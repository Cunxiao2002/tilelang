import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import *
import argparse
import math
from tilelang.layout import make_swizzled_layout
from tvm import tir
import itertools
from tilelang.engine.callback import register_cuda_postproc_callback
from tvm import DataType

torch.manual_seed(42)
tilelang.disable_cache()

'''
Qserve量化方案
weight   -->   temp1   -->   temp2 
bf16     -->   int8    -->   int4
     per-channel     per-group
'''

def get_configs():
    iter_params = dict(
        block_M=[32, 64, 128, 256],
        block_N=[32, 64, 128, 256],
        block_K=[32, 64, 128, 256],
        num_stages=[1, 2, 3, 4, 5],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

def _tir_u8_to_i4_to_i8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    """
    使用左移右移方法将打包在uint8中的4位有符号整数(INT4)转换为8位有符号整数(INT8) b
    """
    assert nbit == 4
    assert dtype == "int8"
    assert val.dtype == "uint8"
    
    # 创建4位掩码
    mask = tir.const((1 << nbit) - 1, "uint8")
    
    # 从uint8中提取4位数据
    i4 = (val >> (pos.astype("uint8") * tir.const(nbit, "uint8"))) & mask
    
    # 使用左移和算术右移
    # 1. 先将低4位数据左移4位，放到int8的高4位
    # 2. 然后算术右移4位，自动进行符号扩展
    i8_shifted = tir.reinterpret("int8", i4 << tir.const(4, "uint8"))
    i8 = i8_shifted >> tir.const(4, "int8")  # 注意这里使用int8类型的右移（算术右移）
    
    return i8

def quantize_per_token_and_smooth(x):
    amax, _ = torch.max(torch.abs(x), dim=1, keepdim=True)
    amax = torch.clamp(amax, min=1e-8)
    quant_scale = 127.0 / amax
    x_quant = (x * quant_scale).round().clamp(-127, 127).to(torch.int8)

    return x_quant, amax / 127.0

# @tilelang.autotune(configs=get_configs())
@tilelang.jit(
    out_idx=[2], pass_configs={"tl.disable_tma_lower": False, "tl.disable_warp_specialized": False}
)
def grouped_gemm_w4a8(
    batch_sizes_list, K, N, block_M, block_N, block_K, num_stages=2, threads=128, in_dtype="int8", out_dtype="bfloat16", num_bits=4, group_size=32
):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
    """
    assert in_dtype == "int8"
    storage_dtype = "uint8"
    accum_dtype = "int32"
    num_elems_per_byte = 8 // num_bits
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)

    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    assert K % (block_K) == 0
    block_QK = block_K // num_elems_per_byte

    # fast dequant for uint4 to int8
    def get_fast_dequant_uint4_int8_func(in_dtype="uint4", out_dtype="int8"):
        assert in_dtype == "uint4"
        assert out_dtype == "int8"

        # Some variables for dequantization in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(out_dtype).bits # 128 // 8 = 16
        local_compress_size = local_size // num_elems_per_byte # 16 // 2 = 8

        @T.macro
        def fast_dequant_uint4_int8(B_shared, B_dequant_shared, scale_shared, k):
            tx = T.get_thread_binding()

            B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
            B_dequant_local_thread = T.alloc_local((local_size,), out_dtype)
            scale_local_thread = T.alloc_local((1,), storage_dtype)
            
            for i in T.serial(0, block_N * block_K // threads // local_size):
                index_base = i * threads * local_compress_size + tx * local_compress_size
                for v in T.vectorized(0, local_compress_size):
                    index = index_base + v
                    B_local_thread[v] = B_shared[index // block_QK, index % block_QK]
                scale_index = index_base // (group_size // num_elems_per_byte)
                si = scale_index // (block_K // group_size)
                sj = scale_index % (block_K // group_size)
                scale_local_thread[0] = scale_shared[si, k * block_K // group_size + sj]

                #TODO(cx) unpack函数
                # 首先将原本的unpack函数放过来。
                # 后续采用liquid gemm的方案时要将这里替换掉
                for i in T.Parallel(local_size):
                    B_dequant_local_thread[i] = _tir_u8_to_i4_to_i8(
                                                num_bits,
                                                B_local_thread[i // num_elems_per_byte],
                                                i % num_elems_per_byte,
                                                dtype="int8",
                                                )

                # last, store the dequantized data to shared memory
                # for v in T.Parallel(local_size):
                #     B_dequant_local_thread[v] *= scale_local_thread[0]
                
                for v in T.vectorized(0, local_size):
                    index = i * threads * local_size + tx * local_size + v
                    B_dequant_shared[index // block_K, index % block_K] = B_dequant_local_thread[v]
            
        return fast_dequant_uint4_int8


    @T.prim_func
    def qserve_kernel(
        A: T.Tensor((batch_sum, K), in_dtype),  # type: ignore
        B: T.Tensor((batch_count, N, K // num_elems_per_byte), storage_dtype),  # type: ignore
        C: T.Tensor((N, batch_sum), out_dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        per_token_scales: T.Tensor([batch_sum], "bfloat16"),  # type: ignore
        expert_weights_qscales: T.Tensor([batch_count, N], "bfloat16"),  # type: ignore
        per_group_scales: T.Tensor((batch_count, N, K // group_size), "int8"),
        per_group_zero_points: T.Tensor((batch_count, N, K // group_size), "int8")
    ):

        with T.Kernel(
            total_m_blocks, T.ceildiv(N, block_N), threads=threads
        ) as (by, bx):
            A_shared = T.alloc_shared([block_M, block_K], in_dtype)
            B_shared = T.alloc_shared([block_N, block_K // num_elems_per_byte], storage_dtype)
            B_dequant_shared = T.alloc_shared([block_N, block_K], in_dtype)
            B_local = T.alloc_fragment([block_N, block_K // num_elems_per_byte], storage_dtype)
            B_unpack_local = T.alloc_fragment([block_N, block_K], in_dtype)
            B_dequant_local = T.alloc_fragment([block_N, block_K], in_dtype)
            Ct_dequant_local = T.alloc_shared([block_N, block_M], out_dtype)
            Ct_local = T.alloc_fragment([block_N, block_M], accum_dtype)
            per_token_scales_shared = T.alloc_shared([block_M], out_dtype)
            expert_weights_qscales_shared = T.alloc_shared([block_N], out_dtype)
            per_group_scales_local = T.alloc_fragment([block_N, block_K // group_size], dtype="int8")
            per_group_zero_points_local = T.alloc_fragment([block_N, block_K // group_size],dtype="int8")

            cur_batch_idx = T.alloc_local([1], "int32")
            cur_batch_size = T.alloc_local([1], "int32")

            # T.use_swizzle(10, order="col", enable=False)
            # T.annotate_layout({
            #     A_shared: make_swizzled_layout(A_shared),
            #     B_shared: make_swizzled_layout(B_shared),
            # })
            m_start_padded = by * block_M

            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx[0] = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx[0])

            cur_batch_size[0] = batch_sizes[cur_batch_idx[0]]
            m_start = (
                m_start_padded
                - batch_padded_offsets[cur_batch_idx[0]]
                + batch_offsets[cur_batch_idx[0]]
            )
            actual_rows = T.max(
                0,
                T.min(
                    block_M,
                    cur_batch_size[0] + batch_padded_offsets[cur_batch_idx[0]] - m_start_padded,
                ),
            )

            T.clear(Ct_local)

            for i in T.Parallel(block_M):
                with T.If(m_start + i < batch_sum), T.Then():
                    per_token_scales_shared[i] = per_token_scales[m_start + i]

            T.copy(
                expert_weights_qscales[cur_batch_idx[0], bx * block_N : (bx + 1) * block_N],
                expert_weights_qscales_shared,
            )

            # 将per-group scale放到local
            T.copy(per_group_scales[cur_batch_idx[0], bx * block_N, by * block_K // group_size], per_group_scales_local)
            T.copy(per_group_zero_points[cur_batch_idx[0], bx * block_N, by * block_K // group_size], per_group_zero_points_local)


            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)

                T.copy(
                    B[
                        cur_batch_idx[0],
                        bx * block_N : (bx + 1) * block_N,
                        k * block_K // num_elems_per_byte : (k + 1) * block_K // num_elems_per_byte,
                    ],
                    B_shared,
                )
                T.copy(B_shared, B_local)

                get_fast_dequant_uint4_int8_func()(B_shared, B_dequant_shared, per_group_scales_local, k)
                # T.gemm(B_dequant_local, A_shared, Ct_local, transpose_B=True)
                T.gemm(B_dequant_shared, A_shared, Ct_local, transpose_B=True)

                
            for i, j in T.Parallel(block_M, block_N):
                Ct_dequant_local[j, i] = Ct_local[j, i] * per_token_scales_shared[i] * expert_weights_qscales_shared[j]
            
            for i, j in T.Parallel(block_N, block_M):
                with T.If(i < actual_rows), T.Then():
                    C[bx * block_N + j, m_start + i] = Ct_dequant_local[j, i]

    return qserve_kernel


def construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(
            batch_padded_offsets_list[-1]
            + math.ceil((batch_sizes_list[i]) / padding_M) * padding_M
        )
    A = torch.randn(batch_sum, K, device=device, dtype=torch.bfloat16)
    #TODO(xiao) 目前这里生成的B是int8，也就是经过per-channel量化完成以后的类型。后面改成随机生成bf16，然后进行per-channel的量化
    B = torch.randint(0, 255, (batch_count, M, K // 2), device=device, dtype=torch.uint8)
    C = torch.empty(M, batch_sum, device=device, dtype=torch.bfloat16)

    A_quant, A_amax = quantize_per_token_and_smooth(A)
    #TODO(xiao) per-channel的scale是随机生成的，后续完成per-channel的函数后对其进行改正
    expert_weights_qscales = (
        torch.randn([batch_count, M], device=device, dtype=torch.bfloat16) * 0.001
    )

    # 随机生成per-group的scale
    group_size = 32
    # per_group_scales = torch.randint(4, 5, (batch_count, M, K // group_size), device=device, dtype=torch.int8)
    # per_group_zero_points = torch.randint(1, 2, (batch_count, M, K // group_size), device=device, dtype=torch.int8)
    
    # for debug
    per_group_scales = torch.ones(batch_count, M, K // group_size, device=device, dtype=torch.int8)
    per_group_zero_points = torch.zeros(batch_count, M, K // group_size, device=device, dtype=torch.int8)

    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
    return (
        A,
        B,
        C,
        batch_sizes,
        batch_offsets,
        batch_padded_offsets,
        A_quant,
        A_amax,
        per_group_scales,
        per_group_zero_points,
        expert_weights_qscales,
    )


def torch_convert(tensor):
    def _convert(val, pos):
        assert val.dtype == torch.uint8
        val = val.view(torch.int8)
        mask = (1 << 4) - 1
        i4_shifted = ((val >> (pos * 4)) & mask)
        i4 = ((i4_shifted << 4) >> 4)

        return i4.view(torch.int8)
    
    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.int8, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2)
    return new_tensor

def torch_gmm_w4a8(
    a, b, batch_sizes, batch_offsets_tensor, per_token_scales, expert_weights_scales, per_group_scales, per_group_zero_points, trans_b=False, group_size=32
):
    """
    Perform grouped matrix multiplication using PyTorch.

    Args:
        a (torch.Tensor): Input tensor of shape (N, K).
        b (torch.Tensor): Input tensor of shape (G, K, M).
        batch_sizes (torch.Tensor): 1D tensor containing the sizes of each group.

    Returns:
        torch.Tensor: Resulting tensor after grouped matrix multiplication.
    """
    assert a.shape[0] == sum(batch_sizes), "Sum of batch_sizes must equal the first dimension of a"
    assert b.shape[0] == len(
        batch_sizes
    ), "The first dimension of b must match the length of batch_sizes"

    b_unpacked_list = []
    for i, b_i in enumerate(b):
        b_i_unpacked = torch_convert(b_i)
        K_packed, M = b_i.shape
        K_unpacked = b_i_unpacked.shape[1]
        
        group_idx = torch.arange(K_unpacked, device=b_i.device) // group_size

        scales = per_group_scales[i][:, group_idx]
        zero_points = per_group_zero_points[i][:, group_idx]
        b_i_dequant = (b_i_unpacked.to(torch.float32) - zero_points.to(torch.float32)) * scales.to(torch.float32)
        b_unpacked_list.append(b_i_dequant.to(torch.int8))
    
    b_unpacked = torch.stack(b_unpacked_list)

    # Initialize output tensor
    output = torch.empty((sum(batch_sizes), b_unpacked.shape[1]), device=a.device, dtype=torch.bfloat16)

    # Perform grouped GEMM
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        part_a = a[start:end].to(torch.bfloat16)
        part_b = b_unpacked[i].transpose(0, 1).to(torch.bfloat16) if trans_b else b_unpacked[i].to(torch.bfloat16)
        part_out = torch.matmul(part_a, part_b)
        output[start:end] = part_out * expert_weights_scales[i, :]
        start = end

    output = output * per_token_scales

    return output.transpose(0, 1)

def run_tilelang_grouped_gemm(
    batch_sizes_list,
    K,
    M,
    block_M,
    block_N,
    block_K,
    trans_b,
    num_stages,
    threads,
    profile=False,
    source=False,
):
    padding_M = block_M
    batch_sum = sum(batch_sizes_list)

    # @register_cuda_postproc_callback
    # def tilelang_callback_cuda_postproc(code, _):
    #     kernel_file_path = "/root/.tilelang/cache/7785140a97ffe139bcaba346aed265fe92f7ae457ac0161b8edece8aaa6e61a7/kernel.cu"
    #     try:
    #         with open(kernel_file_path, 'r') as f:
    #             code = f.read()
    #     except Exception as e:
    #         # 如果读取失败，返回错误信息（这样在编译时会报错）
    #         code = f"// Error reading file {kernel_file_path}: {str(e)}\n"
    #     return code

    kernel = grouped_gemm_w4a8(
        tuple(batch_sizes_list), K, M, block_M, block_N, block_K, num_stages, threads
    )

    # for autotune
    # kernel = grouped_gemm_w4a8(tuple(batch_sizes_list), K, M)
    # print(kernel.config)

    device = torch.device("cuda")
    dtype = torch.int8

    (
        A,
        B,
        C,
        batch_sizes,
        batch_offsets,
        batch_padded_offsets,
        A_quant,
        A_amax,
        per_group_scales,
        per_group_zero_points,
        expert_weights_qscales,
    ) = construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype)

    out = kernel(
        A_quant, B, batch_sizes, batch_offsets, batch_padded_offsets, A_amax, expert_weights_qscales, per_group_scales, per_group_zero_points
    )
    # ref_output = torch_gmm_w4a8(
    #     A_quant, B, batch_sizes, batch_offsets, A_amax, expert_weights_qscales, per_group_scales, per_group_zero_points, trans_b=True
    # )
    print(out)
    print(kernel.get_kernel_source())
    # print(ref_output)
    
    # try:
    #     torch.testing.assert_close(out, ref_output, rtol=0.01, atol=0.01)
    #     print("✅ Tilelang and Torch match")
    # except Exception as e:
    #     print("❌ Tilelang and Torch mismatch")
    #     mismatch_mask = ~torch.isclose(out, ref_output, rtol=0.01, atol=0.01)
    #     mismatch_indices = torch.nonzero(mismatch_mask)

    #     # 打印前100个不匹配点（避免输出过多）
    #     max_print = 10
    #     print(f"\nFirst {min(len(mismatch_indices), max_print)} mismatches:")
    #     print("Row\tCol\tTilelang\tTorch\tDiff")

    #     for idx in mismatch_indices[:max_print]:
    #         row, col = idx.tolist()
    #         tilelang_val = out[row, col].item()
    #         torch_val = ref_output[row, col].item()
    #         diff = tilelang_val - torch_val
    #         print(f"{row}\t{col}\t{tilelang_val:.4f}\t\t{torch_val:.4f}\t{diff:.4f}")

    #     # 统计信息
    #     all_diffs = (out - ref_output)[mismatch_mask]
    #     print("\nStatistics:")
    #     print(f"Total mismatches: {len(mismatch_indices)}")
    #     print(f"Max diff: {all_diffs.max().item():.10f}")
    #     print(f"Min diff: {all_diffs.min().item():.10f}")
    #     print(f"Avg diff: {all_diffs.float().mean().item():.10f}")

    if profile:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        latency = profiler.do_bench(
            warmup=50,
            rep=20,
        )
        # latency = kernel.latency
        print(f"Best config: {kernel.config}")
        print(f"Latency: {latency} ms")
        print(f"TFlops: {batch_sum * K * M * 2 / latency * 1e-9} TFlops")

    if source:
        print("CUDA Source:")
        print(kernel.get_kernel_source())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_sizes", type=str, default="64, 128", help="comma-separated batch sizes"
    )
    parser.add_argument("--K", type=int, default=8192, help="reduce dim")
    parser.add_argument("--M", type=int, default=8192, help="output dim")
    parser.add_argument("--trans_b", action="store_true", help="transpose B")
    parser.add_argument("--profile", action="store_true", help="profile")
    parser.add_argument("--source", action="store_true", help="print source")
    args = parser.parse_args()

    batch_sizes_list = [int(x) for x in args.batch_sizes.split(",")]
    K, M, trans_b = args.K, args.M, args.trans_b

    # only ep=8
    # run_tilelang_grouped_gemm([192]*16, 4096, 2560, 64, 64, 64, num_stages=1, threads=128, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([192]*16, 1280, 4096, 64, 256, 64, num_stages=2, threads=256, trans_b=False, profile=args.profile)
    
    # # w4a8 without tma
    # run_tilelang_grouped_gemm([36]*16, 4096, 2560, 64, 128, 32, num_stages=1, threads=128, trans_b=False, profile=args.profile) #num_tokens=768

    # benchmark w4a8 with tma
    run_tilelang_grouped_gemm([36]*16, 4096, 2560, 64, 64, 128, num_stages=4, threads=128, trans_b=False, profile=args.profile) #num_tokens=768

    # test accuracy
    # run_tilelang_grouped_gemm([64], 128, 128, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([64]*8, 512, 256, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([100, 200, 300, 400, 100, 230, 242, 120], 512, 256, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([100, 50], 512, 256, 64, 64, 64, num_stages=2, threads=128, trans_b=True, profile=args.profile)