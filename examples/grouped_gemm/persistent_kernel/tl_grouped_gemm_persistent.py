import torch
import argparse
import tilelang
import tilelang.language as T
import math
import itertools
from tilelang.autotuner import *
from tilelang.layout import make_swizzled_layout
# tilelang.disable_cache()


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def torch_gmm(a, b, batch_sizes, batch_offsets_tensor, trans_b=False):
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
        batch_sizes), "The first dimension of b must match the length of batch_sizes"

    # Initialize output tensor
    output = torch.empty((sum(batch_sizes), b.shape[1]), device=a.device, dtype=torch.bfloat16)

    # Perform grouped GEMM
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        part_a = a[start:end].to(torch.bfloat16)
        part_b = b[i].transpose(0, 1).to(torch.bfloat16) if trans_b else b[i].to(torch.bfloat16)
        part_out = torch.mm(part_a, part_b)
        output[start:end] = part_out
        start = end

    return output

def get_configs():
    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[32, 64, 128, 256],
        num_stages=[0, 1, 2],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

# @autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[2], pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": False
    })
def grouped_gemm_persistent(batch_sizes_list,
                 K,
                 N,
                 block_M,
                 block_N,
                 block_K,
                 num_stages=2,
                 threads=128,
                 dtype="int8",
                 accum_dtype="int32",
                 use_persistent_primitive=True):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = "int32"

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    @T.prim_func
    def main_persistent_primitive(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        B: T.Tensor([batch_count, N, K], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], "bfloat16"),  # type: ignore
        batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
    ):
        with T.Kernel()



    @T.prim_func
    def kernel(
            A: T.Tensor([batch_sum, K], dtype),  # type: ignore
            # B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
            B: T.Tensor([batch_count, N, K], dtype),  # type: ignore
            C: T.Tensor([batch_sum, N], "bfloat16"),  # type: ignore
            batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
    ):

        with T.Kernel(
                T.ceildiv(batch_sum, block_M) + batch_count, T.ceildiv(N, block_N),
                threads=threads) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            # B_shared = T.alloc_shared([block_K, block_N], dtype)
            B_shared = T.alloc_shared([block_N, block_K], dtype)
            # C_shared = T.alloc_shared([block_M, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            cur_batch_idx = T.alloc_local([1], "int32")
            cur_batch_size = T.alloc_local([1], "int32")

            # T.use_swizzle(10, order="col")
            # T.annotate_layout({
            #     A_shared: make_swizzled_layout(A_shared),
            #     B_shared: make_swizzled_layout(B_shared),
            # })
            m_start_padded = bx * block_M

            for i in range(batch_count):
                in_cur_batch_idx = (m_start_padded >= batch_padded_offsets[i])
                cur_batch_idx[0] = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx[0])

            cur_batch_size[0] = batch_sizes[cur_batch_idx[0]]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx[0]] + batch_offsets[
                cur_batch_idx[0]]
            actual_rows = T.max(
                0,
                T.min(block_M,
                      cur_batch_size[0] + batch_padded_offsets[cur_batch_idx[0]] - m_start_padded))

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start:m_start + block_M, k * block_K:(k + 1) * block_K], A_shared)
                # T.copy(
                #     B[cur_batch_idx[0], k * block_K:(k + 1) * block_K,
                #       by * block_N:(by + 1) * block_N], B_shared)
                T.copy(B[cur_batch_idx[0], by * block_N:(by + 1) * block_N,
                      k * block_K:(k + 1) * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            # with tma存在合并访存的问题
            # T.copy(C_local, C_shared)
            for i, j in T.Parallel(block_M, block_N):
                with T.If(i < actual_rows), T.Then():
                    C[m_start + i, by * block_N + j] = C_local[i, j]

    return kernel


def construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                         math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                         padding_M)
    A = to_int8(torch.randn(batch_sum, K, device=device, dtype=torch.float16))
    # B = torch.randn(batch_count, K, M, device=device, dtype=dtype)
    B = to_int8(torch.randn(batch_count, M, K, device=device, dtype=torch.float16))
    C = torch.empty(batch_sum, M, device=device, dtype=dtype)
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
    # print(batch_sizes_tensor)
    # print(batch_offsets_tensor)
    # print(batch_padded_offsets_tensor)
    return A, B, C, batch_sizes, batch_offsets, batch_padded_offsets


def run_tilelang_grouped_gemm(batch_sizes_list,
                              K,
                              M,
                              block_M,
                              block_N,
                              block_K,
                              trans_b,
                              num_stages,
                              threads,
                              profile=False):
    padding_M = block_M
    batch_sum = sum(batch_sizes_list)
    kernel = grouped_gemm(tuple(batch_sizes_list), K, M, block_M, block_N, block_K, num_stages, threads)
    # kernel = grouped_gemm(tuple(batch_sizes_list), K, M)

    device = torch.device("cuda")
    # dtype = torch.float16
    dtype = torch.int8

    A, B, C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs(
        batch_sizes_list, K, M, trans_b, padding_M, device, dtype)
    out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    ref_output = torch_gmm(A, B, batch_sizes, batch_offsets, trans_b=True)
    # print(out)
    # print(ref_output)
    print(kernel.get_kernel_source())
    # print(kernel.config)
    if torch.allclose(out, ref_output, rtol=0.01, atol=0.01):
        print("✅ Tilelang and Torch match")
    else:
        print("❌ Tilelang and Torch mismatch")
        mismatch_mask = ~torch.isclose(out, ref_output, rtol=0.01, atol=0.01)
        mismatch_indices = torch.nonzero(mismatch_mask)
        
        # 打印前100个不匹配点（避免输出过多）
        max_print = 10
        print(f"\nFirst {min(len(mismatch_indices), max_print)} mismatches:")
        print("Row\tCol\tTilelang\tTorch\tDiff")
        
        for idx in mismatch_indices[:max_print]:
            row, col = idx.tolist()
            tilelang_val = out[row, col].item()
            torch_val = ref_output[row, col].item()
            diff = tilelang_val - torch_val
            print(f"{row}\t{col}\t{tilelang_val:.2f}\t\t{torch_val:.2f}\t{diff:.2f}")
        
        # 统计信息
        all_diffs = (out - ref_output)[mismatch_mask]
        print("\nStatistics:")
        print(f"Total mismatches: {len(mismatch_indices)}")
        print(f"Max diff: {all_diffs.max().item():.2f}")
        print(f"Min diff: {all_diffs.min().item():.2f}")
        print(f"Avg diff: {all_diffs.float().mean().item():.2f}")

    if profile:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        latency = profiler.do_bench(
            warmup=500, input_tensors=[A, B, batch_sizes, batch_offsets, batch_padded_offsets])
        # latency = kernel.latency
        print(f"Best config: {kernel.config}")
        print(f"Latency: {latency} ms")
        print(f"TFlops: {batch_sum * K * M * 2 / latency * 1e-9} TFlops")


def test_grouped_gemm():
    run_tilelang_grouped_gemm([64], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([64, 128, 256], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([63], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([100, 200, 300, 400], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([63, 77, 111, 280], 8192, 8192, 64, 64, 64, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_sizes', type=str, default="64, 128", help='comma-separated batch sizes')
    parser.add_argument('--K', type=int, default=8192, help='reduce dim')
    parser.add_argument('--M', type=int, default=8192, help='output dim')
    parser.add_argument('--trans_b', action="store_true", help="transpose B")
    parser.add_argument('--profile', action="store_true", help="profile")
    args = parser.parse_args()

    batch_sizes_list = [int(x) for x in args.batch_sizes.split(",")]
    K, M, trans_b = args.K, args.M, args.trans_b

    # only tp, tp=8
    # run_tilelang_grouped_gemm([192]*16*8, 4096, 320, 64, 128, 128, num_stages=2, threads=128, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([192]*16*8, 160, 4096, 128, 256, 32, num_stages=2, threads=256, trans_b=False, profile=args.profile)

    # only ep=8
    # with tma
    run_tilelang_grouped_gemm([192]*16, 4096, 2560, 64, 128, 256, num_stages=2, threads=256, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([192]*16, 1280, 4096, 64, 256, 64, num_stages=2, threads=256, trans_b=False, profile=args.profile)

    # without tma
    # run_tilelang_grouped_gemm([192]*16, 4096, 2560, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([192]*16, 1280, 4096, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile)