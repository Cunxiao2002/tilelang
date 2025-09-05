import torch
import argparse
import tilelang
import tilelang.language as T
import math
import itertools
from tilelang.autotuner import *
import pytest
from tilelang.carver.arch import driver

# tilelang.disable_cache()


def get_configs():
    iter_params = dict(
        block_M=[32, 64, 128, 256],
        block_N=[32, 64, 128, 256],
        block_K=[32, 64, 128, 256],
        num_stages=[1, 2, 3],
        threads=[128, 256, 512],
        dtype=["int8"], 
        use_persistent=[False]
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


def quantize_per_token_and_smooth(x):
    amax, _ = torch.max(torch.abs(x), dim=1, keepdim=True)
    amax = torch.clamp(amax, min=1e-8)
    quant_scale = 127.0 / amax
    x_quant = (x * quant_scale).round().clamp(-127, 127).to(torch.int8)

    return x_quant, amax / 127.0


def torch_gmm(
    a, b, batch_sizes, batch_offsets_tensor, per_token_scales, expert_weights_scales, trans_b=False
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

    # Initialize output tensor
    output = torch.empty((sum(batch_sizes), b.shape[1]), device=a.device, dtype=torch.bfloat16)

    # Perform grouped GEMM
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        part_a = a[start:end].to(torch.bfloat16)
        part_b = b[i].transpose(0, 1).to(torch.bfloat16) if trans_b else b[i].to(torch.bfloat16)
        # part_out = torch.mm(part_a, part_b)
        part_out = torch.matmul(part_a, part_b)
        output[start:end] = part_out * expert_weights_scales[i, :]
        start = end

    output = output * per_token_scales

    return output


@autotune(configs=get_configs())
@tilelang.jit(
    out_idx=[2], pass_configs={"tl.disable_tma_lower": False, "tl.disable_warp_specialized": False}
)
def grouped_gemm(
    batch_sizes_list, K, N, block_M, block_N, block_K, num_stages=2, threads=128, dtype="int8", use_persistent=True
):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
    """
    assert dtype == "int8"
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = "int32"
    sm_num = driver.get_num_sms()
    total_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    @T.prim_func
    def non_persistent(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        # B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        B: T.Tensor([batch_count, N, K], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], "bfloat16"),  # type: ignore
        batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        per_token_scales: T.Tensor([batch_sum], "bfloat16"),  # type: ignore
        expert_weights_qscales: T.Tensor([batch_count, N], "bfloat16"),  # type: ignore
    ):

        # with T.Kernel(
        #     T.ceildiv(batch_sum, block_M) + batch_count, T.ceildiv(N, block_N), threads=threads
        # ) as (bx, by):
        with T.Kernel(total_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            # B_shared = T.alloc_shared([block_K, block_N], dtype)
            B_shared = T.alloc_shared([block_N, block_K], dtype)
            # C_shared = T.alloc_shared([block_M, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            per_token_scales_shared = T.alloc_shared([block_M], "bfloat16")
            expert_weights_qscales_shared = T.alloc_shared([block_N], "bfloat16")

            cur_batch_idx = T.alloc_local([1], "int32")
            cur_batch_size = T.alloc_local([1], "int32")

            # T.use_swizzle(10, order="col")
            # T.annotate_layout({
            #     A_shared: make_swizzled_layout(A_shared),
            #     B_shared: make_swizzled_layout(B_shared),
            # })
            m_start_padded = bx * block_M

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

            T.clear(C_local)

            for i in T.Parallel(block_M):
                with T.If(m_start + i < batch_sum), T.Then():
                    per_token_scales_shared[i] = per_token_scales[m_start + i]

            T.copy(
                expert_weights_qscales[cur_batch_idx[0], by * block_N : (by + 1) * block_N],
                expert_weights_qscales_shared,
            )

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)

                T.copy(
                    B[
                        cur_batch_idx[0],
                        by * block_N : (by + 1) * block_N,
                        k * block_K : (k + 1) * block_K,
                    ],
                    B_shared,
                )
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                with T.If(i < actual_rows), T.Then():
                    C[m_start + i, by * block_N + j] = (
                        C_local[i, j]
                        * per_token_scales_shared[i]
                        * expert_weights_qscales_shared[j]
                    )


    @T.prim_func
    def persistent(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        # B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        B: T.Tensor([batch_count, N, K], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], "bfloat16"),  # type: ignore
        batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        per_token_scales: T.Tensor([batch_sum], "bfloat16"),  # type: ignore
        expert_weights_qscales: T.Tensor([batch_count, N], "bfloat16"),  # type: ignore
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            # B_shared = T.alloc_shared([block_K, block_N], dtype)
            B_shared = T.alloc_shared([block_N, block_K], dtype)
            # C_shared = T.alloc_shared([block_M, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            per_token_scales_shared = T.alloc_shared([block_M], "bfloat16")
            expert_weights_qscales_shared = T.alloc_shared([block_N], "bfloat16")

            cur_batch_idx = T.alloc_local([1], "int32")
            cur_batch_size = T.alloc_local([1], "int32")
            
            for bx, by in T.Persistent(
                [T.ceildiv(batch_sum, block_M) + batch_count, T.ceildiv(N, block_N)], sm_num, block_id):
                m_start_padded = bx * block_M

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

                T.clear(C_local)

                for i in T.Parallel(block_M):
                    with T.If(m_start + i < batch_sum), T.Then():
                        per_token_scales_shared[i] = per_token_scales[m_start + i]

                T.copy(
                    expert_weights_qscales[cur_batch_idx[0], by * block_N : (by + 1) * block_N],
                    expert_weights_qscales_shared,
                )

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)

                    T.copy(
                        B[
                            cur_batch_idx[0],
                            by * block_N : (by + 1) * block_N,
                            k * block_K : (k + 1) * block_K,
                        ],
                        B_shared,
                    )
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    with T.If(i < actual_rows), T.Then():
                        C[m_start + i, by * block_N + j] = (
                            C_local[i, j]
                            * per_token_scales_shared[i]
                            * expert_weights_qscales_shared[j]
                        )

        
    return persistent if use_persistent else non_persistent


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
    # B = torch.randn(batch_count, M, K, device=device, dtype=torch.bfloat16)
    B = torch.randint(-127, 127, (batch_count, M, K), device=device, dtype=torch.int8)
    C = torch.empty(batch_sum, M, device=device, dtype=torch.bfloat16)

    A_quant, A_amax = quantize_per_token_and_smooth(A)
    expert_weights_qscales = (
        torch.randn([batch_count, M], device=device, dtype=torch.bfloat16) * 0.001
    )
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
        expert_weights_qscales,
    )


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
    use_persistent=False,
):
    padding_M = block_M
    dtype = "int8"
    batch_sum = sum(batch_sizes_list)
    # kernel = grouped_gemm(
    #     tuple(batch_sizes_list), K, M, block_M, block_N, block_K, num_stages, threads, dtype, use_persistent
    # )
    # print(kernel.get_kernel_source())
    kernel = grouped_gemm(tuple(batch_sizes_list), K, M)

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
        expert_weights_qscales,
    ) = construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype)
    out = kernel(
        A_quant, B, batch_sizes, batch_offsets, batch_padded_offsets, A_amax, expert_weights_qscales
    )
    ref_output = torch_gmm(
        A_quant, B, batch_sizes, batch_offsets, A_amax, expert_weights_qscales, trans_b=True
    )
    print(out)
    print(ref_output)
    # if torch.testing.assert_close(out, ref_output, rtol=0.01, atol=0.01):
    #     print("✅ Tilelang and Torch match")
    # else:
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
    #         print(f"{row}\t{col}\t{tilelang_val:.2f}\t\t{torch_val:.2f}\t{diff:.2f}")

    #     # 统计信息
    #     all_diffs = (out - ref_output)[mismatch_mask]
    #     print("\nStatistics:")
    #     print(f"Total mismatches: {len(mismatch_indices)}")
    #     print(f"Max diff: {all_diffs.max().item():.2f}")
    #     print(f"Min diff: {all_diffs.min().item():.2f}")
    #     print(f"Avg diff: {all_diffs.float().mean().item():.2f}")

    if profile:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        latency = profiler.do_bench(
            warmup=500,
            input_tensors=[
                A_quant,
                B,
                batch_sizes,
                batch_offsets,
                batch_padded_offsets,
                A_amax,
                expert_weights_qscales,
            ],
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
    parser.add_argument("--use_persistent", action="store_true", help="use persistent")
    args = parser.parse_args()

    batch_sizes_list = [int(x) for x in args.batch_sizes.split(",")]
    K, M, trans_b = args.K, args.M, args.trans_b

    # only tp, tp=8
    # run_tilelang_grouped_gemm([192]*16*8, 4096, 320, 64, 128, 128, num_stages=2, threads=128, trans_b=False, profile=args.profile)
    # run_tilelang_grouped_gemm([192]*16*8, 160, 4096, 128, 256, 32, num_stages=2, threads=256, trans_b=False, profile=args.profile)

    # only ep=8
    # with tma
    # run_tilelang_grouped_gemm([192]*16, 4096, 2560, 64, 64, 256, num_stages=1, threads=128, trans_b=False, profile=args.profile, use_persistent=False)
    # run_tilelang_grouped_gemm([192]*16, 1280, 4096, 64, 256, 64, num_stages=2, threads=256, trans_b=False, profile=args.profile, use_persistent=True)

    # without tma
    run_tilelang_grouped_gemm([192]*16, 4096, 2560, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile, use_persistent=False)
    # run_tilelang_grouped_gemm([192]*16, 1280, 4096, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile, use_persistent=False)
    # run_tilelang_grouped_gemm([64]*8, 512, 256, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile, use_persistent=True)
    # run_tilelang_grouped_gemm([100, 200, 300, 400, 100, 230, 242, 120], 512, 256, 64, 64, 64, num_stages=2, threads=128, trans_b=True, profile=args.profile, use_persistent=True)

    # for test
    # run_tilelang_grouped_gemm([64, 128, 128, 256], 1024, 512, 64, 64, 64, num_stages=1, threads=128, trans_b=False, profile=args.profile, use_persistent=args.use_persistent)
    # run_tilelang_grouped_gemm([320]*16, 512, 256, 64, 64, 64, num_stages=2, threads=128, trans_b=False, profile=args.profile, use_persistent=True)
    # run_tilelang_grouped_gemm([64, 256, 100, 200, 300, 400, 128, 64, 100, 230, 242, 120, 64, 256], 512, 256, 64, 64, 64, num_stages=2, threads=128, trans_b=True, profile=args.profile, use_persistent=True)