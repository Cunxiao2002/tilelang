import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
import itertools

# only support sum scatter reduce
def get_configs():
    iter_params = dict(
        blk_tokens=[64, 128, 256, 512, 1024],
        threads=[128, 256, 512, 1024],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[-1])
def scatter_reduce_sum(
    num_tokens,
    topk,
    hidden_dim,
    dtype,
    accum_dtype,
    blk_tokens,
    threads,
):

    @T.prim_func
    def gather(
        expert_cache: T.Tensor([num_tokens * topk, hidden_dim], dtype),
        expert_tokens_idxs: T.Tensor([num_tokens * topk], "int32"),
        output: T.Tensor([num_tokens, hidden_dim], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens * topk, blk_tokens), threads=threads) as (bx):
            idx_shared = T.alloc_shared([num_tokens], "int32")
            val_local = T.alloc_local([1], dtype)
            dst_row = T.alloc_local([1], "int32")
            for i, j in T.Parallel(blk_tokens, hidden_dim):
                dst_row = expert_tokens_idxs[bx * blk_tokens + i]
                val_local = expert_cache[bx * blk_tokens + i, j]
                T.atomic_add(output[dst_row, j], val_local)
        
    
    return gather


def test_scatter_reduce_sum_correctness(num_tokens, topk, hidden_dim, blk_tokens, dtype_str, accum_dtype, torch_dtype, threads):
    """
    Tests the correctness of the scatter_reduce_sum kernel by comparing its output
    with the equivalent operation from torch.scatter_reduce.
    """
    device = "cuda"

    # Create kernel
    # kernel = scatter_reduce_sum(
    #     num_tokens=num_tokens,
    #     topk=topk,
    #     hidden_dim=hidden_dim,
    #     dtype=dtype_str,
    #     accum_dtype=accum_dtype,
    #     blk_tokens=blk_tokens,
    #     threads=threads,
    # )

    kernel = scatter_reduce_sum(
        num_tokens=num_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        dtype=dtype_str,
        accum_dtype=accum_dtype,
    )

    # Create test data
    src_size = (num_tokens * topk, hidden_dim)
    src = torch.randn(src_size, dtype=torch_dtype, device=device)
    
    # Indices for scattering. Should be in range [0, num_tokens - 1]
    idx = torch.randint(0, num_tokens, (num_tokens * topk,), dtype=torch.int32, device=device)
    
    # Output tensor for tilelang kernel, initialized to zeros
    output_tl = torch.zeros(num_tokens, hidden_dim, dtype=torch_dtype, device=device)

    # Run tilelang kernel
    output_tl = kernel(src, idx)
    torch.cuda.synchronize()

    # --- Reference implementation with torch.scatter_reduce ---
    output_ref = torch.zeros(num_tokens, hidden_dim, dtype=torch_dtype, device=device)
    
    # torch.scatter_reduce expects index to have same number of dims as src and be of type LongTensor (int64).
    idx_torch = idx.to(torch.int64).unsqueeze(-1).expand_as(src)
    
    # Perform the scatter-reduce operation
    output_ref = output_ref.scatter_reduce(0, idx_torch, src, reduce="sum", include_self=True).to(torch.float32)

    # Compare results
    # The kernel in MoE_develop/gather/gather.py has a bug and is expected to fail this test.
    # This test serves to demonstrate the incorrectness.
    print(output_tl)
    print(output_ref)
    torch.testing.assert_close(output_tl, output_ref, rtol=1e-2, atol=1e-2)

    # performance
    print(f"best configs:")
    print(kernel.config)
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(warmup=25, rep=100)
    print(f"Latency: {latency:.8f} ms")


if __name__ == "__main__":
    num_tokens = 320
    topk = 6
    hidden_dim = 4096
    blk_tokens = 128
    threads = 128
    dtype_str = "float16"
    accum_dtype = "float32"
    torch_dtype = torch.float16
    test_scatter_reduce_sum_correctness(num_tokens, topk, hidden_dim, blk_tokens, dtype_str, accum_dtype, torch_dtype, threads)