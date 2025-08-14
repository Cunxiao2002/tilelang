import tilelang
import tilelang.language as T
import itertools
import torch

'''
hidden_dim = 4096
num_experts = 128
num_tokens = 4 * 80
dtype = bfloat16
x: (num_tokens, hidden_dim)
gate_net = tl.Linear(hidden_dim, num_experts, bias=False)
logits = gate_net(x.float())
'''

def get_configs():
    iter_params = dict(
        block_M=[32, 64, 128, 256],
        block_N=[32, 64, 128, 256],
        block_K=[32, 64, 128, 256],
        split_k = [2],
        num_stages=[0, 1, 2, 3, 4],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


# @tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[2])
def matmul(M,
           N,
           K,
           block_M,
           block_N,
           block_K,
           split_k,
           num_stages,
           threads,
           dtype="bfloat16",
           accum_dtype="float",
           out_dtype="float32"):

    splitK = K // split_k

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
                T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            T.atomic_add(C[by * block_M, bx * block_N], C_shared)

    return main

def main():
    M = 384
    N = 128
    K = 4096
    block_M = 32
    block_N = 64
    block_K = 128
    split_k = 2
    num_stages = 1
    threads = 128
    kernel = matmul(M, N, K, block_M, block_N, block_K, split_k, num_stages, threads)

    a = torch.randn(M, K).cuda().to(torch.bfloat16)
    b = torch.randn(K, N).cuda().to(torch.bfloat16)
    c = torch.zeros(M, N).cuda().float()
    c = kernel(a, b)

    # benchmark
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench()
    print(f"TileLang latency: {tilelang_latency}")
    print(f"TileLang TFlops: {2 * M * N * K / tilelang_latency * 1e-9}")

if __name__ == "__main__":
    main()