import tilelang
import tilelang.language as T
from tilelang.autotuner import *
import itertools

def get_configs():
    iter_params = dict(
        block_M=[32, 64, 128, 256],
        block_N=[32, 64, 128, 256],
        block_K=[16, 32, 64, 128],
        num_stages=[0, 1, 2, 3],
        threads=[128, 256, 512],
        split_k=[2, 4]
        
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

# @tilelang.autotune(configs=get_configs())
@tilelang.jit(pass_configs={"tl.disable_tma_lower": False, "tl.disable_warp_specialized": False})
def matmul(M,
           N,
           K,
           block_M,
           block_N,
           block_K,
           num_stages,
           threads,
           split_k,
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
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=threads) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])

    return main


def main():
    M = 320
    N = 128
    K = 4096

    # best config
    block_M = 64
    block_N = 32
    block_K = 64
    num_stages = 3
    threads = 128
    split_k = 4

    kernel = matmul(M, N, K, block_M, block_N, block_K, num_stages, threads, split_k)
    # kernel = matmul(M, N, K)

    import torch

    torch.random.manual_seed(42)
    a = torch.randn(M, K).cuda().to(torch.bfloat16)
    b = torch.randn(K, N).cuda().to(torch.bfloat16)
    c = torch.zeros(M, N).cuda().to(torch.float32)
    kernel(a, b, c)

    print(f"no vectorize_atomicadd kernel source:")
    print(kernel.get_kernel_source())

    print(f"tl output:")
    print(c)


    ref_c = a @ b
    print(f"ref output:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c.to(c.dtype), rtol=1e-2, atol=1e-2)
    print("All checks passed!")
    
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(n_warmup=10, n_repeat=10)
    print(f"tl Latency:{latency} ms")



if __name__ == "__main__":
    main()

 
    # for (int i_outer = 0; i_outer < 4; ++i_outer) {
    #   if ((((((tx / 4) >> 2) + i_outer) >> 2) + by) < 5) {
    #     AtomicAddx4((&(C[((((by * 8192 + (i_outer * 2048)) + ((tx / 8) * 512)) + (bx * 32)) + ((tx % 4) * 4))])), (&(((float*)buf_dyn_shmem)[(((i_outer * 512+ ((tx / 4) * 128)) + ((tx % 4) * 4)) + 6144)])));
    #   }


      
    # }




    # (((((tx / 32) >> 2) + i_outer) >> 2) + by)*4*4 < 5

    # tx/4 + i_outer * 4 + by * 16 < 5 * 4 * 4


    # tx/8 + i_outer * 16 +by * 64 < 319
    # [5, 4, 4, 4]
