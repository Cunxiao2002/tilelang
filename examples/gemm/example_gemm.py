import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": False, "tl.disable_warp_specialized": False})
def matmul(M, N, K, block_M, block_N, block_K, dtype="int8", accum_dtype="int32"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            # B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared = T.alloc_shared([block_N, block_K], dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                # T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 512, 128, 128, 32)

    import torch

    # a = torch.randn(1024, 1024).cuda().half()
    # b = torch.randn(1024, 1024).cuda().half()
    a = torch.randn(1024, 512).cuda().to(torch.int8)
    b = torch.randn(1024, 512).cuda().to(torch.int8)

    c = kernel(a, b)

    ref_c = a.to(torch.float16) @ b.to(torch.float16).T
    ref_c = ref_c.to(torch.int8)

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
