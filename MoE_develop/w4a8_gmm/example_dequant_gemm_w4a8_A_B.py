import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from tvm import tir
import itertools
import torch
import argparse

def _tir_u8_to_i4_to_i8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    """
    使用左移右移方法将打包在uint8中的4位有符号整数(INT4)转换为8位有符号整数(INT8)
    """
    assert nbit == 4
    assert dtype == "int8"
    assert val.dtype == "uint8"
    
    # 创建4位掩码
    mask = tir.const((1 << nbit) - 1, "uint8")
    
    # 从uint8中提取4位数据
    i4 = (val >> (pos.astype("uint8") * tir.const(nbit, "uint8"))) & mask
    
    # 方法3: 使用左移和算术右移
    # 1. 先将4位数据左移4位，放到int8的高4位
    # 2. 然后算术右移4位，自动进行符号扩展
    i8_shifted = tir.reinterpret("int8", i4 << tir.const(4, "uint8"))
    i8 = i8_shifted >> tir.const(4, "int8")  # 注意这里使用int8类型的右移（算术右移）
    
    return i8

def get_configs():
    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[64, 128, 256],
        num_stages=[1, 2],
        threads=[128, 256, 512, 1024],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

# test_convert_int4_to_int8()

@tilelang.jit(out_idx=[1])
def _convert_test(N, K, block_N, block_K, in_dtype, num_bits=4, threads=128):
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
    B_shape = (N, K // num_elems_per_byte)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)

    @T.prim_func
    def main(
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((N, K), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)

            for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=1):
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                T.copy(B_shared, B_local)
                for i, j in T.Parallel(block_N, block_K):
                    B_dequantize_local[i, j] = _tir_u8_to_i4_to_i8(
                        num_bits,
                        B_local[i, j // num_elems_per_byte],
                        j % num_elems_per_byte,
                        dtype=in_dtype,
                    )
                T.copy(B_dequantize_local, C[bx * block_N, k * block_K])
    
    return main

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


def test_int4_int8_convert_close():
    N, K = 256, 256
    block_N, block_K = 64, 64
    tl_convert_kernel = _convert_test(N, K, block_N, block_K, "int8")
    B = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda").to(torch.uint8)
    tl_out = tl_convert_kernel(B)
    ref_out = torch_convert(B)

    print(tl_out)
    print(ref_out)
    assert torch.allclose(tl_out, ref_out, rtol=0.01, atol=0.01), (tl_out, ref_out)
    print("pass")


def ref_program(A, qB):
    dtypeC = "int32"
    B = torch_convert(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C.transpose(0, 1)

# @tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[2])
def matmul_int8xint4(M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, threads, num_bits=4):
    # K是解包后的K
    # 因此解包前是K // 2
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
    A_shape = (M, K)
    B_shape = (N, K // num_elems_per_byte)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_local_shape = (block_N, block_K)

    assert K % (block_K) == 0

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, storage_dtype),
            Ct: T.Tensor((N, M), out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_local_shape, in_dtype)
            B_dequantize_prev_local = T.alloc_fragment(B_dequantize_local_shape, in_dtype)
            Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
            Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

            T.annotate_layout({
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                Ct_shared: tilelang.layout.make_swizzled_layout(Ct_shared),
            })

            T.clear(Ct_local)
            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                T.copy(B_shared, B_local)
                for i, j in T.Parallel(block_N, block_K):
                    B_dequantize_local[i, j] = _tir_u8_to_i4_to_i8(
                        num_bits,
                        B_local[i, j // num_elems_per_byte],
                        j % num_elems_per_byte,
                        dtype=in_dtype,
                    )
                T.copy(B_dequantize_local, B_dequantize_prev_local)
                T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
            T.copy(Ct_local, Ct_shared)
            T.copy(Ct_shared, Ct[bx * block_N:(bx + 1) * block_N,
                                    by * block_M:(by + 1) * block_M])

    return main
# test_int4_int8_convert_close()

def assert_matmul_int8xint4_correctness(M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, num_bits, threads):
    kernel = matmul_int8xint4(M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, threads, num_bits)
    # kernel = matmul_int8xint4(M, N, K, in_dtype, out_dtype, accum_dtype)
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()
    input_tensors = profiler._get_inputs()
    tl_output = kernel(*input_tensors)
    ref_output = ref_program(*input_tensors)

    print(tl_output)
    print(ref_output)
    torch.testing.assert_close(tl_output, ref_output, rtol=0.01, atol=0.01)
    
    print("all check pass")

    latency = profiler.do_bench(
        warmup=10,
        rep=100,
    )
    tflops = (2 * M * N * K) / (latency * 1e-3) / 1e12

    # print(kernel.get_kernel_source())

    print(f"tilelang w4a8 latency: {latency}ms")
    print(f"tilelang w4a8 tflops: {tflops}")



assert_matmul_int8xint4_correctness(256, 256, 256, "int8", "int32", "int32", 128, 128, 128, 2, 4, 128)
# assert_matmul_int8xint4_correctness(8192, 8192, 8192, "int8", "int32", "int32", 128, 128, 256, 3, 4, 512)