import tilelang
import tilelang.language as T
import torch
from tvm import tir, DataType
# torch.manual_seed(42)

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
    
    # 使用左移和算术右移
    # 1. 先将低4位数据左移4位，放到int8的高4位
    # 2. 然后算术右移4位，自动进行符号扩展
    i8_shifted = tir.reinterpret("int8", i4 << tir.const(4, "uint8"))
    i8 = i8_shifted >> tir.const(4, "int8")  # 注意这里使用int8类型的右移（算术右移）
    
    return i8

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

            for k in T.Pipelined(0, T.ceildiv(K, block_K)):
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


@tilelang.jit(out_idx=[1])
def fast_quant_test(N, K, block_N, block_K, in_dtype, num_bits=4, threads=128):
    assert in_dtype == "int8"
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
    B_shape = (N, K // num_elems_per_byte)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequant_shared_shape = (block_N, block_K)
    
    # 定义局部变量大小
    MAX_TRANSACTION_SIZE_BITS = 128
    local_size = MAX_TRANSACTION_SIZE_BITS // DataType(in_dtype).bits # 128 // 8 = 16
    local_compress_size = local_size // num_elems_per_byte # 16 // 2 = 8
    group_size = 32  # 根据实际需求调整
    block_QK = block_K // num_elems_per_byte  # 添加 block_QK 定义

    @T.macro
    def _fast_dequant_kernel(B_shared, B_dequant_shared, k):
        tx = T.get_thread_binding()

        B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
        B_dequant_local_thread = T.alloc_local((local_size,), in_dtype)
        # scale_local_thread = T.alloc_local((1,), storage_dtype)
        
        for i in T.serial(0, block_N * block_K // threads // local_size):
            index_base = i * threads * local_compress_size + tx * local_compress_size
            for v in T.vectorized(0, local_compress_size):
                index = index_base + v
                B_local_thread[v] = B_shared[index // block_QK, index % block_QK]
            # scale_index = index_base // (group_size // num_elems_per_byte)
            # si = scale_index // (block_K // group_size)
            # sj = scale_index % (block_K // group_size)
            # scale_local_thread[0] = scale_shared[si, k * block_K // group_size + sj]

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

    @T.prim_func
    def fast_dequant_uint4_int8(
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((N, K), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_dequant_shared = T.alloc_shared(B_dequant_shared_shape, in_dtype)
            
            for k in T.Pipelined(0, T.ceildiv(K, block_K)):
                # 从全局内存加载到共享内存
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                
                # 调用解量化函数
                _fast_dequant_kernel(B_shared, B_dequant_shared, k)
                
                # 从共享内存复制到全局内存
                T.copy(B_dequant_shared, C[bx * block_N, k * block_K])
    

    return fast_dequant_uint4_int8


def test_int4_int8_convert_close():
    N, K = 128, 128
    block_N, block_K = 64, 64
    tl_convert_kernel_1 = _convert_test(N, K, block_N, block_K, "int8")
    tl_convert_kernel_2 = fast_quant_test(N, K, block_N, block_K, "int8")

    B = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda").to(torch.uint8)
    tl_out_1 = tl_convert_kernel_1(B)
    tl_out_2 = tl_convert_kernel_2(B)

    print(tl_out_1)
    print(tl_out_2)
    assert torch.allclose(tl_out_1, tl_out_2, rtol=0.01, atol=0.01), (tl_out_1, tl_out_2)
    print("pass")

test_int4_int8_convert_close()