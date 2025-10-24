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

torch.manual_seed(42)
# tilelang.disable_cache()

'''
liquid量化方案:
fp16 -> int8 -> uint8 -> uint4

dequant gemm:
(Qu4 x Su4 + a) ^ 0x80
uint4的scale采用uint8来存储, 可以采用"IMAD“指令来做反量化
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

# 一般来说B是weight，A是activation
def liquid_w4a8_dequant_gemm(M, N, K, block_M, block_N, block_K, num_stages, threads, in_dtype="int8", out_dtype="bfloat16", storage_dtype="uint8", num_bits=4):
    @T.prim_func
    def liquid_gemm(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), storage_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(M // block_M, N // block_N, threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, K), in_dtype)
            B_shared = T.alloc_shared((block_N, K), storage_dtype)
            B_local = T.alloc_fragment((block_N, K), storage_dtype)
            B_dequantize_local = T.alloc_fragment((block_N, K), in_dtype)
            B_dequantize_prev_local = T.alloc_fragment((block_N, K), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
