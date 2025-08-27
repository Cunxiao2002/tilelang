import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
from tilelang.autotuner import *
import torch.nn.functional as F
import itertools

def get_configs():
    iter_params = dict(
        block_tokens=[32, 64, 128, 256],
        block_experts=[32, 64, 128, 256],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]
    

@autotune(configs=get_configs())
@tilelang.jit(out_idx=[-1])
def tl_swiglu(
    num_tokens,
    num_experts,
    dtype,
    block_tokens,
    block_experts,
    threads,
):
    scale = 1.44269504  # log2(e)
    @T.prim_func
    def kernel(
        gate_logits: T.Tensor([num_tokens, num_experts], dtype),
        up_logits: T.Tensor([num_tokens, num_experts], dtype),
        output: T.Tensor([num_tokens, num_experts], dtype)
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_tokens), T.ceildiv(num_experts, block_experts), threads=threads) as (bx, by):
            gate_logits_local = T.alloc_fragment([block_tokens, block_experts], dtype=dtype)
            up_logits_local = T.alloc_fragment([block_tokens, block_experts], dtype=dtype)

            T.copy(gate_logits[bx * block_tokens, by * block_experts], gate_logits_local)
            T.copy(up_logits[bx * block_tokens, by * block_experts], up_logits_local)

            for i, j in T.Parallel(block_tokens, block_experts):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]
            
            T.copy(up_logits_local, output[bx * block_tokens, by * block_experts])


    return kernel

class SwiGLU(nn.Module):

    def forward(self, gate_logits, up_logits):
        out = gate_logits * F.sigmoid(gate_logits) * up_logits
        return out

# 测试代码
if __name__ == "__main__":
    # 创建输入数据 (batch_size=2, seq_len=3, features=4)
    # x = torch.randn(2, 3, 4)

    num_tokens = 1024
    num_experts = 128
    dtype = "bfloat16"
    block_tokens = 128
    block_experts = 128
    threads = 128
    
    gate_logits = torch.randn([num_tokens, num_experts], dtype=torch.bfloat16).to("cuda")
    up_logits= torch.randn([num_tokens, num_experts], dtype=torch.bfloat16).to("cuda")

    # kernel = tl_swiglu(num_tokens, num_experts, dtype, block_tokens, block_experts, threads)
    kernel = tl_swiglu(num_tokens, num_experts, dtype)

    tl_output = kernel(gate_logits, up_logits)
    
    # 初始化SwiGLU模块
    torch_swiglu = SwiGLU()
    
    # 前向传播
    output = torch_swiglu(gate_logits, up_logits)

    # print(tl_output)
    # print(output)

    torch.testing.assert_close(tl_output, output)
    print("All checks pass.")

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(warmup=500)
    print("Tilelang: {:.10f} ms".format(latency))