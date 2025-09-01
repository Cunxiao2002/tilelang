import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
from tilelang.autotuner import *
import torch.nn.functional as F
import itertools

# tilelang.disable_cache()

# def get_configs():
#     iter_params = dict(
#         blk_tokens=[64, 128, 256],
#         blk_hidden=[64, 128, 256],
#         threads=[128, 256, 512]
#     )
#     return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

# @tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[-3, -2, -1])
def scatter(
    num_tokens,
    hidden_dim,
    topk,
    dtype,
    blk_tokens,
    blk_hidden,
    threads,
):
    total_tokens = num_tokens * topk

    @T.prim_func
    def scatter_kernel(
        input_flat: T.Tensor((num_tokens, hidden_dim), dtype),
        flat_expert_gates: T.Tensor((total_tokens), dtype),
        idxs: T.Tensor((total_tokens), "int32"),
        tokens_idxs: T.Tensor((total_tokens), "int32"),
        stacked_expert_tokens: T.Tensor((total_tokens, hidden_dim), dtype),
        stacked_expert_tokens_idx: T.Tensor((total_tokens), "int32"),
        stacked_expert_weights: T.Tensor((total_tokens), dtype),
    ):
        with T.Kernel(T.ceildiv(total_tokens, blk_tokens), threads=threads) as bx:
            # 直接求scatter后的结果
            tx = T.get_thread_binding(0)
            source_token_idx = tokens_idxs[bx * blk_tokens + tx]
            source_weight_idx = idxs[bx * blk_tokens + tx]

            stacked_expert_weights[bx * blk_tokens + tx] = flat_expert_gates[source_weight_idx]
            stacked_expert_tokens_idx[bx * blk_tokens + tx] = source_token_idx

            for k in T.serial(T.ceildiv(hidden_dim, blk_hidden)):
                for i, j in T.Parallel(blk_tokens, blk_hidden):
                    stacked_expert_tokens[bx * blk_tokens + i, k * blk_hidden +j] = input_flat[tokens_idxs[bx * blk_tokens + i], k * blk_hidden +j]

    return scatter_kernel



def ref_program(x_flat, flat_expert_weights, flat_expert_indices, idxs, num_tokens, topk, hidden_dim):
    # 将下面的计算放到main中
    # x_flat = x.view(-1, x.shape[-1])
    # flat_expert_weights = topk_gates.reshape(-1, 1)
    # flat_expert_indices = topk_indices.reshape(-1)
    # idxs = flat_expert_indices.argsort()

    stacked_expert_tokens = torch.zeros((num_tokens * topk, hidden_dim), dtype=x_flat.dtype, device=x_flat.device)
    stacked_expert_tokens_idx = torch.zeros((num_tokens * topk, 1), dtype=torch.int32, device=x_flat.device)
    stacked_expert_weights = torch.zeros((num_tokens * topk, 1), dtype=flat_expert_weights.dtype, device=x_flat.device)
    
    # 直方图统计要放到kernel中
    counts = flat_expert_indices.bincount()
    tokens_per_expert = counts.cumsum(0)
    token_idxs = idxs // topk

    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x_flat[exp_token_idxs]

        stacked_expert_tokens[start_idx:end_idx] = expert_tokens
        stacked_expert_tokens_idx[start_idx:end_idx, 0] = exp_token_idxs
        stacked_expert_weights[start_idx:end_idx] = flat_expert_weights[idxs[start_idx:end_idx]]
        
    
    return stacked_expert_tokens, stacked_expert_tokens_idx, stacked_expert_weights


def main():
    num_tokens = 320
    hidden_dim = 4096
    num_experts = 128
    topk = 6
    x = torch.randn(num_tokens, hidden_dim).to("cuda").to(torch.float32)
    topk_gates = torch.rand(num_tokens, topk).to("cuda").to(torch.float32)
    topk_indices = torch.randint(0, num_experts, (num_tokens, topk)).to("cuda")

    x_flat = x.view(-1, hidden_dim)
    flat_expert_weights = topk_gates.reshape(-1, 1)
    flat_expert_indices = topk_indices.reshape(-1)

    # 计算排序后的索引，并reshape为kernel期望的形状
    idxs = flat_expert_indices.argsort()

    # 运行ref_program得到参考结果
    ref_stacked_expert_tokens, ref_stacked_expert_tokens_idx, ref_stacked_expert_weights = ref_program(
        x_flat, flat_expert_weights, flat_expert_indices, idxs.squeeze(-1), num_tokens, topk, hidden_dim)

    # 调用tilelang kernel
    blk_tokens = 128
    blk_hidden = 128 # 进一步增加blk_hidden
    threads = 128

    # 将torch dtype转换为tilelang dtype字符串
    dtype_str = "float32" if x.dtype == torch.float32 else ("bfloat16" if x.dtype == torch.bfloat16 else str(x.dtype).split('.')[-1])
    kernel = scatter(num_tokens, hidden_dim, topk, dtype_str, blk_tokens, blk_hidden, threads)
    # kernel = scatter(num_tokens, hidden_dim, num_experts, topk, dtype_str)

    # 准备tilelang kernel的输入
    tl_input_flat = x_flat
    tl_flat_expert_gates = flat_expert_weights  # 使用flatten后的形状 (total_tokens, 1)
    tl_flat_expert_indices = flat_expert_indices.to(torch.int32)  # 转换为int32
    tokens_idxs = idxs // topk

    # 准备输出张量
    total_tokens = num_tokens * topk
    tl_stacked_expert_tokens = torch.zeros((total_tokens, hidden_dim), dtype=x_flat.dtype, device=x_flat.device)
    tl_stacked_expert_tokens_idx = torch.zeros((total_tokens, 1), dtype=torch.int32, device=x_flat.device)
    tl_stacked_expert_weights = torch.zeros((total_tokens, 1), dtype=flat_expert_weights.dtype, device=x_flat.device)

    # 调用kernel - 让tilelang自己创建输出张量
    tl_stacked_expert_tokens, tl_stacked_expert_tokens_idx, tl_stacked_expert_weights = kernel(
        tl_input_flat, tl_flat_expert_gates, idxs.to(torch.int32), tokens_idxs.to(torch.int32))

    print(kernel.config)
    print(kernel.get_kernel_source())
    

    # test accuracy
    torch.testing.assert_close(tl_stacked_expert_tokens, ref_stacked_expert_tokens)
    torch.testing.assert_close(tl_stacked_expert_tokens_idx.view(-1, 1), ref_stacked_expert_tokens_idx)
    print("tl_stacked_expert_weights:")
    print(tl_stacked_expert_weights)
    print("ref_stacked_expert_weights:")
    print(ref_stacked_expert_weights)
    torch.testing.assert_close(tl_stacked_expert_weights.view(-1, 1), ref_stacked_expert_weights)

    # profile
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench(warmup=50)
    print(f"tilelang latency: {tilelang_latency}")



if __name__ == "__main__":
    main()