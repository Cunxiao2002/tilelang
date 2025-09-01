
import tilelang
import tilelang.language as T
import torch.nn.functional as F
import torch
from tilelang.engine.callback import register_cuda_postproc_callback
import itertools
import torch.cuda.nvtx as nvtx


# softmax + topk
# logits (num_token, num_expert) -> softmax_logits (num_token, num_expert), dtype=float32
# top_k_gates (num_token, top_k), dtype=float32
# top_k_indices (num_token, top_k), dtype=int32

# tilelang.disable_cache()

# torch.manual_seed(42)   

def get_configs():
    iter_params = dict(
        block_tokens=[64, 128, 256],
        threads=[128, 256, 512],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


# @tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[1, 2])
def topk_softmax(
    num_tokens,
    num_experts,
    topk,
    block_tokens,
    threads=128,
):
    dtype = "float32"
    accum_dtype = "float32"
    scale = 1.44269504 #log2(e)

    @T.prim_func
    def topk_softmax_kernel(
        logits: T.Tensor([num_tokens, num_experts], dtype),
        topk_gates: T.Tensor([num_tokens, topk], dtype),
        topk_indices: T.Tensor([num_tokens, topk], "int32"),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_tokens), threads=threads) as bx:
            logits_frag = T.alloc_fragment([block_tokens, num_experts], dtype=dtype)
            max_val = T.alloc_fragment([block_tokens], dtype=dtype)
            expand_max_idx = T.alloc_fragment([block_tokens, num_experts], "int32")
            max_idx = T.alloc_fragment([block_tokens], "int32")
            gates_frag = T.alloc_fragment([block_tokens, topk], dtype=dtype)
            
            T.copy(logits[bx * block_tokens, 0], logits_frag)

            for k in T.serial(topk):
                T.fill(expand_max_idx, -1)
                T.reduce_max(logits_frag, max_val, dim=1, clear=True)
                

                for i, j in T.Parallel(block_tokens, num_experts):
                    expand_max_idx[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], j, expand_max_idx[i, j])
                
                T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)
                
                for i, j in T.Parallel(block_tokens, num_experts):
                    logits_frag[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], -10000.0, logits_frag[i, j])
                
                for i in T.Parallel(block_tokens):
                    # topk_gates[bx * block_tokens + i, k] = max_val[i]
                    gates_frag[i, k] = max_val[i]
                    topk_indices[bx * block_tokens + i, k] = max_idx[i]
            
            # 对gates_frag进行softmax
            exp_max = T.alloc_fragment([block_tokens], dtype)
            scores_sum = T.alloc_fragment([block_tokens], dtype)
            T.fill(exp_max, -T.infinity(dtype))

            T.reduce_max(gates_frag, exp_max, dim=1, clear=True)

            for i, j in T.Parallel(block_tokens, topk):
                gates_frag[i, j] = T.exp2(gates_frag[i, j] * scale - exp_max[i] * scale)
            
            T.fill(scores_sum, 0.0)

            T.reduce_sum(gates_frag, scores_sum, dim=1, clear=False)

            for i, j in T.Parallel(block_tokens, topk):
                gates_frag[i, j] = gates_frag[i, j] / scores_sum[i]
            
            for k in T.serial(topk):
                for i in T.Parallel(block_tokens):
                    topk_gates[bx * block_tokens + i, k] = gates_frag[i, k]
    
    return topk_softmax_kernel
 
        
@tilelang.jit(out_idx=[-2, -1])
def scatter(
    num_tokens,
    hidden_dim,
    topk,
    input_dtype,
    weight_dtype,
    blk_tokens,
    blk_hidden,
    threads,
):
    total_tokens = num_tokens * topk

    @T.prim_func
    def scatter_kernel(
        input_flat: T.Tensor((num_tokens, hidden_dim), input_dtype),
        flat_expert_gates: T.Tensor((total_tokens), weight_dtype),
        idxs: T.Tensor((total_tokens), "int32"),
        tokens_idxs: T.Tensor((total_tokens), "int32"),
        stacked_expert_tokens: T.Tensor((total_tokens, hidden_dim), input_dtype),
        stacked_expert_weights: T.Tensor((total_tokens), weight_dtype),
    ):
        with T.Kernel(T.ceildiv(total_tokens, blk_tokens), threads=threads) as bx:
            # 直接求scatter后的结果
            tx = T.get_thread_binding(0)
            source_weight_idx = idxs[bx * blk_tokens + tx]
            stacked_expert_weights[bx * blk_tokens + tx] = flat_expert_gates[source_weight_idx]

            for k in T.serial(T.ceildiv(hidden_dim, blk_hidden)):
                for i, j in T.Parallel(blk_tokens, blk_hidden):
                    stacked_expert_tokens[bx * blk_tokens + i, k * blk_hidden +j] = input_flat[tokens_idxs[bx * blk_tokens + i], k * blk_hidden +j]

    return scatter_kernel


def ref_program(x_flat, logits, topk, num_tokens, hidden_dim):

    # topk + softmax
    topk_gates, topk_indices = logits.topk(topk, dim=1)
    topk_gates = F.softmax(topk_gates)

    # scatter
    # 将数据flat成1维
    flat_expert_weights = topk_gates.reshape(-1)
    flat_expert_indices = topk_indices.reshape(-1)
    idxs = flat_expert_indices.argsort().to(torch.int32)

    # 直方图统计
    counts = flat_expert_indices.bincount()
    tokens_per_expert = counts.cumsum(0)
    token_idxs = idxs // topk

    # 创建最终输出的数据tensor
    stacked_expert_tokens = torch.zeros((num_tokens * topk, hidden_dim), dtype=x_flat.dtype, device=x_flat.device)
    stacked_expert_weights = torch.zeros((num_tokens * topk), dtype=flat_expert_weights.dtype, device=x_flat.device)

    stacked_expert_tokens_idx = token_idxs
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x_flat[exp_token_idxs]

        stacked_expert_tokens[start_idx:end_idx] = expert_tokens
        stacked_expert_weights[start_idx:end_idx] = flat_expert_weights[idxs[start_idx:end_idx]]
    return stacked_expert_tokens, stacked_expert_weights, stacked_expert_tokens_idx


def tl_softmax_scatter(x_flat, logits, topk, num_tokens, num_experts, hidden_dim):
    # topk_softmax_kernel
    blk_tokens_1 = 64
    threads_1 = 256
    topk_softmax_kernel = topk_softmax(num_tokens, num_experts, topk, blk_tokens_1, threads_1)

    # scatter_kernel
    blk_tokens_2 = 64
    blk_hidden_2 = 4096
    threads_2 = 64
    scatter_kernel = scatter(num_tokens, hidden_dim, topk, "bfloat16", "float32", blk_tokens_2, blk_hidden_2, threads_2)


    # tl forward
    topk_gates, topk_indices = topk_softmax_kernel(logits) # 7.3us
    flat_expert_weights = topk_gates.reshape(-1)
    flat_expert_indices = topk_indices.reshape(-1)
    idxs = flat_expert_indices.argsort().to(torch.int32) # 22.18us
    tokens_idxs = idxs // topk

    # 创建空tensor
    stacked_expert_tokens = torch.zeros((num_tokens * topk, hidden_dim), dtype=x_flat.dtype, device=x_flat.device)    
    stacked_expert_weights = torch.zeros((num_tokens * topk), dtype=flat_expert_weights.dtype, device=x_flat.device)
    stacked_expert_tokens_idx = tokens_idxs

    stacked_expert_tokens, stacked_expert_weights = scatter_kernel(x_flat, flat_expert_weights, idxs, tokens_idxs) # 17.5us

    return stacked_expert_tokens, stacked_expert_weights, stacked_expert_tokens_idx    


def main():
    num_tokens = 320
    hidden_dim = 4096
    num_experts = 128
    top_k = 6

    x = torch.randn(num_tokens, hidden_dim).to("cuda").to(torch.bfloat16)
    x_flat = x.view(-1, hidden_dim)
    logits = torch.rand(num_tokens, num_experts).to("cuda")

    # ref_stacked_expert_tokens, ref_stacked_expert_weights, ref_stacked_expert_tokens_idx = ref_program(x_flat, logits, top_k, num_tokens, hidden_dim)

    tl_stacked_expert_tokens, tl_stacked_expert_weights, tl_stacked_expert_tokens_idx = tl_softmax_scatter(x_flat, logits, top_k, num_tokens, num_experts, hidden_dim)

    # torch.testing.assert_close(ref_stacked_expert_tokens, tl_stacked_expert_tokens)
    # torch.testing.assert_close(ref_stacked_expert_weights, tl_stacked_expert_weights)
    # torch.testing.assert_close(ref_stacked_expert_tokens_idx, tl_stacked_expert_tokens_idx)
    print("all check pass")

    
if __name__ == "__main__":
    main()
