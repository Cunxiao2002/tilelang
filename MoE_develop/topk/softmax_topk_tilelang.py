
import tilelang
import tilelang.language as T
import torch.nn.functional as F
import torch
from tilelang.engine.callback import register_cuda_postproc_callback
import itertools


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
        

def ref_program(logits, top_k):

    top_k_gates, top_k_indices = logits.topk(top_k, dim=1)
    top_k_gates = F.softmax(top_k_gates)

    return top_k_gates, top_k_indices.to(torch.int32)


def main():
    num_tokens = 320
    num_experts = 128
    top_k = 6
    block_tokens = 64
    threads = 256

    logits = torch.rand(num_tokens, num_experts).to("cuda")

    # kernel = topk_softmax(num_tokens, num_experts, top_k, block_tokens)
    kernel = topk_softmax(num_tokens=num_tokens, num_experts=num_experts, topk=top_k, block_tokens=block_tokens, threads=threads)
    tl_gates, tl_indices = kernel(logits)
    # print(f"tl_gates:")
    # print(tl_gates)
    
    # print(kernel.get_kernel_source())
    print(kernel.config)

    # torch_gates, torch_indices = ref_program(logits, top_k)
    # print(f"torch_gates:")
    # print(torch_gates)
    
    # test accuracy
    # torch.testing.assert_close(tl_gates, torch_gates)
    # torch.testing.assert_close(tl_indices, torch_indices)

    # profile
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench(warmup=500)
    print(f"Tilelang latency: {tilelang_latency}")

    
if __name__ == "__main__":
    main()
