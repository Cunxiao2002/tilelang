import tilelang
import tilelang.language as T
import torch.nn.functional as F

# softmax + topk
# logits (num_token, num_expert) -> softmax_logits (num_token, num_expert), dtype=float32
# top_k_gates (num_token, top_k), dtype=float32
# top_k_indices (num_token, top_k), dtype=int32
@tilelang.jit
def topk_gate(
    num_tokens,
    num_expert,
    top_k,
    block_M,
    block_N,
    block_K,
    stages=1,
    threads=128,
):
    dtype = "float32"
    accum_dtype = "float32"
    scale = (1.0 / num_expert) ** 0.5 * 1.44269504 # log2(e)

    
    @T.macro
    def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
    ):
        max_val = T.alloc_fragment([block_M], accum_dtype)
        scores_sum = T.alloc_fragment([block_M], accum_dtype)

        T.fill(max_val, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, max_val, dim=1, clear=False)

        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - max_val[i] * scale)

        T.reduce_sum(acc_s, scores_sum, dim=1, clear=False)

        for i in T.Parallel(block_M):
            acc_s[i, j] = acc_s[i, j] / scores_sum[i]
        T.copy(acc_s, acc_s_cast)
    
    @T.macro
    def TopK(
        logits: T.Tensor([num_tokens, num_expert], dtype),
        top_k_gates: T.Tensor([num_tokens, top_k], dtype),
        top_k_indices: T.Tensor([num_tokens, top_k], dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_M), T.ceildiv(num_expert, block_N)) as (by, bx):
            cur_vals = T.alloc_fragment([block_M, block_N], dtype)
            candidate_idx = T.alloc_fragment([block_M, block_N], "int32")

            top_k_gates_local = T.alloc_fragment([block_M, top_k], dtype)
            top_k_indices_local = T.alloc_fragment([block_M, top_k], "int32")

            max_val = T.alloc_fragment([block_M], dtype)
            max_idx = T.alloc_fragment([block_M], "int32")
            T.copy(logits, cur_vals)
            
            for k in T.Parallel(top_k):
                T.fill(max_val, -T.infinity(dtype))
                T.reduce_max(cur_vals, max_val, dim=1, clear=False)

                for i, j in T.Parallel(block_M, block_N):
                    candidate_idx[i, j] = T.if_then_else(
                        cur_vals[i, j] == max_val[i],
                        bx * block_N + j,
                        num_expert
                    )

                T.fill(max_idx, 128)
                T.reduce_min(candidate_idx, max_idx, dim=1, clear=False)
                
                for i in T.Parallel(block_M):
                    top_k_gates_local[i, k] = max_val[i]
                    top_k_indices_local[i, k] = max_idx[i]

                    for j in T.Parallel(block_N):
                        cur_vals[i, j] = T.if_then_else(
                            j + (bx * block_N) == max_idx[i],
                            -T.infinity(dtype),
                            cur_vals[i, j]
                        )
                
            T.copy(top_k_gates_local, top_k_gates[by * block_M, 0])
            T.copy(top_k_indices_local, top_k_indices[by * block_M, 0])
                
    

            
    
    @T.prim_func
    def softmax_topk(
        logits: T.Tensor([num_tokens, num_expert], dtype),
        top_k_gates: T.Tensor([num_tokens, top_k], dtype),
        top_k_indices: T.Tensor([num_tokens, top_k], dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_M), T.ceildiv(num_expert, block_N)) as (by, bx):
            pass
                        

                



            
def ref_program(logits, top_k):

    logits = F.softmax(logits)
    top_k_gates, top_k_indices = logits.topk(top_k, dim=1)

    return top_k_gates, top_k_indices


def main():
    num_tokens, num_expert, top_k = 320, 128, 6
    block_M, block_N, block_K = 128, 64, 128
    kernel = topk_gate(num_tokens, num_expert, top_k, block_M, block_N, block_K)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks passed!")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("ref_program: {:.2f} ms".format(latency))

    latency = profiler.do_bench(n_warmup=10, n_repeat=10)
    print("tilelang: {:.4f} ms".format(latency))
    
if __name__ == "__main__":
    main()
