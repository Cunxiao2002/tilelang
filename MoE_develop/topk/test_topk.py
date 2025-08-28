import tilelang
import tilelang.language as T
import torch
from tilelang.engine.callback import register_cuda_postproc_callback
import itertools

torch.manual_seed(42)

tilelang.disable_cache()

def get_configs():
    iter_params = dict(
        block_M=[64, 128, 256],
        threads=[128, 256, 512],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[1, 2])
def Topk_val(
    num_tokens,
    num_experts,
    topk,
    block_M,
    threads,
):
    dtype = "float32"
    logits_shape = (num_tokens, num_experts)
    topk_gates_shape = (num_tokens, topk)
    topk_indices_shape = (num_tokens, topk)
    @T.prim_func
    def kernel(
        logits: T.Tensor(logits_shape, dtype),
        topk_gates: T.Tensor(topk_gates_shape, dtype),
        topk_indices: T.Tensor(topk_indices_shape, "int32"),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_M), threads=threads) as bx:
            logits_frag = T.alloc_fragment([block_M, num_experts], dtype=dtype)
            max_val = T.alloc_fragment([block_M], dtype=dtype)
            expand_max_idx = T.alloc_fragment([block_M, num_experts], "int32")
            max_idx = T.alloc_fragment([block_M], "int32")

            
            T.copy(logits[bx * block_M, 0], logits_frag) 

            # T.fill(max_idx, -1)
            # T.copy的logits中计算的是某个block的首地址，因此dim1=0的意思是每次都从第0列开始

            # T.Parallel是将其block内部的计算分发给所有的thread

            for k in T.serial(topk):
                T.fill(expand_max_idx, -1)
                T.reduce_max(logits_frag, max_val, dim=1, clear=True)
                

                for i, j in T.Parallel(block_M, num_experts):
                    expand_max_idx[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], j, expand_max_idx[i, j])
                
                T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)
                # T.copy(max_idx, topk_indices[bx * block_M, 0])
                
                for i, j in T.Parallel(block_M, num_experts):
                    logits_frag[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], -10000.0, logits_frag[i, j])
                

                # T.copy(max_val, topk_gates[bx * block_M, k])
                # T.copy(max_idx, topk_indices[bx * block_M, k])
                
                for i in T.Parallel(block_M):
                    topk_gates[bx * block_M + i, k] = max_val[i]
                    topk_indices[bx * block_M + i, k] = max_idx[i]
    return kernel


def ref_program(logits, top_k):
    top_k_gates, top_k_indices = logits.topk(top_k, dim=1)
    return top_k_gates, top_k_indices


def main():
    num_tokens = 320
    num_expert = 128
    top_k = 6
    block_M = 64

    logits = torch.rand(num_tokens, num_expert).to("cuda")
    # kernel = Topk_val(num_tokens, num_expert, top_k, block_M)
    kernel = Topk_val(num_tokens, num_expert, top_k)
    tl_gates, tl_indices = kernel(logits)
    print(kernel.get_kernel_source())
    # print(tl_gates)
    print(tl_indices)

    topk_gates, topk_indices = logits.topk(top_k, dim=1)
    topk_indices = topk_indices.to(torch.int32)
    print(f"torch topk \n")
    # print(topk_gates)
    print(topk_indices)

    # ref_gates, ref_indices = ref_program(logits, top_k)

    torch.testing.assert_close(tl_gates, topk_gates)
    torch.testing.assert_close(tl_indices, topk_indices)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(warmup=500, input_tensors=[logits])
    print(f"Tilelang: {latency} ms")
     


if __name__ == "__main__":
    main()