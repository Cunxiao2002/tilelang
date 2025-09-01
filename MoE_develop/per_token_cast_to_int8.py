import tilelang
import tilelang.language as T
import torch
from tilelang.engine.callback import register_cuda_postproc_callback
import itertools

# bf16 -> int8
# input_smoothed:bf16 <- input:bf16 * smooth_scale: bf16
# input_quant: int8 <- quant(input_smoothed: bf16)

tilelang.disable_cache()

torch.manual_seed(42)

# def get_configs():
#     iter_params = dict(
#         blk_m=[1, 2, 4, 8, 16, 32],
#         threads=[128, 256],
#     )
#     return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

# @tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[2, 3], pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def per_token_cast_to_int8(M, 
                            N, 
                            blk_m,
                            threads=128):
    dtype = "bfloat16"
    int8_min = -127.0
    int8_max = 127.0

    @T.prim_func
    def per_token_cast(
        x: T.Tensor((M, N), dtype),
        expert_scales: T.Tensor((M, N), dtype), # smooth scale
        x_int8: T.Tensor((M, N), "int8"),
        x_scale: T.Tensor((M), "float32")  # 每行一个缩放因子
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:  # 移除列分组
            x_local = T.alloc_fragment((blk_m, N), dtype)
            expert_scales_local = T.alloc_fragment((blk_m, N), dtype)
            amax_local = T.alloc_fragment((blk_m), "float32")
            x_int8_local = T.alloc_fragment((blk_m, N), "int8")

            T.copy(x[bx * blk_m, 0], x_local)
            T.copy(expert_scales[bx * blk_m, 0], expert_scales_local)

            for i, j in T.Parallel(blk_m, N): 
                x_local[i, j] = x_local[i, j] * expert_scales_local[i, j]

            T.reduce_absmax(x_local, amax_local, dim=1)

            for i in T.Parallel(blk_m):
                amax_local[i] = T.max(amax_local[i], 1e-8)
            
            for i, j in T.Parallel(blk_m, N):
                x_int8_local[i, j] = T.clamp(T.floor(x_local[i, j] / amax_local[i] * int8_max + 0.5), int8_min, int8_max)

            T.copy(x_int8_local, x_int8[bx * blk_m, 0])
            # T.copy(amax_local, x_scale[bx * blk_m])
            for i in T.Parallel(blk_m):
                x_scale[bx * blk_m + i] = amax_local[i] / 127.0

    return per_token_cast



def quantize_per_token_and_smooth(x, expert_scales):
    # expert_scales = input_smooth_scale[expert_indices]
    input_smoothed = x * expert_scales
    amax, _ = torch.max(torch.abs(input_smoothed), dim=1, keepdim=True)
    amax = torch.clamp(amax, min=1e-8)
    input_quant = input_smoothed / amax * 127.0
    input_quant = torch.floor(input_quant + 0.5)
    input_quant = torch.clip(input_quant, -127.0, 127.0).to(torch.int8)

    return input_quant, amax / 127.0

def main():
    num_tokens = 320
    topk = 6
    hidden_dim = 4096
    blk_m = 1
    threads = 1024
    x = torch.randn(num_tokens*topk, hidden_dim).to(torch.bfloat16).to("cuda")
    expert_scales = torch.rand(num_tokens*topk, hidden_dim).to(torch.bfloat16).to("cuda")

#     @register_cuda_postproc_callback
#     def tilelang_callback_cuda_postproc(code, _):
#         code = '''
#         '''
#         return code

    kernel = per_token_cast_to_int8(num_tokens*topk, hidden_dim, blk_m=blk_m, threads=threads)

    print(kernel.get_kernel_source())
    print(kernel.config)

    tl_quant, tl_scale = kernel(x, expert_scales)
    print(tl_quant)

    # torch_quant, torch_scale = quantize_per_token_and_smooth(x, expert_scales)

    # print(f"torch quant logits")
    # print(torch_quant)

    # torch.testing.assert_close(tl_quant, torch_quant)
    # tl_scale = tl_scale.reshape(num_tokens*topk, 1).to(torch.float32)
    # torch_scale = torch_scale.reshape(num_tokens*topk, 1).to(torch.float32)
    # torch.testing.assert_close(tl_scale, torch_scale)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(warmup=500)
    print(f"tilelang latency: {latency:.10f} ms")
    

if __name__ == "__main__":
    main()