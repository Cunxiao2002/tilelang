import tilelang
import tilelang.language as T
import torch
from tilelang.engine.callback import register_cuda_postproc_callback


@tilelang.jit(out_idx=[1, 2], pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def per_token_cast_to_int8(M, N, blk_m):
    dtype = "bfloat16"
    int8_min = -128
    int8_max = 127

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((M, N), dtype),
        X_int8: T.Tensor((M, N), "int8"),
        X_scale: T.Tensor((M, 1), dtype)  # 每行一个缩放因子
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:  # 移除列分组
            x_local = T.alloc_fragment((blk_m, N), dtype)
            amax_local = T.alloc_fragment((blk_m,), dtype)
            x_int8_local = T.alloc_fragment((blk_m, N), "int8")

            T.copy(X[bx * blk_m, 0], x_local)
            T.reduce_absmax(x_local, amax_local, dim=1)

            for i in T.Parallel(blk_m):
                amax_local[i] = T.max(amax_local[i], 1e-4)
            
            for i, j in T.Parallel(blk_m, N):
                x_int8_local[i, j] = T.clamp(T.floor(x_local[i, j] / amax_local[i] * int8_max + 0.5), int8_min, int8_max)
            
            T.copy(x_int8_local, X_int8[bx * blk_m, 0])

    
    return per_token_cast



def quantize_per_token_and_smooth(x):
    # expert_scales = input_smooth_scale[expert_indices]
    # input_smoothed = x * input_smooth_scale
    amax, _ = torch.max(torch.abs(x), dim=1, keepdim=True)
    amax = torch.clamp(amax, min=1e-8)
    input_quant = x / amax * 127.0
    input_quant = torch.floor(input_quant.to(torch.float) + 0.5)
    input_quant = torch.clip(input_quant, -127.0, 127.0).to(torch.int8)

    return input_quant, amax / 127.0

def main():
    num_tokens = 320
    topk = 6
    hidden_dim = 4096
    block_tokens = 64
    x = torch.randn(num_tokens*topk, hidden_dim).to(torch.bfloat16).to("cuda")

    @register_cuda_postproc_callback
    def tilelang_callback_cuda_postproc(code, _):
        code = '''
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void per_token_cast_kernel(bfloat16_t* __restrict__ X, signed char* __restrict__ X_int8);
extern "C" __global__ void __launch_bounds__(128, 1) per_token_cast_kernel(bfloat16_t* __restrict__ X, signed char* __restrict__ X_int8) {
  bfloat16_t x_local[2048];
  bfloat16_t amax_local[64];
  extern __shared__ __align__(1024) bfloat16_t workspace[];
  signed char x_int8_local[2048];
  #pragma unroll
  for (int i = 0; i < 256; ++i) {
    *(uint4*)(x_local + (i * 8)) = *(uint4*)(X + (((((int)blockIdx.x) * 262144) + (i * 1024)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 64; ++i_1) {
    amax_local[i_1] = bfloat16_t(0.000000e+00f);
    #pragma unroll
    for (int rv = 0; rv < 32; ++rv) {
      amax_local[i_1] = (bfloat16_t)max(max(amax_local[i_1], x_local[(((i_1 * 32) + ((rv & 3) * 8)) + (rv >> 2))]), (bfloat16_t(0.000000e+00f) - (bfloat16_t)min(amax_local[i_1], x_local[(((i_1 * 32) + ((rv & 3) * 8)) + (rv >> 2))])));
    }
    amax_local[i_1] = tl::AllReduce<tl::MaxOp, 128, 1, 0, 128>::run_hopper(amax_local[i_1], (&(workspace[0])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    amax_local[i_2] = ((bfloat16_t)max(((float)amax_local[i_2]), 1.000000e-04f));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2048; ++i_3) {
    x_int8_local[i_3] = ((signed char)min(max(floorf((((float)((x_local[i_3] / amax_local[(i_3 >> 5)]) * bfloat16_t(1.270000e+02f))) + 5.000000e-01f)), -1.280000e+02f), 1.270000e+02f));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 256; ++i_4) {
    *(int2*)(X_int8 + (((((int)blockIdx.x) * 262144) + (i_4 * 1024)) + (((int)threadIdx.x) * 8))) = *(int2*)(x_int8_local + (i_4 * 8));
  }
}       
        '''
        return code

    
    kernel = per_token_cast_to_int8(num_tokens*topk, hidden_dim, block_tokens)

    print(kernel.get_kernel_source())

    tl_quant, tl_scale = kernel(x)

    torch_quant, torch_scale = quantize_per_token_and_smooth(x)

    torch.testing.assert_close(tl_quant, torch_quant)
    # torch.testing.assert_close(tl_scale, torch_scale)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(warmup=500)
    print(f"tilelang latency: {latency:.10f} ms")
    

if __name__ == "__main__":
    main()