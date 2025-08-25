import tilelang
import tilelang.language as T
import torch
from tilelang.engine.callback import register_cuda_postproc_callback

torch.manual_seed(42)

tilelang.disable_cache()

@tilelang.jit(out_idx=[1, 2])
def Topk_val(
    num_tokens,
    num_experts,
    topk,
    block_M,
    threads=128,
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
            max_idx = T.alloc_fragment([block_M], "int32")
            T.copy(logits[bx * block_M, 0], logits_frag) 

            T.fill(max_idx, -1)
            # T.copy的logits中计算的是某个block的首地址，因此dim1=0的意思是每次都从第0列开始

            # T.Parallel是将其block内部的计算分发给所有的thread

            for k in T.Pipelined(topk):
                T.reduce_max(logits_frag, max_val, dim=1, clear=True)

                for i, j in T.Parallel(block_M, num_experts):
                    max_idx[i] = T.if_then_else(max_val[i] == logits_frag[i, j], j, max_idx[i])
                
                
                for i, j in T.Parallel(block_M, num_experts):
                    logits_frag[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], 0, logits_frag[i, j])
                
                
                # T.print(max_idx)
                T.copy(max_val, topk_gates[bx * block_M, k])
                T.copy(max_idx, topk_indices[bx * block_M, k])
                
    return kernel


def ref_program(logits, top_k):
    top_k_gates, top_k_indices = logits.topk(top_k, dim=1)
    return top_k_gates, top_k_indices


def main():
    num_tokens = 320
    num_expert = 128
    top_k = 2
    block_M = 64
    block_N = 64
    block_K = 64

    logits = torch.rand(num_tokens, num_expert).to("cuda")
    # print(logits)

    @register_cuda_postproc_callback
    def tilelang_callback_cuda_postproc(code, _):
        code = ''' 
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(float* __restrict__ logits, float* __restrict__ topk_gates, int* __restrict__ topk_indices);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(float* __restrict__ logits, float* __restrict__ topk_gates, int* __restrict__ topk_indices) {
  float logits_frag[64];
  int max_idx[16];
  float max_val[16];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float4*)(logits_frag + (i * 4)) = *(float4*)(logits + (((((int)blockIdx.x) * 8192) + (i * 512)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  // for (int i_1 = 0; i_1 < 16; ++i_1) {
  //   max_idx[i_1] = 1;
  // }
  for (int k = 0; k < 2; ++k) {
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      max_idx[i_1] = -1;
    }

    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      max_val[i_2] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        max_val[i_2] = max(max_val[i_2], logits_frag[((i_2 * 4) + rv)]);
      }
      max_val[i_2] = tl::AllReduce<tl::MaxOp, 32, 1, 0, 128>::run_hopper(max_val[i_2], (&(workspace[0])));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 64; ++i_3) {
      int condval_idx;
      if ((max_val[(i_3 >> 2)] == logits_frag[i_3])) {
        condval_idx = (((((int)threadIdx.x) & 31) * 4) + (i_3 & 3));
      } else {
        condval_idx = max_idx[(i_3 >> 2)];
      }
      max_idx[(i_3 >> 2)] = condval_idx;
    }

    for (int i_5 = 0; i_5 < 16; ++i_5) {
      max_idx[i_5] = tl::AllReduce<tl::MaxOp, 32, 1, 0, 128>::run_hopper(max_idx[i_5], ((int*)&(workspace[0])));
    }
    
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      float condval_1;
      if ((max_val[(i_4 >> 2)] == logits_frag[i_4])) {
        condval_1 = 0.000000e+00f;
      } else {
        condval_1 = logits_frag[i_4];
      }
      logits_frag[i_4] = condval_1;
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      // if (((i_5 * 2) + (((((int)threadIdx.x) >> 5) + k) >> 1)) < 1) {
        topk_gates[((((((int)blockIdx.x) * 128) + (i_5 * 4) * 2) + (((int)threadIdx.x) >> 5) * 2) + k)] = max_val[i_5];
      // }
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      // if (((i_6 * 2) + (((((int)threadIdx.x) >> 5) + k) >> 1)) < 1) {
        topk_indices[((((((int)blockIdx.x) * 128) + (i_6 * 4) * 2) + (((int)threadIdx.x) >> 5) * 2) + k)] = max_idx[i_6];
      // }
    }
  }
}
        '''
        return code

    kernel = Topk_val(num_tokens, num_expert, top_k, block_M)
    tl_gates, tl_indices = kernel(logits)
    print(kernel.get_kernel_source())
    # print(tl_gates)
    print(tl_indices)

    topk_gates, topk_indices = logits.topk(top_k, dim=1)
    topk_indices = topk_indices.to(torch.int32)
    # print(topk_gates)
    print(f"torch topk \n")
    print(topk_indices)

    # ref_gates, ref_indices = ref_program(logits, top_k)

    torch.testing.assert_close(tl_gates, topk_gates)
    torch.testing.assert_close(tl_indices, topk_indices)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(warmup=500, input_tensors=[logits])
    print(f"Tilelang: {latency} ms")
     


if __name__ == "__main__":
    main()