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
  for (int k = 0; k < 6; ++k) {
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
        topk_gates[((((((int)blockIdx.x) * 384) + (i_5 * 4) * 6) + (((int)threadIdx.x) >> 5) * 6) + k)] = max_val[i_5];
      // }
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      // if (((i_6 * 2) + (((((int)threadIdx.x) >> 5) + k) >> 1)) < 1) {
        topk_indices[((((((int)blockIdx.x) * 384) + (i_6 * 4) * 6) + (((int)threadIdx.x) >> 5) * 6) + k)] = max_idx[i_6];
      // }
    }
  }
}