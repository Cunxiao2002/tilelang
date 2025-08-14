#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace th = ::torch;
extern "C" __global__ void main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  for (int ko = 0; ko < 8; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(A + ((((((((int)blockIdx.y) * 262144) + (i_1 * 16384)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.z) * 2048)) + (ko * 256)) + ((((int)threadIdx.x) & 31) * 8)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = *(uint4*)(B + ((((((((int)blockIdx.z) * 262144) + (ko * 32768)) + (i_2 * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    tl::fence_proxy_async();
    __syncthreads();
    tl::gemm_ss<64, 32, 256, 4, 1, 0, 0, 0, 256, 32, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(((bfloat16_t*)buf_dyn_shmem)[16384])), (&(C_local[0])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    *(float2*)(((float*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_3 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)) = *(float2*)(C_local + (i_3 * 2));
  }
  tl::fence_proxy_async();
  __syncthreads();
  #pragma unroll
  for (int i_outer = 0; i_outer < 4; ++i_outer) {
    if ((((((((int)threadIdx.x) / 4) >> 2) + i_outer) >> 2) + ((int)blockIdx.y)) < 5) {
      AtomicAddx4((&(C[(((((((int)blockIdx.y) * 8192) + (i_outer * 2048)) + ((((int)threadIdx.x) / 4) * 512)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) % 4) * 4))])), (&(((float*)buf_dyn_shmem)[((((i_outer * 512) + ((((int)threadIdx.x) / 4) * 128)) + ((((int)threadIdx.x) % 4) * 4)) + 8192)])));
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 49152, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, float* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
        main_kernel<<<dim3(4, 5, 2), dim3(128, 1, 1), 49152, stream>>>(A, B, C);
        TILELANG_CHECK_LAST_ERROR("main_kernel");

        return 0;
}

//extern "C" int call(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ C, int* __restrict__ batch_sizes, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, cudaStream_t stream=cudaStreamDefault) {
    th::Tensor gemm_splitK(const th::Tensor& A, const th::Tensor& B) {
      init();
      th::Tensor C = th::empty({A.size(0), B.size(1)}).to(th::kFloat).cuda();
    
      call(
          A.data<nv_bfloat16_t>(),
          B.data<nv_bfloat16_t>(),
          C.data<float>()
          );
      return C;
    }
    
    PYBIND11_MODULE(gemm_splitK, m) {
      m.def(
        "gemm_splitK",
        &gemm_splitK,
        "gemm splitK"
      );
    }