#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void kernel_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, int* __restrict__ batch_sizes);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, int* __restrict__ batch_sizes) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  int cur_batch_idx[1];
  int cur_batch_size[1];
  float C_local[32];
  __shared__ uint64_t _mbarrier[4];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 3; ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 2)], ((k >> 1) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 8192);
        tl::tma_load(A_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[((k & 1) * 4096)])), (k * 64), (((((int)blockIdx.x) * 64) + batch_offsets[cur_batch_idx[(int64_t)0]]) - batch_padded_offsets[cur_batch_idx[(int64_t)0]]));
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[(k & 1)]);
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 8192);
        tl::tma_load(B_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 8192)])), (((int)blockIdx.y) * 64), (k * 64), cur_batch_idx[0]);
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[(k & 1)]);
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    for (int i = 0; i < 128; ++i) {
      int condval;
      if ((batch_padded_offsets[i] <= (((int)blockIdx.x) * 64))) {
        condval = i;
      } else {
        condval = cur_batch_idx[0];
      }
      cur_batch_idx[0] = condval;
    }
    cur_batch_size[0] = batch_sizes[cur_batch_idx[(int64_t)0]];
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      *(float2*)(C_local + (i_1 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    tl::fence_proxy_async();
    for (int k_1 = 0; k_1 < 3; ++k_1) {
      tl::mbarrier_wait(_mbarrier[(k_1 & 1)], (k_1 >> 1));
      tl::gemm_ss<64, 64, 64, 4, 1, 0, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[((k_1 & 1) * 4096)])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 4096) + 8192)])), (&(C_local[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 2)]);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      if (((((((int)threadIdx.x) >> 5) * 16) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < ((cur_batch_size[0] + batch_padded_offsets[cur_batch_idx[(int64_t)0]]) - (((int)blockIdx.x) * 64))) {
        if (0 <= ((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + batch_offsets[cur_batch_idx[(int64_t)0]]) - batch_padded_offsets[cur_batch_idx[(int64_t)0]])) {
          if (((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + batch_offsets[cur_batch_idx[(int64_t)0]]) - batch_padded_offsets[cur_batch_idx[(int64_t)0]]) < 24576) {
            uint1 __1;
            float2 v_ = *(float2*)(C_local + (i_2 * 2));
            ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
            ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
            *(uint1*)(C + (((((((((((int64_t)((int)blockIdx.x)) * (int64_t)262144) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)65536)) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)32768)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)4096)) + (((int64_t)batch_offsets[cur_batch_idx[(int64_t)0]]) * (int64_t)4096)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (((int64_t)batch_padded_offsets[cur_batch_idx[(int64_t)0]]) * (int64_t)4096))) = __1;
          }
        }
      }
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
    
    cudaError_t result_kernel_kernel = cudaFuncSetAttribute(kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    if (result_kernel_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 32768, cudaGetErrorString(result_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, int* __restrict__ batch_sizes, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap A_desc;
	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)6;
	cuuint32_t A_desc_tensorRank= 2;
	void *A_desc_globalAddress= A;
	cuuint64_t A_desc_globalDim[2]= {160,24576};
	cuuint64_t A_desc_globalStride[2]= {2,320};
	cuuint32_t A_desc_boxDim[2]= {64,64};
	cuuint32_t A_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult A_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);

	if (A_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor A_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap B_desc;
	CUtensorMapDataType B_desc_type= (CUtensorMapDataType)6;
	cuuint32_t B_desc_tensorRank= 3;
	void *B_desc_globalAddress= B;
	cuuint64_t B_desc_globalDim[3]= {4096,160,128};
	cuuint64_t B_desc_globalStride[3]= {2,8192,1310720};
	cuuint32_t B_desc_boxDim[3]= {64,64,1};
	cuuint32_t B_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion B_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill B_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult B_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);

	if (B_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor B_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	kernel_kernel<<<dim3(512, 64, 1), dim3(256, 1, 1), 32768, stream>>>(A_desc, B_desc, C, batch_offsets, batch_padded_offsets, batch_sizes);
	TILELANG_CHECK_LAST_ERROR("kernel_kernel");

	return 0;
}

❌ Tilelang and Torch mismatch

First 10 mismatches:
Row	Col	Tilelang	Torch	Diff
0	0	-3.41		7.27	-10.68
0	1	-18.53		-28.61	10.08
0	2	-17.95		-15.07	-2.88
0	3	0.63		-12.30	12.93
0	4	2.09		-2.70	4.79
0	5	-8.27		3.45	-11.71
0	6	-1.52		-19.92	18.40
0	7	-11.20		-15.89	4.70
0	8	8.91		10.73	-1.82
0	9	-9.80		-4.53	-5.27

Statistics:
Total mismatches: 100595771
Max diff: 82.12
Min diff: -81.12
Avg diff: -0.00
