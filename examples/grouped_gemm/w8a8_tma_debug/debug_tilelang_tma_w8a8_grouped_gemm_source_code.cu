#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
namespace th = ::torch;

extern "C" __global__ void kernel_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, signed char* __restrict__ C, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, int* __restrict__ batch_sizes);
extern "C" __global__ void __launch_bounds__(384, 1) kernel_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, signed char* __restrict__ C, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, int* __restrict__ batch_sizes) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  int cur_batch_idx[1];
  int cur_batch_size[1];
  int C_local[32];
  __shared__ uint64_t _mbarrier[6];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 256);
    tl::mbarrier_init(_mbarrier[3], 256);
    tl::mbarrier_init(_mbarrier[4], 256);
    tl::mbarrier_init(_mbarrier[5], 256);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int i = 0; i < 16; ++i) {
      int condval;
      if ((batch_padded_offsets[i] <= (((int)blockIdx.x) * 64))) {
        condval = i;
      } else {
        condval = cur_batch_idx[0];
      }
      cur_batch_idx[0] = condval;
    }
    tl::fence_proxy_async();
    for (int k = 0; k < 16; ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 2)], (((k & 3) >> 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 16384);
        tl::tma_load(A_desc, _mbarrier[(k & 1)], (&(((signed char*)buf_dyn_shmem)[((k & 1) * 16384)])), (k * 256), (((((int)blockIdx.x) * 64) + batch_offsets[cur_batch_idx[(int64_t)0]]) - batch_padded_offsets[cur_batch_idx[(int64_t)0]]));
        tl::tma_load(A_desc, _mbarrier[(k & 1)], (&(((signed char*)buf_dyn_shmem)[(((k & 1) * 16384) + 8192)])), ((k * 256) + 128), (((((int)blockIdx.x) * 64) + batch_offsets[cur_batch_idx[(int64_t)0]]) - batch_padded_offsets[cur_batch_idx[(int64_t)0]]));
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[(k & 1)]);
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 32768);
        tl::tma_load(B_desc, _mbarrier[(k & 1)], (&(((signed char*)buf_dyn_shmem)[(((k & 1) * 32768) + 32768)])), (k * 256), (((int)blockIdx.y) * 128), cur_batch_idx[0]);
        tl::tma_load(B_desc, _mbarrier[(k & 1)], (&(((signed char*)buf_dyn_shmem)[(((k & 1) * 32768) + 49152)])), ((k * 256) + 128), (((int)blockIdx.y) * 128), cur_batch_idx[0]);
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[(k & 1)]);
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      int condval_1;
      if ((batch_padded_offsets[i_1] <= (((int)blockIdx.x) * 64))) {
        condval_1 = i_1;
      } else {
        condval_1 = cur_batch_idx[0];
      }
      cur_batch_idx[0] = condval_1;
    }
    cur_batch_size[0] = batch_sizes[cur_batch_idx[(int64_t)0]];
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      *(int2*)(C_local + (i_2 * 2)) = make_int2(0, 0);
    }
    tl::fence_proxy_async();
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      tl::mbarrier_wait(_mbarrier[(k_1 & 1)], ((k_1 & 3) >> 1));
      tl::gemm_ss<64, 128, 256, 4, 2, 0, 1, 0, true>((&(((signed char*)buf_dyn_shmem)[((k_1 & 1) * 16384)])), (&(((signed char*)buf_dyn_shmem)[(((k_1 & 1) * 32768) + 32768)])), (&(C_local[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 2)]);
    }
    // tl::syncthreads_partial(_mbarrier[4]);
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      // char2 __1;
      // int2 v_ = *(int2*)(C_local + (i_3 * 2));
      // __1.x=((signed char)(v_.x));
      // __1.y=((signed char)(v_.y));
      // *(char2*)(((signed char*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 5) * 2048) + ((i_3 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
      // tl::ptx_stmatrix_x2_trans((void*)(((signed char*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 5) * 2048) + ((i_3 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))), C_local[i_3 * 2], C_local[i_3 * 2 + 1]);
      tl::ptx_stmatrix_x2(buf_dyn_shmem, C_local[i_3 * 2], C_local[i_3 * 2 + 1]);
    }
    // tl::syncthreads_partial(_mbarrier[5]);
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      if (((i_4 * 8) + (((int)threadIdx.x) >> 5)) < ((cur_batch_size[0] + batch_padded_offsets[cur_batch_idx[(int64_t)0]]) - (((int)blockIdx.x) * 64))) {
        if (0 <= (((((((int)blockIdx.x) * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 5)) + batch_offsets[cur_batch_idx[(int64_t)0]]) - batch_padded_offsets[cur_batch_idx[(int64_t)0]])) {
          if ((((((((int)blockIdx.x) * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 5)) + batch_offsets[cur_batch_idx[(int64_t)0]]) - batch_padded_offsets[cur_batch_idx[(int64_t)0]]) < 3072) {
            *(int*)(C + (((((((((int64_t)((int)blockIdx.x)) * (int64_t)163840) + (((int64_t)i_4) * (int64_t)20480)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2560)) + (((int64_t)batch_offsets[cur_batch_idx[(int64_t)0]]) * (int64_t)2560)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)) - (((int64_t)batch_padded_offsets[cur_batch_idx[(int64_t)0]]) * (int64_t)2560))) = *(int*)(((signed char*)buf_dyn_shmem) + ((i_4 * 1024) + (((int)threadIdx.x) * 4)));
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
    
    cudaError_t result_kernel_kernel = cudaFuncSetAttribute(kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    if (result_kernel_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 98304, cudaGetErrorString(result_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ C, int* __restrict__ batch_sizes, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap A_desc;
	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)0;
	cuuint32_t A_desc_tensorRank= 2;
	void *A_desc_globalAddress= A;
	cuuint64_t A_desc_globalDim[2]= {4096,3072};
	cuuint64_t A_desc_globalStride[2]= {1,4096};
	cuuint32_t A_desc_boxDim[2]= {128,64};
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
	CUtensorMapDataType B_desc_type= (CUtensorMapDataType)0;
	cuuint32_t B_desc_tensorRank= 3;
	void *B_desc_globalAddress= B;
	cuuint64_t B_desc_globalDim[3]= {4096,2560,16};
	cuuint64_t B_desc_globalStride[3]= {1,4096,10485760};
	cuuint32_t B_desc_boxDim[3]= {128,128,1};
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
	kernel_kernel<<<dim3(64, 20, 1), dim3(384, 1, 1), 98304, stream>>>(A_desc, B_desc, C, batch_offsets, batch_padded_offsets, batch_sizes);
	TILELANG_CHECK_LAST_ERROR("kernel_kernel");

	return 0;
}

//extern "C" int call(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ C, int* __restrict__ batch_sizes, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, cudaStream_t stream=cudaStreamDefault) {
  th::Tensor qtc_group_gemm(const th::Tensor& A, const th::Tensor& B, const th::Tensor& batch_sizes, const th::Tensor& batch_offsets,
    const th::Tensor& batch_padded_offsets) {
  init();
  th::Tensor C = th::empty({A.size(0), B.size(1)}).to(th::kInt8).cuda();

  call(
      A.data<int8_t>(),
      B.data<int8_t>(),
      C.data<int8_t>(),
      batch_sizes.data<int>(),
      batch_offsets.data<int>(),
      batch_padded_offsets.data<int>()
      );
  return C;
}

PYBIND11_MODULE(qtc, m) {
  m.def(
    "qtc_group_gemm",
    &qtc_group_gemm,
    "qtc group gemm"
  );
}

