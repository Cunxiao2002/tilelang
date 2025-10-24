#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void qserve_kernel_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, bfloat16_t* __restrict__ C, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, int* __restrict__ batch_sizes, __grid_constant__ const CUtensorMap expert_weights_qscales_desc, signed char* __restrict__ per_group_scales, signed char* __restrict__ per_group_zero_points, bfloat16_t* __restrict__ per_token_scales);
extern "C" __global__ void __launch_bounds__(256, 1) qserve_kernel_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, bfloat16_t* __restrict__ C, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, int* __restrict__ batch_sizes, __grid_constant__ const CUtensorMap expert_weights_qscales_desc, signed char* __restrict__ per_group_scales, signed char* __restrict__ per_group_zero_points, bfloat16_t* __restrict__ per_token_scales) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  int cur_batch_idx[1];
  int cur_batch_size[1];
  int Ct_local[32];
  signed char per_group_scales_local[8];
  signed char per_group_zero_points_local[8];
  uchar B_local[32];
  signed char B_dequant_local[64];
  signed char B_dequant_prev_local[64];
  __shared__ uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(expert_weights_qscales_desc);
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
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
    cur_batch_size[0] = batch_sizes[cur_batch_idx[0]];
    tl::fence_proxy_async();
    mbarrier[4].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[5].expect_transaction(128);
      tl::tma_load(expert_weights_qscales_desc, mbarrier[5], (&(((bfloat16_t*)buf_dyn_shmem)[12800])), (((int)blockIdx.y) * 64), cur_batch_idx[0]);
    }
    tl::mbarrier_cp_async_arrive(mbarrier[5]);
    mbarrier[5].arrive();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k & 1) + 2)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].expect_transaction(8192);
        tl::tma_load(A_desc, mbarrier[(k & 1)], (&(((signed char*)buf_dyn_shmem)[((k & 1) * 8192)])), (k * 128), (((((int)blockIdx.x) * 64) + batch_offsets[cur_batch_idx[0]]) - batch_padded_offsets[cur_batch_idx[0]]));
      }
      tl::mbarrier_cp_async_arrive(mbarrier[(k & 1)]);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].expect_transaction(4096);
        tl::tma_load(B_desc, mbarrier[(k & 1)], (&(buf_dyn_shmem[(((k & 1) * 4096) + 16384)])), (k * 64), (((int)blockIdx.y) * 64), cur_batch_idx[0]);
      }
      tl::mbarrier_cp_async_arrive(mbarrier[(k & 1)]);
      mbarrier[(k & 1)].arrive();
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
    tl::fence_proxy_async();
    mbarrier[4].arrive();
    cur_batch_size[0] = batch_sizes[cur_batch_idx[0]];
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      *(int2*)(Ct_local + (i_2 * 2)) = make_int2(0, 0);
    }
    if (((int)threadIdx.x) < 64) {
      if (((((((int)blockIdx.x) * 64) + batch_offsets[cur_batch_idx[0]]) + ((int)threadIdx.x)) - batch_padded_offsets[cur_batch_idx[0]]) < 576) {
        bfloat16_t condval_2;
        if ((0 <= ((((((int)blockIdx.x) * 64) + batch_offsets[cur_batch_idx[0]]) + ((int)threadIdx.x)) - batch_padded_offsets[cur_batch_idx[0]]))) {
          condval_2 = per_token_scales[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) + batch_offsets[cur_batch_idx[0]]) - batch_padded_offsets[cur_batch_idx[0]])];
        } else {
          condval_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        }
        ((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 12288)] = condval_2;
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(int*)(per_group_scales_local + (i_3 * 4)) = *(int*)(per_group_scales + ((((((cur_batch_idx[0] * 327680) + (((int)blockIdx.y) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_3 * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((int)blockIdx.x) * 4)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      *(int*)(per_group_zero_points_local + (i_4 * 4)) = *(int*)(per_group_zero_points + ((((((cur_batch_idx[0] * 327680) + (((int)blockIdx.y) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((int)blockIdx.x) * 4)));
    }
    tl::__sync_thread_partial<3, 128>();
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        *(uchar2*)(B_local + (i_5 * 2)) = *(uchar2*)(buf_dyn_shmem + ((((((((k_1 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_5 >> 3) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_5 & 7) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 64; ++i_6) {
          uchar v_ = ((B_local[(((((i_6 & 7) >> 2) * 16) + ((i_6 >> 3) * 2)) + ((i_6 & 3) >> 1))] >> (((uchar)(i_6 & 1)) * (uchar)4)) & (uchar)15) << (uchar)4;
        B_dequant_local[i_6] = ((*(signed char *)(&(v_))) >> (signed char)4);
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 64; ++i_7) {
        B_dequant_prev_local[i_7] = ((B_dequant_local[i_7] - per_group_zero_points_local[((((i_7 & 7) >> 2) * 4) + (i_7 >> 4))]) * per_group_scales_local[((((i_7 & 7) >> 2) * 4) + (i_7 >> 4))]);
      }
      tl::fence_proxy_async();
      tl::gemm_rs<64, 64, 128, 4, 1, 0, 1, 0, 128, 128, 0, 0, true>((&(B_dequant_prev_local[0])), (&(((signed char*)buf_dyn_shmem)[((k_1 & 1) * 8192)])), (&(Ct_local[0])));
      mbarrier[((k_1 & 1) + 2)].arrive();
    }
    mbarrier[5].wait(0);
    #pragma unroll
    for (int i_8 = 0; i_8 < 32; ++i_8) {
      if (((((i_8 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + ((i_8 & 3) >> 1)) < ((cur_batch_size[0] + batch_padded_offsets[cur_batch_idx[0]]) - (((int)blockIdx.x) * 64))) {
        if (0 <= ((((((((int)blockIdx.x) * 64) + ((i_8 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_8 & 3) >> 1)) + batch_offsets[cur_batch_idx[0]]) - batch_padded_offsets[cur_batch_idx[0]])) {
          if (((((((((int)blockIdx.x) * 64) + ((i_8 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_8 & 3) >> 1)) + batch_offsets[cur_batch_idx[0]]) - batch_padded_offsets[cur_batch_idx[0]]) < 576) {
            C[((((((((((((int)blockIdx.y) * 36864) + ((((int)threadIdx.x) >> 5) * 9216)) + ((i_8 & 1) * 4608)) + (((((int)threadIdx.x) & 31) >> 2) * 576)) + (((int)blockIdx.x) * 64)) + ((i_8 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_8 & 3) >> 1)) + batch_offsets[cur_batch_idx[0]]) - batch_padded_offsets[cur_batch_idx[0]])] = ((((bfloat16_t)Ct_local[((((i_8 >> 2) * 4) + ((i_8 & 1) * 2)) + ((i_8 & 3) >> 1))]) * ((bfloat16_t*)buf_dyn_shmem)[(((((i_8 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + ((i_8 & 3) >> 1)) + 12288)]) * ((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_8 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 12800)]);
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
    
    cudaError_t result_qserve_kernel_kernel = cudaFuncSetAttribute(qserve_kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 25728);
    if (result_qserve_kernel_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 25728, cudaGetErrorString(result_qserve_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(int8_t* __restrict__ A, uint8_t* __restrict__ B, bfloat16_t* __restrict__ C, int* __restrict__ batch_sizes, int* __restrict__ batch_offsets, int* __restrict__ batch_padded_offsets, bfloat16_t* __restrict__ per_token_scales, bfloat16_t* __restrict__ expert_weights_qscales, int8_t* __restrict__ per_group_scales, int8_t* __restrict__ per_group_zero_points, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap A_desc;
	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)0;
	cuuint32_t A_desc_tensorRank= 2;
	void *A_desc_globalAddress= A;
	cuuint64_t A_desc_globalDim[2]= {4096,576};
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
	cuuint64_t B_desc_globalDim[3]= {2048,2560,16};
	cuuint64_t B_desc_globalStride[3]= {1,2048,5242880};
	cuuint32_t B_desc_boxDim[3]= {64,64,1};
	cuuint32_t B_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)0;
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

	CUtensorMap expert_weights_qscales_desc;
	CUtensorMapDataType expert_weights_qscales_desc_type= (CUtensorMapDataType)9;
	cuuint32_t expert_weights_qscales_desc_tensorRank= 2;
	void *expert_weights_qscales_desc_globalAddress= expert_weights_qscales;
	cuuint64_t expert_weights_qscales_desc_globalDim[2]= {2560,16};
	cuuint64_t expert_weights_qscales_desc_globalStride[2]= {2,5120};
	cuuint32_t expert_weights_qscales_desc_boxDim[2]= {64,1};
	cuuint32_t expert_weights_qscales_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave expert_weights_qscales_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle expert_weights_qscales_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion expert_weights_qscales_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill expert_weights_qscales_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult expert_weights_qscales_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &expert_weights_qscales_desc, expert_weights_qscales_desc_type, expert_weights_qscales_desc_tensorRank, expert_weights_qscales_desc_globalAddress, expert_weights_qscales_desc_globalDim, expert_weights_qscales_desc_globalStride + 1, expert_weights_qscales_desc_boxDim, expert_weights_qscales_desc_elementStrides, expert_weights_qscales_desc_interleave, expert_weights_qscales_desc_swizzle, expert_weights_qscales_desc_l2Promotion, expert_weights_qscales_desc_oobFill);

	if (expert_weights_qscales_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor expert_weights_qscales_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	qserve_kernel_kernel<<<dim3(16, 40, 1), dim3(256, 1, 1), 25728, stream>>>(A_desc, B_desc, C, batch_offsets, batch_padded_offsets, batch_sizes, expert_weights_qscales_desc, per_group_scales, per_group_zero_points, per_token_scales);
	TILELANG_CHECK_LAST_ERROR("qserve_kernel_kernel");

	return 0;
}

