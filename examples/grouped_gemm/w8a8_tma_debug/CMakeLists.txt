# ===================== trtllm =========================
cmake_minimum_required(VERSION 3.21)
project(TRTLLM_KERNELS CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_PATH ${CUDA_HOME})
#set(CUDNN_PATH ${CUDNN_HOME})
set(TORCH_PATH ${TORCH_HOME})
#set(CUTLASS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/cutlass)

#set(CMAKE_CUDA_ARCHITECTURES 90-real)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90a,code=sm_90a")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
if(Python3_FOUND)
  message(f"python3 found")
elseif()
  message(f"python3 not found")
endif()
if(Torch_FOUND)
  message(f"torch found")
elseif()
  message(f"torch not found")
endif()

# find_library(TORCH_LIBS torch_python PATHS ${TORCH_PATH})
find_library(TORCH_LIBS torch PATHS ${TORCH_PATH}/lib)
find_library(CUDA_LIBS cudart PATHS ${CUDA_PATH}/lib64)
find_library(CUDNN_LIBS cudnn PATHS ${CUDNN_PATH})
find_library(TORCH_PYTHON_LIBS torch_python PATHS ${TORCH_PATH}/lib)
message("Torch_path--- ${TORCH_PATH}")

include_directories(${CUDA_PATH}/include)
include_directories(${CUDNN_PATH}/include)
include_directories(${TORCH_PATH}/include)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(/home/workspace/tilelang/src/)
include_directories(/home/workspace/tilelang/3rdparty/cutlass/include/)
# include_directories(/home/workspace/qtc/test_qtc/lib/python3.9/site-packages/torch/include)

link_directories(${CUDA_PATH}/lib64)
link_directories(${CUDNN_PATH}/lib64)

add_definitions("-DENABLE_BF16")
add_definitions("-DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(TRTLLM_HEADERS
  ${TRTLLM_KERNEL_PATH}/../..
  ${TRTLLM_KERNEL_PATH}/cutlass_kernels/
  ${TRTLLM_KERNEL_PATH}/../cutlass_extensions/include
  )
include_directories(${TRTLLM_HEADERS})

add_definitions("-DUSE_W2A16")

add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")

set(QTC_SRC_CU
    # ${CMAKE_CURRENT_SOURCE_DIR}/tilelang_tma_w8a8_grouped_gemm_source_code.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/profile_tilelang_grouped_gemm_tma_ep.cu
    )

add_library(qtc_w8a8 SHARED  ${QTC_SRC_CU})
target_link_libraries(qtc_w8a8 ${TORCH_LIBS} ${CUDA_LIBS} ${CUDNN_LIBS} ${TORCH_PYTHON_LIBS})
set_property(TARGET qtc_w8a8 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET qtc_w8a8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)