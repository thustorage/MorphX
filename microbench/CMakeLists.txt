cmake_minimum_required(VERSION 3.20)

option(SMSCHED_ENABLE_PASS "Build CUDA kernels with the SMSched LLVM pass" OFF)

if(SMSCHED_ENABLE_PASS)
    set(SMSCHED_LLVM_ROOT "/home/rtx/llvm-project/install" CACHE PATH "LLVM install prefix for SMSched pass builds")
    set(CMAKE_CUDA_COMPILER "${SMSCHED_LLVM_ROOT}/bin/clang++" CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER "${SMSCHED_LLVM_ROOT}/bin/clang++" CACHE FILEPATH "" FORCE)
    set(CMAKE_C_COMPILER "${SMSCHED_LLVM_ROOT}/bin/clang" CACHE FILEPATH "" FORCE)
endif()

project(microbench LANGUAGES C CXX CUDA)

if(SMSCHED_ENABLE_PASS)
    get_filename_component(SMSCHED_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(SMSCHED_PASS_SO "${SMSCHED_REPO_ROOT}/smsched-pass/build/libsmsched.so" CACHE FILEPATH "SMSched LLVM pass plugin")
    set(SMSCHED_CUDA_COMMON_FLAGS
        -nostdinc++
        -I/usr/local/cuda-12.4/include
        -I/usr/local/cuda-12.4/targets/x86_64-linux/include
        -I${SMSCHED_LLVM_ROOT}/include/c++/v1
        -I${SMSCHED_LLVM_ROOT}/include/x86_64-unknown-linux-gnu/c++/v1
        -I/home/rtx/gpu/cccl/thrust
        -I/home/rtx/gpu/cccl/libcudacxx/include
        -I/home/rtx/gpu/cccl/cub
        --cuda-gpu-arch=sm_80
        -std=c++17
        -O3
        -fplugin=${SMSCHED_PASS_SO}
        -fpass-plugin=${SMSCHED_PASS_SO}
        -mllvm
        -enable-dump-ir
        -mllvm
        -dump-ir-dir=${CMAKE_CURRENT_BINARY_DIR}/ir
        -D__NVCC__
        -D__CUDACC_VER_MAJOR__=12
        -D__CUDACC_VER_MINOR__=4
        -D__NV_NO_HOST_COMPILER_CHECK
        -D_ALLOW_UNSUPPORTED_LIBCPP
        -D_VSTD=std
        -Wno-invalid-specialization
        -D__CUDACC__
        -include
        cuda_fp16.h
    )
    string(REPLACE ";" " " SMSCHED_CUDA_FLAGS_STR "${SMSCHED_CUDA_COMMON_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SMSCHED_CUDA_FLAGS_STR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -I${SMSCHED_LLVM_ROOT}/include/c++/v1 -I${SMSCHED_LLVM_ROOT}/include/x86_64-unknown-linux-gnu/c++/v1 -D_VSTD=std -Wno-error=missing-template-arg-list-after-template-kw -Wno-invalid-specialization -include cuda_fp16.h")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/cuda-12.4/lib64 -stdlib=libc++ -lc++ -lc++abi -lcudart -lcuda -lgdrapi")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/cuda-12.4/lib64 -stdlib=libc++ -lc++ -lc++abi -lcudart -lcuda -lgdrapi")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA SM architectures to target" FORCE)

set(CUTLASS_ROOT "/home/rtx/gpu/cutlass" CACHE PATH "CUTLASS install prefix")
set(CUTLASS_INCLUDE_DIRS
    "${CUTLASS_ROOT}/include"
    "${CUTLASS_ROOT}/tools/util/include"
)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.4" CACHE PATH "CUDA Toolkit root directory")
find_package(CUDAToolkit REQUIRED cublas)

add_library(mycutlass SHARED src/my_cutlass_gemm.cu)
# Shared library provides the kernels consumed by the executables.
target_include_directories(mycutlass
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUTLASS_INCLUDE_DIRS}
)
target_link_libraries(mycutlass
    PUBLIC
        CUDA::cudart
)

add_executable(main src/main.cu)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main PRIVATE mycutlass)

add_executable(bench src/bench.cu)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench PRIVATE mycutlass CUDA::cublas)

add_executable(stream_bench src/stream_bench.cu)
target_include_directories(stream_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stream_bench PRIVATE mycutlass CUDA::cublas)

add_executable(stream_mm_test src/stream_mm_test.cu)
target_include_directories(stream_mm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stream_mm_test PRIVATE mycutlass)

add_executable(green_ctx_bench src/green_ctx_bench.cu)
target_include_directories(green_ctx_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(green_ctx_bench PRIVATE mycutlass CUDA::cuda_driver)

install(TARGETS mycutlass main bench stream_mm_test green_ctx_bench
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
