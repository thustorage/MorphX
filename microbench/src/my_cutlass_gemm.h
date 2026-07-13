#pragma once

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

// 为了确保在C++项目中能够正确链接C风格的符号，使用 extern "C"
#ifdef __cplusplus
extern "C" {
#endif

// 前向声明依赖的类型，避免包含完整的CUDA或CUTLASS头文件
typedef struct CUstream_st* cudaStream_t;
namespace cutlass { struct half_t; }

/**
 * @brief 执行半精度矩阵乘法 D = alpha * A * B + beta * C.
 *
 * 该函数内部通过查找表自动选择最优的CUTLASS内核。
 * 实现被编译在独立的共享库中，以加速主项目的编译。
 */
void myCutlassHgemm(
    int m,
    int n,
    int k,
    const cutlass::half_t* a,
    long long int lda, // 使用标准的 long long int 避免依赖 int64_t 定义
    const cutlass::half_t* b,
    long long int ldb,
    cutlass::half_t* c,
    long long int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
);

#ifdef __cplusplus
} // extern "C"
#endif
