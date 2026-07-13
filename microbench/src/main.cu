#include "my_cutlass_gemm.h"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>

#define CUDA_CHECK(status)                                     \
do {                                                           \
    cudaError_t err = status;                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << std::endl;                                \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while (0)

void run_test(int m, int n, int k) {
    std::cout << "\n--- Running test for M=" << m << ", N=" << n << ", K=" << k << " ---\n";
    
    int lda = m;
    int ldb = k;
    int ldc = m;

    size_t size_a = (size_t)lda * k;
    size_t size_b = (size_t)ldb * n;
    size_t size_c = (size_t)ldc * n;

    std::vector<cutlass::half_t> h_A(size_a, cutlass::half_t(1.0f));
    std::vector<cutlass::half_t> h_B(size_b, cutlass::half_t(1.0f));

    cutlass::half_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(cutlass::half_t) * size_a));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(cutlass::half_t) * size_b));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(cutlass::half_t) * size_c));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(cutlass::half_t) * size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(cutlass::half_t) * size_b, cudaMemcpyHostToDevice));

    try {
        myCutlassHgemm(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, 1.0f, 0.0f, 0);
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel execution successful." << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    // 测试1: 精确匹配 (假设 1, 1, 64 存在于你的 result.csv 中)
    run_test(4096, 4096, 4096);

    // 测试2: 非精确匹配，将会触发最近邻查找
    // 假设 (1, 1, 65) 不在你的csv中, 它可能会匹配到 (1, 1, 64)
    // run_test(1, 1, 65);
    
    // 测试3: 另一个非精确匹配
    // run_test(2, 2, 130);

    return 0;
}