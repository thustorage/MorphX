#include "my_cutlass_gemm.h"
#include <cuda_runtime.h>
#include <cutlass/half.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(status)                                                             \
    do {                                                                               \
        cudaError_t err__ = (status);                                                  \
        if (err__ != cudaSuccess) {                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                  \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

namespace {

__global__ void fill_kernel(cutlass::half_t* data, size_t count, cutlass::half_t value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

void fill_device(cutlass::half_t* data, size_t count, cutlass::half_t value, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    constexpr int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    fill_kernel<<<blocks, threads, 0, stream>>>(data, count, value);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

int main() {
    // Dimensions
    // Task 1: 16 x 65536 x 65536
    int m1 = 128;
    int n1 = 65536;
    int k1 = 65536;

    // Task 2: 8192 x 65536 x 65536
    int m2 = 8192;
    int n2 = 65536;
    int k2 = 65536;

    // Common N and K for sharing B matrix if possible
    // Here n1=n2, k1=k2. So we can share B.
    int n_common = 65536;
    int k_common = 65536;

    int warmup_iters = 5;
    int timed_iters = 20;
    float alpha = 1.0f;
    float beta = 0.0f;

    CUDA_CHECK(cudaSetDevice(0));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Calculate memory sizes
    // Layout: Column Major (as in bench.cu)
    // A: m x k
    // B: k x n
    // C: m x n
    
    size_t elems_b = static_cast<size_t>(k_common) * n_common;
    
    size_t elems_a1 = static_cast<size_t>(m1) * k1;
    size_t elems_c1 = static_cast<size_t>(m1) * n1;

    size_t elems_a2 = static_cast<size_t>(m2) * k2;
    size_t elems_c2 = static_cast<size_t>(m2) * n2;

    size_t total_elems = elems_b + elems_a1 + elems_c1 + elems_a2 + elems_c2;
    size_t total_bytes = total_elems * sizeof(cutlass::half_t);

    size_t free_bytes = 0;
    size_t total_bytes_device = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes_device));

    std::cout << "Required Memory: " << (total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB" << std::endl;
    std::cout << "Free Memory: " << (free_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB" << std::endl;

    if (total_bytes > free_bytes) {
        std::cerr << "Not enough memory!" << std::endl;
        return 1;
    }

    cutlass::half_t *d_B = nullptr;
    cutlass::half_t *d_A1 = nullptr, *d_C1 = nullptr;
    cutlass::half_t *d_A2 = nullptr, *d_C2 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_B, elems_b * sizeof(cutlass::half_t)));
    
    CUDA_CHECK(cudaMalloc(&d_A1, elems_a1 * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_C1, elems_c1 * sizeof(cutlass::half_t)));

    CUDA_CHECK(cudaMalloc(&d_A2, elems_a2 * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_C2, elems_c2 * sizeof(cutlass::half_t)));

    // Initialize data
    // Use stream1 for initialization
    fill_device(d_B, elems_b, cutlass::half_t(1.0f), stream1);
    fill_device(d_A1, elems_a1, cutlass::half_t(1.0f), stream1);
    fill_device(d_A2, elems_a2, cutlass::half_t(1.0f), stream1);
    CUDA_CHECK(cudaMemsetAsync(d_C1, 0, elems_c1 * sizeof(cutlass::half_t), stream1));
    CUDA_CHECK(cudaMemsetAsync(d_C2, 0, elems_c2 * sizeof(cutlass::half_t), stream1));
    
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmup_iters; ++i) {
        myCutlassHgemm(m1, n1, k1, d_A1, m1, d_B, k1, d_C1, m1, alpha, beta, stream1);
        myCutlassHgemm(m2, n2, k2, d_A2, m2, d_B, k2, d_C2, m2, alpha, beta, stream2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::cout << "Benchmarking..." << std::endl;
    
    cudaEvent_t start1, stop1, start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    // Launch Stream 1
    for (int i = 0; i < 200; ++i) {
        myCutlassHgemm(m1, n1, k1, d_A1, m1, d_B, k1, d_C1, m1, alpha, beta, stream1);
    }

    // Launch Stream 2
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 2; ++i) {
        
        myCutlassHgemm(m2, n2, k2, d_A2, m2, d_B, k2, d_C2, m2, alpha, beta, stream2);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaDeviceSynchronize());

    // float ms1 = 0.0f;
    float ms2 = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // float avg_ms1 = ms1 / 200;
    float avg_ms2 = ms2 / 2;

    // std::cout << "Stream 1 (16x65536x65536): " << avg_ms1 << " ms/iter" << std::endl;
    std::cout << "Stream 2 (8192x65536x65536): " << avg_ms2 << " ms/iter" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_C1));
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_C2));
    
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
