#include "my_cutlass_gemm.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutlass/half.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <sstream>

#define CUDA_CHECK(status)                                                             \
    do {                                                                               \
        cudaError_t err__ = (status);                                                  \
        if (err__ != cudaSuccess) {                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                  \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

#define CUBLAS_CHECK(status)                                                           \
    do {                                                                               \
        cublasStatus_t stat__ = (status);                                              \
        if (stat__ != CUBLAS_STATUS_SUCCESS) {                                         \
            std::cerr << "cuBLAS error: " << cublas_status_to_string(stat__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

namespace {

const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 10010
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

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

std::vector<int> parse_int_list_env(const char* name, std::vector<int> fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }

    std::vector<int> values;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        values.push_back(std::stoi(item));
    }
    return values.empty() ? fallback : values;
}

int parse_int_env(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    return std::stoi(raw);
}

float benchmark_cublas(cublasHandle_t handle,
                       int m, int n, int k,
                       const cutlass::half_t* d_A,
                       const cutlass::half_t* d_B,
                       cutlass::half_t* d_C,
                       int lda, int ldb, int ldc,
                       float alpha, float beta,
                       size_t elems_c,
                       cudaStream_t stream,
                       int warmup_iters,
                       int timed_iters) {
    __half alpha_half = __float2half(alpha);
    __half beta_half = __float2half(beta);

    CUDA_CHECK(cudaMemsetAsync(d_C, 0, elems_c * sizeof(cutlass::half_t), stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    for (int i = 0; i < warmup_iters; ++i) {
        CUBLAS_CHECK(cublasHgemm(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alpha_half,
                                 reinterpret_cast<const __half*>(d_A),
                                 lda,
                                 reinterpret_cast<const __half*>(d_B),
                                 ldb,
                                 &beta_half,
                                 reinterpret_cast<__half*>(d_C),
                                 ldc));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemsetAsync(d_C, 0, elems_c * sizeof(cutlass::half_t), stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; ++i) {
        CUBLAS_CHECK(cublasHgemm(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alpha_half,
                                 reinterpret_cast<const __half*>(d_A),
                                 lda,
                                 reinterpret_cast<const __half*>(d_B),
                                 ldb,
                                 &beta_half,
                                 reinterpret_cast<__half*>(d_C),
                                 ldc));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(timed_iters);
}

float benchmark_cutlass(int m, int n, int k,
                        const cutlass::half_t* d_A,
                        const cutlass::half_t* d_B,
                        cutlass::half_t* d_C,
                        int lda, int ldb, int ldc,
                        float alpha, float beta,
                        size_t elems_c,
                        cudaStream_t stream,
                        int warmup_iters,
                        int timed_iters) {
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, elems_c * sizeof(cutlass::half_t), stream));

    for (int i = 0; i < warmup_iters; ++i) {
        myCutlassHgemm(m, n, k,
                       d_A, lda,
                       d_B, ldb,
                       d_C, ldc,
                       alpha, beta,
                       stream);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemsetAsync(d_C, 0, elems_c * sizeof(cutlass::half_t), stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; ++i) {
        myCutlassHgemm(m, n, k,
                       d_A, lda,
                       d_B, ldb,
                       d_C, ldc,
                       alpha, beta,
                       stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(timed_iters);
}

} // namespace

int main() {
    // std::vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384};
    // std::reverse(sizes.begin(), sizes.end());
    std::vector<int> n_sizes = parse_int_list_env("SMSCHED_BENCH_N_SIZES", {65536});
    std::vector<int> m_sizes = parse_int_list_env("SMSCHED_BENCH_M_SIZES", {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384});
    std::vector<int> k_sizes = parse_int_list_env("SMSCHED_BENCH_K_SIZES", {65536});
    const int warmup_iters = parse_int_env("SMSCHED_BENCH_WARMUP_ITERS", 2);
    const int timed_iters = parse_int_env("SMSCHED_BENCH_TIMED_ITERS", 5);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUDA_CHECK(cudaSetDevice(0));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "M,N,K | cuBLAS(ms) | CUTLASS(ms) | |Δ|(ms) | ratio(CUTLASS/cuBLAS)" << std::endl;

    for (int m : m_sizes) {
        for (int k : k_sizes) {
            for (int n : n_sizes) {
                int lda = m;
                int ldb = k;
                int ldc = m;

                size_t elems_a = static_cast<size_t>(lda) * k;
                size_t elems_b = static_cast<size_t>(ldb) * n;
                size_t elems_c = static_cast<size_t>(ldc) * n;
                size_t total_bytes = (elems_a + elems_b + elems_c) * sizeof(cutlass::half_t);

                size_t free_bytes = 0;
                size_t total_bytes_device = 0;
                CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes_device));
                if (total_bytes > free_bytes) {
                    std::cout << "M=" << m << ", N=" << n << ", K=" << k
                              << " | skipped (needs " << (total_bytes / (1024.0 * 1024.0))
                              << " MiB, free " << (free_bytes / (1024.0 * 1024.0)) << " MiB)" << std::endl;
                    continue;
                }

                cutlass::half_t* d_A = nullptr;
                cutlass::half_t* d_B = nullptr;
                cutlass::half_t* d_C = nullptr;
                CUDA_CHECK(cudaMalloc(&d_A, elems_a * sizeof(cutlass::half_t)));
                CUDA_CHECK(cudaMalloc(&d_B, elems_b * sizeof(cutlass::half_t)));
                CUDA_CHECK(cudaMalloc(&d_C, elems_c * sizeof(cutlass::half_t)));

                fill_device(d_A, elems_a, cutlass::half_t(1.0f), stream);
                fill_device(d_B, elems_b, cutlass::half_t(1.0f), stream);
                CUDA_CHECK(cudaMemsetAsync(d_C, 0, elems_c * sizeof(cutlass::half_t), stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                float time_cublas = benchmark_cublas(handle,
                                                     m, n, k,
                                                     d_A, d_B, d_C,
                                                     lda, ldb, ldc,
                                                     alpha, beta,
                                                     elems_c,
                                                     stream,
                                                     warmup_iters,
                                                     timed_iters);

                float time_cutlass = benchmark_cutlass(m, n, k,
                                                       d_A, d_B, d_C,
                                                       lda, ldb, ldc,
                                                       alpha, beta,
                                                       elems_c,
                                                       stream,
                                                       warmup_iters,
                                                       timed_iters);

                CUDA_CHECK(cudaStreamSynchronize(stream));

                float delta = std::fabs(time_cutlass - time_cublas);
                float ratio = (time_cublas > 0.0f) ? (time_cutlass / time_cublas) : std::numeric_limits<float>::infinity();

                std::cout << "M=" << std::setw(5) << m
                          << ", N=" << std::setw(5) << n
                          << ", K=" << std::setw(5) << k
                          << " | " << std::setw(9) << time_cublas
                          << " | " << std::setw(11) << time_cutlass
                          << " | " << std::setw(7) << delta
                          << " | " << std::setw(9) << ratio << std::endl;
                fflush(stdout);

                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_B));
                CUDA_CHECK(cudaFree(d_C));
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
