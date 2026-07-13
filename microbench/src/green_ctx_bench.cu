#include <cuda.h>
#include "my_cutlass_gemm.h"
#include <cuda_runtime.h>
#include <cutlass/half.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

class Barrier {
public:
    explicit Barrier(std::size_t count) : threshold_(count), count_(count), generation_(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto gen = generation_;
        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t threshold_;
    std::size_t count_;
    std::size_t generation_;
};

#define CUDA_CHECK(status)                                                             \
    do {                                                                               \
        cudaError_t err__ = (status);                                                  \
        if (err__ != cudaSuccess) {                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                  \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

#define CU_CHECK(status)                                                               \
    do {                                                                               \
        CUresult err__ = (status);                                                     \
        if (err__ != CUDA_SUCCESS) {                                                   \
            const char* errName;                                                       \
            const char* errStr;                                                        \
            cuGetErrorName(err__, &errName);                                           \
            cuGetErrorString(err__, &errStr);                                          \
            std::cerr << "CUDA Driver error: " << errName << " (" << errStr << ")"    \
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

void run_workload(int ctx_id, CUcontext ctx, int m, int n, int k, int iters, Barrier& barrier) {
    // Set the context for this thread
    CU_CHECK(cuCtxSetCurrent(ctx));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float alpha = 1.0f;
    float beta = 0.0f;

    size_t elems_a = static_cast<size_t>(m) * k;
    size_t elems_b = static_cast<size_t>(k) * n;
    size_t elems_c = static_cast<size_t>(m) * n;

    cutlass::half_t *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, elems_a * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, elems_b * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, elems_c * sizeof(cutlass::half_t)));

    // Initialize
    fill_device(d_A, elems_a, cutlass::half_t(1.0f), stream);
    fill_device(d_B, elems_b, cutlass::half_t(1.0f), stream);
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, elems_c * sizeof(cutlass::half_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warmup
    for (int i = 0; i < 1; ++i) {
        myCutlassHgemm(m, n, k, d_A, m, d_B, k, d_C, m, alpha, beta, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    std::cout << "Context " << ctx_id << " (M=" << m << ", N=" << n << ", K=" << k << ") starting..." << std::endl;
    
    barrier.wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        myCutlassHgemm(m, n, k, d_A, m, d_B, k, d_C, m, alpha, beta, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    std::cout << "Context " << ctx_id << " finished. Avg time: " << ms / iters << " ms/iter" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    int sms = 0;
    if (argc > 1) {
        sms = std::stoi(argv[1]);
    }

    CU_CHECK(cuInit(0));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));

    int total_sms = 108;

    // Get Device Resource (SMs)
    CUdevResource resource;
    CU_CHECK(cuDeviceGetDevResource(device, &resource, CU_DEV_RESOURCE_TYPE_SM));


    // Split resources
    std::vector<CUdevResource> split_resources(total_sms); // Allocate enough space
    unsigned int nbGroups = total_sms / 2;
    CUdevResource remaining;
    printf("nbGroups: %u\n", nbGroups);
    
    // Split into smallest possible chunks
    CU_CHECK(cuDevSmResourceSplitByCount(
        split_resources.data(),
        &nbGroups,
        &resource,
        &remaining,
        0, // flags
        2
    ));

    std::cout << "Split into " << nbGroups << " groups." << std::endl;

    int smPerGroup = total_sms / nbGroups;
    int chunks1 = sms / smPerGroup;
    int chunks2 = nbGroups - chunks1;

    if (chunks1 < 1) { chunks1 = 1; chunks2 = nbGroups - 1; }
    if (chunks2 < 1) { chunks2 = 1; chunks1 = nbGroups - 1; }

    std::cout << "Context 1 chunks: " << chunks1 << std::endl;
    std::cout << "Context 2 chunks: " << chunks2 << std::endl;

    CUcontext ctx1, ctx2;
    CUgreenCtx gctx1, gctx2;
    CUdevResourceDesc desc1, desc2;

    // Create Context 1
    {
        CU_CHECK(cuDevResourceGenerateDesc(&desc1, split_resources.data(), chunks1));
        CU_CHECK(cuGreenCtxCreate(&gctx1, desc1, device, CU_GREEN_CTX_DEFAULT_STREAM));
        CU_CHECK(cuCtxFromGreenCtx(&ctx1, gctx1));
    }

    // Create Context 2
    {
        CU_CHECK(cuDevResourceGenerateDesc(&desc2, split_resources.data() + chunks1, chunks2));
        CU_CHECK(cuGreenCtxCreate(&gctx2, desc2, device, CU_GREEN_CTX_DEFAULT_STREAM));
        CU_CHECK(cuCtxFromGreenCtx(&ctx2, gctx2));
    }


    // Task 1: 128 x 65536 x 65536 (Compute bound-ish? No, 128 is small M)
    // Task 2: 8192 x 65536 x 65536
    
    Barrier barrier(1);

    // We run them in parallel threads
    // std::thread t1(run_workload, 1, ctx1, 8, 65536, 65536, 1, std::ref(barrier));
    std::thread t2(run_workload, 2, ctx2, 8192, 8192, 8192, 1, std::ref(barrier)); // Fewer iters for larger task

    // t1.join();
    t2.join();

    // Cleanup contexts
    // Note: threads are joined, so contexts are not current in those threads anymore.
    // We can destroy them.
    // CU_CHECK(cuCtxDestroy(ctx1));
    // CU_CHECK(cuCtxDestroy(ctx2));

    return 0;
}
