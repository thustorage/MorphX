#include "my_cutlass_gemm.h"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>

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

int main(int argc, char** argv) {
    // 可配置运行时长（秒），默认 60 秒
    int duration_seconds = 60;
    int num_streams = 2; // 新增：默认 stream 数量

    if (argc > 1) {
        duration_seconds = std::stoi(argv[1]);
    }
    if (argc > 2) {
        num_streams = std::stoi(argv[2]);
    }
    if (num_streams <= 0) num_streams = 1;
    if (num_streams > 128) num_streams = 128; // 简单上限防止资源暴涨

    std::vector<int> choices = {8, 64, 1024, 4096, 16384};

    // 动态分配计数数组，大小为 num_streams
    std::vector<std::atomic<long long>> counts(num_streams);
    for (int i = 0; i < num_streams; ++i) counts[i].store(0);

    std::atomic<bool> stop_flag(false);

    auto worker = [&](int idx) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::random_device rd;
        std::mt19937 gen(rd() + idx);
        std::uniform_int_distribution<int> dist(0, (int)choices.size() - 1);

        while (!stop_flag.load(std::memory_order_relaxed)) {
            int m = choices[dist(gen)];
            int n = choices[dist(gen)];
            int k = choices[dist(gen)];

            long long lda = m;
            long long ldb = k;
            long long ldc = m;

            size_t size_a = (size_t)lda * k;
            size_t size_b = (size_t)ldb * n;
            size_t size_c = (size_t)ldc * n;

            std::vector<cutlass::half_t> h_A(size_a, cutlass::half_t(1.0f));
            std::vector<cutlass::half_t> h_B(size_b, cutlass::half_t(1.0f));

            cutlass::half_t *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
            if (size_a) CUDA_CHECK(cudaMalloc(&d_A, sizeof(cutlass::half_t) * size_a));
            if (size_b) CUDA_CHECK(cudaMalloc(&d_B, sizeof(cutlass::half_t) * size_b));
            if (size_c) CUDA_CHECK(cudaMalloc(&d_C, sizeof(cutlass::half_t) * size_c));

            if (d_A) CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), sizeof(cutlass::half_t) * size_a, cudaMemcpyHostToDevice, stream));
            if (d_B) CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), sizeof(cutlass::half_t) * size_b, cudaMemcpyHostToDevice, stream));

            try {
                myCutlassHgemm(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, 1.0f, 0.0f, stream);
            } catch (const std::exception &e) {
                std::cerr << "Exception in myCutlassHgemm: " << e.what() << std::endl;
            }

            counts[idx].fetch_add(1, std::memory_order_relaxed);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (d_A) cudaFree(d_A);
            if (d_B) cudaFree(d_B);
            if (d_C) cudaFree(d_C);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    };

    // 启动 num_streams 个线程，每个线程维护一个 stream
    std::vector<std::thread> threads;
    threads.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        threads.emplace_back(worker, i);
    }

    // 运行固定时长
    std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
    stop_flag.store(true);

    for (auto &t : threads) t.join();

    std::cout << "Run duration: " << duration_seconds << " seconds\n";
    for (int i = 0; i < num_streams; ++i) {
        std::cout << "Stream " << i << " completed GEMM submissions: " << counts[i].load() << "\n";
    }

    return 0;
}
