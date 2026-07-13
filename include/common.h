#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <gdrapi.h>
#include <chrono>
#include <assert.h>
#include <dlfcn.h>
#include <boost/lockfree/queue.hpp>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            assert(0); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CU_ERROR(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char *err_name; \
            cuGetErrorName(err, &err_name); \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << err_name << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            auto cuda_err = cudaGetLastError(); \
            std::cerr << "CUBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(cuda_err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define GDR_POOL_SIZE (GPU_PAGE_SIZE << 10) // 1MB
#define GDR_ENTRY_SIZE 8
#define GDR_BUFFER_SIZE (512) // TODO: 512

struct GdrEntry {
    CUdeviceptr d;
    uintptr_t h;
    GdrEntry half() {
        return {d + GDR_ENTRY_SIZE / 2, h + GDR_ENTRY_SIZE / 2};
    }
};

class GdrPool {
private:
    CUdeviceptr d_pool;
    uintptr_t h_pool;
    gdr_t g;
    gdr_mh_t g_mh;
    boost::lockfree::queue<GdrEntry> buffer{GDR_POOL_SIZE / GDR_ENTRY_SIZE};
public:
    GdrPool() {
        g = gdr_open();
        if(g == nullptr) {
            std::cerr << "Failed to open gdr";
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pool, GDR_POOL_SIZE));
        CHECK_CUDA_ERROR(cudaMemset((void*)d_pool, 0, GDR_POOL_SIZE));
        d_pool = (d_pool + (GPU_PAGE_SIZE - 1)) & ~(GPU_PAGE_SIZE - 1);
        if(gdr_pin_buffer(g, (unsigned long)d_pool, GDR_POOL_SIZE, 0, 0, &g_mh) != 0) {
            std::cerr << "Failed to pin input buffer";
            exit(EXIT_FAILURE);
        }
        if (gdr_map(g, g_mh, (void**)&h_pool, GDR_POOL_SIZE) != 0) {
            std::cerr << "Failed to map input GPU buffer";
            exit(EXIT_FAILURE);
        }
        gdr_info_t info;
        if(gdr_get_info(g, g_mh, &info) != 0) {
            std::cerr << "Failed to get info";
            exit(EXIT_FAILURE);
        }
        int off = info.va - d_pool;
        h_pool = h_pool + off;
        d_pool = d_pool + off;
        // printf("h_pool: %lx, d_pool: %llx\n", h_pool, d_pool);
        for(size_t i = 0; i < GDR_POOL_SIZE; i += GDR_ENTRY_SIZE) {
            GdrEntry entry({d_pool + i, h_pool + i});
            if(!buffer.push(entry)) {
                std::cerr << "GDR pool initialization failed\n";
                exit(EXIT_FAILURE);
            }
        }
    }
    inline GdrEntry gdr_malloc() {
        GdrEntry entry;
        if(!buffer.pop(entry)) {
            std::cerr << "GDR pool is empty\n";
            exit(EXIT_FAILURE);
        }
        return entry;
    }
    inline void gdr_free(GdrEntry entry) {
        if(!buffer.push(entry)) {
            std::cerr << "GDR pool push failed\n";
            exit(EXIT_FAILURE);
        } 
    }

    inline void get_buffer(int bufferId, void *h_buffer) {
        if(gdr_copy_from_mapping(g_mh, (void*)(h_buffer), (void*)(h_pool + bufferId * GDR_BUFFER_SIZE), GDR_BUFFER_SIZE) != 0) {
            std::cerr << "Failed to copy from mapping\n";
            exit(EXIT_FAILURE);
        }
    }
    inline void set_buffer(int bufferId, void *h_buffer) {
        if(gdr_copy_to_mapping(g_mh, (void*)(h_pool + bufferId * GDR_BUFFER_SIZE), (void*)(h_buffer), GDR_BUFFER_SIZE) != 0) {
            std::cerr << "Failed to copy to mapping\n";
            exit(EXIT_FAILURE);
        }
    }

    inline int get_index(GdrEntry &entry) {
        return ((entry.h - h_pool) % GDR_BUFFER_SIZE) / sizeof(int);
    }
    inline int get_bufferId(GdrEntry &entry) {
        return (entry.h - h_pool) / GDR_BUFFER_SIZE;
    }

    inline void set(GdrEntry &entry, uint64_t value) {
        gdr_copy_to_mapping(g_mh, (void*)entry.h, &value, GDR_ENTRY_SIZE);
    }
    inline uint64_t get(GdrEntry &entry) {
        uint64_t ret = 0;
        gdr_copy_from_mapping(g_mh, &ret, (void*)entry.h, GDR_ENTRY_SIZE);
        return ret;
    }

    ~GdrPool() {
        if(gdr_unmap(g, g_mh, (void*)h_pool, GDR_POOL_SIZE) != 0) {
            std::cerr << "Failed to unmap input GPU buffer";
            exit(EXIT_FAILURE);
        }
        if(gdr_unpin_buffer(g, g_mh) != 0) {
            std::cerr << "Failed to unpin input buffer";
            exit(EXIT_FAILURE);
        }
        if(gdr_close(g) != 0) {
            std::cerr << "Failed to close gdr";
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA_ERROR(cudaFree((void*)d_pool));
        while(!buffer.empty()) {
            GdrEntry entry;
            buffer.pop(entry);
        }
    }
};

typedef void (*fixSMHandle)(cudaStream_t, int, int);
typedef void (*suggestSMHandle)(cudaStream_t, int, int);

fixSMHandle getFixSMHandle() {
    void *handle = dlopen(nullptr, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error opening handle: " << dlerror() << std::endl;
        return nullptr;
    }
    fixSMHandle func = (fixSMHandle)dlsym(handle, "fix_SM");
    if (!func) {
        std::cerr << "Error loading symbol: " << dlerror() << std::endl;
        return nullptr;
    }
    return func;
}

suggestSMHandle getSuggestSMHandle() {
    void *handle = dlopen(nullptr, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error opening handle: " << dlerror() << std::endl;
        return nullptr;
    }
    suggestSMHandle func = (suggestSMHandle)dlsym(handle, "suggest_SM");
    if (!func) {
        std::cerr << "Error loading symbol: " << dlerror() << std::endl;
        return nullptr;
    }
    return func;
}

bool fixSMForStream(cudaStream_t stream, int minSM, int maxSM) {
    fixSMHandle func = getFixSMHandle();
    if (func) {
        func(stream, minSM, maxSM);
        return true;
    }
    return false;
}

bool suggestSMForStream(cudaStream_t stream, int minSM, int maxSM) {
    suggestSMHandle func = getSuggestSMHandle();
    if (func) {
        func(stream, minSM, maxSM);
        return true;
    }
    return false;
}