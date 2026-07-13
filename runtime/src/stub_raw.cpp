#include <cuda.h>
#include <stdio.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <string>
#include <queue>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>
#include "common.h"

using hrc = std::chrono::high_resolution_clock;
using NanoSec = std::chrono::nanoseconds::rep;
template <typename Duration> 
inline NanoSec getNano(const Duration &d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

std::unordered_map<std::string, int> func_arg_num;

GdrPool* gGdrPool;
std::mutex gGdrPoolMutex;
#define GDR_SET(entry, value) { \
        std::lock_guard<std::mutex> lock(gGdrPoolMutex); \
        gGdrPool->set(entry, value); }
#define GDR_GET(entry) ({ \
        int value; \
        { \
            std::lock_guard<std::mutex> lock(gGdrPoolMutex); \
            value = gGdrPool->get(entry); \
        } \
        value; })

using cudaLaunchKernelHandler = cudaError_t(
    const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaLaunchKernelHandler *_cuda_func;



extern "C"
{

void register_name_with_arg_num(char *name, int arg_num) {
    // printf("registering kernel %s %d %p\n", name, arg_num, func);
    func_arg_num[std::string(name)] = arg_num;
}

cudaError_t CUDARTAPI cudaLaunchKernel(
    const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream
) {
    if(_cuda_func == nullptr) {
        _cuda_func = (cudaLaunchKernelHandler*)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    } 
    if (_cuda_func == nullptr)  {
        std::cout << "Interception method is not found: cudaLaunchKernel" <<   
                        ", error: " << dlerror() << std::endl; 
        return cudaErrorUnknown; 
    }
    if(stream == 0) {
        std::cerr << "Default stream is not supported" << std::endl;
        return cudaErrorInvalidValue;
    }

    int *sig, *sig_h = (int*)malloc(sizeof(int) * 4);
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)&sig, sizeof(int) * 4, stream));
    sig_h[0] = 0;
    sig_h[1] = 0;
    sig_h[2] = 0;
    sig_h[3] = 108;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(sig, sig_h, sizeof(int) * 4, cudaMemcpyHostToDevice, stream));
    int *fetched = sig, *finished = sig + 1, *minSM = sig + 2, *maxSM = sig + 3;
    const char* funcName;
    CHECK_CUDA_ERROR(cudaFuncGetName(&funcName, func));
    std::string funcNameStr(funcName, strlen(funcName));
    if(!func_arg_num.count(funcNameStr)) {
        std::cerr << "Kernel " << funcNameStr << " is not registered" << std::endl;
        exit(EXIT_FAILURE);
    }
    int nrSM = 108;
    sharedMem = std::max(sharedMem, 4ul);
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int gridSize = gridDim.x * gridDim.y * gridDim.z;
    int occup = 0;
    int numArgs = func_arg_num[funcNameStr];
    void **nargs = new void*[numArgs + 5];
    for(int i = 0; i < numArgs; ++i) {
        nargs[i] = args[i];
    }
    nargs[numArgs] = &gridDim;
    nargs[numArgs + 1] = &fetched;
    nargs[numArgs + 2] = &finished;
    nargs[numArgs + 3] = &minSM;
    nargs[numArgs + 4] = &maxSM;
    CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occup, func, blockSize, sharedMem));
    
    printf("kernel: %s, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMem: %lu, stream: %p, numArgs: %d, occup: %d\n", 
        funcName, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, numArgs, occup);
    fflush(stdout);
    int agents = occup * nrSM;
    auto res = _cuda_func(func, dim3(agents, 1, 1), blockDim, nargs, sharedMem, stream);
    delete[] nargs;
    return res;
}

// #define MAKE_CUDA_STREAM_CREATE_METHOD(symbol, params, ...)                                   \
// cudaError_t CUDARTAPI symbol params {                                                     \
//     using symbol##Handler = cudaError_t(params);                                \
//     auto _cuda_func = (symbol##Handler*)dlsym(RTLD_NEXT, #symbol);                    \
//     if (_cuda_func == nullptr)  {                                                     \
//         std::cout << "Interception method is not found: " << #symbol <<         \
//                         ", error: " << dlerror() << std::endl;                  \
//         return cudaErrorUnknown;                                                \
//     }                                                                           \
//     else {                                                                      \
//         const auto res = _cuda_func(__VA_ARGS__);                               \
//         registerStream(*pStream);                                               \
//         return res;                                                             \
//     }                                                                           \
//     return cudaSuccess;                                                         \
// }

// MAKE_CUDA_STREAM_CREATE_METHOD(cudaStreamCreate, (cudaStream_t *pStream), pStream)
// MAKE_CUDA_STREAM_CREATE_METHOD(cudaStreamCreateWithFlags, 
//     (cudaStream_t *pStream, unsigned int flags), pStream, flags)
// MAKE_CUDA_STREAM_CREATE_METHOD(cudaStreamCreateWithPriority, 
//     (cudaStream_t *pStream, unsigned int flags, int priority), pStream, flags, priority)


// cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
//     if(!streamInfoMap.count(stream)) {
//         return cudaErrorInvalidResourceHandle;
//     }
//     ThreadInfo &threadInfo = streamInfoMap[stream];
//     while(threadInfo.kernelQueuLength.load() > 0) {
//         std::this_thread::yield();
//     }
//     return cudaSuccess;
// }

// cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
//     if(!streamInfoMap.count(stream)) {
//         return cudaErrorInvalidResourceHandle;
//     }
//     ThreadInfo &threadInfo = streamInfoMap[stream];
//     return threadInfo.kernelQueuLength.load() == 0 ? cudaSuccess : cudaErrorNotReady;
// }


} // extern "C"