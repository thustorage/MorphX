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
#include <algorithm>
#include <unistd.h>
#include <cstdlib>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include "common.h"


using hrc = std::chrono::high_resolution_clock;
using NanoSec = std::chrono::nanoseconds::rep;

inline bool isSmschedBypass() {
    const char* val = std::getenv("SMSCHED_BYPASS");
    return val != nullptr && std::atoi(val) != 0;
}

inline int getMIFromCI(float CI) {
    return CI >= 0.9 ? 0 : 1;
}

std::atomic<unsigned long long> gLaunchId{1};
int split_hint = 76;
int debug_log = 0;
std::atomic<int> gDebugPrints{0};

inline int getenv_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if(value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::atoi(value);
}

inline bool shouldDebugPrint() {
    return debug_log && gDebugPrints.fetch_add(1) < 200;
}

using cudaLaunchKernelHandler = cudaError_t(
    const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
using cudaMemcpyAsyncHandler = cudaError_t(
    void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
using cudaMemsetAsyncHandler = cudaError_t(
    void *dst, int value, size_t count, cudaStream_t stream);
using cudaMallocAsyncHandler = cudaError_t(
    void **devPtr, size_t size, cudaStream_t stream);
using cudaEventRecordHandler = cudaError_t(cudaEvent_t event, cudaStream_t stream);
using cudaEventSynchronizeHandler = cudaError_t(cudaEvent_t event);
using cudaStreamSynchronizeHandler = cudaError_t(cudaStream_t stream);
using cudaStreamQueryHandler = cudaError_t(cudaStream_t stream);
using cudaDeviceSynchronizeHandler = cudaError_t();


struct KernelInfo {
    const void *func;
    cudaStream_t stream;
    const char* name;
    int occup;
    int gridSize;
    dim3 logicalGrid;
    dim3 blockDim;
    size_t sharedMem;
    int physicalGridX;
    const char* launchPath;
    unsigned long long launchId;

    GdrEntry fetched;
    GdrEntry finished;
    GdrEntry minSM;
    GdrEntry maxSM;

    int h_minSM;
    int h_maxSM;

    hrc::time_point tLaunch;
};

struct StreamInfo {
    cudaStream_t stream;
    boost::lockfree::spsc_queue<KernelInfo*> *pending;
    std::atomic<int> nPending = 0;
    hrc::time_point tStart;
};

struct KernelCacheInfo {
    std::vector<int> argSizes;
    int occup;
};

// #define PROFILE_ITERATION
// #define NO_QUEUEING
#define MAX_CONCURRENT_STREAMS 16

template <typename Duration> 
inline NanoSec getNano(const Duration &d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

void logKernelLaunch(
    unsigned long long launchId,
    const char* path,
    const char* name,
    dim3 logicalGrid,
    dim3 blockDim,
    int physicalGridX,
    size_t sharedMem,
    cudaStream_t stream,
    int occup) {
    if(!shouldDebugPrint()) {
        return;
    }
    printf(
        "[smsched][launch] id=%llu || path=%s || name=%s || logical_grid=(%u,%u,%u) || block=(%u,%u,%u) || physical_grid=(%d,1,1) || shared_mem=%zu || stream=%p || occup=%d\n",
        launchId,
        path,
        name,
        logicalGrid.x,
        logicalGrid.y,
        logicalGrid.z,
        blockDim.x,
        blockDim.y,
        blockDim.z,
        physicalGridX,
        sharedMem,
        stream,
        occup);
}

void logKernelFinish(
    unsigned long long launchId,
    const char* path,
    const char* name,
    int finished,
    int total,
    cudaStream_t stream,
    double elapsedUs) {
    if(!shouldDebugPrint()) {
        return;
    }
    printf(
        "[smsched][finish] id=%llu || path=%s || name=%s || finished=%d/%d || stream=%p || elapsed_us=%.3f\n",
        launchId,
        path,
        name,
        finished,
        total,
        stream,
        elapsedUs);
}

std::unordered_map<std::string, int> func_arg_num;

GdrPool* gGdrPool;

cudaLaunchKernelHandler *_cudaLaunchKernel = nullptr;
cudaMemcpyAsyncHandler *_cudaMemcpyAsync = nullptr;
cudaMemsetAsyncHandler *_cudaMemsetAsync = nullptr;
cudaMallocAsyncHandler *_cudaMallocAsync = nullptr;
cudaEventRecordHandler *_cudaEventRecord = nullptr;
cudaEventSynchronizeHandler *_cudaEventSynchronize = nullptr;
cudaStreamSynchronizeHandler *_cudaStreamSynchronize = nullptr;
cudaStreamQueryHandler *_cudaStreamQuery = nullptr;
cudaDeviceSynchronizeHandler *_cudaDeviceSynchronize = nullptr;

std::unordered_map<const char*, KernelCacheInfo*> gKernelCache;

std::unordered_map<cudaStream_t, StreamInfo*> streamInfoMap;
StreamInfo *streamInfos[MAX_CONCURRENT_STREAMS];
std::atomic<int> nActiveStreams{0};
std::thread agent;
std::atomic<bool> gAgentRunning{true};

int gNrSM;
inline int getNrSM() {
    if(gNrSM == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        gNrSM = prop.multiProcessorCount;
    }
    return gNrSM;
}

inline uint64_t pack4(int x, int y) {
    uint64_t ret = 0;
    ((int*)&ret)[0] = x;
    ((int*)&ret)[1] = y;
    return ret;
}

std::unordered_map<cudaStream_t, std::pair<int, int>> gFixSM;
std::unordered_map<cudaStream_t, std::pair<int, int>> gSuggestSM;

// std::unordered_map<const char *, std::vector<std::pair<int, float>>> gProfileData; // TODO: thread unsafe 
std::unordered_map<const char *, std::atomic<float>> gCI;
std::unordered_map<const char *, std::atomic<float>> gWaveTime;

void ensureCudaRuntimeHooks() {
    if(_cudaLaunchKernel != nullptr) {
        return;
    }
    _cudaLaunchKernel = (cudaLaunchKernelHandler*)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    _cudaMemcpyAsync = (cudaMemcpyAsyncHandler*)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    _cudaMemsetAsync = (cudaMemsetAsyncHandler*)dlsym(RTLD_NEXT, "cudaMemsetAsync");
    _cudaMallocAsync = (cudaMallocAsyncHandler*)dlsym(RTLD_NEXT, "cudaMallocAsync");
    _cudaEventRecord = (cudaEventRecordHandler*)dlsym(RTLD_NEXT, "cudaEventRecord");
    _cudaEventSynchronize = (cudaEventSynchronizeHandler*)dlsym(RTLD_NEXT, "cudaEventSynchronize");
    _cudaStreamSynchronize = (cudaStreamSynchronizeHandler*)dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    _cudaStreamQuery = (cudaStreamQueryHandler*)dlsym(RTLD_NEXT, "cudaStreamQuery");
    _cudaDeviceSynchronize = (cudaDeviceSynchronizeHandler*)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    if(_cudaLaunchKernel == nullptr
        || _cudaMemcpyAsync == nullptr
        || _cudaMemsetAsync == nullptr
        || _cudaMallocAsync == nullptr
        || _cudaEventRecord == nullptr
        || _cudaEventSynchronize == nullptr
        || _cudaStreamSynchronize == nullptr
        || _cudaStreamQuery == nullptr
        || _cudaDeviceSynchronize == nullptr) {
        printf("Failed to get function pointers\n");
        exit(EXIT_FAILURE);
    }
}

void generateProfileWaves(int nrSM, KernelInfo *info, std::vector<int> &profileWaves, int fixMin = -1, int fixMax = -1) {
    if(fixMin != -1) {
        int gridSize = info->gridSize;
        int wave = (fixMax - fixMin) * info->occup;
        while(gridSize >= wave) {
            profileWaves.push_back(wave);
            gridSize -= wave;
        }
        profileWaves.push_back(gridSize);
        return; 
    }
    float waves = 1.0 * info->gridSize / (nrSM * info->occup);
    // printf("[smsched] kernel %s: gridSize: %d, occup: %d, nrSM: %d, waves: %lf\n", info->name, info->gridSize, info->occup, nrSM, waves);
    // fflush(stdout);
    if(waves <= 1) {
        if(info->gridSize <= 16 * info->occup) {
            profileWaves.push_back(info->gridSize);
        } else {
            profileWaves.push_back(info->gridSize - 8 * info->occup);
            profileWaves.push_back(8 * info->occup);
        }
    } else if(waves < 2) {
        profileWaves.push_back(nrSM * info->occup);
        int left = info->gridSize - nrSM * info->occup;
        if(left <= 16 * info->occup) 
            profileWaves.push_back(left);
        else {
            profileWaves.push_back(left - 8 * info->occup);
            profileWaves.push_back(8 * info->occup);
        }
    } else {
        int baseWaves;
        if(waves >= 5) baseWaves = 9;
        else if(waves >= 3.5) baseWaves = 6;
        else if(waves >= 2.5) baseWaves = 4;
        else baseWaves = 3;
        int dec = nrSM / baseWaves;
        int baseBlocks = (dec + nrSM) * info->occup * baseWaves / 2;
        int repeat = info->gridSize / baseBlocks;
        int remain = info->gridSize % baseBlocks;
        // printf("[smsched] baseWaves: %d, dec: %d, baseBlocks: %d, repeat: %d, remain: %d\n", baseWaves, dec, baseBlocks, repeat, remain);
        // printf("[smsched] nrSM: %d, occup: %d\n", nrSM, info->occup);
        for(int k = nrSM * info->occup; k; k -= dec * info->occup) {
            for(int i = 0; i < repeat; ++i) {
                profileWaves.push_back(k);
            }
            if(remain >= k) {
                profileWaves.push_back(k);
                remain -= k;
            }
        }
        if(remain) {
            profileWaves.push_back(remain);
        }
    }
    int sum = 0;
    for(int i = 0; i < profileWaves.size(); ++i) {
        sum += profileWaves[i];
    }
    assert(sum == info->gridSize);
}

inline int getSMFromWave(int wave, int occup) {
    return (wave + occup - 1) / occup;
}

void generateProfileResult(KernelInfo *info, std::vector<int> &profileWaves, std::vector<NanoSec> &profileTimes) {
    if(profileWaves.size() != profileTimes.size()) {
        fprintf(stderr, "Profile size mismatch: %lu vs %lu\n", profileWaves.size(), profileTimes.size());
        exit(EXIT_FAILURE);
    }
    std::vector<std::pair<int, float>> profileResult;
    for(int l = 0, r; l < profileWaves.size(); l = r) {
        float avg = 0;
        for(r = l; r < profileWaves.size() && profileWaves[r] == profileWaves[l]; ++r) {
            avg += profileTimes[r];
        }
        avg /= (r - l);
        profileResult.push_back({getSMFromWave(profileWaves[l], info->occup), avg});
    }
    float tm = profileResult[0].second;
    float t0 = profileResult.back().second;
    gWaveTime[info->name].store(tm / 1000.0);
    if(profileResult[0].first != getNrSM()) {
        tm = tm / profileResult[0].first * getNrSM();
    }
    float CI = t0 / tm;
    if(CI >= 0.9) CI = 1;
    gCI[info->name].store(CI);
    printf(
        "[smsched][profile] name=%s || blocks=%d || occup=%d || waves=%zu || CI=%.6f || MI=%d || WaveTime_us=%.6f\n",
        info->name,
        info->gridSize,
        info->occup,
        profileResult.size(),
        CI,
        getMIFromCI(CI),
        tm / 1000.0);
    for(int i = 0; i < profileResult.size(); ++i) {
        printf(
            "[smsched][profile][wave] name=%s || idx=%d || sms=%d || avg_latency_us=%.6f\n",
            info->name,
            i,
            profileResult[i].first,
            profileResult[i].second / 1000.0);
    }
    fflush(stdout);
}

void bind_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);   

    pthread_t thread = pthread_self();
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
    }
}
std::atomic<int> gCpu;

inline void writeSM(KernelInfo *info, int minSM, int maxSM) {
    if(minSM == info->h_minSM && maxSM == info->h_maxSM)
        return;
    info->h_minSM = minSM;
    info->h_maxSM = maxSM;
    gGdrPool->set(info->minSM, pack4(minSM, maxSM));
}

void profileThread(KernelInfo *launchInfo, std::vector<int> &profileWaves) {
    const char* name = launchInfo->name;
    if(strstr(name, "flash_fwd_kernel")) {
        gCI[name].store(1.0);
        return;
    }
    std::vector<NanoSec> profileTimes;
    int fetched, finished;
    int tot = 0;
    for(int i = 0; i < profileWaves.size(); ++i) {
        tot += profileWaves[i];
        auto t0 = hrc::now();
        do {
            uint64_t tmp = gGdrPool->get(launchInfo->fetched);
            fetched = ((int*)&tmp)[0];
            finished = ((int*)&tmp)[1];
            // printf("[smsched][profile]: wave %d/%lu, fetched: %d, finished: %d, tot: %d\n", 
            //     i + 1, profileWaves.size(), fetched, finished, tot);
            if(fetched >= tot && i + 1 < profileWaves.size() && profileWaves[i+1] != profileWaves[i]) {
                writeSM(launchInfo, 0, getSMFromWave(profileWaves[i+1], launchInfo->occup));
            }
        } while(finished < tot);
        auto t1 = hrc::now();
        profileTimes.push_back(getNano(t1 - t0));
    }
    generateProfileResult(launchInfo, profileWaves, profileTimes);
}

int getStreamRank(cudaStream_t stream) {
    for(int i = 0; i < nActiveStreams.load(); ++i) {
        if(streamInfos[i]->stream == stream) {
            return i;
        }
    }
    return -1;
}

void applySplitByStreamRank(KernelInfo *info, int nrSM) {
    if(info == nullptr || nrSM <= 1) {
        return;
    }
    int split = std::max(1, std::min(split_hint, nrSM - 1));
    int rank = getStreamRank(info->stream);
    int minSM = rank <= 0 ? 0 : split;
    int maxSM = rank <= 0 ? split : nrSM;
    writeSM(info, minSM, maxSM);
    if(shouldDebugPrint()) {
        fprintf(stderr, "[smsched-debug] stub assign kernel=%s stream=%p rank=%d range=[%d,%d) nStreams=%d\n",
            info->name, info->stream, rank, minSM, maxSM, nActiveStreams.load());
    }
}

void schedule(KernelInfo *kernels[], int n, int nrSM) {
    if(n <= 1) 
        return;
    for(int i = 0; i < n; ++i) {
        applySplitByStreamRank(kernels[i], nrSM);
    }
}

void agentThread() {
    // bind_thread_to_cpu(gCpu.fetch_add(1));
    int nrSM = getNrSM();
    KernelInfo *kernels[MAX_CONCURRENT_STREAMS];
    memset(kernels, 0, sizeof(kernels));
    int nActiveKernels = 0;
    KernelInfo *activeKernels[MAX_CONCURRENT_STREAMS];
    while(gAgentRunning.load(std::memory_order_relaxed)) {
        int nStreams = nActiveStreams.load();
        nActiveKernels = 0;
        for(int i = 0; i < nStreams; i++) {
            if(kernels[i] == nullptr) {
                if(!streamInfos[i]->pending->empty()) {
                    streamInfos[i]->pending->pop(kernels[i]);
                }
            }
            if(kernels[i] == nullptr) {
                continue;
            }
            uint64_t tmp = gGdrPool->get(kernels[i]->fetched);
            int fetched = ((int*)&tmp)[0];
            int finished = ((int*)&tmp)[1];
            if(fetched == 0) {
                continue;
            }
            if(finished >= kernels[i]->gridSize) {
                float elapsedUs = getNano(hrc::now() - kernels[i]->tLaunch) / 1000.0;
                logKernelFinish(
                    kernels[i]->launchId,
                    kernels[i]->launchPath,
                    kernels[i]->name,
                    finished,
                    kernels[i]->gridSize,
                    kernels[i]->stream,
                    elapsedUs);
                gGdrPool->gdr_free(kernels[i]->fetched);
                gGdrPool->gdr_free(kernels[i]->minSM);
                delete kernels[i];
                kernels[i] = nullptr;
                continue;
            }
            assert(fetched > finished);
            activeKernels[nActiveKernels++] = kernels[i];
        }
        schedule(activeKernels, nActiveKernels, nrSM);
    }
}

StreamInfo* registerStream(cudaStream_t stream) {
    // printf("Register stream %p\n", stream);
    StreamInfo *streamInfo = new StreamInfo();
    streamInfo->stream = stream;
    streamInfo->pending = new boost::lockfree::spsc_queue<KernelInfo*>(65536);
    streamInfo->nPending.store(0);
    streamInfoMap[stream] = streamInfo;
    if(nActiveStreams.load() >= 16) {
        std::cerr << "Too many streams" << std::endl;
        exit(EXIT_FAILURE);
    }
    streamInfos[nActiveStreams.load()] = streamInfo;
    nActiveStreams.fetch_add(1);
    return streamInfo;
}

bool isInitialized = false;

// Ensure the background agent thread is properly joined before process exit
static void runtimeShutdown() {
    if(!isInitialized) return;
    gAgentRunning.store(false, std::memory_order_relaxed);
    if(agent.joinable()) {
        agent.join();
    }
}

void runtimeInit() {
    ensureCudaRuntimeHooks();
    split_hint = getenv_int("SMSCHED_SPLIT_HINT", 76);
    debug_log = getenv_int("SMSCHED_DEBUG", 0);
    gGdrPool = new GdrPool();
    agent = std::thread(agentThread);
    // Register graceful shutdown to avoid std::terminate at process exit
    std::atexit(runtimeShutdown);
    isInitialized = true;
}

extern "C"
{

void register_name_with_arg_num(char *name, int arg_num) {
    // printf("registering kernel %s %d\n", name, arg_num);
    // func_arg_num[std::string(name)] = arg_num;
}

void fix_SM(cudaStream_t stream, int minSM, int maxSM) {
    gFixSM[stream] = std::make_pair(minSM, maxSM);
    if(shouldDebugPrint()) {
        printf("fix SM for stream %p: %d %d\n", stream, minSM, maxSM);
    }
}

void suggest_SM(cudaStream_t stream, int minSM, int maxSM) {
    gSuggestSM[stream] = std::make_pair(minSM, maxSM);
    if(shouldDebugPrint()) {
        printf("suggest SM for stream %p: %d %d\n", stream, minSM, maxSM);
    }
}

inline StreamInfo* checkStream(cudaStream_t stream) {
    if(stream == 0) {
        std::cerr << "Default stream is not supported" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(!isInitialized) {
        runtimeInit();
    }
    auto streamInfoIter = streamInfoMap.find(stream);
    if(streamInfoIter == streamInfoMap.end()) {
        return registerStream(stream);
    } else {
        return streamInfoIter->second;
    }
}

cudaError_t CUDARTAPI cudaLaunchKernel(
    const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream
) {
    ensureCudaRuntimeHooks();
    if(isSmschedBypass()) {
        return _cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }
    StreamInfo *streamInfo = checkStream(stream);
    sharedMem = std::max(sharedMem, 4ul);
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    KernelInfo *launchInfo = new KernelInfo;
    launchInfo->func = func;
    launchInfo->stream = stream;
    const char* funcName;
    CHECK_CUDA_ERROR(cudaFuncGetName(&funcName, launchInfo->func));
    launchInfo->name = funcName;
    launchInfo->logicalGrid = gridDim;
    launchInfo->blockDim = blockDim;
    launchInfo->sharedMem = sharedMem;
    // printf("Launch kernel to stream %p in thread %ld\n", stream, std::this_thread::get_id());
    // fflush(stdout);
    // if(!func_arg_num.count(std::string(launchInfo->name))) {
    //     std::cerr << "Kernel " << launchInfo->name << " is not registered" << std::endl;
    //     return cudaErrorInvalidValue;
    // }
    KernelCacheInfo *cacheInfo;
    if(!gKernelCache.count(launchInfo->name)) {
        cacheInfo = new KernelCacheInfo();
        gKernelCache[launchInfo->name] = cacheInfo;
        int i = 0;
        cacheInfo->argSizes.clear();
        size_t offset, size;
        while(cudaFuncGetParamInfo(func, i, &offset, &size) == cudaSuccess) {
            cacheInfo->argSizes.push_back(size);
            i++;
        }
        auto last_error = cudaGetLastError();
        CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cacheInfo->occup, func, blockSize, sharedMem));
    } else {
        cacheInfo = gKernelCache[launchInfo->name];
    }
    int numArgs = cacheInfo->argSizes.size();
    void **nargs = new void*[numArgs];
    for(int i = 0; i < numArgs - 5; ++i) {
        nargs[i] = malloc(cacheInfo->argSizes[i]);
        memcpy(nargs[i], args[i], cacheInfo->argSizes[i]);
    }
    launchInfo->occup = cacheInfo->occup;

    int nrSM = getNrSM();
    int gridSize = gridDim.x * gridDim.y * gridDim.z;
    launchInfo->gridSize = gridSize;
    launchInfo->fetched = gGdrPool->gdr_malloc();
    launchInfo->finished = launchInfo->fetched.half();
    launchInfo->minSM = gGdrPool->gdr_malloc();
    launchInfo->maxSM = launchInfo->minSM.half();
    gGdrPool->set(launchInfo->fetched, 0);
    int fixMin = -1, fixMax = -1;
    if(gFixSM.count(stream)) {
        auto p = gFixSM[stream];
        fixMin = p.first; fixMax = p.second;
    } 
    nargs[numArgs - 5] = &gridDim;
    nargs[numArgs - 4] = &launchInfo->fetched.d;
    nargs[numArgs - 3] = &launchInfo->finished.d;
    nargs[numArgs - 2] = &launchInfo->minSM.d;
    nargs[numArgs - 1] = &launchInfo->maxSM.d;
    int agents = launchInfo->occup * nrSM; // NOT min(occup * nrSM, gridSize)!!!!
    launchInfo->physicalGridX = agents;
    // printf("Time to launch kernel %s: %lfus\n", launchInfo->name, getNano(hrc::now() - launchInfo->tLaunch) / 1000.0);
    // printf("[smsched] kernel launched: %s, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMem: %lu, stream: %p, numArgs: %d, occup: %d, agents: %d\n", 
    //     launchInfo->name, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, launchInfo->stream, numArgs, occup, agents);
    // fflush(stdout);
    std::thread* profile = nullptr;
    std::vector<int> profileWaves;
    // if(!gCI.count(launchInfo->name)) {
    //     generateProfileWaves(nrSM, launchInfo, profileWaves, fixMin, fixMax);
    //     launchInfo->h_minSM = 0;
    //     launchInfo->h_maxSM = getSMFromWave(profileWaves[0], launchInfo->occup);
    //     gGdrPool->set(launchInfo->minSM, pack4(0, launchInfo->h_maxSM));
    //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //     profile = new std::thread(profileThread, launchInfo, std::ref(profileWaves));
    // } else {
        launchInfo->h_minSM = 0;
        launchInfo->h_maxSM = nrSM;
        gGdrPool->set(launchInfo->minSM, pack4(0, nrSM));
        if(nActiveStreams.load() >= 2) {
            applySplitByStreamRank(launchInfo, nrSM);
        }
    // }

    launchInfo->launchPath = profile ? "profile" : "scheduled";
    launchInfo->launchId = gLaunchId.fetch_add(1, std::memory_order_relaxed);
    logKernelLaunch(
        launchInfo->launchId,
        launchInfo->launchPath,
        launchInfo->name,
        launchInfo->logicalGrid,
        launchInfo->blockDim,
        launchInfo->physicalGridX,
        launchInfo->sharedMem,
        launchInfo->stream,
        launchInfo->occup);
    auto res = _cudaLaunchKernel(func, dim3(agents, 1, 1), blockDim, nargs, sharedMem, stream);

    if(profile) {
        profile->join();
        delete profile;
        if(res == cudaSuccess) {
            uint64_t tmp = gGdrPool->get(launchInfo->fetched);
            int finished = ((int*)&tmp)[1];
            float elapsedUs = getNano(hrc::now() - launchInfo->tLaunch) / 1000.0;
            logKernelFinish(
                launchInfo->launchId,
                launchInfo->launchPath,
                launchInfo->name,
                finished,
                launchInfo->gridSize,
                launchInfo->stream,
                elapsedUs);
        }
        gGdrPool->gdr_free(launchInfo->fetched);
        gGdrPool->gdr_free(launchInfo->minSM);
        delete launchInfo;
    } else {
        launchInfo->tLaunch = hrc::now();
        streamInfo->nPending.fetch_add(1);
        streamInfo->pending->push(launchInfo);
    }

    for(int i = 0; i < numArgs - 5; ++i) {
        free(nargs[i]);
    }
    delete[] nargs;
    return res;
}

#define MAKE_CUDA_METHOD(symbol, params, ...)                                   \
cudaError_t CUDARTAPI symbol params {                                                     \
    using symbol##Handler = cudaError_t(params);                                \
    auto _cuda_func = (symbol##Handler*)dlsym(RTLD_NEXT, #symbol);                    \
    if (_cuda_func == nullptr)  {                                                     \
        std::cout << "Interception method is not found: " << #symbol <<         \
                        ", error: " << dlerror() << std::endl;                  \
        return cudaErrorUnknown;                                                \
    }                                                                           \
    else {                                                                      \
        const auto res = _cuda_func(__VA_ARGS__);                               \
        printf("Using %s that is not implemented!\n", #symbol);                \
        exit(EXIT_FAILURE);                                                \
        return res;                                                             \
    }                                                                           \
    return cudaSuccess;                                                         \
}

// Rarely used functions, but may cause error. Raise error if called.
// MAKE_CUDA_METHOD(cudaMemPrefetchAsync, (const void* devPtr, size_t count, int dstDevice, cudaStream_t stream), devPtr, count, dstDevice, stream)
// MAKE_CUDA_METHOD(cudaMemcpy2DAsync, (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), dst, dpitch, src, spitch, width, height, kind, stream)
// MAKE_CUDA_METHOD(cudaMemcpy2DFromArrayAsync, (void *dst, size_t dpitch, cudaArray_const_t srcArray, size_t srcOffsetXInBytes, size_t srcOffsetY, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), dst, dpitch, srcArray, srcOffsetXInBytes, srcOffsetY, width, height, kind, stream)
// MAKE_CUDA_METHOD(cudaMemcpy2DToArrayAsync, (cudaArray_t dstArray, size_t dstOffsetXInBytes, size_t dstOffsetY, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), dstArray, dstOffsetXInBytes, dstOffsetY, src, spitch, width, height, kind, stream)
// MAKE_CUDA_METHOD(cudaMemcpy3DAsync, (const cudaMemcpy3DParms *p, cudaStream_t stream), p, stream)
// MAKE_CUDA_METHOD(cudaMemcpy3DBatchAsync, (const cudaMemcpy3DParms *p, cudaStream_t stream), p, stream)
// MAKE_CUDA_METHOD(cudaMemcpy3DPeerAsync, (const cudaMemcpy3DPeerParms *p, cudaStream_t stream), p, stream)
// MAKE_CUDA_METHOD(cudaMemcpyBatchAsync, (const cudaMemcpy3DParms *p, cudaStream_t stream), p, stream)
// MAKE_CUDA_METHOD(cudaMemcpyFromSymbolAsync, (void *dst, const void *symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), dst, symbol, count, offset, kind, stream)
// MAKE_CUDA_METHOD(cudaMemcpyPeerAsync, (void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream), dst, dstDevice, src, srcDevice, count, stream)
// MAKE_CUDA_METHOD(cudaMemcpyToSymbolAsync, (const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), symbol, src, count, offset, kind, stream)
// MAKE_CUDA_METHOD(cudaMemset2DAsync, (void *dst, size_t dpitch, int value, size_t width, size_t height, cudaStream_t stream), dst, dpitch, value, width, height, stream)
// MAKE_CUDA_METHOD(cudaMemset3DAsync, (cudaPitchedPtr dst, int value, cudaExtent extent, cudaStream_t stream), dst, value, extent, stream)
// MAKE_CUDA_METHOD(cudaMallocFromPoolAsync, (void **devPtr, size_t size, cudaMemPool_t memPool, cudaStream_t stream), devPtr, size, memPool, stream)
MAKE_CUDA_METHOD(cudaStreamBeginCapture, (cudaStream_t stream, cudaStreamCaptureMode mode), stream, mode)
MAKE_CUDA_METHOD(cudaStreamEndCapture, (cudaStream_t stream, cudaGraph_t *graph), stream, graph)

// MAKE_CUDA_METHOD(cudaMemcpy, (void *dst, const void *src, size_t count, cudaMemcpyKind kind), dst, src, count, kind)
// MAKE_CUDA_METHOD(cudaMemset, (void *dst, int value, size_t count), dst, value, count)
// MAKE_CUDA_METHOD(cudaStreamWaitEvent, (cudaStream_t stream, cudaEvent_t event, unsigned int flags), stream, event, flags)

} // extern "C"
