#include "preload.h"
#include "common.h"
#include <cuda.h>   // for cuda related definition

#include <dlfcn.h>
#include <fcntl.h>
#include <execinfo.h>  // for backtrace and backtrace_symbols
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <unistd.h>
#include <cstdlib>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#define USE_INCREMENTAL

/**
 * Undefine some symbols updated to v2. These are some historical issue with
 * 32bit machine. Now NVIDIA update them to v2 for 64bit.
 */
#undef cuMemAlloc
#undef cuStreamGetCaptureInfo
#undef cuArray3DCreate
#undef cuArray3DGetDescriptor
#undef cuArrayCreate
#undef cuArrayGetDescriptor
#undef cuCtxCreate
#undef cuCtxDestroy
#undef cuCtxPopCurrent
#undef cuCtxPushCurrent
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxSetFlags
#undef cuDeviceTotalMem
#undef cuEventDestroy
#undef cuGetProcAddress
#undef cuGraphAddKernelNode
#undef cuGraphExecKernelNodeSetParams
#undef cuGraphExecUpdate
#undef cuGraphicsResourceGetMappedPointer
#undef cuGraphicsResourceSetMapFlags
#undef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeSetParams
#undef cuIpcOpenMemHandle
#undef cuLinkAddData
#undef cuLinkAddFile
#undef cuLinkCreate
#undef cuMemAllocHost
#undef cuMemAllocPitch
#undef cuMemcpy2DAsync
#undef cuMemcpy2DUnaligned
#undef cuMemcpy2D
#undef cuMemcpy3DAsync
#undef cuMemcpy3D
#undef cuMemcpyAtoA
#undef cuMemcpyAtoD
#undef cuMemcpyAtoHAsync
#undef cuMemcpyAtoH
#undef cuMemcpyDtoA
#undef cuMemcpyDtoDAsync
#undef cuMemcpyDtoD
#undef cuMemcpyDtoHAsync
#undef cuMemcpyDtoH
#undef cuMemcpyHtoAAsync
#undef cuMemcpyHtoA
#undef cuMemcpyHtoDAsync
#undef cuMemcpyHtoD
#undef cuMemFree
#undef cuMemGetAddressRange
#undef cuMemGetInfo
#undef cuMemHostGetDevicePointer
#undef cuMemHostRegister
#undef cuMemsetD16
#undef cuMemsetD2D16
#undef cuMemsetD2D32
#undef cuMemsetD2D8
#undef cuMemsetD32
#undef cuMemsetD8
#undef cuModuleGetGlobal
#undef cuStreamBatchMemOp
#undef cuStreamBeginCapture
#undef cuStreamDestroy
#undef cuStreamWaitValue32
#undef cuStreamWaitValue64
#undef cuStreamWriteValue32
#undef cuStreamWriteValue64
#undef cuTexRefGetAddress
#undef cuTexRefSetAddress2D
#undef cuTexRefSetAddress

#define WARP_SIZE 32 // NVIDIA GPUs use 32 for WARP_SIZE

typedef CUresult (*cuModuleLoadData_func_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleLoadDataEx_func_t)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
typedef CUresult (*cuModuleGetFunction_func_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuKernelGetFunction_func_t)(CUfunction*, CUkernel);
typedef CUresult (*cuLibraryGetKernel_func_t)(CUkernel*, CUlibrary, const char*);
typedef CUresult (*cuLibraryGetModule_func_t)(CUmodule*, CUlibrary);
typedef CUresult (*cuLibraryLoadData_func_t)(CUlibrary*, const void*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int);
typedef CUresult (*cuLaunchKernel_func_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
typedef CUresult (*cuModuleLoad_func_t)(CUmodule*, const char*);
typedef CUresult (*cuModuleLoadFatBinary_func_t)(CUmodule*, const void*);
typedef CUresult (*cuLaunchKernelEx_func_t)(const CUlaunchConfig*, CUfunction, void**, void**);
typedef CUresult (*cuGetProcAddress_func_t)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
typedef CUresult (*cuGetProcAddress_v2_func_t)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

static cuModuleLoadData_func_t real_cuModuleLoadData = NULL;
static cuModuleLoadDataEx_func_t real_cuModuleLoadDataEx = NULL;
static cuModuleGetFunction_func_t real_cuModuleGetFunction = NULL;
static cuKernelGetFunction_func_t real_cuKernelGetFunction = NULL;
static cuLibraryGetKernel_func_t real_cuLibraryGetKernel = NULL;
static cuLibraryGetModule_func_t real_cuLibraryGetModule = NULL;
static cuLibraryLoadData_func_t real_cuLibraryLoadData = NULL;
static cuLaunchKernel_func_t real_cuLaunchKernel = NULL;
static cuModuleLoad_func_t real_cuModuleLoad = NULL;
static cuModuleLoadFatBinary_func_t real_cuModuleLoadFatBinary = NULL;
static cuLaunchKernelEx_func_t real_cuLaunchKernelEx = NULL;
static cuGetProcAddress_func_t real_cuGetProcAddress = NULL;
static cuGetProcAddress_v2_func_t real_cuGetProcAddress_v2 = NULL;

// helper macro to check cuda error
#define CU_CHECK(call) \
    do { \
        CUresult result = (call); \
        if (result != CUDA_SUCCESS) { \
            const char* errorStr; \
            cuGetErrorString(result, &errorStr); \
            printf("%s failed with error: %d, %s (at %s:%d)\n", #call, result, errorStr, __FILE__, __LINE__); \
            exit(-1); \
        } \
    } while (0)

#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

/////////////////////////////////////////////////////////////////////
// Below is the implementation of SM scheduler
/////////////////////////////////////////////////////////////////////

using hrc = std::chrono::high_resolution_clock;
using NanoSec = std::chrono::nanoseconds::rep;

struct KernelInfo {
    CUfunction func;
    CUstream stream;
    const char* name;
    int occup;
    int gridSize;
    int numArgs;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    dim3 logicalGrid;

    GdrEntry d_agents;
    GdrEntry fetched;
    GdrEntry finished;
    GdrEntry minSM;
    GdrEntry maxSM;

    int h_minSM;
    int h_maxSM;

    int isScheduled;
    int isReclaimed;

    std::vector<void*> kernelArgs;
    std::vector<GdrEntry> reclaimAgents;
    std::vector<GdrEntry> reclaimMinSMs;
};

struct StreamInfo {
    CUstream stream;
    boost::lockfree::spsc_queue<KernelInfo*> *pending;
    hrc::time_point tStart;
};

enum KernelType {
    MEMORY_INTENSIVE = 0, 
    COMPUTE_INTENSIVE = 1
};

KernelType getKernelType(float CI) {
    if(CI >= 0.9) return COMPUTE_INTENSIVE;
    return MEMORY_INTENSIVE;
}

// #define PROFILE_ITERATION
// #define NO_QUEUEING
#define MAX_CONCURRENT_STREAMS 16

template <typename Duration> 
inline NanoSec getNano(const Duration &d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

GdrPool* gGdrPool;

std::unordered_map<CUstream, StreamInfo*> streamInfoMap;
StreamInfo *streamInfos[MAX_CONCURRENT_STREAMS];
std::atomic<int> nActiveStreams{0};

std::thread agent;
std::atomic<bool> gAgentRunning{true};
std::shared_mutex launchMutex;

std::unordered_map<CUfunction, CUfunction> gPkFuncMap;

int with_profile;
int profile_log;
int split_hint;
int debug_log;
int debug_limit;
int only_whitelist;
int register_bypass_streams;
int reclaim_enabled;
std::vector<std::string> targetKernelPatterns;

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

std::unordered_map<CUstream, std::pair<int, int>> gFixSM;
std::unordered_map<CUstream, std::pair<int, int>> gSuggestSM;
std::unordered_map<CUfunction, std::string> gFuncNameMap;
std::unordered_map<CUkernel, std::string> gKernelNameMap;
CUstream gReclaimStream = nullptr;

// std::unordered_map<const char *, std::vector<std::pair<int, float>>> gProfileData; // TODO: thread unsafe 
std::unordered_map<const char *, float> gCI;
std::unordered_map<const char *, float> gWaveTime;

bool isTargetKernelName(const char* name);

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

void generateProfileResult(KernelInfo *info, std::vector<std::pair<int, float>> &profileResult) {
    float tm = profileResult[0].second;
    float t0 = profileResult.back().second;
    gWaveTime[info->name] = tm / 1000.0;
    if(profileResult[0].first != getNrSM()) {
        tm = tm / profileResult[0].first * getNrSM();
    }
    float CI = t0 / tm;
    if(CI >= 0.75) CI = 1;
    gCI[info->name] = CI;
    if(profile_log) {
        printf("[MorphX] kernel %.40s: blocks: %d, occup: %d, waves: %d, CI: %lf, WaveTime: %lfus\n", info->name, info->gridSize, info->occup, (int)profileResult.size(), CI, tm / 1000.0);
        for(int i = 0; i < profileResult.size(); ++i) {
            printf("[MorphX] AvgWave %d: %d, %lfus\n", i, profileResult[i].first, profileResult[i].second / 1000.0);
        }
        fflush(stdout);
    }
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
std::atomic<int> gDebugPrints;

bool shouldDebugPrint() {
    return debug_log && gDebugPrints.fetch_add(1) < debug_limit;
}

void parseTargetKernelPatterns() {
    targetKernelPatterns.clear();
    const char* raw = std::getenv("SMSCHED_TARGET_PATTERNS");
    if(raw == nullptr || raw[0] == '\0') {
        targetKernelPatterns.emplace_back("blocked_floyd_phase");
        targetKernelPatterns.emplace_back("ggnn5query");
        return;
    }
    const char* start = raw;
    for(const char* p = raw; ; ++p) {
        if(*p == ',' || *p == '\0') {
            if(p > start) {
                targetKernelPatterns.emplace_back(start, p - start);
            }
            if(*p == '\0') break;
            start = p + 1;
        }
    }
}

inline void writeSM(KernelInfo *info, int minSM, int maxSM) {
    // if(maxSM >= info->h_maxSM && minSM <= info->h_minSM) 
    //     return;
    info->h_minSM = minSM;
    info->h_maxSM = maxSM;
    gGdrPool->set(info->minSM, pack4(minSM, maxSM));
}

void fillKernelArgs(KernelInfo *info, GdrEntry &agents, GdrEntry &minSM, GdrEntry &maxSM, void **args) {
    int baseArgs = info->numArgs - 6;
    for(int i = 0; i < baseArgs; ++i) {
        args[i] = info->kernelArgs[i];
    }
    args[baseArgs] = &info->logicalGrid;
    args[baseArgs + 1] = &agents.d;
    args[baseArgs + 2] = &info->fetched.d;
    args[baseArgs + 3] = &info->finished.d;
    args[baseArgs + 4] = &minSM.d;
    args[baseArgs + 5] = &maxSM.d;
}

void freeKernelInfo(KernelInfo *info) {
    if(info == nullptr) {
        return;
    }
    if(!info->reclaimAgents.empty() && gReclaimStream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(gReclaimStream)));
    }
    for(void *arg : info->kernelArgs) {
        free(arg);
    }
    for(GdrEntry &entry : info->reclaimAgents) {
        gGdrPool->gdr_free(entry);
    }
    for(GdrEntry &entry : info->reclaimMinSMs) {
        gGdrPool->gdr_free(entry);
    }
    gGdrPool->gdr_free(info->fetched);
    gGdrPool->gdr_free(info->minSM);
    gGdrPool->gdr_free(info->d_agents);
    delete info;
}

void initReclaimStream() {
    if(!reclaim_enabled || gReclaimStream != nullptr) {
        return;
    }
    int leastPriority = 0;
    int greatestPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
    gReclaimStream = reinterpret_cast<CUstream>(stream);
    if(shouldDebugPrint()) {
        fprintf(stderr, "[MorphX-debug] reclaim stream=%p priority=%d least=%d\n",
            gReclaimStream, greatestPriority, leastPriority);
    }
}

bool kernelFinished(KernelInfo *info) {
    uint64_t tmp = gGdrPool->get(info->fetched);
    int finished = ((int*)&tmp)[1];
    return finished >= info->gridSize;
}

bool launchReclaim(KernelInfo *info, int minSM, int maxSM, int nrSM) {
    if(!reclaim_enabled || gReclaimStream == nullptr || info == nullptr) {
        return false;
    }
    if(info->isReclaimed || maxSM <= minSM || minSM < 0 || maxSM > nrSM) {
        return false;
    }
    if(info->h_minSM <= minSM && info->h_maxSM >= maxSM) {
        return false;
    }
    if(kernelFinished(info)) {
        return false;
    }

    GdrEntry agentsEntry = gGdrPool->gdr_malloc();
    GdrEntry minEntry = gGdrPool->gdr_malloc();
    GdrEntry maxEntry = minEntry.half();
    int agents = info->occup * (maxSM - minSM);
    gGdrPool->set(agentsEntry, pack4(agents, 0));
    gGdrPool->set(minEntry, pack4(0, nrSM));

    void **args = new void*[info->numArgs];
    fillKernelArgs(info, agentsEntry, minEntry, maxEntry, args);
    std::shared_lock<std::shared_mutex> readLock(launchMutex);
    CUresult res = real_cuLaunchKernel(
        info->func,
        agents, 1, 1,
        info->blockDimX, info->blockDimY, info->blockDimZ,
        info->sharedMemBytes,
        gReclaimStream,
        args,
        nullptr);
    delete[] args;
    if(res != CUDA_SUCCESS) {
        gGdrPool->gdr_free(agentsEntry);
        gGdrPool->gdr_free(minEntry);
        return false;
    }

    info->reclaimAgents.push_back(agentsEntry);
    info->reclaimMinSMs.push_back(minEntry);
    info->isReclaimed = 1;
    if(shouldDebugPrint()) {
        fprintf(stderr, "[MorphX-debug] reclaim func=%s range=[%d,%d) stream=%p reclaim_stream=%p\n",
            info->name, minSM, maxSM, info->stream, gReclaimStream);
    }
    return true;
}

bool getNonOverlappingRange(KernelInfo *target, int minSM, int maxSM, int *outMinSM, int *outMaxSM) {
    int targetMin = target->h_minSM;
    int targetMax = target->h_maxSM;
    if(targetMin >= targetMax) {
        return false;
    }
    if(targetMin <= minSM && targetMax >= maxSM) {
        return false;
    }
    if(targetMax <= minSM || targetMin >= maxSM) {
        *outMinSM = minSM;
        *outMaxSM = maxSM;
        return true;
    }
    if(targetMin <= minSM) {
        *outMinSM = targetMax;
        *outMaxSM = maxSM;
        return *outMaxSM > *outMinSM;
    }
    if(targetMax >= maxSM) {
        *outMinSM = minSM;
        *outMaxSM = targetMin;
        return *outMaxSM > *outMinSM;
    }

    int leftMin = minSM;
    int leftMax = targetMin;
    int rightMin = targetMax;
    int rightMax = maxSM;
    if(leftMax - leftMin >= rightMax - rightMin) {
        *outMinSM = leftMin;
        *outMaxSM = leftMax;
    } else {
        *outMinSM = rightMin;
        *outMaxSM = rightMax;
    }
    return *outMaxSM > *outMinSM;
}

void profileThread(KernelInfo *launchInfo, std::vector<int> &profileWaves) {
    if(isTargetKernelName(launchInfo->name)) {
        gCI[launchInfo->name] = 1.0;
        gWaveTime[launchInfo->name] = 100.0;
        return;
    }
    const char* name = launchInfo->name;
    std::vector<std::pair<int, float>> profileTimes;
    int fetched, finished;
    int tot = 0;
    for(int i = 0, j = 0; i < profileWaves.size(); i = j) {
        while(j < profileWaves.size() && profileWaves[j] == profileWaves[i]) {
            tot += profileWaves[j];
            j++;
        }
        auto t0 = hrc::now();
        do {
            uint64_t tmp = gGdrPool->get(launchInfo->fetched);
            fetched = ((int*)&tmp)[0];
            finished = ((int*)&tmp)[1];
            // printf("[smsched][profile]: wave %d/%lu, fetched: %d, finished: %d, tot: %d\n", 
            //     i + 1, profileWaves.size(), fetched, finished, tot);
            if(fetched >= tot && j < profileWaves.size()) {
                writeSM(launchInfo, 0, getSMFromWave(profileWaves[j], launchInfo->occup));
            }
        } while(finished < tot);
        auto t1 = hrc::now();
        profileTimes.push_back(std::make_pair(getSMFromWave(profileWaves[i], launchInfo->occup), getNano(t1 - t0) / (j - i)));
    }
    generateProfileResult(launchInfo, profileTimes);
}

bool isTargetKernelName(const char* name) {
    if(name == nullptr) return false;
    for(const std::string& pattern : targetKernelPatterns) {
        if(!pattern.empty() && strstr(name, pattern.c_str())) {
            return true;
        }
    }
    return false;
}

bool isCI(KernelInfo *info) {
    float CI = gCI[info->name];
    if(CI >= 0.9) return true;
    return false;
}

int getStreamRank(CUstream stream) {
    for(int i = 0; i < nActiveStreams.load(); ++i) {
        if(streamInfos[i]->stream == stream) 
            return i;
    }
    return -1;
}

void schedule_1(KernelInfo *kernels[], KernelInfo *nexts[], int nStreams, int nrSM) {
    if(nStreams <= 1) 
        return;
    int flag = 0;
    for(int j = 0; j < nStreams; ++j) {
        if(kernels[j] && isCI(kernels[j])) {
            flag = 1;
            break;
        }
        if(nexts[j] && isCI(nexts[j])) {
            flag = 1;
            break;
        }
    }
    if(!flag) {
        return;
    }
    for(int i = 0; i < nStreams; ++i) {
        if(kernels[i] && !isCI(kernels[i]) && !kernels[i]->isScheduled) {
            writeSM(kernels[i], split_hint, nrSM);
            if(shouldDebugPrint()) {
                fprintf(stderr, "[MorphX-debug] schedule_1 kernel=%s ci=0 range=[%d,%d) nStreams=%d\n",
                    kernels[i]->name, split_hint, nrSM, nStreams);
            }
            kernels[i]->isScheduled = 1;
        }
        if(nexts[i] && !isCI(nexts[i]) && !nexts[i]->isScheduled) {
            writeSM(nexts[i], split_hint, nrSM);
            if(shouldDebugPrint()) {
                fprintf(stderr, "[MorphX-debug] schedule_1 next=%s ci=0 range=[%d,%d) nStreams=%d\n",
                    nexts[i]->name, split_hint, nrSM, nStreams);
            }
            nexts[i]->isScheduled = 1;
        }
    }
    // if(!flag) {
    //     int n = nrSM / nStreams;
    //     for(int i = 0; i < nStreams; ++i) {
    //         if(kernels[i] && !kernels[i]->isScheduled) {
    //             int minSM = i * n;
    //             int maxSM = (i == nStreams - 1) ? nrSM : (i + 1) * n;
    //             // printf("[smsched] scheduling kernel %s to use SM [%d, %d)\n", kernels[i]->name, minSM, maxSM);
    //             writeSM(kernels[i], minSM, maxSM);
    //             kernels[i]->isScheduled = 1;
    //         }
    //     }
    // }
}

void schedule(std::vector<KernelInfo*> &actives, std::vector<KernelInfo*> &pendings, int nrSM) {

    if(actives.empty()) 
        return; // Wait for the hardware to choose the next kernel to execute.
    if(actives.size() + pendings.size() <= 1) 
        return; // Only one stream, no need to schedule.
    if(actives.size() >= 2) 
        return; // Either we have achieved optimal configuration by previous scheduling, or it's kernel boundaries.
    KernelInfo *info0 = actives[0], *info1 = nullptr;
    if(info0->h_minSM != 0 || info0->h_maxSM != nrSM) 
        return; // The active kernel has already been limited by SM, wait for it to finish.
    // printf("[smsched] scheduling... gCI = %f, actives[0] = %s\n", gCI[info0->name].load(), info0->name);
    float CI0 = gCI[info0->name];
    //// special strategy
    if (CI0 > 0.9) {
        // printf("[smsched] Adjusting in stream %p\n", info0->stream);
        writeSM(info0, 0, 75);
    }
    return;
    //// end
    for (auto p : pendings) {
        float CI1 = gCI[p->name];
        if(getKernelType(CI0) != getKernelType(CI1)) {
            info1 = p;
        }
    }
    if(info1 == nullptr) 
        return; // No suitable kernel to co-locate.
    if(info1->h_minSM != 0) {
        writeSM(info0, 0, nrSM - info1->h_minSM);
        return;
    }
    if(info1->h_maxSM != nrSM) {
        writeSM(info0, info1->h_maxSM, nrSM);
        return;
    }
    float CI1 = gCI[info1->name];
    if(CI0 > 0.9) CI0 = 3.0;
    if(CI1 > 0.9) CI1 = 3.0;
    int div = (1 - CI0) / (CI1 - CI0) * nrSM;
    assert(div > 0 && div < nrSM);
    writeSM(info0, 0, div);
    writeSM(info1, div, nrSM);
}

void schedule_2(KernelInfo *kernels[], KernelInfo *nexts[], int nStreams, int nrSM) {
    for(int i = 0; i < nStreams; ++i) {
        if(kernels[i] && isCI(kernels[i])) {
            // if(!nexts[i]->isScheduled) {
            //     writeSM(nexts[i], 0, split_hint);
            //     nexts[i]->isScheduled = 1;
            // }
            for(int j = 0; j < nStreams; ++j) {
                if(j != i && kernels[j] && !kernels[j]->isScheduled) {
                    writeSM(kernels[j], split_hint, nrSM);
                    kernels[j]->isScheduled = 1;
                }
                if(j != i && nexts[j] && !nexts[j]->isScheduled) {
                    writeSM(nexts[j], split_hint, nrSM);
                    nexts[j]->isScheduled = 1;
                }
            }
        }

    }
}

void reclaimOnCIFinish(int finishedStream, KernelInfo *finishedKernel, KernelInfo *kernels[], int nStreams, int nrSM) {
    if(!reclaim_enabled || finishedKernel == nullptr || !isCI(finishedKernel)) {
        return;
    }
    int minSM = finishedKernel->h_minSM;
    int maxSM = finishedKernel->h_maxSM;
    if(maxSM <= minSM || minSM < 0 || maxSM > nrSM) {
        return;
    }
    for(int i = 0; i < nStreams; ++i) {
        if(i == finishedStream || kernels[i] == nullptr) {
            continue;
        }
        KernelInfo *target = kernels[i];
        if(isCI(target)) {
            continue;
        }
        auto waveTimeIt = gWaveTime.find(target->name);
        if(waveTimeIt == gWaveTime.end() || waveTimeIt->second <= 100.0f) {
            continue;
        }
        int reclaimMin = 0;
        int reclaimMax = 0;
        if(!getNonOverlappingRange(target, minSM, maxSM, &reclaimMin, &reclaimMax)) {
            continue;
        }
        if(launchReclaim(target, reclaimMin, reclaimMax, nrSM)) {
            return;
        }
    }
}

void agentThread() {
    bind_thread_to_cpu(gCpu.fetch_add(1));
    int nrSM = getNrSM();
    KernelInfo *kernels[MAX_CONCURRENT_STREAMS], *nexts[MAX_CONCURRENT_STREAMS];
    memset(kernels, 0, sizeof(kernels));
    memset(nexts, 0, sizeof(nexts));
    // std::vector<KernelInfo*> actives;
    // std::vector<KernelInfo*> pendings;
    int cnt = 0;
    cudaStream_t tStream;
    CUDA_CHECK(cudaStreamCreate(&tStream));
    while(gAgentRunning.load(std::memory_order_relaxed)) {
        int nStreams = nActiveStreams.load();
        // actives.clear();
        // pendings.clear();
        for(int i = 0; i < nStreams; i++) {
            if(i == 0)
                cnt = (cnt + 1) % 500;
            if(nexts[i] == nullptr) {
                streamInfos[i]->pending->pop(nexts[i]);
            }
            if(kernels[i] == nullptr) {
                kernels[i] = nexts[i];
                nexts[i] = nullptr;
                streamInfos[i]->pending->pop(nexts[i]);
            }
            if(kernels[i] == nullptr) {
                continue;
            }
            uint64_t tmp = gGdrPool->get(kernels[i]->fetched);
            int fetched = ((int*)&tmp)[0];
            int finished = ((int*)&tmp)[1];
            if(cnt == 0) {
                // printf("[smsched] stream %p, fetched: %d, finished: %d, gridSize: %d, kernels[i]: %p, nexts[i]: %p, func: %p\n", 
                //     kernels[i]->stream, fetched, finished, kernels[i]->gridSize, kernels[i], nexts[i], kernels[i]->func);
                // int agents;
                // CUDA_CHECK(cudaMemcpyAsync(&agents, kernels[i]->d_agents, sizeof(int), cudaMemcpyDeviceToHost, tStream));
                // printf("[smsched] stream %p, agents: %d\n", kernels[i]->stream, agents);
            }
            assert(fetched >= finished);
            if(finished >= kernels[i]->gridSize) {
                reclaimOnCIFinish(i, kernels[i], kernels, nStreams, nrSM);
                freeKernelInfo(kernels[i]);
                kernels[i] = nullptr;
                continue;
            }
            // if(fetched == 0) {
            //     pendings.push_back(kernels[i]);
            // } else {
            //     actives.push_back(kernels[i]);
            // }

        }
        // schedule_2(kernels, nexts, nStreams, nrSM);
        schedule_1(kernels, nexts, nStreams, nrSM);
    }
}

StreamInfo* registerStream(CUstream stream) {
    // printf("Register stream %p\n", stream);
    StreamInfo *streamInfo = new StreamInfo();
    streamInfo->stream = stream;
    streamInfo->pending = new boost::lockfree::spsc_queue<KernelInfo*>(65536);
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
    gGdrPool = new GdrPool();
    initReclaimStream();
    agent = std::thread(agentThread);
    // Register graceful shutdown to avoid std::terminate at process exit
    std::atexit(runtimeShutdown);
    isInitialized = true;
}

void fix_SM(CUstream stream, int minSM, int maxSM) {
    gFixSM[stream] = std::make_pair(minSM, maxSM);
}

void suggest_SM(CUstream stream, int minSM, int maxSM) {
    gSuggestSM[stream] = std::make_pair(minSM, maxSM);
}

inline StreamInfo* checkStream(CUstream stream) {
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

int getenv_int(const char* varname, int default_value) {
    const char* val_str = std::getenv(varname);
    if (val_str != nullptr) {
        return std::atoi(val_str);
    }
    return default_value;
}

/////////////////////////////////////////////////////////////////////////
// Below is the implementation of dynamic loader and function hooking
/////////////////////////////////////////////////////////////////////////



#if defined(__cplusplus)
extern "C" {
#endif

// include auto-generated signatures for unmodified functions
// @note signature.c will be auto generated by parse.py
#include "signature.c" 

/**
 * Function to initialize the environment, including
 * * init the cuda driver module via dlopen
 * * init the file system as specified above
 * * init the hashmap for binaries and CUfunction
 * * init commonly used functions like real_cuModuleLoad...
 * 
 * @note this will be called only once when any hooked driver function is called
 */
static void ld_init(void) {
    pthread_once(&mutex_is_initialized, mutex_init);
    // init() is critical section to be protected
    pthread_mutex_lock(&mutex);
    if (shared_lib != NULL) { // then it has been initialized by another
        pthread_mutex_unlock(&mutex);
        return;
    }
    with_profile = getenv_int("SMSCHED_PROFILE", 1);
    profile_log = getenv_int("SMSCHED_PROFILE_LOG", 0);
    split_hint = getenv_int("SMSCHED_SPLIT_HINT", 0);
    if(split_hint == 0) {
        split_hint = 76; // default value
    }
    debug_log = getenv_int("SMSCHED_DEBUG", 0);
    debug_limit = getenv_int("SMSCHED_DEBUG_LIMIT", 200);
    only_whitelist = getenv_int("SMSCHED_ONLY_WHITELIST", 0);
    register_bypass_streams = getenv_int("SMSCHED_REGISTER_BYPASS_STREAMS", 1);
    reclaim_enabled = getenv_int("SMSCHED_RECLAIM", 1);
    parseTargetKernelPatterns();
    gDebugPrints.store(0);
    common_init(); // init common modules
    if(shouldDebugPrint()) {
        fprintf(stderr, "[MorphX-debug] ld_init split=%d profile=%d only_whitelist=%d\n",
            split_hint, with_profile, only_whitelist);
    }
    // load hooked function of Neutrino
    real_cuModuleLoadData      = (cuModuleLoadData_func_t) dlsym(shared_lib, "cuModuleLoadData");
    real_cuModuleLoadDataEx    = (cuModuleLoadDataEx_func_t) dlsym(shared_lib, "cuModuleLoadDataEx");
    real_cuModuleGetFunction   = (cuModuleGetFunction_func_t) dlsym(shared_lib, "cuModuleGetFunction");
    real_cuKernelGetFunction   = (cuKernelGetFunction_func_t) dlsym(shared_lib, "cuKernelGetFunction");
    real_cuLibraryGetKernel    = (cuLibraryGetKernel_func_t) dlsym(shared_lib, "cuLibraryGetKernel");
    real_cuLibraryGetModule    = (cuLibraryGetModule_func_t) dlsym(shared_lib, "cuLibraryGetModule");
    real_cuLibraryLoadData     = (cuLibraryLoadData_func_t) dlsym(shared_lib, "cuLibraryLoadData");
    real_cuLaunchKernel        = (cuLaunchKernel_func_t) dlsym(shared_lib, "cuLaunchKernel");
    real_cuModuleLoad          = (cuModuleLoad_func_t) dlsym(shared_lib, "cuModuleLoad");
    real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_func_t) dlsym(shared_lib, "cuModuleLoadFatBinary");
    real_cuLaunchKernelEx      = (cuLaunchKernelEx_func_t) dlsym(shared_lib, "cuLaunchKernelEx");
    real_cuGetProcAddress      = (cuGetProcAddress_func_t) dlsym(shared_lib, "cuGetProcAddress");
    real_cuGetProcAddress_v2    = (cuGetProcAddress_v2_func_t) dlsym(shared_lib, "cuGetProcAddress_v2");
    init_unmodified(); // init unmodified functions, defined in signature.c
    CHECK_DL(); // checking if any dl error presented
    // initialzie the L2 Flush Memory if benchmark is enabled
    // leaving critical section, unlock
    pthread_mutex_unlock(&mutex);
    return;
}


CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    CUresult result = real_cuModuleLoadData(module, image);
    return result;
}

CUresult cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    CUresult result = real_cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
    return result;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    CUresult ret = real_cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
    return ret;
}

// JAX use this API, but they don't pass in fatbin but cubin, so a wrong API to use...
CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    CUresult result = real_cuModuleLoadFatBinary(module, fatCubin); // call the symbol
    return result;
}

// @todo handle the multiple function with different name problem
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    // fprintf(stderr, "[mod] cuModuleGetFunction called for %s\n", name);
    CUresult result = real_cuModuleGetFunction(hfunc, hmod, name);
    std::string pk_name = std::string(name) + "_pk";
    CUfunction pk_func;
    CUresult pk_result = real_cuModuleGetFunction(&pk_func, hmod, pk_name.c_str());
    gFuncNameMap[*hfunc] = name;
    if(pk_result == CUDA_SUCCESS) {
        gPkFuncMap[*hfunc] = pk_func;
        gFuncNameMap[pk_func] = pk_name;
        // fprintf(stderr, "[smsched] found patched kernel %s for original kernel %s\n", pk_name.c_str(), name);
    } else {
        gPkFuncMap[*hfunc] = nullptr;
    }
    return result;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    CUresult result = real_cuKernelGetFunction(pFunc, kernel);
    auto nameIter = gKernelNameMap.find(kernel);
    if(result == CUDA_SUCCESS && nameIter != gKernelNameMap.end()) {
        gFuncNameMap[*pFunc] = nameIter->second;
    }
    return result;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    if (shared_lib == NULL) {
        ld_init(); 
    }
    CUresult result = real_cuLibraryGetKernel(pKernel, library, name);
    if(result == CUDA_SUCCESS && name != nullptr) {
        gKernelNameMap[*pKernel] = name;
    }
    return result;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    CUresult result = real_cuLibraryGetModule(pMod, library);
    return result;
}

class FairSpinLock {
public:
    FairSpinLock() : next_ticket_(0), owner_ticket_(0) {}

    // 禁止拷贝和移动
    FairSpinLock(const FairSpinLock&) = delete;
    FairSpinLock& operator=(const FairSpinLock&) = delete;

    void lock() {
        const unsigned int my_ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed);
        while (owner_ticket_.load(std::memory_order_acquire) != my_ticket) {
            __builtin_ia32_pause();
        }
    }

    void unlock() {
        unsigned int current_owner = owner_ticket_.load(std::memory_order_relaxed);
        owner_ticket_.store(current_owner + 1, std::memory_order_release);
    }

private:
    std::atomic<unsigned int> next_ticket_;
    std::atomic<unsigned int> owner_ticket_;
};

/**
 * Execution Control, cuLaunchXXX and cuFuncXXX
 * @see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
 * 
 * aims at providing runtime probing support
 */

std::unordered_map<CUstream, bool> hasMI;

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, 
    CUstream hStream, void** kernelParams, void** extra) 
{
    if (shared_lib == NULL) { 
        ld_init(); 
    }

    int nrSM = getNrSM();
    int gridSize = gridDimX * gridDimY * gridDimZ;

    const char* funcName = nullptr;
    auto funcNameIter = gFuncNameMap.find(f);
    if(funcNameIter != gFuncNameMap.end()) {
        funcName = funcNameIter->second.c_str();
    } else {
        CU_CHECK(real_cuFuncGetName(&funcName, f));
        if(funcName != nullptr) {
            gFuncNameMap[f] = funcName;
            funcName = gFuncNameMap[f].c_str();
        }
    }

    int inWhiteList = isTargetKernelName(funcName) ? 1 : 0;
    bool debugThisLaunch = shouldDebugPrint();
    if(debugThisLaunch) {
        fprintf(stderr, "[MorphX-debug] launch func=%s grid=%d block=%u stream=%p split=%d nActive=%d inWhiteList=%d\n",
            funcName, gridSize, blockDimX * blockDimY * blockDimZ, hStream, split_hint, nActiveStreams.load(), inWhiteList);
    }
    if(only_whitelist && !inWhiteList) {
        if(register_bypass_streams && hStream != 0 && streamInfoMap.find(hStream) == streamInfoMap.end()
            && nActiveStreams.load() < MAX_CONCURRENT_STREAMS) {
            checkStream(hStream);
        }
        if(debugThisLaunch) {
            fprintf(stderr, "[MorphX-debug] bypass-not-whitelist func=%s\n", funcName);
        }
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
    if (!inWhiteList && gWaveTime.count(funcName) && gWaveTime[funcName] < 50.0) {
        if(debugThisLaunch) {
            fprintf(stderr, "[MorphX-debug] bypass-short func=%s waveTime=%.3f\n", funcName, gWaveTime[funcName]);
        }
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
    if(strstr(funcName, "Decode")) {
        hasMI[hStream] = true;
    }
    if(inWhiteList && !gCI.count(funcName)) {
        gCI[funcName] = 1.0;
        gWaveTime[funcName] = 100.0;
    }

    CUfunction newF = gPkFuncMap[f];
    if(newF == nullptr) {
        // fprintf(stderr, "[smsched] cannot find kernel %s, use original kernel\n", newFuncName.c_str());
        if(debugThisLaunch) {
            fprintf(stderr, "[MorphX-debug] bypass-unpatched func=%s\n", funcName);
        }
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    } else {
        // printf("[smsched] using patched kernel %s\n", newFuncName.c_str());
        f = newF;
    }
    StreamInfo *streamInfo = checkStream(hStream);
    sharedMemBytes = std::max(sharedMemBytes, 4u);
    CU_CHECK(real_cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));
    int blockSize = blockDimX * blockDimY * blockDimZ;
    KernelInfo *launchInfo = new KernelInfo;
    launchInfo->func = f;
    launchInfo->stream = hStream;
    launchInfo->name = funcName;
    launchInfo->isScheduled = 0;
    launchInfo->isReclaimed = 0;
    launchInfo->blockDimX = blockDimX;
    launchInfo->blockDimY = blockDimY;
    launchInfo->blockDimZ = blockDimZ;
    launchInfo->sharedMemBytes = sharedMemBytes;
    launchInfo->logicalGrid = dim3(gridDimX, gridDimY, gridDimZ);
    CU_CHECK(real_cuOccupancyMaxActiveBlocksPerMultiprocessor(&launchInfo->occup, f, blockSize, sharedMemBytes));
    assert(launchInfo->occup > 0);
    std::vector<size_t> argSizes;
    int i = 0;
    size_t offset, size;
    while(real_cuFuncGetParamInfo(f, i, &offset, &size) == CUDA_SUCCESS) {
        argSizes.push_back(size);
        i++;
    } 
    int numArgs = argSizes.size();
    assert(numArgs >= 6);
    launchInfo->numArgs = numArgs;
    bool keepKernelArgs = reclaim_enabled;
    if(keepKernelArgs) {
        launchInfo->kernelArgs.resize(numArgs - 6);
        for(int i = 0; i < numArgs - 6; ++i) {
            launchInfo->kernelArgs[i] = malloc(argSizes[i]);
            memcpy(launchInfo->kernelArgs[i], kernelParams[i], argSizes[i]);
        }
    }

    launchInfo->gridSize = gridSize;
    launchInfo->d_agents = gGdrPool->gdr_malloc();
    launchInfo->fetched = gGdrPool->gdr_malloc();
    launchInfo->finished = launchInfo->fetched.half();
    launchInfo->minSM = gGdrPool->gdr_malloc();
    launchInfo->maxSM = launchInfo->minSM.half();
    gGdrPool->set(launchInfo->fetched, 0);
    int fixMin = -1, fixMax = -1;
    if(gFixSM.count(hStream)) {
        auto p = gFixSM[hStream];
        fixMin = p.first; fixMax = p.second;
    } 
    int agents = launchInfo->occup * nrSM; // NOT min(occup * nrSM, gridSize)!!!!
    gGdrPool->set(launchInfo->d_agents, pack4(agents, 0));
    // CUDA_CHECK(cudaMallocAsync(&launchInfo->d_agents, sizeof(int), hStream));
    // CUDA_CHECK(cudaMemcpyAsync(launchInfo->d_agents, &agents, sizeof(int), cudaMemcpyHostToDevice, hStream));
    void **nargs = new void*[numArgs];
    dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    if(keepKernelArgs) {
        fillKernelArgs(launchInfo, launchInfo->d_agents, launchInfo->minSM, launchInfo->maxSM, nargs);
    } else {
        for(int i = 0; i < numArgs - 6; ++i) {
            nargs[i] = malloc(argSizes[i]);
            memcpy(nargs[i], kernelParams[i], argSizes[i]);
        }
        nargs[numArgs - 6] = &gridDim;
        nargs[numArgs - 5] = &launchInfo->d_agents.d;
        nargs[numArgs - 4] = &launchInfo->fetched.d;
        nargs[numArgs - 3] = &launchInfo->finished.d;
        nargs[numArgs - 2] = &launchInfo->minSM.d;
        nargs[numArgs - 1] = &launchInfo->maxSM.d;
    }
    // printf("Time to launch kernel %s: %lfus\n", launchInfo->name, getNano(hrc::now() - launchInfo->tLaunch) / 1000.0);
    // printf("[smsched] kernel launched: %.40s, func: %p, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMemBytes: %u, stream: %p, numArgs: %d, occup: %d, agents: %d\n", 
    //     launchInfo->name, (void*)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, numArgs, launchInfo->occup, agents);
    // fflush(stdout);
    std::thread* profile = nullptr;
    std::vector<int> profileWaves;
    CUresult res = CUDA_SUCCESS;
    if(with_profile && !gCI.count(launchInfo->name)) {
        std::unique_lock<std::shared_mutex> writeLock(launchMutex);
        generateProfileWaves(nrSM, launchInfo, profileWaves, fixMin, fixMax);
        launchInfo->h_minSM = 0;
        launchInfo->h_maxSM = getSMFromWave(profileWaves[0], launchInfo->occup);
        gGdrPool->set(launchInfo->minSM, pack4(0, launchInfo->h_maxSM));
        if(debugThisLaunch) {
            fprintf(stderr, "[MorphX-debug] profile func=%s range=[%d,%d) waves=%zu inWhiteList=%d\n",
                launchInfo->name, launchInfo->h_minSM, launchInfo->h_maxSM, profileWaves.size(), inWhiteList);
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        profile = new std::thread(profileThread, launchInfo, std::ref(profileWaves));
        res = real_cuLaunchKernel(f, agents, 1, 1, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, nargs, extra);
        profile->join();
        delete profile;
        freeKernelInfo(launchInfo);
        launchInfo = nullptr;
    } else {
        std::shared_lock<std::shared_mutex> readLock(launchMutex);
        // if(getStreamRank(hStream) <= 0) {
        //     writeSM(launchInfo, 0, 54);
        //     launchInfo->isScheduled = 1;
        // } else {
        //     writeSM(launchInfo, 54, nrSM);
        //     launchInfo->isScheduled = 1;
        // }
        if(inWhiteList && nActiveStreams.load() >= 2) {
            writeSM(launchInfo, 0, split_hint);
            launchInfo->isScheduled = 1;
        } else {
            writeSM(launchInfo, 0, nrSM);
        }
        if(debugThisLaunch) {
            fprintf(stderr, "[MorphX-debug] assign func=%s range=[%d,%d) inWhiteList=%d nActive=%d isScheduled=%d\n",
                launchInfo->name, launchInfo->h_minSM, launchInfo->h_maxSM, inWhiteList, nActiveStreams.load(), launchInfo->isScheduled);
        }
        // printf("[smsched] kernel launched: %.40s, func: %p, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMemBytes: %u, stream: %p, numArgs: %d, occup: %d, agents: %d\n", 
        //     launchInfo->name, (void*)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, numArgs, launchInfo->occup, agents);
        // if(!inWhiteList)
        res = real_cuLaunchKernel(f, agents, 1, 1, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, nargs, extra);
        streamInfo->pending->push(launchInfo);
        // if(strstr(launchInfo->name, "blocked_floyd_phase")) {
        //     // printf("[smsched] scheduling kernel %s to use max SM\n", launchInfo->name);
        //     writeSM(launchInfo, 32, nrSM);
        //     launchInfo->isScheduled = 1;
        // }
        // if(strstr(launchInfo->name, "ggnn5query")) {
        //     // printf("[smsched] scheduling kernel %s to use max SM\n", launchInfo->name);
        //     // fflush(stdout);
        //     writeSM(launchInfo, 32, nrSM);
        //     launchInfo->isScheduled = 1;
        // }

        // if(nActiveStreams.load() >= 2 && isCI(launchInfo) && !hasMI[hStream]) {
        //     writeSM(launchInfo, 0, split_hint);
        //     launchInfo->isScheduled = 1;
        // }

        // if(isCI(launchInfo)) {
        //     writeSM(launchInfo, 32, nrSM);
        //     launchInfo->isScheduled = 1;
        // }
        // if(strstr(launchInfo->name, "redLoadIteratorMixedINS1S_1")) {
        //     writeSM(launchInfo, 0, split_hint);
        //     launchInfo->isScheduled = 1;
        // }
        // if(strstr(launchInfo->name, "NSL_IfLi4ELb1EEESF_EENS1R_25Til")) {
        //     writeSM(launchInfo, 76, nrSM);
        // }
    }

    if(!keepKernelArgs) {
        for(int i = 0; i < numArgs - 6; ++i) {
            free(nargs[i]);
        }
    }
    delete[] nargs;
    // fprintf(stderr, "[smsched] kernel finished\n");
    CU_CHECK(res);
    return res;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }

    if(config == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuLaunchKernel(
        f,
        config->gridDimX,
        config->gridDimY,
        config->gridDimZ,
        config->blockDimX,
        config->blockDimY,
        config->blockDimZ,
        config->sharedMemBytes,
        config->hStream,
        kernelParams,
        extra);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra) {
    return cuLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    return cuLaunchKernelEx(config, f, kernelParams, extra);
}

/**
 * Following functions shall also be hooked but we don't observe any workload
 * calling them, thus having a [info] section for tracing, add if needed
 */
CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    
    CUresult result = real_cuModuleLoad(module, fname); // call the symbol
    return result;
}

#define cuLaunchKernel_ptsz cuLaunchKernel_ptsz_unmodified
#define cuLaunchKernelEx_ptsz cuLaunchKernelEx_ptsz_unmodified
#include "unmodified.c" // include the auto-generated code
#undef cuLaunchKernel_ptsz
#undef cuLaunchKernelEx_ptsz

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    std::string sym_str(symbol);
    if(debug_log && (
        sym_str.find("Launch") != std::string::npos
        || sym_str.find("Module") != std::string::npos
        || sym_str.find("Library") != std::string::npos
        || sym_str.find("Kernel") != std::string::npos
    ) && shouldDebugPrint()) {
        fprintf(stderr, "[MorphX-debug] cuGetProcAddress_v2 symbol=%s version=%d flags=%llu\n",
            symbol, cudaVersion, (unsigned long long)flags);
    }
    if (sym_str == "cuGetProcAddress") {
        *pfn = (void*)cuGetProcAddress_v2;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel") {
        *pfn = (void*)cuLaunchKernel;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel_ptsz") {
        *pfn = (void*)cuLaunchKernel_ptsz;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadData") {
        *pfn = (void*)cuModuleLoadData;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryLoadData") {
        *pfn = (void*)cuLibraryLoadData;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadDataEx") {
        *pfn = (void*)cuModuleLoadDataEx;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleGetFunction") {
        *pfn = (void*)cuModuleGetFunction;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuKernelGetFunction") {
        *pfn = (void*)cuKernelGetFunction;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetKernel") {
        *pfn = (void*)cuLibraryGetKernel;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetModule") {
        *pfn = (void*)cuLibraryGetModule;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadFatBinary") {
        *pfn = (void*)cuModuleLoadFatBinary;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx") {
        *pfn = (void*)cuLaunchKernelEx;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx_ptsz") {
        *pfn = (void*)cuLaunchKernelEx_ptsz;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    }
    CUresult result = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    return result;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    if (shared_lib == NULL) { 
        ld_init(); 
    }
    std::string sym_str(symbol);
    if(debug_log && (
        sym_str.find("Launch") != std::string::npos
        || sym_str.find("Module") != std::string::npos
        || sym_str.find("Library") != std::string::npos
        || sym_str.find("Kernel") != std::string::npos
    ) && shouldDebugPrint()) {
        fprintf(stderr, "[MorphX-debug] cuGetProcAddress symbol=%s version=%d flags=%llu\n",
            symbol, cudaVersion, (unsigned long long)flags);
    }
    if (sym_str == "cuGetProcAddress") {
        *pfn = (void*)cuGetProcAddress;
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel") {
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    } else if (sym_str == "cuLaunchKernel_ptsz") {
        *pfn = (void*)cuLaunchKernel_ptsz;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadData") {
        *pfn = (void*)cuModuleLoadData;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryLoadData") {
        *pfn = (void*)cuLibraryLoadData;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadDataEx") {
        *pfn = (void*)cuModuleLoadDataEx;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleGetFunction") {
        *pfn = (void*)cuModuleGetFunction;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuKernelGetFunction") {
        *pfn = (void*)cuKernelGetFunction;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetKernel") {
        *pfn = (void*)cuLibraryGetKernel;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLibraryGetModule") {
        *pfn = (void*)cuLibraryGetModule;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuModuleLoadFatBinary") {
        *pfn = (void*)cuModuleLoadFatBinary;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx") {
        *pfn = (void*)cuLaunchKernelEx;
        return CUDA_SUCCESS;
    } else if(sym_str == "cuLaunchKernelEx_ptsz") {
        *pfn = (void*)cuLaunchKernelEx_ptsz;
        return CUDA_SUCCESS;
    }
    CUresult result = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
    return result;
}

#ifdef __cplusplus
}
#endif
