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
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>

// #define PROFILE_ITERATION
// #define NO_QUEUEING

using hrc = std::chrono::high_resolution_clock;
using NanoSec = std::chrono::nanoseconds::rep;
template <typename Duration> 
inline NanoSec getNano(const Duration &d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

std::unordered_map<std::string, int> func_arg_num;

GdrPool* gGdrPool;

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
cudaLaunchKernelHandler *_cudaLaunchKernel = nullptr;
cudaMemcpyAsyncHandler *_cudaMemcpyAsync = nullptr;
cudaMemsetAsyncHandler *_cudaMemsetAsync = nullptr;
cudaMallocAsyncHandler *_cudaMallocAsync = nullptr;
cudaEventRecordHandler *_cudaEventRecord = nullptr;
cudaEventSynchronizeHandler *_cudaEventSynchronize = nullptr;
cudaStreamSynchronizeHandler *_cudaStreamSynchronize = nullptr;
cudaStreamQueryHandler *_cudaStreamQuery = nullptr;
cudaDeviceSynchronizeHandler *_cudaDeviceSynchronize = nullptr;

__attribute__((constructor))
void init() {
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

struct KernelCacheInfo {
    std::vector<int> argSizes;
    int occup;
};

std::unordered_map<const char*, KernelCacheInfo> gKernelCache;

struct MemcpyInfo {
    void *dst;
    const void *src;
    size_t count;
    cudaMemcpyKind kind;
    cudaStream_t stream;
};

struct MemsetInfo {
    void *dst;
    int value;
    size_t count;
    cudaStream_t stream;
};

struct MallocInfo {
    void **devPtr;
    size_t size;
    cudaStream_t stream;
};

struct EventRecordInfo {
    cudaEvent_t event;
    cudaStream_t stream;
};

std::unordered_map<cudaEvent_t, std::atomic<int>> eventInfoMap;

struct KernelRuntimeInfo {
    const char* name;
    int gridSize = 0;
    int occup = 0;
    GdrEntry fetched;
    GdrEntry finished;
    GdrEntry minSM;
    GdrEntry maxSM;
    hrc::time_point tIssue;
};
#define NON_KERNEL_MAGIC ((KernelRuntimeInfo*)0xDEADBEEF)

struct KernelInfo {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
    const char* name;
    int numArgs;
    int occup;
    KernelRuntimeInfo *runtimeInfo;
    hrc::time_point tLaunch;
};

struct StreamInfo;

struct TaskInfo {
    enum TaskType {
        KERNEL,
        MEMCPY,
        MEMSET,
        MALLOC, 
        EVENT_RECORD
    } type;
    void *info;
    StreamInfo *streamInfo;
};

struct StreamInfo {
    cudaStream_t stream;
    boost::lockfree::spsc_queue<TaskInfo> *pending;
    std::atomic<int> nPending = 0;
    boost::lockfree::spsc_queue<KernelRuntimeInfo*> *queueing; 
    KernelRuntimeInfo* running;
    KernelRuntimeInfo* last;
    int hMinSM;
    int hMaxSM;
    bool started;
    hrc::time_point tStart;
};

struct CommInfo {
    struct Modification {
        GdrEntry entry;
        int value;
        int doFree;
    };
    char* buffer[2];
    std::atomic<char*> readBuffer;
    boost::lockfree::queue<std::vector<Modification>*> *ops;
}; 

boost::lockfree::spsc_queue<TaskInfo> *taskQueue;

std::unordered_map<cudaStream_t, StreamInfo*> streamInfoMap;
StreamInfo *streamInfos[16];
std::atomic<int> nActiveStreams = 0;
CommInfo *commInfo;
std::thread communicator;
std::thread agent;
std::thread launcher;
std::atomic<int> commCreated = 0;
std::atomic<int> launcherCreated = 0;
std::atomic<int> agentCreated = 0;

int gNrSM;
inline int getNrSM() {
    if(gNrSM == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        gNrSM = prop.multiProcessorCount;
    }
    return gNrSM;
}

std::unordered_map<cudaStream_t, std::pair<int, int>> gFixSM;
std::unordered_map<cudaStream_t, std::pair<int, int>> gSuggestSM;

// std::unordered_map<const char *, std::vector<std::pair<int, float>>> gProfileData; // TODO: thread unsafe 
std::unordered_map<const char *, std::atomic<float>> gCI;
std::unordered_map<const char *, std::atomic<float>> gWaveTime;

std::atomic<int> launchBlock;

void writeSM(KernelRuntimeInfo *runtimeInfo, int minSM, int maxSM, bool emerg = false) {
    if(emerg) launchBlock.store(1);
    std::vector<CommInfo::Modification>* mdfs = new std::vector<CommInfo::Modification>(2);
    (*mdfs)[0] = CommInfo::Modification({runtimeInfo->minSM, minSM, emerg ? -1 : 0});
    (*mdfs)[1] = CommInfo::Modification({runtimeInfo->maxSM, maxSM, emerg ? -1 : 0});
    commInfo->ops->push(mdfs);
    if(emerg) {
        while(launchBlock.load()) {}
    }
}

void reset(KernelRuntimeInfo *runtimeInfo) {
    std::vector<CommInfo::Modification> *mdfs = new std::vector<CommInfo::Modification>(4);
    (*mdfs)[0] = CommInfo::Modification({runtimeInfo->fetched, 0, 1});
    (*mdfs)[1] = CommInfo::Modification({runtimeInfo->finished, 0, 0});
    (*mdfs)[2] = CommInfo::Modification({runtimeInfo->minSM, 0, 1});
    (*mdfs)[3] = CommInfo::Modification({runtimeInfo->maxSM, getNrSM(), 0});
    commInfo->ops->push(mdfs);
}

cudaError_t launchKernel_impl(KernelInfo *launchInfo) {
    int nrSM = getNrSM();
    dim3 gridDim = launchInfo->gridDim;
    dim3 blockDim = launchInfo->blockDim;
    size_t sharedMem = launchInfo->sharedMem;
    KernelRuntimeInfo *runtimeInfo = launchInfo->runtimeInfo;
    int gridSize = runtimeInfo->gridSize;
    int occup = launchInfo->occup;
    int numArgs = launchInfo->numArgs;
    void** args = launchInfo->args;
    args[numArgs - 5] = &gridDim;
    args[numArgs - 4] = &runtimeInfo->fetched.d;
    args[numArgs - 3] = &runtimeInfo->finished.d;
    args[numArgs - 2] = &runtimeInfo->minSM.d;
    args[numArgs - 1] = &runtimeInfo->maxSM.d;
    int agents = occup * nrSM; // NOT min(occup * nrSM, gridSize)!!!!
    // printf("minSM: %d, maxSM: %d\n", gGdrPool->get(runtimeInfo->minSM), gGdrPool->get(runtimeInfo->maxSM));
    // printf("Time to launch kernel %s: %lfus\n", launchInfo->name, getNano(hrc::now() - launchInfo->tLaunch) / 1000.0);
    // printf("[smsched] kernel launched: %s, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMem: %lu, stream: %p, numArgs: %d, occup: %d, agents: %d\n", 
    //     launchInfo->name, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, launchInfo->stream, numArgs, occup, agents);
    // fflush(stdout);
    auto res = _cudaLaunchKernel(launchInfo->func, dim3(agents, 1, 1), blockDim, args, sharedMem, launchInfo->stream);
    CHECK_CUDA_ERROR(res);
    runtimeInfo->tIssue = hrc::now();
    for(int i = 0; i < numArgs - 5; ++i) {
        free(args[i]);
    }
    delete[] args;
    return res;
}

void allocateRuntime(KernelInfo *launchInfo) {
    KernelRuntimeInfo *runtimeInfo = new KernelRuntimeInfo();
    runtimeInfo->fetched = gGdrPool->gdr_malloc(0);
    runtimeInfo->finished = {runtimeInfo->fetched.d + sizeof(int), runtimeInfo->fetched.h + sizeof(int)};
    runtimeInfo->minSM = gGdrPool->gdr_malloc(1);
    runtimeInfo->maxSM = {runtimeInfo->minSM.d + sizeof(int), runtimeInfo->minSM.h + sizeof(int)};
    runtimeInfo->name = launchInfo->name;
    dim3 gridDim = launchInfo->gridDim;
    runtimeInfo->gridSize = gridDim.x * gridDim.y * gridDim.z;
    runtimeInfo->occup = launchInfo->occup;
    launchInfo->runtimeInfo = runtimeInfo;
}

void issueTask(StreamInfo *streamInfo, TaskInfo &taskInfo) {
    if(taskInfo.type == TaskInfo::KERNEL) {
        KernelInfo *launchInfo = (KernelInfo*)taskInfo.info;
        streamInfo->queueing->push(launchInfo->runtimeInfo);
        // printf("[smsched] kernel issued: %p, stream: %p\n", launchInfo->name, launchInfo->stream);
    } else {
        streamInfo->queueing->push(NON_KERNEL_MAGIC);
    }
    taskInfo.streamInfo = streamInfo;
    taskQueue->push(taskInfo);
}

void generateProfileWaves(int nrSM, KernelRuntimeInfo *info, std::vector<int> &profileWaves) {
    float waves = 1.0 * info->gridSize / (nrSM * info->occup);
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

void generateProfileResult(KernelRuntimeInfo *info, std::vector<int> &profileWaves, std::vector<NanoSec> &profileTimes) {
    if(profileWaves.size() != profileTimes.size()) {
        return;
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
    // printf("[smsched] kernel %s: CI: %lf, WaveTime: %lfus\n", info->name, CI, tm / 1000.0);
    // for(int i = 0; i < profileResult.size(); ++i) {
    //     printf("[smsched] AvgWave %d: %d, %lfus\n", i, profileResult[i].first, profileResult[i].second / 1000.0);
    // }
    
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

void communicatorThread() {
    bind_thread_to_cpu(gCpu.fetch_add(1));
    if(gGdrPool == nullptr) {
        gGdrPool = new GdrPool();
    }
    commInfo = new CommInfo();
    commInfo->buffer[0] = (char*)malloc(GDR_BUFFER_SIZE + 4);
    memset(commInfo->buffer[0], 0, GDR_BUFFER_SIZE + 4);
    commInfo->buffer[1] = (char*)malloc(GDR_BUFFER_SIZE);
    for(int i = 0; i < GDR_BUFFER_SIZE / GDR_ENTRY_SIZE * 2; i += 2) {
        ((int*)commInfo->buffer[1])[i] = 0;
        ((int*)commInfo->buffer[1])[i + 1] = getNrSM();
    }
    gGdrPool->set_buffer(1, commInfo->buffer[1]);
    commInfo->readBuffer.store((char*)malloc(GDR_BUFFER_SIZE + 4));
    commInfo->ops = new boost::lockfree::queue<std::vector<CommInfo::Modification>*>(128);
    std::vector<CommInfo::Modification>* op;
    std::vector<GdrEntry> toFree; 
    int cnt = 0;
    int cnt1 = 0;
    auto t0 = hrc::now();
    commCreated.fetch_add(1);
    while(1) {
#if defined(PROFILE_ITERATION)
        auto dur0 = getNano(hrc::now() - t0);
        if(dur0 > 5000) {
            printf("communicator thread: %3.3fus, cnt: %d\n", dur0 / 1000.0, cnt);
        }
        ++cnt;
        t0 = hrc::now();
#endif 
#if !defined(NO_QUEUEING)
        bool modified = false;
        bool toRelease = false;
        toFree.clear();
        while(commInfo->ops->pop(op)) {
            for(auto mdf : *op) {
                int bufferId = gGdrPool->get_bufferId(mdf.entry);
                if(bufferId == 0) {
                    gGdrPool->set(mdf.entry, mdf.value);
                } else {
                    int index  = gGdrPool->get_index(mdf.entry);
                    *((int*)commInfo->buffer[1] + index) = mdf.value;
                    modified = true;
                }
                if(mdf.doFree == 1) {
                    toFree.push_back(mdf.entry);
                } else if(mdf.doFree == -1) {
                    toRelease = true;
                }
            }
            delete op;
        }
        if(modified) {
            gGdrPool->set_buffer(1, commInfo->buffer[1]);
            if(toRelease) launchBlock.store(0);
        }
        for(auto entry : toFree) {
            gGdrPool->gdr_free(entry);
        }
        __sync_fetch_and_add((int*)(commInfo->buffer[0] + GDR_BUFFER_SIZE), 1);
        gGdrPool->get_buffer(0, commInfo->buffer[0]);
        __sync_fetch_and_add((int*)(commInfo->buffer[0] + GDR_BUFFER_SIZE), 1);
        commInfo->buffer[0] = commInfo->readBuffer.exchange(commInfo->buffer[0]);
#endif
    }
}

void launcherThread() {
    bind_thread_to_cpu(gCpu.fetch_add(1));
    TaskInfo taskInfo;
    cudaError_t ret;
    launcherCreated.fetch_add(1);
    while(true) {
        if(!taskQueue->pop(taskInfo)) {
            continue;
        }
        if(taskInfo.type == TaskInfo::KERNEL) {
            KernelInfo *launchInfo = (KernelInfo*)taskInfo.info;
            ret = launchKernel_impl(launchInfo);
            delete launchInfo;
        } else if(taskInfo.type == TaskInfo::MEMCPY) {
            MemcpyInfo *memcpyInfo = (MemcpyInfo*)taskInfo.info;
            ret = _cudaMemcpyAsync(memcpyInfo->dst, memcpyInfo->src, memcpyInfo->count, memcpyInfo->kind, memcpyInfo->stream);
            delete memcpyInfo;
            // printf("cudaMemcpyAsync\n");
        } else if(taskInfo.type == TaskInfo::MEMSET) {
            MemsetInfo *memsetInfo = (MemsetInfo*)taskInfo.info;
            ret = _cudaMemsetAsync(memsetInfo->dst, memsetInfo->value, memsetInfo->count, memsetInfo->stream);
            delete memsetInfo;
            // printf("cudaMemsetAsync\n");
        } else if(taskInfo.type == TaskInfo::MALLOC) {
            MallocInfo *mallocInfo = (MallocInfo*)taskInfo.info;
            ret = _cudaMallocAsync(mallocInfo->devPtr, mallocInfo->size, mallocInfo->stream);
            delete mallocInfo;
            // printf("cudaMallocAsync\n");
        } else if(taskInfo.type == TaskInfo::EVENT_RECORD) {
            EventRecordInfo *eventRecordInfo = (EventRecordInfo*)taskInfo.info;
            ret = _cudaEventRecord(eventRecordInfo->event, eventRecordInfo->stream);
            eventInfoMap[eventRecordInfo->event].fetch_sub(1); 
            delete eventRecordInfo;
            // printf("cudaEventRecord\n");
        } else {
            std::cerr << "Unknown task type" << std::endl;
            exit(EXIT_FAILURE);
        }
        taskInfo.streamInfo->nPending.fetch_sub(1);
        // if(taskInfo.type == TaskInfo::KERNEL) {
        //     KernelInfo *launchInfo = (KernelInfo*)taskInfo.info;
        //     assert(launchInfo->runtimeInfo != nullptr);
        //     taskInfo.streamInfo->queueing->push(((KernelInfo*)taskInfo.info)->runtimeInfo);
        //     delete launchInfo;
        // } else {
        //     taskInfo.streamInfo->queueing->push(NON_KERNEL_MAGIC);
        // }
    }
}

inline bool checkNoRunning(KernelRuntimeInfo *running) {
    return running == nullptr || running == NON_KERNEL_MAGIC;
}

inline bool checkProfile(TaskInfo &taskInfo) {
    // return false;
    if(taskInfo.type != TaskInfo::KERNEL) {
        return false;
    }
    KernelInfo *launchInfo = (KernelInfo*)taskInfo.info;
    int gridSize = launchInfo->gridDim.x * launchInfo->gridDim.y * launchInfo->gridDim.z;
    int occup = launchInfo->occup;
    int nrSM = getNrSM();
    return strstr(launchInfo->name, "flash_fwd_kernel") == nullptr && gCI[launchInfo->name].load() == -1;
}

uint64_t total_it = 0;
uint64_t ideal_it = 0;
uint64_t same_it = 0;
uint64_t t0_it = 0;
uint64_t t1_it = 0;

__attribute__((destructor))
void destroy() {
    // printf("[smsched] total iterations: %lu, ideal: %.6f, same: %.6f, t0: %.6f, t1: %.6f\n", 
    //     total_it, ideal_it * 1.0 / total_it, same_it * 1.0 / total_it, t0_it * 1.0 / total_it, t1_it * 1.0 / total_it);
}

void agentThread() {
    bind_thread_to_cpu(gCpu.fetch_add(1));
    int nrSM = getNrSM();
    int profiling = -1;
    int toProfile = -1;
    std::vector<int> profileWaves(256);
    int profileNextWave = 0;
    int profileFetched = 0;
    bool profilePending = false;
    TaskInfo profileTask;
    std::vector<NanoSec> profileTimes(256);
    hrc::time_point tWaveStart;
    hrc::time_point tIter = hrc::now();
    float avgIterTime = 0;
    int cnt = 0;
    hrc::time_point waveTrackTime[16];
    int trackingWave[16];
    int nFinished[16];
    int tFinishedk = 0;
    int tK = -1;
    for(int i = 0; i < 16; ++i) {
        trackingWave[i] = 0;
        nFinished[i] = 0;
    }
    agentCreated.fetch_add(1);

    while(true) {
#if defined(PROFILE_ITERATION)
        NanoSec iterDelay = getNano(hrc::now() - tIter);
        if(iterDelay > 5000) {
            printf("iter delay: %lfus, cnt: %d\n", iterDelay / 1000.0, cnt);
        }
        avgIterTime += iterDelay / 1000.0;
        ++cnt; 
        // if(cnt >= 1000) {
        //     printf("avg iter time: %lfus\n", avgIterTime / cnt);
        // }
        tIter = hrc::now();
#endif 
        int n = nActiveStreams.load();
        if(n == 0) continue;
        // assert(n == 1);
        int nIdle = 0;
        for(int i = 0; i < n; ++i) {
            if(checkNoRunning(streamInfos[i]->running) 
                && streamInfos[i]->queueing->empty()) {
                ++nIdle;
            }
        }
        if(toProfile != -1 && nIdle == n) {
            int i = toProfile;
            auto si = streamInfos[i];
            si->pending->pop(profileTask);
            assert(profileTask.type == TaskInfo::KERNEL);
            KernelInfo *launchInfo = (KernelInfo*)profileTask.info;
            allocateRuntime(launchInfo);
            KernelRuntimeInfo *ri = launchInfo->runtimeInfo;
            assert(profiling == -1);
            profiling = i;
            toProfile = -1;
            profileTimes.clear();
            profileWaves.clear();
            profileNextWave = 0;
            profileFetched = 0;
            profilePending = false;
            generateProfileWaves(nrSM, ri, profileWaves);
            si->hMinSM = 0;
            si->hMaxSM = profileWaves[0];
            writeSM(ri, si->hMinSM, si->hMaxSM);
            issueTask(si, profileTask);
            while(!si->queueing->pop(si->running)) {}
        }

        if(streamInfos[0] != nullptr && !checkNoRunning(streamInfos[0]->running) && streamInfos[1] != nullptr && !checkNoRunning(streamInfos[1]->running)) {
            auto s0 = streamInfos[0];
            auto s1 = streamInfos[1];
            if(s0->running->name[3] == '7' && s1->running->name[3] == '7') {
                if(s0->running->name == s1->running->name) {
                    ++same_it;
                } else {
                    ++ideal_it;
                }
            }
        }
        if(streamInfos[0] != nullptr && !checkNoRunning(streamInfos[0]->running)) {
            auto s0 = streamInfos[0];
            if(s0->running->name[3] == '7') {
                ++t0_it;
            }
        }
        if(streamInfos[1] != nullptr && !checkNoRunning(streamInfos[1]->running)) {
            auto s1 = streamInfos[1];
            if(s1->running->name[3] == '7') {
                ++t1_it;
            }
        }
        if((streamInfos[0] != nullptr && !checkNoRunning(streamInfos[0]->running)) || (streamInfos[1] != nullptr && !checkNoRunning(streamInfos[1]->running))) {
            ++total_it;
        }
        
        for(int i = 0; i < n; ++i) {
            if(profiling != -1 && i != profiling) {
                continue;
            }
            StreamInfo *si = streamInfos[i];
#if defined(NO_QUEUEING)
            if(si->pending->read_available() != 0) {
                auto taskInfo = si->pending->front();
                if(taskInfo.type == TaskInfo::KERNEL) {
                    KernelInfo *launchInfo = (KernelInfo*)taskInfo.info;
                    allocateRuntime(launchInfo);
                    if(gFixSM.find(launchInfo->stream) != gFixSM.end()) {
                        auto [minSM, maxSM] = gFixSM[launchInfo->stream];
                        gGdrPool->set(launchInfo->runtimeInfo->minSM, minSM);
                        gGdrPool->set(launchInfo->runtimeInfo->maxSM, maxSM);
                    } 
                }
                issueTask(si, taskInfo);
                si->pending->pop();
            }
#endif
            // while(si->pending->read_available() != 0 && si->queueing->read_available() <= 30) {
            //     auto taskInfo = si->pending->front();
            //     if(taskInfo.type == TaskInfo::KERNEL) {
            //         si->pending->pop();
            //         issueTask(si, taskInfo);
            //     }
            // }
            if(checkNoRunning(si->running)) {
#if defined(NO_QUEUEING)
                if(si->queueing->read_available() != 0) {
                    si->queueing->pop(si->running);
                }
                continue;
#endif
                KernelInfo *launchInfo = nullptr;
                int k = -1, cnt = 0;
                StreamInfo *sk = nullptr;
                float CI0, CI1;
                int div;
                if(toProfile != -1) {
                    // Some other stream is waiting for profiling, do not issue
                    continue;
                }
                assert(si->queueing->read_available() == 0);
                if(si->pending->read_available() == 0) {
                    // Not task to issue
                    continue;
                } 
                TaskInfo &taskInfo = si->pending->front();
                if(checkProfile(taskInfo)) {
                    // Mark current stream as waiting for profiling
                    toProfile = i;
                    continue;
                } 
                if(taskInfo.type != TaskInfo::KERNEL) {
                    // Launch non-kernel tasks directly
                    goto ready_to_issue;
                }
                launchInfo = (KernelInfo*)taskInfo.info;
                if(gFixSM.find(launchInfo->stream) != gFixSM.end()) {
                    auto [minSM, maxSM] = gFixSM[launchInfo->stream];
                    si->hMinSM = minSM;
                    si->hMaxSM = maxSM;
                    allocateRuntime(launchInfo);
                    writeSM(launchInfo->runtimeInfo, si->hMinSM, si->hMaxSM, true);
                    goto ready_to_issue; 
                } 
                k = -1, cnt = 0; 
                for(int j = 0; j < n; ++j) if(j != i && !checkNoRunning(streamInfos[j]->running)) {
                    k = j;
                    ++cnt;
                }
                if(cnt == 0) {
                    si->hMinSM = 0;
                    si->hMaxSM = nrSM;
                    allocateRuntime(launchInfo);
                    goto ready_to_issue;
                }
                if(cnt >= 2) {
                    continue;
                }
                sk = streamInfos[k];

                // Fall through to use profiled results
                if(gWaveTime[launchInfo->name].load() <= 30) {
                    // Directly launch kernels that all too small
                    si->hMinSM = 0;
                    si->hMaxSM = nrSM;
                    allocateRuntime(launchInfo);
                    goto ready_to_issue;
                }

                if(gCI[launchInfo->name].load() < 0.5 && gCI[sk->running->name].load() > 0.8 && sk->hMinSM == 0) {
                    sk->hMinSM = 20;
                    sk->hMaxSM = nrSM;
                    writeSM(sk->running, sk->hMinSM, sk->hMaxSM);
                    // printf("[smsched] Adjust %s from %s\n", sk->running->name, launchInfo->name);
                }
                // if(gCI[launchInfo->name].load() < 0.5 && gCI[sk->running->name].load() > 0.8 && sk->hMinSM == 0) {
                //     // block current stream
                //     continue;
                // }
                si->hMinSM = 0;
                si->hMaxSM = nrSM;
                allocateRuntime(launchInfo);
                goto ready_to_issue;

                if(sk->hMinSM != 0 || sk->hMaxSM != nrSM) {
                    // If the other stream has been adjusted, then directly launch to occupy idle SMs
                    si->hMinSM = 0;
                    si->hMaxSM = nrSM;
                    allocateRuntime(launchInfo);
                    goto ready_to_issue;
                }
                if(gSuggestSM.find(launchInfo->stream) != gSuggestSM.end()) {
                    // Use suggested SM number
                    auto [minSM, maxSM] = gSuggestSM[launchInfo->stream];
                    sk->hMinSM = minSM;
                    sk->hMaxSM = maxSM;
                    writeSM(sk->running, sk->hMinSM, sk->hMaxSM);
                    if(minSM == 0) {
                        si->hMinSM = maxSM;
                        si->hMaxSM = nrSM;
                    } else {
                        si->hMinSM = 0;
                        si->hMaxSM = minSM;
                    }
                    allocateRuntime(launchInfo);
                    writeSM(launchInfo->runtimeInfo, si->hMinSM, si->hMaxSM);
                    goto ready_to_issue;
                }


                CI0 = gCI[launchInfo->name].load();
                CI1 = gCI[sk->running->name].load();
                assert(CI0 != -1 && CI1 != -1);
                if((CI0 == 1 && CI1 == 1) || (CI0 < 1 && CI1 < 1)) {
                    // printf("[smsched] Waiting for launch...\n");
                    continue;
                }
                if(CI0 == 1) CI0 = 4;
                if(CI1 == 1) CI1 = 4;
                div = CI0 * (1 - CI1) / (CI0 - CI1) * nrSM;
                assert(div != 0 && div != nrSM);
                allocateRuntime(launchInfo);
                if(k < i) {
                    div = nrSM - div;
                    si->hMinSM = sk->hMaxSM = div;
                    writeSM(launchInfo->runtimeInfo, div, nrSM);
                    writeSM(sk->running, 0, div);
                    // printf("[smsched] Adjust: %.2f : %.2f, %d\n", CI1, CI0, div);
                } else {
                    si->hMaxSM = sk->hMinSM = div;
                    writeSM(launchInfo->runtimeInfo, 0, div);
                    writeSM(sk->running, div, nrSM);
                    // printf("[smsched] Adjust: %.2f : %.2f, %d\n", CI0, CI1, div);
                }
                // if(CI0 == 1) {
                //     trackingWave[i] = 1;
                //     tK = k;
                //     tFinishedk = nFinished[k];
                // }
ready_to_issue:
                si->pending->pop();
                issueTask(si, taskInfo);
                while(!si->queueing->pop(si->running)) {}
                assert(si->running != nullptr);
                if(si->running == NON_KERNEL_MAGIC) {
                    continue;
                }
            }
            assert(!checkNoRunning(si->running));
            int fetched_ind = gGdrPool->get_index(si->running->fetched);
            int finished_ind = gGdrPool->get_index(si->running->finished);
            volatile int fetched, finished;
            char *readBuffer = commInfo->readBuffer.load();
            while(1) {
                volatile int ver0 = *(int*)(readBuffer + GDR_BUFFER_SIZE);
                fetched = ((int*)readBuffer)[fetched_ind];
                finished = ((int*)readBuffer)[finished_ind];
                volatile int ver1 = *(int*)(readBuffer + GDR_BUFFER_SIZE);
                if(ver0 == ver1 && ver0 % 2 == 0) {
                    break;
                }
            }
            // printf("[smsched] kernel running: %s %d/%d/%d\n", si->running->name, fetched, finished, si->running->gridSize);
            // printf("[smsched] cudaStreamQuery: %d\n", (int)_cudaStreamQuery(si->stream));
            if(fetched > 0) {
                if(!si->started) {
                    si->tStart = hrc::now();
                    si->started = 1;
                }
                if(si->last != nullptr) {
                    reset(si->last);
                    delete si->last;
                    si->last = nullptr;
                }
            }
            if(profiling == i) {
                if(!profilePending && profileNextWave < profileWaves.size() 
                    && fetched >= profileFetched + profileWaves[profileNextWave]) {
                    tWaveStart = hrc::now();
                    profileFetched += profileWaves[profileNextWave];
                    profileNextWave++;
                    profilePending = true;
                }
                if(profileNextWave < profileWaves.size() && si->hMaxSM > 
                    getSMFromWave(profileWaves[profileNextWave], si->running->occup)) {
                    si->hMaxSM = getSMFromWave(profileWaves[profileNextWave], si->running->occup);
                    writeSM(si->running, si->hMinSM, si->hMaxSM);
                }
                if(profilePending && finished >= profileFetched) {
                    profileTimes.push_back(getNano(hrc::now() - tWaveStart));
                    profilePending = false;
                }
                if(!(profilePending == false && profileNextWave >= profileWaves.size())) {
                    continue;
                }
            }
            if(trackingWave[i] == 1 && finished >= si->hMaxSM - si->hMinSM) {
                waveTrackTime[i] = hrc::now();
                trackingWave[i] = 2;
            }
            if(trackingWave[i] == 2 && finished >= 2 * (si->hMaxSM - si->hMinSM)) {
                auto tEnd = hrc::now();
                float waveTime = getNano(tEnd - waveTrackTime[i]) / 1000.0;
                float waveTimeFull = gWaveTime[si->running->name].load();
                float CI_ = 1 / (waveTime / waveTimeFull - 1);
                if(nFinished[tK] == tFinishedk) {
                    gCI[si->running->name].store(CI_);
                    // printf("[smsched] Time increase: %lf\n", waveTime / waveTimeFull);
                }
                // printf("[smsched] Time : %lfus, increase: %lf with %lf SMs\n", waveTime, waveTime / waveTimeFull, (si->hMaxSM - si->hMinSM) / (float)nrSM);
                trackingWave[i] = 0;
            }
            if(finished >= si->running->gridSize) {
                auto tEnd = hrc::now();
                float elapsed = getNano(tEnd - si->tStart) / 1000.0;
                // printf("[smsched] kernel finished: %s, %d/%d, stream: %p, elapsed: %lfus\n", si->running->name, finished, si->running->gridSize, si->stream, elapsed);
                // fflush(stdout);
                if(profiling == i) {
                    generateProfileResult(si->running, profileWaves, profileTimes);
                    // for(int i = 0; i < profileTimes.size(); ++i) {
                    //     printf("[smsched] wave %2d:  %3d   %lfus\n", i, profileWaves[i], profileTimes[i] / 1000.0);
                    // }
                    // printf("[smsched] Compute Intensity: %lf, wave time: %lf\n", gCI[si->running->name].load(), gWaveTime[si->running->name].load());
                    profiling = -1;
                }
                nFinished[i]++;
                si->last = si->running;
                si->running = nullptr;
                si->started = false;
            } 
        }
    }
}

void registerStream(cudaStream_t stream) {
    // printf("Register stream %p\n", stream);
    StreamInfo *streamInfo = new StreamInfo();
    streamInfo->stream = stream;
    streamInfo->pending = new boost::lockfree::spsc_queue<TaskInfo>(65536);
    streamInfo->queueing = new boost::lockfree::spsc_queue<KernelRuntimeInfo*>(65536);
    streamInfo->nPending.store(0);
    streamInfo->running = nullptr;
    streamInfo->last = nullptr;
    streamInfo->hMinSM = 0;
    streamInfo->hMaxSM = getNrSM();
    streamInfo->started = false;
    streamInfoMap[stream] = streamInfo;
    if(nActiveStreams.load() >= 16) {
        std::cerr << "Too many streams" << std::endl;
        exit(EXIT_FAILURE);
    }
    streamInfos[nActiveStreams.load()] = streamInfo;
    nActiveStreams.fetch_add(1);
}

extern "C"
{

void register_name_with_arg_num(char *name, int arg_num) {
    // printf("registering kernel %s %d\n", name, arg_num);
    // func_arg_num[std::string(name)] = arg_num;
}

void fix_SM(cudaStream_t stream, int minSM, int maxSM) {
    gFixSM[stream] = std::make_pair(minSM, maxSM);
    printf("fix SM for stream %p: %d %d\n", stream, minSM, maxSM);
}

void suggest_SM(cudaStream_t stream, int minSM, int maxSM) {
    gSuggestSM[stream] = std::make_pair(minSM, maxSM);
    printf("suggest SM for stream %p: %d %d\n", stream, minSM, maxSM);
}

inline bool checkStream(cudaStream_t stream) {
    if(stream == 0) {
        std::cerr << "Default stream is not supported" << std::endl;
        return false;
    }
    if(launcherCreated.fetch_add(1) == 0) {
        taskQueue = new boost::lockfree::spsc_queue<TaskInfo>(65536);
        launcher = std::thread(launcherThread);
        while(launcherCreated.load() < 2) {}
        launcher.detach();
    }
    if(commCreated.fetch_add(1) == 0) {
        communicator = std::thread(communicatorThread);
        while(commCreated.load() < 2) {}
        communicator.detach();
    }
    if(agentCreated.fetch_add(1) == 0) {
        agent = std::thread(agentThread);
        while(agentCreated.load() < 2) {}
        agent.detach();
    } 
    if(streamInfoMap.find(stream) == streamInfoMap.end()) {
        registerStream(stream);
    }
    return true;
}

cudaError_t CUDARTAPI cudaLaunchKernel(
    const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream
) {
    if(!checkStream(stream)) {
        return cudaErrorInvalidValue;
    }

    StreamInfo *streamInfo = streamInfoMap[stream];
    sharedMem = std::max(sharedMem, 4ul);
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    KernelInfo *launchInfo = new KernelInfo({func, gridDim, blockDim, 0, sharedMem, stream, "", 0, 0, nullptr, hrc::now()});
    const char* funcName;
    CHECK_CUDA_ERROR(cudaFuncGetName(&funcName, launchInfo->func));
    launchInfo->name = funcName;
    // printf("Launch kernel to stream %p in thread %ld\n", stream, std::this_thread::get_id());
    // fflush(stdout);
    // if(!func_arg_num.count(std::string(launchInfo->name))) {
    //     std::cerr << "Kernel " << launchInfo->name << " is not registered" << std::endl;
    //     return cudaErrorInvalidValue;
    // }
    KernelCacheInfo cacheInfo;
    if(!gKernelCache.count(launchInfo->name)) {
        int i = 0;
        cacheInfo.argSizes.clear();
        size_t offset, size;
        while(cudaFuncGetParamInfo(func, i, &offset, &size) == cudaSuccess) {
            cacheInfo.argSizes.push_back(size);
            i++;
        }
        auto last_error = cudaGetLastError();
        CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cacheInfo.occup, func, blockSize, sharedMem));
    } else {
        cacheInfo = gKernelCache[launchInfo->name];
    }
    void **nargs = new void*[cacheInfo.argSizes.size()];
    for(int i = 0; i < cacheInfo.argSizes.size() - 5; ++i) {
        nargs[i] = malloc(cacheInfo.argSizes[i]);
        memcpy(nargs[i], args[i], cacheInfo.argSizes[i]);
    }
    launchInfo->args = nargs;
    launchInfo->numArgs = cacheInfo.argSizes.size();
    launchInfo->occup = cacheInfo.occup;
    // Create the elements in user thread, then all other threads do not insert elements
    if(!gCI.count(launchInfo->name)) {
        gCI[launchInfo->name].store(-1);
        gWaveTime[launchInfo->name].store(-1);
    }
    streamInfo->nPending.fetch_add(1);
    streamInfo->pending->push({TaskInfo::KERNEL, (void*)launchInfo, nullptr});
    return cudaSuccess; 
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    if(!checkStream(stream)) {
        return cudaErrorInvalidValue;
    }
    // printf("cudaMemcpyAsync %p %p %zu %d %p\n", dst, src, count, kind, stream);
    StreamInfo *streamInfo = streamInfoMap[stream];
    MemcpyInfo *memcpyInfo = new MemcpyInfo({dst, src, count, kind, stream});
    streamInfo->nPending.fetch_add(1);
    streamInfo->pending->push({TaskInfo::MEMCPY, (void*)memcpyInfo, nullptr});
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream) {
    if(!checkStream(stream)) {
        return cudaErrorInvalidValue;
    }
    // printf("cudaMemsetAsync\n");
    StreamInfo *streamInfo = streamInfoMap[stream];
    MemsetInfo *memsetInfo = new MemsetInfo({dst, value, count, stream});
    streamInfo->nPending.fetch_add(1);
    streamInfo->pending->push({TaskInfo::MEMSET, (void*)memsetInfo, nullptr});
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    if(!checkStream(stream)) {
        return cudaErrorInvalidValue;
    }
    // printf("cudaMallocAsync\n");
    StreamInfo *streamInfo = streamInfoMap[stream];
    MallocInfo *mallocInfo = new MallocInfo({devPtr, size, stream});
    streamInfo->nPending.fetch_add(1);
    streamInfo->pending->push({TaskInfo::MALLOC, (void*)mallocInfo, nullptr});
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    // printf("cudaEventRecord %p\n", event);
    if(!streamInfoMap.count(stream)) {
        return cudaSuccess;
    }
    StreamInfo *streamInfo = streamInfoMap[stream];
    EventRecordInfo *eventRecordInfo = new EventRecordInfo({event, stream});
    if(!eventInfoMap.count(event)) {
        eventInfoMap[event].store(0);
    }
    eventInfoMap[event].fetch_add(1);
    streamInfo->nPending.fetch_add(1);
    streamInfo->pending->push({TaskInfo::EVENT_RECORD, (void*)eventRecordInfo, nullptr});
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    // printf("cudaEventSynchronize %p in thread %ld\n", event, std::this_thread::get_id());
    auto &eventCounter = eventInfoMap[event];
    while(eventCounter.load()) {
        // std::this_thread::yield();
    }
    auto ret = _cudaEventSynchronize(event);
    return ret;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    // printf("cudaStreamSynchronize %p in thread %ld\n", stream, std::this_thread::get_id());
    if(!streamInfoMap.count(stream)) {
        printf("stream %p not found\n", stream);
        return cudaSuccess;
    }
    StreamInfo *streamInfo = streamInfoMap[stream];
    while(streamInfo->nPending.load() > 0) {
        // auto ret = cudaGetLastError();
        // if(ret != cudaSuccess) {
        //     printf("cudaStreamSynchronize error: %s\n", cudaGetErrorString(ret));
        //     return ret;
        // }
        // printf("Synchronizing stream %p, pending: %d\n", stream, streamInfo->nPending.load());
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // std::this_thread::yield();
    }
    auto ret = _cudaStreamSynchronize(stream);
    return ret;
}

cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    if(!streamInfoMap.count(stream)) {
        return cudaSuccess;
    }
    StreamInfo *streamInfo = streamInfoMap[stream];
    if(streamInfo->nPending.load() > 0) {
        return cudaErrorNotReady;
    } else {
        return _cudaStreamQuery(stream);
    }
}

cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    // printf("cudaDeviceSynchronize\n");
    int n = nActiveStreams.load();
    for(int i = 0; i < n; ++i) {
        auto ret = cudaStreamSynchronize(streamInfos[i]->stream);
        if(ret != cudaSuccess) {
            return ret;
        }
    }
    return cudaSuccess;
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
MAKE_CUDA_METHOD(cudaMemPrefetchAsync, (const void* devPtr, size_t count, int dstDevice, cudaStream_t stream), devPtr, count, dstDevice, stream)
MAKE_CUDA_METHOD(cudaMemcpy2DAsync, (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), dst, dpitch, src, spitch, width, height, kind, stream)
MAKE_CUDA_METHOD(cudaMemcpy2DFromArrayAsync, (void *dst, size_t dpitch, cudaArray_const_t srcArray, size_t srcOffsetXInBytes, size_t srcOffsetY, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), dst, dpitch, srcArray, srcOffsetXInBytes, srcOffsetY, width, height, kind, stream)
MAKE_CUDA_METHOD(cudaMemcpy2DToArrayAsync, (cudaArray_t dstArray, size_t dstOffsetXInBytes, size_t dstOffsetY, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), dstArray, dstOffsetXInBytes, dstOffsetY, src, spitch, width, height, kind, stream)
MAKE_CUDA_METHOD(cudaMemcpy3DAsync, (const cudaMemcpy3DParms *p, cudaStream_t stream), p, stream)
MAKE_CUDA_METHOD(cudaMemcpy3DBatchAsync, (const cudaMemcpy3DParms *p, cudaStream_t stream), p, stream)
MAKE_CUDA_METHOD(cudaMemcpy3DPeerAsync, (const cudaMemcpy3DPeerParms *p, cudaStream_t stream), p, stream)
MAKE_CUDA_METHOD(cudaMemcpyBatchAsync, (const cudaMemcpy3DParms *p, cudaStream_t stream), p, stream)
MAKE_CUDA_METHOD(cudaMemcpyFromSymbolAsync, (void *dst, const void *symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), dst, symbol, count, offset, kind, stream)
MAKE_CUDA_METHOD(cudaMemcpyPeerAsync, (void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream), dst, dstDevice, src, srcDevice, count, stream)
MAKE_CUDA_METHOD(cudaMemcpyToSymbolAsync, (const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), symbol, src, count, offset, kind, stream)
MAKE_CUDA_METHOD(cudaMemset2DAsync, (void *dst, size_t dpitch, int value, size_t width, size_t height, cudaStream_t stream), dst, dpitch, value, width, height, stream)
MAKE_CUDA_METHOD(cudaMemset3DAsync, (cudaPitchedPtr dst, int value, cudaExtent extent, cudaStream_t stream), dst, value, extent, stream)
MAKE_CUDA_METHOD(cudaMallocFromPoolAsync, (void **devPtr, size_t size, cudaMemPool_t memPool, cudaStream_t stream), devPtr, size, memPool, stream)
MAKE_CUDA_METHOD(cudaStreamBeginCapture, (cudaStream_t stream, cudaStreamCaptureMode mode), stream, mode)
MAKE_CUDA_METHOD(cudaStreamEndCapture, (cudaStream_t stream, cudaGraph_t *graph), stream, graph)

// MAKE_CUDA_METHOD(cudaMemcpy, (void *dst, const void *src, size_t count, cudaMemcpyKind kind), dst, src, count, kind)
// MAKE_CUDA_METHOD(cudaMemset, (void *dst, int value, size_t count), dst, value, count)
MAKE_CUDA_METHOD(cudaStreamWaitEvent, (cudaStream_t stream, cudaEvent_t event, unsigned int flags), stream, event, flags)

} // extern "C"