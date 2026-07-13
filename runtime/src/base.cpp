#include "common.h"

using cudaLaunchKernelHandler = cudaError_t(
    const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);


extern "C" 
{

void register_name_with_arg_num(char *name, int arg_num) {
    // printf("registering kernel %s %d\n", name, arg_num);
    // func_arg_num[std::string(name)] = arg_num;
}

cudaError_t CUDARTAPI cudaLaunchKernel(
    const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream
) {
    auto _cudaLaunchKernel = (cudaLaunchKernelHandler*)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    if(!func) {
        std::cerr << "Failed to get cudaLaunchKernel function pointer" << std::endl;
        return cudaErrorUnknown;
    }
    sharedMem = std::max(4UL, sharedMem);
    int gridSize = gridDim.x * gridDim.y * gridDim.z;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int occup;
    CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occup, func, blockSize, sharedMem));
    int n = 0;
    size_t size, offset;
    while(cudaFuncGetParamInfo(func, n, &offset, &size) == cudaSuccess) {
        n++;
    }
    auto last_error = cudaGetLastError();
    void** nargs = new void*[n];
    int *fetched, *finished, *minSM, *maxSM;
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)&fetched, sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)&finished, sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)&minSM, sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)&maxSM, sizeof(int), stream));
    int t = 0, nrSM = 108;
    int h_fetched = 0, h_finished = 0, h_minSM = 0, h_maxSM = 0;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(fetched, &t, sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(finished, &t, sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(minSM, &t, sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(maxSM, &nrSM, sizeof(int), cudaMemcpyHostToDevice, stream));
    for(int i = 0; i < n - 5; ++i) {
        nargs[i] = args[i];
    }
    nargs[n - 5] = &gridDim;
    nargs[n - 4] = &fetched;
    nargs[n - 3] = &finished;
    nargs[n - 2] = &minSM;
    nargs[n - 1] = &maxSM;
    // printf("Launch kernel %p, grid: (%d, %d, %d), block: (%d, %d, %d), sharedMem: %lu, stream: %p, numArgs: %d, occup: %d\n", 
    //     func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, n - 5, occup);
    auto ret = _cudaLaunchKernel(func, dim3(occup * 108, 1, 1), blockDim, nargs, sharedMem, stream);
    // cudaStreamSynchronize(stream);
    // cudaMemcpyAsync(&h_fetched, fetched, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(&h_finished, finished, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(&h_minSM, minSM, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(&h_maxSM, maxSM, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // printf("fetched: %d, finished: %d, minSM: %d, maxSM: %d\n", h_fetched, h_finished, h_minSM, h_maxSM);
    return ret;
}

}