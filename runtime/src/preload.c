#define _GNU_SOURCE
#include <dlfcn.h>     // for dynamic library
#include <stdio.h>     // for I/O
#include <string.h>    // for strcmp
#include <execinfo.h>  // for backtrace and backtrace_symbols
#include <stdlib.h>    // for malloc and free
#include <time.h>      // for timing terms
#include <assert.h>

#define CHECK_DL() do {                    \
    const char *dl_error = dlerror();      \
    if (dl_error) {                        \
        fprintf(stderr, "%s\n", dl_error); \
        exit(EXIT_FAILURE);                \
    }                                      \
} while (0)

#ifndef STACK_TRACE_SIZE
#define STACK_TRACE_SIZE 5
#endif

#ifndef DL_VERBOSE
#define DL_VERBOSE 0
#endif

// Pointer to GLIBC dlopen function, by dlsym(RTLD_NEXT, "dlopen")
static void* (*real_dlopen)(const char *filename, int flags) = NULL;

static char* NEUTRINO_REAL_DRIVER = "/usr/lib/x86_64-linux-gnu/libcuda.so";
static char* NEUTRINO_HOOK_DRIVER = "/home/rtx/gpu/sm-sched/runtime/build/libcuda.so";
static char* NEUTRINO_DRIVER_NAME = "libcuda.so.1";

void* dlopen(const char *filename, int flags) {
    if (!real_dlopen) 
        real_dlopen = dlsym(RTLD_NEXT, "dlopen");

    char* hook_driver_env = getenv("NEUTRINO_HOOK_DRIVER");
    if (hook_driver_env != NULL && hook_driver_env[0] != '\0') {
        NEUTRINO_HOOK_DRIVER = hook_driver_env;
    }
    char* real_driver_env = getenv("NEUTRINO_REAL_DRIVER");
    if (real_driver_env != NULL && real_driver_env[0] != '\0') {
        NEUTRINO_REAL_DRIVER = real_driver_env;
    }
    char* driver_name_env = getenv("NEUTRINO_DRIVER_NAME");
    if (driver_name_env != NULL && driver_name_env[0] != '\0') {
        NEUTRINO_DRIVER_NAME = driver_name_env;
    }
    
    if (!NEUTRINO_DRIVER_NAME) {
        NEUTRINO_DRIVER_NAME = getenv("NEUTRINO_DRIVER_NAME");
        // fprintf(stderr, "[info] NEUTRINO_DRIVER_NAME: %s\n", NEUTRINO_DRIVER_NAME);
    }   

    // if (filename != NULL && (strstr(filename, "libcuda.so") != NULL)) {
    //     printf("[dlopen] Intercepted loading of %s\n", filename);
    //     fflush(stdout);
    // }

    int is_real_driver_path = (
        filename != NULL
        && NEUTRINO_REAL_DRIVER != NULL
        && strcmp(filename, NEUTRINO_REAL_DRIVER) == 0
    );
    if (filename != NULL && !is_real_driver_path && (strstr(filename, NEUTRINO_DRIVER_NAME) != NULL || strstr(filename, "libcuda.so") != NULL)) {
        
        // Keep AE logs focused on workload output; enable DL_VERBOSE for loader diagnostics.

        void* array[STACK_TRACE_SIZE];
        int size       = backtrace(array, STACK_TRACE_SIZE);
        char** strings = backtrace_symbols(array, size);
        int call_from_cublas = 0;
        if (strings != NULL){
            for (int i = 0; i < size; i++) {
                // we will add ALL Nvidia Propietray Product here
                if (strstr(strings[i], "libcublas") != NULL) {
                    call_from_cublas = 1;
                    break;
                }
            }
        }
        free(strings);
        char* force_hook_cublas = getenv("SMSCHED_FORCE_HOOK_CUBLAS");
        if (force_hook_cublas != NULL && atoi(force_hook_cublas) != 0) {
            call_from_cublas = 0;
        }
        void* ptr;
        if (call_from_cublas) {
            if (NEUTRINO_REAL_DRIVER == NULL) {
                NEUTRINO_REAL_DRIVER = getenv("NEUTRINO_REAL_DRIVER");
                if (NEUTRINO_REAL_DRIVER == NULL) { // fault
                    fprintf(stderr, "[error] NEUTRINO_REAL_DRIVER not set\n");
                    exit(1);
                }
            }
            ptr = real_dlopen(NEUTRINO_REAL_DRIVER, flags);
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            long long time = ts.tv_nsec + ts.tv_sec * 1e9;
            // printf("[info] %lld cublas use real: %s %p %d\n", time, NEUTRINO_REAL_DRIVER, ptr, flags);
            fflush(stdout);
        } else {
            if (NEUTRINO_HOOK_DRIVER == NULL) {
                NEUTRINO_HOOK_DRIVER = getenv("NEUTRINO_HOOK_DRIVER");
            }
            if (NEUTRINO_HOOK_DRIVER == NULL) {
                fprintf(stderr, "[error] NEUTRINO_HOOK_DRIVER not set\n");
                ptr = real_dlopen(filename, flags); // try to backup
            }
            // @note fix the multiple initialization bug
            ptr = real_dlopen(NEUTRINO_HOOK_DRIVER, flags | RTLD_GLOBAL);
            CHECK_DL();

            // fprintf(stderr, "[dlopen] %s : %d, %p\n", NEUTRINO_HOOK_DRIVER, flags | RTLD_GLOBAL, ptr);
            if (DL_VERBOSE) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                long long time = ts.tv_nsec + ts.tv_sec * 1e9;
                printf("[info] %lld use hooked: %s %p %d\n", time, NEUTRINO_HOOK_DRIVER, ptr, flags);
                fflush(stdout);
            }
        }
        return ptr;
    } else { // not interested, just let them go via loading the correct
        // Call the original dlopen
        void* ptr = real_dlopen(filename, flags);
        // Print the name of the module being loaded
        if (DL_VERBOSE) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            long long time = ts.tv_nsec + ts.tv_sec * 1e9;
            printf("[info] %lld Loading: %s %p %d\n", time, filename, ptr, flags);
            fflush(stdout);
        }
        return ptr;
    }
}
