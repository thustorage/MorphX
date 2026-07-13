#include <unistd.h>   // for many thing
#include <stdlib.h>   // for standard library
#include <stdio.h>    // for file dump
#include <time.h>     // for timing
#include <dlfcn.h>    // for loading real shared library
#include <stdint.h>   // for uint64_t defn
#include <stdbool.h>  // for true false
#include <elf.h>      // for ELF Header
#include <sys/wait.h> // for waiting subprocess
#include <sys/stat.h> // for directory
#include <pthread.h>  // for mutex lock
#include <stdatomic.h> // for atomic operations
#include <atomic>

#define PROBE_TYPE_THREAD 0
#define PROBE_TYPE_WARP 1
#define CDIV(a,b) (a + b - 1) / (b)

static FILE* event_log = stderr; // file pointer to event_log:  NEUTRINO_TRACEDIR/MM_DD_HH_MM_SS/event.event_log

/**
 * System Configuration and Setup
 */

static void* shared_lib           = NULL; // handle to real cuda driver
static const char* NEUTRINO_REAL_DRIVER = "/usr/lib/x86_64-linux-gnu/libcuda.so"; // path to real cuda driver, loaded by env_var NEUTRINO_REAL_DRIVER

// simple auto-increasing idx to distinguish kernels of the same name
static int kernel_idx = 0;

// start time for event_logging. Neutrino trace are named as time since start
static struct timespec start;

// verbose setting -> to prevent event_log file too large due to unimportant setting
static int VERBOSE = 0; 

// dynamic setting -> enable it leads to a count kernel launched to detect the dynamic part
static int DYNAMIC = 0;

// helper macro to check dlopen/dlsym error
#define CHECK_DL() do {                    \
    const char *dl_error = dlerror();      \
    if (dl_error) {                        \
        fprintf(stderr, "%s\n", dl_error); \
        exit(EXIT_FAILURE);                \
    }                                      \
} while (0)


/**
 * @note semaphores for thread safety: Neutrino don't envision multi-threading
 *       but upper layer, like PyTorch may use multi-threading for their need
 * There's only a few critical section like init and hashmaps
 */
static pthread_once_t mutex_is_initialized = PTHREAD_ONCE_INIT; // for safe initialization of mutex
static pthread_mutex_t mutex; // initialization is protected by the mutex_is_initialized
void mutex_init(void) { pthread_mutex_init(&mutex, NULL); }

/**
 * initialize event_log, dir, envvar, these kind of platform-diagnostic commons
 * need to be called at the beginning of platform-specific init()
 * @note shall be executed with mutex protection!!!
 */
static void common_init(void) {
    // if(NEUTRINO_REAL_DRIVER == NULL) {
    //     NEUTRINO_REAL_DRIVER = getenv("NEUTRINO_REAL_DRIVER");
    // }
    if (NEUTRINO_REAL_DRIVER == NULL) {
        fprintf(stderr, "[error] envariable NEUTRINO_REAL_DRIVER not set\n");
        exit(EXIT_FAILURE);
    } 
    char* dynamic = getenv("NEUTRINO_DYNAMIC");
    if (dynamic != NULL && atoi(dynamic) != 0) {
        DYNAMIC = 1;
    }
    char* verbose = getenv("NEUTRINO_VERBOSE");
    if (verbose != NULL && atoi(verbose) != 0) { // otherwise, default is 0
        VERBOSE = 1;
    } 
    shared_lib = dlopen(NEUTRINO_REAL_DRIVER, RTLD_LAZY);
    CHECK_DL();
    // Keep normal AE runs quiet; loader diagnostics can be re-enabled locally.
    fflush(event_log);
    // get the starting time
    clock_gettime(CLOCK_REALTIME, &start);
    // don't free RESULT_DIR and KERNEL_DIR, we will use it later
}
