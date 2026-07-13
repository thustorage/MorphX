import ctypes
import os
import sys

def load_library():
    lib_name = "libmycutlass.so"
    
    # Search paths
    # 1. Environment variable
    # 2. Relative to this file (assuming standard build structure)
    # 3. Current working directory
    # 4. System paths (handled by CDLL)
    
    paths = []
    if "MYCUTLASS_LIB_DIR" in os.environ:
        paths.append(os.environ["MYCUTLASS_LIB_DIR"])
        
    # .../microbench/python/mycutlass/binding.py -> .../microbench/build/lib
    paths.append(os.path.join(os.path.dirname(__file__), "../../../build/lib"))
    # .../microbench/python/mycutlass/binding.py -> .../microbench/build
    paths.append(os.path.join(os.path.dirname(__file__), "../../build"))
    paths.append(os.getcwd())

    lib_path = None
    for p in paths:
        candidate = os.path.join(p, lib_name)
        if os.path.exists(candidate):
            lib_path = candidate
            break
    
    if lib_path is None:
        # Try system load
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            raise FileNotFoundError(f"Could not find {lib_name}. Please set MYCUTLASS_LIB_DIR environment variable or ensure it is in the library path.")
            
    return ctypes.CDLL(lib_path)

try:
    _lib = load_library()
except FileNotFoundError as e:
    # Allow import even if lib is missing, but fail on usage
    _lib = None
    _error_msg = str(e)

if _lib:
    # extern "C" void myCutlassHgemm(
    #     int m, int n, int k,
    #     const cutlass::half_t* a, long long int lda,
    #     const cutlass::half_t* b, long long int ldb,
    #     cutlass::half_t* c, long long int ldc,
    #     float alpha, float beta, cudaStream_t stream
    # )
    _lib.myCutlassHgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_longlong,
        ctypes.c_void_p, ctypes.c_longlong,
        ctypes.c_void_p, ctypes.c_longlong,
        ctypes.c_float, ctypes.c_float, ctypes.c_void_p
    ]
    _lib.myCutlassHgemm.restype = None

def myCutlassHgemm(m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, alpha=1.0, beta=0.0, stream=0):
    """
    Raw wrapper for myCutlassHgemm C function.
    
    Args:
        m, n, k: int
        a_ptr, b_ptr, c_ptr: int (pointers to device memory)
        lda, ldb, ldc: int (strides)
        alpha, beta: float
        stream: int (cudaStream_t)
    """
    if _lib is None:
        raise RuntimeError(f"Library not loaded: {_error_msg}")
    
    _lib.myCutlassHgemm(
        m, n, k,
        ctypes.c_void_p(a_ptr), ctypes.c_longlong(lda),
        ctypes.c_void_p(b_ptr), ctypes.c_longlong(ldb),
        ctypes.c_void_p(c_ptr), ctypes.c_longlong(ldc),
        alpha, beta, ctypes.c_void_p(stream)
    )
