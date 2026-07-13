import ctypes
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
smsched_api = ctypes.CDLL(str(REPO_ROOT / 'runtime' / 'build' / 'libapi.so'))
cudart_api = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')

smsched_api.fixSMForStream.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
def fix_sm_for_stream(stream, min_sm, max_sm):
    cuda_stream = stream.cuda_stream
    return smsched_api.fixSMForStream(ctypes.c_void_p(cuda_stream), ctypes.c_int(min_sm), ctypes.c_int(max_sm))

smsched_api.suggestSMForStream.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
def suggest_sm_for_stream(stream, min_sm, max_sm):
    cuda_stream = stream.cuda_stream
    return smsched_api.suggestSMForStream(ctypes.c_void_p(cuda_stream), ctypes.c_int(min_sm), ctypes.c_int(max_sm))

def cudaSetDeviceFlags(flags):
    return cudart_api.cudaSetDeviceFlags(ctypes.c_uint(flags))

def cudaGetDeviceFlags():
    flags = ctypes.c_uint()
    cudart_api.cudaGetDeviceFlags(ctypes.byref(flags))
    return flags.value
