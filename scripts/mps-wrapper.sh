python ./test-flashinfer.py
nvidia-cuda-mps-control -d
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python ./test-flashinfer.py
echo quit | nvidia-cuda-mps-control
