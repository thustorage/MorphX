# mycutlass Python Package

This package provides a Python interface to the `libmycutlass.so` library.

## Installation

1. Ensure `libmycutlass.so` is built.
   ```bash
   cd ../
   mkdir build
   cd build
   cmake ..
   make -j
   ```

2. Install the Python package.
   ```bash
   cd ../python
   pip install -e .
   ```

## Usage

### Raw C Interface

```python
import ctypes
from mycutlass import myCutlassHgemm

# You need to manage pointers and memory yourself
# myCutlassHgemm(m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, alpha, beta, stream)
```

### PyTorch Interface

```python
import torch
from mycutlass import hgemm

# Create CUDA tensors (float16)
m, n, k = 1024, 1024, 1024
a = torch.randn((m, k), dtype=torch.float16, device='cuda')
b = torch.randn((k, n), dtype=torch.float16, device='cuda')

# Run GEMM
c = hgemm(a, b)

# Verify
expected = torch.mm(a, b)
diff = (c - expected).abs().max()
print(f"Max difference: {diff}")
```

## Environment Variables

If `libmycutlass.so` is not found automatically, set `MYCUTLASS_LIB_DIR`:

```bash
export MYCUTLASS_LIB_DIR=/path/to/directory/containing/libmycutlass.so
```
