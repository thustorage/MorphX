import tvm
from tvm import te
import numpy as np

# Define GEMV: A * x = y
n = 4096
m = 4096

# Create TVM placeholders for matrices and vectors
A = te.placeholder((m, n), dtype="float32", name="A")
x = te.placeholder((n,), dtype="float32", name="x")
k = te.reduce_axis((0, n), name="k")
y = te.compute((m,), lambda i: te.sum(A[i, k] * x[k], axis=k), name="y")

# Schedule the operation (this is where optimization happens)
s = te.create_schedule(y.op)

# Target: CUDA
tgt = tvm.target.Target(target="cuda", host="llvm")

bx, tx = s[y].split(y.op.axis[0], factor=64)
s[y].bind(bx, te.thread_axis("blockIdx.x"))
s[y].bind(tx, te.thread_axis("threadIdx.x"))

# Compile the operation to CUDA
with tvm.transform.PassContext(opt_level=3):
    f = tvm.build(s, [A, x, y], target=tgt)

dev = tvm.device(tgt.kind.name, 0)
dev_module = f.imported_modules[0]

# View the generated CUDA kernel
source_file = open("t.cu", "w")
source_file.write(dev_module.get_source())