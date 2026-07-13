from .binding import myCutlassHgemm

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = ['myCutlassHgemm', 'hgemm']

def hgemm(a, b, c=None, alpha=1.0, beta=0.0, stream=None):
    """
    High-level wrapper for myCutlassHgemm using PyTorch tensors.
    Performs C = alpha * A * B + beta * C.
    
    Args:
        a (torch.Tensor): Input matrix A (M x K), float16, CUDA.
        b (torch.Tensor): Input matrix B (K x N), float16, CUDA.
        c (torch.Tensor, optional): Output matrix C (M x N), float16, CUDA.
        alpha (float): Scalar alpha.
        beta (float): Scalar beta.
        stream (torch.cuda.Stream, optional): CUDA stream.
        
    Returns:
        torch.Tensor: The result matrix C.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this high-level wrapper.")
    
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")
    
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Inputs must be float16")
        
    # A is (M, K)
    # B is (K, N)
    # C is (M, N)
    
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Inputs must be 2D matrices")
        
    m, k = a.shape
    k_b, n = b.shape
    
    if k != k_b:
        raise ValueError(f"Shape mismatch: A({m}, {k}) vs B({k_b}, {n})")
        
    if c is None:
        c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    else:
        if c.shape != (m, n):
            raise ValueError(f"Output shape mismatch: expected ({m}, {n}), got {c.shape}")
        if c.dtype != torch.float16:
            raise ValueError("Output must be float16")
            
    # Check contiguity (Row Major standard layout)
    if a.stride(1) != 1 or b.stride(1) != 1 or c.stride(1) != 1:
        # If not contiguous, we might need to make them contiguous
        # But for performance, we should probably warn or error.
        # Let's error for now to be safe.
        raise ValueError("Inputs must be contiguous in the last dimension (Row Major)")

    # Pointers
    a_ptr = a.data_ptr()
    b_ptr = b.data_ptr()
    c_ptr = c.data_ptr()
    
    # Strides
    # PyTorch is Row Major.
    # A stride(0) is K. stride(1) is 1.
    # We treat A as A^T (Col Major K x M). Leading dimension is K.
    lda = a.stride(0)
    ldb = b.stride(0)
    ldc = c.stride(0)
    
    # Stream
    stream_ptr = 0
    if stream is not None:
        if isinstance(stream, torch.cuda.Stream):
            stream_ptr = stream.cuda_stream
        else:
            stream_ptr = stream
    else:
        stream_ptr = torch.cuda.current_stream().cuda_stream
        
    # Call kernel
    # We want C = A * B (Row Major).
    # We call gemm(n, m, k, b, ldb, a, lda, c, ldc, ...)
    # This computes C^T = B^T * A^T (Col Major) which is C = A * B (Row Major)
    
    myCutlassHgemm(n, m, k, b_ptr, ldb, a_ptr, lda, c_ptr, ldc, alpha, beta, stream_ptr)
    
    return c
