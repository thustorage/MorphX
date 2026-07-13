import numpy as np
import cupy as cp
import cupy.cuda.cudnn as cudnn
import cupyx
# Set up CuDNN
cudnn.benchmark = True

# Create CUDA streams
stream_conv = cp.cuda.Stream()
stream_bn = cp.cuda.Stream()

# Create a random input tensor
batch_size, channels, height, width = 32, 3, 64, 64
input_tensor = cp.random.rand(batch_size, channels, height, width).astype(np.float32)
# Create convolution descriptor
conv_desc = cudnn.createConvolutionDescriptor()
cudnn.setConvolution2dDescriptor_v4(conv_desc,1,1,1,1,1,1, mode='forward', compute_type=cudnn.CUDNN_DATA_FLOAT)

# Create filter tensor
filter_tensor = cp.random.rand(64, channels, 3, 3).astype(np.float32)

# Create convolution kernel descriptor
conv_kernel_desc = cudnn.create_filter_descriptor(filter_tensor)

# Create output tensor descriptor
output_desc = cudnn.create_tensor_descriptor(input_tensor)

# Allocate memory for output tensor
output_tensor = cp.empty_like(input_tensor)

# Perform convolution in the specified stream
with stream_conv:
    cudnn.convolution_forward(conv_desc, 1.0, input_tensor, conv_kernel_desc, filter_tensor, 0.0, output_tensor)

# Batch normalization parameters
epsilon = 1e-5
exp_avg_factor = 0.1

# Allocate memory for batch normalization parameters
scale = cp.ones((channels,), dtype=np.float32)
bias = cp.zeros((channels,), dtype=np.float32)
running_mean = cp.zeros((channels,), dtype=np.float32)
running_var = cp.ones((channels,), dtype=np.float32)

# Create batch normalization descriptor
bn_mode = cudnn.batch_norm_mode['spatial']
bn_desc = cudnn.create_batch_norm_descriptor(bn_mode)

# Perform batch normalization in the specified stream
with stream_bn:
    cudnn.batch_normalization_forward_training(bn_desc, 1.0, 0.0, input_tensor, output_tensor, scale, bias, exp_avg_factor, running_mean, running_var, epsilon)

# Synchronize the streams to ensure all operations are completed
cp.cuda.Stream.null.synchronize()