#include "activations.cuh"

#include <cuda_runtime.h>

namespace tensor_engine::cuda
{

__global__ void relu_kernel(const float* input, float* output, size_t size)
{
    // 1. calculate this thread's global ID index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. Boundary Check (crucial!)
    if (idx < size) {
        // If x > 0, output is x, otherwise 0
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}

std::vector<float> relu(const std::vector<float>& host_input)
{
    // Calculate byte size
    size_t size = host_input.size();
    size_t bytes = size * sizeof(float);

    // Allocate GPU Memory (cudaMalloc)
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy CPU Data to GPU (cudaMemcpyHostToDevice)
    cudaMemcpy(d_input, host_input.data(), bytes, cudaMemcpyHostToDevice);

    // Configure Grid & Launch the Kernel (<<<grid, block>>>)
    int block_size = 256;
    int grid_size = static_cast<int>((size + block_size - 1) / block_size); // Ceiling division!

    relu_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    // Copy result from GPU back to CPU (cudaMemcpyDeviceToHost)
    std::vector<float> host_output(size);
    cudaMemcpy(host_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    // Free GPU Memory (cudaFree)
    cudaFree(d_input);
    cudaFree(d_output);

    return host_output;
}

void launchRelu(const float* input, float* output, size_t size, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = static_cast<int>((size + block_size - 1) / block_size);

    relu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
}

} // namespace tensor_engine::cuda