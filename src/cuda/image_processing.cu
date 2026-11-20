#include "image_processing.cuh"

namespace tensor_engine::cuda
{

__global__ void bgr_to_grayscale_kernel(const uint8_t* d_in, uint8_t* d_out, int width, int height)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    // Why don't I need something like __shared__ anymore? I can actually implement that?

    // Guess that the row and col is spanning so that I don't need to do the loop just like the matmul
    // Guess: d_in is something like [B G R B G R B G R B G...]
    if (row < height && col < width)
    {
        size_t pixel_idx = row * width + col;
        uint8_t B_in = d_in[pixel_idx * 3 + 0];
        uint8_t G_in = d_in[pixel_idx * 3 + 1];
        uint8_t R_in = d_in[pixel_idx * 3 + 2];

        float luminance_gray = 0.299F * static_cast<float>(R_in) + 0.587F * static_cast<float>(G_in) + 0.114F * static_cast<float>(B_in);

        d_out[pixel_idx] = static_cast<uint8_t>(luminance_gray);
    }
}

void launchBgrToGrayscale(const uint8_t* d_in, uint8_t* d_out, int width, int height, cudaStream_t stream)
{
    // This function is taking const Type* and Type*, and the var name is d_, so this is already got assigned to GPU already.
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        static_cast<unsigned int>((width + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((height + block_dim.y - 1) / block_dim.y));

    bgr_to_grayscale_kernel<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, width, height); // Why is this structure? Why another 0, stream as well? Guess: The stream which is cudaStream_t will divide this to different flows/streams so, define this and I don't need to do the whole cullaMalloc() -> cudaMemcpy() -> cudaDeviceSynchronize() -> cudaFree() anymore?
    // Why no cudaDeviceSynchronize()
}

}