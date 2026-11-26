#include "vision_kernels.cuh"

namespace vision
{

// preprocessing kernel: resize, switch B->G, value range [-1.0F, 1.0F]
__global__ void preprocessing_kernel(const uint8_t* d_in, float* d_out, int in_width, int in_height, int out_width, int out_height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < out_height && col < out_width)
    {
        int out_pixel_idx = row * out_width + col;

        float src_x = fminf(fmaxf((col + 0.5F) * (static_cast<float>(in_width) / out_width) - 0.5F, 0.0F), in_width - 1.0F);
        float src_y = fminf(fmaxf((row + 0.5F) * (static_cast<float>(in_height) / out_height) - 0.5F, 0.0F), in_height - 1.0F);

        // Sample 4 neighbor pixels
        int x0 = static_cast<int>(floorf(src_x));
        int x1 = min(x0 + 1, in_width - 1);
        int y0 = static_cast<int>(floorf(src_y));
        int y1 = min(y0 + 1, in_height - 1);

        // Compute bilinear weights
        float alpha = src_x - static_cast<float>(x0);
        float beta = src_y - static_cast<float>(y0);

        for (int i = 0; i < 3; ++i)
        {
            int switch_B_to_R = 2 - i;
            // Bilinear interpolation with swap B-> R on the fly
            float horizontal_top_blend = (1.0F - alpha) * d_in[(y0 * in_width + x0) * 3 + switch_B_to_R] + alpha * d_in[(y0 * in_width + x1) * 3 + switch_B_to_R];
            float horizontal_bottom_blend = (1.0F - alpha) * d_in[(y1 * in_width + x0) * 3 + switch_B_to_R] + alpha * d_in[(y1 * in_width + x1) * 3 + switch_B_to_R];

            float final_value = (1.0F - beta) * horizontal_top_blend + beta * horizontal_bottom_blend;

            // Normalize value
            d_out[out_pixel_idx * 3 + i] = (final_value / 127.5F) - 1.0F;
        }
    }
}

__global__ void postprocessing_kernel(const float* d_in, uint8_t* d_out, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width)
    {
        int out_pixel_idx = row * width + col;
        for (int i = 0; i < 3; ++i)
        {
            int switch_R_to_B = 2 - i;
            d_out[out_pixel_idx * 3 + switch_R_to_B] = static_cast<uint8_t>(fminf(fmaxf((d_in[out_pixel_idx * 3 + i] + 1.0F) * 127.5F, 0.0F), 255.0F));
        }
    }
}

void launchPreProcessing(const uint8_t* d_in, float* d_out, int in_width, int in_height, int out_width, int out_height, cudaStream_t stream)
{
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        static_cast<unsigned int>((out_width + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((out_height + block_dim.y - 1) / block_dim.y));

    preprocessing_kernel<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, in_width, in_height, out_width, out_height);
}

void launchPostProcessing(const float* d_in, uint8_t* d_out, int width, int height, cudaStream_t stream)
{
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        static_cast<unsigned int>((width + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((height + block_dim.y - 1) / block_dim.y));

    postprocessing_kernel<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, width, height);
}

} // namespace vision