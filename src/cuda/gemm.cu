#include "gemm.cuh"
#include <stdexcept>

#include <cuda_runtime.h>

using namespace tensor_engine::cuda;

__global__ void gemm_naive_matmul_kernel(const float* a, const float* b, float* c, size_t m, size_t k, size_t n)
{
    // Calculate row and col
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n)
    {
        float sum = 0.0F;
        for (size_t i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }

}

Matrix tensor_engine::cuda::cudaMatmul(const Matrix& mat_a, const Matrix& mat_b)
{
    if (mat_a.cols() != mat_b.rows())
    {
        throw std::invalid_argument("Incompatible matrix dimension for multiplication.");
    }
    Matrix mat_c(mat_a.rows(), mat_b.cols());
    
    // Calculate byte size
    size_t size_mat_a = mat_a.rows() * mat_a.cols();
    size_t bytes_mat_a = size_mat_a * sizeof(float);

    size_t size_mat_b = mat_b.rows() * mat_b.cols();
    size_t bytes_mat_b = size_mat_b * sizeof(float);

    size_t size_mat_c = mat_c.rows() * mat_c.cols();
    size_t bytes_mat_c = size_mat_c * sizeof(float);

    // Allocate GPU memory
    float* d_mat_a = nullptr;
    float* d_mat_b = nullptr;
    float* d_mat_c = nullptr;
    cudaMalloc(&d_mat_a, bytes_mat_a);
    cudaMalloc(&d_mat_b, bytes_mat_b);
    cudaMalloc(&d_mat_c, bytes_mat_c);

    // Copy CPU data to GPU
    cudaMemcpy(d_mat_a, mat_a.data(), bytes_mat_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_b, mat_b.data(), bytes_mat_b, cudaMemcpyHostToDevice);

    // Config Grid and launch the kernel
    dim3 block_dim(16, 16); // 16x16=256 threads in each block
    dim3 grid_dim((mat_b.cols() + block_dim.x - 1) / block_dim.x, (mat_a.rows() + block_dim.y - 1) / block_dim.y);

    gemm_naive_matmul_kernel<<<grid_dim, block_dim>>>(d_mat_a, d_mat_b, d_mat_c, mat_a.rows(), mat_a.cols(), mat_b.cols());
    cudaDeviceSynchronize(); // Wait for GPU to finish

    // Copy result from GPU back CPU
    cudaMemcpy(mat_c.data(), d_mat_c, bytes_mat_c, cudaMemcpyDeviceToHost);

    // Free GPU Memory (cudaFree)
    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
    cudaFree(d_mat_c);

    return mat_c;
}