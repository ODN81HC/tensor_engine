#include <gtest/gtest.h>
#include <chrono>

#include "activations.cuh"
#include "gemm.cuh"
#include "matrix.hpp"
#include "thread_pool.hpp"
#include "pinned_vector.cuh"
#include "cuda_stream.cuh"

TEST(ActivationsTest, ReLUTest)
{
    std::vector<float> test_input = {-1.0, 5.0, -2.3, 110.7};
    std::vector<float> test_output;
    test_output = tensor_engine::cuda::relu(test_input);
    std::vector<float> result_to_compare = {0.0, 5.0, 0.0, 110.7};
    EXPECT_EQ(test_output, result_to_compare);
}

TEST(GPUGemmTest, CorrectnessCheck)
{
    Matrix test_matrix_a = Matrix::random(10, 15);
    Matrix test_matrix_b = Matrix::random(15, 20);
    Matrix test_matrix_error = Matrix::random(19, 20);

    EXPECT_THROW(tensor_engine::cuda::cudaMatmul(test_matrix_a, test_matrix_error), std::invalid_argument);
    Matrix gpu_matmul = tensor_engine::cuda::cudaMatmul(test_matrix_a, test_matrix_b);
    Matrix cpu_matmul = Matrix::multiplySequential(test_matrix_a, test_matrix_b);

    for (size_t i = 0; i < gpu_matmul.rows(); ++i) {
        for (size_t j = 0; j < gpu_matmul.cols(); ++j) {
            EXPECT_NEAR(gpu_matmul(i, j), cpu_matmul(i, j), 1e-4F);
        }
    }
}

TEST(GPUGemmTest, GPUSpeedBenchmark)
{
    Matrix test_matrix_a = Matrix::random(1000, 1200);
    Matrix test_matrix_b = Matrix::random(1200, 1500);

    ThreadPool test_pool(4);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    Matrix::multiplyParallel(test_matrix_a, test_matrix_b, test_pool);
    auto duration_cpu = std::chrono::high_resolution_clock::now() - t1;
    std::cout << "Duration when matrix multiplication by CPU " << duration_cpu << std::endl;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    tensor_engine::cuda::cudaMatmul(test_matrix_a, test_matrix_b);
    auto duration_gpu = std::chrono::high_resolution_clock::now() - t2;
    std::cout << "Duration when matrix multiplication by GPU " << duration_gpu << std::endl;

    EXPECT_TRUE(duration_gpu < duration_cpu);
}

TEST(GPUTiledGemmTest, CorrectnessCheck)
{
    Matrix test_matrix_a = Matrix::random(10, 15);
    Matrix test_matrix_b = Matrix::random(15, 20);
    Matrix test_matrix_error = Matrix::random(19, 20);

    EXPECT_THROW(tensor_engine::cuda::cudaTiledMatmul(test_matrix_a, test_matrix_error), std::invalid_argument);
    Matrix gpu_matmul = tensor_engine::cuda::cudaTiledMatmul(test_matrix_a, test_matrix_b);
    Matrix cpu_matmul = Matrix::multiplySequential(test_matrix_a, test_matrix_b);

    for (size_t i = 0; i < gpu_matmul.rows(); ++i) {
        for (size_t j = 0; j < gpu_matmul.cols(); ++j) {
            EXPECT_NEAR(gpu_matmul(i, j), cpu_matmul(i, j), 1e-4F);
        }
    }
}

TEST(GPUTiledGemmTest, GPUSpeedBenchmark)
{
    Matrix test_matrix_a = Matrix::random(1000, 1200);
    Matrix test_matrix_b = Matrix::random(1200, 1500);

    ThreadPool test_pool(4);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    tensor_engine::cuda::cudaMatmul(test_matrix_a, test_matrix_b);
    auto duration_naive_matmul = std::chrono::high_resolution_clock::now() - t1;
    std::cout << "Duration when matrix multiplication by CPU " << duration_naive_matmul << std::endl;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    tensor_engine::cuda::cudaTiledMatmul(test_matrix_a, test_matrix_b);
    auto duration_tiled_matmul = std::chrono::high_resolution_clock::now() - t2;
    std::cout << "Duration when matrix multiplication by GPU " << duration_tiled_matmul << std::endl;

    EXPECT_TRUE(duration_tiled_matmul < duration_naive_matmul);
}

TEST(CudaStreamTest, ObjectCheck)
{
    using namespace tensor_engine::cuda;
    CudaStream test_cuda_stream = CudaStream();
    CudaStream test_cuda_stream_moved = std::move(test_cuda_stream);

    EXPECT_EQ(test_cuda_stream.get(), nullptr);
    EXPECT_TRUE(test_cuda_stream_moved.get() != nullptr);

    PinnedVector<float> test_pinned_vector(100);
    for (size_t i = 0; i < test_pinned_vector.size(); ++i)
    {
        test_pinned_vector[i] = static_cast<float>(i);
    }
    EXPECT_FLOAT_EQ(test_pinned_vector[10], 10.0F);

    PinnedVector<float> test_pinned_vector_moved = std::move(test_pinned_vector);
    EXPECT_EQ(test_pinned_vector.data(), nullptr);
    EXPECT_FLOAT_EQ(test_pinned_vector_moved[15], 15.0F);
}