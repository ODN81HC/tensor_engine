#include <gtest/gtest.h>
#include <chrono>

#include "activations.cuh"
#include "gemm.cuh"
#include "matrix.hpp"
#include "thread_pool.hpp"

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