#include <gtest/gtest.h>
#include <chrono>

#include "matrix.hpp"
#include "thread_pool.hpp"

TEST(MatrixTest, MatrixCorrectnessCheck)
{
    Matrix test_mat_a = Matrix::random(4, 5);
    Matrix test_mat_b = Matrix::random(5, 6);
    Matrix test_mat_error = Matrix::random(900, 1000);
    
    ThreadPool test_pool(4);

    EXPECT_THROW(test_mat_a(501, 401), std::out_of_range);
    EXPECT_THROW(Matrix::multiplySequential(test_mat_a, test_mat_error), std::invalid_argument);
    EXPECT_THROW(Matrix::multiplyParallel(test_mat_b, test_mat_error, test_pool), std::invalid_argument);

    Matrix seq_matrix = Matrix::multiplySequential(test_mat_a, test_mat_b);
    Matrix parallel_matrix = Matrix::multiplyParallel(test_mat_a, test_mat_b, test_pool);
    for (size_t i = 0; i < seq_matrix.rows(); ++i) {
        for (size_t j = 0; j < seq_matrix.cols(); ++j) {
            EXPECT_EQ(seq_matrix(i, j), parallel_matrix(i, j));
        }
    }
}

TEST(MatrixTest, SpeedBenchmark)
{
    Matrix test_mat_a = Matrix::random(500, 600);
    Matrix test_mat_b = Matrix::random(600, 700);

    ThreadPool test_pool(10);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Matrix::multiplySequential(test_mat_a, test_mat_b);
    auto duration_sequential = std::chrono::high_resolution_clock::now() - t1;
    std::cout << "Duration when matrix multiplication sequentially: " << duration_sequential << std::endl;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    Matrix::multiplyParallel(test_mat_a, test_mat_b, test_pool);
    auto duration_parallel = std::chrono::high_resolution_clock::now() - t2;
    std::cout << "Duration when matrix multiplication parallely: " << duration_parallel << std::endl;

    EXPECT_TRUE(duration_parallel < duration_sequential);
}