#pragma once

#include "thread_pool.hpp"

#include <vector>

class Matrix
{
public:
    Matrix(size_t rows, size_t cols, float initial_val = 0.0F);

    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    const float* data() const;
    float* data();

    static Matrix random(size_t rows, size_t cols);
    static Matrix multiplySequential(const Matrix& mat_a, const Matrix& mat_b);
    static Matrix multiplyParallel(const Matrix& mat_a, const Matrix& mat_b, ThreadPool& pool);

    size_t rows() const;
    size_t cols() const;

private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
};

