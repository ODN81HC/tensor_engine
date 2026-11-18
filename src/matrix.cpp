#include "matrix.hpp"

#include <random>
#include <stdexcept>
#include <latch>

Matrix::Matrix(size_t rows, size_t cols, float initial_val):
    rows_(rows), cols_(cols), data_(rows * cols, initial_val)
{}

float& Matrix::operator()(size_t row, size_t col)
{
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range{"Matrix subscript out of range"};
    }
    return data_[row * cols_ + col];
}

const float& Matrix::operator()(size_t row, size_t col) const
{
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range{"Matrix subscript out of range"};
    }
    return data_[row * cols_ + col];
}

const float* Matrix::data() const
{
    return data_.data();
}

float* Matrix::data()
{
    return data_.data();
}

size_t Matrix::rows() const
{
    return rows_;
}

size_t Matrix::cols() const
{
    return cols_;
}

Matrix Matrix::random(size_t rows, size_t cols)
{
    Matrix mat(rows, cols);
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_real_distribution<float> dis(0.0F, 1.0F);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = dis(gen);
        }
    }
    return mat;
}

Matrix Matrix::multiplySequential(const Matrix& mat_a, const Matrix& mat_b)
{
    if (mat_a.cols() != mat_b.rows()) {
        throw std::invalid_argument{"Incompatible matrix dimensions for multiplication"};
    }
    Matrix result(mat_a.rows(), mat_b.cols());
    for (size_t i = 0; i < mat_a.rows(); ++i) {
        for (size_t j = 0; j < mat_b.cols(); ++j) {
            float sum = 0.0F;
            for (size_t k = 0; k < mat_a.cols(); ++k) {
                sum += mat_a(i, k) * mat_b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::multiplyParallel(const Matrix& mat_a, const Matrix& mat_b, ThreadPool& pool)
{
    if (mat_a.cols() != mat_b.rows()) {
        throw std::invalid_argument{"Incompatible matrix dimensions for multiplication"};
    }
    Matrix result(mat_a.rows(), mat_b.cols());
    std::latch done_latch(static_cast<std::ptrdiff_t>(result.rows()));

    for (size_t i = 0; i < result.rows(); ++i) {
        pool.enqueue([&result, &mat_a, &mat_b, &done_latch, i]() {
            for (size_t k = 0; k < mat_a.cols(); ++k) {
                const float mat_aik = mat_a(i, k);
                for (size_t j = 0; j < mat_b.cols(); ++j) {
                    result(i, j) += mat_aik * mat_b(k, j);
                }
            }
            done_latch.count_down();
        });
    }
    done_latch.wait();
    return result;
}