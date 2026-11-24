#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "matrix.hpp"

namespace tensor_engine::cuda
{

constexpr int TILE_DIM = 16;

Matrix cudaMatmul(const Matrix& mat_a, const Matrix& mat_b);
Matrix cudaTiledMatmul(const Matrix& mat_a, const Matrix& mat_b);

void launchTiledGemm(const float* a, const float* b, float* c, size_t m, size_t k, size_t n, cudaStream_t stream = nullptr);
}