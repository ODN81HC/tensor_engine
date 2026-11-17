#pragma once

#include <cstddef>

#include "matrix.hpp"

namespace tensor_engine::cuda
{

constexpr int TILE_DIM = 16;

Matrix cudaMatmul(const Matrix& mat_a, const Matrix& mat_b);
Matrix cudaTiledMatmul(const Matrix& mat_a, const Matrix& mat_b);

}