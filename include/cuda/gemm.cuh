#pragma once

#include <cstddef>

#include "matrix.hpp"

namespace tensor_engine::cuda
{

Matrix cudaMatmul(const Matrix& mat_a, const Matrix& mat_b);

}