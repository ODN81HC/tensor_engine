#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace tensor_engine::cuda
{

void launchBgrToGrayscale(const uint8_t* d_in, uint8_t* d_out, int width, int height, cudaStream_t stream);

}