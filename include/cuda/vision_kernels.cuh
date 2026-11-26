#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include <opencv2/core.hpp>

namespace vision
{

void launchPreProcessing(const uint8_t* d_in, float* d_out, int in_width, int in_height, int out_width, int out_height, cudaStream_t stream);
void launchPostProcessing(const float* d_in, uint8_t* d_out, int width, int height, cudaStream_t stream);

} // namespace vision