#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace tensor_engine::cuda {

// Host function: takes CPU vector, runs ReLU on the GPU, returns CPU vector
std::vector<float> relu(const std::vector<float>& host_input);

void launchRelu(const float* input, float* output, size_t size, cudaStream_t stream = nullptr);
}