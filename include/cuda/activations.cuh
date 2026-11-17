#pragma once

#include <vector>

namespace tensor_engine::cuda {

// Host function: takes CPU vector, runs ReLU on the GPU, returns CPU vector
std::vector<float> relu(const std::vector<float>& host_input);

}