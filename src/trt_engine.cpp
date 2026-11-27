#include "trt_engine.hpp"

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace vision
{
TRTEngine::TRTEngine(const std::string& engine_path)
{
    // Read .engine file into std::vector<char>
    std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
    if (!engine_file.is_open())
    {
        throw std::invalid_argument{"Cannot open the engine file. Please check if the directory is available for the .engine file is valid."};
    }
    auto fsize = engine_file.tellg();
    if (fsize <= 0)
    {
        throw std::runtime_error{"Engine file is empty of unreachable."};
    }
    std::vector<char> buffer(static_cast<size_t>(fsize)); // This is temp CPU RAM
    engine_file.seekg(0, std::ios::beg);
    engine_file.read(buffer.data(), fsize);

    // Create runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
    {
        throw std::runtime_error{"Cannot reset runtime."};
    }
    // Deserialize engine: Call this to cudaMalloc() + cudaMemcpy(H2D) to engine_ in VRAM
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine_)
    {
        throw std::runtime_error{"Cannot reset engine."};
    }
    // Create execution context: Create model graph (just like in PyTorch)
    context_.reset(engine_->createExecutionContext());
    if (!context_)
    {
        throw std::runtime_error{"Cannot reset context."};
    }
    
    // Inspect I/O tensor
    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            input_name_ = name;
            input_dims_ = engine_->getTensorShape(name);
        }
        else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            output_name_ = name;
            output_dims_ = engine_->getTensorShape(name);
        }
    }
}

// Act like a launcher, but for NN
void TRTEngine::inferAsync(const void* d_input, void* d_output, cudaStream_t stream)
{
    // Bind device buffers to tensor names
    context_->setTensorAddress(input_name_.c_str(), const_cast<void*>(d_input));
    context_->setTensorAddress(output_name_.c_str(), d_output);

    // Launch asynchronous inference
    if (!context_->enqueueV3(stream))
    {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
}

// Getters
const std::string& TRTEngine::inputName() const
{
    return input_name_;
}

const std::string& TRTEngine::outputName() const
{
    return output_name_;
}

const nvinfer1::Dims& TRTEngine::inputDims() const
{
    return input_dims_;
}

const nvinfer1::Dims& TRTEngine::outputDims() const
{
    return output_dims_;
}

void TRTLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cerr << msg << "\n";
    }
}

}