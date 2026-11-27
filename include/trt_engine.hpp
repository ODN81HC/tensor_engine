#pragma once

#include <sstream>
#include <memory>

#include <cuda_runtime.h>
#include <NvInfer.h>

namespace vision
{

class TRTLogger : public nvinfer1::ILogger
{

public:
    TRTLogger() {}

private:
    void log(Severity severity, const char* msg) noexcept override;
};

class TRTEngine
{

public:
    explicit TRTEngine(const std::string& engine_path);
    ~TRTEngine() = default;

    // Delete copy and move
    TRTEngine(const TRTEngine&) noexcept = delete;
    TRTEngine& operator=(const TRTEngine&) noexcept = delete;
    TRTEngine(TRTEngine&&) noexcept = delete;
    TRTEngine& operator=(TRTEngine &&) noexcept = delete;

    void inferAsync(const void* d_input, void* d_output, cudaStream_t stream);

    // Getters
    const std::string& inputName() const;
    const std::string& outputName() const;
    const nvinfer1::Dims& inputDims() const;
    const nvinfer1::Dims& outputDims() const; 

private:
    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string output_name_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};
};

} // namespace vision
