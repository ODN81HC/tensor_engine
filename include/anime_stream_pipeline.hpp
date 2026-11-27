#pragma once

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "trt_engine.hpp"
#include "cuda_stream.cuh"
#include "pinned_vector.cuh"
#include "vision_kernels.cuh"
#include "tensor.hpp"

namespace vision
{

class AnimeStreamPipeline
{

private:
    int in_width_{0};
    int in_height_{0};
    TRTEngine engine_;
    int out_width_{0};
    int out_height_{0};
    size_t in_size_{0};
    size_t in_bytes_{0};
    size_t out_size_{0};
    size_t out_bytes_{0};
    // Host pinned buffers
    tensor_engine::cuda::PinnedVector<uint8_t> pinned_in_;
    tensor_engine::cuda::PinnedVector<uint8_t> pinned_out_;
    // Tensor GPU buffers
    Tensor d_trt_in_;
    Tensor d_trt_out_;
    
    // GPU buffers
    uint8_t* d_raw_bgr_{nullptr};
    uint8_t* d_out_bgr_{nullptr};

    tensor_engine::cuda::CudaStream stream_;

public:
    AnimeStreamPipeline(int width, int height, const std::string& engine_path);
    ~AnimeStreamPipeline() noexcept;

    // Delete copy and move assignment
    AnimeStreamPipeline(const AnimeStreamPipeline& ) noexcept = delete;
    AnimeStreamPipeline& operator=(const AnimeStreamPipeline& ) noexcept = delete;
    AnimeStreamPipeline(AnimeStreamPipeline&& ) noexcept = delete;
    AnimeStreamPipeline& operator=(AnimeStreamPipeline&& ) noexcept = delete; 

    // Non-blocking asynchronous stream
    void submit_async(const cv::Mat& frame);
    cv::Mat sync_and_get_frame();
};

}