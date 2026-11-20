#pragma once

#include "cuda_stream.cuh"
#include "image_processing.cuh"
#include "pinned_vector.cuh"

#include <opencv2/core.hpp>

namespace tensor_engine::cuda
{
class CameraStreamPipeline
{
public:
    CameraStreamPipeline(int width, int height);
    ~CameraStreamPipeline() noexcept;
    // Delete copy and move constructor and assignment
    CameraStreamPipeline(const CameraStreamPipeline&) = delete;
    CameraStreamPipeline& operator=(const CameraStreamPipeline&) = delete;
    CameraStreamPipeline(CameraStreamPipeline&&) noexcept = delete;
    CameraStreamPipeline& operator=(CameraStreamPipeline&&) noexcept = delete;

    void submit_async(const cv::Mat& frame); // Copy frame to pinned_in_, queue H2D, queue kernel, queue D2H. DO NOT Synchronize!
    cv::Mat sync_and_get_frame(); // call stream_.synchronize()

private:
    int width_{0};
    int height_{0};
    size_t size_{0};
    size_t bytes_{0};

    CudaStream stream_;
    uint8_t* d_in_{nullptr};
    uint8_t* d_out_{nullptr};
    PinnedVector<uint8_t> pinned_in_;
    PinnedVector<uint8_t> pinned_out_;
};

} // namespace tensor_engine::cuda
