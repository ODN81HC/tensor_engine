#include "camera_stream_pipeline.hpp"

namespace tensor_engine::cuda
{
CameraStreamPipeline::CameraStreamPipeline(int width, int height): width_(width), height_(height), size_(static_cast<unsigned int>(width * height)), bytes_(size_ * sizeof(uint8_t)), pinned_in_(size_ * 3), pinned_out_(size_)
{
    // Allocate d_in_ and d_out_
    cudaMalloc(&d_in_, size_ * 3);
    cudaMalloc(&d_out_, size_);
}

CameraStreamPipeline::~CameraStreamPipeline() noexcept
{
    cudaFree(d_in_);
    cudaFree(d_out_);
}

void CameraStreamPipeline::submit_async(const cv::Mat& frame)
{
    // Copy input frame to pinned_in_ memory
    std::memcpy(pinned_in_.data(), frame.data, bytes_ * 3);
    // Queue H2D
    cudaMemcpyAsync(d_in_, pinned_in_.data(), bytes_ * 3, cudaMemcpyHostToDevice, stream_.get());
    // Queue kernel
    launchBgrToGrayscale(d_in_, d_out_, width_, height_, stream_.get());
    // Queue D2H
    cudaMemcpyAsync(pinned_out_.data(), d_out_, bytes_, cudaMemcpyDeviceToHost, stream_.get());
}

cv::Mat CameraStreamPipeline::sync_and_get_frame()
{
    // Call cudaSynchronize()
    stream_.synchronize();

    // Output with OpenCV
    cv::Mat mat_out = cv::Mat(height_, width_, CV_8UC1, pinned_out_.data());
    return mat_out;
}

}