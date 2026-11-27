#include "anime_stream_pipeline.hpp"

namespace vision
{

AnimeStreamPipeline::AnimeStreamPipeline(int width, int height, const std::string& engine_path) : 
                in_width_{width}, 
                in_height_{height}, 
                engine_(engine_path), 
                out_width_{static_cast<int>(engine_.outputDims().d[2])}, 
                out_height_{static_cast<int>(engine_.outputDims().d[1])}, 
                in_size_{static_cast<size_t>(in_width_ * in_height_) * 3},
                in_bytes_{in_size_ * sizeof(uint8_t)},
                out_size_{static_cast<size_t>(out_width_ * out_height_) * 3},
                out_bytes_{out_size_ * sizeof(uint8_t)}, 
                pinned_in_(in_size_), 
                pinned_out_(out_size_), 
                d_trt_in_({1, static_cast<size_t>(out_width_), static_cast<size_t>(out_height_), 3}, Device::CUDA), 
                d_trt_out_({1, static_cast<size_t>(out_width_), static_cast<size_t>(out_height_), 3}, Device::CUDA)
{
    // Allocate d_raw_bgr_ and d_out_bgr_
    cudaMalloc(&d_raw_bgr_, in_size_);
    cudaMalloc(&d_out_bgr_, out_size_);
}

AnimeStreamPipeline::~AnimeStreamPipeline() noexcept
{
    cudaFree(d_raw_bgr_);
    cudaFree(d_out_bgr_);
}

void AnimeStreamPipeline::submit_async(const cv::Mat& frame)
{   
    // This is the main pipeline flow
    // 1. Pinned the CV Frame to pinned_in.data() with the OpenCV in_size
    // 2. Copy that pinned_in.data() -> d_raw_bgr (this is because OpenCV is read in BGR)
    // 3. Preprocessing kernel launcher: in: d_raw_bgr -> stores in d_trt_in_
    // 4. AnimeLauncher d_trt_in_ -> d_trt_out_
    // 5. Postprocessing kernel launcher: in: d_trt_out_ -> stores results in d_out_bgr
    // 6. Copy result back to pinned_out.data()

    std::memcpy(pinned_in_.data(), frame.data, in_bytes_);

    cudaMemcpyAsync(d_raw_bgr_, pinned_in_.data(), in_bytes_, cudaMemcpyHostToDevice, stream_.get());

    launchPreProcessing(d_raw_bgr_, d_trt_in_.data(), in_width_, in_height_, out_width_, out_height_, stream_.get());
    engine_.inferAsync(d_trt_in_.data(), d_trt_out_.data(), stream_.get());
    launchPostProcessing(d_trt_out_.data(), d_out_bgr_, out_width_, out_height_, stream_.get());

    cudaMemcpyAsync(pinned_out_.data(), d_out_bgr_, out_bytes_, cudaMemcpyDeviceToHost, stream_.get());
}

cv::Mat AnimeStreamPipeline::sync_and_get_frame()
{
    stream_.synchronize();

    cv::Mat mat_out = cv::Mat(out_height_, out_width_, CV_8UC3, pinned_out_.data());
    return mat_out;
}

}