#include "cuda_stream.cuh"
#include "image_processing.cuh"
#include "pinned_vector.cuh"

#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace tensor_engine::cuda;

int main()
{
    // Open laptop webcam (0) and remote webcam (4)
    cv::VideoCapture cap_0(0);
    cv::VideoCapture cap_4(4);

    int width_cap_0 = 0, height_cap_0 = 0;
    int width_cap_4 = 0, height_cap_4 = 0;

    // Verify if webcams are working and query frame width and height
    if (!cap_0.isOpened())
    {
        throw std::runtime_error{"Laptop webcam cannot be opened."};
    }
    else
    {
        width_cap_0 = static_cast<int>(cap_0.get(cv::CAP_PROP_FRAME_WIDTH));
        height_cap_0 = static_cast<int>(cap_0.get(cv::CAP_PROP_FRAME_HEIGHT));
    }

    if (!cap_4.isOpened())
    {
        throw std::runtime_error{"Remote webcam cannot be opened."};
    }
    else
    {
        width_cap_4 = static_cast<int>(cap_4.get(cv::CAP_PROP_FRAME_WIDTH));
        height_cap_4 = static_cast<int>(cap_4.get(cv::CAP_PROP_FRAME_HEIGHT));
    }

    CudaStream stream_0;
    CudaStream stream_4;

    // Allocate device VRAM
    uint8_t* d_in_0 = nullptr;
    uint8_t* d_out_0 = nullptr;
    uint8_t* d_in_4 = nullptr;
    uint8_t* d_out_4 = nullptr;

    size_t size_cap_0 = width_cap_0 * height_cap_0;
    size_t bytes_cap_0 = size_cap_0 * sizeof(uint8_t);
    size_t size_cap_4 = width_cap_4 * height_cap_4;
    size_t bytes_cap_4 = size_cap_4 * sizeof(uint8_t);

    cudaMalloc(&d_in_0, size_cap_0 * 3);
    cudaMalloc(&d_out_0, size_cap_0);
    cudaMalloc(&d_in_4, size_cap_4 * 3);
    cudaMalloc(&d_out_4, size_cap_4);

    // Allocate pinned host memory
    // Thought that when we got the pinnedvector here, meaning that it already do cudaMallocHost() without doing the cudaMalloc() anymore?
    PinnedVector<uint8_t> pinned_in_0(size_cap_0 * 3);
    PinnedVector<uint8_t> pinned_out_0(size_cap_0);

    PinnedVector<uint8_t> pinned_in_4(size_cap_4 * 3);
    PinnedVector<uint8_t> pinned_out_4(size_cap_4);

    cv::Mat frame_0, frame_4;
    while (cap_0.read(frame_0) && cap_4.read(frame_4))
    {
        // Copy to pinned memory
        std::memcpy(pinned_in_0.data(), frame_0.data, bytes_cap_0 * 3);
        std::memcpy(pinned_in_4.data(), frame_4.data, bytes_cap_4 * 3); // Only copy input, not output?

        // Async pipeline for cam0
        cudaMemcpyAsync(d_in_0, pinned_in_0.data(), bytes_cap_0 * 3, cudaMemcpyHostToDevice, stream_0.get());
        cudaMemcpyAsync(d_in_4, pinned_in_4.data(), bytes_cap_4 * 3, cudaMemcpyHostToDevice, stream_4.get());

        launchBgrToGrayscale(d_in_0, d_out_0, width_cap_0, height_cap_0, stream_0.get());
        launchBgrToGrayscale(d_in_4, d_out_4, width_cap_4, height_cap_4, stream_4.get());

        cudaMemcpyAsync(pinned_out_0.data(), d_out_0, bytes_cap_0, cudaMemcpyDeviceToHost, stream_0.get());
        cudaMemcpyAsync(pinned_out_4.data(), d_out_4, bytes_cap_4, cudaMemcpyDeviceToHost, stream_4.get());
        
        // Stream synchronize
        stream_0.synchronize();
        stream_4.synchronize();

        // Display with OpenCV
        cv::Mat mat_out_0 = cv::Mat(height_cap_0, width_cap_0, CV_8UC1, pinned_out_0.data());
        cv::Mat mat_out_4 = cv::Mat(height_cap_4, width_cap_4, CV_8UC1, pinned_out_4.data());
        // Stitch them horizontally
        if (height_cap_0 != height_cap_4 || width_cap_0 != width_cap_4)
        {
            // Resize cap_4 to the same size as cap_0
            cv::resize(mat_out_4, mat_out_4, cv::Size(width_cap_0, height_cap_0));
        }
        cv::Mat side_by_side = cv::Mat(height_cap_0, width_cap_0 * 2, CV_8UC1);
        cv::hconcat(mat_out_0, mat_out_4, side_by_side);
        cv::imshow("Dual camera GPU stream", side_by_side);

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
    // Cleanup
    cudaFree(d_in_0);
    cudaFree(d_out_0);
    cudaFree(d_in_4);
    cudaFree(d_out_4);
}