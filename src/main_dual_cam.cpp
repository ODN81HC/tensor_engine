#include "camera_stream_pipeline.hpp"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace tensor_engine::cuda;

int main()
{
    int constexpr esc_key = 27;
    // Open laptop webcam (0) and remote webcam (4)
    cv::VideoCapture cap_0(0);
    cv::VideoCapture cap_4(4);

    int width_cap_0 = 0;
    int height_cap_0 = 0;
    int width_cap_4 = 0;
    int height_cap_4 = 0;

    // Verify if webcams are working and query frame width and height
    if (!cap_0.isOpened())
    {
        std::cerr << "Cannot open laptop webcam!\n";
        return 1;
    }
    else
    {
        width_cap_0 = static_cast<int>(cap_0.get(cv::CAP_PROP_FRAME_WIDTH));
        height_cap_0 = static_cast<int>(cap_0.get(cv::CAP_PROP_FRAME_HEIGHT));
    }

    if (!cap_4.isOpened())
    {
        std::cerr << "Cannot open remote webcam!\n";
        return 1;
    }
    else
    {
        width_cap_4 = static_cast<int>(cap_4.get(cv::CAP_PROP_FRAME_WIDTH));
        height_cap_4 = static_cast<int>(cap_4.get(cv::CAP_PROP_FRAME_HEIGHT));
    }

    CameraStreamPipeline camera_stream_pipeline_0(width_cap_0, height_cap_0);
    CameraStreamPipeline camera_stream_pipeline_4(width_cap_4, height_cap_4);

    cv::Mat frame_0; 
    cv::Mat frame_4;
    while (cap_0.read(frame_0) && cap_4.read(frame_4))
    {
        camera_stream_pipeline_0.submit_async(frame_0);
        camera_stream_pipeline_4.submit_async(frame_4);
        // Process and get output frame to CPU
        const cv::Mat mat_out_0 = camera_stream_pipeline_0.sync_and_get_frame();
        const cv::Mat mat_out_4 = camera_stream_pipeline_4.sync_and_get_frame();

        // Stitch them horizontally
        if (height_cap_0 != height_cap_4 || width_cap_0 != width_cap_4)
        {
            // Resize cap_4 to the same size as cap_0
            cv::resize(mat_out_4, mat_out_4, cv::Size(width_cap_0, height_cap_0));
        }
        const cv::Mat side_by_side = cv::Mat(height_cap_0, width_cap_0 * 2, CV_8UC1);
        cv::hconcat(mat_out_0, mat_out_4, side_by_side);
        cv::imshow("Dual camera GPU stream", side_by_side);

        if (cv::waitKey(1) == esc_key)
        {
            break;
        }
    }
    return 0;
}