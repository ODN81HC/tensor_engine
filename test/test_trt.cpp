#include "trt_engine.hpp"
#include "tensor.hpp"
#include "cuda_stream.cuh"
#include <gtest/gtest.h>

TEST(TRTEngineTest, SmokeTest)
{
    using namespace vision;

    std::string animegan_engine_path = "/home/tonan/study/cpp_study/tensor_engine/models/animeganv3.engine";
    TRTEngine test_gan_engine(animegan_engine_path);
    const auto& input_dims = test_gan_engine.inputDims();

    EXPECT_EQ(input_dims.nbDims, 4);
    EXPECT_EQ(input_dims.d[0], 1);
    EXPECT_EQ(input_dims.d[1], 512);
    EXPECT_EQ(input_dims.d[2], 512);
    EXPECT_EQ(input_dims.d[3], 3);

    // Infer
    Tensor test_tensor_1({1, 512, 512, 3}, Device::CUDA);
    Tensor test_tensor_2({1, 512, 512, 3}, Device::CUDA);
    tensor_engine::cuda::CudaStream test_cuda_stream;
    EXPECT_NO_THROW(test_gan_engine.inferAsync(test_tensor_1.data(), test_tensor_2.data(), test_cuda_stream.get()));
    test_cuda_stream.synchronize();

    // Map d_out back to test_tensor_2, so we can transfer that back to CPU and assert something
    Tensor cpu_out = test_tensor_2.to(Device::CPU);
    EXPECT_GE(cpu_out({0, 0, 0, 0}), -1.0F);
    EXPECT_LE(cpu_out({0, 0, 0, 0}), 1.0F);
}
