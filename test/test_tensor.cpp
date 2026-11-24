#include <gtest/gtest.h>
#include <vector>

#include "tensor.hpp"

TEST(TensorTest, CreateAndOperationsWithTensorShape)
{
    TensorShape test_tensor_shape(std::vector<size_t>{2, 3, 4, 4});
    EXPECT_EQ(test_tensor_shape.total_elements(), 96);
    
    std::vector<size_t> test_strides{48, 16, 4, 1};
    EXPECT_EQ(test_tensor_shape.contiguous_strides(), test_strides);

    EXPECT_EQ(test_tensor_shape[0], 2);

    size_t test_offset = compute_offset(std::vector<size_t>{1, 0, 2, 3}, test_tensor_shape.contiguous_strides());
    EXPECT_EQ(test_offset, 59);

    TensorShape test_1D_element(std::vector<size_t>{10});
    std::vector<size_t> strides_1d{1};
    EXPECT_EQ(test_1D_element.contiguous_strides(), strides_1d);
    EXPECT_EQ(test_1D_element.total_elements(), 10);

}

TEST(TensorTest, CreateAndOperationsWithTensor)
{
    Tensor test_cpu_tensor({2, 3, 4}, Device::CPU);
    Tensor test_cuda_tensor({2, 6, 4}, Device::CUDA);

    // Operations on CPU
    test_cpu_tensor({0, 0, 1}) = 2.0f;
    test_cpu_tensor({0, 0, 2}) = 3.0f;
    test_cpu_tensor({1, 0, 0}) = 10.0f;

    EXPECT_FLOAT_EQ(test_cpu_tensor({1, 0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(test_cpu_tensor({0, 0, 2}), 3.0f);

    // Reshape
    Tensor new_reshaped_tensor = test_cuda_tensor.reshape({12, 4});
    // Check if the underlying is the same data
    EXPECT_EQ(new_reshaped_tensor.data(), test_cuda_tensor.data());
}

TEST(TensorTest, TransposeTensorOperation)
{
    Tensor test_cpu_tensor({2, 3, 4}, Device::CPU);

    // Transpose
    Tensor transpose_tensor = test_cpu_tensor.transpose(0, 2);
    EXPECT_EQ(transpose_tensor.shape()[0], 4);
    EXPECT_EQ(transpose_tensor.shape()[2], 2);

    EXPECT_FALSE(transpose_tensor.is_contiguous());
}

TEST(TensorTest, SliceTensorOperation)
{
    Tensor test_cpu_tensor({2, 3}, Device::CPU);

    // Add some elements
    test_cpu_tensor({0, 1}) = 50.0F;
    test_cpu_tensor({1, 1}) = 49.9F;
    // Slice
    Tensor slice_tensor = test_cpu_tensor.slice(1, 1, 2);
    EXPECT_FLOAT_EQ(slice_tensor({0, 0}), test_cpu_tensor({0, 1}));
    EXPECT_FLOAT_EQ(slice_tensor({1, 0}), test_cpu_tensor({1, 1}));
}

TEST(TensorTest, GEMMTiledMatmul)
{
    Tensor tensor_cpu_1({4, 5}, Device::CPU);
    Tensor tensor_cpu_2({5, 10}, Device::CPU);
    Tensor tensor_cuda_wrong({2, 3, 4}, Device::CUDA);

    // Fill in some values
    tensor_cpu_1({0, 0}) = 10.0F;
    tensor_cpu_2({0, 0}) = 10.0F;

    // Convert to CUDA Tensor
    Tensor tensor_cuda_1 = tensor_cpu_1.to(Device::CUDA);
    Tensor tensor_cuda_2 = tensor_cpu_2.to(Device::CUDA);

    EXPECT_THROW(tensor_cuda_1.matmul(tensor_cuda_wrong), std::invalid_argument);
    Tensor result_matmul_cuda = tensor_cuda_1.matmul(tensor_cuda_2);

    // Convert back to CPU
    Tensor result_matmul_cpu = result_matmul_cuda.to(Device::CPU);
    EXPECT_FLOAT_EQ(result_matmul_cpu({0, 0}), 100.0F);
}

TEST(TensorTest, ReLUOperation)
{
    Tensor tensor_cpu_1({4, 5}, Device::CPU);
    
    tensor_cpu_1({0, 0}) = 10.0F;
    tensor_cpu_1({1, 0}) = -10.0F;

    // Convert to CUDA
    Tensor tensor_cuda_1 = tensor_cpu_1.to(Device::CUDA);

    Tensor result_relu_tensor = tensor_cuda_1.relu();

    // Convert back to CPU
    Tensor result_cpu = result_relu_tensor.to(Device::CPU);

    EXPECT_FLOAT_EQ(result_cpu({0, 0}), 10.0F);
    EXPECT_FLOAT_EQ(result_cpu({1, 0}), 0.0F);
    
    // Test in-place
    tensor_cuda_1.relu_();
    Tensor in_place_cpu = tensor_cuda_1.to(Device::CPU);
    EXPECT_FLOAT_EQ(in_place_cpu({1, 0}), 0.0F);
}