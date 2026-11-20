#include <gtest/gtest.h>
#include <vector>

#include "tensor.hpp"

TEST(TensorTest, CreateAndOperationsWithTensor)
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