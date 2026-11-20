#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

enum class Device
{
    CPU,
    CUDA
};

class TensorShape
{
public:
    TensorShape(std::vector<size_t> dims): dims_(std::move(dims))
    {
        calculate_total_elements_();
        calculate_contiguous_strides_();
    }

    size_t total_elements() const
    {
        return total_elements_;
    }

    const std::vector<size_t>& contiguous_strides() const
    {
        return strides_;
    }

    const std::vector<size_t>& dims() { return dims_; }
    size_t nDims() const { return dims_.size(); }
    size_t operator[](size_t i) {return dims_[i]; }

private:
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
    size_t total_elements_{0};

    void calculate_total_elements_()
    {
        if (dims_.empty())
        {
            total_elements_ = 0;
        }
        else
        {
            total_elements_ = 1;
            for (size_t i = 0; i < dims_.size(); ++i)
            {
                total_elements_ *= dims_[i];
            }
        }
    }

    void calculate_contiguous_strides_()
    {
        if (dims_.empty())
        {
            strides_ = {};
        }
        else
        {
            strides_ = std::vector<size_t>(dims_.size(), 1);
            for (size_t i = strides_.size() - 1; i > 0 ; --i)
                {
                    strides_[i - 1] = strides_[i] * dims_[i];
                }
        }
    }

};

inline size_t compute_offset(const std::vector<size_t>& indices, const std::vector<size_t>& strides)
{
    size_t offset{0};
    if (indices.size() != strides.size())
    {
        throw std::invalid_argument{"The size of indices and strides are not the same!"};
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * strides[i];
    }
    return offset;
}