#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <stdexcept>

namespace tensor_engine::cuda
{

template <typename T>
class PinnedVector
{
public:
    explicit PinnedVector(size_t size) : size_(size)
    {
        // Allocate pinned vector
        if (size_ > 0)
        {
            cudaMallocHost(reinterpret_cast<void**>(&data_), size * sizeof(T));
        }
    }
    ~PinnedVector() noexcept
    {
        if (data_ != nullptr)
        {
            cudaFreeHost(data_);
            data_ = nullptr;
        }
    }

    // Rule of 5, disable constructor copy and assignment
    // Implement move
    PinnedVector(const PinnedVector&) noexcept = delete;
    PinnedVector& operator=(const PinnedVector&) noexcept = delete;

    PinnedVector(PinnedVector&& other_pinned_vector) noexcept : data_(other_pinned_vector.data_), size_(other_pinned_vector.size_)
    {
        other_pinned_vector.data_ = nullptr;
        other_pinned_vector.size_ = 0;
    }
    PinnedVector& operator=(PinnedVector&& other_pinned_vector) noexcept
    {
        if (this != &other_pinned_vector)
        {
            if (data_ != nullptr && size_ != 0)
            {
                cudaFreeHost(data_);
                data_ = nullptr;
                size_ = 0;
            }
            data_ = other_pinned_vector.data_;
            size_ = other_pinned_vector.size_;

            other_pinned_vector.data_ = nullptr;
            other_pinned_vector.size_ = 0;
        }
        return *this;
    }

    // Member access
    T* data() noexcept
    {
        return data_;
    }
    const T* data() const noexcept
    {
        return data_;
    }
    size_t size() const noexcept
    {
        return size_;
    }
    T& operator[](size_t idx)
    {
        if (idx >= size_) {
            throw std::out_of_range{"The index is out of range."};
        }
        return data_[idx];
    }
    const T& operator[](size_t idx) const
    {
        if (idx >= size_) {
            throw std::out_of_range{"The index is out of range."};
        }
        return data_[idx];
    }

private:
    T* data_{nullptr};
    size_t size_{0};

};

} // namespace tensor_engine::cuda