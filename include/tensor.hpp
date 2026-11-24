#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

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

    const std::vector<size_t>& dims() const { return dims_; }
    size_t nDims() const { return dims_.size(); }
    size_t operator[](size_t i) const {return dims_[i]; }

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

class Storage
{
private:
    float* data_{nullptr};
    size_t size_{0};
    Device device_{Device::CPU};

public:
    Storage(size_t size, Device device) : size_(size), device_(device)
    {
        // Initialize data_ based on device type
        if (device_ == Device::CPU)
        {
            data_ = new float[size_]();
        }
        else if (device_ == Device::CUDA)
        {
            cudaMalloc(&data_, size_ * sizeof(float));
            cudaMemset(data_, 0, size_ * sizeof(float));
        }
    }
    ~Storage() noexcept
    {
        if (device_ == Device::CPU)
        {
            delete[] data_;
        }
        else if (device_ == Device::CUDA)
        {
            cudaFree(data_);
        }
        data_ = nullptr;
    }

    // Delete copy assignment
    Storage(const Storage&) noexcept = delete;
    Storage& operator=(const Storage&) noexcept = delete;

    // Custom move assignment
    Storage(Storage&& other_storage) noexcept : data_(other_storage.data_), size_(other_storage.size_), device_(other_storage.device_)
    {
        other_storage.data_ = nullptr;
        other_storage.size_ = 0;
    }

    Storage& operator=(Storage&& other_storage) noexcept
    {
        if (this != &other_storage)
        {
            if (data_ != nullptr && size_ != 0)
            {
                if (device_ == Device::CPU)
                {
                    delete[] data_;
                }
                else if (device_ == Device::CUDA)
                {
                    cudaFree(data_);
                }
                size_ = 0;
            }
            data_ = other_storage.data_;
            size_ = other_storage.size_;
            device_ = other_storage.device_;

            other_storage.data_ = nullptr;
            other_storage.size_ = 0;
        }
        return *this;
    }

    float* data() noexcept {return data_; };
    const float* data() const noexcept {return data_; };
    size_t size() const noexcept { return size_; };
    Device device() const noexcept { return device_; };
};

class Tensor
{
private:
    Tensor(TensorShape tensor_shape, std::vector<size_t> strides, std::shared_ptr<Storage> storage, size_t storage_offset) : tensor_shape_(tensor_shape), strides_(std::move(strides)), storage_(storage), storage_offset_(storage_offset) {}

    TensorShape tensor_shape_;
    std::vector<size_t> strides_;
    std::shared_ptr<Storage> storage_;
    size_t storage_offset_{0};

public:
    Tensor(std::vector<size_t> dims, Device device = Device::CPU) : tensor_shape_(std::move(dims))
    {
        strides_ = tensor_shape_.contiguous_strides();
        storage_ = std::make_shared<Storage>(tensor_shape_.total_elements(), device);
    }

    // Accessor
    const TensorShape& shape() const noexcept { return tensor_shape_; };
    const std::vector<size_t>& strides() const noexcept { return strides_; };
    Device device() const noexcept { return storage_->device(); };
    size_t total_elements() const noexcept { return tensor_shape_.total_elements(); };
    float* data() noexcept { return storage_->data() + storage_offset_; };
    const float* data() const noexcept { return storage_->data() + storage_offset_; };

    // operators()
    float& operator()(const std::vector<size_t>& indices)
    {
        if (storage_->device() != Device::CPU)
        {
            throw std::runtime_error{"Indexing cannot be performed in CUDA tensor."};
        }
        if (indices.size() != tensor_shape_.nDims())
        {
            throw std::invalid_argument{"Input indices and tensor shape doesn't have the same size!"};
        }
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= tensor_shape_[i])
            {
                throw std::out_of_range{"Index at " + std::to_string(i) + "cannot be larger than the tensor shape."};
            }
        }
        return storage_->data()[compute_offset(indices, strides_) + storage_offset_];
    }
    const float& operator()(const std::vector<size_t>& indices) const
    {
        if (storage_->device() != Device::CPU)
        {
            throw std::runtime_error{"Indexing cannot be performed in CUDA tensor."};
        }
        if (indices.size() != tensor_shape_.nDims())
        {
            throw std::invalid_argument{"Input indices and tensor shape doesn't have the same size!"};
        }
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= tensor_shape_[i])
            {
                throw std::out_of_range{"Index at " + std::to_string(i) + "cannot be larger than the tensor shape."};
            }
        }
        return storage_->data()[compute_offset(indices, strides_) + storage_offset_];
    }

    // Zero-copy operations
    bool is_contiguous() const noexcept { return strides_ == tensor_shape_.contiguous_strides(); };
    Tensor reshape(const std::vector<size_t>& new_dims) const
    {
        TensorShape new_tensor_shape(new_dims);
        if (new_tensor_shape.total_elements() != tensor_shape_.total_elements())
        {
            throw std::invalid_argument{"Dimensions are not equal, cannot reshape!"};
        }
        else
        {
            if (is_contiguous())
            {   
                // Create a zero-copy based on private Tensor constructor
                Tensor reshaped_tensor(new_tensor_shape, new_tensor_shape.contiguous_strides(), storage_, storage_offset_);
                return reshaped_tensor;
            }
            throw std::invalid_argument{"Cannot reshape because the tensor is not contiguous!"};
        }
    }

    Tensor transpose(size_t dim0 = 0, size_t dim1 = 1) const
    {
        if (dim0 < tensor_shape_.nDims() && dim1 < tensor_shape_.nDims())
        {
            auto new_dims = tensor_shape_.dims();
            auto new_strides = strides_;

            std::swap(new_dims[dim0], new_dims[dim1]);
            std::swap(new_strides[dim0], new_strides[dim1]);

            Tensor transpose_tensor(new_dims, new_strides, storage_, storage_offset_);
            return transpose_tensor;
        }
        else
        {
            throw std::invalid_argument{"Make sure that dim0 and dim1 are smaller than the actual size of a tensor."};
        }
    }

    Tensor slice(size_t dim, size_t start, size_t end) const
    {
        if (dim < tensor_shape_.nDims() && start < end && end <= tensor_shape_[dim])
        {
            auto new_dims = tensor_shape_.dims();
            new_dims[dim] = end - start;

            size_t new_offset = storage_offset_ + start * strides_[dim];

            Tensor sub_tensor(new_dims, strides_, storage_, new_offset);
            return sub_tensor;
        }
        else
        {
            throw std::out_of_range{"Make sure that the arguments satisfy!"};
        }
    }

    // Device transfer
    Tensor to(Device target_device) const
    {
        if (device() == target_device)
        {
            return *this;
        }
        else
        {   
            Tensor destination_tensor(tensor_shape_.dims(), target_device);
            if (target_device == Device::CPU)
            {
                cudaMemcpy(destination_tensor.data(), this->data(), tensor_shape_.total_elements() * sizeof(float), cudaMemcpyDeviceToHost);
            }
            else if (target_device == Device::CUDA)
            {
                cudaMemcpy(destination_tensor.data(), this->data(), tensor_shape_.total_elements() * sizeof(float), cudaMemcpyHostToDevice);
            }
            return destination_tensor;
        }
    }
};