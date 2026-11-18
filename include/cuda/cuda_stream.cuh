#pragma once

#include <cuda_runtime.h>

namespace tensor_engine::cuda
{
class CudaStream
{
public:
    CudaStream();
    ~CudaStream() noexcept;

    // Disallow copy and assignment
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Default move
    CudaStream(CudaStream&& other_stream) noexcept;
    CudaStream& operator=(CudaStream&& other_stream) noexcept;

    // Methods
    cudaStream_t get() const;
    void synchronize() const;

private:
    cudaStream_t stream_;
};

} // namespace tensor_engine::cuda