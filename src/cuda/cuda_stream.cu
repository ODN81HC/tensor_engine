#include "cuda_stream.cuh"

using namespace tensor_engine::cuda;

CudaStream::CudaStream()
{
    cudaStreamCreate(&stream_);
}

CudaStream::~CudaStream() noexcept
{
    if (stream_ != nullptr)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

CudaStream::CudaStream(CudaStream&& other_stream) noexcept : stream_(other_stream.stream_)
{
    // Because move and move assigment in a raw pointer cudaStream_t will do a member-wise copy
    other_stream.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other_stream) noexcept
{
    if (this != &other_stream)
    {
        if (stream_ != nullptr)
        {
            // 1. Destroy the current existence
            cudaStreamDestroy(stream_);
        }
        // 2. Steal other_stream resource
        stream_ = other_stream.stream_;
        // 3. Set other_stream to nullptr
        other_stream.stream_ = nullptr;
    }
    return *this;
}

cudaStream_t CudaStream::get() const
{
    return stream_;
}

void CudaStream::synchronize() const
{
    cudaStreamSynchronize(stream_);
}
