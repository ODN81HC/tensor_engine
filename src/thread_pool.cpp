#include "thread_pool.hpp"
#include <iostream>
#include <stdexcept>

ThreadPool::ThreadPool(size_t num_threads)
{
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this](const std::stop_token& stoken) {
            while (!stoken.stop_requested()) {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this]() {
                    return !tasks_.empty() || stop_;
                });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                auto task_item = std::move(tasks_.front());
                tasks_.pop();
                // Unlock the mutex before running the task
                lock.unlock();
                // Execute the task outside the lock
                task_item();
            }
        });
    };
}

ThreadPool::~ThreadPool()
{
    stop_ = true;
    cv_.notify_all();
    // Join all worker threads to ensure they have finished execution before destruction.
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::enqueue(std::function<void()> task)
{
    const std::unique_lock<std::mutex> lock(queue_mutex_);
    // Push the task to tasks_
    tasks_.push(std::move(task));
    cv_.notify_one();
}
