# pragma once

#include<cstddef>
#include<vector>
#include<thread>
#include<queue>
#include<mutex>
#include<condition_variable>
#include<functional>

class ThreadPool 
{
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    void enqueue(std::function<void()> task);

private:
    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
};
