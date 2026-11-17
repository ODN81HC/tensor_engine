#include <gtest/gtest.h>

#include "thread_pool.hpp"

TEST(ThreadPoolTest, EnqueueAndExecuteTask) {
    std::atomic<int> counter{0};
    constexpr int num_tasks = 100;
    {
        ThreadPool thread_pool(4);
        for (int i = 0; i < num_tasks; ++i) {
            thread_pool.enqueue([&counter]() { 
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                counter++; });
        }
    }
    EXPECT_EQ(counter.load(), num_tasks);
}