#include "thread_pool.h"

ThreadPool::ThreadPool(int num_threads) : threads_(num_threads), stop_(false) {
  for (int i = 0; i < num_threads; i++) {
    threads_[i].thread =
        std::thread([this](int tid) { this->InternalLoop(tid); }, i);
  }
}

void ThreadPool::InternalLoop(int tid) {
  while (1) {
    std::unique_lock<std::mutex> lock(threads_[tid].mutex);
    threads_[tid].cv.wait(lock, [this, tid]() {
      return this->stop_.load() || !this->threads_[tid].tasks.empty();
    });
    if (stop_.load()) {
      return;
    }
    if (!threads_[tid].tasks.empty()) {
      auto task = threads_[tid].tasks.front();
      threads_[tid].tasks.pop_front();

      task();
    }
  }
}

ThreadPool::~ThreadPool() {
  stop_.store(true);
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i].cv.notify_all();
  }
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i].thread.join();
  }
}