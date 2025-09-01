#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
 private:
  using Task = std::function<void()>;

  struct Thread {
    std::thread thread;
    std::deque<Task> tasks;
    std::mutex mutex;
    std::condition_variable cv;
  };

  std::vector<Thread> threads_;

  std::atomic_bool stop_;
  std::atomic<int64_t> choice_;

  void InternalLoop(int tid);

 public:
  ThreadPool(int num_threads);

  inline int num_threads() { return threads_.size(); }

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> Enqueue(
      F&& f, Args&&... args) {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> future = task->get_future();
    {
      int tid = choice_.fetch_add(1) % threads_.size();
      std::unique_lock<std::mutex> lock(threads_[tid].mutex);
      threads_[tid].tasks.push_back([task]() { (*task)(); });
      threads_[tid].cv.notify_all();
    }

    return future;
  }

  ~ThreadPool();
};

#endif