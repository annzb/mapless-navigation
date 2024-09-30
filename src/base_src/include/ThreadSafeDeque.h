#include <mutex>
#include <condition_variable>
#include <deque>
#include <stdexcept>

#ifndef THREADSAFEDEQUE_H
#define THREADSAFEDEQUE_H


template<typename T>
class ThreadSafeDeque
{
public:

  ThreadSafeDeque();
  void setTimeout(int timeout_ms);
  bool getItem(T &item);
  void addItem(T &item);
  int size();
  bool empty();
  T& front();
  T& back();
  void pop_front();
  void pop_back();
  T& operator[](int i);

private:
  std::deque<T> deque_;
  std::mutex deque_mtx_;
  std::condition_variable cv_;
  std::chrono::milliseconds timeout_;
};

template<typename T>
ThreadSafeDeque<T>::ThreadSafeDeque()
{
  timeout_ = std::chrono::milliseconds(500);
}

template<typename T>
void ThreadSafeDeque<T>::setTimeout(int timeout_ms)
{
  timeout_ = std::chrono::milliseconds(timeout_ms);
}

template<typename T>
bool ThreadSafeDeque<T>::getItem(T& item)
{
  std::unique_lock<std::mutex> lock(deque_mtx_);
  if (!cv_.wait_for(lock,
    timeout_,
    [this]{return !empty();}))
  {
    throw std::runtime_error("waiting for new item has timed out");
    return false;
  }
  item = deque_.back();
  deque_.pop_back();
  return true;
}

template<typename T>
void ThreadSafeDeque<T>::addItem(T& item)
{
  {
    std::lock_guard<std::mutex> lock(deque_mtx_);
    deque_.push_front(item);
  }
  cv_.notify_one();
}

template<typename T>
int ThreadSafeDeque<T>::size()
{
  return deque_.size();
}

template<typename T>
bool ThreadSafeDeque<T>::empty()
{
  return deque_.empty();
}

template<typename T>
T& ThreadSafeDeque<T>::front()
{
  std::lock_guard<std::mutex> lock(deque_mtx_);
  return deque_.front();
}

template<typename T>
T& ThreadSafeDeque<T>::back()
{
  std::lock_guard<std::mutex> lock(deque_mtx_);
  return deque_.back();
}

template<typename T>
void ThreadSafeDeque<T>::pop_front()
{
  std::lock_guard<std::mutex> lock(deque_mtx_);
  deque_.pop_front();
}

template<typename T>
void ThreadSafeDeque<T>::pop_back()
{
  std::lock_guard<std::mutex> lock(deque_mtx_);
  deque_.pop_back();
}

template<typename T>
T& ThreadSafeDeque<T>::operator[](int i)
{
  std::lock_guard<std::mutex> lock(deque_mtx_);
  return deque_[i];
}

#endif
