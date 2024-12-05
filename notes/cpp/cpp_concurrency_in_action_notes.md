# _C++ Concurrency In Action Second Edition_ 笔记

- Notes of reading [_C++ Concurrency In Action Second Edition_](https://github.com/xiaoweiChen/CPP-Concurrency-In-Action-2ed-2019).
- A good Chinese localization is available at [HERE](https://github.com/xiaoweiChen/CPP-Concurrency-In-Action-2ed-2019/).



## 第一章 C++ 并发概述

### 🌱 1.1 何谓并发

- 硬件角度
  - 真正并行：N 核 N 线程
  - 任务切换：超标量 
- 进程线程角度
  - 多进程并发
    - 独立的进程之间通过信号、socket、文件、管道、共享内存、消息队列等方式传递信息
    - 进程开销大
  - 多线程并发
    - 单进程中多线程
- 并发（Concurrency）与并行（Parallelism）
  - 并发：更注重性能
  - 并行：更关注算法，“程序和算法的（可）并行性”
  - 对多线程来说，这两个概念是重叠的

### 🌱 1.3 并发和多线程

- C++11/14/17 标准对并发的支持
  - 管理线程（参见第二章）
  - 保护共享数据（参见第三章）
  - 线程间同步操作（参见第四章）
  - 原子操作（参见第五章）
  - 一整套的并行算法（参见第十章）



## 第二章 线程管理

- 主要内容
  - 启动新线程
  - join 和 detach
  - 唯一标识符 handle

### 🌱 2.1 线程的基本操作

#### 📌 2.1.1 启动线程

- 线程在[std::thread](https://en.cppreference.com/w/cpp/thread/thread)对象创建时启动
```c++
void do_some_task() {}
std::thread t1(do_some_task);

struct background_task { void operator()() {} };
std::thread t2(background_task());  // triggers most vexing parse!
std::thread t3((background_task()));
std::thread t4 {background_task()};

std::thread t5([]
{
    do_something();
    do_something_else();
});
```
- 线程对象析构前必须先[join](https://en.cppreference.com/w/cpp/thread/thread/join)或者[detach](https://en.cppreference.com/w/cpp/thread/thread/detach)，否则析构函数会调用[terminate](https://en.cppreference.com/w/cpp/error/terminate)终止整个程序

























## Chapter 1 -- Hello, World of Concurrency in C++!

### 🌱 [Usage Guide](https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py)

#### 📌 A Simple Example


## Chapter 1 -- Hello, World of Concurrency in C++!

### 🌱 [Usage Guide](https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py)

#### 📌 A Simple Example

