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

### 🌱 2.1 线程的基本操作

#### 📌 2.1.1 启动线程

- 线程在 [std::thread](https://en.cppreference.com/w/cpp/thread/thread) 对象创建时启动
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
- 线程对象析构前必须先 [join](https://en.cppreference.com/w/cpp/thread/thread/join) 或者 [detach](https://en.cppreference.com/w/cpp/thread/thread/detach)，否则析构函数会调用 [terminate](https://en.cppreference.com/w/cpp/error/terminate) **终止整个程序**
- join：汇入
  - 调用者阻塞住，直到这个线程执行完毕
  - 确保线程在主函数完成前结束
- detach：分离
  - 调用者不阻塞，调用者所在线程和这个线程完全脱钩
  - **注意线程数据的生命周期**
    - 线程对象引用父线程的局部数据时要额外注意
    - 父线程的局部数据 go out of scope 后被销毁
    - 而这个线程此时可能还没执行完，就会访问到悬垂引用
```c++
void oops()
{
    int local_state = 0;

    std::thread t([
        local_state_ref = &local_state]  // 1 潜在访问隐患：空引用
    {  
        for (int i = 0; i < 1000000; ++i)
        {
            do_something(local_state_ref);
        }
    });

    t.detach();                          // 2 不等待线程结束
}                                        // 3 新线程可能还在运行
```

#### 📌 2.1.2 join

- 使用 join 等待线程
  - 使用 join 的场景
    - 原始线程有自己的工作要做
    - 原始线程启动多个子线程来做并行做一些有用的工作
    - 原始线程需要等待这些线程结束（例如获取运算结果）
- 想对等待中的线程有更灵活的控制，则需其他机制辅助，比如condition_variable和future
  - 比如：看一下某个线程是否结束，或者只等待一段时间，超过时间就判定为超时
- 调用 join，还可以清理线程相关的内存，这样 `std::thread` 对象将不再与已经完成的线程有任何关联
  - 只能对一个线程使用一次 join，一旦使用过 join，`std::thread` 对象就不能再次汇入了
  - 当对其使用 joinable 时，将返回 false

#### 📌 2.1.3 线程函数抛出异常导致 join 被跳过

- `std::thread` 对象创建后必须 join 或 detach
  - detach：可以在线程启动后，直接使用 detach 进行分离
  - join：需要细心挑选使用 join 的位置
    - 线程运行后若产生异常，则会在 join 调用之前 throw，这样就会**跳过join**！
- **避免线程函数异常导致线程没有join**
  - 在无异常的情况下使用join时，需要**在异常处理过程中调用join**，从而避免生命周期的问题
```c++
struct func;

void f()
{
    int some_local_state = 0;

    func my_func(some_local_state);
    std::thread t(my_func);

    try
    {
        do_something_in_current_thread();
    }
    catch (...)
    {
        t.join();  // 1：一旦进入这个 catch block，下面的 join 就会被跳过！
        throw;
    }
    
    t.join();  // 2
}
```
- RAII Thread Guard：引用线程对象，线程对象析构时，如果还没 join，则自动 join
```c++
class thread_guard
{
public:
    // 这里是引用，不会拷贝线程对象
    explicit thread_guard(std::thread & t_) : t(t_) {}

    ~thread_guard()
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    thread_guard(thread_guard const &) = delete;
    thread_guard & operator=(thread_guard const &) = delete;

    // A user-defined destructor 
    // implicitly surpasses generation of 
    // move constructor and move assignment operator. 

private:
    // 这里是引用，不会拷贝线程对象
    std::thread & t;
};
```

#### 📌 2.1.4 后台运行线程

- 使用 detach 会让线程在后台运行，这就意味着与主线程不能直接交互
  - 被 detach 的线程永远无法再被 `std::thread` 对象引用到
  - C++运行库保证，当线程退出时，相关资源的能够正确回收
- detach 掉的线程通常称为*守护线程*（daemon threads）

### 🌱 2.2 传递参数

- 传递参数：
  - 将参数作为 `std::thread` 构造函数的附加参数即可
  - `std::thread`的 **[构造函数](https://en.cppreference.com/w/cpp/thread/thread/thread)无视函数参数类型，盲目地拷贝已提供的变量**
    - 这些参数会被拷贝至新线程的内存空间中，同临时变量一样
    - 即使函数中的参数是引用的形式，拷贝操作也会执行
    - 被**拷贝的参数会以右值的方式传递**，以兼容只支持移动语义的参数类型
- [含参数版构造函数](https://en.cppreference.com/w/cpp/thread/thread/thread)的实现：
```c++
template <class Func, class ... Args> 
thread(Func && func, Args && ... args)
{
    do_something();

    // decay_copy(value) returns a decayed prvalue copy of value.
    INVOKE(decay_copy(forward<Func>(func)), decay_copy(forward<Args>(args)...));
}
```
- 注：[decay-copy](https://en.cppreference.com/w/cpp/standard_library/decay-copy)
```c++
/// Returns a decayed prvalue copy of value. 
/// Ensures that arguments are decayed when passing-by-value. 
/// decay-copy always materializes value and produces a copy. 
template <class T>
typename std::decay<T>::type decay-copy(T && value)
{
    // Implicitly converted to the decayed type. 
    return std::forward<T>(value);
}
```
- **线程函数的参数必须注意生命周期问题，严禁传入局部对象自动存储期的对象！**
```c++
// 这两种 f 都有问题：
void f(int i, std::string const & s);
void f(int i, std::string s);

void oops(int some_param)
{
    char buffer[1024];
    sprintf(buffer, "%i", some_param);

    // oops 可能会在 buffer 转换成 std::string 前结束，导致悬垂指针
    std::thread t(f, 3, buffer);
    t.detach();
}

// 解决办法：
void correct(int some_param)
{
    // 直接构造临时 string 对象，不依赖临时对象。
    // C++ 标准规定，std::to_string(some_param) 的求值一定先于 t1 的构造，
    // 因此一定先于 correct 返回，是合法的。
    std::thread t1(f, 3, std::to_string(some_param));
    t1.detach();

    // 等价于上面一条
    std::string str = std::to_string(some_param);
    std::thread t2(f, 3, str);
    t2.detach();
    
    // 显式地将 buffer 的内容分配到堆上
    auto buffer = std::make_shared<std::string>(std::to_string(some_param));
    std::thread t3(f, 3, *buffer);
    t3.detach();

    // 使用 std::async 的延迟启动机制，自动管理线程对象的生命周期
    std::async(std::launch::async, f, 3, std::to_string(some_param));
}
```
- 线程函数的参数如果是左值引用，则不能给线程对象参数传临时量，会报编译错误：
  - 也就是说，想接收引用但最后复制了一份对象的情况**不会**发生。
```c++
void update_data_for_widget(widget_id w, widget_data & data);

void oops_again(widget_id w)
{
    widget_data data;
    std::thread t(update_data_for_widget, w, data);  // 编译错误：左值引用不能绑定到右值上
    display_status();
    t.join();
    process_widget_data(data);
}
```

#### 📌 2.3 转移所有权

- `std::thread` 对象**可以移动，但不能拷贝**
  - 移动包括移动语义和 swap 成员函数
  - 移动的目标 `std::thread` 对象**不能**已经关联了实际的线程，不然程序直接会 `terminate`
- `std::thread` 对象的传参和返回
  - 传参和返回的操作参考 [std::unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr) 即可
  - 只能移动的类型的局部实例可以作为函数返回值，不需要额外的 `move`
  - 注意函数返回值类型一定不能是右值引用
```c++
std::thread f()
{
    void some_function();

    // 返回 std::thread 临时量，OK
    // 这是在直接构造一个 std::thread，不是拷贝，OK
    return std::thread(some_function);
}

std::thread g()
{
    void some_other_function(int);

    // Copy/Move Elision: 
    // https://en.cppreference.com/w/cpp/language/copy_elision
    // 在某些情况下，编译器可以或必须省略拷贝或移动，即使拷贝构造函数或移动构造函数是非平凡的：
    // 
    // - 返回值优化（Named Return Value Optimization NRVO）【可以】
    //   当函数返回一个局部变量（*），编译器直接构造返回值在调用者的存储位置，无需拷贝或移动；
    //   （*）：
    //   有名字（没名字的直接适用下一条）、非 volatile 的自动存储期对象，
    //   且不是函数参数、不是 handler（catch 语句括号里的东西），
    //   且类型和返回值类型相同（不考虑 cv 限定）。
    //
    // - 纯右值（prvalue）的推迟实例化（since C++17）【必须】
    //   纯右值直到被显式使用为止都不会被实例化，当被显式使用时，编译器直接在调用点构造对象，无需拷贝或移动。
    std::thread t(some_other_function, 42);

    // 返回 std::thread 局部实例，OK
    // 按拷贝初始化的定义，这句 return statement 要触发拷贝初始化的：
    // https://en.cppreference.com/w/cpp/language/copy_initialization
    // 但前面提到的 RVO 使得 t 没有被构造在上一行，而是直接构造在了函数外面接收返回值的地方。
    return t;
}
```
```c++
void f(std::thread t);

void g()
{
    void some_function();

    // 传参 std::thread 临时量，OK
    // 这是在直接构造一个 std::thread，不是拷贝，OK
    f(std::thread(some_function));

    // 传参 std::thread 局部对象，则必须 move
    // 没有这个 move 就要触发拷贝啦
    std::thread t(some_function);
    f(std::move(t));
}
```
- Scoped Thread：RAII Joining Thread：移动传入的线程，而不是创建一个新的
```c++
class scoped_thread
{
public:
    // 必须移动初始化，没有这个 move 就成拷贝初始化啦
    explicit scoped_thread(std::thread t_) : t(std::move(t_))  // 1
    {
        if (!t.joinable())  // 2
        {
            throw std::logic_error("No thread");
        }
    }

    ~scoped_thread()
    {
        t.join();  // 3
    }

    scoped_thread(scoped_thread const &) = delete;
    scoped_thread & operator=(scoped_thread const &) = delete;

private:
    // 注意，这里不是引用啦
    std::thread t;
};

struct func;

void f()
{
    int some_local_state;
    scoped_thread t(std::thread(func(some_local_state)));  // 4
    do_something_in_current_thread();
}  // 5
```
- Joining Thread
```c++
class joining_thread
{
public:
    joining_thread() noexcept = default;

    template <typename Callable, typename ... Args>
    explicit joining_thread(Callable && func, Args && ... args)
            : t(std::forward<Callable>(func), std::forward<Args>(args)...)
    {}

    explicit joining_thread(std::thread t_) noexcept
            : t(std::move(t_))
    {}
    
    joining_thread(joining_thread && other) noexcept
            : t(std::move(other.t))
    {}

    joining_thread & operator=(joining_thread && other) noexcept
    {
        if (joinable())
        {
            join();
        }

        t = std::move(other.t);
        return *this;
    }

    joining_thread & operator=(std::thread other) noexcept
    {
        if (joinable())
        {
            join();
        }
            
        t = std::move(other);
        return *this;
    }

    ~joining_thread() noexcept
    {
        if (joinable())
        {
            join();
        }
    }

    void swap(joining_thread & other) noexcept
    {
        t.swap(other.t);
    }

    std::thread::id get_id() const noexcept
    {
        return t.get_id();
    }

    bool joinable() const noexcept
    {
        return t.joinable();
    }

    void join()
    {
        t.join();
    }

    void detach()
    {
        t.detach();
    }

    std::thread & as_thread() noexcept
    {
        return t;
    }

    const std::thread & as_thread() const noexcept
    {
        return t;
    }

private:
    std::thread t;
};
```
- 使用容器量产并管理线程
```c++
void do_work(unsigned id);

void f()
{
    std::vector<std::thread> threads;

    for (unsigned i = 0; i < 20; ++i)
    {
        threads.emplace_back(do_work, i);  // 产生线程
    } 

    for (auto & entry : threads)  // 对每个线程调用 join()
    {
        entry.join(); 
    }
}
```






## 

### 🌱 

#### 📌 


## 

### 🌱 

#### 📌 A Simple Example

