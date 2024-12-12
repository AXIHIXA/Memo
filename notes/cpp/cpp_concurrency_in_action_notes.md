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
    // Creates a new std::thread object and associates it with a thread of execution. 
    // The new thread of execution starts executing (asynchronously):
    // INVOKE(decay_copy(forward<Func>(func)), decay_copy(forward<Args>(args)...));
    // where decay_copy(value) returns a decayed prvalue copy of value.

    // Note that when the constructor returns, this INVOKE might NOT have happened yet. 
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

    // oops 可能会在 buffer 转换成 std::string 前结束，导致悬垂指针。
    // t 的构造函数只会复制一份 buffer 指针，然后就返回了。
    // char * 到 std::string 的转换要等到 f 开始执行时
    // （即 t 的构造函数生成的线程真正开始执行、并调用了 f 时）才会发生，
    // 至于这时候 oops 返回了没有，那就只有神仙知道啦。
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

### 🌱 2.3 转移所有权

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

### 🌱 2.4 确定线程数量

- [std::thread::hardware_concurrency](https://en.cppreference.com/w/cpp/thread/thread/hardware_concurrency)
  - 返回并发线程的数量
  - 多核系统中，返回值可以是 CPU 核心的数量
  - 无法获取时，函数返回0
- 例子：并行版 accumulate
```c++
template <typename Iterator, typename T>
struct accumulate_block
{
    void operator()(Iterator first, Iterator last, T & result)
    {
        result = std::accumulate(first, last, result);
    }
};

template <typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init)
{
    unsigned long const length = std::distance(first, last);

    if (0 == length)  // 1
    {
        return init;
    }
    
    unsigned long const min_per_thread = 25;
    unsigned long const max_threads = 
        (length + min_per_thread - 1) / min_per_thread;  // 2

    unsigned long const hardware_threads =
        std::thread::hardware_concurrency();

    unsigned long const num_threads =  // 3
        std::min((hardware_threads != 0 ? hardware_threads : 2), max_threads);

    unsigned long const block_size = length / num_threads;  // 4

    std::vector<T> results(num_threads);

    // 因为在启动之前已经有了一个线程（主线程），所以启动的线程数比 num_threads 少 1
    std::vector<std::thread> threads(num_threads - 1);  // 5

    Iterator block_start = first;

    for (unsigned long i = 0; i < num_threads - 1; ++i)
    {
        Iterator block_end = block_start;
        std::advance(block_end, block_size);  // 6

        threads[i] = std::thread(  // 7
            accumulate_block<Iterator, T>(),
            block_start, block_end, std::ref(results[i])
        );

        block_start = block_end;  // 8
    }

    accumulate_block<Iterator,T>()(
        block_start, last, results[num_threads - 1]
    );  // 9
        
    for (auto & t : threads)
    {
        t.join();  // 10
    }
    
    return std::accumulate(results.begin(), results.end(), init); // 11
}
```

### 🌱 2.5 线程标识

- 线程标识为 [std::thread::id](https://en.cppreference.com/w/cpp/thread/thread/id) 类型，可以通过两种方式进行检索。
  - 第一种，可以通过调用 [std::thread::get_id](https://en.cppreference.com/w/cpp/thread/thread/get_id) 来直接获取。
    - 如果 `std::thread` 对象没有与任何执行线程相关联，`get_id` 将返回默认构造的 `std::thread` 的 `id` ，这个值表示“无线程”。
  - 第二种，当前线程中调用静态成员函数 [std::this_thread::get_id](https://en.cppreference.com/w/cpp/thread/get_id) 也可以获得线程标识。
- `std::thread::id` 对象支持拷贝、比大小、哈希、输出
  - 如果两个对象的 `std::thread::id` 相等，那就是同一个线程，或者都“无线程”。
  - 如果不等，那么就代表了两个不同线程，或者一个有线程，另一没有线程。
  - `std::thread::id` 可用作 associative container 的 key（有序无序均可）
- `std::thread::id` 常用作检测线程是否需要进行一些操作。
  - 比如,当用线程来分割一项工作，主线程可能要做一些与其他线程不同的工作
  - 启动其他线程前，可以通过 `std::this_thread::get_id()` 得到自己的线程 ID
  - 每个线程都要检查一下，其拥有的线程ID是否与初始线程的 ID 相同
  - 这是真 TM 像 fork 的返回值啊
```c++
std::thread::id master_thread;

void some_core_part_of_algorithm()
{
    if (std::this_thread::get_id() == master_thread)
    {
        do_master_thread_work();
    }

    do_common_work();
}
```

## 第三章 共享数据

- 数据竞争 Data Race
- 使用互斥锁（Mutex）保护数据
- 互斥锁的替代方案

### 🌱 3.1 数据竞争 Data Race

- 涉及到共享数据时，问题就是因为共享数据的**修改**所导致
  - 如果共享数据只读，那么不会影响到数据，更不会对数据进行修改，所有线程都会获得同样的数据
  - 但当一个或多个线程要修改共享数据时，就会产生很多麻烦
- 最简单的办法就是对数据结构采用某种保护机制，确保只有修改线程才能看到**非原子操作的中间状态**
  - 从其他访问线程的角度来看，修改不是已经完成了，就是还没开始
  - C++ 标准库提供很多类似的机制，下面会逐一介绍

### 🌱 3.2 互斥锁 Mutex

- 编排代码来保护数据的正确性（见3.2.2节）
- 避免接口间的条件竞争（见3.2.3节）
- 互斥量也会造成死锁（见3.2.4节）
- 或对数据保护的太多（或太少）（见3.2.8节）

#### 📌 3.2.1 互斥锁 Mutex

- [std::mutex](https://en.cppreference.com/w/cpp/thread/mutex)
  - [std::mutex::lock](https://en.cppreference.com/w/cpp/thread/mutex/lock) 为上锁
  - [std::mutex::unlock](https://en.cppreference.com/w/cpp/thread/mutex/unlock) 为解锁
- RAII 模板类 [std::lock_guard](https://en.cppreference.com/w/cpp/thread/lock_guard)
  - 在构造时就能提供已锁的互斥量
  - 在析构时进行解锁
  - 保证了互斥量能被正确解锁
- 带有互斥锁的封装接口**不能**完全保护数据
  - 如果接口在持有锁期间返回了引用或指针，则这一引用或指针可以绕过锁
  - **谨慎设计接口**，要确保互斥量能锁住数据访问，并且**不留后门**
```c++
class Data
{
public:
    void push_back(int new_value)
    {
        std::lock_guard<std::mutex> guard(some_mutex);    // 3
        some_list.push_back(new_value);
    }

    bool contains(int value_to_find)
    {
        // 模板类参数推导 (since C++17) 
        // std::lock_guard 的模板参数列表可以省略
        std::lock_guard guard(some_mutex);    // 4
        return std::find(some_list.begin(), some_list.end(), value_to_find) != some_list.end();
    }

    std::list<int> & oops()
    {
        // 返回值可以绕过互斥锁修改 some_list！
        std::lock_guard g(some_mutex);  // 5
        return list;
    }

private:
    std::list<int> some_list;    // 1
    std::mutex some_mutex;    // 2
};
```

#### 📌 3.2.2 保护共享数据

- **切勿将受保护数据的指针或引用传递到互斥锁作用域之外**
```c++
class data_wrapper
{

public:
    template <typename Function>
    void process_data(Function func)
    {
        std::lock_guard<std::mutex> l(m);
        func(data);    // 1 传递“保护”数据给用户函数
    }

private:
    some_data data;
    std::mutex m;
};

data_wrapper x;

some_data * unprotected;

void malicious_function(some_data & protected_data)
{
    unprotected = &protected_data;
}

void oops()
{
    x.process_data(malicious_function);    // 2 恶意函数绕过锁留下了后门
    unprotected->do_something();    // 3 在无保护的情况下访问保护数据
}
```

#### 📌 3.2.3 接口间的条件竞争

- 考虑一个栈，**即使 top 和 pop 各自内部都加了锁，这两个接口间依旧存在竞争**
  - 如下，两个线程同时操作一个栈
  - 接口内部各自加锁，而不是整个栈上加锁
    - 效率稍稍好点
    - 但只能保证同一时刻只有一个接口被访问
    - 单个线程的多次访问的间隙中，可能与其他线程的操作互相交叠
  - 当接口访问顺序如注释时，4 处将产生未定义行为
    - 注意如果没有先 empty 后 top/pop 的间隙，是不会产生条件竞争的
  - 这是一个经典的条件竞争
    - *使用互斥量对栈内部数据进行保护，但依旧不能阻止条件竞争的发生*
    - 这是 *接口固有的问题*
```c++
std::stack<int> stk;
stk.emplace(1);

void thread_one()
{
    if (!stk.empty())  // 1 此时栈里有一个数
    {
        // 2 注意执行到这里时，线程一是没有锁的，线程二可以在此处横插一脚！
        std::cout << stk.top() << '\n';  // 5 此时，栈空了，未定义行为
    }
}

void thread_two()
{
    if (!stk.empty())  // 3 此时栈里依旧有一个数
    {
        stk.pop();  // 4 弹出栈顶，这回栈空了
    }
}
```
- 比起上面的 *未定义行为 UB Undefined Behavior*，下面由于条件竞争产生的 *重复处理数据* 的 BUG 更加隐秘、难以排查：
```c++
std::stack<int> stk;
if (!s.empty())  // 1
{    
    const int value = s.top();  // 2
    s.pop();  // 3
    do_something(value);
}
```
- 表3.1 一种可能执行顺序：**同一个栈顶被处理了两次**！

| Thread A                   | Thread B                   |
| -------------------------- | -------------------------- |
| if (!s.empty());           |                            |
|                            | if (!s.empty());           |
| const int value = s.top(); |                            |
|                            | const int value = s.top(); |
| s.pop();                   |                            |
| do_something(value);       | s.pop();                   |
|                            | do_something(value);       |

- 注释：为什么 C++ STL `std::stack::pop` 不返回被弹出的元素？（为了异常安全）
  - 假设有一个 `std::stack<std::vector<int>>`
  - `std::vector` 的拷贝构造函数可能会抛出一个 `std::bad_alloc` 异常
  - 当 `pop` 函数将栈顶弹出并返回“弹出值”时，会有一个潜在的问题
    - `pop` 函数：首先用栈顶创建一个临时量，然后弹出栈顶，最后临时量拷贝到返回值
    - 如果最后一步**拷贝抛出异常**，就会导致它的确从栈上移出了，但是接收却失败了
    - 要**弹出的数据将会丢失**
  - `std::stack` 的设计人员将这个操作分为两部分：`top` 和 `pop`
    - 这样，在不能安全的将元素拷贝出去的情况下，栈中的这个数据还依旧存在，没有丢失
  - 但这个设计在并发环境中**引入了条件竞争**！
- `top` `pop` 拆分带来的**条件竞争如何解决**？有多种解决方案，但都有代价：
  - 接口之间的**条件竞争**问题：
    - 将 `top` `pop` 合并为一个函数，不要拆分，只有一个接口，自然就没有条件竞争了
  - 拷贝操作的**异常安全**问题：
    - 避免拷贝，而是返回指针或者修改入参
      - 返回 `shared_ptr`：代价是动态内存分配的额外开销
      - 修改入参：代价是栈内元素的类型需要支持默认构造，且默认构造的开销也可以很大
    - 使用 `noexcept` 的拷贝或移动构造函数（不是所有数据结构都支持这个）
```c++
struct empty_stack : public std::exception
{
    const char * what() const noexcept
    {
        return "empty stack";
    }
};

template <typename T>
class threadsafe_stack
{
public:
    threadsafe_stack() = default;

    threadsafe_stack(const threadsafe_stack & other)
    {
        std::lock_guard lock(other.m);
        data = other.data; // 在构造函数体中的执行拷贝
    }

    threadsafe_stack & operator=(const threadsafe_stack &) = delete; // 赋值操作被删除

    void push(T new_value)
    {
        std::lock_guard lock(m);
        data.push(new_value);
    }
  
    std::shared_ptr<T> pop()
    {
        std::lock_guard lock(m);

        if (data.empty())
        {
            throw empty_stack(); // 在调用pop前，检查栈是否为空
        }
        
        std::shared_ptr<T> const res = std::make_shared<T>(data.top()); // 在修改堆栈前，分配出返回值
        data.pop();

        return res;
    }

    void pop(T & value)
    {
        std::lock_guard lock(m);

        if (data.empty())
        {
            throw empty_stack();
        }
        
        value = data.top();
        data.pop();
    }

    bool empty() const
    {
        std::lock_guard lock(m);
        return data.empty();
    }

private:
    std::stack<T> data;
    mutable std::mutex m;
};
```

#### 📌 3.2.4 死锁：问题描述及解决方案

- [std::lock](https://en.cppreference.com/w/cpp/thread/lock) 可以一次性锁住多个（两个以上）的互斥量，并且没有死锁风险（**不建议裸着用**）
```c++
template <class Lockable1, class Lockable2, class ... LockableN>
void lock(Lockable1 & lock1, Lockable2 & lock2, LockableN & ... lockn);
```
- 用例
```c++
class some_big_object;

void swap(some_big_object & lhs, some_big_object & rhs)
{
    if (&lhs == &rhs)
    {
        return;
    }
      
    std::lock(lhs.m, rhs.m); // 1
    std::lock_guard<std::mutex> lock_a(lhs.m, std::adopt_lock); // 2 adopt_lock：假设构造时已经预先占用了锁
    std::lock_guard<std::mutex> lock_b(rhs.m, std::adopt_lock); // 3
    swap(lhs.some_detail, rhs.some_detail);
}
```
- [std::scoped_lock](https://en.cppreference.com/w/cpp/thread/scoped_lock) 提供 RAII 封装，标准建议用这个
```c++
explicit scoped_lock(MutexTypes & ... m);
scoped_lock(std::adopt_lock_t, MutexTypes & ... m);
scoped_lock(const scoped_lock &) = delete;
```
- 上面用例改为：
```c++
void swap(some_big_object & lhs, some_big_object & rhs)
{
    if (&lhs == &rhs)
    {
        return;
    }

    std::scoped_lock guard(lhs.m, rhs.m); // 1
    swap(lhs.some_detail, rhs.some_detail);
}
```

#### 📌 3.2.5 避免死锁的进阶指导

- 避免嵌套锁
  - 最简单的：线程获得一个锁时，就别再去获取第二个。
  - 每个线程只持有一个锁，就不会产生死锁。
  - 当需要获取多个锁，使用 `std::lock` 上锁，避免产生死锁。
- 避免在持有锁时调用外部代码
  - 因为代码是外部提供的，所以没有办法确定外部要做什么。
  - 外部程序可能做任何事情，包括获取锁。
  - 在持有锁的情况下，如果用外部代码要获取一个锁，就会违反第一个指导意见，并造成死锁。
- 使用固定顺序获取锁
  - 当硬性要求获取两个或两个以上的锁，并且不能使用 `std::lock` 单独上锁时，最好在每个线程上，用固定的顺序获取锁
- **使用层次锁结构**
  - 如将一个 `hierarchical_mutex` 实例进行上锁，那么只能获取更低层级实例上的锁，这就会对代码进行一些限制。
  - 层级互斥量不可能死锁，因为互斥量本身会严格遵循约定进行上锁。
```c++
hierarchical_mutex high_level_mutex(10000); // 1
hierarchical_mutex low_level_mutex(5000);  // 2
hierarchical_mutex other_mutex(6000); // 3

void low_level_func()
{
    std::lock_guard<hierarchical_mutex> lk(low_level_mutex); // 4
    do_low_level_stuff();
}

void high_level_func()
{
    std::lock_guard<hierarchical_mutex> lk(high_level_mutex); // 6
    low_level_func();
    do_high_level_stuff(); // 5
}

void thread_a()  // 7 遵守规则
{
    high_level_func();
}

void thread_b() // 8 无视规则，因此在运行时会失败
{
    // 9 锁了 6000 级的 other_mutex，禁止获取更高级的锁
    std::lock_guard<hierarchical_mutex> lk(other_mutex); 

    // 10 试图锁 10000 级的 high_level_mutex，抛出异常
    high_level_func();  

    do_other_stuff();
}
```
- 层级锁的实现
```c++
class hierarchical_mutex
{
public:
    explicit hierarchical_mutex(unsigned long value)
            : hierarchy_value(value)
            , previous_hierarchy_value(0)
    {
        
    }

    void lock()
    {
        check_for_hierarchy_violation();
        internal_mutex.lock();  // 4
        update_hierarchy_value();  // 5
    }

    void unlock()
    {
        if (this_thread_hierarchy_value != hierarchy_value)
        {
            throw std::logic_error("mutex hierarchy violated");  // 9
        }
            
        this_thread_hierarchy_value = previous_hierarchy_value;  // 6
        internal_mutex.unlock();
    }

    bool try_lock()
    {
        check_for_hierarchy_violation();

        if (!internal_mutex.try_lock())  // 7
        {
            return false;
        }
            
        update_hierarchy_value();

        return true;
    }

private:
    std::mutex internal_mutex;
  
    unsigned const long hierarchy_value;
    unsigned long previous_hierarchy_value;

    static thread_local unsigned long this_thread_hierarchy_value;  // 1

    void check_for_hierarchy_violation()
    {
        if (this_thread_hierarchy_value <= hierarchy_value)  // 2
        {
            throw std::logic_error("mutex hierarchy violated");
        }
    }

    void update_hierarchy_value()
    {
        previous_hierarchy_value = this_thread_hierarchy_value;  // 3
        this_thread_hierarchy_value = hierarchy_value;
    }
};

// 使用了 thread_local 的值来代表当前线程的层级值。
// 初始化为最大值，所以最初所有线程都能被锁住。
// 因为声明中有 thread_local，所以每个线程都有其副本，这样线程中变量状态完全独立，
// 当从另一个线程进行读取时，变量的状态也完全独立。
thread_local unsigned long hierarchical_mutex::this_thread_hierarchy_value(ULONG_MAX);  // 8
```

#### 📌 3.2.6 [std::unique_lock](https://en.cppreference.com/w/cpp/thread/unique_lock) 灵活的锁

- [std::unique_lock](https://en.cppreference.com/w/cpp/thread/unique_lock)
  - `std::lock_guard` 只是 minimal RAII 管理器，没有任何其他功能
    - `std::lock_guard` 只有构造函数和析构函数，不支持其他操作
    - `std::unique_lock` 支持 `std::lock_guard` 的全部特性，包括析构时自动释放占有的锁
  - `std::unique_lock` 更灵活，支持更多的构造时拿锁策略
    - 可将 `std::adopt_lock` 作为第二个参数传入构造函数，对互斥量进行管理
    - 可将 `std::defer_lock` 作为第二个参数传入构造函数，表明互斥量应保持解锁状态
  - `std::unique_lock` 完全适配普通互斥锁对象所有的操作，比如传给 `std::lock`
  - `std::unique_lock` 会占用**更多空间**，并且比 `std::lock_guard` 稍**慢**一些
```c++
void swap(X & lhs, X & rhs)
{
    if (&lhs == &rhs)
    {
        return;
    }
        
    std::unique_lock<std::mutex> lock_a(lhs.m, std::defer_lock); // 1 
    std::unique_lock<std::mutex> lock_b(rhs.m, std::defer_lock); // 1 std::defer_lock 留下未上锁的互斥量
    std::lock(lock_a, lock_b); // 2 互斥量在这里上锁
    swap(lhs.some_detail, rhs.some_detail);
}
```

#### 📌 3.2.7 `std::unique_lock` 的传递

- 和 `std::unique_ptr` 类似，被传递的对象如果是右值（或不会被拷贝），则不需要显式地 `std::move`，否则需要
- 一个例子，一个函数获取锁，并将所有权转移给调用者
```c++
std::unique_lock<std::mutex> get_lock()
{
    extern std::mutex some_mutex;
    std::unique_lock<std::mutex> lk(some_mutex);
    preprocess();

    // 1：NRVO，无需 move
    return lk;
}

void process_data()
{
    // 2：get_lock 里的 lock 实际上直接构造在了这一行
    std::unique_lock<std::mutex> lk(get_lock());
    do_something();
}
```

### 🌱 3.3 多线程下保护共享数据的其他方式

#### 📌 3.3.1 保护共享数据的初始化过程

- [std::once_flag](https://en.cppreference.com/w/cpp/thread/once_flag)
  - `std::call_once` 的辅助结构，用于传参
  - 只有一个默认构造函数，默认初始化为尚未调用
  - 不可拷贝、不可移动
- [std::call_once](https://en.cppreference.com/w/cpp/thread/call_once)
  - 使用 `std::call_once` 比显式使用互斥量消耗的资源更少，特别是当初始化完成后
```c++
template <class Callable, class ... Args>
void call_once(std::once_flag & flag, Callable && f, Args && ... args);
```
- 例子：Lazy Initialization
```c++
std::shared_ptr<some_resource> resource_ptr;
std::once_flag resource_flag;  // 1

void init_resource()
{
    resource_ptr.reset(new some_resource);
}

void foo()
{
    std::call_once(resource_flag, init_resource);  // 可以线程安全地进行且仅进行一次初始化
    resource_ptr->do_something();
}
```
- 例子：Meyer's Singleton：线程安全 since C++11
  - C++11 开始，局部静态对象的初始化及定义完全在一个线程中发生，并且没有其他线程可在初始化完成前对其进行处理
  - 在只需要一个全局实例情况下，这是 `std::call_once` 的一个替代方案
```c++
class my_class;

// 多线程可以安全的调用，不用为数据竞争而担心
my_class & get_my_class_instance()
{
    static my_class instance;  // 线程安全的初始化过程
    return instance;
}
```

#### 📌 3.3.2 读写锁：保护不常更新的数据结构

- [std::shared_mutex](https://en.cppreference.com/w/cpp/thread/shared_mutex)
  - 更高的性能优势
  - 更少的操作方式
- [std::shared_timed_mutex](https://en.cppreference.com/w/cpp/thread/shared_timed_mutex)
  - 支持更多操作方式
- 读锁定：
  - [std::shared_lock](https://en.cppreference.com/w/cpp/thread/shared_lock)
- 写锁定：
  - [std::lock_guard](https://en.cppreference.com/w/cpp/thread/lock_guard)
  - [std::unique_lock](https://en.cppreference.com/w/cpp/thread/unique_lock)
- 示例：DNS Cache
```c++
class dns_entry;

class dns_cache
{
public:
    std::optional<dns_entry> find(const std::string & domain) const
    {
        std::shared_lock<std::shared_mutex> lk(entry_mutex);  // 1 读锁
        const Entries::const_iterator it = entries.find(domain);
        return (it == entries.end()) ? std::nullopt : it->second;
    }

    void update(const std::string & domain, const dns_entry & detail)
    {
        std::lock_guard<std::shared_mutex> lk(entry_mutex);  // 2 写锁
        entries[domain] = detail;
    }

private:
    using Entries = std::unordered_map<std::string, dns_entry>;
    Entries entries;
    mutable std::shared_mutex entry_mutex;
};
```

#### 📌 3.3.3 嵌套锁

- [std::recursive_mutex](https://en.cppreference.com/w/cpp/thread/recursive_mutex)
  - 线程对已经获取的 `std::mutex` 再次上锁是错误的，尝试这样做会导致未定义行为
  - 其他线程对互斥量上锁前，当前线程必须释放拥有的所有锁，所以如果你调用 `lock` 三次，也必须调用 `unlock` 三次
  - `std::lock_guard<std::recursive_mutex>` 和 `std::unique_lock<std::recursive_mutex>` 可以帮你处理这些问题

## 第四章 同步操作

- C++ 标准库提供了一些工具可用于同步，形式上表现为
  - 条件变量 [std::condition_variable](https://en.cppreference.com/w/cpp/thread/condition_variable)
  - [std::future](https://en.cppreference.com/w/cpp/thread/future)
- 并发技术规范中，为future添加了非常多的操作，并可与新工具一起使用：
  - [std::latch](https://en.cppreference.com/w/cpp/thread/latch) ：轻量级锁资源
  - [std::barrier](https://en.cppreference.com/w/cpp/thread/barrier)

### 🌱 4.1 等待事件或条件

- 等待的平凡实现
```c++
bool flag;
std::mutex m;

void wait_for_flag()
{
    std::unique_lock<std::mutex> lk(m);

    while (!flag)
    {
        lk.unlock();  // 1 解锁互斥量
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 2 休眠100ms
        lk.lock();   // 3 再锁互斥量
    }
}
```

#### 📌 4.1.1 等待条件达成

- C++标准库对条件变量有两套实现，两者都需要与互斥量一起才能工作（互斥量是为了同步）：
  - [std::condition_variable](https://en.cppreference.com/w/cpp/thread/condition_variable)
    - 仅能与 [std::mutex](https://en.cppreference.com/w/cpp/thread/mutex) 一起工作
  - [std::condition_variable_any](https://en.cppreference.com/w/cpp/thread/condition_variable_any)
    - 可以和任何自定义的互斥量一起工作
    - 在性能和系统资源的使用方面会有**更多开销**
- [std::condition_variable::wait](https://en.cppreference.com/w/cpp/thread/condition_variable/wait)
```c++
/// 调用前，unique_lock 中已经占有了一个锁。
/// 首先，释放 unique_lock 中占有的锁，并阻塞。
/// 之后，会被 notify 唤醒，或被 伪唤醒。
/// 被唤醒后，重新上锁（这一步可能又会阻塞），成功上锁后返回。
void wait(std::unique_lock<std::mutex> & lock);

/// 等价于：while (!pred()) { wait(lock); }
template <class Predicate>
void wait(std::unique_lock<std::mutex> & lock, Predicate pred);
```
- **伪唤醒** Spurious Wakeup
  - 调用 `wait` 的过程中，在互斥量锁定时，可能会去检查条件变量若干次，当提供测试条件的函数返回 `true` 就会立即返回。
  - 当等待线程重新获取互斥量并检查条件变量时，并非直接响应另一个线程的通知，就是所谓的 *伪唤醒*（Spurious Wakeup）。
  - 任意的函数和可调用对象，不仅限于 lambda 表达式，都可以传入 `wait` 用于 *检测伪唤醒并重新等待* 。
  - 因为任何伪唤醒的数量和频率都是不确定的，所以**不建议**使用有 *副作用* 的函数做条件检查。
```c++
template <typename Predicate>
void minimal_wait(std::unique_lock<std::mutex> & lk, Predicate pred)
{
    while (!pred())
    {
        lk.unlock();
        lk.lock();
    }
}
```
- 代码 4.1 使用 `std::condition_variable` 处理数据等待的生产者——消费者模型
```c++
std::mutex data_mut;
std::queue<Data> data_queue;  // 1
std::condition_variable data_cond;

void produce()
{
    while (!should_exit())
    {
        const Data data = produce_data();
        std::lock_guard<std::mutex> lk(data_mut);
        data_queue.push(data);  // 2
        data_cond.notify_one();  // 3
    }
}

void consume()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> lk(mut);  // 4
            data_cond.wait(lk, [] { return !data_queue.empty(); });  // 5

            Data data = data_queue.front();
            data_queue.pop();
        }  // 6

        comsume(data);

        if (is_last_chunk(data))
        {
            break;
        }
    }
}
```

#### 📌 4.1.2 构建线程安全队列

- 线程安全的队列：
  - 支持多线程**并行 push 和 pop，无需额外加锁**
  - 传统的 `front` 和 `pop` 之间**即使各自上锁，依旧存在条件竞争**，需要合并为一个函数
```c++
template <typename T>
class threadsafe_queue
{
public:
    threadsafe_queue() = default;

    threadsafe_queue(const threadsafe_queue & other)
    {
        std::lock_guard<std::mutex> lk(other.mut);
        data_queue = other.data_queue;
    }

    bool empty() const
    {
        // 因为其他线程可能有非 const 引用对象，并调用变种成员函数，所以这里有必要对互斥量上锁。
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    }

    void push(T new_value)
    {
        std::lock_guard<std::mutex> lk(mut);
        data_queue.push(new_value);
        data_cond.notify_one();
    }

    wait_and_pop(T & value)
    {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        value = data_queue.front();
        data_queue.pop();
    }

    std::shared_ptr<T> wait_and_pop()
    {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        std::shared_ptr<T> res = std::make_shared<T>(data_queue.front());
        data_queue.pop();
        return res;
    }

    bool try_pop(T & value)
    {
        std::lock_guard<std::mutex> lk(mut);

        if (data_queue.empty())
        {
            return false;
        }
        
        value = data_queue.front();
        data_queue.pop();
        return true;
    }

    std::shared_ptr<T> try_pop()
    {
        std::lock_guard<std::mutex> lk(mut);

        if (data_queue.empty())
        {
            return nullptr;
        }
        
        std::shared_ptr<T> res = std::make_shared<T>(data_queue.front());
        data_queue.pop();
        return res;
    }

private:
    std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;
};

threadsafe_queue<data_chunk> data_queue;  // 1

void data_preparation_thread()
{
    while (more_data_to_prepare())
    {
        const data_chunk data = prepare_data();
        data_queue.push(data);  // 2
    }
}

void data_processing_thread()
{
    while (true)
    {
        data_chunk data;
        data_queue.wait_and_pop(data);  // 3
        process(data);

        if (is_last_chunk(data))
        {
            break;
        } 
    }
}
```

### 🌱 4.2 使用 [std::future](https://en.cppreference.com/w/cpp/thread/future)

- 线程需要等待特定事件的结果（例如异步任务的返回值）
  - 之后，线程会周期性地等待或检查事件是否触发，检查期间也会执行其他任务。
  - 另外，等待任务期间也可以先执行另外的任务，直到对应的任务触发，而后等待 `future` 的状态会变为就绪状态。
  - `future` 一旦就绪，这个 `future` 就不能重置了。
- `std::thread` 执行的任务不能有返回值
  - [std::async](https://en.cppreference.com/w/cpp/thread/async) 启动一个异步任务，会返回一个 `std::future<V>` 对象
  - `V = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args> ...>;`
  - `policy` 是一个 bitmask，`enum launch { async, deferred };`
    - `std::launch::async`：开一个新线程执行任务。
    - `dstd::launch::eferred`：Lazy evaluation，直到 future 被 wait 或 get 时，才在同一线程内求值。
    - 不带 `policy` 的版本，默认 `async | deferred`，即哪个都行，C++ 标准建议实现在有空余算力时采用 `async`。
```c++
template <class F, class ... Args>
std::future<V> async(F && f, Args && ... args );

template <class F, class ... Args>
std::future<V> async(std::launch policy, F && f, Args && ... args);
```
- [std::future](https://en.cppreference.com/w/cpp/thread/future)
  - 只能与指定事件相关联，类似于 `unique_ptr`
  - 与数据无关的 `future`，可以使用 `std::future<void>`
- [std::shared_future](https://en.cppreference.com/w/cpp/thread/shared_future)
  - 能关联多个事件，类似于 `shared_ptr`
  - 与数据无关的，用 `std::shared_future<void>`
- 代码4.6 `std::future` 从异步任务中获取返回值
```c++
int find_the_answer(int, int &, std::unique_ptr<int>);
void do_other_stuff();

void foo()
{
    int a = 1, b = 2, c = 3;
    std::future<int> the_answer = std::async(
            std::launch::async,
            find_the_answer, 
            a, std::ref(b), std::make_unique<int>(c)
    );
    do_other_stuff();
    std::cout << "The answer is " << the_answer.get() << std::endl;
}
```

#### 📌 4.2.2 [std::packaged_task]() ：[std::future](https://en.cppreference.com/w/cpp/thread/future) 与任务关联




















## 

### 🌱 

#### 📌 


## 

### 🌱 

#### 📌 A Simple Example

