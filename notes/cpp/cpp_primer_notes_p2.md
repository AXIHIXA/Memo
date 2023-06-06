# *C++ Primer* Notes Part 2

- [Part 1](./cpp_primer_notes_p1.md)
- [**Part 2**](./cpp_primer_notes_p2.md)



### 🌱 [Chap 12] [动态内存管理](https://en.cppreference.com/w/cpp/memory)（Dynamic memory management）

- 程序中使用的对象都有严格的 *存储期* （生存期）
    - *全局对象* 
        - 程序启动时分配，结束时销毁
        - 存储于静态存储区（程序的静态内存）
    - *局部静态对象* 
        - 程序进入其所在的程序块时分配，离开该块时销毁
        - 存储于静态存储区（程序的静态内存）
    - *局部非静态对象* （ *自动对象* ）
        - 第一次使用前分配，程序结束时销毁
        - 存储于自动存储区（程序的栈内存）
    - *动态对象* 
        - 从被创建一直存在到被 *显式释放* 为止
            - *智能指针* 可以自动释放该被释放的对象
        - 存储于动态存储区（程序的堆内存）

#### 动态内存和智能指针（Dynamic memory and smart pointers）

- `C++`直接管理动态内存
- 动态申请内存：[`new`表达式](https://en.cppreference.com/w/cpp/language/new)（`new` expression, New expression）
  - Is **different** from [`operator new`](https://en.cppreference.com/w/cpp/memory/new/operator_new)
      - `operator new` is only for memory allocation, no object construction occurs
      - Size-unware versions of `operator new`s are preferred over size-aware versions (when both are present)
      - `new` expression first calls `operator new` to allocate memory, then calls the constructor to construct the object.
    ```c++
    struct MyStruct
    {
    public:
        static void * operator new(std::size_t count)
        {
            std::cout << __PRETTY_FUNCTION__ << ' ' << count << '\n';
            return ::operator new(count);
        }
    
        static void operator delete(void * ptr)
        {
            std::cout << __PRETTY_FUNCTION__ << '\n';
            return ::operator delete(ptr);
        }
    
    public:
        explicit MyStruct()
        {
            std::cout << __PRETTY_FUNCTION__ << '\n';
        }
    
        ~MyStruct()
        {
            std::cout << __PRETTY_FUNCTION__ << '\n';
        }
    
    public:
        int a {1};
        int b {2};
    };
        
    auto p = new MyStruct();
    delete p;
    
    // OUTPUT:
    // static void* MyStruct::operator new(std::size_t) 8
    // MyStruct::MyStruct()
    // MyStruct::~MyStruct()
    // static void MyStruct::operator delete(void*)
    ```
    - 初始化可以选择
      - *默认初始化* 
          - *不提供* 初始化器 
          - 对象的值 *未定义* 
      ```c++
      int * pi = new int;
      std::string * ps = new std::string;
      ```
      - *值初始化* 
          - 提供 *空的* 初始化器 
          - 如类类型没有合成的默认构造函数，则值初始化进行的也是默认初始化，没有意义
          - 对于内置类型，值初始化的效果则是 *零初始化* 
      ```c++
      std::string * ps1 = new std::string;   // default initialized to the empty string
      std::string * ps = new std::string();  // value initialized to the empty string
      int * pi1 = new int;                   // default initialized; *pi1 is undefined
      int * pi2 = new int();                 // value initialized to 0; *pi2 is 0
      ```
      - *直接初始化* 
          - 提供 *非空* 的初始化器 
          - 显式指定对象初值，可以使用 *括号* 或 *花括号* 初始化器
      ```c++
      int * pi = new int(1024);
      std::string * ps = new std::string(10, '9');
      std::vector<int> * pv = new std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      ```
    - 使用`auto`
      - 需提供 *初始化器* ，且初始化器中 *只能有一个值* 
          - 编译器需要从初始化器中推断类型
    ```c++
    auto p1 = new auto(obj);      // p points to an object of the type of obj
                                // that object is initialized from obj
    auto p2 = new auto{a, b, c};  // error: must use parentheses for the initializer
    ```
    - 动态分配`const`对象
      - 用`new`分配`const`对象是合法的，返回指向`const`的指针
      - 类似于其他`const`对象，动态分配的`const`对象亦必须进行初始化
          - 对于有 *默认构造函数* 的类类型，可以默认初始化
          - 否则，必须直接初始化
    ```c++
    // allocate and direct-initialize a const int
    const int * pci = new const int(1024);

    // allocate a default-initialized const empty string
    const std::string * pcs = new const std::string;
    ```
    - 内存耗尽
      - 无内存可用时，`new`会抛出`std::bad_alloc`异常，返回 *空指针*
      - 可以使用 *定位`new`* 表达式`new (std::nothrow)`（placement new）阻止抛出异常 => 19.1.2
          - 定位`new`本质作用是在指定地点`new`个东西出来，配合`std::allocator<T>`用的
    ```c++
    // if allocation fails, new returns a null pointer
    int * p1 = new int;                 // if allocation fails, new throws std::bad_alloc
    int * p2 = new (std::nothrow) int;  // if allocation fails, new returns a null pointer
    ```
    - `new` and `operator new`
    ```c++
    // allocates memory by calling: operator new(sizeof(MyClass))
    // and then constructs an object at the newly allocated space
    MyClass * p1 = new MyClass;
    
    // allocates memory by calling: operator new(sizeof(MyClass), std::nothrow)
    // and then constructs an object at the newly allocated space
    MyClass * p2 = new (std::nothrow) MyClass;
    
    // does not allocate memory; calls: operator new(sizeof(MyClass), p2)
    // but constructs an object at p2
    new (p2) MyClass;
    
    // Notice though that calling this function directly does not construct an object. 
    // allocates memory by calling: operator new(sizeof(MyClass))
    // but does not call MyClass's constructor
    MyClass * p3 = (MyClass *) ::operator new(sizeof(MyClass));
    ```
    - 动态释放内存：[`delete`表达式](https://en.cppreference.com/w/cpp/language/delete)（`delete` expression, Delete expression）
      - Still different from [`operator delete`](https://en.cppreference.com/w/cpp/memory/new/operator_delete)
          - `operator delete` just deallocates the memory, no object destruction is done
          - `delete` expression first calls destructor to destruct the object, 
            then calls `operator delete` to deallocate the memory. 
      - 传递给`delete`的指针必须是 *指向被动态分配的对象* 的指针或者 *空指针* 
      - 将同一个对象反复释放多次是 *未定义行为*
      - *`const`对象* 虽然不能更改，但却 *可以销毁* 
      - `delete`之后指针成为了 *空悬指针* （dangling pointer）
          - *你就是一个没有对象的野指针*
    ```c++
    int i; 
    int * pi1 = &i; 
    int * pi2 = nullptr;
    
    double * pd = new double(33); 
    double * pd2 = pd;
    
    delete i;    // error: i is not a pointer
    delete pi1;  // undefined: pi1 refers to a local
    delete pd;   // ok
    delete pd2;  // undefined: the memory pointed to by pd2 was already freed
    delete pi2;  // ok: it is always ok to delete a null pointer    
    
    const int * pci = new const int(1024);
    delete pci;  // ok: free a const object 
    ```
    - 动态对象的生存期直到被释放时为止
      - `std::shared_ptr`管理的对象会在引用计数降为`0`时被自动释放
      - 内置类型指针管理的对象则一直存在到被显式释放为止
- *智能指针*
    - 定义于头文件`<memory>`中，包括 
        - [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr)：允许多个指针指向同一个对象
        - [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr)： *独占* 指向的对象
        - [`std::weak_ptr`](https://en.cppreference.com/w/cpp/memory/weak_ptr)： *伴随类* ， *弱引用* ，指向`std::shared_ptr`所指向的对象
    - 行为类似于 *常规指针* ，但负责 *自动释放* 所指向的对象
        - 下文中的 *指针* 除非特别说明，都是指 *常规指针* 
    - *默认初始化* 的智能指针中保存着一个 *空指针* 
    - 智能指针使用方法与普通指针类似
        - *解引用* 返回对象 *左值* 
        - *条件判断* 中使用智能指针就是判断它 *是否为空* 
    - 智能指针使用规范
        1. **不**使用相同的内置指针初始化（或`reset`）多个智能指针，否则是 *未定义行为*
        2. **不**`delete`从智能指针`get()`到的内置指针
        3. **不**使用智能指针的`get()`初始化（或`reset`） *另一个* 智能指针
        4. 使用智能指针的`get()`返回的内置指针时，记住当最后一个对应的智能指针被销毁后，这个内置指针就 *无效* 了
        5. 使用内置指针管理的资源而不是`new`出来的内存时，记住传递给它一个 *删除器*
- 智能指针支持的操作
    - `std::shared_ptr`和`std::unique_ptr`都支持的操作
        - `p`：将`p`用作一个条件判断，若`p`指向一个对象，则为`true`
        - `*p`：解引用`p`，获得它指向的对象
        - `p->mem`：等价于`(*p).mem`
        - `p.get()`：返回`p`中保存的指针。若智能指针释放了其对象，则这一指针所指向的对象亦会失效
        - `std::swap(p, q)`：交换`p`和`q`中的指针
        - `p.swap(q)`：交换`p`和`q`中的指针
    - `std::shared_ptr`独有的操作
        - `std::shared_ptr<T> p`：定义一个 *空的* `std::shared_ptr<T>`
        - `std::shared_ptr<T> p(p2)`：`p`是`std::shared_ptr<T> p2`的拷贝。此操作会递增`p2`的引用计数。`p2`中的指针必须能被转换程`T *`
        - `std::shared_ptr<T> p(p2, d)`：`p`是`std::shared_ptr<T> p2`的拷贝。此操作会递增`p2`的引用计数。`p2`中的指针必须能被转换程`T *`。`p`将调用 *删除器* `d`来代替`delete`
        - `std::shared_ptr<T> p(u)`：`p`从`std::unique_ptr<T> u`处 *接管* 对象管辖权，将`u` *置空*
        - `std::shared_ptr<T> p(q)`：`p`管理内置指针`q`所指向的对象，`q`必须指向`new`分配的内存，且能够转换成`T *`类型
        - `std::shared_ptr<T> p(q, d)`：`p` *接管* 内置指针`q`所指向的对象的所有权，`q`能够转换成`T *`类型。`p`将调用 *删除器* `d`来代替`delete`
        - `std::make_shared<T>(args)`：返回一个`std::shared_ptr<T>`用`args`初始化
        - `p = q`：`p`和`q`都是`std::shared_ptr`，且保存的指针能够相互转换。此操作会递减`p`的引用计数、递增`q`的引用计数；若`p`的引用计数变为`0`，则将其管理的 *原内存释放* 
        - `p.use_count()`：返回`p`的 *引用计数* （与`p`共享对象的`std::shared_ptr`的数量）。 *可能很慢，主要用于调试* 
        - `p.unique()`：`return p.use_count() = 1;`
        - `p.reset()`：若`p`是唯一指向其对象的`std::shared_ptr`，则释放此对象，将`p` *置空*
        - `p.reset(q)`：若`p`是唯一指向其对象的`std::shared_ptr`，则释放此对象，令`p` *指向内置指针* `q`
        - `p.reset(q, d)`：若`p`是唯一指向其对象的`std::shared_ptr`，则 *调用`d`* 释放此对象，将`p` *置空*
    - `std::unique_ptr`独有的操作
        - `std::unique_ptr<T> u1`：定义一个 *空的* `std::unique_ptr<T>`，使用默认删除器`delete`
        - `std::unique_ptr<T> u1(q)`：`u1`管理内置指针`q`所指向的对象，`q`必须指向`new`分配的内存，且能够转换成`T *`类型
        - `std::unique_ptr<T, D> u2`：定义一个 *空的* `std::unique_ptr<T, D>`，`D` *删除器* 的类型
        - `std::unique_ptr<T, D> u(d)`：定义一个 *空的* `std::unique_ptr<T, D>`，`D` *删除器* 的类型，`d`为指定的 *删除器* 
        - `std::unique_ptr<T, D> u(q, d)`：`u1`管理内置指针`q`所指向的对象，`q`必须指向`new`分配的内存，且能够转换成`T *`类型；调用`D`类型 *删除器* `d`
        - [`std::make_unique`](https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique)：返回一个`std::unique_ptr<T>`，用`new(args)`初始化 `(since C++14)`
        - [`u1 = u2`](https://en.cppreference.com/w/cpp/memory/unique_ptr/operator%3D)：`u2`必须是 *右值* 。若`u2`是`std::unique_ptr &&`，则等价于`u1.reset(u2.release())`；`u2`还可以是自定义删除器的数组类型指针 `(since C++17)`；若`u2`是`nullptr`，`u1`将释放`u1`指向的对象，将`u1` *置空* 
            - `Clang-Tidy`：相比`u1.reset(u2.release())`，更推荐`u1 = std::move(u2)`
        - `u.release()`：`u` *放弃* 对指针的控制权，返回内置指针，并将`u` *置空* 。注意`u.release()`只是放弃所有权，并**没有**释放`u`原先指向的对象
        - `u.reset()`：释放指向`u`的对象，将`u` *置空*
        - `u.reset(q)`：释放指向`u`的对象，令`u` *指向内置指针* `q`。常见转移操作：`u1.reset(u2.release())`
            - `Clang-Tidy`：相比`u1.reset(u2.release())`，更推荐`u1 = std::move(u2)`
        - `u.reset(nullptr)`：释放指向`u`的对象，将`u` *置空*
    - `std::unique_ptr<T[]>`独有的操作
        - `std::unique_ptr<T[]> u`：定义一个 *空的* `std::unique_ptr<T[]>`，使用默认删除器`delete []`，可以指向动态分配的数组
        - `std::unique_ptr<T[]> u(q)`：`u`管理内置指针`q`所指向的动态分配的数组，`q`能够转换成`T *`类型
        - `u[i]`：返回`u`拥有的数组中的第`i`个元素
        - **不支持**`->`和`.`
        - 其他 *不变* 
    - `std::weak_ptr`支持的操作
        - `std::weak_ptr<T> w`：定义一个 *空的* `std::weak_ptr<T>`
        - `std::weak_ptr<T> w(sp)`：与`std::shared_ptr sp`指向相同对象的`std::weak_ptr`，`T`必须能转换成`sp`指向的类型
        - `w = p`：`p`可以是`std::shared_ptr`或者`std::weak_ptr`，赋值后`w`和`p` *共享* 对象
        - `w.reset()`：将`w` *置空* （**并不**释放对象）
        - `w.use_count()`：与`w`共享对象的`std::shared_ptr`的数量
        - `w.expired()`：`return w.use_count() == 0;`
        - `w.lock()`：（线程安全）如果`w.expired() == true`，则返回一个 *空的* `std::shared_ptr`；否则，返回一个指向`w`对象的`std::shared_ptr`
- `std::shared_ptr`
    - 智能指针使用方法与普通指针类似
        - *解引用* 返回对象 *左值* 
        - *条件判断* 中使用智能指针就是判断它 *是否为空* 
    ```
    std::shared_ptr<std::string> p1;
    if (p1 && p1->empty()) *p1 = "hi";
    ```
    - `std::make_shared`函数
        - 最安全的分配和使用动态内存的方法
        - 在动态内存中分配一个对象并 *用其参数构造对象* ，返回指向该对象的`shared_ptr`
            - 就类似与顺序容器的`c.emplace(args)`
            - 不提供任何参数就是 *值初始化* 对象
        ```
        std::shared_ptr<int>         p3 = std::make_shared<int>(42);                     // int 42
        std::shared_ptr<std::string> p4 = std::make_shared<std::string>(10, '9');        // std::string "9999999999"
        std::shared_ptr<int>         p5 = std::make_shared<int>();                       // int 0 (value initialized)
        auto                         p6 = std::make_shared<std::vector<std::string>>();  // std::vector<std::string>
        ```
    - `p.reset()` 函数   
        - 示例1
        ```
        p = new int(1024);                 // error: cannot assign a pointer to a shared_ptr
        p.reset(new int(1024));            // ok: p points to a new object
        ```
        - 和赋值类似，`p.reset`会更新引用计数，可能会释放掉对象。`p.reset`常常和`p.unique()`一起使用，来控制多个`std::shared_ptr`之间共享的对象。在改变底层对象之前，我们检查自己是否是当前对象仅有的用户。如果不是，在改变之前需要制作一份新的拷贝
        ```
        if (!p.unique())
        {
            p.reset(new std::string(*p));  // we aren't alone; allocate a new copy
        }

        *p += newVal;                      // now that we know we're the only pointer, okay to change this object
        ```
    - `std::shared_ptr`拷贝和赋值
        - 每个`std::shared_ptr`都有其 *引用计数* （reference count），记录有多少个其他`std::shared_ptr`指向相同的对象
            - *拷贝* 时，引用计数会 *递增* ，例如
                - 用一个`std::shared_ptr`初始化另一个`std::shared_ptr`
                - 将`std::shared_ptr`作为参数传递给一个函数
                - 将`std::shared_ptr`作为函数返回值
            - *赋值* 或 *销毁* 时，引用计数会 *递减* ，例如
                - 局部的`std::shared_ptr`离开其作用域时
            - 一旦`std::shared_ptr`的引用计数降为`0`，它就会 *自动释放* 自己所管理的对象
        ```
        auto p = std::make_shared<int>(42);   // object to which p points has one user
        auto q(p);                            // p and q point to the same object
                                              // object to which p and q point has two users
                                             
        auto r = std::make_shared<int>(42);   // int to which r points has one user assign to r, 
                                              // making it point to a different address
                                              // increase the use count for the object to which q points
                                              // reduce the use count of the object to which r had pointed
                                              // the object r had pointed to has no users; 
                                              // that object is automatically freed
        ```
    - `std::shared_ptr` *自动销毁* 所管理的对象
        - 销毁工作通过调用对象的 *析构函数* （destructor）来完成
            - 析构函数一般负责释放该对象所占用的资源
        - `std::shared_ptr`的析构函数会递减它所指向的对象的引用计数
            - 降为`0`后就会销毁对象并释放占用的内存
        - 如果将`std::shared_ptr`存放于容器中，而后不再需要全部元素，要使用`c.erase`删除不再需要的元素
        - 如果两个对象 *共享底层数据* ，则某个对象被销毁时，**不能**单方面地销毁底层数据
        ```
        std::vector<std::string> v1;                           // empty vector
        
        {                                                      // new scope
            std::vector<std::string> v2 = {"a", "an", "the"};
            v1 = v2;                                           // copies the elements from v2 into v1
        }                                                      // v2 is destroyed, which destroys the elements in v2
                                                               // v1 has three elements, 
                                                               // which are copies of the ones originally in v2
        ```
        - 工厂例程
        ```
        std::shared_ptr<Foo> factory(T arg)
        {
            return std::make_shared<Foo>(arg);      // shared_ptr will take care of deleting this memory, ++ref_cnt
        }                                           // goes out of scope; however the memory remains
        
        void use_factory(T arg)
        {
            std::shared_ptr<Foo> p = factory(arg);
            // do something...                      // use p...
        }                                           // p goes out of scope; 
                                                    // the memory to which p points is AUTOMATICALLY freed
        ```
- `std::shared_ptr`和`new`结合使用
    - 可以使用`new`的返回值初始化`std::shared_ptr`
        - 接受指针参数的智能指针构造函数是`explicit`的，因此，必须直接初始化，而**不能**将内置指针隐式转化为智能指针
            - 类似的，返回智能指针的函数也不能在其返回语句中隐式转换普通指针
        - 用来初始化智能指针的普通指针必须指向动态内存
            - 因为智能指针默认使用`delete`释放对象
            - 如果绑定到其他指针上，则必须自定义释放操作 => 12.1.4
    ```
    std::shared_ptr<int> p0;                  // shared_ptr that can point at a int

    std::shared_ptr<int> p1 = new int(1024);  // error: must use direct initialization
    std::shared_ptr<int> p2(new int(1024));   // ok: uses direct initialization
    
    std::shared_ptr<int> clone(int p) 
    {
        // error: implicit conversion to shared_ptr<int>
        return new int(p); 
    }

    std::shared_ptr<int> clone(int p) 
    {
        // ok: explicitly create a shared_ptr<int> from int*
        return std::shared_ptr<int>(new int(p));
    }
    ```
    - **不要**混用智能指针和内置指针
        - `std::shared_ptr`可以调节对象的析构，但这仅限于其自身的拷贝（即`std::shared_ptr`）之间
            - 这也是我们推荐使用`make_shared<T>(args)`而不是`new`的原因
        - 混用这俩玩意可能导致该释放的没释放，或者内置指针指向的对象被`std::shared_ptr`释放了
    ```
    // ptr is created and initialized when process is called
    void process(std::shared_ptr<int> ptr)
    {
        // use ptr
    } // ptr goes out of scope and is destroyed
        
    std::shared_ptr<int> p(new int(42));  // reference count is 1
    process(p);                           // copying p increments its count; 
                                          // in process the reference count is 2
    int i = *p;                           // ok: reference count is 1
    
    int * x(new int(1024));               // dangerous: x is a plain pointer, not a smart pointer
    process(x);                           // error: cannot convert int* to shared_ptr<int>
    process(shared_ptr<int>(x));          // legal, but the memory will be deleted!
    int j = *x;                           // undefined: x is a dangling pointer!
    ```
    - **不要**用智能指针的`get`方法初始化 *另一个* 智能指针或者为智能指针赋值
        - 智能指针的`get`方法设计用途是向不能用智能指针的代码传递一个内置指针
        - 使用此指针的代码自然**不能**`delete`此指针
        - 两个独立的`std::shared_ptr` 绑定到同一块内存上是 *未定义行为*
    ```
    std::shared_ptr<int> p(new int(42));  // reference count is 1
    int * q = p.get();                    // ok: but don't use q in any way 
                                          // that might delete its pointer
    
    {                                     // new block
        shared_ptr<int>(q);               // undefined: two independent shared_ptrs point to the same memory
    }                                     // block ends, q is destroyed, 
                                          // and the memory to which q points is freed
    
    int foo = *p;                         // undefined; the memory to which p points was freed
    ```
- [Guideline: How to pass smart pointers](https://herbsutter.com/2013/06/05/gotw-91-solution-smart-pointer-parameters/):
    - **Don’t** pass a smart pointer as a function parameter unless you want to use or manipulate the smart pointer itself, 
      such as to share or transfer ownership.
    - Prefer passing objects by value, `*`, or `&`, **not** by smart pointer.
    - Express a “sink” function using a by-value `std::unique_ptr` parameter.
    - Use a non-`const` `std::unique_ptr &` parameter only to modify the `std::unique_ptr`.
    - **Don’t** use a `const std::unique_ptr &` as a parameter; use `Widget *` instead.
    - Express that a function will store and share ownership of a heap object using a by-value `std::shared_ptr` parameter.
    - Use a non-`const` `std::shared_ptr &` parameter only to modify the `std::shared_ptr`. 
      Use a `const std::shared_ptr &` as a parameter only if 
      you’re not sure whether you’ll take a copy and share ownership; 
      otherwise use `Widget *` instead (or if not nullable, a `Widget &`).
- 智能指针和异常
    - 即使程序出现异常、过早结束，智能指针也能确保内存被释放
        - 与之相对的，直接管理的内存不会被释放
    - *删除器* （deleter）
        - 用于自定义析构智能指针管理的对象的方法
        - `std::shared_ptr<T>`的删除器接受一个`T *`类型的内置指针
    ```
    struct destination;                 // represents what we are connecting to
    struct connection;                  // information needed to use the connection
    connection connect(destination *);  // open the connection
    void disconnect(connection);        // close the given connection
    
    void f(destination & d /* other parameters */)
    {
        // get a connection; must remember to close it when done
        connection c = connect(&d);
        // use the connection
        // if we forget to call disconnect before exiting f, there will be no way to close c
    }

    void end_connection(connection * p) { disconnect(*p); }

    void f(destination & d /* other parameters */)
    {
        connection c = connect(&d);
        std::shared_ptr<connection> p(&c, end_connection);
        // use the connection
        // when f exits, even if by an exception, the connection will be properly closed
    }
    ```
- `std::unique_ptr`
    - *拥有* 自己指向的对象
    - 同一时刻只能有一个`std::unique_ptr`指向一个给定对象，被销毁时其指向的对象也立即被销毁
    - 没有`make_unique`函数，想要显式指定初值只能传入`new`出来的内置指针进行 *直接初始化* 
    ```
    std::unique_ptr<double> p1;                     // unique_ptr that can point at a double
    std::unique_ptr<int> p2(new int(42));           // p2 points to int with value 42
    ```
    - 虽然**不能**拷贝或赋值`std::unique_ptr`，但可以通过调用`u1.reset(u2.release())`将指针的所有权从一个非`const` `std::unique_ptr`转移至另一个
    ```
    std::unique_ptr<std::string> p1(new string("Stegosaurus"));
    std::unique_ptr<std::string> p2(p1);            // error: no copy for unique_ptr
    std::unique_ptr<std::string> p3;
    p3 = p2;                                        // error: no assign for unique_ptr
    
    // transfers ownership from p1 (which points to the string Stegosaurus) to p2
    std::unique_ptr<std::string> p2(p1.release());  // release makes p1 null
    std::unique_ptr<std::string> p3(new string("Trex"));
    
    // transfers ownership from p3 to p2
    p2.reset(p3.release());                         // reset deletes the memory to which p2 had pointed
    ```
    - 调用`u.release()`会切断`u`和它原来所管理的对象之间的关系，返回值通常用来初始化另一个智能指针，或者给另一个智能指针赋值
        - 只是管理内存的责任转移了而已，如果不保存`release`的返回值，那这块内存可就真的永生了
    ```
    p2.release();                                   // WRONG: p2 won't free the memory and we've lost the pointer
    auto p = p2.release();                          // ok, but we must remember to delete(p)
    ```
    - `std::unique_ptr`用于传参与返回
        - *不能拷贝* `std::unique_ptr` 这一规则有一个例外：`std::unique_ptr`将亡值的拷贝和赋值操作编译器都默认成 *移动* 操作 => 13.6.2
        - 最常见的例子就是从函数返回一个`std::unique_ptr`
        ```
        std::unique_ptr<int> clone(int p) 
        {
            // ok: explicitly create a unique_ptr<int> from int *
            return std::unique_ptr<int>(new int(p));
        }
        ```
        - 还可以返回一个局部对象的拷贝
        ```
        std::unique_ptr<int> clone(int p) 
        {
            std::unique_ptr<int> ret(new int (p));
            // . . .
            return ret;
        }
            ```
    - 向`std::unique_ptr`传递 *删除器* 
        - `std::unique_ptr`管理 *删除器* 的方式和`std::shared_ptr`不同 => 16.1.6
    ```
    void f(destination & d /* other needed parameters */)
    {
        // open the connection
        connection c = connect(&d);
        
        // when p is destroyed, the connection will be closed
        std::unique_ptr<connection, decltype(end_connection)*> p(&c, end_connection);
        
        // use the connection
        
        // when f exits, even if by an exception, the connection will be properly closed
    }
    ```
- `std::weak_ptr`
    - `std::weak_ptr`指向`std::shared_ptr`管理的对象，但**不影响**`std::shared_ptr`的 *引用计数*
        - `std::weak_ptr` 不控制被管理对象的生存期
        - 一旦该对象最后一个`std::shared_ptr`被销毁，即使还有`std::weak_ptr`指向该对象，该对象还是会被销毁
        - *弱* 共享对象
    - 创建`std::weak_ptr`时，要用`std::shared_ptr`初始化
    ```
    auto p = std::make_shared<int>(42);
    std::weak_ptr<int> wp(p);            // wp weakly shares with p; use count in p is unchanged
    ```
    - 由于对象 *可能不存在* ， *必须* 调用`wp.lock()`访问对象，而**不能**直接解引用
    ```
    if (std::shared_ptr<int> np = wp.lock()) 
    { 
        // true if np is not null
        // inside the if, np shares its object with p
    }
    ```    

#### 动态数组（Dynamic arrays）

- `C++`语言和标准库提供了 *两种* 一次分配一个 *对象数组* 的方法
    1. [`new`表达式](https://en.cppreference.com/w/cpp/language/new)
    2. [`allocator`类](https://en.cppreference.com/w/cpp/memory/allocator)
        - 将分配和初始化分离
        - 更好的性能和更灵活的内存管理能力
- 大多数应用都应该使用 *标准库容器* 而**不是**动态分配的数组。使用容器更为简单，更不容易出现内存管理错误，并且可能有更好的性能
- [`new`表达式](https://en.cppreference.com/w/cpp/language/new)和数组
    - 在`new`表达式的类型名之后跟一对方括号，其中指明要分配的对象的数目
        - 数目必须是 *整形* ，但 *不必是常量*
        - 成功后返回指向 *第一个* 对象的指针
        - 也可以使用数组类型的 *类型别名*
    ```
    int * pia = new int[get_size()];  // pia points to the first of these ints
    
    typedef int intarr42_t1[42];      // intarr42_t1 names the type array of 42 ints
    int * p1 = new intarr42_t1{};     // allocates an array of 42 ints; p1 points to the first one
    
    using intarr42_t2 = int [42];     // intarr42_t2 names the type array of 42 ints
    int * p2 = new IntArr42_t2{};     // allocates an array of 42 ints; p2 points to the first one
    ```
    - 分配一个数组会得到一个元素类型的指针，分配的内存**不是**数组类型
        - **不能**对动态数组调用`std::begin()`或`std::end()`
        - **不能**使用范围`for`遍历动态数组
    - 初始化动态分配对象的数组
        - *默认初始化* 
            - *不提供* 初始化器 
            - 对象的值 *未定义* 
        ```
        int * pia = new int[10];
        std::string * psa = new std::string[10];
        ```
        - *值初始化* 
            - 提供 *空的* 初始化器 
            - 如类类型没有合成的默认构造函数，则值初始化进行的也是默认初始化，没有意义
            - 对于内置类型，值初始化的效果则是 *零初始化* 
        ```
        int * pia = new int[10]();
        std::string * ps = new std::string[10]();
        ```
        - *聚合初始化* 
            - 提供 *非空* 的初始化器 
            - 显式指定对象初值，可以使用 *花括号* 初始化器
            - 初始化器数目小于元素数目时，剩余元素将进行 *值初始化* 
            - 初始化器数目大于元素数目时，`new`表达式抛出`std::bad_array_new_length`异常，**不会**分配任何内存
        ```
        int * pia = new int[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::string * ps = new std::string[10]{"a", "an", "the", std::string(3, 'x')};
        ```
        - 由于不能通过 *直接初始化* 提供单一初始化器，**不能**用`auto`分配数组
    - 动态分配 *空数组* 是合法的
        - 与之相对，**不能**创建大小为`0`的静态数组对象
        - 创建出的指针
            - **不能**解引用
            - 保证和`new`返回的任何其他指针都 *不同* 
            - 可以像使用 *尾后迭代器* 一样使用它
    ```
    char arr[0];              // error: cannot define a zero-length array
    char * cp = new char[0];  // ok: but cp can't be dereferenced
    
    size_t n = get_size();    // get_size returns the number of elements needed
    int * p = new int[n];     // allocate an array to hold the elements
    
    for (int * q = p; q != p + n; ++q)  // don't go into loop if n == 0
    {
        // process the array... 
    }
    ```
    - 释放动态数组：`delete []`表达式
        - 数组中的元素按照 *逆序* 被销毁
        - 销毁动态数组是使用普通的`delete`是 *未定义行为*
        - 销毁普通动态对象时使用`delete []`同样是 *未定义行为*
            - 以上两条编译器很可能还没有`warning`，那可真是死都不知道怎么死的了
        - 只要内存 *实际上* 是动态数组，就必须使用`delete []`
            - 包括使用 *类型别名* 定义的动态数组
    ```
    typedef int intarr42_t[42];        // intarr42_t names the type array of 42 ints
    int * p = new intarr_42_t{};       // allocates an array of 42 ints; p points to the first one
    delete[] p;                        // brackets are necessary because we allocated an array
    ```
    - 智能指针和动态数组
        - 标准库提供可以管理`new T[]`分配的数组的`std::unique_ptr<T[]>`版本
            - 自动销毁时，会自动调用`delete []`
        - 这种`std::unique_ptr<T[]>`提供的操作与普通`std::unique_ptr`稍有不同
            - `std::unique_ptr<T[]> u`：定义一个 *空的* `std::unique_ptr<T[]>`，使用默认删除器`delete []`，可以指向动态分配的数组
            - `std::unique_ptr<T[]> u(q)`：`u`管理内置指针`q`所指向的动态分配的数组，`q`能够转换成`T *`类型
            - `u[i]`：返回`u`拥有的数组中的第`i`个元素
            - **不能**使用 *成员访问运算符* （ *点运算符* `.`和 *箭头运算符* `->`）
                - 这俩货对数组没意义
            - 其他操作 *不变* 
        ```c++
        // up points to an array of ten uninitialized ints
        std::unique_ptr<int[]> up(new int[10]);
        
        for (size_t i = 0; i != 10; ++i)
        {
            up[i] = i;  // assign a new value to each of the elements
        }
        
        // automatically uses delete[] to destroy its pointer
        up.release();             
        ```
        - `std::shared_ptr`**不**直接支持动态数组，
            - 如果一定要用`std::shared_ptr`，则需自行提供 *删除器* 
                - 这种情况下不提供删除器是 *未定义行为*
            - *智能指针类型* **不**支持下标运算符、**不**支持指针算术运算
                - 必须使用`sp.get()`获取内置指针进行访问
        ```c++
        // to use a shared_ptr we must supply a deleter
        std::shared_ptr<int> sp(new int[10], [] (int *p) { delete[] p; });

        // shared_ptrs don't have subscript operator and don't support pointer arithmetic
        for (size_t i = 0; i != 10; ++i)
        {
            *(sp.get() + i) = i; // use get to get a built-in pointer
        }
        
        // uses the lambda we supplied that uses delete[] to free the array
        sp.reset();     
        ```
- [`allocator`类](https://en.cppreference.com/w/cpp/memory/allocator)
    - *内存分配* 解耦 *对象初始化* 
        - `new`将内存的分配和对象的初始化绑定
        - `delete`将内存的释放和对象的析构绑定
        - 对于单个对象这无可厚非，但对于动态数组我们则需要在内存上 *按需构造对象*
            - 否则将造成不必要的浪费（对象先被初始化，之后又被重复赋值）
                - 比如下面的例子，`p`中每个`std::string`都先被默认初始化，之后又被赋值
            - 且没有默认构造函数的类类型干脆就不能动态分配数组了
        ```c++
        std::string * const p = new std::string[n];  // construct n empty strings
        std::string s;
        std::string * q = p;                         // q points to the first string
        
        while (cin >> s && q != p + n)
        {
            *q++ = s;                                // assign a new value to *q
        }
            
        const size_t size = q - p;                   // remember how many strings we read
        
        // use the array
        
        delete[] p;                                  // p points to an array; 
                                                     // must remember to use delete[]
        ```
    - 标准库`std::allocator`类定义于`<memory>`中
        - 将 *内存分配* 和 *对象构造* 分离开
        - `std::allocator`是一个 *模板* ，定义时需指明将分配的对象类型
        - `std::allocotor<T>`的 *对象* 分配 *未构造的内存* 时，它将根据`T`的类型确定 *内存大小* 和 *对齐位置*
        ```c++
        // default allocator for ints
        std::allocator<int> alloc;

        // demonstrating the few directly usable members
        int * p = alloc.allocate(1);  // space for one int
        alloc.deallocate(p, 1);       // and it is gone

        // Even those can be used through traits though, so no need
        using traits_t = std::allocator_traits<decltype(alloc)>;
        p = traits_t::allocate(alloc, 1);
        traits_t::construct(alloc, p, 7);   // construct the int
        std::cout << *p << '\n';            // 7
        traits_t::deallocate(alloc, p, 1);  // dealloocate space for one int
        ```
        ```c++
        // default allocator for strings
        std::allocator<std::string> alloc;
        using traits_t = std::allocator_traits<decltype(alloc)>;
 
        // Rebinding the allocator using the trait for strings gets the same type
        traits_t::rebind_alloc<std::string> alloc_ = alloc;
 
        std::string * p = traits_t::allocate(alloc, 2);  // space for 2 strings
 
        traits_t::construct(alloc, p, "foo");
        traits_t::construct(alloc, p + 1, "bar");
 
        std::cout << p[0] << ' ' << p[1] << '\n';        // boo far
 
        traits_t::destroy(alloc, p + 1);
        traits_t::destroy(alloc, p);
        traits_t::deallocate(alloc, p, 2);
        ```
    - 标准库`std::allocator`类
        - `std::allocator<T> a`：定义一个`std::allocator<T>`类型对象`a`，用于为`T`类型对象分配 *未构造的内存*
        - 构造使用
          - Member functions:
            - [`a.allocate(n)`](https://en.cppreference.com/w/cpp/memory/allocator/allocate)：分配一段能保存`n`个`T`类对象的 *未构造的内存* ，返回`T *`.
                - Calls `::operator new(n)` (which in turn calls `std::malloc(std::size_t)`), but how and when to call is unspecified
            - [`a.deallocate(p, n)`](https://en.cppreference.com/w/cpp/memory/allocator/deallocate)：释放`T * p`开始的内存，这块内存保存了`n`个`T`类型对象。
                - `p`必须是先前由`a.allocate(n)`返回的指针，且`n`必须是之前所要求的大小。
                - 调用`a.deallocate(p, n)`之前，这块内存中的对象必须已经被析构
                - Calls `::operator delete(void *)`, but it is unspecified when and how it is called. 
            - `a.construct` and `a.destruct` are removed `(since C++20)`, call the traits' static methods. 
          - [`std::allocator_traits`](https://en.cppreference.com/w/cpp/memory/allocator_traits)'s static methods 
            - [`std::allocator_traits::allocate`](https://en.cppreference.com/w/cpp/memory/allocator_traits/allocate)
            - [`std::allocator_traits::deallocate`](https://en.cppreference.com/w/cpp/memory/allocator_traits/deallocate)
            - [`std::allocator_traits::construct`](https://en.cppreference.com/w/cpp/memory/allocator_traits/construct)
            - [`std::allocator_traits::destory`](https://en.cppreference.com/w/cpp/memory/allocator_traits/destory)
          - NOT RECOMMENDED:
            - [Placement `new`](https://en.cppreference.com/w/cpp/language/new#Placement_new)
            - Manually call object destructor
    - 标准库 *未初始化内存* 算法（`<memory>`）
        - [`std::construct_at`](https://en.cppreference.com/w/cpp/memory/construct_at) `(C++20)`
          - Creates a `T` object initialized with arguments `args...` at given address `p`. 
          ```c++
          template <class T, class ... Args>
          constexpr T * construct_at(T * p, Args && ... args);
          ```
          - Specialization of this function template participates in overload resolution 
            only if `::new(std::declval<void *>()) T(std::declval<Args>()...)` is well-formed in an unevaluated context.
          - Equivalent to the following except that `std::construct_at` is `constexpr`. 
          ```c++
          return ::new (const_cast<void *>(static_cast<const volatile void *>(p)))
              T(std::forward<Args>(args)...);
          ``` 
          - When `std::construct_at` is called in the evaluation of some constant expression `e`, 
            the argument `p` must point to either storage obtained by `std::allocator<T>::allocate `
            or an object whose lifetime began within the evaluation of `e`. 
          - **Parameters**
            - `p`: Pointer to the uninitialized storage on which a `T` object will be constructed
            - `args...`: Arguments used for initialization
          - **Return value**: `p`
        - [`std::destroy_at`](https://en.cppreference.com/w/cpp/memory/destroy_at) `(C++17)`
            - 可能的实现
            ```c++
            // since C++20
            template <class T>
            constexpr void 
            destroy_at(T * p) 
            {
                if (std::is_array_v<T>)
                {
                    for (auto & elem : *p)
                    {
                        destroy_at(std::addressof(elem));
                    }
                }  
                else
                {
                    p->~T(); 
                } 
            }
            
            // until C++17
            template <class T> 
            void 
            destroy_at(T * p) 
            { 
                p->~T(); 
            }
            ```
            - 若`T`不是 *数组* 类型，则调用`p`所指向对象的析构函数，如同用`p->~T()`
            - 若`T`是 *数组* 类型，则
                - 程序非良构 `(until C++20)`
                - 按顺序递归地销毁`*p`的元素，如同通过调用`std::destroy(std::begin(*p), std::end(*p))` `(since C++20)`
        - [`std::destroy`](https://en.cppreference.com/w/cpp/memory/destroy) `(C++17)`
            - 可能的实现
            ```c++
            template <class ForwardIt>
            constexpr void 
            destroy(ForwardIt first, 
                    ForwardIt last)
            {
                for (; first != last; ++first)
                {
                    std::destroy_at(std::addressof(*first));
                }
            }
            ```
            - 销毁范围`[first, last)`中的对象
            - 复杂度：`Omega(last - first)`
        - [`std::destroy_n`](https://en.cppreference.com/w/cpp/memory/destroy_n) `(C++17)`
            - 可能的实现
            ```c++
            template <class ForwardIt, class Size>
            constexpr ForwardIt 
            destroy_n(ForwardIt first, 
                      Size      n)
            {
                for (; n > 0; (void) ++first, --n)
                {
                    std::destroy_at(std::addressof(*first));
                }
                  
                return first;
            }
            ```
            - 销毁从`first`开始的范围中的`n`个对象
            - 返回：已被销毁的元素的范围结尾，即`std::next(first, n)`
            - 复杂度：`Omega(n)`
          - [`std::uninitialized_copy`](https://en.cppreference.com/w/cpp/memory/uninitialized_copy)
              - 可能的实现
              ```c++
              template <class InputIt, class ForwardIt>
              ForwardIt 
              uninitialized_copy(InputIt   first, 
                                 InputIt   last, 
                                 ForwardIt d_first)
              {
                  typedef typename std::iterator_traits<ForwardIt>::value_type Value;
                  ForwardIt current = d_first;
                
                  try 
                  {
                      for (; first != last; ++first, (void) ++current) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value(*first);
                      }
                  } 
                  catch (...) 
                  {
                      for (; d_first != current; ++d_first) 
                      {
                          d_first->~Value();
                      }
                    
                      throw;
                  }
                
                  return current;
              }
              ```
              - 用来自范围`[first, last)`的元素，在始于`d_first`的 *未初始化内存* 中 *构造* 新元素 
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
                  - 注意是 *构造* ，**不是**单纯的 *迭代器解引用赋值* ，后者是`std::copy`
              - 返回：指向最后复制的元素后一元素的迭代器
              - 复杂度：`Omega(last - first)`
          - [`std::uninitialized_copy_n`](https://en.cppreference.com/w/cpp/memory/uninitialized_copy_n)
              - 可能的实现
              ```c++
              template <class InputIt, class Size, class ForwardIt>
              ForwardIt 
              uninitialized_copy_n(InputIt   first, 
                                   Size      count, 
                                   ForwardIt d_first)
              {
                  typedef typename std::iterator_traits<ForwardIt>::value_type Value;
                  ForwardIt current = d_first;
                
                  try 
                  {
                      for (; count > 0; ++first, (void) ++current, --count) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value(*first);
                      }
                  } 
                  catch (...) 
                  {
                      for (; d_first != current; ++d_first) 
                      {
                          d_first->~Value();
                      }
                    
                      throw;
                  }
                
                  return current;
              }
              ```
              - 从始于`first`的范围复制`count`个元素到始于`d_first`的 *未初始化内存* 
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 返回：指向最后复制的元素后一元素的迭代器
              - 复杂度：`Omega(count)`
          - [`std::uninitialized_fill`](https://en.cppreference.com/w/cpp/memory/uninitialized_fill) `(C++17)`
              - 可能的实现
              ```c++
              template <class ForwardIt, class T>
              void 
              uninitialized_fill(ForwardIt first, 
                                 ForwardIt last, 
                                 const T & value)
              {
                  typedef typename std::iterator_traits<ForwardIt>::value_type Value;
                  ForwardIt current = first;
                
                  try 
                  {
                      for (; current != last; ++current) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value(value);
                      }
                  }  
                  catch (...) 
                  {
                      for (; first != current; ++first) 
                      {
                          first->~Value();
                      }
                    
                      throw;
                  }
              }
              ```
              - 复制给定值`value`到以`[first, last)`定义的 *未初始化内存* 区域
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 复杂度：`Omega(last - first)`
          - [`std::uninitialized_fill_n`](https://en.cppreference.com/w/cpp/memory/uninitialized_fill_n) `(C++17)`
              - 可能的实现
              ```c++
              template <class ForwardIt, class Size, class T>
              ForwardIt 
              uninitialized_fill_n(ForwardIt first, 
                                   Size      count, 
                                   const T & value)
              {
                  typedef typename std::iterator_traits<ForwardIt>::value_type Value;
                  ForwardIt current = first;
                
                  try 
                  {
                      for (; count > 0; ++current, (void) --count) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value(value);
                      }
                    
                      return current;
                  } 
                  catch (...) 
                  {
                      for (; first != current; ++first) 
                      {
                          first->~Value();
                      }
                    
                      throw;
                  }
              }
              ```
              - 复制给定值`value`到始于`first`的 *未初始化内存区域* 的首`count`个元素
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 返回：指向最后复制的元素后一位置元素的迭代器
              - 复杂度：`Omega(count)`
          - [`std::uninitialized_move`](https://en.cppreference.com/w/cpp/memory/uninitialized_move) `(C++17)`
              - 可能的实现
              ```c++
              template <class InputIt, class ForwardIt>
              ForwardIt 
              uninitialized_move(InputIt   first, 
                                 InputIt   last, 
                                 ForwardIt d_first)
              {
                  typedef typename std::iterator_traits<ForwardIt>::value_type Value;
                  ForwardIt current = d_first;
                
                  try 
                  {
                      for (; first != last; ++first, (void) ++current) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value(std::move(*first));
                      }

                      return current;
                  } 
                  catch (...) 
                  {
                      for (; d_first != current; ++d_first) 
                      {
                          d_first->~Value();
                      }
                    
                      throw;
                  }
              }
              ```
              - 从范围`[first, last)` *移动* 元素到始于`d_first`的 *未初始化内存区域* 
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 返回：指向最后被移动元素的后一元素的迭代器
              - 复杂度：`Omega(last - first)`
          - [`std::uninitialized_move_n`](https://en.cppreference.com/w/cpp/memory/uninitialized_move_n) `(C++17)`
              - 可能的实现
              ```c++
              template <class InputIt, class Size, class ForwardIt>
              std::pair<InputIt, ForwardIt> 
              uninitialized_move_n(InputIt   first, 
                                   Size      count, 
                                   ForwardIt d_first)
              {
                  typedef typename std::iterator_traits<ForwardIt>::value_type Value;
                  ForwardIt current = d_first;
                
                  try 
                  {
                      for (; count > 0; ++first, (void) ++current, --count) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value(std::move(*first));
                      }
                  } 
                  catch (...) 
                  {
                      for (; d_first != current; ++d_first) 
                      {
                          d_first->~Value();
                      }
                    
                      throw;
                  }
                
                  return {first, current};
              }
              ```
              - 从始于`first`的范围 *移动* `count`个元素到始于`d_first`的 *未初始化内存区域* 
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 返回：指向源范围中最后被移动的元素后一元素的迭代器，和指向目标范围中最后移动到的元素后一元素的迭代器
              - 复杂度：`Omega(count)`
          - [`std::uninitialized_default_construct`](https://en.cppreference.com/w/cpp/memory/uninitialized_default_construct) `(C++17)`
              - 可能的实现
              ```c++
              template <class ForwardIt>
              void 
              uninitialized_default_construct(ForwardIt first, 
                                              ForwardIt last)
              {
                  using Value = typename std::iterator_traits<ForwardIt>::value_type;
                  ForwardIt current = first;
                
                  try 
                  {
                      for (; current != last; ++current) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value;
                      }
                  }  
                  catch (...) 
                  {
                      std::destroy(first, current);
                      throw;
                  }
              }
              ```
              - 以 *默认初始化* 在范围`[first, last)`所指代的 *未初始化内存* 上构造`typename iterator_traits<ForwardIt>::value_type`类型对象
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 复杂度：`Omega(last - first)`
          - [`std::uninitialized_default_construct_n`](https://en.cppreference.com/w/cpp/memory/uninitialized_default_construct_n) `(C++17)`
              - 可能的实现
              ```c++
              template <class ForwardIt, class Size>
              ForwardIt 
              uninitialized_default_construct_n(ForwardIt first, 
                                                Size      n)
              {
                  using Value = typename std::iterator_traits<ForwardIt>::value_type;
                  ForwardIt current = first;
                
                  try 
                  {
                      for (; n > 0 ; (void) ++current, --n) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value;
                      }
                    
                      return current;
                  }  
                  catch (...) 
                  {
                      std::destroy(first, current);
                    
                      throw;
                  }
              }
              ```
              - 在`first`起始的 *未初始化内存* 中以 *默认初始化* 构造`n`个`typename iterator_traits<ForwardIt>::value_type`类型对象
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 返回：对象范围的结尾，即`std::next(first, n)`
              - 复杂度：`Omega(n)`
          - [`std::uninitialized_value_construct`](https://en.cppreference.com/w/cpp/memory/uninitialized_value_construct) `(C++17)`
              - 可能的实现
              ```c++
              template <class ForwardIt>
              void 
              uninitialized_value_construct(ForwardIt first, 
                                            ForwardIt last)
              {
                  using Value = typename std::iterator_traits<ForwardIt>::value_type;
                  ForwardIt current = first;
                
                  try 
                  {
                      for (; current != last; ++current) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value();
                      }
                  } 
                  catch (...) 
                  {
                      std::destroy(first, current);
                      throw;
                  }
              }
              ```
              - 以 *值初始化* 在范围`[first, last)`所指代的 *未初始化内存* 上构造`typename iterator_traits<ForwardIt>::value_type`类型对象
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 复杂度：`Omega(last - first)`
          - [`std::uninitialized_value_construct_n`](https://en.cppreference.com/w/cpp/memory/uninitialized_value_construct_n) `(C++17)`
              - 可能的实现
              ```c++
              template <class ForwardIt, class Size>
              ForwardIt 
              uninitialized_value_construct_n(ForwardIt first, 
                                              Size      n)
              {
                  using Value = typename std::iterator_traits<ForwardIt>::value_type;
                  ForwardIt current = first;
                
                  try 
                  {
                      for (; n > 0 ; (void) ++current, --n) 
                      {
                          ::new (static_cast<void*>(std::addressof(*current))) Value();
                      }
                    
                      return current;
                  }  
                  catch (...)
                  {
                      std::destroy(first, current);
                      throw;
                  }
              }
              ```
              - 在`first`起始的 *未初始化内存* 中以 *值初始化* 构造`n`个`typename iterator_traits<ForwardIt>::value_type`类型对象
                  - 若期间抛出异常，则以 *未指定顺序* 销毁已构造的对象
              - 返回：对象范围的结尾，即`std::next(first, n)`
              - 复杂度：`Omega(n)`

#### 共享指针应用类`StrBlob`

```
class StrBlob
{
public:
    friend class StrBlobPtr;
    typedef std::vector<std::string>::size_type size_type;

    StrBlob() : data(std::make_shared<std::vector<std::string>>())
    {
    }

    StrBlob(std::initializer_list<std::string> il) : data(std::make_shared<std::vector<std::string>>(il))
    {
    }

    // add and remove elements
    void push_back(const std::string & t)
    {
        data->push_back(t);
    }

    void pop_back()
    {
        check(0, "pop_back on empty StrBlob");
        data->pop_back();
    }

    // statistics
    [[nodiscard]] size_type size() const
    {
        return data->size();
    }

    [[nodiscard]] bool empty() const
    {
        return data->empty();
    }

    // element access
    std::string & front()
    {
        // if the vector is empty, check will throw
        check(0, "front on empty StrBlob");
        return data->front();
    }

    std::string & back()
    {
        check(0, "back on empty StrBlob");
        return data->back();
    }

private:
    // throws msg if data[i] isn't valid
    void check(size_type i, const std::string & msg) const
    {
        if (i >= data->size())
        {
            throw std::out_of_range(msg);
        }
    }

private:
    std::shared_ptr<std::vector<std::string>> data;
};
```

#### 弱指针和重载运算符应用类`StrBlobPtr` => 14.6, 14.7

```
// StrBlobPtr throws an exception on attempts to access a nonexistent element
class StrBlobPtr
{
public:
    StrBlobPtr() : curr(0)
    {
    }

    explicit StrBlobPtr(StrBlob & a, size_t sz = 0) : wptr(a.data), curr(sz)
    {
    }

    // prefix: return a reference to the incremented/decremented object
    StrBlobPtr & operator++()
    {
        // if curr already points past the end of the container, can't increment it
        check(curr, "increment past end of StrBlobPtr");
        // advance the current state
        ++curr;
        return *this;
    }

    StrBlobPtr & operator--()
    {
        // move the current state back one element
        --curr;
        // if curr is zero, decrementing it will yield an invalid subscript
        check(-1, "decrement past begin of StrBlobPtr");
        return *this;
    }

    // postfix: increment/decrement the object but return the unchanged value
    StrBlobPtr operator++(int)
    {
        // no check needed here; the call to prefix increment will do the check
        StrBlobPtr ret = *this;      // save the current value
        ++*this;                     // advance one element; prefix ++ checks the increment
        return ret;                  // return the saved state
    }

    StrBlobPtr operator--(int)
    {
        // no check needed here; the call to prefix decrement will do the check
        StrBlobPtr ret = *this;      // save the current value
        --*this;                     // move backward one element; prefix -- checks the decrement
        return ret;                  // return the saved state
    }

    std::string & operator*() const
    {
        std::shared_ptr<std::vector<std::string>> p = check(curr, "dereference past end");
        return (*p)[curr];           // (*p) is the vector to which this object points
    }

    std::string * operator->() const
    {
        return & this->operator*();  // delegate the real work to the dereference operator
    }

private:
    // check returns a shared_ptr to the vector if the check succeeds
    std::shared_ptr<std::vector<std::string>> check(std::size_t i, const std::string & msg) const
    {
        std::shared_ptr<std::vector<std::string>> ret = wptr.lock();  // is the vector still around?

        if (!ret)
        {
            throw std::runtime_error("unbound StrBlobPtr");
        }

        if (i >= ret->size())
        {
            throw std::out_of_range(msg);
        }

        return ret; // otherwise, return a shared_ptr to the vector
    }

private:
    // store a weak_ptr, which means the underlying vector might be destroyed
    std::weak_ptr<std::vector<std::string>> wptr;

    // current position within the array
    std::size_t curr;
};
```






### 🌱 [Chap 13] 拷贝控制（Copy Control）

- *拷贝控制操作* （Copy Control）
    - 定义一个类时，我们显式或隐式地定义在此类型的对象 *拷贝* 、 *移动* 、 *赋值* 和 *销毁* 时做什么
    - *拷贝控制成员* ：一个类通过 *五种* 特殊的成员函数控制这些操作
        1. [*拷贝构造函数*](https://en.cppreference.com/w/cpp/language/copy_constructor)（copy constructor）
            - 用同类型 *另一对象* 初始化本对象是会发生什么
        2. [*拷贝赋值运算符*](https://en.cppreference.com/w/cpp/language/copy_assignment)（copy-assignment operator）
            - 将一个对象赋值给同类型 *另一对象* 时会发生什么
        3. [*移动构造函数*](https://en.cppreference.com/w/cpp/language/move_constructor)（move constructor）
            - 用同类型 *另一对象* 初始化本对象是会发生什么
        4. [*移动赋值运算符*](https://en.cppreference.com/w/cpp/language/move_assignment)（move-assignment operator）
            - 将一个对象赋值给同类型 *另一对象* 时会发生什么
        5. [*析构函数*](https://en.cppreference.com/w/cpp/language/destructor)（destructor）
            - 此类型对象销毁时会发生什么

#### 拷贝、赋值与销毁（Copy, Assign And Destroy）

- [*拷贝构造函数*](https://en.cppreference.com/w/cpp/language/copy_constructor)
    - *第一个参数* 是自身类类型的 *左值引用* 的构造函数
        - 可以有额外参数，但必须提供 *默认实参* 
        - 在几种情况下都会被 *隐式* 地使用，因此**不应该**是`explicit`的
    ```
    class Foo 
    {
    public:
        Foo();             // default constructor
        Foo(const Foo &);  // copy constructor
    // ...
    };
    ```
    - *合成拷贝构造函数* （Synthesized Copy Constructor）
        - 不管有没有人工定义拷贝构造函数时，编译器都会隐式定义一个
        - 对某些类，用于阻止拷贝该类型对象（`= delete;`）
        - 一般情况，将其参数的 *非静态数据成员* 逐个拷贝到正在创建的对象中
            - 类类型：调用其拷贝构造函数
            - 内置类型：直接拷贝
                - 数组：不能直接拷贝，因此逐元素拷贝
    - *拷贝初始化* （copy initialization）
        - 直接初始化：从 *明确的构造函数实参的集合* 初始化对象。要求编译器使用普通的函数匹配来选择相应的构造函数
        - 拷贝初始化：从 *另一个对象* 初始化对象。要求编译器将右侧运算对象（如需要，隐式类型转换后）拷贝到正在创建的对象中
            - 通常使用拷贝构造函数完成
            - 如果有 *移动构造函数* ，则拷贝初始化有时会使用移动构造函数来完成
        ```
        std::string dots(10, '.');                  // direct initialization
        std::string s(dots);                        // direct initialization
        std::string s2 = dots;                      // copy initialization
        std::string null_book = "9-999-99999-9";    // copy initialization
        std::string nines = std::string(100, '9');  // copy initialization
        ```
        - 拷贝初始化会在以下情况发生
            - 容器中的`push`、`insert`使用拷贝初始化
            - 容器中的`emplace`使用直接初始化
        ```
        T object = other;                      (1)     
        T object = {other} ;                   (2)
        function(other)                        (3)  // 函数非引用形参    
        return other;                          (4)     
        throw object;
        catch (T object)                       (5)     
        T array[N] = {other};                  (6)  // 聚合初始化中以初始化提供了初始化器的每个元素   
        ```
        - 拷贝初始化的限制
            - 当使用的初始化值要求通过`explicit`构造函数，就必须显式进行类型转换
        ```
        std::vector<int> v1(10);   // ok: direct initialization
        std::vector<int> v2 = 10;  // error: constructor that takes a size is explicit
        void f(std::vector<int>);  // f's parameter is copy initialized
        f(10);                     // error: can't use an explicit constructor to copy an argument
        f(std::vector<int>(10));   // ok: directly construct a temporary vector from an int
        ```
        - 编译器 *可以* 但 *不是必须* 绕过拷贝构造函数，直接创建对象
            - 但即使绕过了，拷贝构造函数仍必须 *存在* 且 *可访问* （如，不能是`= delete;`或`private`）
        ```
        std::string nullBook = "9-999-99999-9";  // copy initialization
        
        // is rewritten into
        std::string nullBook("9-999-99999-9");   // compiler omits the copy constructor
        ```
- [*拷贝赋值运算符*](https://en.cppreference.com/w/cpp/language/copy_assignment)
    - 要求
        - 赋值运算符应该返回一个指向其左侧运算对象的 *引用* 
        - 必须正确处理 *自赋值* （ *拷贝并交换赋值运算符* 则自动能处理自赋值）
        - 大多数拷贝赋值运算符组合了 *析构函数* 和 *拷贝构造函数* 二者的工作
            - 公共的工作应放到 *私有的工具函数* 中完成
    - *合成拷贝赋值运算符* （Synthesized Copy-Assignment Operator）
        - 如果没有定义拷贝赋值运算符，编译器会自动定义一个
        - 对某些类，用于阻止拷贝该类型对象（`= delete;`）
        - 一般情况，将其参数的 *非静态数据成员* 逐个拷贝到正在创建的对象中
            - 类类型：调用其拷贝赋值运算符
            - 内置类型：直接赋值
                - 数组：不能直接赋值，因此逐元素赋值
    ```
    // equivalent to the synthesized copy-assignment operator
    Sales_data & Sales_data::operator=(const Sales_data & rhs)
    {
        bookNo = rhs.bookNo;          // calls the string::operator=
        units_sold = rhs.units_sold;  // uses the built-in int assignment
        revenue = rhs.revenue;        // uses the built-in double assignment
        return *this;                 // return a reference to this object
    }
    ```
- [*析构函数*](https://en.cppreference.com/w/cpp/language/destructor)
    - 析构函数
        - 签名：`T::~T();`，不接受参数，没有返回值（注意不是返回`void`）
            - 没有参数意味着析构函数**不能**被重载
        - 负责释放对象使用的资源，并销毁非`static`数据成员
            - 成员按照初始化顺序的 *逆序* 销毁
            - *隐式* 销毁内置指针类型**不会**`delete`它指向的对象
            - *智能指针* 是类类型，有自己的析构函数，因此它被隐式销毁时也会`delete`其成员
    - 何时会调用析构函数
        - 无论何时一个对象被销毁，就会自动调用其析构函数
            - 非静态变量离开其作用域时会被销毁
                - 当一个对象的 *引用* 
            - 当对象被销毁时，其成员也会被销毁
            - 容器（标准库容器和数组）被销毁时，其元素被销毁
            - 对于动态分配的对象，当对指向它的指针使用`delete`时，会被销毁
            - 对于临时对象，当创建它的表达式结束时被销毁
    - *合成析构函数* （synthesized destructor）
        - 类未定义自己的析构函数时，编译器会自动定义一个
        - 对某些类，用于阻止该类型对象被销毁（`= delete;`）
- *三五法则* （The rule of three/five）
    - 三个基本操作可以控制类的拷贝操作
        1. 拷贝构造函数
        2. 拷贝赋值运算符
        3. 析构函数
    - `C++11`多添加了两个
        1. 移动构造函数
        2. 移动赋值运算符
    - 千言万语汇聚成一句话， *三五法则* ，五个拷贝控制成员要定义就 *都定义全* ，就没这么多破事儿了
        - 还有一句： *拷贝并交换赋值运算符* 好哇，天生异常安全、不怕自赋值，还同时能充当拷贝和移动两种运算符
    ```
    struct S35
    {
        S35() { printf("S35::S35()\n"); }
        explicit S35(const int i) : p(new int(i)) { printf("S35::S35(const int &)\n"); }
        S35(const S35 & rhs) : p(new int(*rhs.p)) { printf("S35::S35(const S35 &)\n"); }
        S35(S35 && rhs) noexcept : p(std::move(rhs.p)) { printf("S35::S35(S35 &&)\n"); }
        virtual ~S35() { printf("S35::~S35()\n"); };

        S35 & operator=(const S35 & rhs)
        {
            printf("S35::operator=(const S35 &)\n");
            if (this != &rhs) p = std::make_unique<int>(*rhs.p);
            return *this;
        }

        S35 & operator=(S35 && rhs) noexcept
        {
            printf("S35::operator=(S35 &&)\n");
            if (this != &rhs) p = std::move(rhs.p);
            return *this;
        }

    //    // copy-and-swap assign operator deals with self-assignment 
    //    // and servers automatically as both copy and move assign operator
    //    S35 & operator=(S35 rhs)
    //    {
    //        printf("S35::operator=(S35)\n");
    //        using std::swap;
    //        swap(p, rhs.p);
    //        return *this;
    //    }

        // when used as condition, this explicit operator will still be applied by compiler implicitly
        // "this is a feature, NOT a bug. " -- Microsoft
        explicit operator bool() const { return static_cast<bool>(*p); }

        std::unique_ptr<int> p{new int(0)};
    };
    
    S35 s1{0};              // S35::S35(const int &)
    S35 s2{s1};             // S35::S35(const S35 &)
    S35 s3{std::move(s2)};  // S35::S35(S35 &&)
    
    S35 s4{1};              // S35::S35(const int &)
    s4 = s3;                // S35::operator=(const S35 &)
    s4 = S35{2};            // S35::S35(const int &)
                            // S35::operator=(S35 &&)
    s4 = std::move(s3);     // S35::operator=(S35 &&)
    ```
- *显式默认* 和 *删除函数* 
    - 大多数类应该定义默认构造函数、拷贝构造函数和拷贝赋值运算符，不论是隐式地还是显式地
    - 有些情况反而应当 *阻止* 拷贝或赋值，方法有
        - 对应控制成员定义为 *删除* 的函数（正确做法）
        - 对应控制成员 *声明但不定义* 为 *私有* 的函数（早期没有`= delete;`时的做法，现在**不应**这么干）
            - *声明但不定义成员函数* 是合法操作，除一个**例外** => 15.2.1
                - 试图访问未定义的成员将导致 *链接时错误* （link-time failure）
            - 试图拷贝对象的用户代码将产生 *编译错误* 
            - 成员函数或友元函数中的拷贝操作将导致 *链接时错误*
    - *显式默认* `= default;`
        - 可以通过将拷贝控制成员定义为`= default;`来 *显式地* 要求编译器生成合成版本
        - 只能对具有合成版本的成员函数使用`= default;`
        - *不必* 出现在函数第一次声明的时候
    - *删除的函数* （deleted function）`= delete;`
        - 可以对 *任何函数* （即，不一定是拷贝控制成员，可以是任何成员或全局函数、函数模板等等）指定`= delete;`
        - *必须* 出现在函数第一次声明的时候
    - 删除析构函数
        - 这种对象无法销毁
        - **不能** *定义* 该类变量或创建临时对象
        - 可以 *动态分配* 该类变量或创建临时对象，但仍旧**不能**释放
        ```
        struct NoDtor 
        {
            NoDtor() = default;     // use the synthesized default constructor
            ~NoDtor() = delete;     // we can't destroy objects of type NoDtor
        };
        
        NoDtor nd;                  // error: NoDtor::~NoDtor() is deleted
        NoDtor * p = new NoDtor();  // ok: but we can't delete p
        delete p;                   // error: NoDtor::~NoDtor() is deleted
        ```
- 如类有数据成员 *不能默认构造、拷贝、赋值或销毁* ，则对应的合成的拷贝控制成员是 *删除的* 
    - 合成的默认构造函数
        - 类的某个数据成员的 *析构函数* 函数是删除的或不可访问的（例如是`private`的）
            - 否则，将自动搞出无法销毁的对象
        - 类的某个数据成员是`const`的，且 *没有类内初始化器* 
            - `const`必须显式初始化、不能赋值
        - 类的某个数据成员是 *引用* 的、 *没有类内初始化器* 且 *未显式定义默认构造函数* 
            - 引用必须显式初始化，赋值改变被引用的对象而不是引用本身
    - 合成拷贝构造函数
        - 类的某个数据成员的 *拷贝构造函数* 是删除的或不可访问的
        - 类的某个数据成员的 *析构函数* 函数是删除的或不可访问的
    - 合成拷贝赋值运算符
        - 类的某个数据成员的 *拷贝赋值运算符* 是删除的或不可访问的
        - 类的某个数据成员是`const`的
        - 类的某个数据成员是 *引用* 的
    - 合成析构函数
        - 类的某个数据成员的 *析构函数* 是析构的或者不可访问的
    - => 13.6.2，15.7.2，19.6

#### 深浅拷贝

- *深拷贝*
    - 拷贝语义：拷贝副本和原对象完全独立
    - 行为像 *值* ，例如：`std::string`
    - 深拷贝`Entry<int, std::string>`的实现
        1. 定义 *拷贝构造函数* ，完成`std::string`的拷贝，而不是拷贝指针
        2. 定义 *析构函数* 来释放`std::string`
        3. 定义 *拷贝赋值运算符* 来释放当前的`std::string`，并从右侧运算对象拷贝`std::string`
    ```
    struct Entry
    {
        explicit Entry(const int & _i = 0, const std::string & s = std::string()) :
                i(_i), ps(new std::string(s))
        {
        }

        ~Entry()
        {
            delete ps;
        }

        Entry(const Entry & p) : i(p.i), ps(new std::string(*p.ps))
        {
        }

        Entry & operator=(const Entry & rhs)
        {
            if (this != &rhs)                   // deal with self-assignemnt!
            {
                i = rhs.i;
                delete ps;
                ps = new std::string(*rhs.ps);  // otherwise will use deleted memory!
            }

            return *this;
        }

        int i;                                  // key
        std::string * ps;                       // value
    };
    ```
- *浅拷贝* 
    - 拷贝语义：拷贝副本和原对象 *共享* 底层数据。改变一个，另一个也会随之改变
    - 行为像 *指针* ，例如：`std::shared_ptr<T>`
    - 析构函数不能简单地直接释放内存，而要使用 *引用计数*
        - 引用计数器可以保存到动态内存中，拷贝时直接拷贝指向引用计数的指针即可
        - 工作守则
            1. 除了初始化对象外，每个构造函数（拷贝构造函数除外）还要创建一个引用计数，用于记录有多少对象与正在创建的对象共享底层数据。当创建第一个对象时，引用计数初始化为`1`
            2. 拷贝构造函数不分配新的计数器，而是拷贝包括计数器在内的给定对象的数据成员。拷贝构造函数递增共享的计数器，指出给定对象的底层数据又被一个新用户所共享
            3. 析构函数递减计数器，指出共享数据的用户又少了一个。如果计数器变为`0`，则析构函数释放底层数据
            4. 拷贝赋值运算符递增右侧运算对象的计数器，递减左侧运算对象的计数器。如果左侧运算对象的计数器变为`0`，则销毁其底层数据
    ```
    class Entry
    {
    public:
        explicit Entry(const int & _i = 0, const std::string & s = std::string()) :
                i(_i), ps(new std::string(s)), useCount(new std::size_t(1))
        {
        }

        Entry(const Entry & p) :
                ps(p.ps), i(p.i), useCount(p.useCount)
        {
            ++*useCount;
        }

        ~Entry()
        {
            if (--*useCount == 0)
            {
                delete ps;
                delete useCount;
            }
        }

        Entry & operator=(const Entry & rhs)
        {
            if (this != &rhs)
            {
                ++*rhs.useCount;

                if (--*useCount == 0)
                {
                    delete ps;
                    delete useCount;
                }

                i = rhs.i;
                ps = rhs.ps;
                useCount = rhs.useCount;
            }

            return *this;
        }

    private:
        int i;                   // if only for preformance, trivial types don't have to be shared
        std::string * ps;        // share this std::string (as an example of some huge data type)
        std::size_t * useCount;  // how many instances are sharing *ps
    };
    ```

#### `std::move`探究

- [复制消除](https://en.cppreference.com/w/cpp/language/copy_elision)（Copy Elision）
    - 省略 *拷贝构造函数* 和 *移动构造函数* ，达成无拷贝的按值传递语义，分为
        1. *强制消除* （Mandatory elision） `(since C++17)`
            - 编译器被 *强制要求* 省略拷贝和移动构造函数，哪怕它们还有其他效果（side effects，比如输出语句等）
            - 对象会被 *一步到位地直接构造* 于它们本来会被 *拷贝* 或 *移动* 的存储位置中
            - 拷贝构造函数和移动构造函数**不**需要可见或可访问
            - 析构函数 *必须* 可见可访问，哪怕并没有对象被析构
            - 具体发生于如下情景
                1. *返回语句* 中，操作数是和返回值 *同类型纯右值* （不考虑`cv`限定）
                ```
                T f() { return T(); }
                f();                    // only one call to default constructor of T
                ```
                2. *初始化* 对象时，初始化器表达式是和变量 *同类型纯右值* （不考虑`cv`限定）
                    - 这一条仅当被初始化的对象**不是** *潜在重叠的子对象* 时有效
                    - 这条规则并不是优化，因为`C++17`中纯右值是被 *未实质化传递* 的，甚至不会构造临时量
                    - 特别地：返回类型不是引用的函数类型的返回值都是纯右值
                ```
                T x = T(T(f()));        // only one call to default constructor of T, to initialize x
                
                struct C { /* ... */ };
                C f();
                struct D;
                D g();
                struct D : C 
                {
                    D() : C(f()) {}     // no elision when initializing a base-class subobject
                    D(int) : D(g()) {}  // no elision because the D object being initialized 
                                        // might be a base-class subobject of some other class
                };
                ```
        2. *非强制消除* (Non-mandatory elision) `(since C++11)`
            - 编译器被 *允许* 省略拷贝和移动构造函数，哪怕它们还有其他效果（side effects，比如输出语句等）
            - 对象会被 *一步到位地直接构造* 于它们本来会被 *拷贝* 或 *移动* 的存储位置中
            - 这是优化：即使没有调用拷贝构造函数和移动构造函数，它们也 *必须* 需要可见可访问
            - 可以连锁多次复制消除，以消除多次复制
            - 具体发生于如下情景
                1. *具名返回值优化* （Named Return Value Optimization，`NRVO`）
                    - *返回语句* 中，操作数是 *非`vloatile`自动对象* ，且**不是** *函数形参* 或 *`catch`子句形参* ，且与返回值 *同类型* （不考虑`const`限定）
                2. *初始化* 对象时，源对象是和变量 *同类型无名临时量* （不考虑`cv`限定） `(until C++17)`
                    - 临时量是返回语句操作数时，被称作 *返回值优化* （Return Value Optimization，`VRO`）
                    - `C++17`开始， *返回值优化* 已经变成了 *强制消除* 
                3. `throw`表达式和`catch`子句中某些情况
                4. *常量表达式* 和 *常量初始化* 中，保证进行`RVO`，但禁止`NRVO` `(since C++14)`
- 构造函数中的最速形参传递
    - 平凡形参：直接传 *常量* 就完事了，你搞什么右值啊引用啊什么的反而还慢了
    - `Clang-Tidy`规定构造函数的非平凡形参应该是 *传值加`std::move`* ，而**不是** *传常引用* 
        - 传 *常引用* 
            - 不论实参是左值还是右值，都是一步 *引用初始化* 
            - 接着如果用形参进行赋值，则是一步 *拷贝* 
        - 传 *值* 加`std::move`：不知道高到哪儿去了
            - 实参为 *左值* ，一步 *拷贝* 和一步 *移动* （如果用形参赋值）
            - 实参为 *右值* ，两步 *移动* （如果用形参赋值）
    ```
    struct T
    {
        T(const std::string & _s, const int _i) : s(_s), i(_i) {}     // Clang-Tidy: NOT good
        
        T(std::string _s, const int _i) : s(std::move(_s)), i(_i) {}  // Clang-Tidy: good
        
        std::string s;
        int i;
    }
    ```
- `std::move`不能乱用
    - `例1`：作大死
        - 看，看什么看，这是 *悬垂引用* ，没事作死玩儿啊
        - 再说一遍：函数**不能**返回临时量的引用，左值右值，常或非常都不行
        - 返回语句中生成的右值引用得绑定到普通类型返回值上，才能发生移动构造
        - 绑到引用上，资源压根儿就没转移，函数结束就被析构了
    ```
    std::vector<int> && return_vector()
    {
        std::vector<int> tmp {1, 2, 3, 4, 5};
        return std::move(tmp);
    }

    std::vector<int> && rval_ref = return_vector();
    ```
    - `例2`：弄巧成拙
        - 乱用`std::move`抑制了 *拷贝消除* ，反而**不好**
    ```
    std::vector<int> return_vector()
    {
        std::vector<int> tmp {1, 2, 3, 4, 5};
        return std::move(tmp);
    }

    std::vector<int>         val      = return_vector();  // 强制拷贝消除
    std::vector<int> &&      rval_ref = return_vector();
    const std::vector<int> & c_ref    = return_vector();
    ```
    - 如何做到不乱用`std::move`
        - 很多情况下需要使用`std::move`来提升性能，但不是所有时候都该这么用
        - 实际情况很复杂，但`gcc 9`开始支持了[如下两个编译器选项](https://developers.redhat.com/blog/2019/04/12/understanding-when-not-to-stdmove-in-c/)来识别
            - `-Wpessimizing-move`
                - `std::move`阻碍`NRVO`时报`warning`
                - 这种情况下是有性能损失的，必须避免
                - 包含在`-Wall`中
            - `-Wredundant-move`
                - 当满足 *强制拷贝消除* 时还写了`std::move`时报`warning`
                - 这种情况下没有性能损失，纯粹只是多余而已
                - 包含在`-Wextra`中，`-Wall`中没有
        - 所以函数首先不能返回引用，其次返回的临时量时姑且先加上`std::move`，看报不报`warning`好了
        - `CMake`使用示例
        ```
        target_compile_options(${PROJECT_NAME} PUBLIC -Wpessimizing-move -Wredundant-move)
        ```
    
#### 动态内存管理类`StrVec`

```
// simplified implementation of the dynamic memory allocation strategy for a vector-like class
class StrVec
{
public:
    // the allocator member is default initialized
    StrVec() : elements(nullptr), first_free(nullptr), cap(nullptr)
    {
    }

    // copy constructor
    StrVec(const StrVec & s)
    {
        // call alloc_n_copy to allocate exactly as many elements as in s
        std::pair<std::string *, std::string *> newdata = alloc_n_copy(s.begin(), s.end());
        elements = newdata.first;
        first_free = cap = newdata.second;
    }

    // copy assignment
    StrVec & operator=(const StrVec & rhs)
    {
        if (this != &rhs)
        {
            // call alloc_n_copy to allocate exactly as many elements as in rhs
            std::pair<std::string *, std::string *> data = alloc_n_copy(rhs.begin(), rhs.end());
            free();
            elements = data.first;
            first_free = cap = data.second;
        }

        return *this;
    }

    // move constructor
    // move won't throw any exceptions
    // member initializers take over the resources in s
    StrVec(StrVec && s) noexcept : elements(s.elements), first_free(s.first_free), cap(s.cap)
    {
        // leave s in a state in which it is safe to run the destructor
        s.elements = s.first_free = s.cap = nullptr;
    }

    // move assignment
    StrVec & operator=(StrVec && rhs) noexcept
    {
        // direct test for self-assignment
        if (this != &rhs)
        {
            // free existing elements
            free();
            // take over resources from rhs
            elements = rhs.elements;
            first_free = rhs.first_free;
            cap = rhs.cap;
            // leave rhs in a destructible state
            rhs.elements = rhs.first_free = rhs.cap = nullptr;
        }

        return *this;
    }

    // destructor
    ~StrVec()
    {
        free();
    }

    // copy the element
    void push_back(const std::string & s)
    {
        // ensure that there is room for another element
        chk_n_alloc();
        // construct a copy of s in the element to which first_free points
        new(first_free++) std::string(s);
        // alloc.construct(first_free++, s);  // deprecated in C++14
    }

    [[nodiscard]] size_t size() const
    {
        return first_free - elements;
    }

    [[nodiscard]] size_t capacity() const
    {
        return cap - elements;
    }

    [[nodiscard]] const std::string * begin() const
    {
        return elements;
    }

    [[nodiscard]] const std::string * end() const
    {
        return first_free;
    }

private:
    // used by the functions that add elements to the StrVec
    inline void chk_n_alloc()
    {
        if (size() == capacity())
        {
            reallocate();
        }
    }

    // utilities used by the copy constructor, assignment operator, and destructor
    std::pair<std::string *, std::string *>
    alloc_n_copy(const std::string * b, const std::string * e)
    {
        // allocate space to hold as many elements as are in the range
        std::string * data = alloc.allocate(e - b);
        // initialize and return a pair constructed from data and
        // the value returned by uninitialized_copy
        return {data, uninitialized_copy(b, e, data)};
    }

    // destroy the elements and free the space
    void free()
    {
        // may not pass deallocate a 0 pointer; if elements is 0, there's no work to do
        if (elements)
        {
            // destroy the old elements in reverse order
            for (std::string * p = first_free; p != elements; /* empty */)
            {
                std::destroy_at(--p);
                // alloc.destroy(--p);  // deprecated in C++14
            }

            alloc.deallocate(elements, cap - elements);
        }
    }

    // get more space and copy the existing elements
    void reallocate()
    {
        // we'll allocate space for twice as many elements as the current size
        size_t newcapacity = size() ? 2 * size() : 1;
        // allocate new memory
        std::string * newdata = alloc.allocate(newcapacity);

        // move the data from the old memory to the new
        std::string * dest = newdata;   // points to the next free position in the new array
        std::string * elem = elements;  // points to the next element in the old array

        for (size_t i = 0; i != size(); ++i)
        {
            new (dest++) std::string(std::move(*elem++));
            // alloc.construct(dest++, std::move(*elem++));  // deprecated in C++14
        }

        free();                // free the old space once we've moved the elements

        // update our data structure to point to the new elements
        elements = newdata;
        first_free = dest;
        cap = elements + newcapacity;
    }

private:
    std::allocator<std::string> alloc;  // allocates the elements

    std::string * elements;             // pointer to the first element in the array
    std::string * first_free;           // pointer to the first free element in the array
    std::string * cap;                  // pointer to one past the end of the array
};
```

#### 交换操作

- `std::swap`会产生三次 *移动* 赋值，例如`gcc`的实现可以约等价为
```
template <class T>
void
swap(T & a, T & b)
{
    T tmp = std::move(a);
    a = std::moveE(b);
    b = std::move(tmp);
}
```
- 这些移动赋值虽说比拷贝赋值强多了，有时仍旧是不必要的
    - 例如交换前面的 *浅复制型* `Entry`类，`swap`就没必要交换`ps`和`useCount`，实际只需要
        - `ps`和`useCount`两边压根就是共享的，为嘛闲的没事换那俩指针玩儿
```
class Entry
{
    friend void swap(Entry &, Entry &);  // this is just declaration! still need a definition outside
    // other members remain the same
};

inline void swap(Entry & lhs, Entry & rhs)
{
    swap(lhs.i, rhs.i);                  // swap the int members
}
```
- 接受类参数的`swap`函数应当调用类成员自己的`swap`，而**不是**`std::swap`
```
void swap(Foo & lhs, Foo & rhs)
{
    // WRONG: this function uses the library version of swap, not the HasPtr version
    std::swap(lhs.h, rhs.h);
    // swap other members of type Foo
}

void swap(Foo & lhs, Foo & rhs)
{
    using std::swap;
    swap(lhs.h, rhs.h);  // uses the HasPtr version of swap
    // swap other members of type Foo
}
```
- *拷贝并交换赋值运算符* （copy-and-swap assign operator）
    - 接受普通形参而不是常引用
    - 天然就是异常安全的，且能正确处理自赋值
    - 天然能同时充当 *拷贝赋值运算符* 和 *移动赋值运算符*
        - 前提是类定义了 *移动构造函数*
        - 传参的时候，会根据实参是左值还是右值调用对应的拷贝或移动构造函数
```
// note rhs is passed by value, which means the Entry copy constructor
// copies the string in the right-hand operand into rhs
Entry & operator=(Entry rhs)
{
    // swap the contents of the left-hand operand with the local variable rhs
    using std::swap;
    swap(*this, rhs);  // rhs now points to the memory this object had used
    return *this;      // rhs is destroyed, which deletes the pointer in rhs
}
```

#### 对象移动

- 移动对象
    - 提升性能
    - 某些对象不能拷贝，例如 *流对象* ，`std::unique_ptr`等
        - 标准库容器、`std::string`和`std::shared_ptr`既支持拷贝又支持移动
        - `I/O`类和`std::unique_ptr`可以移动但不能拷贝
- *右值引用* （rvalue references）
    - *必须* 绑定到 *右值* （包括 *纯右值* 、 *将亡值* ，都是没有用户、即将被销毁的）的引用
        - 复习一下 *右值* 的性质
            1. 不能取地址
            2. 不能赋值
            3. 不能初始化非常量左值引用
            4. 可以初始化右值引用或常量左值引用
    - 通过`&&`来获得
    - 可以自由地将一个右值的资源 *移动* ，或者说， *窃取* 到别处去
        - 反正没人要，而且马上就要被销毁了，不如拿走，待会儿销毁个寂寞
        - 变量都是 *左值* ， *左值* 不能直接绑定到 *右值引用* 上，即使这个变量自己也是 *右值引用* 类型也不行
            - 搞不懂这句话的人都是把 *（值的）类型* （type）和 *值类别* （value category）这俩货给搞混了
            - 比如`T && a;`， *（值的类型）是右值引用* 说的是`T &&`， *（值类别）是左值* 说的是`a`，压根不是一回事儿
    ```
    int i = 42;
    
    int & r1 = i;             // ok: r1 refers to i
    int & r2 = i * 42;        // error: i * 42 is an rvalue
    const int & r3 = i * 42;  // ok: we can bind a reference to const to an rvalue
    
    int && rr1 = i;           // error: cannot bind an rvalue reference to an lvalue
    int && rr2 = 42;          // ok: literal 42 is an rvalue
    int && rr3 = rr2;         // error: cannot bind an rvalue reference to an lvalue
    int && rr4 = i * 42;      // ok: bind rr2 to the result of the multiplication
    ```
- 从左值获取右值的两个方法
    1. 通过 *强制类型转换* 显式地将左值变为右值
        - 复习：所有`cast<T>`的结果的 *值类别* （value category）是
            - *左值* ，如果`T`为 *左值引用* 或 *函数类型的右值引用*  
            - *将亡值* ，如果`T`为 *对象类型的右值引用*
            - *纯右值* ，其他情况。此时生成转换结果需要一次 *拷贝构造* 
        - 辨析
            - 只要实参的值类别是右值，就绑定到右值引用版本
            - 至于实参的类型是对象，还是对象的右值引用，那根本无所谓
        ```
        S35 s1;
        S35 s2 {static_cast<S35>(s1)};  // cast result needs copy initialization
                                        // compiler will do copy elision
                                        // and construct s2 directly
                                        // and thus avoids an extra move initialization
        
        S35 s3;                         // default initialization
        S35 s4;                         // default initialization
        s4 = static_cast<S35>(s3);      // 1. copy initialization of the cast result
                                        // 2. move assignment
        ```
    2. `std::move`
        - 实际就是一个封装版的`static_cast`
    ```
    S35 s1;
    S35 s2;
    S35 s3;

    S35 && r1 = static_cast<S35 &&>(s1);
    S35 && r2 = reinterpret_cast<S35 &&>(s2);
    S35 && r3 = std::move(s3);
    ```
- [`std::move`](https://en.cppreference.com/w/cpp/utility/move)
    - 具体实现 => 16.2.6
    - 告诉编译器：我们有一个左值，但我们希望像处理一个右值一样处理它
    - 调用`std::move(var)`就意味着承诺：除了对`var` *赋值* 或 *销毁* 它外，我们将不再使用它
        - 调用`std::move`之后，移后源对象的值 *未定义* ；可以被 *赋值* 或 *销毁* ，但**不能** *使用它的值* 
    - 对`std::move`，调用时**不提供**`using`声明，而是直接调用`std::move` => 18.2.3
        - 避免名字冲突
- 移动操作成员
    - [*移动构造函数*](https://en.cppreference.com/w/cpp/language/move_constructor)
        - *第一个* 参数是自身类类型的 *右值引用* 的构造函数
            - 可以有额外参数，但必须提供 *默认实参* 
            - *必须标注* `noexcept`
                - 向编译器承诺 *不抛出异常* ，避免编译器为了处理异常做出额外操作（将被操作对象恢复原状）
                - 如果出现异常，被移动对象无法恢复原状，此时只能使用 *拷贝构造函数*
            - 从对象 *窃取* 资源， *接管* 对象的全部内存
            - 必须保证完事后，移后源对象必须保持 *有效的、可析构的* 状态，但用户**不能**对其值做任何假设
                1. 移后源对象**不再**指向被移动的资源
                2. *销毁* 移后源对象是无害的
                    - 指针全部 *置空* 就完事儿了
        ```
        // move constructor
        // move won't throw any exceptions
        // member initializers take over the resources in s
        StrVec(StrVec && s) noexcept : elements(s.elements), first_free(s.first_free), cap(s.cap)
        {
            // leave s in a state in which it is safe to run the destructor
            s.elements = s.first_free = s.cap = nullptr;
        }
        ```
    - [*移动赋值运算符*](https://en.cppreference.com/w/cpp/language/move_assignment)
        - 应标记为`noexcept`，必须妥善处理自赋值
        ```
        // move assignment
        StrVec & operator=(StrVec && rhs) noexcept
        {
            // direct test for self-assignment
            if (this != &rhs)
            {
                // free existing elements
                free();
                // take over resources from rhs
                elements = rhs.elements;
                first_free = rhs.first_free;
                cap = rhs.cap;
                // leave rhs in a destructible state
                rhs.elements = rhs.first_free = rhs.cap = nullptr;
            }

            return *this;
        }
        ```    
    - 合成的移动操作
        - 只有当类没有自定义任何拷贝控制成员、且类的每个非静态数据成员都可 *移动构造* 或 *移动赋值* 时，编译器会合成 *移动构造函数* 或 *移动赋值运算符* 
        - 编译器可以移动内置类型的的成员
        - 如果一个类没有移动操作，编译器会匹配到对应的拷贝操作
    ```
    // the compiler will synthesize the move operations for X and hasX
    struct X 
    {
        int i;                 // built-in types can be moved
        std::string s;         // std::string defines its own move operations
    };
    
    struct hasX 
    {
        X mem;                 // X has synthesized move operations
    };
    
    X x;
    X x2 = std::move(x);       // uses the synthesized move constructor
    
    hasX hx;
    hasX hx2 = std::move(hx);  // uses the synthesized move constructor
    ```
    - 编译器 *自动删除* 拷贝或移动成员
        - 当且仅当我们显式要求`= default;`的移动成员，而编译器不能移动该类的全部非静态数据成员时，编译器会定义 *被删除的* 移动成员
            - 移动构造函数
                - 类成员定义了自己的拷贝构造函数且未定义移动构造函数
                - 类成员未定义自己的拷贝构造函数且编译器不能为其合成移动构造函数
                - 类成员的移动构造函数被定义为删除的或者不可访问
                - 类的 *析构函数* 被定义为删除的或者不可访问
            - 移动赋值运算符
                - 类成员定义了自己的拷贝赋值运算符且未定义移动赋值运算符
                - 类成员未定义自己的拷贝赋值运算符且编译器不能为其合成移动赋值运算符
                - 类成员的移动赋值运算符被定义为删除的或者不可访问
                - 类成员有 *`const`的* 或者 *引用* 
        - 定义了移动成员后，类也必须定义对应的拷贝成员，否则，这些成员也被默认成删除的
        - 千言万语汇聚成一句话， *三五法则* ，五个拷贝控制成员要定义就 *都定义全* ，就没这么多破事儿了
            - 还有一句： *拷贝并交换赋值运算符* 好哇，天生异常安全、不怕自赋值，还同时能充当拷贝和移动两种运算符
    ```
    // assume Y is a class that defines its own copy constructor but not a move constructor
    struct Y
    {
        Y & operator=(const Y & rhs) = default;
        // move operator should be deleted by compiler
        // move operator should be deleted by compiler
        int v;
    };
    
    
    struct hasY 
    {
        hasY() = default;
        hasY(hasY &&) = default;
        Y mem;                 // hasY will have a deleted move constructor
    };
    hasY hy;
    hasY hy2 = std::move(hy);  // error: move constructor is deleted
                               // at least on gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)
                               // this one passes compiling
    ```
    - 类既有拷贝操作成员，又有移动操作成员时
        1. *移动右值，拷贝左值*
        ```
        StrVec v1, v2;
        v1 = v2;                        // v2 is an lvalue; copy assignment
        StrVec getVec(std::istream &);
        v2 = getVec(cin);               // getVec(cin) is an rvalue; move assignment
        ```
        2. 如果 *没有移动* 操作成员，则 *右值也被拷贝* 
        ```
        class Foo 
        {
        public:
            Foo() = default;
            Foo(const Foo &);     // copy constructor
            // other members, but Foo does not define a move constructor
            // so it is deleted by compiler
        };
        
        Foo x;
        Foo y(x);                 // copy constructor; x is an lvalue
        Foo z(std::move(x));      // copy constructor, because there is no move constructor
        ```
    - *拷贝并交换赋值运算符* 和移动
        - 定义了移动构造函数的类的 *拷贝并交换赋值运算符* 天然就同时是 *拷贝赋值运算符* 和 *移动赋值运算符*
- 右值引用和成员函数
    - 成员函数一样可以同时提供 *拷贝版本* 和 *移动版本*
        - 例如标准库容器的`c.push_back`就同时定义了
        ```
        void push_back(const X &);  // copy: binds to any kind of X
        void push_back(X &&);       // move: binds only to modifiable rvalues of type X
        ```
    - *引用限定符* （reference qualifier）
        - 我们调用成员函数时，通常不关心对象是左值还是右值
            - 但`this`指针还是知道自己是 *左值* 还是 *右值* 的
        - *标准库类型* 还允许 *向该类型的右值赋值* 
            - 这也是为了向前兼容啊，总不能学`python`吧
        ```
        std::string s1 = "a value", s2 = "another";
        auto n = (s1 + s2).find('a');           // (s1 + s2) is rvalue, and we are calling member function
        s1 + s2 = "wow!";                       // assigning an rvalue
        ```
        - 通过对类成员函数添加 *引用限定符* 可以限制`this`的 *值类别* 
            - 方法是，在和定义`const`成员函数时`const`一样的位置放置`&`或`&&`
        ```
        class Foo 
        {
        public:
            Foo & operator=(const Foo &) &;     // may assign only to modifiable lvalues
            // other members of Foo
        };
        
        Foo & Foo::operator=(const Foo & rhs) &
        {
            // do whatever is needed to assign rhs to this object
            return *this;
        }
        
        Foo & retFoo();  // returns a reference; a call to retFoo is an lvalue
        Foo retVal();    // returns by value; a call to retVal is an rvalue
        Foo i, j;        // i and j are lvalues
        
        i = j;           // ok: i is an lvalue
        retFoo() = j;    // ok: retFoo() returns an lvalue
        retVal() = j;    // error: retVal() returns an rvalue
        i = retVal();    // ok: we can pass an rvalue as the right-hand operand to assignment
        ```
        - 成员函数可以 *同时* 使用`const`限定和 *引用限定*
            - 此时， *引用限定符* 必须跟在`const` *之后* 
        ```
        class Foo 
        {
        public:
            Foo someMem() & const;     // error: const qualifier must come first
            Foo anotherMem() const &;  // ok: const qualifier comes first
        };
        ```
    - 重载和引用函数
        - 成员函数的`const`限定和 *引用限定* 均可用于重载函数
        ```
        class Foo 
        {
        public:
            Foo sorted() &&;                         // may run on modifiable rvalues
            Foo sorted() const &;                    // may run on any kind of Foo
            // other members of Foo
            
        private:
            std::vector<int> data;
        };
        
        // this object is an rvalue, so we can sort in place
        Foo Foo::sorted() &&
        {
            std::sort(data.begin(), data.end());
            return *this;
        }
        
        // this object is either const or it is an lvalue; either way we can't sort in place
        Foo Foo::sorted() const & 
        {
            Foo ret(*this);                          // make a copy
            sort(ret.data.begin(), ret.data.end());  // sort the copy
            return ret;                              // return the copy
        }
        
        retVal().sorted();  // retVal() is an rvalue, calls Foo::sorted() &&
        retFoo().sorted();  // retFoo() is an lvalue, calls Foo::sorted() const &
        ```
        - 如果一个成员函数有 *引用限定符* ，则所有具有相同形参列表的函数都必须也有
            - 如果根据`const`限定区分重载函数，两个函数可以一个加`const`另一个不加
            - 如果根据 *引用限定* 区分重载函数，两个函数 *必须都加* *引用限定符*
        ```
        class Foo 
        {
        public:
            Foo sorted() &&;
            Foo sorted() const;        // error: must have reference qualifier
            
            // Comp is type alias for the function type that can be used to compare int values
            using Comp = bool(const int&, const int&);
            
            Foo sorted(Comp *);        // ok: different parameter list
            Foo sorted(Comp *) const;  // ok: neither version is reference qualified
        };
        ```






### 🌱 [Chap 14] [重载运算符](https://en.cppreference.com/w/cpp/language/operators)（Overloaded Operations and Conversions）

#### 基本概念

- 重载的运算符是具有 *特殊名字* （`operator`和 *运算符号* ）的函数，也包含返回类型、参数列表以及函数体
    - 参数数量和该运算符作用的运算对象数量一样多
        - 一元运算符：一个
        - 二元运算符：两个，左侧运算对象传递给第一个参数，右侧传给第二个
        - 除重载的 *函数调用运算符* `operator()`之外，其他重载运算符**不能**有 *默认实参*
        - 重载的运算符如果是 *成员函数* ，则第一个（左侧）运算对象绑定到隐式的`this`指针上，只需指定右侧运算符（如有）
        - 成员运算符函数的（显式）参数数量比运算符的运算对象总数 *少一个* 
    - 重载的运算符和对应的内置运算符享有 *相同的优先级和结合律* 
- 什么运算符能被重载
    - 重载的运算符要么是 *类成员* ，要么含有 *至少一个类类型参数*
        - 这意味着只作用于 *内置类型* 的运算对象的运算符**不能**重载
    ```
    int operator+(int, int);  // error: cannot redefine the built-in operator for ints
    ```
    - 只能重载 *一部分内置运算符* ，**不能**发明新符号
        - 例如，不能提供`operator**`来执行幂运算
    - 能重载的内置运算符
        - 有四个符号（`+`，`-`，`*`，`&`）既是一元运算符又是二元运算符
        - 从 *参数数量* 推断具体重载的是哪种
    ```
    +        -        *        /        %        ^
    &        |        ~        !        ,        =
    <        >        <=       >=       ++       --
    <<       >>       ==       !=       &&       ||
    +=       -=       /=       %=       ^=       &=
    |=       *=       <<=      >>=      []       ()
    ->       ->*      new      new[]    delete   delete[]
    ```
    - **不能**重载的内置运算符
    ```
    ::      .*        .        ? :
    ```
- 成员函数版本和非成员函数版本重载运算符的等价调用
    - `@`代表对应的 *前置* 、 *中置* 或 *后置* *运算符* 
    - `a`、`b`代表对应的 *操作数* 
    
| 表达式     | 成员函数              | 非成员函数          | 示例                                                        |
|-----------|----------------------|-------------------|-----------------------------------------------------------|
| `@a`      | `(a).operator@()`    | `operator@(a)`    | `!std::cin => std::cin.operator!()`                       |
| `a@`      | `(a).operator@(0)`   | `operator@(a, 0)` | `std::vector<int>::iterator i;`，`i++ => i.operator++(0)`  |
| `a @ b`   | `(a).operator@(b)`   | `operator@(a, b)` | `std::cout << 42 => std::cout.operator<<(42)`             |
| `a = b`   | `(a).operator=(b)`   | *必须为成员函数*     | `std::string s;`，`str = "abc" => str.operator=("abc")`    |
| `a(b...)` | `(a).operator(b...)` | *必须为成员函数*     | `std::greater(1, 2) => std::greater.operator()(1, 2)`     |
| `a[b]`    | `(a).operator[](b)`  | *必须为成员函数*     | `std::map<int, int> m;`，`m[1] => m.operator[](1)`         |
| `a->   `  | `(a).operator->()`   | *必须为成员函数*     | `std::unique_ptr<S> p;`，`p->bar() => p.operator->()`      |

- 直接调用重载的运算符函数
```
// equivalent calls to a nonmember operator function
data1 + data2;            // normal expression
operator+(data1, data2);  // equivalent function call

data1 += data2;           // expression-based ''call''
data1.operator+=(data2);  // equivalent call to a member operator function that
                          // implicitly binds this to its 1st parameter 
```

#### 重载守则

- 重载应使用与内置类型一致的含义
    - 类使用`I/O`操作，则将重载 *移位运算符* `<<`、`>>` 使其与内置类型的`I/O`保持一致
    - 如果类的某个操作是检查相等性，则定义`operator==`；如果类有了`operator==`，意味着它通常也应该有`operator!=`
    - 如果类包含一个内在的单序比较操作，则定义`operator<`；如果类有了`operator<`，意味着它通常也应该有 *其他关系操作* 
    - 重载运算符的 *返回类型* 通常情况下应与其内置版本的返回类型兼容
        - *逻辑运算符* 和 *关系运算符* 返回`bool`
        - *算术运算符* 返回 *类类型* 
        - *赋值运算符* 和 *符合赋值运算符* 返回 *左侧运算对象的左值引用* 
- 一般情况下**不应该**重载、 *逻辑与* 、 *逻辑或* 、 *逗号* 和 *取地址* 运算符
    - *逻辑与* `&&`， *逻辑或* `||`， *逗号* `,`：由于重载的运算符本质上是 *函数调用* ，运算对象求值顺序会变
    - *逻辑与* `&&`， *逻辑或* `||`：无法保留 *短路求值* 属性，运算对象一定都会被求值
    - *逗号* `,`， *取地址* `&`：`C++`已经定义了它们用于 *类对象* 时的语义，无需重载即可使用，硬要重载成不一样的，会破坏用户的三观
- 选择是否重载为 *成员函数* 
    - *赋值* `=`、 *调用* `()` 、 *下标* `[]`和 *成员访问箭头* `->` 必须是成员函数
    - *复合赋值运算符* 一般应为成员函数
    - 改变对象状态的运算符或者与给定类型关系密切的运算符，例如 *自增* 、 *自减* 和 *解引用* 运算符通常应该是成员函数
    - 具有对称性的运算符可能任意转换任意一个对象，例如 *算术* 、 *相等性* 、 *关系* 和 *位置* 运算符等，通常应为 *非成员函数* 
        - 如果`operator+`是`std::string`的成员函数，则第一个加法等价于`s.operator+("!")`，而第二个等价于`"hi".operator+(s)`
    ```
    std::string s = "world";
    std::string t = s + "!";   // ok: we can add a const char* to a string
    std::string u = "hi" + s;  // would be an error if + were a member of string
    ```
    - 如果 *非成员函数* 运算符需要接触私有成员，一般定义成 *友元函数*

#### 输入和输出流运算符（Input and Output Operators）

- `I/O`库分别使用`>>`和`<<`进行输入和输出操作，定义了对 *内置类型* 的版本，但对于自定义类类型则需人工重载
- 重载输出流运算符`<<`
    - 第一个形参是非常量`std::ostream`对象的引用
        - 非常量：因为输出会改变流对象的状态
        - 引用：流对象无法复制
    - 第二个形参是要输出的对象的常量引用
        - 输出操作不应该改变被输出对象的状态
    - 输出运算符应当 *尽量减少格式化操作* 
        - 尤其**不会**打印`std::endl`
    - `I/O`运算符 *必须* 是 *非成员函数* 
        - 如果需要输出私有数据成员，会定义成 *友元函数*
```
std::ostream & operator<<(std::ostream & cout, const Sales_data & item)
{
    cout << item.isbn() << " " << item.units_sold << " " << item.revenue << " " << item.avg_price();
    return cout;
}
```
- 重载输入流运算符`>>`
    - 第一个形参是非常量`std::istream`对象的引用
    - 第二个形参是要读入到的对象的非常量引用
    - 输入运算符 *必须* 处理 *输入失败* 的情况，而输出运算符不需要
```
std::istream & operator>>(std::istream & cin, Sales_data & item)
{
    // backup used to restore input when error
    Sales_data tmp = item;
    
    // no need to initialize; we'll read into price before we use it
    double price;                 
    cin >> item.bookNo >> item.units_sold >> price;
    
    // check that the inputs succeeded
    if (cin) 
    {
        item.revenue = item.units_sold * price;
    }
    else
    {
        item = std::move(tmp);  // input failed: give the object its input state
    }
        
    return cin;
}
```

#### 算术和关系运算符（Arithmetic and Relational Operators）

- 算术运算符
    - 接受两个 *常引用* ，返回 *新生成的副本* ，不是引用
    - 通常情况下 *算术运算符* 和 *关系运算符* 应定义为 *非成员函数* 以允许左右操作数互相转换
    - 如果类实现了 *算术运算符* ，则通常也会实现对应的 *复合赋值运算符* ，则应使用 *复合赋值运算符* 来实现 *算术运算符* 
```
// assumes that both objects refer to the same book
Sales_data operator+(const Sales_data &l hs, const Sales_data & rhs)
{
    Sales_data sum = lhs;  // copy data members from lhs into sum
    sum += rhs;            // add rhs into sum
    return sum;
}
```
- 相等运算符
    - 如果类支持判等，就应该实现`operator==`而**不是**具名函数
        - 便于记忆和使用
        - 更容易用于标准库容器和算法
    - 如果类定义了`operator==`，那么该运算符应该能判断一组给定的对象中是否含有 *重复数据* 
    - 如果类定义了`operator==`，那么也应该定义`operator!=`
    - `operator==`应当具有 *等价关系* 的三条性质： *自反性* 、 *对称性* 和 *传递性* 
    - `operator==`和`operator!=`中的一个应该把工作 *委托给另一个*  
- 关系运算符
    - 通常情况下， *关系运算符* 应该
        1. 定义 *顺序关系* ，令其与 *关联容器* 中对 *键* 的要求一致
        2. 如果类还含有`operator==`，则定义一种关系（比如 *小于* ）令其与之保持一致
            - 特别是，如果两个对象是`!=`的，则一个对象应 *小于* 另一个
    - 定义了 *相等运算符* 的类通常（但不总是）也会定义 *关系运算符*     
        - 如果存在唯一一种逻辑可靠的 *小于* 关系的定义，则应该考虑定义`operator<`
        - 如果类还包含`operator==`，则当且仅当`<`和`==`的定义不冲突时才定义`operator<`

#### 赋值运算符（Assignment Operators）

- *赋值运算符* 必须为 *成员函数* ，返回左操作数的 *左值引用* 
```
StrVec & StrVec::operator=(std::initializer_list<std::string> il)
{
    // alloc_n_copy allocates space and copies elements from the given range
    auto data = alloc_n_copy(il.begin(), il.end());
    // destroy the elements in this object and free the space
    free();                                          
    // update data members to point to the new space
    elements = data.first; 
    first_free = cap = data.second;
    return *this;
}

StrVec v;
v = {"a", "an", "the"};
```
- *复合赋值运算符* 通常情况下也应该
    - 定义为 *成员函数* 
    - 返回左操作数的 *左值引用* 
    - 用它实现 *算术运算*
```
// member binary operator: left-hand operand is bound to the implicit this pointer
// assumes that both objects refer to the same book
Sales_data & Sales_data::operator+=(const Sales_data & rhs)
{
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

Sales_data operator+(const Sales_data & lhs, const Sales_data & rhs)
{
    Sales_data res(lhs);
    return res += rhs;
}
```

#### 下标运算符（Subscript Operator）

- 下标运算符必须是 *成员函数* 
- 下标运算符通常定义两个版本
    - 一个返回 *普通引用* 
    - 一个是 *常量成员* ，并返回 *常量引用* 
```
class StrVec 
{
public:
    std::string & operator[](std::size_t n) { return elements[n]; }
    const std::string & operator[](std::size_t n) const { return elements[n]; }

private:
    std::string * elements;  // pointer to the first element in the array
};

// assume svec is a StrVec
const StrVec cvec = svec;    // copy elements from svec into cvec

// if svec has any elements, run the string empty function on the first one
if (svec.size() && svec[0].empty()) 
{
    svec[0] = "zero";        // ok: subscript returns a reference to a string
    cvec[0] = "Zip";         // error: subscripting cvec returns a reference to const
}
```

#### 自增和自减运算符（Increment and Decrement Operators）

- *迭代器* 类中通常定义自增运算符`++`和自减运算符`--`
- 因为改变的是所操作的对象的状态，建议将其设定为 *成员函数*
- 应当 *同时定义* *前置* 版本和 *后置* 版本
    - *前置* 版本返回自增或自减 *之后* 的对象的 *引用* ，**无**参数
    - *后置* 版本返回自增或自减 *之前* 的对象的 *拷贝* ，接受一个`int`类型参数
        - 因为不能仅靠返回值区分重载版本，因此由这个`int`类型参数作为和前置版本的区分
        - 编译器调用 *后置* 类型时，会自动传一个`0`
        - 后置运算符一般不用这个`0`，因此**不必**为之命名
```
// prefix: return a reference to the incremented/decremented object
StrBlobPtr & StrBlobPtr::operator++()
{
    // if curr already points past the end of the container, can't increment it
    check(curr, "increment past end of StrBlobPtr");
    // advance the current state
    ++curr; 
    return *this;
}

StrBlobPtr & StrBlobPtr::operator--()
{
    // move the current state back one element
    --curr; 
    // if curr is zero, decrementing it will yield an invalid subscript
    check(-1, "decrement past begin of StrBlobPtr");
    return *this;
}

// postfix: increment/decrement the object but return the unchanged value
StrBlobPtr StrBlobPtr::operator++(int)
{
    // no check needed here; the call to prefix increment will do the check
    StrBlobPtr ret = *this;  // save the current value
    ++*this;                 // advance one element; prefix ++ checks the increment
    return ret;              // return the saved state
}

StrBlobPtr StrBlobPtr::operator--(int)
{
    // no check needed here; the call to prefix decrement will do the check
    StrBlobPtr ret = *this;  // save the current value
    --*this;                 // move backward one element; prefix -- checks the decrement
    return ret;              // return the saved state
}
```
- 显式调用
```
StrBlobPtr p(a1);            // p points to the vector inside a1
p.operator++(0);             // call postfix operator++
p.operator++();              // call prefix operator++
```

#### 成员访问运算符（Member Access Operators）

- *解引用* 运算符`*`
    - *解引用* 运算符`*`通常是`const`类成员函数
        - 成员访问并不应该改变状态
- *箭头* 运算符`->`
    - *箭头* 运算符`->`必须是`const`类成员函数
        - 成员访问并不应该改变状态
    - 重载的 *箭头* 运算符必须返回 *类的指针* 或者 *自定义了箭头运算符的某个类的对象* 
        - `operator->()` 一般**不执行任何操作**，而是调用`operator*()`并返回其结果的 *地址* （即返回 *类的指针* ）
    - 重载箭头时，可以改变的是从 *哪个* 对象访问成员，不能改变的是访问成员这一事实
    - 形如`point->mem`的表达式等价于下面情况。除此之外，代码都将 *发生错误* 
    ```
    (*point).mem;                       // point is a built-in pointer type
    point.operator()->mem;              // point is an object of class type
    ```
    - `point->mem`的执行过程如下
        1. 如果`point`是 *指针* ，则应用内置的箭头运算符，表达式等价于`(*point).mem`
            - 首先解引用指针，然后从从所得的对象中获取指定成员
            - 如果指定成员`mem`不存在，则报错
        2. 如果`point`是 *定义了`operator->()`的类的一个对象* ，则使用`point.operator->()`的 *结果* 来获取`mem`
            - 如果 *该结果* 是一个 *指针* ，则 *执行第`1`步* 
            - 如果 *该结果* *本身含有重载的`operator->()`* ，则 *重复调用当前步骤* 
            - 最终，过程结束，程序返回所需内容或报错
```
std::string & StrBlobPtr::operator*() const
{ 
    // check whether curr is still valid
    std::shared_ptr<std::vector<std::string>> p = check(curr, "dereference past end");
    return (*p)[curr];                  // (*p) is the vector to which this object points
}

std::string * StrBlobPtr::operator->() const
{ 
    return & this->operator*();         // delegate the real work to the dereference operator
}

StrBlob a1 = {"hi", "bye", "now"};
StrBlobPtr p(a1);                       // p points to the vector inside a1
*p = "okay";                            // assigns to the first element in a1
std::cout << p->size() << std::endl;    // prints 4, the size of the first element in a1
std::cout << (*p).size() << std::endl;  // equivalent to p->size()
```

#### 函数调用运算符（Function-Call Operator）

- *函数调用运算符* 必须是 *成员函数*
- 重载了 *函数调用运算符* `operator()()`的类的对象可以像函数一样被调用，被称作 *函数对象* （function object）
    - 类能存储 *状态* ，相比普通函数更灵活
    - 函数对象常常作为谓词的一种用于标准库算法中，例如`std::sort(b, e, std::greater<T>())`
```
template <class T = void>
struct greater
{
    bool operator()(const T & lhs, const T & rhs) const 
    {
        return lhs > rhs;
    }
}
```
- `lambda`表达式的本质是函数对象
    - 具体实现讲`lambda`表达式的时候已经说过了，小作用域内的闭包类`Closure`的函数对象
- 标准库定义的函数对象
    - 定义于`<funtional>`
    - 讲泛型算法的时候说过了
- *调用签名* （call signature）
    - 指明了调用返回的类型以及传递给调用的实参类型，与函数类型一一对应
    - 格式：`result_type (first_argument_type, second_argument_type...)`
    ```
    bool (int, int)  // e.g. signature of std::greater<int>
    ```
    - 标准库[`std::function`](https://en.cppreference.com/w/cpp/utility/functional/function)类型
        - 定义于`<funtional>`
        - 操作
            - `std::function<T> f;`：`f`是一个用来存储 *签名* 为`T`的 *可调用对象* 的空`std::function<T>`
            - `std::function<T> f(nullptr);`显式地构造一个空`std::function<T>`
            - `std::function<T> f(obj);`：用`obj`拷贝构造`std::function<T>`
            - `f`：将`f`作为 *条件* ，当含有可调用对象时为`true`，否则为`false`
            - `f(args)`：调用`f`中的对象，实参列表为`args`
        - 静态类型成员
            - `result_type`：该`std::function<T>`类型的可调用对象的返回值类型
            - `argument_type`，`first_argument_type`，`second_argument_type`：当`T`有一个或两个实参时定义的类型。
                - 一个：`argument_type`和`first_argument_type`等价
                - 两个：`first_argument_type`，`second_argument_type`分别代表第一个和第二个实参的类型
    ```
    int add_func(int a, int b) { return a + b; }
    int (*add_fp)(int, int) = add_func;
    
    std::function<int (int, int)> f1 = add_fp;                               // function pointer
    std::function<int (int, int)> f2 = std::add<int>();                      // object of a function-object class
    std::function<int (int, int)> f3 = [] (int i, int j) { return i + j; };  // lambda
    
    std::cout << f1(4, 2) << std::endl;                                      // 6
    std::cout << f2(4, 2) << std::endl;                                      // 6
    std::cout << f3(4, 2) << std::endl;                                      // 6
    
    auto mod = [] (int i, int j) { return i % j; };
    
    struct div 
    {
        int operator()(int denominator, int divisor) 
        {
            return denominator / divisor;
        }
    };
    
    std::map<std::string, std::function<int (int, int)>> binops = \
    {
        {"+", add_fp},                                                       // function pointer
        {"-", std::minus<int>()},                                            // library function object
        {"*", [] (int i, int j) { return i * j; }},                          // unnamed lambda
        {"/", div()},                                                        // user-defined function object
        {"%", mod}                                                           // named lambda
    };
    
    binops["+"](10, 5);                                                      // 15
    binops["-"](10, 5);                                                      // 5
    binops["*"](10, 5);                                                      // 50
    binops["/"](10, 5);                                                      // 2
    binops["%"](10, 5);                                                      // 0
    ```
    - **不能**将 *重载函数* 的名字存入`std::function`中
        - 会有 *二义性*
        - 解决方法
            1. 传 *函数指针* 
            2. *用`lambda`调用* `add`
    ```
    int add(int i, int j) { return i + j; }
    Sales_data add(const Sales_data &, const Sales_data &);
    std::map<std::string, std::function<int (int, int)>> binops;
    binops.insert({"+", add});  // error: which add?
    
    int (*fp)(int,int) = add;   // pointer to the version of add that takes two ints
    binops.insert({"+", fp});   // ok: fp points to the right version of add
    
    // ok: use a lambda to disambiguate which version of add we want to use
    binops.insert({"+", [] (int a, int b) { return add(a, b); }});
    ```

#### 类型转换运算符（Conversion Operators）

- *用户定义转换* （user-defined conversions）
    - 又称 *类类型转换* （class-type conversions），包括
        - *转换构造函数* (conversion constructor)
        - *类型转换运算符* (conversion operator)
- *类型转换运算符* 
    - 类的一种特殊 *成员函数* ，负责将该类类型转换为`type`类型
        - **没有**显式的返回类型
        - **没有**形参
        - 必须定义成 *类成员函数* 
        - 一般定义成`const`成员
            - 类型转换运算符**不应该**改变待转换对象的内容
    ```
    operator type() const;
    ```
    - 可以面向**除`void`之外**的任何 *能被函数返回的* 类型进行定义
        - **不能**转换成 *数组* 或 *函数* ，但可以转换成这俩的 *指针* 或 *引用* 
    - *显式类型转换运算符* (explicit conversion operator)
        - 告诉编译器**不能**用此运算符进行 *隐式类型转换* (implicit conversion)
        - 一个**例外**：表达式被用作 *条件* 时，类型转换运算符即使是`explicit`的，仍会被 *隐式应用* **例外**：表达式被用作 *条件* 时，类型转换运算符即使是`explicit`的，仍会被 *隐式应用* 
            - `if`，`while`，`do while`语句的条件部分
            - `for`语句头的条件表达式
            - 逻辑非运算符`!`、逻辑与运算符`&&`、逻辑或运算符`&&`的运算对象
            - 条件运算符`? :`的条件表达式
    - 一般很少定义类型转换运算符，因为用户会感到意外而不是舒适
        - 除了向`bool`的类型转换运算符
            - 通常用在条件部分，`operator bool()`一般定义成`explicit`的 => `struct S35`
            - 那真是人手一个，谁用谁说好，大家都习惯了
        - 应当**避免** *二义性型转换运算符* ，比如
        ```
        struct A 
        {
            A(int = 0);               // usually a bad idea to have two
            A(double);                // conversions from arithmetic types
            operator int() const;     // usually a bad idea to have two
            operator double() const;  // conversions to arithmetic types
            // other members
        };
        
        void f2(long double);
        A a;
        f2(a);                        // error ambiguous: f(A::operator int()) or f(A::operator double())
        long lg;
        A a2(lg);                     // error ambiguous: A::A(int) or A::A(double)
        ```






### 🌱 [Chap 15] 面向对象程序设计（Object-Oriented Programming，`OOP`）

#### `OOP`概述

- 核心思想
    1. *数据抽象* （data abstraction）
        - 类的 *接口* 与 *实现* 分离
    2. *继承* （inheritance）
        - 定义相似的类型并对其相似关系建模
    3. *动态绑定* （dynamic binding）
        - *多态性* （Polymorphism）：一定程度上忽略相似类型的区别，以统一的方式使用它们的对象
- 继承
    - *基类* （base class）产生 *派生类* （derived class）
        - 派生类通过 *派生类列表* （class derivation list）指出他继承谁
        ```
        class BulkQuote : public Quote 
        {
            // ...
        }
        ```
    - *虚函数* （virtual function）：由派生类各自实现更适合自身的版本
        - 派生类中重新定义的虚函数也需要声明`virtual`
        - 通过`override`限定符显式注明此函数是改写的基类的虚函数（此时不必再加`virtual`）
- 动态绑定
    - 使用基类的 *引用* 、 *对象指针* 或 *成员指针* 调用虚函数时，将发生动态绑定
    - 调用指针或引用实际指向的对象的函数

#### 继承

- 基类
    - `Quote`类定义
    ```
    class Quote
    {
    public:
        Quote() = default;
        Quote(std::string book, double sales_price) : bookNo(std::move(book)), price(sales_price) {}
        virtual ~Quote() = default;  // dynamic binding for the destructor, see 15.7.1 for virtual destructors

        std::string isbn() const
        {
            return bookNo;
        }

        // returns the total sales price for the specified number of items
        // derived classes will override and apply different discount algorithms
        virtual double net_price(std::size_t n) const
        {
            return n * price;
        }

    protected:
        double price = 0.0;          // normal, undiscounted price

    private:
        std::string bookNo;          // ISBN number of this item
    };
    ```
    - 基类通常应该定义一个 *虚析构函数* ，即使这个函数不执行任何操作也是如此
        - 为了`delete base_ptr`时能正确调用到派生类的析构函数
    - 成员函数和继承
        - 派生类可以 *覆盖* （override）基类函数
            - 基类希望派生类覆盖的函数： *虚函数* （virtual function） => 15.3
                - 函数声明语句之前加上`virtual`
                - *只能* 出现于类内部的声明语句之前
                - 执行 *动态绑定* 
                - 基类中的虚函数在派生类中也 *隐式* 地是虚函数
            - 非虚函数解析过程发生于编译时而不是执行时
    - 访问控制和继承
        - 派生类**不能**访问基类的`private`成员
        - `protected`除了能被派生类访问到以外，其余和`private`一样
    - 防止继承
        - 在类名后面跟一个`final`关键字可以防止此类被继承
    ```
    class NoDerived final { /* */ };    // NoDerived can't be a base class
    class Base { /* */ };
    // Last is final; we cannot inherit from Last
    class Last final : Base { /* */ };  // Last can't be a base class
    class Bad : NoDerived { /* */ };    // error: NoDerived is final
    class Bad2 : Last { /* */ };        // error: Last is final
    ```
- 派生类
    - `BulkQuote`类定义
    ```
    // BulkQuote inherits from Quote
    class BulkQuote : public Quote
    {
        BulkQuote() = default;

        BulkQuote(std::string book, double p, std::size_t qty, double disc) : 
                Quote(std::move(book), p), min_qty(qty), discount(disc)
        {
        }
        

        // overrides the base version in order to implement the bulk purchase discount policy
        double net_price(std::size_t cnt) const override
        {
            return cnt >= min_qty ? cnt * (1 - discount) * price : cnt * price;
        }

    private:
        std::size_t min_qty = 0;  // minimum purchase for the discount to apply
        double discount = 0.0;    // fractional discount to apply
    };
    ```
    - 派生类在 *定义时* 使用 *类派生列表* （class derivation list）指出自己继承的类
        - 格式
            - 每个基类之前都可以有三种 *访问说明符* 之一
            - *默认* 为`public`
        ```
        class Derived : public Base1, private Base2, protected Base3...
        {
            // ...
        };
        ```
        - 类派生列表只能出现于定义处，**不能**出现于声明中
        ```
        class BulkQuote : public Quote;          // error: derivation list can't appear here
        class BulkQuote;                         // ok: right way to declare a derived class
        ```
        - 类派生列表中的基类必须 *已经定义* ，**不能**仅是声明过的
        ```
        class Quote;                             // declared but not defined
        class BulkQuote : public Quote { ... };  // error: Quote must be defined
        ```
        - *直接基类* （direct base）和 *间接基类* （indirect base）
        ```
        class Base { /* ... */ } ;               // direct base for D1, indirect for D2
        class D1: public Base { /* ... */ };
        class D2: public D1 { /* ... */ };
        ```
    - 派生类中的虚函数
        - 派生类经常（但并不总是）覆盖它继承的虚函数
        - 没有覆盖则直接使用继承到的基类的版本
        - 可以在覆盖的函数前继续使用`virtual`关键字
    - `C++`**并未**规定派生类的对象在内存中如何分布
        - 基类成员和派生类新成员很可能是混在一起、而非泾渭分明的
    - 派生类构造函数
        - 每个类控制它自己的成员初始化过程
            - 派生类并不默认调用基类构造函数，自然也不能直接初始化从基类继承来的成员
            - 除非特别指出，派生类对象的基类部分会像数据成员一样执行 *默认初始化* 
        - 派生类构造函数应 *首先调用基类构造函数* 初始化 *基类部分* ， *之后* 再按照 *声明的顺序* 依次初始化 *派生类成员* 
    - 派生类使用基类成员
        - 派生类对象中含有与其基类对应的组成部分，这一事实是继承的关键所在
        - 派生类可以访问基类的 *公有* 及 *受保护* 成员
    - 继承与静态成员
        - 如果基类定义了 *静态成员* ，则在整个体系中只存在该成员的 *唯一* 定义
            - 不论从基类中派生出多少派生类，它们都 *与基类共享同一个静态成员实例* 
        - 静态成员遵循通用的 *访问控制* 规则，即
            - 派生类和基类都能访问基类的 *公有* 或 *受保护* 成员
            - 派生类**不能**访问基类的 *私有* 成员
    ```
    class Base 
    {
    public:
        static void statmem();
    };
    
    class Derived : public Base 
    {
        void f(const Derived &);
    };
    
    void Derived::f(const Derived & derived_obj)
    {
        Base::statmem();         // ok: Base defines statmem
        Derived::statmem();      // ok: Derived inherits statmem
        
        // ok: derived objects can be used to access static from base
        derived_obj.statmem();   // accessed through a Derived object
        statmem();               // accessed through this object
    }
    ```
- 类型转换与继承
    - *派生类到基类的* （derived-to-base）类型转换
        - 编译器 *隐式* 执行
        - 可以把 *派生类对象* 当成 *基类对象* 使用（此时派生类部分被 *切掉* （sliced down））
        - 可以把 *基类指针或引用* 绑定到 *派生类对象* 上，通过此指针或引用访问对象时
            - 成员访问仅限基类成员
            - 虚函数调用执行 *动态绑定* 
        - *智能指针* 和 *内置指针* 一样，都支持派生类向基类的类型转换
    ```
    BulkQuote bulk;              // object of derived type
    Quote item(bulk);            // uses the Quote::Quote(const Quote&) constructor
    item = bulk;                 // calls Quote::operator=(const Quote &)
    
    Quote item;                  // object of base type
    BulkQuote bulk;              // object of derived type
    Quote * p = &item;           // p points to a Quote object
    p = & bulk;                  // p points to the Quote part of bulk
    Quote & r = bulk;            // r bound to the Quote part of bulk
    ```
    - *静态类型* （static type）和 *动态类型* （dynamic type）
        - 如果表达式既不是 *指针* 也不是 *引用* ，则其 *动态类型* 与 *静态类型* 一致
        - 基类的 *指针* 或 *引用* 的 *静态类型* 和它所表示对象的 *动态类型* 可能不同
            - *静态类型* ：编译时已知，变量声明时的类型或表达式生成的类型
            - *动态类型* ：变量或表达式所表示的内存中的对象的类型。知道运行时才可知
    - **不存在**基类向派生类的 *隐式类型转换* 
        - 多态的基类指针或引用一样**无法**隐式转换为派生类
        - 编译器只能检查静态类型确定类型安全，如果基类中含有虚函数，可以使用`dynamic_cast`请求显式类型转换，进行运行时安全检查 => 19.2.1
    ```
    Quote base;
    BulkQuote * bulkP = &base;   // error: can't convert base to derived
    BulkQuote & bulkRef = base;  // error: can't convert base to derived

    BulkQuote bulk;
    Quote * itemP = &bulk;       // ok: dynamic type is BulkQuote
    BulkQuote * bulkP = itemP;   // error: can't convert base to derived
    ```

#### 虚函数

- 所有虚函数都必须被定义
    - *普通函数* 如不被使用， *可以不被定义* 
    - *虚函数* 不管有没有被用到，都 *必须提供定义* 
- 动态绑定只在通过基类指针或引用调用虚函数时才会发生
```
Quote base("0-201-82470-1", 50);
print_total(std::cout, base, 10);     // calls Quote::net_price
BulkQuote derived("0-201-82470-1", 50, 5, .19);
print_total(std::cout, derived, 10);  // calls BulkQuote::net_price
```
- 派生类中的虚函数
    - 在派生类中覆盖某个虚函数时，可以重复`virtual`关键字，但不是必须
        - 基类中的虚函数在其所有派生类中都默认还是虚函数
    - 派生类中的虚函数的 *函数签名* 必须和基类的版本 *完全一致* 
        - 如不一致，则会被理解成重载的新函数，**无法**执行动态绑定
    - `final`和`override`说明符
        - `override`说明符用于显式指定派生类中的虚函数，编译器会对 *函数签名* 执行检查，帮助发现错误
            - 在 *形参列表* 之后，或`const`限定符之后（如有）、或引用限定符之后（如有）、或尾置返回类型之后（如有）使用`override`关键字
            - `override`函数必须在基类中是虚函数
            - `override`函数的签名必须与基类版本一致
            - 此时不必再加`virtual`
        ```
        struct B 
        {
            virtual auto f1(int) const & -> void;
            virtual void f2();
            void f3();
        };
        
        struct D1 : B 
        {
            auto f1(int) const & -> void override;  // ok: f1 matches f1 in the base
            void f2(int) override;                  // error: B has no f2(int) function
            void f3() override;                     // error: f3 not virtual
            void f4() override;                     // error: B doesn't have a function named f4
        }
        ```
        - `final`说明符用于指定此函数**不能**被派生类覆盖
            - 在 *形参列表* 之后，或`const`限定符之后（如有）、或引用限定符之后（如有）、或尾置返回类型之后（如有）使用`final`关键字
            - 此时不必再加`virtual`
        ```
        struct D2 : B 
        {
            // inherits f2() and f3() from B and overrides f1(int)
            void f1(int) const final;     // subsequent classes can't override f1(int)
        };
            
        struct D3 : D2 
        {
            void f1(int) const;           // error: D2 declared f2 as final
            void f2();                    // ok: overrides f2 inherited from the indirect base, B
        };
        ```
    - 虚函数和 *默认实参*
        - 虚函数也可以有默认实参
        - 通过动态绑定调用的派生类虚函数，传入的默认实参是 *基类版本* 的
        - 如果虚函数使用默认实参，**必须**和基类中的定义一致
    - 回避虚函数机制
        - 如果不想使用动态绑定，可以通过 *域运算符* `::`强制执行某一版本的虚函数
        ```
        // calls the version from the base class regardless of the dynamic type of baseP
        double undiscounted = baseP->Quote::net_price(42);
        ```
        - 一般只有成员函数或友元才需要使用域运算符来回避虚函数机制
            - 当派生类虚函数调用其覆盖的基类的虚函数版本时需要强制执行某一版本的虚函数
            - 此时基类版本通常完成继承层次中所有类型都要做的共同任务
            - 而派生类版本只负责执行与派生类本身密切相关的操作
        - 如果派生类虚函数需要调用它的基类版本，但是没有使用域运算符，则在运行时该调用将被解析为递归调用自己，将导致 *死递归* 

#### 抽象基类（abstract base class）

- *纯虚函数* （Pure Virtual Functions）
    - 将函数定义为 *纯虚* 的，明确告知用户定义此函数没有意义
    - 纯虚函数不需定义，而是用`= 0;`代替函数体
        - `= 0;` *只能* 出现于类内部虚函数声明语句处
        - 也可以为纯虚函数提供定义，不过 *必须在类外单独定义* 
- `DiscQuote`类定义
```
// class to hold the discount rate and quantity
// derived classes will implement pricing strategies using these data
class DiscQuote : public Quote 
{
public:
    DiscQuote() = default;
    
    DiscQuote(const std::string & book, double price, std::size_t qty, double disc) :
        Quote(book, price), quantity(qty), discount(disc) 
    { 
    }
    
    double net_price(std::size_t) const = 0;  // pure virtual function
    
protected:
    std::size_t quantity = 0;                 // purchase size for the discount to apply
    double discount = 0.0;                    // fractional discount to apply
};
```
- *抽象基类* 就是含有 *纯虚函数* 的类
    - 负责定义 *接口* ，后续派生类负责实现接口
        - 隔壁`Java`更狠，直接搞了个`interface`出来，就相当于`C++`里的 *抽象基类* 
        - 于是就有了`class Derived extends Base implements Interface`这种操作
        - 虽说`Java`不能直接搞多重继承，这也算能凑合用了吧
    - **不能**创建纯虚基类的对象
    - 只能定义确实覆盖了纯虚函数的派生类的对象
```
// Disc_quote declares pure virtual functions, which Bulk_quote will override
DiscQuote discounted;                            // error: can't define a Disc_quote object
BulkQuote bulk;                                  // ok: Bulk_quote has no pure virtual functions
```
- `BulkQuote`类重定义
```
// the discount kicks in when a specified number of copies of the same book are sold
// the discount is expressed as a fraction to use to reduce the normal price
class BulkQuote : public DiscQuote 
{
public:
    BulkQuote() = default;
    
    BulkQuote(const std::string & book, double price, std::size_t qty, double disc):
        Disc_quote(book, price, qty, disc)
    { 
    }
    
    // overrides the base version to implement the bulk purchase discount policy
    double net_price(std::size_t) const override;
};
```
- *重构* （refactoring）
    - 在类的继承体系中添加 *抽象基类* 就是 *重构* 操作
    - 重构负责重新设计类的体系以便将操作和（或）数据从一个类移动到另一个类中
    - 重构不需重新编写已有代码，但需重新编译

#### 访问控制与继承

- 公有，私有和受保护成员
    - *受保护成员* 
        - 对于类的用户不可访问
        - 对于派生类的成员及其友元可访问
            - 派生类的成员及其友元 *只能* 通过 *派生类对象* 访问基类的受保护成员
            - 派生类的成员及其友元**不能**通过 *基类对象* 访问基类的受保护成员
                - 很好理解，友元**不能**传递、**不能**继承，儿子的哥们又不是老子的哥们
    ```
    class Base 
    {
    protected:
        int prot_mem;                   // protected member
    };
    
    class Sneaky : public Base 
    {
        friend void clobber(Sneaky &);  // can access Sneaky::prot_mem
        friend void clobber(Base &);    // can't access Base::prot_mem
        
        int j;                          // j is private by default
    };
    
    // ok: clobber can access the private and protected members in Sneaky objects
    void clobber(Sneaky & s) 
    { 
        s.j = s.prot_mem = 0; 
    }
    
    // error: clobber can't access the protected members in Base
    void clobber(Base & b) 
    { 
        b.prot_mem = 0; 
    }
    ```
- 公有，私有和受保护继承
    - 公有继承：保持基类中的访问控制不变
    - 私有继承：基类全部内容一律变为私有
    - 受保护继承：基类 *公有* 内容一律变为受保护
```
class Base 
{
public:
    void pub_mem(); // public member
    
protected:
    int prot_mem;   // protected member
    
private:
    char priv_mem;  // private member
};

struct PubDerv : public Base 
{
    // ok: derived classes can access protected members
    int f() { return prot_mem; }
    
    // error: private members are inaccessible to derived classes
    char g() { return priv_mem; }
};

struct PrivDerv : private Base 
{
    // private derivation doesn't affect access in the derived class
    int f1() const { return prot_mem; }
};

PubDerv d1;         // members inherited from Base are public
PrivDerv d2;        // members inherited from Base are private
d1.pub_mem();       // ok: pub_mem is public in the derived class
d2.pub_mem();       // error: pub_mem is private in the derived class
```
- 派生类向基类的转换是否可访问由使用该转换的代码决定，同时派生类的访问说明符也会有影响
    - 对于 *用户代码* 中某个节点来说，当且仅当 *基类公有成员可访问* 时， *派生类向基类的类型转换可用* 
        - 反之，则**不可用**
    - 具体来说，假定`D`继承自`B`
        - 当且仅当`D` *公有继承* `B`时， *用户代码* 才能使用派生类向基类的转换
            - *私有继承* 和 *受保护继承* 则**不能**使用
        - 不论`D`如何继承`B`，*`D`的成员和友元* 都能使用派生类向基类的转换
            - 派生类向直接基类的类型转换对派生类的成员和友元 *永远可见* 
        - 如果`D` *公有继承* 或 *受保护继承* `B`，则 *`D`的派生类的成员和友元* 都能使用`D`向`B`的转换
            - *私有继承* 则**不行**
- 友元和继承
    - 友元**不能**传递、**不能**继承，哪怕是基类和派生类之间
        - 派生类的友元一样只能通过基类对象访问基类的公有成员
        - 但可以通过派生类对象访问到派生类的基类部分
- 改变个别成员的 *可访问性* 
    - 使用`using`声明在对应的访问限定符下指明基类成员
    - 只能对 *派生类可见* 的名字使用`using`声明
        - 也就是说基类的`private`必须是没救的
```
class Base 
{
public:
    std::size_t size() const { return n; }
    
protected:
    std::size_t n;
};

class Derived : private Base  // note: private inheritance
{ 
public:
    // maintain access levels for members related to the size of the object
    using Base::size;
    
protected:
    using Base::n;
};
```
- 默认的继承保护级别
    - `struct`成员默认 *公有* ，继承时默认 *公有继承*
    - `class`成员默认 *私有* ，继承时默认 *私有继承*
    - 这也是`struct`和`class`唯一的区别
    
#### 继承中的类作用域

- 派生类的作用域 *嵌套* 在其基类的作用域之内
    - 每个类拥有自己的 *类作用域* 
    - 如果一个名字在派生类作用域内无法解析，则编译器会 *回溯至其上一级作用域* （即其直接基类的作用域）
    - 这也解释了为什么动态绑定的基类指针和引用虽然实际指向派生类对象，但却无法通过它们访问派生类成员
        - 因为 *名字查找* 直接从基类作用域开始了，自然找不到派生类作用域里才有的东西
- 名字冲突和继承
    - 派生类可以重用定义在其直接或间接基类中的名字
        - 此时定义在内部（派生类）作用域的名字将隐藏定义在外部（基类）作用域中的 *同名实体* 
            - 包括对象和函数
        - 可以通过 *作用域运算符* 显式使用被隐藏的成员
    - 除了继承来的 *虚函数* ，派生类**不应该**重用那些定义在其基类中的名字
    ```
    struct Base 
    {
    public:
        Base(): mem(0) {}
        
    protected:
        int mem;
    };
    
    struct Derived : Base 
    {
    public:
        Derived(int i): mem(i) {}           // initializes Derived::mem to i
        
        // Base::mem is default initialized
        int get_mem() { return mem; }       // returns Derived::mem
    
    protected:
        int mem;                            // hides mem in the base
    };
    
    Derived d(42);
    std::cout << d.get_mem() << std::endl;  // prints 42
    ```
    - *名字查找* 先于 *类型匹配* 
        - *函数重载* 一节中已经强调过，不同的作用域中**无法**重载函数
        - 同理，派生类中无法重载基类的函数，如果函数同名，将在其作用域内 *隐藏* **而不是**重载该基类成员 
            - 即使形参列表不一样，也仍旧是隐藏而不是重载
        - 仍然可以通过 *作用域运算符* 显式指定访问哪个版本
        ```
        struct Base 
        {
            int memfcn();
        };
        
        struct Derived : Base 
        {
            int memfcn(int);  // hides memfcn in the base
        };
        
        Derived d; 
        Base b;
        b.memfcn();           // calls Base::memfcn
        d.memfcn(10);         // calls Derived::memfcn
        d.memfcn();           // error: memfcn with no arguments is hidden
        d.Base::memfcn();     // ok: calls Base::memfcn
        ```
    - 虚函数与作用域
        - 派生类覆盖的虚函数必须和基类具有相同的签名
        - 否则无法通过基类指针或引用调用派生类版本的虚函数
    ```
    class Base 
    {
    public:
        virtual int fcn();
    };
    
    class D1 : public Base 
    {
    public:
        // hides fcn in the base; this fcn is not virtual
        // D1 inherits the definition of Base::fcn()
        int fcn(int);       // parameter list differs from fcn in Base
        
        virtual void f2();  // new virtual function that does not exist in Base
    };
    
    class D2 : public D1 
    {
    public:
        int fcn(int);       // nonvirtual function hides D1::fcn(int)
        int fcn();          // overrides virtual fcn from Base
        void f2();          // overrides virtual f2 from D1
    };
    
    Base bobj; 
    D1 d1obj; 
    D2 d2obj;
    
    Base * bp1 = &bobj;
    Base * bp2 = &d1obj;
    Base * bp3 = &d2obj;
    
    bp1->fcn();             // virtual call, will call Base::fcn at run time
    bp2->fcn();             // virtual call, will call Base::fcn at run time
    bp3->fcn();             // virtual call, will call D2::fcn at run time
    
    D1 * d1p = &d1obj; 
    D2 * d2p = &d2obj;
    
    bp2->f2();              // error: Base has no member named f2
    d1p->f2();              // virtual call, will call D1::f2() at run time
    d2p->f2();              // virtual call, will call D2::f2() at run time
    
    Base * p1 = &d2obj; 
    D1 * p2 = &d2obj; 
    D2 * p3 = &d2obj;
    
    p1->fcn(42);            // error: Base has no version of fcn that takes an int
    p2->fcn(42);            // statically bound, calls D1::fcn(int)
    p3->fcn(42);            // statically bound, calls D2::fcn(int)
    ```
    - 覆盖重载的函数
        - 成员函数不论是否是虚函数都能被重载
        - 派生类可以覆盖重载函数的零或多个实例
        - 如果派生类希望所有重载版本均对齐可见，则它要么需要覆盖所有重载，要么一个也不覆盖
        - 为重载的成员提供`using`声明，就无须覆盖每一个版本了

#### 构造函数与拷贝控制

- *虚析构函数* （virtual destructor）
    - 继承关系对类拷贝控制最直接的影响就是基类通常应该定义一个 *虚析构函数* ，用于动态分配继承体系中的对象
        - 虚析构函数可以保证析构对象时执行正确的版本
            - 当`delete`动态分配对象指针时，将执行析构函数
            - 如果基类的析构函数不是虚函数，则`delete`动态绑定到派生类的指针是 *未定义行为* 
        - 基类的虚析构函数很可能是空的，此时并不需要遵循 *三五法则* 
            - 虚析构函数将阻止编译器自动 *合成移动操作* 
    ```
    class Quote 
    {
    public:
        // virtual destructor needed if a base pointer pointing to a derived object is deleted
        virtual ~Quote() = default;  // dynamic binding for the destructor
    };
    
    Quote * itemP = new Quote;       // same static and dynamic type
    delete itemP;                    // destructor for Quote called
    itemP = new BulkQuote;           // static and dynamic types differ
    delete itemP;                    // destructor for Bulk_quote called
    ```
- 合成拷贝控制与继承
    - 某些基类的定义方式会导致派生类的合成拷贝控制成员被定义成 *删除的* 
        - *基类* 的 *默认构造函数* 、 *拷贝构造函数* 、 *拷贝赋值运算符* 或 *析构函数* 是 *删除的或者不可访问* 的，则 *派生类* 中 *对应成员* 将是 *被删除的* 
            - 因为编译器**不能**使用基类成员来执行派生类对象基类部分的构造、复制或销毁操作
        - 如果在 *基类* 中有一个 *不可访问或删除* 掉的 *析构函数* ，则 *派生类* 中 *合成的默认构造函数和拷贝构造函数* 将是 *被删除* 的
            - 因为编译器**无法**销毁派生类对象的基类部分
        - 编译器将**不会**合成一个 *删除掉的移动操作* 
            - 当我们使用`= default;`请求一个移动操作时，如果基类中的对应操作是删除的或不可访问的，那么派生类中该函数将是被删除的
                - 原因是派生类对象的基类部分不可移动
            - 同样，如果基类的析构函数是删除的或不可访问的，则派生类的移动构造函数也将是被删除的
    ```
    class B 
    {
    public:
        B();
        B(const B &) = delete;
        // other members, not including a move constructor
    };
    
    class D : public B 
    {
        // no constructors
    };
    
    D d;                 // ok: D's synthesized default constructor uses B's default constructor
    D d2(d);             // error: D's synthesized copy constructor is deleted
    D d3(std::move(d));  // error: implicitly uses D's deleted copy constructor 
                         // (no synthesized move constructor as copy constructor is user-defined => 13.6.2)
    ```
    - 派生类中需要执行移动操作时，应先在基类中定义
        - 基类缺少移动操作会阻止派生类有自己的合成移动操作
        - 基类可以使用合成的版本，但必须显式定义`= default;`
        - 这种情况下基类需要遵循 *三五法则* 
    ```
    class Quote 
    {
    public:
        Quote() = default;                           // memberwise default initialize
        Quote(const Quote &) = default;              // memberwise copy
        Quote(Quote &&) = default;                   // memberwise copy
        Quote & operator=(const Quote &) = default;  // copy assign
        Quote & operator=(Quote &&) = default;       // move assign
        virtual ~Quote() = default;
        // other members as before
    };
    ```
- 派生类拷贝控制成员
    - 具体职责
        - 派生类 *构造函数* 要 *同时负责* 初始化 *自己和基类* 的部分
            - 首先调用 *基类对应成员负责基类部分* ，再做自己的那部分
        - 派生类 *拷贝构造函数* 和 *移动构造函数* 要 *同时负责* 拷贝和移动 *自己和基类* 的部分
            - 首先调用 *基类对应成员负责基类部分* ，再做自己的那部分
        - 派生类 *赋值运算符* 也必须 *同时负责自己和基类* 的部分
            - 首先调用 *基类对应成员负责基类部分* ，再做自己的那部分
        - 派生类 *析构函数* *只负责* 销毁派生类 *自己* 分配的资源
            - *编译器会自动调用基类析构函数* 销毁派生类对象的基类部分
    - 基类的拷贝控制成员中， *析构函数* 必须是虚函数，除 *析构函数* 外均**不应**定义为虚函数
        - 也就是解引用基类指针并赋值是**不好**的
    - 派生类 *拷贝或移动构造函数* 
        - **必须**显式调用基类对应构造函数，否则基类部分将被 *默认初始化* ，产生 *未定义值* 
        - 对移动构造函数初始化器列表，应委托`Base(std::move(d))`
    ```
    class Base { /* ... */ } ;

    class D: public Base 
    {
    public:
        // by default, the base class default constructor initializes the base part of an object
        // to use the copy or move constructor, we must explicitly call that
        // constructor in the constructor initializer list
        
        D(const D & d): Base(d)        // copy the base members
        /* initializers for members of D */ { /* ... */ }
        
        D(D && d): Base(std::move(d))  // move the base members
        /* initializers for members of D */ { /* ... */ }
        // note: we are using (derived part) of moved object d
        // d's base part should NOT be accessed, but the derived part will remain valid
    };
    ```
    - 派生类 *赋值运算符* 
        - 必须显式调用基类对应版本为其基类部分赋值
    ```
    // Base::operator=(const Base &) is NOT invoked automatically
    D & D::operator=(const D & rhs)
    {
        Base::operator=(rhs); // assigns the base part
        // assign the members in the derived class, as usual,
        // handling self-assignment and freeing existing resources as appropriate
        return *this;
    }
    ```
    - 派生类 *析构函数* 
        - 派生类析构函数 *只负责* 销毁 *自己* 分配的资源
            - 析构函数体执行完后，对象的成员会被 *隐式销毁* 
            - 类似地，派生类对象的基类部分也是在派生类析构函数执行完后、由编译器 *隐式* 调用基类析构函数销毁的
        - 对象销毁的顺序和被创建的顺序正好相反：派生类虚构函数先执行，然后是其直接基类的析构函数，依此类推
    ```
    class D: public Base 
    {
    public:
        // Base::~Base invoked automatically at end of ~D()
        ~D() { /* do what it takes to clean up derived members */ }
    };
    ```
    - 在构造函数和析构函数中调用虚函数
        - 如果构造函数或析构函数调用了某个虚函数，则我们应该执行与构造函数或析构函数所属类型（基类或派生类）相对应的虚函数版本
        - 即：基类的构造函数**不能**调用派生类版本的虚函数
            - 派生类对象被构造时，先执行基类构造函数，此时派生类部分 *未定义* 
            - 被委托的基类构造函数如果调用派生类版本的虚函数，则可能访问未定义内容，造成崩溃
- *继承的构造函数* 
    - 派生类能够重用基类的构造函数
        - 当然，这些基类构造函数不是常规继承得来的，不过姑且这么叫
        - 派生类只能继承其直接基类的构造函数，且不继承 *默认* 、 *拷贝* 和 *移动* 构造函数
            - 如派生类没有直接定义这些构造函数，则编译器为它们合成一个
        - 编译器还可以根据用户要求，利用基类的构造函数自动为派生类生成构造函数，这种构造函数称作 *继承的构造函数* 
    - 通过一条`using`声明`using Base::Base;`通知编译器生成 *继承的构造函数* 
        - 通常的 *`using`声明* 只是令名字可见，但这里的`using`会令编译器 *产生一或多个对应版本的**派生类构造函数**的代码* 
            - 对于基类的每个构造函数，编译器都在派生类中生成一（或多）个形参列表完全相同的构造函数
            - *默认* 、 *拷贝* 和 *移动* 构造函数**除外**
            - 多个：对于有`n`个默认实参的基类构造函数，编译器生成的派生类构造函数额外再有`n`个版本，每个分别省略掉一个有默认实参的形参
            - 继承的构造函数**不会**被作为用户定义的构造函数来使用
                - 因此，如果一个类只有一个继承的构造函数，则编译器也会为其合成一个默认构造函数
        ```
        struct B1
        {
            B1() = default;
            B1(int _a, int _b = 1) : a(_a), b(_b) {}
            
            int a {0};
            int b {1};
        };

        struct B2 : public B1
        {
            using B1::B1;  // compiler generates the following inherited constructors for B2:
                           // B2(int _a, int _b) : B1(_a, _b) {}
                           // B2(int _a)         : B1(_a)     {}
            
            int c {2};
        };

        B2 obj1(3, 4);     // ok
        B2 obj2(5);        // ok
        B2 obj3(6, 7, 8);  // error: no matching constructor for initialization of B2
        ```
        - 与通常的`using`不同，构造函数`using`声明**不**改变 *访问控制* 
        - （与通常的`using`相同），`using`**不能**声明`explicit`以及`constexpr`
            - 即基类中的`explicit`以及`constexpr`会被原样保留

#### 容器与继承

- 骚操作：在容器中放置 *智能指针* 






### 🌱 [Chap 16] 模板与泛型编程

#### 定义模板

- *函数模板* （function template）
    - 一个函数模板就是一个公式，用于生成针对特定类型的函数版本
    ```
    template <template_parameter_list>
    ```
    - *模板参数列表* （template parameter list）
        - 逗号分隔的列表，**不能**为空
        - 就像是函数形参列表，定义了若干特定类型的局部变量，但并未指出如何初始化它们
            - 运行时由调用者提供实参来初始化形参
        - 表示类或函数定义中用到的类型或值
            - 使用时 *隐式* 或 *显式* 地指定 *模板实参* （template argument）并绑定到模板参数上
        - 可以包含如下内容
            1. *模板类型参数* （template type parameter）
                - 可以将类型参数看做类型说明符，就像内置类型或者类类型说明符一样使用
                    - 特别地，板类型参数可以作为 *函数返回值* ，或用作 *类型转换目标类型* 
                - 类型参数前必须加上关键字`class`或`typename`，模板参数列表中二者 *等价* 
            ```
            // error: must precede U with either typename or class
            template <typename T, U> 
            T calc(const T &, const U &);
            
            // ok: no distinction between typename and class in a template parameter list
            template <typename T, class U> 
            calc (const T &, const U &);
            ```
            2. *非类型模板参数* （nontype template parameter）
                - 非类型参数是一个 *值* ，而不是类型
                - 通过 *特定的类型名* ，而非关键字`class`或`template`来指定
                    - 可以是 *整形* ，或指向对象或函数类型的 *指针* 或 *左值引用* 
                - 模板被实例化时，非类型参数被用户提供后编译器推断出的值所代替
                    - 这些值必须是 *常量表达式* ，以便模板实例化能 *在编译期发生* 
                    - **不能**用普通 *局部非静态变量* 或 *动态对象* 作为指针或引用非类型模板参数的实参
            ```
            template <unsigned N, unsigned M>
            int compare(const char (&p1)[N], const char (&p2)[M])
            {
                return strcmp(p1, p2);
            }
            
            // call of 
            compare("hi", "mom");
            // instantiates the following
            int compare(const char (&p1)[3], const char (&p2)[4])  // len + 1 for '\0' terminator
            ```
    - `inline`和`constexpr`函数模板
        - 函数模板可以被声明为`inline`的或`constexpr`，就像非模板函数一样
        - `inline`或`constexpr`说明符放在模板形参列表之后，返回类型之前
    ```
    // ok: inline specifier follows the template parameter list
    template <typename T> 
    inline T min(const T &, const T &);
    
    // error: incorrect placement of the inline specifier
    inline template <typename T> 
    T min(const T &, const T &);
    ```
    - 模板 *实例化* （Instantiating a Template）
        - 发生于 *编译期*
        - 调用函数模板时，编译器 （通常）用函数实参来 *推断* 模板实参，并 *实例化* 一个特定版本的函数
            - 编译器用实际的模板实参代替对应的模板参数来创建出模板的一个新 *实例* （instantiation）
        ```
        template <typename T>
        int compare(const T & v1, const T & v2)
        {
            if (v1 < v2) return -1;
            if (v2 < v1) return 1;
            return 0;
        }
        
        // instantiates int compare(const int &, const int &)
        std::cout << compare(1, 0) << std::endl;        // T is int
        // instantiates int compare(const vector<int> &, const vector<int> &)
        vector<int> vec1{1, 2, 3}, vec2{4, 5, 6};
        std::cout << compare(vec1, vec2) << std::endl;  // T is vector<int>
        
        int compare(const int & v1, const int & v2)
        {
            if (v1 < v2) return -1;
            if (v2 < v1) return 1;
            return 0;
        }
        ```
    - 模板编译
        - 编译器遇到模板定义时，并不生成代码，只有当实例化出模板的一个特定版本时，编译器才会生成代码
            - 当我们 *使用* 而不是定义模板时编译器才生成代码，这一特性影响了我们如何组织代码，以及错误何时被检测
        - 调用函数时，编译器只需要掌握函数的声明；类似地，使用类类型对象时，类定义必须可用，但成员函数定义不必已经出现
            - 因此函数声明和类的定义被放在 *头文件* （header file）中，而普通函数和类的成员函数的定义放在 *源文件* （source file）中
        - 模板则**不同**
            - 为了生成实例化版本，编译器需要掌握 *函数模板* 或 *类模板成员函数* 的 *定义* 
            - 因此，**函数模板和类模板的头文件都既需包括声明、也需包含定义**
    - 大多数编译错误在实例化期间报告
        - 第一阶段：编译模板本身时。只能检查语法错误
        - 第二阶段：遇到模板使用时。检查模板调用实参数目是否准确、参数类型是否匹配
        - 第三阶段：模板实例化时。只有这个阶段可以发现类型相关错误
            - 依赖于编译器如何管理实例化，有可能到 *链接* 时才报告
    - 编写类型无关的代码
        - 模板程序应尽量减少对 *实参* 的要求
        - 例如，比较运算符只用`<`，不要混用好几个
- *类模板* （class template）
    - 类模板及其成员的定义中，模板参数可以代替使用模板是用户需要提供的类型或值
    - `Blob`类定义
    ```
    // needed for friendship declaration
    template <typename> 
    class BlobPtr;
    
    // needed for the following declaraton
    template <typename>
    class Blob;
    
    // needed for friendship declaration!!!
    // template specialization MUST be present for any reference on its one specific instance
    template <typename T>
    bool operator==(const Blob<T> &, const Blob<T> &);
    
    template <typename T>
    class Blob
    {
    public:
        friend class BlobPtr<T>;
        friend bool operator==<T>(const Blob<T> &, const Blob<T> &);
    
        typedef T value_type;
        typedef typename std::vector<T>::size_type size_type;

        // constructors
        Blob() : data(std::make_shared<std::vector<T>>()) {}
        Blob(std::initializer_list<T> il) : data(std::make_shared<std::vector<T>>(il)) {}

        // number of elements in the Blob
        size_type size() const
        {
            return data->size();
        }

        bool empty() const
        {
            return data->empty();
        }

        // add and remove elements
        void push_back(const T & t)
        {
            data->push_back(t);
        }
        
        void push_back(T && t)
        {
            data->push_back(std::move(t));
        }

        void pop_back()
        {
            check(0, "pop_back on empty Blob");
            data->pop_back();
        }

        // element access
        T & back()
        {
            check(0, "back on empty Blob");
            return data->back();
        }

        T & operator[](size_type i)
        {
            // if i is too big, check will throw, preventing access to a nonexistent element
            check(i, "subscript out of range");
            return (*data)[i];
        }

    private:
        // throws msg if data[i] isn't valid
        void check(size_type i, const std::string & msg) const
        {
            if (i >= data->size()) throw std::out_of_range(msg);
        }

    private:
        std::shared_ptr<std::vector<T>> data;
    };
    
    Blob<std::string> articles = {"a", "an", "the"};
    ```
    - 实例化类模板
        - 编译器**不能**为类模板推断模板参数类型，必须在 *显式模板实参* （explicit template argument）列表中指出
            - 类模板的名字**不是**类型名
                - 类模板用于实例化类型，实例化的类型总是包含 *显式模板实参列表* 
        ```
        Blob<int> ia;                     // empty Blob<int>
        Blob<int> ia2 = {0, 1, 2, 3, 4};  // Blob<int> with five elements
        ```
        - 从这两个定义，编译器实例化出一个和下面的类等价的类
        ```
        template <> 
        class Blob<int> 
        {
        public:
            typedef typename std::vector<int>::size_type size_type;
            
            Blob();
            Blob(std::initializer_list<int> il);
            
            // ...
            
            int & operator[](size_type i);
        
        private:
            void check(size_type i, const std::string & msg) const;
            
        private:
            std::shared_ptr<std::vector<int>> data;
        };
        ```
        - 对于指定的 *每一种元素类型* ，编译器都生成 *一个不同的类* 
            - 每一个类模板的每个实例都是一个独立的类，`Blob<std::string>`和其他`Blob`类没有任何关联，也不对这些类有特殊的访问权限
        ```
        // these definitions instantiate two distinct Blob types
        Blob<std::string> names;  // Blob that holds strings
        Blob<double> prices;      // different element type
        ```
    - 类模板的成员函数
        - 与任何其他类相同，既可以在类模板内部，也可以在类模板外部定义成员函数，且定义在类模板内的成员函数被隐式声明为`inline`函数
        - 在类模板外使用类模板名
            - 在类模板外定义其成员时，必须记住：我们此时并不在其作用域中，类作用域从遇到类名处才开始
            - 定义在类模板之外的成员函数必须以`template`开始，后接模板参数列表
                - 类模板的成员函数本身就是一个普通函数，但是，类模板的每个实例都有其自己版本的成员函数
        ```
        // in-class declaration
        ret mem_func(param_list);
        
        // out-of-class declaration
        template <typename T>
        ret Class<T>::mem_func(param_list);
        
        // out-of-class definition for Blob<T>::pop_back
        template <typename T> 
        void Blob<T>::pop_back()
        {
            check(0, "pop_back on empty Blob");
            data->pop_back();
        }
        ```
        - 默认情况下，一个类模板成员函数只有在被用到时才进行实例化
        ```
        // instantiates Blob<int> and the initializer_list<int> constructor
        Blob<int> squares = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        
        // instantiates Blob<int>::size() const
        for (size_t i = 0; i != squares.size(); ++i)
        {
            squares[i] = i * i;     // instantiates Blob<int>::operator[](size_t)
        }
        ```
    - 在类模板代码中简化模板类名的使用
        - 在类模板自己的作用域中，可以 *直接使用模板名而不提供实参* 
        ```
        // inside template class scope, the following are equivalent
        BlobPtr & operator++(); 
        BlobPtr & operator--();
        
        BlobPtr<T> & operator++();
        BlobPtr<T> & operator--();
        ```
        - `BlobPtr`类定义
        ```
        // BlobPtr throws an exception on attempts to access a nonexistent element
        template <typename T>
        class BlobPtr
        {
        public:
            BlobPtr() : curr(0) {}
            BlobPtr(Blob<T> & a, size_t sz = 0) : wptr(a.data), curr(sz) {}

            T & operator*() const
            {
                auto p = check(curr, "dereference past end");
                return (*p)[curr];               // (*p) is the vector to which this object points
            }

            // increment and decrement
            BlobPtr & operator++();              // prefix operators
            
            BlobPtr & operator--();
            
            BlobPtr & operator++(int)            // postfix operators
            {
                // no check needed here; the call to prefix increment will do the check
                BlobPtr ret = *this;             // save the current value
                ++*this;                         // advance one element; prefix ++ checks the increment
                return ret;                      // return the saved state
            }
            
            BlobPtr & operator--(int);

        private:
            // check returns a shared_ptr to the vector if the check succeeds
            std::shared_ptr<std::vector<T>> check(std::size_t, const std::string &) const;

        private:
            // store a weak_ptr, which means the underlying vector might be destroyed
            std::weak_ptr<std::vector<T>> wptr;
            std::size_t curr;                    // current position within the array
        };
        ```
    - 类模板的静态成员
        - 类模板可以声明静态成员
            - 不同模板类 *实例之间是相互独立* 的类，其静态成员自然也是相互独立的
                - 对于模板中声明的每一个静态数据成员，此模板的每一个实例都拥有一个 *独立的* 该成员
            - 类外定义类模板的静态成员时也需 *定义成模板* 
                - 类的静态成员必须有且仅有一个定义
                - 类的静态成员只能在类内声明`static`并在类外定义一次，且不能重复`static`
            - 类似其他类模板成员函数，类模板静态成员函数也是只有在被用到时才进行实例化
        ```
        template <typename T> 
        class Foo 
        {
        public:
            static std::size_t count() { return ctr; }
            // other interface members
            
        private:
            static std::size_t ctr;
            // other implementation members
        };
        
        template <typename T>                // define and initialize ctr
        size_t Foo<T>::ctr = 0;              // for all instance classes of this class template
        
        // instantiates static members Foo<std::string>::ctr and Foo<std::string>::count
        Foo<std::string> fs;
        // all three objects share the same Foo<int>::ctr and Foo<int>::count members
        Foo<int> fi1, fi2, fi3;
        
        Foo<int> fi;                         // instantiates Foo<int> class and the static data member ctr
        std::size_t ct = Foo<int>::count();  // instantiates Foo<int>::count
        ct = fi.count();                     // uses Foo<int>::count
        ct = Foo::count();                   // error: which template instantiation?
        ```
    - 类模板和友元
        - 引用类或函数模板的 *一个特定实例* 之前 *必须前向声明模板自身* ；如果引用的是 *全部实例* ，则 *不需前向声明*
        - 当一个类包含一个友元声明时，类与友元各自是否是模板是相互无关的
        - 如果一个类模板包含一个非模板友元，则友元被授权可以访问 *所有* 模板实例
        ```
        // needed for friendship declaration
        template <typename> 
        class BlobPtr;
        
        // needed for the following declaraton
        template <typename>
        class Blob;
        
        // needed for friendship declaration!!!
        // template specialization MUST be present for any reference on its one specific instance
        template <typename T>
        bool operator==(const Blob<T> &, const Blob<T> &);
        
        template <typename T>
        class Blob
        {
        public:
            friend class BlobPtr<T>;
            friend bool operator==<T>(const Blob<T> &, const Blob<T> &);
            
            // others are the same
        }
        ```
        - 如果友元自身是模板，类可以授权给友元模板的 *所有实例* ，也可以只授权给 *特定实例* 
            - 为了让 *所有实例* 成为友元，友元声明中必须使用与类模板本身 *不同的参数* 
        ```
        // forward declaration necessary to befriend a specific instantiation of a template
        template <typename T> 
        class Pal;
        
        class C                   // C is an ordinary, nontemplate class
        { 
            friend class Pal<C>;  // Pal instantiated with class C is a friend to C
            
            // all instances of Pal2 are friends to C;
            // no forward declaration required when we befriend all instantiations
            template <typename T> 
            friend class Pal2;
        };
        
        template <typename T> 
        class C2                  // C2 is itself a class template
        { 
            // each instantiation of C2 has the same instance of Pal as a friend
            friend class Pal<T>;  // a template declaration for Pal must be in scope
            
            // all instances of Pal2 are friends of each instance of C2, prior declaration needed
            template <typename X> friend class Pal2;
            
            // Pal3 is a nontemplate class that is a friend of every instance of C2
            friend class Pal3;    // prior declaration for Pal3 not needed
        };
        ```
        - 将 *模板类型参数* 声明为友元
            - 可以与 *内置类型* 成为友元
        ```
        template <typename Type> 
        class Bar 
        {
        public: 
            friend Type;          // grants access to the type used to instantiate Bar
            // ...
        };
        ```
- 模板类型别名（Template Type Aliases）
    - 可以使用`typedef`引用 *实例化的类模板* 
        - 类模板的实例确实定义了一个类类型
        - 类模板本身不是类类型，因此**不能**`typedef`一个 *模板本身* 
    ```
    typedef Blob<std::string> StrBlob;
    ```
    - 但可以使用`using`类型别名引用 *模板本身* 
    ```
    template <typename T> 
    using twin = std::pair<T, T>;
    twin<std::string> authors;     // authors is a std::pair<string, string>
    ```
    - 一个 *模板类型别名* 是一族类的别名
    ```
    twin<int> win_loss;            // win_loss is a std::pair<int, int>
    twin<double> area;             // area is a std::pair<double, double>
    ```
    - 定义 *模板类型别名* 时，可以 *固定* 一或多个模板参数
    ```
    template <typename T> 
    using partNo = std::pair<T, unsigned>;
    
    partNo<std::string> books;     // books is a std::pair<std::string, unsigned>
    partNo<Vehicle> cars;          // cars is a std::pair<Vehicle, unsigned>
    partNo<Student> kids;          // kids is a std::pair<Student, unsigned>
    ```
- 模板参数
    - 模板参数可以是任何名字
        - 比如类型参数不一定非要是`T`
    ```
    template <typename Foo> 
    Foo calc(const Foo & a, const Foo & b)
    {
        Foo tmp = a;  // tmp has the same type as the parameters and return type
        // ...
        return tmp;   // return type and parameters have the same type
    }
    ```
    - 模板参数与作用域
        - 模板参数作用域起始于声明之后，终止于模板声明或定义结束之前
        - 会覆盖外层定义域中的同名实体
        - 模板内**不能**重用模板参数名
            - 自然，一个参数在模板形参列表中只能出现一次
    ```
    typedef double A;
    template <typename A, typename B> void f(A a, B b)
    {
        A tmp = a;    // tmp has same type as the template parameter A, not double
        double B;     // error: redeclares template parameter B
    }
    
    // error: illegal reuse of template parameter name V
    template <typename V, typename V> // ...
    ```
    - 模板声明
        - 模板声明必须包含参数
        - 与函数声明相同，模板声明中的模板参数名字不需与定义中相同
            - 当然，声明和定义时模板参数的种类数量和顺序必须是一样的
        - 一个特定文件所需要的 *所有模板声明* 通常 *一起放置在文件开始* 位置，出现于任何使用这些模板的代码之前 => 16.3
    ```
    // declares but does not define compare and Blob
    template <typename T> int compare(const T &, const T &);
    template <typename T> class Blob;
    
    // all three uses of calc refer to the same function template
    template <typename T> T calc(const T &, const T &);  // declaration
    template <typename U> U calc(const U &, const U &);  // declaration
    
    // definition of the template
    template <typename Type>
    Type calc(const Type & a, const Type & b) { /* . . . */ }
    ```
    - 在模板中使用 *类的类型成员* 
        - 通常使用`T::mem`访问类的静态类型成员或者静态数据成员
            - 例如`std::string::size_type`
            - 对于确定的类`T`，编译器有`std::string`的定义，自然知道`mem`是类型成员还是数据成员
            - 对于模板参数`T`，编译器直到模板实例化时才会知道`T`是什么，自然也直到那时才会知道`mem`究竟是类型成员还是数据成员
                - 但为了处理模板，编译器必须在模板定义时就知道名字`mem`究竟是类型成员还是数据成员
                - 例如遇到`T::size_type * p;`这一语句时，编译器必须立即知道这是在
                    1. 定义指向`T::size_type`类型的指针，还是
                    2. 在用一个名为`T::size_type`的静态数据成员和`p`相乘
        - 默认情况下，`C++`语言假定通过作用域运算符访问的名字**不是**类型
            - 希望使用模板类型参数的 *类型成员* 时，必须 *显示指明`typename`*
                - 希望通知编译器一个名字表示类型时，必须使用关键字`typename`，**不能**使用`class`
        ```
        template <typename T>
        typename T::value_type top(const T & c)
        {
            if (!c.empty())
                return c.back();
            else
                return typename T::value_type();
        }
        ```
    - *默认模板实参* （default template argument）
        - 就像函数（噶）默认（韭）实参（菜）一样
            - 可以提供给 *函数模板* 或 *类模板* 
            - 对于一个模板参数，当且仅当右侧所有参数都有模板实参时，它才可以有默认模板实参
        ```
        // compare has a default template argument, less<T>
        // and a default function argument, F()
        template <typename T, typename F = std::less<T>>
        int compare(const T & v1, const T & v2, F f = F())
        {
            if (f(v1, v2)) return -1;
            if (f(v2, v1)) return 1;
            return 0;
        }
        
        bool i = compare(0, 42);  // uses f = std::less<int>(); i is -1
        // result depends on the isbns in item1 and item2
        SalesData item1(cin), item2(cin);
        bool j = compare(item1, item2, compareIsbn);
        ```
        - 模板默认实参与类模板
            - 无论何时使用一个类模板，都必须在模板名后面接上尖括号`<>`，尖括号中指出类必须从模板实例化而来
            - 特别是，如果一个类模板为其所有模板参数都提供了默认实参，且我们希望使用这些模板实参，就必须在模板名之后跟一个空尖括号
        ```
        template <class T = int> 
        class Numbers                            // by default T is int
        { 
        public:
            Numbers(T v = 0): val(v) { }
            // various operations on numbers
            
        private:
            T val;
        };
        
        Numbers<long double> lots_of_precision;
        Numbers<> average_precision;             // empty <> says we want the default type
        ```
- *成员模板* （member template）
    - 一个类（不论是普通类还是类模板）可以包含 *本身是模板的成员函数* ，这种成员函数被称作 *成员模板*
    - 普通（非模板）类的成员模板
        - `DebugDelete`类定义
            - 此类类似`std::unique_ptr`所用的 *默认删除器* 类型
        ```
        // function-object class that calls delete on a given pointer
        class DebugDelete 
        {
        public:
            DebugDelete(std::ostream & os = std::cerr): cout(os) {}
            
            // as with any function template, the type of T is deduced by the compiler
            template <typename T> 
            void operator()(T * p) const
            { 
                cout << "deleting std::unique_ptr" << std::endl; 
                delete p;
            }
            
        private:
            std::ostream & cout;
        };
        
        double * p = new double{};
        DebugDelete d;              // an object that can act like a delete expression
        d(p);                       // calls DebugDelete::operator()(double *), which deletes p
        
        int * ip = new int{};
        DebugDelete()(ip);          // calls operator()(int *) on a temporary DebugDelete object
        
        // destroying the the object to which p points
        // instantiates DebugDelete::operator()<int>(int *)
        std::unique_ptr<int, DebugDelete> p(new int, DebugDelete());
        
        // destroying the the object to which sp points
        // instantiates DebugDelete::operator()<string>(string*)
        std::unique_ptr<std::string, DebugDelete> sp(new string, DebugDelete());
        ```
    - 类模板的成员模板
        - 类模板也可以定义成员模板
            - 类模板和成员模板拥有 *各自的* 模板参数
        ```
        template <typename T> 
        class Blob 
        {
            template <typename It> 
            Blob(It b, It e);
            // ...
        };
        ```
        - 类模板外定义成员模板时，需要 *连续写两个`template`* ，类模板的在前，成员函数模板的在后
        ```
        template <typename T>   // type parameter for the class
        template <typename It>  // type parameter for the constructor
        Blob<T>::Blob(It b, It e): data(std::make_shared<std::vector<T>>(b, e)) 
        {
        }
        ```
    - 实例化与成员模板
        - 实例化类模板的成员模板时，必须同时提供类模板和成员函数模板的实参
    ```
    int ia[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<long> vi = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::list<const char *> w = {"now", "is", "the", "time"};
    
    // instantiates the Blob<int> class
    // and the Blob<int> constructor that has two int * parameters
    // Blob<int>::Blob(int *, int *);
    Blob<int> a1(begin(ia), end(ia));
    
    // instantiates the Blob<int> constructor that has
    // two vector<long>::iterator parameters
    Blob<int> a2(vi.begin(), vi.end());
    
    // instantiates the Blob<string> class and the Blob<string>
    // constructor that has two (std::list<const char *>::iterator parameters
    Blob<std::string> a3(w.begin(), w.end());
    ```
- 控制实例化
    - [*显式实例化*](https://en.cppreference.com/w/cpp/language/class_template)（explicit instantiation）
        - 模板被实例化的相同实例可能出现在多个对象文件中，会造成严重的额外开销
        - *显式实例化* 用于避免这种额外开销
            - 编译器遇到 *显式模板声明* 时，不会再本文件中生成实例化代码
            - 将一个实例化声明为`extern`就意味着承诺在程序的其他位置会有一个非`extern`声明（定义）
            - 对于一个给定的实例化版本，可能会有多个`extern`声明，但必须 *有且仅有一个实例化定义* 
        - An explicit instantiation definition forces instantiation of the class, struct, or union they refer to. 
          - It may appear in the program anywhere after the template definition.  
          - For a given argument-list, is only allowed to appear once in the entire program.
        - An explicit instantiation declaration (an extern template) skips implicit instantiation step: 
          - The code that would otherwise cause an implicit instantiation instead uses 
            the explicit instantiation definition provided elsewhere 
            (resulting in link errors if no such instantiation exists). 
          - This can be used to reduce compilation times by explicitly declaring a template instantiation 
            in all but one of the source files using it, and explicitly defining it in the remaining file.
        ```
        extern template class SomeClass<Arguments...>;          // instantiation declaration
        template class SomeClass<Arguments...>;                 // instantiation definition
        ```
        - `declaration`是类或函数声明，其中模板参数全部替换为模板实参
        ```
        // instantion declaration and definition
        extern template class Blob<std::string>;                // declaration
        template class Blob<std::string>;                       // definition
        
        extern template int compare(const int &, const int &);  // definition
        template int compare(const int &, const int &);         // definition
        ```
        - 由于编译器在使用一个模板时自动对其初始化，因此`extern`声明必须出现在任何使用此实例化版本的代码 *之前* 
        ```
        // Application.cpp
        
        // these template types must be instantiated elsewhere in the program
        extern template class Blob<std::string>;
        extern template int compare(const int &, const int &);
        
        // instantiation will appear elsewhere
        Blob<std::string> sa1, sa2;     
        
        // Blob<int> and its initializer_list constructor instantiated in this file
        Blob<int> a1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        Blob<int> a2(a1);               // copy constructor instantiated in this file
        int i = compare(a1[0], a2[0]);  // instantiation will appear elsewhere
        ```
    - 类模板实例化定义会实例化所有成员
        - 类模板实例化定义会实例化该模板的所有成员，包括 *内联* 的成员函数
            - 编译器遇到类模板的实例化定义时，它不了解具体要用哪些成员，所以干脆全部实例化
        - 因此，用来显式实例化一个类模板的类型必须能用于模板的全部成员
- 案例分析：[`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr)
    - 定义
    ```
    template <class T, class Deleter = std::default_delete<T>> 
    class unique_ptr;
    
    template <class T, class Deleter> 
    class unique_ptr<T[], Deleter>;
    
    template <class T> 
    class shared_ptr;
    ```
    - 与`std::shared_ptr`的不同
        1. 保存指针的策略
            - 前者共享
            - 后者独占
        2. 允许用户重载 *默认删除器* 的方式
            - 前者：定义或`reset`时作为函数参数传入
            - 后者：定义时以显式模板实参传入
    - 在 *运行时* 绑定删除器
        - `std::shared_ptr`必须能够直接访问删除器
            - 删除器保存为指针或封装了指针的类（如`std::function`）
            - **不能**直接保存为成员，因为删除器的类型在运行期时刻会变，而成员的类型编译期确定后就不能变了
        - 假定`std::shared_ptr`将删除器保存为名为`del`的 *指针* 中
            - 则其析构函数中应有如下语句
            - 由于删除器是 *间接* 保存的，因此调用时需要一次额外的 *跳转* 操作
            ```
            // value of del known only at run time; call through a pointer
            del ? del(p) : delete p;  // del(p) requires run-time jump to del's location
            ```
    - 在 *编译时* 绑定删除器
        - `std::unique_ptr`的删除器是类类型的一部分
        - 删除器成员的类型在编译时就已知，且不会改变，可以直接保存为类成员
        - 其析构函数中应有如下语句，避免了间接调用删除器的额外的运行时开销
        ```
        // del bound at compile time; direct call to the deleter is instantiated
        del(p);                       // no run-time overhead
        ```

#### 模板实参推断（template argument deduction）

- 类型转换与模板类型参数
    - 与非模板函数一样，传递给函数模板的实参被用来初始化函数的形参
        - 如果函数形参类型使用了模板参数，那么它采用特殊的初始化规则
        - 只有很有限的几种类型转换会自动地应用于这些实参
        - 编译器通常**不是**对实参进行类型转换，而是生成一个新的实例
    - 能在调用中应用于函数模板的 *类型转换* 有
        - `const_cast`：可以添加 *底层`const`* ，将非`const`对象的引用或指针传递给一个`const`的引用或指针
            - *顶层`const`* 不论是在形参中还是在实参中都会 *被忽略* 
        - *数组或函数指针* 转换：如果函数形参**不是** *引用* 类型，则可以对数组或函数类型的实参应用正常的指针转换
            - 一个数组实参可以转换为指向其首元素的指针
            - 一个函数实参可以转换为指向该函数的函数指针
        - 其他类型转换，如 *算数转换* ， *派生类向基类的转换* ， *用户定义的转换* ，都**不能**应用于函数模板
        ```
        template <typename T> T fobj(T, T);                  // arguments are copied
        template <typename T> T fref(const T &, const T &);  // references
        
        std::string s1("a value");
        const std::string s2("another value");
        fobj(s1, s2);                                        // calls fobj(std::string, std::string); const is ignored
        fref(s1, s2);                                        // calls fref(const std::string &, const std::string &)
        
        // uses premissible conversion to const on s1
        int a[10], b[42];
        fobj(a, b);                                          // calls f(int *, int *)
        fref(a, b);                                          // error: array types don't match
        ```
    - 使用相同模板参数类型的函数形参
        - 一个模板类型参数可以用作多个函数形参的类型，此时这些函数形参的类型 *必须精确匹配* 
        ```
        template <typename T> 
        int compare(const T &, const T &);
        
        long lng;
        compare(lng, 1024);                                  // error: cannot instantiate compare(long, int)
        ```
        - 如果希望允许对函数实参进行正常的类型转换，则应将函数模板定义成`2`个参数
            - 此时`A`和`B`之间则必须 *兼容*
        ```
        // argument types can differ but must be compatible
        template <typename A, typename B>
        int flexibleCompare(const A & v1, const B & v2)
        {
            if (v1 < v2) return -1;
            if (v2 < v1) return 1;
            return 0;
        }
        
        long lng;
        flexibleCompare(lng, 1024);                          // ok: calls flexibleCompare(long, int)
        ```
   - 正常类型转换应用于普通函数实参
        - 函数模板中，对于参数类型**不是**模板参数的形参，可以接受对实参的正常的类型转换
    ```
    template <typename T> 
    std::ostream & print(std::ostream & cout, const T & obj)
    {
        return cout << obj;
    }
    
    print(cout, 42);                   // instantiates print(std::ostream &, int)
    std::ofstream fout("output.txt");
    print(f, 10);                      // uses print(std::ostream &, int); converts f to ostream &
    fout.close();
    ```
- 函数模板显式实参
    - 某些情况下，编译器无法推断出模板实参的类型
    - 其他情况下，我们希望允许用户控制模板实例化
    - 当函数返回类型与参数列表中任何类型都不相同时，这两种情况最常出现
    - 我们可以定义表示返回类型的 *第三个* 模板参数，从而允许用户控制返回类型
    - 本例中，没有任何函数实参的类型可用来推断`T1`的类型，每次`sum`调用时，调用者都必须为`T1`提供一个 *显式模板实参* 
    ```
    // T1 cannot be deduced: it doesn't appear in the function parameter list
    template <typename T1, typename T2, typename T3>
    T1 sum(T2, T3);
    ```
    - *显式模板实参* 在 *尖括号* 中给出，位于函数名之后、形参列表之前
        - 与定义模板实例的方式相同
    ```
    // T1 is explicitly specified; T2 and T3 are inferred from the argument types
    auto val3 = sum<long long>(i, lng);  // long long sum(int, long)
    ```
    - *显式模板实参* 按 *从左至右* 的顺序与对应的模板参数匹配，第一个模板实参与第一个模板参数匹配，第二个模板实参与第二个模板参数匹配，依次类推
        - 只有 *最右* 参数的显式模板实参才可以忽略，且前提是能被推断出来
    ```
    // poor design: users must explicitly specify all three template parameters
    template <typename T1, typename T2, typename T3>
    T3 alternative_sum(T2, T1);

    // error: can't infer initial template parameters
    auto val3 = alternative_sum<long long>(i, lng);
    
    // ok: all three parameters are explicitly specified
    auto val2 = alternative_sum<long long, int, long>(i, lng);
    ```
    - *正常类型转换* 用于 *显式指定的形参* 
    ```
    long lng;
    compare(lng, 1024);        // error: template parameters don't match
    compare<long>(lng, 1024);  // ok: instantiates compare(long, long)
    compare<int>(lng, 1024);   // ok: instantiates compare(int, int)
    ```
- 尾置返回类型与类型转换
    - 希望用户确定返回类型时，用显式模板形参表示函数模板的返回类型很简单明了
    - 但有时返回值类型无法用模板形参表示，只能从返回对象直接获取，比如返回所处理序列的元素类型，此时
        - 则需要 *尾置返回* `decltype(ret)` `(until C++11)`
        - 不需要任何操作，编译器自动根据`return`语句推导返回值类型 `(since C++14)`
    ```
    // a trailing return lets us declare the return type after the parameter list is seen
    template <typename It>
    auto fcn(It beg, It end) -> decltype(*beg)
    {
        // process the range
        return *beg;  // return a reference to an element from the range
    }
    
    std::vector<int> vi = {1, 2, 3, 4, 5};
    Blob<std::string> ca = {"hi", "bye"};
    auto & i = fcn(vi.begin(), vi.end());  // fcn should return int &
    auto & s = fcn(ca.begin(), ca.end());  // fcn should return std::string &
    ```
- *类型转换模板* （type transformation template）
    - [`<type_traits>`](https://en.cppreference.com/w/cpp/header/type_traits)
        - 常用于 *模板元程序* 设计，常用的类型转换模板有
            - [`std::remove_reference<T>::type`](https://en.cppreference.com/w/cpp/types/remove_reference)：若`T`为`X &`或`X &&`，则为`X`；否则，为`T`
            - [`std::add_const<T>::type`](https://en.cppreference.com/w/cpp/types/add_const)：若`T`为`X &`，`X &&`或 *函数* ，则为`T`；否则，为`const T`
            - [`std::add_lvalue_reference<T>::type`](https://en.cppreference.com/w/cpp/types/add_lvalue_reference)：若`T`为`X &`，则为`T`；若`T`为`X &&`，则为`X &`；否则，为`T &`
            - [`std::add_rvalue_reference<T>::type`](https://en.cppreference.com/w/cpp/types/add_rvalue_reference)：若`T`为`X &`或`X &&`，则为`T`；否则，为`T &&`
            - [`std::remove_pointer<T>::type`](https://en.cppreference.com/w/cpp/types/remove_pointer)：若`T`为`X *`，则为`X`；否则，为`T`
            - [`std::add_pointer<T>::type`](https://en.cppreference.com/w/cpp/types/add_pointer)：若`T`为`X &`或`X &&`，则为`X *`；否则，为`T`
            - [`std::make_signed<T>::type`](https://en.cppreference.com/w/cpp/types/make_signed)：若`T`为`unsigned X`，则为`X`；否则，为`T`
            - [`std::make_unsigned<T>::type`](https://en.cppreference.com/w/cpp/types/make_unsigned)：若`T`为`X`，则为`unsigned X`；否则，为`T`
            - [`std::remove_extent<T>::type`](https://en.cppreference.com/w/cpp/types/remove_extent)：若`T`为`X[n]`，则为`X`；否则，为`T`
            - [`std::make_all_extents<T>::type`](https://en.cppreference.com/w/cpp/types/make_all_extents)：若`T`为`X[n1][n2]...`，则为`X`；否则，为`T`
        - 工作方式举例
        ```
        template <class T> struct remove_reference       { typedef T type; };
        template <class T> struct remove_reference<T &>  { typedef T type; };
        template <class T> struct remove_reference<T &&> { typedef T type; };
        ```
    - 无法直接从模板参数以及返回对象获得所需要的返回类型时使用
        - 例如，要求上面的`fcn(vi.begin(), vi.end());`返回`int`而不是`int &`
        ```
        // must use typename to use a type member of a template parameter
        template <typename It>
        auto fcn2(It beg, It end) -> typename remove_reference<decltype(*beg)>::type
        {
            // process the range
            return *beg;  // return a copy of an element from the range
        }
        ```
- 函数指针与实参推断
    - 用函数模板初始化函数指针或为函数指针赋值时，编译器使用 *函数指针的类型* 推断模板实参
        - 如果无法推断实参，则产生 *错误* 
    ```
    template <typename T> int compare(const T &, const T &);
    // pf1 points to the instantiation int compare(const int &, const int &)
    int (*pf1)(const int &, const int &) = compare;
    ```
    - 特别地，当 *参数* 是一个函数模板实例的地址时，程序上下文必须满足：对于每个模板参数，能唯一确定其类型或值
    ```
    // overloaded versions of func; each takes a different function pointer type
    void func(int(*)(const std::string &, const std::string &));
    void func(int(*)(const int &, const int &));
    func(compare);       // error: which instantiation of compare?

    // ok: explicitly specify which version of compare to instantiate
    func(compare<int>);  // passing compare(const int &, const int &)
    ```
- 模板实参推断与引用
    - 从 *非常量左值引用形参* 推断类型
        - 传递规则
            1. 只能传递 *左值* 
            2. 实参可以是`const`类型，也可以不是。如果实参是`const`的，则`T`将被推断成`const`类型
                - 编译器会应用正常的引用绑定规则：`const`是底层的，不是顶层的
    ```
    template <typename T> void f1(T &);  // argument must be an lvalue
    
    // calls to f1 use the referred-to type of the argument as the template parameter type
    int i = 0;
    const int ci = 0;
    f1(i);                               // i is an int; template parameter T is int
    f1(ci);                              // ci is a const int; template parameter T is const int
    f1(5);                               // error: argument to a & parameter must be an lvalue
    ```
    - 从 *常量左值引用形参* 推断类型
        - 传递规则
            1. 可以传递 *任何类型* 的实参：常量或非常量对象、临时量，字面值
            2. `T`**不会**被推断为`const`类型，不论提供的实参本身是不是`const`
                - `const`已经是函数参数类型的一部分，因此不会是模板参数类型的一部分
    ```
    template <typename T> void f2(const T &);  // can take an rvalue
    
    // parameter in f2 is const &; const in the argument is irrelevant
    // in each of these three calls, f2's function parameter is inferred as const int &
    int i = 0;
    const int ci = 0;
    f2(i);                                     // i is an int; template parameter T is int
    f2(ci);                                    // ci is a const int, but template parameter T is int
    f2(5);                                     // a const & parameter can be bound to an rvalue; T is int
    ```
    - 从 *右值引用形参* 推断类型
        - 正常传递：可以传递 *右值* ，`T`推断为该右值实参的类型
            - 即`typename std::remove_reference<decltype(argument)>::type`
            ```
            template <typename T> void f3(T &&);
            f3(42);                                    // argument is an rvalue of type int; template parameter T is int
            ```
        - 一条例外：还可以传递 *左值* ，`T`推断为该左值实参类型的引用（保留 *底层`const`* ）
            - 即`typename std::remove_reference<decltype(argument)>::type &`
    - *引用坍缩* 和 *右值引用形参* （Reference Collapsing and Rvalue Reference Parameters）
        - 正常情况下， *右值引用* **不能**绑定到 *左值* 上，以下 *两种* 情况**例外**
            1. *右值引用的特殊类型推断* 
                - 将 *左值实参* 传递给函数的 *指向模板参数的右值引用形参* （如`T &&`）时，编译器推断模板类型参数为 *实参的左值引用类型* 
                    - *左值实参* 的 *底层`const`* 会被原样保留
                ```
                template <typename T> void f3(T &&);
                f3(argument);  // T is deducted to typename std::add_lvalue_reference<decltype(argument)>::type
                ```
                - 影响右值引用参数的推断如何进行
            2. *引用坍缩* 
                - 仅适用于 *间接创建引用的引用* 
                    - 比如通过`typedef`、 *类型别名* 或 *模板* 
                - 除 *右值引用的右值引用* 坍缩为 *右值引用* 外， *其余组合* 均坍缩为 *左值引用* 
        - 组合引用坍缩和右值引用的特殊类型推断规则，意味着
            - 如果 *函数形参* 是 *指向模板类型参数的右值引用* ，则它可以被绑定到一个 *左值* ，且
            - 如果 *函数实参* 是 *左值* ，则推断出的 *模板实参类型* 将是 *左值引用* ，且 *函数形参* 将被实例化为 *普通左值引用参数* 
        ```
        f3(i);   // argument is an lvalue; template parameter T is int&
        f3(ci);  // argument is an lvalue; template parameter T is const int&
        
        // invalid code, for illustration purposes only
        void f3<int &>(int & &&);  // when T is int &, function parameter is int & &&, which collapses into int &
        // actual function template instance for previous code
        void f3<int &>(int &);     // when T is int &, function parameter collapses to int &
        ```
    - 编写 *接受右值引用形参的函数模板* 
        - 考虑如下函数
        ```
        template <typename T> 
        void f3(T && val)
        {
            T t = val;                   // copy or binding a reference?
            t = fcn(t);                  // does the assignment change only t, or both val and t?
            if (val == t) { /* ... */ }  // always true if T is a reference type
        }
        ```
        - 情况很复杂，容易出事故
            1. 传入 *右值* 时，例如 *字面值常量* `42`， 则
                - `T`被推断为`int`
                - 此时局部变量`t`被 *拷贝初始化* 
                - 赋值`t`**不**改变`val`
            2. 传入 *左值* 时，例如`int i = 0; f3(i);`， 则
                - `T`被推断为`int &`
                - 此时局部变量`t`被 *（左值）引用初始化* ，绑定到了`val`上
                - 赋值`t` *会改变* `val`
        - 实际应用时， *接受右值引用形参的函数模板* 通常只应用于
            1. *转发* （forwarding） => 16.2.7
            2. *模板重载* （template overloading）=> 16.3
        - 使用 *接受右值引用形参的函数模板* 通常使用如下方式重载 => 13.6.3
        ```
        template <typename T> void f(T &&);       // binds to nonconst rvalues
        template <typename T> void f(const T &);  // lvalues and const rvalues
        ```
- 详解[`std::move`](https://en.cppreference.com/w/cpp/utility/move)
    - `gcc`的实现
    ```
    /// <type_traits>
    /// remove_reference
    template <typename T> struct remove_reference       { typedef T type; };
    template <typename T> struct remove_reference<T &>  { typedef T type; };
    template <typename T> struct remove_reference<T &&> { typedef T type; };
    
    /// <move.h>
    /// @brief     Convert a value to an rvalue.
    /// @param  t  A thing of arbitrary type.
    /// @return    The parameter cast to an rvalue-reference to allow moving it.
    template <typename T>
    constexpr typename std::remove_reference<T>::type &&
    move(T && t) noexcept
    { 
        return static_cast<typename std::remove_reference<T>::type &&>(t); 
    }
    ```
    - 通过 *引用坍缩* ，`std::move`的形参`T && t`可以与任何 *类型* 、任何 *值类别* 的实参匹配
        - 可以传递 *左值* 
        - 也可以传递 *右值* 
    ```
    std::string s1("hi!"), s2;
    s2 = std::move(std::string("bye!"));  // ok: moving from an rvalue
    s2 = std::move(s1);                   // ok: but after the assigment s1 has indeterminate value
    ```
    - 工作流程梳理
        1. `std::move(std::string("bye!"));`：传入右值实参时
            - 推断出`T = std::string`
            - 返回值类型`std::remove_reference<std::string>::type &&`就是`std::string &&`
            - 形参`t`的类型`T &&`为`std::string &&`
            - 因此，此调用实例化`std::move<std::string>`，即`std::string && std::move(std::string &&)`
            - 返回`static_cast<std::string &&>(t)`，但`t`已是`std::string &&`，因此这步强制类型转换只是直接赋值引用
        2. `std::move(s1);`：传入左值实参时
            - 推断出`T = std::string &`
            - 返回值类型`std::remove_reference<std::string>::type &&`仍然是`std::string &&`
            - 形参`t`的类型`T &&` *坍缩* 为`std::string &`
            - 因此，此调用实例化`std::move<std::string &>`，即`std::string && std::move(std::string &)`
            - 返回`static_cast<std::string &&>(t)`，这步强制类型转换将`t`从`std::string &`转换为`std::string &&`
    - 将 *左值* `static_cast`成 *右值引用* 是允许的
        - 但实际使用时应当使用封装好的`std::move`而**不是**`static_cast`
- *完美转发* （perfect forwarding）
    - 某些函数需要将其一个或多个实参原封不动地 *转发* 给其他函数，具体需要保持实参的以下性质
        1. *类型* ，包括底层`cv`限定（对于引用和指针）
        2. *值类别* ，是左值还是右值
    - *完美转发* 具体做法：同时采取如下措施
        1. 将 *函数形参类型* 定义为 *指向模板类型参数的右值引用* 就可以保持实参的所有 *类型信息* 
            - 使用引用参数还可以保持`const`属性，因为引用中`const`是 *底层* 的
        2. 在函数中 *调用`std::forward`转发实参* 
            - 使用时指明`std::forward`，不使用`using`声明，和`std::move`类似，避免作用域问题
    - 完美转发语法样例
    ```
    template <typename T1, typename T2>
    void fun1(T1 && t1, T2 && t2)
    {
        fun2(std::forward<T1>(t1), std::forward<T2>(t2));
    }
    
    template <typename ... Args>
    void fun3(Args && ... args)
    {
        fun4(std::forward<Args>(args) ...);
    }
    ```
    - 详解[`std::forward`](https://en.cppreference.com/w/cpp/utility/forward)
        - `gcc`的实现
        ```
        /// <move.h>
        /// @brief     Forward an lvalue.
        /// @return    The parameter cast to the specified type.
        /// This function is used to implement "perfect forwarding".
        template <typename T>
        constexpr T &&
        forward(typename std::remove_reference<T>::type & t) noexcept
        { 
            return static_cast<T &&>(t); 
        }
        
        /// <move.h>
        /// @brief     Forward an rvalue.
        /// @return    The parameter cast to the specified type.
        /// This function is used to implement "perfect forwarding".
        template <typename T>
        constexpr T &&
        forward(typename std::remove_reference<T>::type && t) noexcept
        {
            static_assert(!std::is_lvalue_reference<T>::value, 
                          "template argument substituting T is an lvalue reference type");
            return static_cast<T &&>(t);
        }
        ```
        - 保持传入参数的 *值类别* 
            1. 转发 *左值* 为 *左值或右值* ，依赖于`T`
                - 考虑如下例子
                ```
                template <class T>
                void wrapper(T && arg) 
                {
                    // arg is always lvalue
                    foo(std::forward<T>(arg));  // Forward as lvalue or as rvalue, depending on T
                }
                ```
                - 若对`wrapper`的调用传递 *右值* `std::string`，则推导`T = std::string`，将`std::string &&`类型的 *右值* 传递给`foo`
                - 若对`wrapper`的调用传递 *`const`左值* `std::string`，则推导`T = const std::string &`，将`const std::string &`类型的 *左值* 传递给`foo`
                - 若对`wrapper`的调用传递 *非`const`左值* `std::string`，则推导`T = std::string &`，将`std::string &`类型的 *左值* 传递给`foo`
            2. 转发 *右值* 为 *右值* ，并 *禁止右值转发为左值* 
                - 此重载用于 *转发表达式的结果* （如 *函数调用* ），结果可以是 *右值* 或 *左值* ，值类别与实参的原始值相同
                - 考虑如下例子
                ```                
                // transforming wrapper 
                template <class T>
                void wrapper(T && arg)
                {
                    foo(std::forward<decltype(std::forward<T>(arg).get())>(std::forward<T>(arg).get()));
                }
                
                struct Arg
                {
                    int i = 1;
                    int   get() && const { return i; }  // call to this overload is rvalue
                    int & get() &  const { return i; }  // call to this overload is lvalue
                }; 
                ```
                - 试图转发右值为左值，例如通过以左值引用类型`T`实例化模板`(2)`，会产生 *编译错误*  
    - 考虑如下例子
        - 代码
        ```
        template <typename F, typename T1, typename T2>
        void flip(F f, T1 && t1, T2 && t2)
        {
            f(std::forward<T1>(t1), std::forward<T1>(t1));
        }
        
        void f(int v1, int & v2)  // note v2 is a reference
        {
            std::cout << v1 << " " << ++v2 << std::endl;
        }
        
        int j = 0;
        filp(f, j, 42);          // now j == 1
        ```
        - 传给`t1`左值`j`，推断出`T1 = int &`，`T1 &&`坍缩为`int &`，原样转发 *左值* `int &`
        - 传给`t2`右值`42`，推断出`T2 = int`，`T2 &&`就是`int &&`，原样转发 *右值* `int &&`
        - `f`能够改变`j`
    - 可变模板完美转发测试
        - 代码
        ```
        // a test on perfect forwarding of variadic template functions

        // variadic template function expansion
        // via recursion
        template <typename T>
        void fun4(T && t)
        {
            std::cout << "void fun3(T && t) " << t << std::endl;
        }

        void fun4(std::string && s)
        {
            std::cout << "void fun3(std::string && s) " << s << std::endl;
        }

        template <typename T, typename ... Args>
        void fun4(T && t, Args && ... args)
        {
            std::cout << "void fun4(T && t, Args && ... args) " << t << std::endl;
            fun4(std::forward<Args>(args) ...);
        }

        // this one does perfect forwarding, successfully calling the std::string && specialization
        template <typename ... Args>
        void fun3_1(Args && ... args)
        {
            fun4(std::forward<Args>(args) ...);
        }

        // this one doesn't do perfect forwarding, only calling the template
        template <typename ... Args>
        void fun3_2(Args && ... args)
        {
            fun4(args ...);
        }
        
        fun3_1(1, std::string {"rval"});  // void fun4(T && t, Args && ... args) 1
                                          // void fun3(std::string && s) rval
        fun3_2(1, std::string {"rval"});  // void fun4(T && t, Args && ... args) 1
                                          // void fun3(T && t) rval
        ```
        - 可以看到，没有完美转发，就无法匹配到接受右值引用形参的特例化版本`void fun3(std::string && s)`了

#### 模板重载（template overloading）

- 函数模板可以被另一模板或非模板函数重载
- 名字相同的函数必须具有不一样的形参列表
- 函数模板的重载确定/重载决议 (overload resolution involving function templates)
    - 对于一个调用，其 *候选函数* (Candidate functions) 包括 *所有* 模板实参推断 (template argument deduction) 成功的模板实例
    - 候选的函数模板总是 *可行的* ，因为模板实参会排除掉任何不可行的模板
    - *可行函数* (Viable functions)（模板的和非模板的）按 *类型转换* 来排序
        - 可用于函数模板的类型转换很有限，只有`const_cast`，和数组、函数向指针的转换 => 16.2.1
    - 如果恰有一个函数提供比任何其他函数都 *更好的匹配* (Best viable function)，则选择此函数；如有多个函数，则
        1. 如果 *只有一个非模板函数* ，则选择之
        2. 如果没有非模板函数 ，而 *全是函数模板* ，而一个模板比其他模板 *更特例化* （specialized），则选择之
        3. 否则，报错 *二义性调用*
    - 一句话：形参匹配，特例化（非模板才是最特例化的），完犊子
- *模板重载* 案例分析
    - `例1`
        - 考虑如下调用
        ```
        // print any type we don't otherwise handle
        template <typename T> std::string debug_rep(const T & t)
        {
            std::ostringstream ret;  // see § 8.3 (p. 321)
            ret << t;                // uses T's output operator to print a representation of t
            return ret.str();        // return a copy of the string to which ret is bound
        }

        // print pointers as their pointer value, followed by the object to which the pointer points
        // NOTICE: this function will not work properly with char*; see § 16.3 (p. 698)
        template <typename T> std::string debug_rep(T * p)
        {
            std::ostringstream ret;
            ret << "pointer: " << p;          // print the pointer's own value
            if (p)
                ret << " " << debug_rep(*p);  // print the value to which p points
            else
                ret << " null pointer";       // or indicate that the p is null
            return ret.str();                 // return a copy of the string to which ret is bound
        }

        std::string s("hi");
        std::cout << debug_rep(s) << std::endl;
        std::cout << debug_rep(&s) << std::endl;
        ```
        - 对于`debug_rep(s)`，只有第一个版本可行
            - 第二个要指针，`std::string`对象又不是
        - 对于`debug_rep(&s)`，两个版本都可行
            - 各自实例化出
                - `debug_rep(const std::string * &)`：第一个版本实例化而来，`T = std::string *`，需要指针的`const_cast`
                - `debug_rep(std::string *)`：第二个版本实例化而来，`T = std::string`，是 *精确匹配* 
            - 选择第二个
    - `例2`：多个可行模板
        - 考虑如下调用
        ```
        const std::string * sp = &s;
        std::cout << debug_rep(sp) << std::endl;
        ```
        - 此例中两个版本都可行
            - 各自实例化出
                - `debug_rep(const std::string * &)`：第一个版本实例化而来，`T = std::string *`，是 *精确匹配* 
                - `debug_rep(const std::string *)`：第二个版本实例化而来，`T = const std::string`，是 *精确匹配* ，更 *特例化* 
            - 选择第二个
                - 没有 *特例化* 这一条，将**无法**对 *`const`指针* 调用 *指针* 版本的`debug_rep`
                - 问题在于`const T &`可以匹配 *任何类型* ，包括 *指针类型* ，是万金油
                - 而`T *` *只能* 匹配 *指针* 
    - `例3`：非模板和模板重载
        - 考虑如下调用
        ```
        // print strings inside double quotes
        std::string debug_rep(const std::string & s)
        {
            return '"' + s + '"';
        }
        
        const std::string * sp = &s;
        std::cout << debug_rep(sp) << std::endl;
        ```
        - 此例中第一个模板和上面的非模板版本都可行
            - 实际调用
                - `debug_rep(const std::string * &)`：第一个模板实例化而来，`T = std::string *`，是 *精确匹配* 
                - `debug_rep(const std::string &)`：非模板版本，是 *精确匹配* 
            - 选择 *非模板版本* 
                - 非模板才是最特例化的嘛
    - `例4`：重载模板和类型转换
        - 玩一玩`C`风格字符串指针和字符串字面常量
        - 考虑如下调用
        ```
        std::cout << debug_rep("hi world!") << std::endl;  // calls debug_rep(T *)
        ```
        - 此例中三个函数都可行
            - 实际调用
                - `debug_rep(const T &)`：第一个模板实例化而来，`T = char[10]`，是 *精确匹配* 
                - `debug_rep(T *)`：第二个模板实例化而来，`T = const char`，需要数组转指针，是 *精确匹配* 
                - `debug_rep(const std::string &)`：非模板版本，需要`const char *`转`std::string`，是 *用户定义转换* 
            - 首先`pass`掉非模板版本，然后两个精确匹配的模板里面选择更特例化的模板二
        - 如果希望 *字符指针* 按照`std::string`处理，可以定义另外两个非模板重载版本
        ```
        // convert the character pointers to string and call the string version of debug_rep
        std::string debug_rep(char * p)
        {
            return debug_rep(std::string(p));
        }
        
        std::string debug_rep(const char * p)
        {
            return debug_rep(std::string(p));
        }
        ```
        - 缺少 *声明* 可能导致程序行为异常
            - 为了使上述`char *`版本正常工作，`debug_rep(const std::string &)` *必须在作用域中* 
            - 否则，就会调用错误的版本，找到模板版本去
        ```
        template <typename T> std::string debug_rep(const T & t);
        template <typename T> std::string debug_rep(T * p);
        
        // the following declaration must be in scope
        // for the definition of debug_rep(char* ) to do the right thing
        std::string debug_rep(const std::string &);
        
        std::string debug_rep(char * p)
        {
            // if the declaration for the version that takes a const string & is not in scope
            // the return will call debug_rep(const T &) with T instantiated to std::string
            return debug_rep(std::string(p));
        }
        ```
        - 在定义任何函数之前，记得 *声明所有重载的函数版本* ，这样就不必担心编译器由于未遇到希望调用的版本而实例化并非所需的函数模板

#### 模板特例化（Template Specializations）

- *模板特例化* 版本就是一个模板的独立定义，其中一个或多个模板参数被指定为特定的类型
    - 特例化函数模板时，必须为原模板中的每个模板参数都提供实参
    - 为了指出我们正在实例化一个模板，应使用关键字`template <>`指出我们将为原模板的所有模板参数提供实参
        - 注意这与模板默认实参的`template <>`的区别在于
            - 这是在定义一个类模板
            - 后者是在实例化一个类模板的对象，且模板实参全部使用默认参数
- *函数模板* 特例化
        - 提供的模板参数实参必须与原模板的形参类型相匹配
            - 例如下面的例子，`const char * const &`匹配`const T &`，其中`T = const char * const`
    - 对于如下例子，特例化的第三个版本可以使得`char *`实参调用第三个，而不是第一个版本
    ```
    // first version; can compare any two types
    template <typename T> 
    int compare(const T &, const T &);

    // second version to handle string literals
    template<size_t N, size_t M>
    int compare(const char (&)[N], const char (&)[M]);

    // third version
    // special version of compare to handle pointers to character arrays
    template <>
    int compare(const char * const & p1, const char * const & p2)  // reference of const pointer to const char
    {
        return strcmp(p1, p2);
    }
    ```
- 函数重载与模板特例化
    - *特例化* 的本质是 *实例化* 一个模板，**而非** *重载* 它。因此，特例化**不影响**函数匹配
        - 特例化函数模板时，实际上相当于接管了编译器匹配到此函数模板之后的实例化工作
    - 将一个特殊的函数定义为 *特例化函数模板* 还是 *普通函数* 则会影响函数匹配
    - 普通作用域规则应用于特例化
        - 特例化模板时，原模板声明必须在作用域中
        - 任何使用模板实例的代码之前，特例化版本的声明也必须在作用域中
        - 模板及其特例化版本应该声明在同一个头文件中，所有同名模板的声明应该放在前面，然后是这些模板的特例化版本
            - 否则一旦声明不在域中，编译器不会报错，而是会错误地使用非特例化的模板，造成难以排查到的错误
- *类模板* 特例化
    - 举例：定义`template<> std::hash<Sales_data>`，用于 *无序关联容器* 对于`Sales_data`的散列
    ```
    namespace std 
    {
    template <>                            // we're defining a specialization with
    struct hash<Sales_data>                // the template parameter of Sales_data
    {
        // the type used to hash an unordered container must define these types
        typedef size_t result_type;
        typedef Sales_data argument_type;  // by default, this type needs ==
        
        size_t operator()(const Sales_data & s) const;
        
        // our class uses synthesized copy control and default constructor
    };
    
    size_t hash<Sales_data>::operator()(const Sales_data & s) const
    {
        return hash<string>()(s.bookNo) ^ hash<unsigned>()(s.units_sold) ^ hash<double>()(s.revenue);
    }
    }
    
    template <class T> class std::hash;  // needed for the friend declaration
    
    class Sales_data 
    {
        friend class std::hash<Sales_data>;
        // other members as before
    };
    ```
    - 类模板 *偏特化* （partial specialization）
        - *偏特化* （又称 *部分实例化* ）只适用于类模板，**不**适用于函数模板
            - *偏特化* 时 *不必* 提供全部模板实参
                - 可以只指定一部分而非所有模板参数
                - 或是参数的一部分而非全部特性
            - 偏特化定义时仍旧需要`template <parameters>`
                - `parameters`为这个偏特化版本没有显式提供的模板参数
                - 如果显式提供了全部模板参数（这时候就是普通的特例化），则用空的尖括号
        - 举例
            - 一个小例子
            ```
            template <typename K = size_t, typename V = std::string>
            struct Entry
            {
                void fun() { std::cout << boost::core::demangle(typeid(*this).name()) << std::endl; }

                K k {};
                V v {};
            };

            template <typename K>
            struct Entry<K, char *>
            {
                ~Entry() { delete v; }

                void fun() { std::cout << "partial Entry<K, char *>" << std::endl; }

                K      k {};
                char * v {nullptr};
            };

            template <>
            struct Entry<int, int>
            {
                void fun() { std::cout << "partial Entry<int, int>" << std::endl; }

                int k {233};
                int v {666};
            };
            ```
            - `std::remove_reference`的实现
            ```
            // original, most general template
            template <class T> struct remove_reference       { typedef T type; };
            
            // partial specializations that will be used for lvalue and rvalue references
            template <class T> struct remove_reference<T &>  { typedef T type; };
            template <class T> struct remove_reference<T &&> { typedef T type; };
            
            int i;
            
            // decltype(42) is int, uses the original template
            remove_reference<decltype(42)>::type a;
            
            // decltype(i) is int &, uses first (T &) partial specialization
            remove_reference<decltype(i)>::type b;
            
            // decltype(std::move(i)) is int &&, uses second (i.e., T &&) partial specialization
            remove_reference<decltype(std::move(i))>::type c;
            ```
            - 第一个模板定义了最通用的模板
                - 它可以用 *任意类型* 实例化
                - 将模板实参作为`type`成员的类型
            - 与普通的特例化不一样，偏特化版本需要定义 *模板参数* 
                - 普通特例化 *模板参数* 是空的，因为全都人工指定好了
                - 对每个 *未完全确定类型* 的参数，在特例化版本的模板参数列表中都有一项与之对应
                - 在类名之后，我们要为偏特化的模板参数指定实参，这些实参列于模板名之后的尖括号中
                - 这些实参与原始模板中的参数 *按位置对应* 
            - 偏特化版本的模板参数列表是原始模板的参数列表的一个子集（针对指定模板参数）、或一个特例化版本（针对指定参数特性）
                - 本例中，偏特化版本的模板参数的数目与原始模板相同，但是类型不同
                    - 两个偏特化版本分别用于 *左值引用* 和 *右值引用* 类型
    - 特例化 *成员* 而不是类
        - 可以只特例化类模板的特定成员函数，而不特例化整个模板
    ```
    template <typename T> 
    struct Foo 
    {
        Foo(const T & t = T()): mem(t) { }
        void Bar() { /* ... */ }
        T mem;
        // other members of Foo
    };
    
    template<>            // we're specializing a template
    void Foo<int>::Bar()  // we're specializing the Bar member of Foo<int>
    {
        // do whatever specialized processing that applies to ints
    }
    
    Foo<std::string> fs;  // instantiates Foo<string>::Foo()
    fs.Bar();             // instantiates Foo<string>::Bar()
    Foo<int> fi;          // instantiates Foo<int>::Foo()
    fi.Bar();             // uses our specialization of Foo<int>::Bar()
    ```

#### 可变参数模板（Variadic Templates）

- *可变参数模板* 就是一个接受可变数目的参数的函数模板或类模板
    - 可变数目的参数被称作 *参数包* （parameter packet），包括
        - *模板参数包* （template parameter pack）：零或多个 *模板参数* 
            - *模板形参列表* 中
                - `class ...`和`typename ...`指出接下来的参数表示 *零或多个类型的列表* 
                - 一个 *类型* 后面跟一个 *省略号* 表示 *零或多个给定类型的非类型参数的列表* 
        - *函数参数包* （function parameter pack）：零或多个 *函数参数* 
            - *函数形参列表* 中
                - 如果一个形参的类型是一个 *模板参数包* ，则此参数也是一个 *函数参数包* 
    - 例如
        - 对于如下调用
        ```
        // Args is a template parameter pack; rest is a function parameter pack
        // Args represents zero or more template type parameters
        // rest represents zero or more function parameters
        template <typename T, typename ... Args>
        void foo(const T & t, const Args & ... rest);

        int i = 0; 
        double d = 3.14; 
        std::string s = "how now brown cow";

        foo(i, s, 42, d);  // three parameters in the pack
        foo(s, 42, "hi");  // two parameters in the pack
        foo(d, s);         // one parameter in the pack
        foo("hi");         // empty pack
        ```
        - 编译器会为`foo`实例化出四个版本
        ```
        void foo(const int &, const string &, const int&, const double &);
        void foo(const string &, const int &, const char[3] &);
        void foo(const double &, const string &);
        void foo(const char[3] &);
        ```
    - `sizeof...`运算符
        - 当我们需要知道包中有多少元素时，可以使用`sizeof...`运算符
        - 类似`sizeof`，`sizeof...`也返回 *常量表达式* ，而且**不会**对其实参求值
    ```
    template <typename ... Args> 
    void g(Args ... args) 
    {
        std::cout << sizeof...(Args) << std::endl;  // number of type parameters
        std::cout << sizeof...(args) << std::endl;  // number of function parameters
    }
    ```
- 编写 *可变参数模板函数* 
    - *递归* 包扩展
        - 第一步调用处理包中的 *第一个实参* ，然后用剩下的实参包递归调用自己
            - 剩下的实参包一般也会 *转发* 
        - 为了 *终止递归* ，需要额外定义一个 *非可变参数* 版本
    ```
    template <typename T>
    void variadic_template_recursion_expansion(std::ostream & cout, T && t)
    {
        cout << t << std::endl;
    }

    template <typename T, typename ... Args>
    void variadic_template_recursion_expansion(std::ostream & cout, T && t, Args && ... args)
    {
        cout << t << ", ";
        variadic_template_recursion_expansion(cout, std::forward<T>(args) ...);
    }
    
    variadic_template_recursion_expansion(std::cout, 0, 1, 2, 3);    // 0, 1, 2, 3
    
    template <typename T>
    T sum(T && t)
    {
        return t;
    }

    template <typename T, typename ... Args>
    T sum(T && t, Args && ... rest)
    {
        return t + sum(std::forward<T>(rest) ...);
    }
    
    sum(0, 1, 2, 3)                                                  // 6
    ```
    - *逗号表达式初始化列表* 包扩展
        - 扩展后`(printArg(args), 0) ...`会被替换成由 *逗号表达式* 组成、由逗号分隔的列表
        - 和外面的花括号`{}`正好构成 *初始化列表*
    ```
    template <typename T>
    void printArg(T && t)
    {
        std::cout << t << ", ";
    }

    template <typename ... Args>
    void expand(Args && ... args)
    {
        int arr[] = {(printArg(args), 0) ...};
    }

    expand(0, 1, 2, 3);                                              // 0, 1, 2, 3, 
    ```
- 编写 *可变参数模板类*
    - *模板偏特化递归* 包扩展
        - 基本写法
        ```
        // 基本定义
        template <typename T, typename ... Args>
        struct Sum
        {
            enum
            {
                value = Sum<T>::value + Sum<Args ...>::value
            };
        };

        // 偏特化，递归至只剩一个模板类型参数时终止
        template <typename T>
        struct Sum<T>
        {
            enum
            {
                value = sizeof(T)
            };
        };
        
        std::cout << Sum<char, short, int, double>::value << std::endl;  // 15
        ```
        - 递归终止类还可以有如下写法
        ```
        // 偏特化，递归至只剩两个模板类型参数时终止
        template <typename First, typename Last>
        struct sum<First, Last>
        { 
            enum
            { 
                value = sizeof(First) + sizeof(Last) 
            };
        };
        
        // 偏特化，递归至模板类型参数一个不剩时终止
        template<>
        struct sum<> 
        { 
            enum
            { 
                value = 0 
            }; 
        };
        ```
    - *继承* 包扩展
        - 代码
        ```
        // 整型序列的定义
        template <int ...>
        struct IndexSeq
        {
        };

        // 继承方式，开始展开参数包
        template <int N, int ... Indexes>
        struct MakeIndexes : MakeIndexes<N - 1, N - 1, Indexes ...>
        {
        };

        // 模板特化，终止展开参数包的条件
        template <int ... Indexes>
        struct MakeIndexes<0, Indexes ...>
        {
            typedef IndexSeq<Indexes ...> type;
        };
        
        #include <cxxabi.h>

        std::string demangle(const char * name)
        {
            int status = -4;  // some arbitrary value to eliminate the compiler warning
            std::unique_ptr<char> res {abi::__cxa_demangle(name, nullptr, nullptr, &status)};
            return status ? name : res.get();
        }
        
        using T = MakeIndexes<3>::type;
        std::cout << demangle(typeid(T).name()) << std::endl;  // IndexSeq<0, 1, 2>
        ```
        - 其中`MakeIndexes`的作用是为了生成一个可变参数模板类的整数序列，最终输出的类型是：`IndexSeq<0, 1, 2>`
            - `MakeIndexes`继承于自身的一个特化的模板类
            - 这个特化的模板类同时也在展开参数包
            - 这个展开过程是通过继承发起的，直到遇到特化的终止条件展开过程才结束
            - `MakeIndexes<3>::type`的展开过程是这样的
            ```
            struct MakeIndexes<3> : MakeIndexes<2, 2>
            {
            }
            
            struct MakeIndexes<2, 2> : MakeIndexes<1, 1, 2>
            {
            }
            
            struct MakeIndexes<1, 1, 2> : MakeIndexes<0, 0, 1, 2>
            {
                typedef IndexSeq<0, 1, 2> type;
            }
            ```
            - 通过不断的继承递归调用，最终得到整型序列`IndexSeq<0, 1, 2>`
        - 如果不希望通过继承方式去生成整形序列，则可以通过下面的方式生成
        ```
        template <int N, int ... Indexes>
        struct MakeIndexes3
        {
            using type = typename MakeIndexes3<N - 1, N - 1, Indexes ...>::type;
        };

        template <int... Indexes>
        struct MakeIndexes3<0, Indexes ...>
        {
            typedef IndexSeq<Indexes ...> type;
        };
        ```
- 理解 *包扩展* （Pack Expansion）
    - 对于一个 *参数包* ，我们能对它做得唯一一件事就是 *扩展* 它
        - *扩展* 一个包时，我们还要提供 *模式* （pattern）
            - *模式* 具体就是参数包中的一个元素的 *表达式* 也可以说是应用于一个元素的操作
        - *扩展* 一个包就是把它分解为构成的元素， *对每个元素独立地应用模式* ，在源代码中用扩展后生成的列表替代扩展前的内容
            - `C++`的模板实质就是 *宏* （macros） 
        - *扩展* 操作通过在 *模式* 右边放一个 *省略号* `...` 来触发 
    - 比如，以下可变参数模板函数`print`中包含 *两个扩展*
        ```
        template <typename T, typename ... Args>
        std::ostream & 
        print(std::ostream & os, const T & t, const Args & ... rest)  // expand Args
        {
            os << t << ", ";
            return print(os, rest ...);                               // expand rest
        }
        ```
        - 第一个扩展模板参数包`Args`，为`print`生成函数参数列表
            - 对`Args`的 *扩展* 中，编译器将 *模式* `const Arg &` 应用到模板参数包`Args`中的每个元素
            - 因此，此模式的扩展结果是一个逗号分隔的零个或多个类型的列表，每个类型都形如`const Type &`，例如
            ```
            print(std::cout, i, s, 42);                                    // 包中有 2 个参数
            ```
            - 最后两个实参的类型和模式一起确定了尾置参数的类型，此调用被实例化为
            ```
            std::ostream & print(std::ostream &, const int &, const string &, const int &);
            ```
        - 第二个扩展发生于对`print`的递归调用中， *模式* 是函数参数包的名字`rest`，为`print`生成函数参数列表
            - 此模式扩展出一个由包中元素组成的、逗号分隔的列表
            - 因此，这个调用等价于
            ```
            print(os, s, 42);
            ```
    - 理解包扩展
        - 上述`print`函数的扩展仅仅将包扩展为其构成元素，`C++`语言还允许 *更复杂的扩展模式* 
        - 例如，可以编写第二个`print`，对其每个实参调用`debug_dup`，然后调用`print`打印结果`std::string`
        ```
        // call debug_rep on each argument in the call to print
        template <typename ... Args>
        std::ostream & errorMsg(std::ostream & os, const Args & ... rest)
        {
            // equivlent to: print(os, debug_rep(a1), debug_rep(a2), ..., debug_rep(an)
            return print(os, debug_rep(rest) ...);
        }
        ```
        - 这个`print`使用了模式`debug_rep(rest)`
            - 此模式表示我们希望对函数参数包`rest`中的每个元素调用`debug_rep`
            - 扩展结果是一个逗号分隔的`debug_rep`调用列表，即如下调用
            ```
            errorMsg(std::cerr, fcnName, code.num(), otherData, "other", item);
            ```
            - 就好像我们这样编写代码一样
            ```
            print(std::cerr, debug_rep(fcnName), debug_rep(code.num()),
                             debug_rep(otherData), debug_rep("otherData"),
                             debug_rep(item));
            ```
            - 与之相对地，如下模式将会失败
            ```
            // passes the pack to debug_rep; print(os, debug_rep(a1, a2, ..., an))
            print(os, debug_rep(rest...));  // error: no matching function to call
            ```
            - 其问题就是在`debug_rep`的 *调用之中* ，而不是 *之外* ，扩展了`rest`，它实际等价于
            ```
            print(cerr, debug_rep(fcnName, code.num(), otherData, "otherData", item));
            ```
- 转发参数包
    - 可以组合使用`std::forward`和 *可变参数模板* 来编写函数
        - 实现将其实参不变地传递给其他函数
        - 标准库容器的`emplace_back`方法就是可变参数成员函数模板
    - 以`StrVec::emplace_back`为例
        - 代码
        ```
        class StrVec 
        {
        public:
            template <class ... Args> void emplace_back(Args && ...);
            // remaining members as in § 13.5 (p. 526)
        };

        template <class... Args>
        inline void StrVec::emplace_back(Args && ... args)
        {
            chk_n_alloc(); // reallocates the StrVec if necessary
            alloc.construct(first_free++, std::forward<Args>(args) ...);
        }
        ```
        - `alloc.construct (since C++11)(deprecated in C++17)(removed in C++20)`调用的扩展为`std::forward<Args>(args) ...`
            - 它既扩展了 *模板参数包* `Args`，又扩展了 *函数参数包* `args`
            - 此 *模式* 生成如下形式的元素`std::forawrd<T_i>(t_i)`，例如
                - `svec.emplace_back(10, 'c');`会被扩展为`std::forward<int>(10), std::forward<char>(c)`
                - `svec.emplace_back(s1 + s2);`会被扩展为`std::forward<std::string>(std::string("the end"))`
            - 这保证了如果`emplace_back`接受的是右值实参，则`construct`也会接受到右值实参
- 建议：转发和可变参数模板
    - 可变参数模板通常将它们的参数转发给其他函数。这种函数通常具有如下形式
    ```
    // fun has zero or more parameters each of which is
    // an rvalue reference to a template parameter type
    template <typename ... Args>
    void fun(Args && ... args)  // expands Args as a list of rvalue references
    {
        // the argument to work expands both Args and args
        work(std::forward<Args>(args) ...);
    }
    ```
    - 这里我们希望将`fun`的所有实参转发给另一个名为`work`的函数，假定由它完成函数的实际工作
    - 类似`emplace_back`对`construct`的调用，`work`调用中的扩展既扩展了模板参数包又扩展了函数参数包
    - 由于`fun`的形参是右值引用，因此我们既可以传递左值又可以传递右值
    - 由于`std::forward<Args>(args) ...`，`fun`所有实参的类型信息在调用`work`时都能得到保持






### 🌱 [Chap 17] 标准库特殊设施

#### [`std::tuple`](https://en.cppreference.com/w/cpp/utility/tuple)

- 定义于`<utility>`中，是`std::pair`的推广
```
template <class ... Types>
class tuple;
```
- 支持的操作
    - 定义和初始化
        - `std::tuple<T1, T2, T3...> t;`： *默认初始化* ，创建`std::tuple`，成员进行 *值初始化* 
        - `std::tuple<T1, T2, T3...> t(v1, v2, v3...);`： *显式构造* ，创建`std::tuple`，成员初始化为给定值。此构造函数为`explicit`的
        - `std::tuple<T1, T2，T3...> t = {v1, v2, v3...};`： *列表初始化* ，创建`std::tuple`，成员初始化为给定值
        - [`std::make_tuple(v1, v2, v3...);`](https://en.cppreference.com/w/cpp/utility/tuple/make_tuple)：创建`std::tuple`，元素类型由`v1`、`v2`、`v3`等自动推断。成员初始化为给定值
    - 关系运算
        - `t1 == t2`：字典序判等
        - `t1 != t2`，`t1 relop t2`：字典序比较 `(removed in C++20)`
        - `t1 <=> t2`：字典序比较 `(since C++20)`
    - 赋值和对换
        - `operator=`：拷贝或移动赋值
        - `swap`：对换`std::tuple`的内容
        ```
        std::tuple<int, std::string, float> p1, p2;
        p1 = std::make_tuple(10, "test", 3.14);
        p2.swap(p1);
        printf("%d %s %f\n", std::get<0>(p2), std::get<1>(p2).c_str(), std::get<2>(p2));  // 10 test 3.14
        ```
        - `std::swap<TupleType>(t1, t2)`：`std::swap<T>`关于`std::tuple`类型的重载，相当于`t1.swap(t2)`
    - 成员访问
        - [`std::get<i>(t)`](https://en.cppreference.com/w/cpp/utility/tuple/get)：获取`t`的第`i`个数据成员的引用，`或元素类型为 i 的数据成员的引用 (since C++14)`
            - 如果`t`为 *左值* ，则返回 *左值引用* ；否则，返回 *右值引用* 
        ```
        auto t = std::make_tuple(1, "Foo", 3.14);
        // index-based access
        std::cout << "(" << std::get<0>(t) << ", " << std::get<1>(t)
                  << ", " << std::get<2>(t) << ")\n";
                  
        // type-based access (since C++14)
        std::cout << "(" << std::get<int>(t) << ", " << std::get<const char*>(t)
                  << ", " << std::get<double>(t) << ")\n";
                  
        // Note: std::tie and structured binding may also be used to decompose a tuple
        ```
        - [`std::tie`](https://en.cppreference.com/w/cpp/utility/tuple/tie)
            - 可能的实现
            ```
            namespace detail 
            {
            struct ignore_t 
            {
                template <typename T>
                const ignore_t & operator=(const T &) const { return *this; }
            };
            }
            
            const detail::ignore_t ignore;
             
            template <typename ... Args>
            auto tie(Args & ... args) 
            {
                return std::tuple<Args & ...>(args ...);
            }
            ```
            - 用其实参的 *左值引用* 创建一个`std::tuple`
                - 常用于用指定的参数解包`std::tuple`或`std::pair`
                - 可以传入`std::ignore`表示该位置元素不需解包
            ```
            std::tuple<int, std::string, double, double> tup {0, "pi", 3.14, 3.14159};
            int a;
            std::string s;
            double pi;
            std::tie(a, s, pi, std::ignore) = tup;
            printf("%d, %s, %lf\n", a, s, pi);                   // 0, pi, 3.14
            ```
        - [Structured Binding](https://skebanga.github.io/structured-bindings/) `(since C++17)`
            - `C++`早晚得活成`Python`的样子
            ```
            std::tuple<int, std::string, double, double> tup {0, "pi", 3.14, 3.14159};
            auto [a, s, pi, pi2] = tup;
            printf("%d %s %lf %lf\n", a, s.c_str(), pi, pi2);
            
            auto tup = std::make_tuple(0, "pi", 3.14, 3.14159);  // 0 pi 3.140000 3.141590
            auto [a, s, pi, pi2] = tup;
            printf("%d %s %lf %lf\n", a, s, pi, pi2);            // 0 pi 3.140000 3.141590
            ```
        - [`std::forward_as_tuple`](https://en.cppreference.com/w/cpp/utility/tuple/forward_as_tuple)
            - 可能的实现
            ```
            template < class... Types >
            tuple<Types && ...> 
            forward_as_tuple(Types && ... args) noexcept
            {
                return std::tuple<Types && ...>(std::forward<Types>(args) ...);
            }
            ```
            - 将接受的实参完美转发并用之构造一个`std::tuple`
            ```
            std::map<int, std::string> m;
            m.emplace(std::forward_as_tuple(10), std::forward_as_tuple(20, 'a'));
            std::cout << "m[10] = " << m[10] << std::endl;
         
            // The following is an error: it produces a
            // std::tuple<int &&, char &&> holding two dangling references.
            auto t = std::forward_as_tuple(20, 'a');                            // error: dangling reference
            m.emplace(std::piecewise_construct, std::forward_as_tuple(10), t);  // error
            ```
        - [`std::tuple_cat`](https://en.cppreference.com/w/cpp/utility/tuple/tuple_cat)
            - 签名
            ```
            template <class ... Tuples>
            std::tuple<CTypes ...> tuple_cat(Tuples && ... args);
            ```
            - 用`args`中所有`std::tuple`中的元素创建一个大`std::tuple`
            ```
            int n = 1;
            auto t = std::tuple_cat(std::make_tuple("Foo", "bar"), std::tie(n));  // ("Foo", "bar", 1)
            ```
    - *辅助模板类* （helper template classes）
        - `std::tuple_size<TupleType>::value`：类模板，通过一个`std::tuple`的类型来初始化。有一个名为`value`的`public constexpr static`数据成员，类型为`size_t`，表示给定`std::tuple`类型中成员的数量
        - `std::tuple_element<i, TupleType>::type`：类模板，通过一个 *整形常量* 和一个`std::tuple`的类型来初始化。有一个名为`type`的`public typedef`，表示给定`std::tuple`类型中指定成员的类型
        - `std::ignore`：未指定类型的对象，任何值均可赋此对象，且无任何效果。用作`std::tie(a, b, c...)`解包`std::tuple`时的 *占位符* 
- 定义和初始化
    - 定义`std::tuple`时需要指出每个成员的类型
    ```
    std::tuple<size_t, size_t, size_t> threeD;         // all three members value initialized to 0
    
    std::tuple<std::string, std::vector<double>, int, std::list<int>> 
    someVal("constants", {3.14, 2.718}, 42, {0, 1, 2, 3, 4, 5});
    
    tuple<size_t, size_t, size_t> threeD = {1, 2, 3};  // error: explicit tuple(Args && ... arg)
    tuple<size_t, size_t, size_t> threeD {1, 2, 3};    // ok
    
    // tuple that represents a bookstore transaction: ISBN, count, price per book
    auto item = std::make_tuple("0-999-78345-X", 3, 20.00);
    ```
- 成员访问
    - `std::tuple`的成员一律为 *未命名* 的
    - 使用`std::get<i>(tup)`
    ```
    auto book = get<0>(item);       // returns the first member of item
    auto cnt = get<1>(item);        // returns the second member of item
    auto price = get<2>(item)/cnt;  // returns the last member of item
    get<2>(item) *= 0.8;            // apply 20% discount
    ```
    - 不知道`std::tuple`的准确类型信息时，使用 *辅助模板类* 来查询成员数量和类型
        - 使用`std::tuple_size`和`std::tuple_element`需要知道`std::tuple`的类型，可以使用`decltype(t)`
    ```
    typedef decltype(item) trans;                           // trans is the type of item
    
    // returns the number of members in object's of type trans
    size_t sz = std::tuple_size<trans>::value;              // returns 3
    
    // cnt has the same type as the second member in item
    std::tuple_element<1, trans>::type cnt = get<1>(item);  // cnt is an int
    ```
- 返回
```
std::tuple<int, int> foo_tuple() 
{
    return {1, -1};
    return std::tuple<int, int> {1, -1};
    return std::make_tuple(1, -1);
}
```
- 答疑：为什么`std::tuple`**不**支持 *下标* ，而非要用`std::get<i>(t)`
    - 因为`operator[]`是 *函数* ，函数的返回值类型必须在编译期确定
    - 而`std::tuple`元素数量和类型偏偏都不确定，因此无法定义出函数，只能用带模板的`std::get<i>(t)`

#### [`std::bitset`](https://en.cppreference.com/w/cpp/utility/bitset)

- `std::bitset`
    - 使得位运算的使用变得更容易
    - 能够处理超过 *最长整形类型大小* （`unsigned long long`有`64 bit`）的位集合
    - *下标* ： *最低位* 为`0`，以此开始向高位递增
- 定义和初始化
    - 签名
    ```
    template <std::size_t N>
    class bitset;
    ```
    - 构造函数
        - `std::bitset<n> b;`：`b`有`n bit`，每一位都是`0`。此构造函数是`constexpr`
        - `std::bitset<n> b(u);`：`b`是`unsigned long long`类型值`u`的 *低`n`位* 的拷贝，若`n > 64`则多出的高位置为`0`。此构造函数是`constexpr`
        - `std::bitset<n> b(s, pos = 0, m = std::string::npos, zero = '0', one = '1');`：`b`是`std::string`类型值`s`的 *从`pos`开始的`m`个字符* 的拷贝。`s` *只能* 包含 *字符* `zero`或`one`，否则抛出`std::invalid_argument`异常。此构造函数是`explicit`的
        - `std::bitset<n> b(cp, pos = 0, m = std::string::npos, zero = '0', one = '1');`：`b`是 *指向`C`风格字符数组的指针* `cp`所指向的字符串的 *从`pos`开始的`m`个字符* 的拷贝。`cp` *只能* 包含 *字符* `zero`或`one`，否则抛出`std::invalid_argument`异常。此构造函数是`explicit`的
    - 初始化`std::bitset`
    ```
    std::bitset<32> bitvec(1U);       // bits are 0000 0000 0000 0000 0000 0000 0000 0001
    
    // bitvec1 is smaller than the initializer; high-order bits from the initializer are discarded
    std::bitset<13> bitvec1(0xbeef);  // bits are    1 1110 1110 1111
    
    // bitvec2 is larger than the initializer; high-order bits in bitvec2 are set to zero
    std::bitset<20> bitvec2(0xbeef);  // bits are 0000 1011 1110 1110 1111
    
    // on machines with 64-bit long long 0ULL is 64 bits of 0, so ~0ULL is 64 ones
    std::bitset<128> bitvec3(~0ULL);  // bits 0 ... 63 are one; 63 ... 127 are zero
    
    std::bitset<32> bitvec4("1100");  // bits are 0000 0000 0000 0000 0000 0000 0000 1100 
    
    std::string str("1111111000000011001101");
    std::bitset<32> bitvec5(str, 5, 4);            // four bits starting at str[5], 1100
    std::bitset<32> bitvec6(str, str.size() - 4);  // use last four characters
    ```
- 支持的操作
    - `b.any()`：`b`中是否存在 *置位* 的二进制位
    - `b.all()`：`b`中是否都是 *置位* 的二进制位
    - `b.none()`：`b`中是否**没有** *置位* 的二进制位
    - `b.count()`：`b`中 *置位* 的二进制位的个数
    - `b.size()`：`b`中的位数，`constexpr`函数
    - `b.test(pos)`：若`b`中`pos`是 *置位* 的，则返回`true`，否则返回`false`。若`pos`非法，则抛出`std::out_of_range`异常 
    - `b.set(pos, v = true)`：将`b`中`pos`处设置为`bool`值`v`
    - `b.set()`：将`b`中所有位全部 *置位* 
    - `b.reset(pos)`：将`b`中`pos`处 *复位*
    - `b.reset()`：将`b`中所有位全部 *复位* 
    - `b.flip(pos)`：将`b`中`pos`处 *置反*
    - `b.flip()`：将`b`中所有位全部 *置反* 
    - `b[pos]`：返回`b`中第`pos`位的 *引用* 。如果`b`为`const`，则返回`true`或`false`。`pos`非法时 *行为未定义*
    - `b.to_ulong()`：返回对应的`unsigned long`。如果放不下，抛出`std::overflow_error`异常
    - `b.to_ullong()`：返回对应的`unsigned long long`。如果放不下，抛出`std::overflow_error`异常
    - `b.to_string(zero = '0', one = '1')`：返回对应的`std::string`
    - `std::cout << b`：相当于`std::cout << b.to_string();`
    - `std::cin >> b`：读入到`b`，当下一个字符不是`'0'`或`'1'`时、或已经读入`b.size()`个位时，读取过程停止
```
std::bitset<32> bitvec(1U);       // 32 bits; low-order bit is 1, remaining bits are 0
bool is_set = bitvec.any();       // true, one bit is set
bool is_not_set = bitvec.none();  // false, one bit is set
bool all_set = bitvec.all();      // false, only one bit is set
size_t onBits = bitvec.count();   // returns 1
size_t sz = bitvec.size();        // returns 32
bitvec.flip();                    // reverses the value of all the bits in bitvec
bitvec.reset();                   // sets all the bits to 0
bitvec.set();                     // sets all the bits to 1

bitvec.flip(0);                   // reverses the value of the first bit
bitvec.set(bitvec.size() - 1);    // turns on the last bit
bitvec.set(0, 0);                 // turns off the first bit
bitvec.reset(i);                  // turns off the ith bit
bitvec.test(0);                   // returns false because the first bit is off

bitvec[0] = 0;                    // turn off the bit at position 0
bitvec[31] = bitvec[0];           // set the last bit to the same value as the first bit
bitvec[0].flip();                 // flip the value of the bit at position 0
~bitvec[0];                       // equivalent operation; flips the bit at position 0
bool b = bitvec[0];               // convert the value of bitvec[0] to bool

unsigned long ulong = bitvec3.to_ulong();
std::cout << "ulong = " << ulong << std::endl;

std::bitset<16> bits;
std::cin >> bits;                            // read up to 16 1 or 0 characters from cin
std::cout << "bits: " << bits << std::endl;  // print what we just read
```
- 使用`std::bitset`
```
bool status;

// version using bitwise operators
unsigned long quizA = 0;          // this value is used as a collection of bits
quizA |= 1UL << 27;               // indicate student number 27 passed
status = quizA & (1UL << 27);     // check how student number 27 did
quizA &= ~(1UL << 27);            // student number 27 failed

// equivalent actions using the bitset library
std::bitset<30> quizB;            // allocate one bit per student; all bits initialized to 0
quizB.set(27);                    // indicate student number 27 passed
status = quizB[27];               // check how student number 27 did
quizB.reset(27);                  // student number 27 failed
```

#### [正则表达式库](https://en.cppreference.com/w/cpp/regex)

- *`ECMAScript`正则表达式* 
    - `C++`正则表达式标准库`<regex>`采用的的默认文法
    - 基础文法
        - *普通字符* 
            - 未被显式指定为元字符的所有可打印和不可打印字符，包括
                - 所有大写
                - 所有小写字母
                - 所有数字
                - 所有标点符号
                - 一些其他符号
        - *非打印字符* 
            - 也可以是正则表达式的组成部分，包括
                - `cX`：匹配`Ctrl + X`或对应的控制字符，`X`必须为`[a-zA-Z]`之一，否则视为字面的`c`
                    - 例如`\cM`匹配`Ctrl + M`或 *回车*
                - `\f`：匹配 *换页符* （form feed），等价于`\x0c`或`\cL`
                - `\n`：匹配 *换行符* （line feed），等价于`\x0a`或`\cJ`
                - `\r`：匹配 *回车符* （carriage return），等价于`\x0d`或`\cM`
                - `\t`：匹配 *水平制表符* （horizontal tab），等价于`\x09`或`\cI`
                - `\v`：匹配 *垂直制表符* （vertical tab），等价于`\x0b`或`\cK`
                - `\s`：匹配任何 *空白字符* ，等价于`[ \f\n\r\t\v]`
                - `\S`：匹配任何 *非空白字符* ，等价于`[^ \f\n\r\t\v]`
        - *元字符* 
            - 如果要字面匹配如下有特殊意义的元字符，必须加`\`转义
            - `^`：匹配输入字符串 *开始* 。如在 *字符簇* `[]`中使用，则表示 *不接受字符簇* 中的字符
            - `$`：匹配输入字符串 *结尾* 
            - `()`：匹配 *子表达式* 开始和结尾
            - `[]`：匹配 *字符簇* 开始和结尾
            - `{}`：匹配 *限定符* 开始和结尾
            - `*`：匹配前面的子表达式 *零或多次* 
            - `+`：匹配前面的子表达式 *一或多次* 
            - `?`：匹配前面的子表达式 *零或一次* 
            - `.`：匹配**除 *换行符* `\n`之外**的 *任意字符* 
            - `\`： *转义* 字符
            - `|`：两项之间的 *选择* 
        - *限定符* 
            - 限定符用来指定正则表达式的一个给定组件必须要出现多少次才能满足匹配
            - `*`：匹配前面的子表达式 *零或多次* 
            - `+`：匹配前面的子表达式 *一或多次* 
            - `?`：匹配前面的子表达式 *零或一次* 
            - `{n}`：匹配前面的子表达式 *`n`次* 
            - `{n,}`：匹配前面的子表达式 *至少`n`次* 
                - `{0,}`等价于`*`
                - `{1,}`等价于`+`
            - `{n,m}`：匹配前面的子表达式 *`n`到`m`次* 
        - *贪婪* 和 *非贪婪*
            - `*`和`+`默认贪婪，即尽可能多的匹配文字
            - 在它们的后面加上一个`?`就可以实现非贪婪或最小匹配，例如对于输入字符串`<h1>RUNOOB</h1>`
                - `<.*>`会匹配整个字符串`<h1>RUNOOB</h1>`
                - `<.*?>`只会匹配整个字符串`<h1>`
        - *定位符* 
            - 定位符使您能够将正则表达式固定到行首或行尾
            - 它们还使您能够创建这样的正则表达式，这些正则表达式出现在一个单词内、在一个单词的开头或者一个单词的结尾
            - `^`：匹配输入字符串 *开始* 
            - `$`：匹配输入字符串 *结尾* 
            - `\b`：匹配 *单词边界* ，即字与空格间的位置
            - `\B`：匹配 *非单词边界* 
        - 其他黑话
            - `\d`：匹配 *数字* ，等价于`[0-9]`
            - `\D`：匹配 *非数字* ，等价于`[^0-9]`
            - `\w`：匹配 *字母、数字、下划线* ，等价于`[A-Za-z0-9_]`
            - `\W`：匹配 *非字母、数字、下划线* ，等价于`[^A-Za-z0-9_]`
            - `\xN`：匹配十六进制转义值，`N`必须长度为`2`
                - 如`\x041`匹配`\x04`和`1`
    - *选择* 
        - 用圆括号`()`将所有选择项括起来，相邻的选择项之间用`|`分隔
    - *簇* 
        - 用方括号`[]`将一些单体元素括起来，匹配它们中的任一个，加入`^`表示**不**匹配其中任一个
            - 举例
                - `[AaEeIiOoUu]`：匹配所有的元音
                - `[a-z]`：匹配所有的小写字母 
                - `[A-Z]`：匹配所有的大写字母 
                - `[a-zA-Z]`：匹配所有的字母 
                - `[^a-zA-Z]`：匹配任何非字母
                - `[0-9]`：匹配所有的数字 
                - `[0-9\.\-]`：匹配所有的数字，句号和减号 
                - `[ \f\r\t\n]`：匹配所有的空白字符，相当于`\s`
                - `[^ \f\r\t\n]`：匹配所有的非空白字符，相当于`^\s`
            - 一些预设黑话
                - `[[:alpha:]]`：任何字母，相当于`[a-zA-Z]`
                - `[^[:alpha:]]`：任何字母，相当于`[^a-zA-Z]`
                - `[[:digit:]]`：任何数字，相当于`\d`
                - `[^[:digit:]]`：任何非数字，相当于`\D`
                - `[[:alnum:]]`：任何字母和数字
                - `[[:space:]]`：任何空白字符，相当于`\s`
                - `[^[:space:]]`：任何非空白字符，相当于`\S`
                - `[[:upper:]]`：任何大写字母
                - `[[:lower:]]`：任何小写字母
                - `[[:punct:]]`：任何标点符号
                - `[[:xdigit:]]`：任何十六进制的数字，即`[0-9a-fA-F]`
- `C++` *正则表达式* 标准库`<regex>`
    - `std::regex`：表示有一个正则表达式的类
    - `std::regex_match`：将一个字符序列与一个正则表达式相匹配，要求 *全文匹配*
    - `std::regex_search`：寻找第一个与正则表达式匹配的子序列
    - `std::regex_replace`：使用给定格式替换一个正则表达式
    - `std::sregex_iterator`：迭代器适配器，调用`regex_search`来遍历一个`std::string`中所有匹配的子串
    - `std::sregex_token_iterator`：迭代器适配器，按照正则表达式将输入序列划分成子串并一一遍历
    - `std::smatch`：容器类，保存在`std::string`中搜索的结果
    - `std::ssub_match`：`std::string`中匹配的子表达式的结果
- `std::regex`系列函数
    - 这些函数都返回`bool`，指示是否找到了匹配，且都被重载了
    - `std::regex_search`和`std::regex_match`：在字符序列`seq`中查找`regex`对象`r`中的正则表达式
        - 形参列表
            - `(seq, m, r, mft)`
            - `(seq, r, mft)`
        - `seq`可以是
            - `std::string`
            - 表示范围的一对迭代器
            - `C`风格字符串
        - `r`是一个`std::regex`对象
        - `m`是一个`std::smatch`对象，用来保存匹配结果的相关细节。`m`和`seq`必须具有兼容的类型
        - `mft`是一个 *可选* 的 *匹配标志* => 17.3.4
- 使用正则表达式库    
    - 指定`std::regex`对象的选项
        - `std::regex(re);`：`re`是一个 *正则表达式* ，可以是一个`std::string`、表示字符范围的 *迭代器对* 、 *`C`风格字符串* 、 *`char *`和计数器对* 或是 *花括号包围的字符列表* 
        - `std::regex(re, f);`：在上一项的基础上，按照`f`指出的 *选项标志* 处理对象
            - `f`是`std::regex_constants::syntax_option_type`类型的 *`unsigned int`枚举* 值，具体可以是
                - 匹配规则（同时是`std::regex`和`std::regex_constants`的静态成员）
                    - `std::regex::icase`：匹配时忽略大小写
                    - `std::regex::nosubs`：**不**保存匹配的表达式
                    - `std::regex::optimize`：执行速度优先于构造速度
                - 正则表达式语言（同时是`std::regex`和`std::regex_constants`的静态成员）， *只能有一个* 
                    - `std::regex::ECMAScript`：使用`ECMA-262`语法， *默认选项* 
                    - `std::regex::basic`：使用`POSIX` *基本* 正则表达式语法
                    - `std::regex::extended`：使用`POSIX` *扩展* 正则表达式语法
                    - `std::regex::awk`：使用`POSIX` `awk`正则表达式语法
                    - `std::regex::grep`：使用`POSIX` `grep`正则表达式语法
                    - `std::regex::egrep`：使用`POSIX` `egrep`正则表达式语法
        - `r1 = re;`：将`r1`中的正则表达式替换为`re`。`re`可以是一个`std::string`、表示字符范围的 *迭代器对* 、 *`C`风格字符串* 、 *`char *`和计数器对* 或是 *花括号包围的字符列表* 
        - `r1.assign(re, f);`：与使用 *赋值运算符* `=`效果相同，`f`为 *选项标志* 
        - `r.mark_count()`：`r`中 *子表达式* 的数目
        - `r.flags()`：返回`r`的 *标志集* ，`typedef regex_constants::syntax_option_type flag_type`
        - 注： *构造函数* 和 *赋值* 操作可能抛出类型为`std::regex_error`的异常
    - `例17.1`：查找拼写错误（违反规则 *除在`c`之后时以外，`i`必须在`e`之前* ）
        - 默认情况下使用的正则表达式语言是`ECMAScript`
            - `[^c]`匹配 *任意不是`c`的字母*
            - `[[:alpha:]]`匹配 *任意字母* 
            - `[[:alnum:]]`匹配 *任意数字 
            - `+`匹配 *一或多个* 
            - `*`匹配 *零或多个*   
    ```
    // find the characters ei that follow a character other than c
    std::string pattern("[^c]ei");
    
    // we want the whole word in which our pattern appears
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    std::regex r(pattern);                        // construct a regex to find pattern
    std::smatch results;                          // define an object to hold the results of a search
    
    // define a string that has text that does and doesn't match pattern
    std::string test_str = "receipt freind theif receive";
    
    // use r to find a match to pattern in test_str
    if (std::regex_search(test_str, results, r))  // if there is a match
    {
        std::cout << results.str() << std::endl;  // print the matching word: freind
    }  
    ```  
    - `例17.2`：匹配`C++`源文件扩展名
        - 默认情况下使用的正则表达式语言是`ECMAScript`
            - `.`匹配 *任意字符*
            - `\\.`转义为匹配字面`.`
                - `\`在`C++`字符串字面量中本身又是转义字符，因此其本身也需要一次转义
    ```
    // one or more alphanumeric characters followed by a '.' followed by "cpp" or "cxx" or "cc"
    std::regex r("[[:alnum:]]+\\.(cpp|cxx|cc)$", std::regex::icase);
    std::smatch results;
    std::string filename;
    
    while (std::cin >> filename)
    {
        if (std::regex_search(filename, results, r))
        {
            std::cout << results.str() << std::endl;  // print the current match
        }
    }  
    ```
    - 指定或使用正则表达式时的错误
        - 正则表达式本身有自己的语法，不由`C++`编译器编译，是否正确需要在运行时解析
        - 如果正则表达式本身有语法错误，则运行时会抛出`std::regex_error`异常
            - `e.what()`：描述发生了什么错误
            - `e.code()`：错误类型对应的编码，具体数值 *由实现定义* 
        - *错误类型* ：正则表达式库能抛出的标准错误，`std::regex_constants::error_type`枚举类型值
            - `std::regex_constants::error_collate`：无效的元素校对请求
            - `std::regex_constants::error_ctype`：无效的字符类
            - `std::regex_constants::error_escape`：无效的转义字符或无效的尾置转义
            - `std::regex_constants::error_backref`：无效的向后引用
            - `std::regex_constants::error_brack`：不匹配的方括号`[]`
            - `std::regex_constants::error_paren`：不匹配的圆括号`()`
            - `std::regex_constants::error_brace`：不匹配的花括号`{}`
            - `std::regex_constants::error_badbrace`：花括号`{}`中的无效范围
            - `std::regex_constants::error_range`：无效的字符范围，如`[z-a]`
            - `std::regex_constants::error_space`：内存不足，无法处理此正则表达式
            - `std::regex_constants::error_badrepeat`：重复字符`*`、`?`、`+`或`{n}`之前没有有效的正则表达式
            - `std::regex_constants::error_complexity`：要求的匹配过于复杂
            - `std::regex_constants::error_stack`：栈空间不足，无法处理匹配
    - `例17.3`：捕获错误
    ```
    try 
    {
        // error: missing close bracket after alnum; the constructor will throw
        std::regex r("[[:alnum:]+\\.(cpp|cxx|cc)$", std::regex::icase);
    } 
    catch (std::regex_error e)
    { 
        std::cout << e.what() << "\ncode: " << e.code() << std::endl; 
    }
    
    // Unexpected character in bracket expression.
    // code: 4
    ```
    - 避免创建不必要的正则表达式
        - 正则表达式的编译发生于程序运行时，非常耗时
        - 为了最小化开销，应当避免创建不必要的正则表达式
        - 例如，在循环中使用正则表达式时，应该在循环之外创建而不是每步迭代时都编译一次
    - 正则表达式与输入序列类型
        - 可以搜索多种类型的输入序列
            - 输入可以是包括`char`、`wchar_t`数据
            - 字符可以保存于`std::string`或`C`风格字符串`const char *`中（或对应的宽字符版本，`std::wstring`以及`const wchar_t *`）
        - *输入类型* 及其对应的 *正则表达式库类型* 
            - `std::string`：`std::regex`、`std::smatch`、`std::ssub_match`和`std::sregex_iterator`
            - `const char *`：`std::regex`、`std::cmatch`、`std::csub_match`和`std::cregex_iterator`
            - `std::wstring`：`std::wregex`、`std::wsmatch`、`std::wssub_match`和`std::wsregex_iterator`
            - `const wchar_t *`：`std::wregex`、`std::wcmatch`、`std::wcsub_match`和`std::wcregex_iterator`
        - 使用的 *正则表达式库类型* 必须与 *输入类型* 匹配
            - 例如`std::smatch`用于保存`std::string`的匹配结果，对于`C`风格字符串则必须使用`std::cmatch`
            ```
            // <regex.h>
            // namespace std
            typedef match_results<const char *>               cmatch;
            typedef match_results<string::const_iterator>     smatch;
            typedef match_results<const wchar_t *>            wcmatch;
            typedef match_results<wstring::const_iterator>    wsmatch;
            ```
            - 以下程序会报编译错误
            ```
            // wrong
            std::regex r("[[:alnum:]]+\\.(cpp|cxx|cc)$", std::regex::icase);
            std::smatch results;  // will match a string input sequence, but not char *
            
            if (std::regex_search("myfile.cc", results, r))  // error: char * input
            {
                std::cout << results.str() << std::endl;
            }
            ```
            - 正确写法
            ```
            // correct
            std::cmatch results;  // will match character array input sequences
            
            if (std::regex_search("myfile.cc", results, r))
            {
                std::cout << results.str() << std::endl;     // print the current match
            }
            ```
- [`std::regex_iterator`](https://en.cppreference.com/w/cpp/regex/regex_iterator)
    - 下面以`std::string`输入为例，对其他输入类型对应的正则表达式库类型一样适用
    - `std::sregex_iterator`操作
        - `std::sregex_iterator it(b, e, r);`：创建一个`std::sregex_iterator`，遍历迭代器`[b, e)`表示的`std::string`。它调用`std::sregex_search(b, e, r)`将`it`定位到输入中 *第一个* 匹配的位置
        - `std::sregex_iterator end;`：`std::sregex_iterator`的 *尾后迭代器*
        - `*it`：根据上一次调用`std::regex_match`的结果，返回一个`sts::smatch`对象的 *引用* 
        - `it->`：根据上一次调用`std::regex_match`的结果，返回一个`sts::smatch`对象的 *指针* 
        - `++it`，`it++`：从输入序列当前匹配位置开始调用`std::regex_search`，前置版本返回递增后的迭代器；后置版本返回旧值
        - `it1 == it2`，`it1 != it2`：如果两个`std::sregex_iterator`都是尾后迭代器，或都是从同一个序列构造出的非尾后迭代器，则它们相等
    - 可以使用`std::sregex_iterator`来获得所有匹配
    ```
    // find the characters ei that follow a character other than c
    std::string str_to_test {"receipt freind theif receive"};
    std::string pattern {"[^c]ei"}; 
    // we want the whole word in which our pattern appears
    pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
    // we'll ignore case in doing the match
    std::regex r {pattern, std::regex::icase};  
    
    // it will repeatedly call regex_search to find all matches in file
    for (std::sregex_iterator it {str_to_test.begin(), str_to_test.end(), r}, end_it; it != end_it; ++it)
    {
        std::cout << it->str() << std::endl;  // matched word
    }
    ```
    - `std::smatch`操作，也适用于`std::cmatch`、`std::wsmatch`、`std::wcmatch`以及对应的`std::ssub_match`、`std::csub_match`、`std::wssub_match`、`std::wcsub_match`
        - `m.ready()`：如果已经通过调用`std::regex_match`或`std::regex_search`设置了`m`，则返回`true`；否则返回`false`。访问未设置的`m`是 *未定义行为* 
        - `m.size()`：如果匹配失败，则返回`0`；否则，返回最近一次正则表达式中 *子表达式* 的数目
        - `m.empty()`：`return m.size() == 0;`
        - `m.prefix()`：一个`std::ssub_match`对象，表示当前匹配之前的序列
        - `m.suffix()`：一个`std::ssub_match`对象，表示当前匹配之后的部分
        - `m.format(...)`：用于正则表达式 *替换* 操作`std::regex_replace` => 17.3.4
        - 在接受 *索引* 的操作中，`n`默认值为`0`，且必须小于`m.size()`。第一个子匹配（索引为`0`）表示 *整个匹配*
        - `m.length(n)`：第`n`个匹配的子表达式的大小
        - `m.position(n)`：第`n`个匹配的子表达式距序列开始的距离
        - `m.str(n)`：第`n`个匹配的子表达式匹配的`std::string`
        - `m[n]`：对应第`n`个子表达式的`std::ssub_match`对象
        - `m.begin()`，`m.end()`：表示`m`中`std::sub_match`元素范围的迭代器
        - `m.cbegin()`，`m.cend()`：表示`m`中`std::sub_match`元素范围的常迭代器
    - 使用匹配数据
        - `std::smatch`的`prefix`和`suffix`成员函数分别表示输入序列中当前匹配之前和之后部分的`std::ssub_match`对象
        - 一个`std::ssub_match`对象有两个名为`str`和`length`的成员函数，分别返回匹配的`std::string`及其长度
        - 可以用这些操作重写语法程序的循环，输出匹配的上下文
        ```
        // find the characters ei that follow a character other than c
        std::string file {"receipt freind theif receive"};
        std::string pattern {"[^c]ei"};
        // we want the whole word in which our pattern appears
        pattern = "[[:alpha:]]*" + pattern + "[[:alpha:]]*";
        // we'll ignore case in doing the match
        std::regex r {pattern, std::regex::icase};

        // same for loop header as before
        for (std::sregex_iterator it {file.begin(), file.end(), r}, end_it; it != end_it; ++it)
        {
            // size of the prefix
            // we want up to 40 characters
            std::string::size_type pos = std::max(it->prefix().length() - 40, 0);                      
            
            std::cout << it->prefix().str().substr(pos)          // last part of the prefix
                      << "\n\t\t>>> " << it->str() << " <<<\n"   // matched word
                      << it->suffix().str().substr(0, 40)        // first part of the suffix
                      << std::endl;
        }
        ```
- *子表达式* （Subexpressions）
    - 正则表达式中的 *模式* （pattern）通常包含一或多个 *子表达式* 
        - 一个子表达式是模式的一部分，本身也有意义
        - 正则表达式语法通常用 *括号* `()` 表示子表达式
        - 例如
        ```
        // r has two subexpressions: the first is the part of the file name before the period
        // the second is the file extension
        std::regex r("([[:alnum:]]+)\\.(cpp|cxx|cc)$", std::regex::icase);
        ```
        - 包含 *两个* 子表达式
            1. `([[:alnum:]]+)`匹配一或多个字符
            2. `(cpp|cxx|cc)`匹配`cpp`、`cxx`或`cc`
        - 可以重写之前的扩展名匹配程序`例17.2`，使之只输出文件名
            - 例如`foo.cpp`的`results.str(0)`为`foo.cpp`、`results.str(1)`为`foo`、`results.str(2)`为`cpp`
        ```
        // one or more alphanumeric characters followed by a '.' followed by "cpp" or "cxx" or "cc"
        std::regex r("[[:alnum:]]+\\.(cpp|cxx|cc)$", std::regex::icase);
        std::smatch results;
        std::string filename;
        
        while (std::cin >> filename)
        {
            if (std::regex_search(filename, results, r))
            {
                std::cout << results.str(1) << std::endl;  // print only the 1st subexpression
            }
        }  
        ```
    - 子表达式用于数据验证
        - `ECMAScript`正则表达式语言的一些特性
            - `\{d}`表示 *单个数字* ，`\{d}{n}`表示 *`n`个数字的序列* 
                - 例如，`\{d}{3}`匹配三个数字的序列
            - *方括号`[]`中的字符集合* 匹配 *这些字符中的任意一个*
                - 例如，`[-. ]`匹配一个短横线`'-'`、一个点`'.'`或一个空格`' '`
                - 注意，点`.`在方括号中**没有**特殊含义
            - *后接`?`* 的组件是 *可选* 的
                - 例如，`\{d}{3}[-. ]?\{d}{4}`匹配三个数字加可选的短横线或点或空格加四个数字
                - 可以匹配`555-0132`或`555.0132`或`555 0132`或`5550132`
        - `C++`中的`ECMAScript`字面量中，表示转义`ECMAScript`的 *反斜线* `\`应写为`\\`
            - `C++`和`ECMAScript`都使用`\`表示转义
            - 所以如果想要匹配字面括号，就需要转义成`\\(`或`\\)`，否则会被认为是子表达式的边界
                - `\\`是因为`\`在`C++`字符串中也是转义的，因此第一个`\`表示转义第二个`\`，由被转义的第二个`\`去转义`(`和`)`
            - 类似地，`\{d}{3}[-. ]?\{d}{4}`在`C++`编程时也应写成`\\{d}{3}[-. ]?\\{d}{4}`
            - 使用 *原始字符串字面量* （raw string literal）`R"(str)"`则可以避免两个`\\`这种难看的东西
        - `例17.4`：匹配美国电话号码
            - 匹配模式解析
                - 整体模式
                ```
                // our overall expression has seven subexpressions: ( ddd ) separator ddd separator dddd
                // subexpressions 1, 3, 4, and 6 are optional; 2, 5, and 7 hold the number
                std::string p1 {"(\\()?(\\d{3})(\\))?([-. ])?(\\d{3})([-. ]?)(\\d{4})"};
                
                // use of raw string literals can avoid `\\`
                std::string p2 {R"((\()?(\d{3})(\))?([-. ])?(\d{3})([-. ]?)(\d{4}))"};
                
                std::cout << std::boolalpha << (p1 == p2) << std::endl;  // true
                ```
                - 子表达式
                1. `(\\()?`：区号部分可选的左括号
                2. `(\\d{3})`：区号
                3. `(\\))?`：区号部分可选的左括号
                4. `([-. ])?`：区号后面可选的分隔符
                5. `(\\d{3})`：号码的下三位数字
                6. `([-. ])?`：可选的分隔符
                7. `(\\d{4})`：号码的最后四位数字
            - 初版程序
            ```
            const std::string phone = R"((\()?(\d{3})(\))?([-. ])?(\d{3})([-. ]?)(\d{4}))";
            std::regex r(phone); // a regex to find our pattern
            std::smatch m;

            // read each record from the input file
            for (std::string s; std::getline(std::cin, s);)
            {
                // for each matching phone number
                for (std::sregex_iterator it(s.begin(), s.end(), r), end_it; it != end_it; ++it)
                {
                    // check whether the number's formatting is valid
                    if (valid(*it))
                    {
                        std::cout << "valid: " << it->str() << std::endl;
                    }
                    else
                    {
                        std::cout << "not valid: " << it->str() << std::endl;
                    }
                }
            }
            ```
    - 使用 *子匹配操作* （Submatch Operations）
        - 子匹配操作，适用于`std::ssub_match`、`std::csub_match`、`std::wssub_match`以及`std::wcsub_match`对象
            - `m.matched`：一个`public bool`数据成员，指出此`ssub_match`是否匹配了
            - `m.first`，`m.second`：`public`数据成员，指向匹配序列首元素和尾后位置的迭代器。如果
            - `m.length()`：匹配的大小。如果`matched`部分为`false`，则返回`0`
            - `m.str()`：返回一个包含输入中匹配部分的`std::string`，如果`matched`为`false`，则返回空`std::string`
            - `str = ssub`：将`std::ssub_match`对象转化为`std::string`，等价于`str = ssub.str()`。`std::ssub_match`向`std::string`的 *类型转换运算符* **不是**`explicit`的
        - 可以使用子匹配操作来编写`例17.4`中的`valid`函数
            - `pattern`有 *七个* 子表达式，从而匹配结果`std::ssmatch m`会有一共 *八个* `std::ssub_match ssub`子匹配对象
                - 其中`m[0]`表示 *完整匹配* 
                - `m[1]`至`m[7]`分别对应七个子表达式的匹配结果
            - 调用我们即将编写的这个`valid`函数时，我们已经知道有一个完整匹配，但不知道每个可选的子表达式是否是完整匹配的一部分
                - 如果一个子表达式是完整匹配的一部分，则其对应的`std::ssub_match`对象的`matched`成员为`true`
            - `valid`函数实现
            ```
            bool valid(const std::smatch & m)
            {
                
                if (m[1].matched)
                {
                    // if there is an open parenthesis before the area code
                    // the area code must be followed by a close parenthesis
                    // and followed immediately by the rest of the number or a space
                    return m[3].matched && (!m[4].matched || m[4].str() == " ");
                }
                else
                {
                    // then there can't be a close after the area code
                    // the delimiters between the other two components must match
                    return !m[3].matched && m[4].str() == m[6].str();
                }
            }
            ```
- [正则表达式 *替换*](http://www.cplusplus.com/reference/regex/regex_replace/)
    - 正则表达式 *替换* 操作，适用于`std::smatch`、`std::cmatch`、`std::wsmatch`、`std::wcmatch`以及对应的`std::ssub_match`、`std::csub_match`、`std::wssub_match`、`std::wcsub_match`
        - `m.format(dest, fmt, mft)`：使用 *格式字符串* `fmt`、`m`中的匹配，以及 *可选* 的 *匹配标志* `mft`生成格式化输出，写入迭代器`dest`指向的目的位置。`fmt`可以是`std::string`，也可以是表示字符数组范围的 *一对指针* 。`mft`默认参数为`std::regex_constants::match_default`
        - `m.format(fmt, mft)`：返回一个`std::string`，其余与前者相同
        - `std::regex_replace(dest, b, e, r, fmt, mft)`：遍历迭代器`[b, e)`表示的范围，用`std::regex_match`寻找与`std::regex r`匹配的子串。使用 *格式字符串* `fmt`，以及 *可选* 的 *匹配标志* `mft`生成格式化输出，写入迭代器`dest`指向的位置。`fmt`可以是`std::string`，也可以是 *`C`风格字符串* 。`mft`默认参数为`std::regex_constants::match_default`
        - `std::regex_replace(seq, r, fmt, mft)`：遍历`seq`，用`std::regex_match`寻找与`std::regex r`匹配的子串。使用 *格式字符串* `fmt`，以及 *可选* 的 *匹配标志* `mft`生成格式化输出并作为`std::string`返回。`seq`可以是`std::string`或 *`C`风格字符串* 。`fmt`可以是`std::string`，也可以是 *`C`风格字符串* 。`mft`默认参数为`std::regex_constants::match_default`
    - `fmt`是 *格式化字符串* ，具体可以含有 
        - `$n`：第`n`个 *反向引用* ，即第`n`个匹配到的`std::ssub_match`对象。`n`必须是 *非负* 的，且 *最多有两位数* 
        - `$&`：整个`std::smatch`
        - `$^`：`std::match`的前缀`prefix()`
        - `$'`：`std::match`的后缀`suffix()`
        - `$$`：字面`$`
        - 其他普通字符
    - `mft`是 *匹配标志* ，具体是`std::regex_constants::match_flag_type`类型的 *`unsigned int`枚举* 值
        - `std::regex_constants::match_default`：等价于`std::regex_constants::format_default`， *默认参数*
        - `std::regex_constants::match_not_bol`：不将首字符作为行首处理
        - `std::regex_constants::match_not_eol`：不将尾字符作为行尾处理
        - `std::regex_constants::match_not_bow`：不将首字符作为词首处理
        - `std::regex_constants::match_not_eow`：不将首字符作为词尾处理
        - `std::regex_constants::match_any`：如果存在多个匹配，则可返回任意一个匹配
        - `std::regex_constants::match_not_null`：不匹配任何空序列
        - `std::regex_constants::match_continuous`：匹配必须从输入的首字符开始
        - `std::regex_constants::match_prev_avail`：输入序列包含第一个匹配之前的内容
        - `std::regex_constants::format_default`：用`ECMAScript`规则替换字符串。 *默认* 
        - `std::regex_constants::format_sed`：用`POSIX sed`规则替换字符串
        - `std::regex_constants::format_no_copy`：不输出输入序列中未匹配的部分
        - `std::regex_constants::format_first_only`：只替换子表达式的第一次出现
    - 使用示例
    ```
    std::string s {"there is a subsequence in the string\n"};
    std::regex e {R"(\b(sub)([^ ]*))"};               // matches words beginning by "sub"

    std::cout << std::regex_replace(s, e, "sub-$2");  // there is a sub-sequence in the string

    std::string result;
    std::regex_replace(std::back_inserter(result), s.begin(), s.end(), e, "$2");
    std::cout << result;                              // there is a sequence in the string

    // with flags:
    std::cout << std::regex_replace(s, e, "$1 and $2", std::regex_constants::format_no_copy);
    std::cout << std::endl;                           // sub and sequence
    ```
- [`std::regex_token_iterator`](http://www.cplusplus.com/reference/regex/regex_token_iterator/)
    - 只读`LegacyForwardIterator`，用于遍历给定字符串中、给定正则表达式的 *每一次匹配* 的子匹配
    - 四种输入的对应版本
        - `std::sregex_token_iterator`：`std::regex_token_iterator<std::string::const_iterator>`
        - `std::cregex_token_iterator`：`std::regex_token_iterator<const char *>`
        - `std::wcregex_token_iterator`：`std::regex_token_iterator<const wchar_t *>`
        - `std::wsregex_token_iterator`：`std::regex_token_iterator<std::wstring::const_iterator>`
    - 支持的操作
        - [`std::sregex_token_iterator srt_it (b, e, r, submatch, mft)`](http://www.cplusplus.com/reference/regex/regex_token_iterator/regex_token_iterator/)：就像调用`std::regex_search(b, e, r, mft)`一样进行匹配。如成功，则保留`std::smatch`的结果，迭代器指向这次匹配结果的第`submatch`个`std::ssub_match`对象。如不成功，则初始化为尾后迭代器。`submatch`可以是`int`、 *数组* 、`std::vector<int>`或`std::initializer-list<int>`
            - `int`：指明在迭代器的每个位置要选择的`std::ssub_match`。如果是`0`，选择整个匹配；如果是`-1`，则使用`match`作为分隔符，选择未被匹配到的序列。 *默认值* 为`0`
            - 其余：指定数个`std::ssub_match`。注意，此时迭代器需要的多递增相应的次数，以到达下一次匹配的位置
            - *警告* ：编程者必须确保 *`r`生存期* 比迭代器长。特别，**不能**传入临时量
        - `std::sregex_token_iterator srt_it_end`：默认初始化，创建尾后迭代器
        - `*it`：根据上一次调用`std::regex_search`的结果，返回一个`sts::ssub_match`对象的 *引用* 
        - `it->`：根据上一次调用`std::regex_search`的结果，返回一个`sts::ssub_match`对象的 *指针* 
        - `++it`，`it++`：从输入序列当前匹配位置开始调用`std::regex_search`，前置版本返回递增后的迭代器；后置版本返回旧值
        - `it1 == it2`，`it1 != it2`：两个`std::sregex_token_iterator`在如下情况下相等
            1. 都是尾后迭代器
            2. 都指向同一个序列的同一处匹配（这句话是错的，先这么写着，具体看文档去吧）
    - 使用示例`1`
    ```
    // Tokenization (non-matched fragments)
    // Note that regex is matched only two times; 
    // when the third value is obtainedn the iterator is a suffix iterator.
    const std::string text = "Quick brown fox.";
    const std::regex ws_re(R"(\s+)");             // whitespace
    std::copy(std::sregex_token_iterator(text.begin(), text.end(), ws_re, -1),
              std::sregex_token_iterator(),
              std::ostream_iterator<std::string>(std::cout, "\n"));
    std::cout << std::endl;
 
    // Iterating the first submatches
    const std::string html = R"(<p><a href="http://google.com">google</a> )"
                             R"(< a HREF ="http://cppreference.com">cppreference</a>\n</p>)";
    const std::regex url_re(R"!!(<\s*A\s+[^>]*href\s*=\s*"([^"]*)")!!", std::regex::icase);
    std::copy(std::sregex_token_iterator(html.begin(), html.end(), url_re, 1),
              std::sregex_token_iterator(),
              std::ostream_iterator<std::string>(std::cout, "\n"));
    std::cout << std::endl;
    
    // OUTPUT: 
    Quick
    brown
    fox.
    http://google.com
    http://cppreference.com
    ```
    - 使用示例`2`
    ```
    std::string s {"this subject has a submarine as a subsequence"};
    std::regex r {R"(\b(sub)([^ ]*))"};   // matches words beginning by "sub"

    // default constructor = end-of-sequence:
    std::regex_token_iterator<std::string::iterator> rend;

    std::cout << "entire matches:"; 
    std::regex_token_iterator<std::string::iterator> a {s.begin(), s.end(), r};
    while (a != rend) std::cout << " [" << *a++ << "]";
    std::cout << std::endl;  // entire amtches: [subject] [submarine] [subsequence]

    std::cout << "2nd submatches:";
    std::regex_token_iterator<std::string::iterator> b {s.begin(), s.end(), r, 2};
    while (b != rend) std::cout << " [" << *b++ << "]";
    std::cout << std::endl;  // 2nd submatches: [ject] [marine] [sequence]

    std::cout << "1st and 2nd submatches:";
    int submatches[] {1, 2};
    std::regex_token_iterator<std::string::iterator> c {s.begin(), s.end(), r, submatches};
    while (c != rend) std::cout << " [" << *c++ << "]";
    std::cout << std::endl;  // 1st and 2nd submatches: [sub] [ject] [sub] [marine] [sub] [sequence]

    std::cout << "1st and 2nd submatches";
    std::regex_token_iterator<std::string::iterator> d {s.begin(), s.end(), r, {1, 2}};
    while (d != rend) std::cout << " [" << *d++ << "]";
    std::cout << std::endl;  // 1st and 2nd submatches: [sub] [ject] [sub] [marine] [sub] [sequence]
    
    std::cout << "matches as splitters:";
    std::regex_token_iterator<std::string::iterator> e {s.begin(), s.end(), r, -1};
    while (e != rend) std::cout << " [" << *e++ << "]";
    std::cout << std::endl;  // matches as splitters: [this ] [ has a ] [ as a ]
    ```
    - 最好懂的一个
    ```
    std::string line {"as 1df 1gh"};
    std::regex r {R"(( )(1))"};

    for (std::sregex_iterator it {line.begin(), line.end(), r}, end; it != end; ++it)
    {
        std::smatch m {*it};
        std::cout << "[ " << m.prefix() << " > " << m.str() << " < " << m.suffix() << " ]" << std::endl;
    }
    
    // OUTPUT:
    [ as >  1 < df 1gh ]
    [ df >  1 < gh ]

    for (std::sregex_token_iterator it {line.begin(), line.end(), r, {1, 2}}, end; it != end; ++it)
    {
        std::ssub_match m {*it};
        std::cout << "[ " <<  m << " ]" << std::endl;
    }
    
    // OUTPUT:
    [   ]
    [ 1 ]
    [   ]
    [ 1 ]
    
    for (std::sregex_token_iterator it {line.begin(), line.end(), r, 0}, end; it != end; ++it)
    {
        std::ssub_match m {*it};
        std::cout << "[ " <<  m << " ]" << std::endl;
    }
    
    // OUTPUT:
    [  1 ]
    [  1 ]

    for (std::sregex_token_iterator it {line.begin(), line.end(), r, -1}, end; it != end; ++it)
    {
        std::ssub_match m {*it};
        std::cout << "[ " <<  m << " ]" << std::endl;
    }
    
    // OUTPUT:
    [ as ]
    [ df ]
    [ gh ]
    ```

#### [随机数](https://en.cppreference.com/w/cpp/numeric/random)

- `C++`随机数标准库`<random>`中定义了
    - *随机数引擎* （random-number engine）
        - 生成随机的 *无符号整数* 序列
    - *随机数分布类* （random-number distribution）
        - 使用引擎返回服从特定概率分布的随机数
- `C++`程序**不应**使用`C`库函数`rand`，而应使用`std::default_random_engine`和恰当的分布类对象
- *随机数引擎* 
    - 随机数引擎是函数对象类，定义了一个调用运算符，不接收参数，返回一个随机的 *无符号整数* 
    ```
    std::default_random_engine e;  // generates random unsigned integers
    
    for (size_t i = 0; i != 10; ++i)
    {
        // e() "calls" the object to produce the next random number
        std::cout << e() << " ";
    }
    ```
    - 标准库定义了很多随机数引擎，区别在于性能和随机性质量不同
        - 每个编译器都会指定一个`std::default_random_engine`类型
        - 此类型一般具有最常用的特性
    - 随机数引擎操作
        - `std::default_random_engine e;`：默认构造函数，使用该引擎类型的默认种子
        - `std::default_random_engine e(s);`：使用整型值`s`作为种子
        - `e.seed(s);`：使用种子`s`重置引擎状态
        - `e()`：返回一个随机数
        - `e.min()`：此引擎可生成的最小值
        - `e.max()`：此引擎可生成的最大值
        - `std::default_random_engine::result_type`：此引擎生成的随机数的类型
        - `e.discard(u)`：将引擎推进`u`步；`u`为`unsigned long long`
    - *分布类型* 和引擎
        - 为了得到在一个指定范围内的数，我们使用一个分布类型对象
        ```
        // uniformly distributed unsigned int from [0, 9]
        std::uniform_int_distribution<unsigned> u(0, 9);
        
        // generates unsigned random integers
        std::default_random_engine e;
        
        for (size_t i = 0; i < 10; ++i)
        {
            // u uses e as a source of numbers
            // each call returns a uniformly distributed value in the specified range
            std::cout << u(e) << " ";
        }
        ```
        - 我们说 *随机数发生器* 一词时，是指 *分布对象* 和 *引擎对象* 的组合
        - 每个新引擎生成的序列都是一样的，因此要么定义成 *全局* 的，要么定义为函数的 *局部静态* 变量
        ```
        std::vector<unsigned> good_randVec()
        {
            static std::default_random_engine e;
            static std::uniform_int_distribution<unsigned> u(0, 9);
            
            std::vector<unsigned> ret;
            
            for (size_t i = 0; i < 100; ++i)
            {
                ret.push_back(u(e));
            }
                
            return std::move(ret);
        }
        ```
        - 设置 *种子* （seed）
            - 可以创建时设置，也可以随后设置
            - 可以设置为随机的`time(NULL)`，返回当前时间（到秒为止）
                - 如果程序是作为一个自动过程反复运行，将`time`的返回值作为种子的方式就无效了；它可能多次使用的都是相同的种子
        ```
        std::default_random_engine e1;              // uses the default seed
        std::default_random_engine e2(2147483646);  // use the given seed value
        
        // e3 and e4 will generate the same sequence because they use the same seed
        std::default_random_engine e3;              // uses the default seed value
        e3.seed(32767);                             // call seed to set a new seed value
        std::default_random_engine e4(32767);       // set the seed value to 32767

        for (size_t i = 0; i != 100; ++i) 
        {
            if (e1() == e2())
            {
                std::cout << "unseeded match at iteration: " << i << std::endl;
            }
            
            if (e3() != e4())
            {
                std::cout << "seeded differs at iteration: " << i << std::endl;
            }   
        }
        ```
- 其他随机数分布
    - 分布类型的操作
        - `Dist d;`：默认构造函数，使`d`准备好被使用。其他构造函数依赖于`Dist`类型。`Dist`类型的构造函数都是`explicit`的
        - `d(e)`：用相同的随机数引擎对象`e`连续调用`d`的话，会根据`d`的分布式类型生成一个随机序列
        - `d.min()`：返回`d(e)`能生成的最小值
        - `d.max()`：返回`d(e)`能生成的最大值
        - `d.reset()`：重建`d`的状态，使得随后对`d`的使用不依赖于`d`已经生成的值
    - 可用的 *随机数分布类* 
        - 均匀分布
            - [`std::uniform_int_distribution`](https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution)：产生在一个范围上均匀分布的整数值
                - `std::uniform_int_distribution<IntT> u(m, n);`
                - 生成指定类型的在给定包含范围之内的值
                - `m`是可以返回的最小值，默认为`0`
                - `n`为可以返回的最大值，默认为`IntT`类型对象可以表示的最大值
            - [`std::uniform_real_distribution`](https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution)：产生在一个范围上均匀分布的实数值
                - `std::uniform_real_distribution<RealT> u(x, y);`
                - 生成指定类型的在给定包含范围之内的值
                - `x`是可以返回的最小值，默认为`0`
                - `y`为可以返回的最大值，默认为`RealT`类型对象可以表示的最大值
        - 伯努利分布
            - [`std::bernoulli_distribution`](https://en.cppreference.com/w/cpp/numeric/random/bernoulli_distribution)：产生伯努利分布上的布尔值
                - `std::bernoulli_distribution b(p);`
                - 以概率`p`生成`true`
                - `p`默认为`0.5`
            - [`std::binomial_distribution`](https://en.cppreference.com/w/cpp/numeric/random/binomial_distribution)：产生二项分布上的整数值
                - `std::binomial_distribution<IntT> b(t, p);`
                - 以概率`p`采样`t`次，成功次数的二项分布
                - `t`为整数，默认为`1`
                - `p`默认为`0.5`
            - [`std::negative_binomial_distribution`](https://en.cppreference.com/w/cpp/numeric/random/negative_binomial_distribution)：产生负二项分布上的整数值
                - `std::negative_binomial_distribution<IntT> nb(k, p);`
                - 以概率`p`采样直至第`k`次成功时，所经历的失败次数的负二项分布
                - `k`为整数，默认为`1`
                - `p`默认为`0.5`
            - [`std::geometric_distribution`](https://en.cppreference.com/w/cpp/numeric/random/geometric_distribution)：产生几何分布上的整数值
                - `std::geometric_distribution<IntT> g(p);`
                - 以概率`p`采样直至第一次成功时，所经历的失败次数的几何分布
                - `p`默认为`0.5`
        - 泊松分布
            - [`std::poisson_distribution`](https://en.cppreference.com/w/cpp/numeric/random/poisson_distribution)：产生泊松分布上的整数值
                - `std::poisson_distribution<IntT> p(x);`
                - 均值为`double`值`x`的分布
            - [`std::exponential_distribution`](https://en.cppreference.com/w/cpp/numeric/random/exponential_distribution)：产生指数分布上的实数值
                - `std::exponential_distribution<RealT> e(lam);`
                - 指数分布，参数`lambda`通过浮点值`lam`给出
                - `lam`默认值为`1.0`
            - [`std::gamma_distribution`](https://en.cppreference.com/w/cpp/numeric/random/gamma_distribution)：产生`Γ`分布上的实数值
                - `std::gamma_distribution<RealT> g(a, b);`
                - 形状参数`alpha`为`a`，默认值`1.0`
                - 尺度参数`beta`为`b`，默认值`1.0`
            - [`std::weibull_distribution`](https://en.cppreference.com/w/cpp/numeric/random/weibull_distribution)：产生威布尔分布上的实数值
                - `std::weibull_distribution<RealT> w(a, b);`
                - 形状参数`alpha`为`a`，默认值`1.0`
                - 尺度参数`beta`为`b`，默认值`1.0`
            - [`std::extreme_value_distribution`](https://en.cppreference.com/w/cpp/numeric/random/extreme_value_distribution)：产生极值分布上的实数值
                - `std::extreme_value_distribution<RealT> e(a, b);`
                - `a`默认值为`1.0`
                - `b`默认值为`1.0`
        - 正态分布
            - [`std::normal_distribution`](https://en.cppreference.com/w/cpp/numeric/random/normal_distribution)：产生标准正态分布上的实数
                - `std::normal_distribution<RealT> n(m, s);`
                - 均值`m`、标准差`s`的正态分布
                - `m`默认值为`0.0`
                - `s`默认值为`1.0`
            - [`std::lognormal_distribution`](https://en.cppreference.com/w/cpp/numeric/random/lognormal_distribution)：产生对数正态分布上的实数值
                - `std::lognormal_distribution<RealT> ln(m, s);`
                - 均值`m`、标准差`s`的对数正态分布
                - `m`默认值为`0.0`
                - `s`默认值为`1.0`
            - [`std::chi_squared_distribution`](https://en.cppreference.com/w/cpp/numeric/random/chi_squared_distribution)：产生`χ2`分布上的实数值
                - `std::chi_squared_distribution<RealT> c(x);`
                - 自由度为`x`的`χ2`分布
                - `x`默认值为`0.0`
            - [`std::cauchy_distribution`](https://en.cppreference.com/w/cpp/numeric/random/cauchy_distribution)：产生柯西分布上的实数值
                - `std::cauchy_distribution<RealT> c(a, b);`
                - 位置参数为`a`，默认值`0.0`
                - 尺度参数为`b`，默认值`1.0`
            - [`std::fisher_f_distribution`](https://en.cppreference.com/w/cpp/numeric/random/fisher_f_distribution)：产生费舍尔`F`分布上的实数值
                - `std::fisher_f_distribution<RealT> f(m, n);`
                - 自由度为`m`和`n`的费舍尔`F`分布
                - `m`和`n`默认值均为`1`
            - [`std::student_t_distribution`](https://en.cppreference.com/w/cpp/numeric/random/student_t_distribution)：产生学生`t`分布上的实数值
                - `std::student_t_distribution<RealT> s(n);`
                - 自由度为和`n`的学生`t`分布
                - `n`默认值为`1`
        - 采样分布
            - [`std::discrete_distribution`](https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution)：产生离散分布上的随机整数
                - 构造函数
                ```
                std::discrete_distribution<IntT> d(i, j);
                std::discrete_distribution<IntT> d{il};
                ```
                - `i`和`j`是一个权重序列的输入迭代器
                - `il`是一个权重的初始化列表
                - 权重必须能转换为`double`
            - [`std::piecewise_constant_distribution`](https://en.cppreference.com/w/cpp/numeric/random/piecewise_constant_distribution)：产生分布在常子区间上的实数值
                - `std::piecewise_constant_distribution<RealT> pc(b, e, w);`
                - `b`，`e`和`w`是输入迭代器
            - [`std::piecewise_linear_distribution`](https://en.cppreference.com/w/cpp/numeric/random/piecewise_linear_distribution)：产生分布在定义的子区间上的实数值
                - `std::piecewise_linear_distribution<RealT> pl(b, e, w);`
                - `b`，`e`和`w`是输入迭代器
    - 生成随机实数
    ```
    // generates unsigned random integers
    std::default_random_engine e; 
    
    // uniformly distributed from 0 to 1 inclusive
    std::uniform_real_distribution<double> u(0, 1);
    
    for (size_t i = 0; i != 10; ++i)
    {
        std::cout << u(e) << " ";
    }
    ```
    - 使用分布的默认结果类型
        - 每个分布模板都有一个默认实参
            - 生成浮点值的分布类型默认`double`
            - 生成整数类型的分布类型默认`int`
        - 希望使用模板默认实参时，跟空的尖括号
    ```
    // empty <> signify we want to use the default result type
    std::uniform_real_distribution<> u(0,1); // generates double by default
    ```
    - 生成非均匀分布的随机数
        - 例如正态分布
        ```
        std::default_random_engine e;  // generates random integers
        std::normal_distribution<> n(4, 1.5);  // mean 4, standard deviation 1.5
        std::vector<unsigned> vals(9);  // nine elements each 0

        for (size_t i = 0; i != 200; ++i)
        {
            unsigned v = lround(n(e));  // round to the nearest integer

            if (v < vals.size())
            {  // if this result is in range
                ++vals[v];
            }  // count how often each number appears
        }

        for (size_t j = 0; j != vals.size(); ++j)
        {
            std::cout << j << ": " << std::string(vals[j], '*') << std::endl;
        }
        ```
        - 实测输出
        ```
        0: ***
        1: ********
        2: ********************
        3: **************************************
        4: **********************************************************
        5: ******************************************
        6: ***********************
        7: *******
        8: *
        ```
- `bernoulli_distribution`类

#### `I/O`库再探

- 放`Chap 8`了


#### [用户自定义字面量](https://en.cppreference.com/w/cpp/language/user_literal)（User Literal）

```c++
#include <iostream>


struct Price
{
    friend std::ostream & operator <<(std::ostream &, const Price &);
    long double val;
};


std::ostream & operator <<(std::ostream & cout, const Price & price)
{
    cout << '$' << price.val;
    return cout;
}


constexpr Price operator "" _USD(long double p)
{
    return Price {p};
}


int main(int argc, char * argv[])
{
    std::cout << 12.5132_USD << '\n';  // $12.5132
    
    return EXIT_SUCCESS;
}
```

#### [日期时间库](https://en.cppreference.com/w/cpp/chrono)（Date and time utilities）

- `C++`日期时间库[`<chrono>`](https://en.cppreference.com/w/cpp/header/chrono)定义了三个核心类
    - [`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration)：表示一段 *时间间隔* 
    - [`std::chrono::time_point`](https://en.cppreference.com/w/cpp/chrono/time_point)：表示一个 *时刻* 
    - *时钟* ：用于计时，具体有如下 *三种* 
        1. [`std::chrono::system_clock`](https://en.cppreference.com/w/cpp/chrono/system_clock)
        2. [`std::chrono::steady_clock`](https://en.cppreference.com/w/cpp/chrono/steady_clock)
        3. [`std::chrono::high_resolution_clock`](https://en.cppreference.com/w/cpp/chrono/high_resolution_clock)
- [`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration)
    - 定义
    ```
    template <class Rep, class Period = std::ratio<1>> 
    class duration;
    ```
    - `Rep`
        - 表示`period`的数量，是一个 *数值类型* ，例如`int`，`float`，`double`
    - `Period`
        - 在这里用来表示此`std::duration`的时长单位（period），是一个[`std::ratio`](https://en.cppreference.com/w/cpp/numeric/ratio/ratio)类型，单位为秒
        - `std::ratio`是一个模板类，代表一个 *分数值* `Num / Denom`
        ```
        template <std::intmax_t Num, std::intmax_t Denom = 1> 
        class ratio;
        ```
        - 预定义好的`std::ratio`类型
        ```
        // <ratio>
        // namespace std
        
        typedef ratio<1, 1000000000000000000> atto;
        typedef ratio<1,    1000000000000000> femto;
        typedef ratio<1,       1000000000000> pico;
        typedef ratio<1,          1000000000> nano;
        typedef ratio<1,             1000000> micro;
        typedef ratio<1,                1000> milli;
        typedef ratio<1,                 100> centi;
        typedef ratio<1,                  10> deci;
        typedef ratio<                 10, 1> deca;
        typedef ratio<                100, 1> hecto;
        typedef ratio<               1000, 1> kilo;
        typedef ratio<            1000000, 1> mega;
        typedef ratio<         1000000000, 1> giga;
        typedef ratio<      1000000000000, 1> tera;
        typedef ratio<   1000000000000000, 1> peta;
        typedef ratio<1000000000000000000, 1> exa;
        ```
    - 预定义好的`duration`类型及其`gcc`实现
        - `std::chrono::nanoseconds`：`std::chrono::duration<int64_t, std::nano>`
        - `std::chrono::microseconds`：`std::chrono::duration<int64_t, std::micro>`
        - `std::chrono::milliseconds`：`std::chrono::duration<int64_t, std::milli>`
        - `std::chrono::seconds`：`std::chrono::duration<int64_t>`
        - `std::chrono::minutes`：`std::chrono::duration<int64_t, std::ratio<60>>`
        - `std::chrono::hours`：`std::chrono::duration<int64_t, std::ratio<3600>>`
        - `std::chrono::days`：`std::chrono::duration<int64_t, std::ratio<86400>>` `(since C++20)`
        - `std::chrono::weeks`：`std::chrono::duration<int64_t, std::ratio<604800>>` `(since C++20)`
        - `std::chrono::months`：`std::chrono::duration<int64_t, std::ratio<2629746>>` `(since C++20)`
        - `std::chrono::years`：`std::chrono::duration<int64_t, std::ratio<31556952>>` `(since C++20)`
    - `std::chrono::duration`字面量 `(since C++14)`
        - [`std::literals::chrono_literals::operator""h`](https://en.cppreference.com/w/cpp/chrono/operator%22%22h)
            - `gcc`实现
            ```
            template<typename _Dur, char ... _Digits>
            constexpr _Dur __check_overflow()
            {
                using _Val = __parse_int::_Parse_int<_Digits ...>;
                constexpr typename _Dur::rep __repval = _Val::value;
                static_assert(__repval >= 0 && __repval == _Val::value,
                              "literal value cannot be represented by duration type");
                return _Dur(__repval);
            }
            
            template <char ... _Digits>
            constexpr chrono::hours operator""h()
            { 
                return __check_overflow<chrono::hours, _Digits ...>(); 
            }
            
            constexpr std::chrono::duration<long double, ratio<3600, 1>> operator""h(long double h)
            {
                return std::chrono::duration<long double, std::ratio<3600, 1>>(h);
            }
            ```
            - 示例
            ```
            using namespace std::chrono_literals;
            auto day = 24h;
            auto halfhour = 0.5h;
            std::cout << "one day is " << day.count() << " hours\n"             // one day is 24 hours
                      << "half an hour is " << halfhour.count() << " hours\n";  // half an hour is 0.5 hours
            ```
            ```
            std::chrono::hours day = std::chrono_literals::operator""h<'2', '4'>();
            std::chrono::duration<long double, std::ratio<3600, 1>> halfhour = std::chrono_literals::operator""h(0.5);
            std::cout << "one day is " << day.count() << " hours\n"             // one day is 24 hours
                      << "half an hour is " << halfhour.count() << " hours\n";  // half an hour is 0.5 hours
            ```
        - [`std::literals::chrono_literals::operator""min`](https://en.cppreference.com/w/cpp/chrono/operator%22%22min)
            - `gcc`实现
            ```
            template <char ... _Digits>
            constexpr chrono::minutes operator""min()
            { 
                return __check_overflow<chrono::minutes, _Digits ...>(); 
            }
            
            constexpr std::chrono::duration<long double, std::ratio<60, 1>> operator""min(long double m)
            {
                return std::chrono::duration<long double, ratio<60, 1>> (m);
            }
            ```
            - 示例
            ```
            using namespace std::chrono_literals;
            auto lesson = 45min;
            auto halfmin = 0.5min;
            
            // one lesson is 45 minutes
            std::cout << "one lesson is " << lesson.count() << " minutes\n";       
            
            // half a minute is 0.5 minutes
            std::cout << "half a minute is " << halfmin.count() << " minutes\n";  
            ```
        - [`std::literals::chrono_literals::operator""s`](https://en.cppreference.com/w/cpp/chrono/operator%22%22s)
            - `gcc`实现
            ```
            template <char ... _Digits>
            constexpr chrono::seconds operator""s()
            { 
                return __check_overflow<chrono::seconds, _Digits ...>(); 
            }
            
            constexpr std::chrono::duration<long double> operator""s(long double s)
            {
                return std::chrono::duration<long double>(s);
            }
            ```
            - 示例
            ```
            using namespace std::chrono_literals;
            auto halfmin = 30s;
            
            // half a minute is 30 seconds
            std::cout << "half a minute is " << halfmin.count() << " seconds\n";              
            
            // a minute and a second is 61 seconds
            std::cout<< "a minute and a second is " << (1min + 1s).count() << " seconds\n";
            ```
        - [`std::literals::chrono_literals::operator""ms`](https://en.cppreference.com/w/cpp/chrono/operator%22%22ms)
            - `gcc`实现
            ```
            template <char ... _Digits>
            constexpr chrono::milliseconds operator""ms()
            { 
                return __check_overflow<chrono::milliseconds, _Digits ...>(); 
            }
            
            constexpr std::chrono::duration<long double, std::milli> operator""ms(long double ms)
            {
                return std::chrono::duration<long double, std::milli>(ms);
            }
            ```
            - 示例
            ```
            using namespace std::chrono_literals;
            auto d1 = 250ms;
            std::chrono::milliseconds d2 = 1s;
            std::cout << "250ms = " << d1.count() << " milliseconds\n"  // 250ms = 250 milliseconds
                      << "1s = " << d2.count() << " milliseconds\n";    // 1s = 1000 milliseconds
            ```
        - [`std::literals::chrono_literals::operator""us`](https://en.cppreference.com/w/cpp/chrono/operator%22%22us)
            - `gcc`实现
            ```
            template <char ... _Digits>
            constexpr chrono::microseconds operator""us()
            { 
                return __check_overflow<chrono::microseconds, _Digits ...>(); 
            }
            
            constexpr std::chrono::duration<long double, std::micro> operator""us(long double us)
            {
                return std::chrono::duration<long double, std::micro>(us);
            }
            ```
            - 示例
            ```
            using namespace std::chrono_literals;
            auto d1 = 250us;
            std::chrono::microseconds d2 = 1ms;
            std::cout << "250us = " << d1.count() << " microseconds\n"  // 250us = 250 microseconds
                      << "1ms = " << d2.count() << " microseconds\n";   // 1ms = 1000 microseconds
            ```
        - [`std::literals::chrono_literals::operator""ns`](https://en.cppreference.com/w/cpp/chrono/operator%22%22ns)  
            - `gcc`实现
            ```
            template <char ... _Digits>
            constexpr chrono::nanoseconds operator""ns()
            { 
                return __check_overflow<chrono::nanoseconds, _Digits ...>(); 
            }
            
            constexpr std::chrono::duration<long double, std::nano> operator""ns(long double ns)
            {
                return std::chrono::duration<long double, std::nano>(ns);
            }
            ```
            - 示例
            ```
            using namespace std::chrono_literals;
            auto d1 = 250ns;
            std::chrono::nanoseconds d2 = 1us;                         // 250ns = 250 nanoseconds
            std::cout << "250ns = " << d1.count() << " nanoseconds\n"  // 1us = 1000 nanoseconds
            ```
    - 支持的操作
        - 一元操作
            - `std::chrono::duration<Rep, Period> t;`：默认构造
            - `std::chrono::duration<Rep, Period> t(r);`：创建时长为`r`个`Period`的`std::chrono::duration`。是`explicit`的
            - `std::chrono::duration::zero()`：返回一个零长度时间间隔
            - `std::chrono::duration::min()`：返回此时间间隔的最小值
            - `std::chrono::duration::max()`：返回此时间间隔的最大值
            - `t.count()`：返回其`Ref`的值
            - `t++`，`++t`
            - `t--`，`--t`
            - `std::chrono::duration_cast<Duration>(t)`：有精度损失的时间间隔转换不能自动执行，必须显式调用`std::chrono::duration_cast`
            ```
            void f()
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
             
            int main()
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                f();
                auto t2 = std::chrono::high_resolution_clock::now();
             
                // floating-point duration: no duration_cast needed
                std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
             
                // integral duration: requires duration_cast
                auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
             
                // converting integral duration to integral duration of shorter divisible time unit:
                // no duration_cast needed
                std::chrono::duration<long, std::micro> int_usec = int_ms;
             
                std::cout << "f() took " << fp_ms.count() << " ms, "
                          << "or " << int_ms.count() << " whole milliseconds "
                          << "(which is " << int_usec.count() << " whole microseconds)" << std::endl;
            }
            
            // OUTPUT:
            f() took 1000.23 ms, or 1000 whole milliseconds (which is 1000000 whole microseconds)
            ```
        - 二元操作
            - `t1 = t2;`
            - `t1 + t2`，`t1 - t2`，`t1 * t2`，`t1 / t2`，`t1 % t2`
            - `t1 += t2;`，`t1 -= t2;`，`t1 *= t2;`，`t1 /= t2;`，`t1 %= t2;`
            - `t1 == t2`，`t1 <=> t2 (since C++20)` 
- [`std::chrono::time_point`](https://en.cppreference.com/w/cpp/chrono/time_point)
    - 定义
    ```
    template <class Clock, class Duration = typename Clock::duration> 
    class time_point;
    ```
    - `std::chrono::time_point`表示一个时刻
        - 这个时刻具体到什么程度，由选用的单位决定
        - 一个`std::chrono::time_point`必须有一个 *时钟* 计时
            - 实际上应该说 *时刻* 是 *时钟* 的属性
            - 有意义的获取时刻对象的方式也是通过时钟的`time_point`类型成员
    - 支持的操作
        - 构造
            - 默认构造：构造时钟零时`epoch`
            - *显式* 构造：接受一个`std::chrono::duration`对象，表示此时刻距`epoch`的时间
            - 接收 *时钟* 返回的时刻
        ```
        std::chrono::time_point<std::chrono::high_resolution_clock> t0;                             // 0ms
        std::chrono::time_point<std::chrono::high_resolution_clock> t4  {std::chrono::seconds(4)};  // 4ms
        std::chrono::time_point<std::chrono::high_resolution_clock> now  \
                                                      {std::chrono::high_resolution_clock::now()};  // now
        std::chrono::high_resolution_clock::time_point              now2 \ 
                                                      {std::chrono::high_resolution_clock::now()};  // now
        ```
        - 一元操作
            - `t++`，`++t`
            - `t--`，`--t`
            - `t.time_since_epoch()`：返回距零时的`std::chrono::duration`
            - `std::chrono::time_point_cast<Duration>(t)`
            ```
            using Clock = std::chrono::high_resolution_clock;
            using Ms = std::chrono::milliseconds;
            using Sec = std::chrono::seconds;
             
            template <class Duration>
            using TimePoint = std::chrono::time_point<Clock, Duration>;
            
            TimePoint<Sec> time_point_sec(Sec(4));
         
            // implicit cast, no precision loss
            TimePoint<Ms> time_point_ms(time_point_sec);
            print_ms(time_point_ms);   // 4000 ms
         
            time_point_ms = TimePoint<Ms>(Ms(5756));
         
            // explicit cast, need when precision loss may happens
            // 5756 truncated to 5000
            time_point_sec = std::chrono::time_point_cast<Sec>(time_point_ms);
            print_ms(time_point_sec);  // 5000 ms
            ```
        - 二元操作
            - `t1 + t2`，`t1 - t2`
            - `t1 += t2;`，`t1 -= t2;`
            - `t1 == t2`，`t1 <=> t2 (since C++20)` 
- *时钟* 
    - 三种时钟
        1. [`std::chrono::system_clock`](https://en.cppreference.com/w/cpp/chrono/system_clock)
            - 系统时钟
            - 记录距协调世界时零时`epoch`（Thu Jan 1 1970 00:00:00 UTC±00:00）的时间间隔
            - 系统中运行的所有进程使用`now()`得到的时间是一致的
        2. [`std::chrono::steady_clock`](https://en.cppreference.com/w/cpp/chrono/steady_clock)
            - 稳定时钟
            - 表示稳定的时间间隔，后一次调用`now()`得到的时间总是比前一次的值大
                - 如果中途修改了系统时间，也不影响`now()`的结果
        3. [`std::chrono::high_resolution_clock`](https://en.cppreference.com/w/cpp/chrono/high_resolution_clock)
            - 系统可用的最高精度的时钟
            - 实际上只是`std::chrono::system_clock`或者`std::chrono::steady_clock`的`typedef`
    - 常用的操作
        - `std::chrono::high_resolution_clock::now()`：返回记录当前时刻的`std::chrono::time_point`
    - 类型成员
        - `std::chrono::high_resolution_clock::time_point`
        - `std::chrono::high_resolution_clock::duration`
- 整体使用示例
```
using CLK = std::chrono::high_resolution_clock;

CLK::time_point t0 {CLK::now()};
using namespace std::chrono_literals;
std::this_thread::sleep_for(1234.56ms);
CLK::time_point t1 {CLK::now()};
std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;  // 1235
```

#### [线程支持库](https://en.cppreference.com/w/cpp/thread)（Thread Support Library）

- *线程* ：使得程序能在数个处理器核心同时执行
    - 线程类，定义于`<thread>`
        - [`std::thread`](https://en.cppreference.com/w/cpp/thread/thread)
            - 综述
                - 表示一个线程，在与其关联的 *线程对象被构造之后立即开始执行* 
                    - 以被提供给其构造函数的顶层函作为入口
                    - 默认情况下顶层函数的返回值被忽略
                        - 顶层函数可以通过[`std::promise`](https://en.cppreference.com/w/cpp/thread/promise))，或修改 *共享变量* 将其返回值或异常传递给调用方
                    - 如果顶层函数以抛异常终止，则程序将调用[`std::terminate`](https://en.cppreference.com/w/cpp/error/terminate)
                - 在以下情况下，`std::thread`对象将处于**不**表示任何线程的状态，从而可安全销毁
                    - 被默认构造
                    - 已被移动
                    - 已被[`detach`](https://en.cppreference.com/w/cpp/thread/thread/detach)
                    - 已被[`join`](https://en.cppreference.com/w/cpp/thread/thread/join)
                - 没有两个`std::thread`会表示同一线程
                    - `std::thread`**不可**复制构造、**不可**复制赋值
                    - `std::thread`可移动构造、可移动赋值
                - 当`std::thread`对象销毁之前还没有被`join`或`detach`，程序就会异常终止
                    - `std::thread`的析构函数会调用`std::terminate()`
                    - 因此，即便是有异常存在，也需要确保线程能够正确`join`或`detach`
            - 构造和赋值
                - `std::thread t;`：默认构造**不**表示线程的新`std::thread`对象
                - `std::thread t(fun, ...)`：显式构造，传入 *可调用对象* （Callable）`fun`和其参数`...`
                    - 参数传递
                        - 方式和`std::bind`相同
                        ```
                        class X
                        {
                        public:
                            void do_lengthy_work(int);
                        };
                        X my_x;
                        int num(0);
                        std::thread t(&X::do_lengthy_work, &my_x, num); // 提供成员函数指针和对象指针，并传参
                        ```
                        - 提供的函数对象和参数会 **复制** 到新线程的存储空间中，函数对象的执行和调用都在线程的内存空间中进行
                        ```
                        int n = 0;
                        foo f;
                        baz b;
                        
                        std::thread t1;                   // t1不是线程
                        std::thread t2(f1, 233);          // 按值传递
                        std::thread t3(f2, std::ref(n));  // 按引用传递
                        std::thread t4(std::move(t3));    // t4现在运行f2() 。t3不再是线程
                        std::thread t5(&foo::bar, &f);    // t5在对象f上运行foo::bar()
                        std::thread t6(std::ref(b));      // t6在对象b上运行baz::operator()
                        ```
                        - 即使函数中的参数是引用的形式，拷贝操作也会执行
                        - **注意**：指向动态变量的指针作为参数的情况
                        ```
                        void f(int i, const std::string & s);
                        
                        void not_oops(int some_param)
                        {
                            char buffer[1024];
                            sprintf(buffer, "%i", some_param);
                            std::thread t(f, 3, std::string(buffer));  // 不显式构造，可能在构造执行之前oops函数就结束了，造成引用野指针
                                                                       // 使用std::string，避免悬空指针
                            t.detach();
                        }
                        ```
                    - C++'s most vexing parse：如何给`std::thread`构造函数传递 *无名临时变量* ？
                    ```
                    class background_task
                    {
                    public:
                        void operator()() const
                        {
                            do_something();
                            do_something_else();
                        }
                    };

                    background_task f;
                    std::thread my_thread(f);                    // OK：命名对象
                    
                    std::thread my_thread(background_task());    // 错误
                                                                 // 这里的background_task()会被解释为“参数列表为空、返回类型为background_task”的函数指针
                                                                 // 则my_thread变成了函数声明，而非对象定义！
                                                                 
                    std::thread my_thread((background_task()));  // OK：使用多组括号
                    
                    std::thread my_thread {background_task()};   // OK：花括号初始化列表
                    
                    std::thread my_thread([]
                    {
                        do_something();
                        do_something_else();
                    });                                          // OK：lambda表达式
                    ```
                - `std::thread t1(t2);`，`t1 = t2;`：移动构造和赋值
                    - 用例
                    ```
                    void some_function();
                    void some_other_function();
                    
                    std::thread t1(some_function);          // 1
                    std::thread t2 = std::move(t1);         // 2
                    t1 = std::thread(some_other_function);  // 3
                    std::thread t3;                         // 4
                    t3 = std::move(t2);                     // 5
                    t1 = std::move(t3);                     // 6 赋值操作将使程序崩溃
                    ```
                    - scoped_thread
                    ```
                    class scoped_thread
                    {
                    public:
                        explicit scoped_thread(std::thread t_) : t(std::move(t_))
                        {
                            if (!t.joinable())
                            {
                                throw std::logic_error(“No thread”);
                            }
                        }
                        
                        scoped_thread(const scoped_thread &) = delete;
                        
                        ~scoped_thread()
                        {
                            t.join();
                        }
                        
                        scoped_thread & operator=(const scoped_thread &) = delete;
                        
                    private:
                        std::thread t;
                    };

                    struct func;

                    void f()
                    {
                        int some_local_state;
                        scoped_thread t(std::thread(func(some_local_state)));
                        do_something_in_current_thread();
                    }
                    ```
                    - joining_thread
                    ```
                    class joining_thread
                    {
                    public:
                        joining_thread() noexcept = default;
                        
                        template<typename Callable, typename ... Args>
                        explicit joining_thread(Callable && func, Args && ... args) : t(std::forward<Callable>(func), std::forward<Args>(args) ...)
                        {
                        
                        }
                        
                        explicit joining_thread(std::thread t_) noexcept : t(std::move(t_))
                        {
                        
                        }
                        
                        joining_thread(joining_thread && other) noexcept : t(std::move(other.t))
                        {
                        
                        }
                        
                        ~joining_thread() noexcept
                        {
                            if (joinable())
                            {
                                join();
                            }
                            
                        }
                        
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
            - 支持的操作
                - `t.join()`： *合并* 线程
                    - 阻塞 *当前线程* `std::this_thread`，直至`t`关联的线程结束
                        - `t`所关联的线程的结束 *同步* 于对应的`join()`成功返回
                        - `t`自身**不**进行 *同步* ，同时从多个线程对同一个`std::thread`调用`join`构成 *数据竞争* ，是 *未定义行为* 
                        - `t.join`之后，`t.joinable()`为`false`
                    - 出现异常，则抛出`std::system_error`
                        - 若`t.get_id() == std::this_thread::get_id()`（检测到死锁），则抛出`std::resource_deadlock_would_occur`
                        - 若线程非法，则抛出`std::no_such_process`
                        - 若`!t.joinable()`，则抛出`std::invalid_argument`
                    - **注意**：生命周期问题
                        - 线程运行后产生的异常，会在`join()`调用之前抛出，这样就会跳过`join()`。因此在 *异常处理过程* 中也要记得调用`join()`
                        ```
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
                                t.join();  // 1
                                throw;
                            }
                            
                            t.join();      // 2
                        }
                        ```
                        - 另一种解决方法：`RAII`（Resource Acquisition Is Initialization），提供一个线程封装类，在析构函数中调用`join()`
                            - 如果不想等待线程结束，可以分离线程，从而避免异常
                            - 不过，这就打破了线程与std::thread对象的联系
                            - 即使线程仍然在后台运行着，分离操作也能确保在std::thread对象销毁时不调用std::terminate()
                        ```
                        class thread_guard
                        {
                        public:
                            explicit thread_guard(std::thread & t_) : t(t_)
                            {
                            
                            }
                            
                            thread_guard(const thread_guard &) = delete;              // 3 保证thread_guard和std::thread对象一样不可复制
                            
                            thread_guard & operator=(const thread_guard &) = delete;  // 3 保证thread_guard和std::thread对象一样不可复制
                            
                            ~thread_guard()
                            {
                                if (t.joinable())  // 1
                                {
                                    t.join();      // 2
                                }
                            }
                        
                        private:
                            std::thread & t;
                        };
                        
                        void f()
                        {
                            int some_local_state=0;
                            func my_func(some_local_state);
                            std::thread t(my_func);
                            thread_guard g(t);
                            do_something_in_current_thread();
                        }    
                        ```
                - `t.detach()`： *分离* 线程
                    - 将`t`关联的线程从`t`中 *分离* ，独立地执行
                        - 分离线程通常称 *守护线程* （daemon threads）
                        - 让线程在后台运行，这就意味着与主线程**不能**直接交互
                        - C++运行库保证，当线程退出时，相关资源的能够正确回收
                        - `t.detach()`后，`t`不再占有任何线程，`t.joinable()`为`false`
                        ```
                        std::thread t(do_background_work);
                        
                        if (t.joinable())
                        {
                            t.detach();
                        }
                        
                        assert(!t.joinable());
                        ```
                    - 异常
                        - 若`!t.joinable()`或出现任何错误，则抛出`std::system_error`
                        - 使用`detach()`前必须检查`t.joinable()`，返回的是`true`，才能`detach()`
                    - **注意**：如不等待线程`join`而是将其`detach`，就必须保证线程结束时其占用的资源仍是有效的，否则是 *未定义行为* 
                    ```
                    struct func
                    {
                        func(int & i_) : i(i_) 
                        {
                        
                        }
                        
                        void operator() ()
                        {
                            for (unsigned j = 0 ; j < 1000000 ; ++j)
                            {
                                do_something(i);           // 潜在访问隐患：空引用
                            }
                        }
                        
                        int & i;
                    };

                    void oops()
                    {
                        int some_local_state = 0;
                        func my_func(some_local_state);
                        std::thread my_thread(my_func);
                        my_thread.detach();                // 不等待线程结束
                                                           // oops返回时some_local_state便被析构
                                                           // 此时my_thread还在执行，其对some_local_state的引用便是非法引用！
                    }         
                    ```
                    - `detach`的应用场景举例：
                    ```
                    void edit_document(std::string const& filename)
                    {
                        open_document_and_display_gui(filename);
                        
                        while (!done_editing())
                        {
                            user_command cmd = get_user_input();
                            
                            if (cmd.type == user_command::open_new_document)
                            {
                                std::string const new_name = get_filename_from_user();
                                std::thread t(edit_document, new_name);
                                t.detach();
                            }
                            else
                            {
                                process_user_input(cmd);
                            }
                        }
                    }
                    ```
                - `t1.swap(t2)`，`std::swap(t1, t2)`： *交换* 线程
                    - 交换二个`std::thread`对象的底层柄
                    ```
                    std::thread t1([] () { std::this_thread::sleep_for(std::chrono::seconds(1)); });
                    std::thread t2([] () { std::this_thread::sleep_for(std::chrono::seconds(1)); });
                    std::cout << t1.get_id() << ' ' << t2.get_id() << '\n';  // 1 2
                 
                    std::swap(t1, t2);
                    std::cout << t1.get_id() << ' ' << t2.get_id() << '\n';  // 2 1
                 
                    t1.swap(t2);
                    std::cout << t1.get_id() << ' ' << t2.get_id() << '\n';  // 1 2
                 
                    t1.join();
                    t2.join();
                    ```
            - 观察器
                - `t.joinable()`：返回线程是否 *可合并* 
                    - 返回
                        - 若`t`标识活跃的执行线程，即`get_id() != std::thread::id()`，则返回`true`
                            - 因此， *默认构造* 的`std::thread` **不可合并**
                        - 已结束执行、但仍未被合并的线程仍被当作活跃的执行线程，从而 *可合并* 
                    - 示例
                    ```
                    std::thread t;
                    std::cout << t.joinable() << '\n';  // false
                 
                    t = std::thread([] () { std::this_thread::sleep_for(std::chrono::seconds(1)); });
                    std::cout << t.joinable() << '\n';  // true
                    
                    t.join();
                    std::cout << t.joinable() << '\n';  // false
                    ```
                - `t.get_id()`：返回线程`id`
                    - 返回
                        - 返回一个`std::thread::id`，表示与`t`关联的线程的`id`
                        - 若**无**关联的线程，则返回默认构造的`std::thread::id()`
                    - [`std::thread::id`](https://en.cppreference.com/w/cpp/thread/thread/id)
                        - 轻量的可频繁复制类，它作为`std::thread`对象的 *唯一标识符* 工作
                        - *默认构造* 的实例保有不表示任何线程的特殊辨别值
                        - 一旦线程结束，则`std::thread::id`的值可为另一线程复用
                        - 还被用于有序和无序 *关联容器* 的键值
                - `t.native_handle()`：返回实现定义的底层线程柄
                - `std::thread::hardware_concurrency()`：返回`unsigned int`，代表实现支持的并发线程数
                    - 应该只把该值当做提示
                    - 若该值非良定义或不可计算，则返回`0​` 
    - 管理 *当前线程* `std::this_thread`的静态函数，定义于`<this_thread>`
        - [`std::this_thread::yield()`](https://en.cppreference.com/w/cpp/thread/yield)：提供提示给实现，以重调度线程的执行，允许其他线程运行
        - [`std::this_thread::get_id()`](https://en.cppreference.com/w/cpp/thread/get_id)：返回当前线程的`id`
        - [`std::this_thread::sleep_for(duration)`](https://en.cppreference.com/w/cpp/thread/sleep_for)：阻塞当前线程执行，至少经过指定的`std::chrono::duration`
        - [`std::this_thread::sleep_until(time_point)`](https://en.cppreference.com/w/cpp/thread/sleep_until)：阻塞当前线程，直至抵达指定的`std::chrono::time_point`
    - 线程取消（thread cancellation），定义于`<stop_token>`
        - [`std::stop_token`](https://en.cppreference.com/w/cpp/thread/stop_token) `(since C++20)`
        - [`std::stop_source`](https://en.cppreference.com/w/cpp/thread/stop_source) `(since C++20)`
        - [`std::stop_callback`](https://en.cppreference.com/w/cpp/thread/stop_callback) `(since C++20)`
    - [`std::jthread`](https://en.cppreference.com/w/cpp/thread/jthread)：支持自动`join`以及`cancel`的`std::thread` `(since C++20)`
- *互斥* （mutual exclusion），定义于`<mutex>`
    - 互斥锁，定义于`<mutex>`
        - [`std::mutex`](https://en.cppreference.com/w/cpp/thread/mutex)
            - 互斥锁，具有如下特性
                - 调用方线程从它成功调用`mutex.lock()`或`mutex.try_lock()`开始，直到它调用`mutex.unlock()`为止占用`mutex`
                - 线程占有`mutex`时，所有其他线程若试图要求占有此`mutex`，则将
                    - 被阻塞，对于`mutex.lock()`
                    - 收到`false`返回值，对于`mutex.try_lock()`
                - 再次锁定已经被自己锁定的锁，或解锁不是被自己锁定的锁都是 *未定义行为* 
                - `std::mutex`既**不可** *复制* 亦**不可** *移动*  
            - 操作
                - `mutex.lock()`
                - `mutex.try_lock()`
                - `mutex.unlock()`
        - [`std::timed_mutex`](https://en.cppreference.com/w/cpp/thread/timed_mutex)
            - 限时互斥锁
                - `mutex.lock()`
                - `mutex.try_lock()`：如不能立即获得锁，则返回`false`
                - `mutex.try_lock_for(timeout_duration)`：如不能在`timeout_duration`时间内获得锁，则返回`false`
                - `mutex.try_lock_until(timeout_time_point)`：如不能在`timeout_time_point`时刻前获得锁，则返回`false`
                - `mutex.unlock()`
        - [`std::recursive_mutex`](https://en.cppreference.com/w/cpp/thread/recursive_mutex)
            - 递归互斥锁，允许已获得锁的线程递归获得锁，之后需调用相应次数的`unlock`来释放
            - 操作
                - `mutex.lock()`
                - `mutex.try_lock()`
                - `mutex.unlock()`
        - [`std::recursive_timed_mutex`](https://en.cppreference.com/w/cpp/thread/recursive_timed_mutex)
            - 递归限时互斥锁
            - 操作
                - `mutex.lock()`
                - `mutex.try_lock()`：如不能立即获得锁，则返回`false`
                - `mutex.try_lock_for(timeout_duration)`：如不能在`timeout_duration`时间内获得锁，则返回`false`
                - `mutex.try_lock_until(timeout_time_point)`：如不能在`timeout_time_point`时刻前获得锁，则返回`false`
                - `mutex.unlock()`
    - 共享互斥锁，定义于`<shared_mutex>`
        - [`std::shared_mutex`](https://en.cppreference.com/w/cpp/thread/shared_mutex) `(since C++17)`
            - 共享互斥锁
                - 拥有二个访问级别
                    - *共享* ：多个线程能共享同一互斥的所有权
                    - *独占* ：仅一个线程能占有互斥
                - 若一个线程已获取独占性锁（通过`lock`、`try_lock`），则无其他线程能获取该锁（包括共享的）
                - 仅当任何线程均未获取独占性锁时，共享锁能被多个线程获取（通过`lock_shared`、`try_lock_shared`）
                - 在一个线程内，同一时刻只能获取一个锁（共享或独占性）
                - 共享互斥锁在能由任何数量的线程同时读共享数据，但一个线程只能在无其他线程同时读写时写同一数据时特别有用
            - 共享锁定
                - `mutex.lock()`
                - `mutex.try_lock()`
                - `mutex.unlock()`
            - 互斥锁定
                - `mutex.lock_shared()`
                - `mutex.try_lock_shared()`
                - `mutex.unlock_shared()`
        - [`std::shared_timed_mutex`](https://en.cppreference.com/w/cpp/thread/shared_timed_mutex) `(since C++14)`
            - 限时共享互斥锁
            - 共享锁定
                - `mutex.lock()`
                - `mutex.try_lock()`
                - `mutex.try_lock_for(timeout_duration)`
                - `mutex.try_lock_until(timeout_time_point)`
                - `mutex.unlock()`
            - 互斥锁定
                - `mutex.lock_shared()`
                - `mutex.try_lock_shared_for(timeout_duration)`
                - `mutex.try_lock_shared_until(timeout_time_point)`
                - `mutex.try_lock_shared()`
                - `mutex.unlock_shared()`
    - 通用互斥管理
        - [`std::lock_guard`](https://en.cppreference.com/w/cpp/thread/lock_guard)
            - 签名
            ```
            template <class Mutex>
            class lock_guard;
            ```
            - 特性
                - 互斥锁封装器，提供[`RAII`](https://en.cppreference.com/w/cpp/language/raii)（Resource Acquisition Is Initialization）风格的块作用域内的互斥锁获取
                    - 实例被创建时，将获取互斥锁
                    - 块作用域结束，实例被析构时，将释放互斥锁
                - **不可**复制
            - 构造
                - `std::lock_guard<Mutex> lock(mutex);`：构造关联到`mutex`上的`std::lock_guard`，并调用`mutex.lock()`获得互斥。若`mutex`不是递归锁且当前线程已获得此锁，则 *行为未定义* 。若`mutex`先于`lock`被销毁，则 *行为未定义* 
                - `std::lock_guard<Mutex> lock(mutex, std::adopt_lock);`：构造关联到`mutex`上的`std::lock_guard`，且假设当前线程已经获得`mutex`。若实际未占有，则 *行为未定义* 
            - 用例
            ```
            int g_i = 0;
            std::mutex g_i_mutex;  // 保护 g_i
             
            void safe_increment()
            {
                std::lock_guard<std::mutex> lock(g_i_mutex);
                ++g_i;
             
                std::cout << std::this_thread::get_id() << ": " << g_i << '\n';
             
                // g_i_mutex 在锁离开作用域时自动释放
            }
             
            int main()
            {
                std::cout << "main: " << g_i << '\n';
             
                std::thread t1(safe_increment);
                std::thread t2(safe_increment);
             
                t1.join();
                t2.join();
             
                std::cout << "main: " << g_i << '\n';
            }
            ```
        - [`std::scoped_lock`](https://en.cppreference.com/w/cpp/thread/scoped_lock) `(since C++17)`
            - 签名
            ```
            template <class ... MutexTypes>
            class scoped_lock;
            ```
            - 特性
                - `RAII`风格互斥锁封装器，在块作用域存在期间占有 *一或多个* 互斥
                - 采用 *免死锁* 算法，如同`std::lock`
                - **不可**复制
            - 构造
                - `std::lock_guard<Mutex> lock(m1, m2, ...);`：构造关联到`mutex`上的`std::lock_guard`，并调用`mutex.lock()`获得互斥。若`mutex`不是递归锁且当前线程已获得此锁，则 *行为未定义* 。若`mutex`先于`lock`被销毁，则 *行为未定义* 
                - `std::lock_guard<Mutex> lock(std::adopt_lock, m1, m2, ...);`：构造关联到`mutex`上的`std::lock_guard`，且假设当前线程已经获得`mutex`。若实际未占有，则 *行为未定义* 
        - [`std::unique_lock`](https://en.cppreference.com/w/cpp/thread/unique_lock)
            - 签名
            ```
            template <class Mutex>
            class unique_lock;
            ```
            - 特性
                - 互斥锁封装器，根据封装的互斥锁，还会允许
                    - 延迟锁定`defer_lock`
                    - 锁定的有时限尝试`try_to_lock`
                    - 递归锁定
                    - 所有权转移
                    - 与条件变量一同使用  
                - 可 *移动* ，但**不可** *复制* 
            - 构造和赋值
                - `std::unique_lock<Mutex> u;`：默认构造关联类型为`Mutex`类型、且目前无关联互斥的`std::unique_lock`
                - `std::unique_lock<Mutex> u(mutex);`：显式构造与`mutex`关联的`std::unique_lock`，并调用`mutex.lock()`获得互斥的所有权。此构造函数为`explicit`的
                - `std::unique_lock<Mutex> u(mutex, tag);`：显式构造与`mutex`关联的`std::unique_lock`，同时遵循如下 *三种* `tag`
                    - `std::defer_lock`：`std::defer_lock_t`类型的内联字面值常量，不获得互斥的所有权
                    - `std::try_to_lock`：`std::try_to_lock_t`类型的内联字面值常量，调用`mutex.try_lock()`尝试获得互斥的所有权而不阻塞
                    - `std::adopt_lock`：`std::adopt_lock_t`类型的内联字面值常量，假设调用方线程已拥有互斥的所有权
                - `std::unique_lock<Mutex> u(mutex, duration)`：创建`std::unique_lock`并调用`mutex.try_lock_for(duration)`
                - `std::unique_lock<Mutex> u(mutex, time_point)`：创建`std::unique_lock`并调用`mutex.try_lock_until(time_point)`
                - `std::unique_lock<Mutex> u1(u2)`，`u1 = u2;`：移动构造和移动赋值
            - 操作
                - `u.lock()`
                - `u.try_lock()`
                - `u.try_lock_for(duration)`
                - `u.try_lock_until(time_point)`
                - `u.unlock()`
                - `u1.swap(ul2)`，`std::swap(u1, u2)`
                - `u.release()`：将关联的互斥锁解关联，但并不释放它
                - `u.mutex()`：返回指向其关联的互斥的指针。若无关联，则返回 *空指针* 
                - `u.owns_lock()`：返回其是否占有关联互斥
                - `operator bool()`：作为条件使用时，返回其是否占有关联互斥
        - [`std::shared_lock`](https://en.cppreference.com/w/cpp/thread/shared_lock) `(since C++14)`
            - 签名
            ```
            template <class Mutex>
            class shared_lock;
            ```
            - 特性
                - 互斥锁封装器，根据封装的互斥锁，还会允许
                    - 延迟锁定
                    - 锁定的有时限尝试
                    - 所有权转移
                - 锁定`std::shared_lock`将 *共享锁定* 与其关联的互斥锁
                    - 想要独占锁定，可以使用`std::unique_lock`
                - 可 *移动* ，但**不可** *复制* 
            - 构造和赋值
                - `std::shared_lock<Mutex> s;`：默认构造关联类型为`Mutex`类型、且目前无关联互斥的`std::shared_lock`
                - `std::shared_lock<Mutex> s(mutex);`：显式构造与`mutex`关联的`std::shared_lock`，并调用`mutex.lock_shared()`获得互斥的所有权。此构造函数为`explicit`的
                - `std::shared_lock<Mutex> s(mutex, tag);`：显式构造与`mutex`关联的`std::shared_lock`，同时遵循如下 *三种* `tag`
                    - `std::defer_lock`：`std::defer_lock_t`类型的内联字面值常量，不获得互斥的所有权
                    - `std::try_to_lock`：`std::try_to_lock_t`类型的内联字面值常量，调用`mutex.try_lock_shared()`尝试获得互斥的所有权而不阻塞
                    - `std::adopt_lock`：`std::adopt_lock_t`类型的内联字面值常量，假设调用方线程已拥有互斥的所有权
                - `std::shared_lock<Mutex> s(mutex, duration)`：创建`std::shared_lock`并调用`mutex.try_lock_shared_for(duration)`
                - `std::shared_lock<Mutex> s(mutex, time_point)`：创建`std::shared_lock`并调用`mutex.try_lock_shared_until(time_point)`
                - `std::shared_lock<Mutex> s1(s2)`，`s1 = s2;`：移动构造和移动赋值
            - 操作
                - `s.lock()`
                - `s.try_lock()`
                - `s.try_lock_for(duration)`
                - `s.try_lock_until()`
                - `s.unlock()`
                - `s1.swap(s2)`，`std::swap(s1, s2)`
                - `s.release()`：将关联的互斥锁解关联，但并不释放它
                - `s.mutex()`：返回指向其关联的互斥的指针。若无关联，则返回 *空指针* 
                - `s.owns_lock()`：返回其是否占有关联互斥
                - `operator bool()`：作为条件使用时，返回其是否占有关联互斥
    - 通用锁定算法
        - [`std::try_lock`](https://en.cppreference.com/w/cpp/thread/try_lock)
            - 签名
            ```
            template <class Lockable1, class Lockable2, class ... LockableN>
            int try_lock(Lockable1 & lock1, Lockable2 & lock2, LockableN & ... lockn);
            ```
            - 功能
                - 尝试锁定每个给定的锁，通过以从头开始的顺序调用`lockn.try_lock()`
                - 若调用`try_lock`失败，则不再进一步调用`try_lock`，并对任何已锁对象调用`unlock`，返回锁定失败对象的下标
                - 若调用`try_lock`抛出异常，则在重新抛出之前对任何已锁对象调用`unlock`
            - 返回值
                - 成功时为`-1`
                - 否则为锁定失败对象的下标值 
        - [`std::lock`](https://en.cppreference.com/w/cpp/thread/lock)
            - 签名
            ```
            template <class Lockable1, class Lockable2, class ... LockableN>
            void lock(Lockable1 & lock1, Lockable2 & lock2, LockableN & ... lockn);
            ```
            - 功能
                - 尝试锁定每个给定的锁，通过 *免死锁* 算法避免死锁的出现 
                - 给定的锁将以`lock`、`try_lock`和`unlock`的未给定序列锁定
                - 若上述过程中抛出异常，则在重新抛出之前对任何已锁对象调用`unlock`
            - 注意
                - 前面讲过的[`std::scoped_lock`](https://en.cppreference.com/w/cpp/thread/scoped_lock)提供此函数的`RAII`包装，通常它比裸调用`std::lock`更好
    - 单次调用
        - [`std::call_once`](https://en.cppreference.com/w/cpp/thread/call_once)
            - 签名
            ```
            template <class Callable, class ... Args>
            void call_once(std::once_flag & flag, Callable && f, Args && ... args);
            ```
            - 保证准确执行一次`f`
                - *消极* ：若在调用时刻`flag`指示已经调用了`f`，则什么也不做
                - *积极* ：否则，调用`std::invoke(std::forward<Callable>(f), std::forward<Args>(args) ...)`
                    - 不同于`std::thread`的构造函数或`std::async`，不移动或复制参数，因为不需要将它们转移至另一线程
                    - *异常* ：如果调用出现异常，则传播异常给`call_once`的调用方，且不反转`flag`
                    - *返回* ：若调用正常返回，则反转`flag`，并保证其它调用为 *消极* 
            - 注解
                - 同一`flag`上的所有 *积极调用* 组成单独全序，它们由零或多个异常调用后随一个 *返回调用* 组成
                    - 该顺序中，每个 *积极调用* 的结尾同步于下个积极调用
                - 从 *返回调用* 的返回同步于同一`flag`上的所有 *消极调用* 
                    - 这表示保证所有对`call_once`的同时调用都能观察到积极调用产生的副效应，而无需额外同步 
                - 若对`call_once`的同时调用传递不同的`f`，则调用哪个`f`是 *未指定* 的
            - [`std::once_flag`](https://en.cppreference.com/w/cpp/thread/once_flag)
                - `std::call_once`的辅助类
                    - 一个`std::once_flag`实例将被传递给多个`std::call_once`实例，用于多个`std::call_once`实例之间相互协调，保证最终只有一个`std::call_once`真正被完整执行
                - **不可** *复制* ，**不可** *移动*  
                - 默认构造函数：`constexpr once_flag() noexcept;`：默认构造一个指示目前还没有一个函数被调用的`std::once`实例
            - 示例
            ```
            std::once_flag flag;
            
            void simple_do_once()
            {
                std::call_once(flag1, [](){ std::cout << "Simple example: called once\n"; });
            }
            
            std::thread st1(simple_do_once);
            std::thread st2(simple_do_once);
            std::thread st3(simple_do_once);
            std::thread st4(simple_do_once);
            st1.join();
            st2.join();
            st3.join();
            st4.join();
            
            // OUTPUT: 
            Simple example: called once
            ```
- *条件变量* （condition variable），定义于`<condition_variable>`
    - [`std::condition_variable`](https://en.cppreference.com/w/cpp/thread/condition_variable)
        - 特性
            - 条件变量**不**包含 *互斥锁* 的 *条件* 
                - *互斥锁* 的 *条件* 需要被单独定义，并配合 *条件变量* 一同使用
                - 条件变量实现的是 *等待队列* 和 *广播* 功能
                - 线程还可以等待在条件变量上，并在需要时通过广播被唤醒
            - 条件变量用于阻塞一个线程或同时阻塞多个线程，直至另一线程修改共享 *条件* 、并 *通知* 此条件变量
                - 任何有意 *修改条件变量* 的线程必须
                    - 获得`std::mutex`（常通过`std::lock_guard`）
                    - 在保有锁时进行修改
                        - 即使共享变量是原子的，也必须在互斥下修改它，以正确地发布修改到等待的线程
                    - 在`std::condition_variable`上执行`notify_one`或`notify_all`（**不**需要为通知保有锁）
                - 任何有意 *在条件变量上等待* 的线程必须
                    - 在用于保护此条件变量的`std::mutex`上获得封装器`std::unique_lock<std::mutex>`
                    - 执行如下两种操作中的一种
                        - 第一种
                            - 检查 *条件* 是否为 *已更新* 或 *已被提醒* 
                            - 执行`wait`、`wait_for`或`wait_until`
                                - 等待操作将自动释放互斥锁，并挂起此线程
                            - 当此条件变量 *被通知* 、 *超时* 或 *伪唤醒* （被唤醒但条件仍不满足时），于此等待的线程被唤醒，并自动获得互斥锁
                                - 此线程应自行检查 *条件* ，如果是 *伪唤醒* ，则应继续进行一轮 *等待*  
                        - 第二种
                            - 使用`wait`、`wait_for`及`wait_until`的 *有谓词重载* ，它们包揽以上三个步骤 
            - 只工作于`std::unique_lock<std::mutex>`上的条件变量，在一些平台上此配置可以达到效率最优
                - 想用其他的互斥封装器，可以用楼下的[`std::condition_variable_any`](https://en.cppreference.com/w/cpp/thread/condition_variable_any)
            - 容许`wait`、`wait_for`、`wait_until`、`notify_one`及`notify_all`的并发调用
            - **不可** *复制* 、**不可** *移动* 
        - 操作
            - `std::conditional_variable cv;`：默认构造
            - `cv.notify_one()`：唤醒一个在`cv`上等待的线程
            - `cv.notify_all()`：唤醒全部在`cv`上等待的线程
            - `cv.wait(unique_lock, pred)`
                - 当前线程阻塞直至条件变量被通知，或伪唤醒发生
                    - 原子地解锁`unique_lock`，阻塞当前线程，并将它添加到`cv`的等待列表上
                    - 当前线程被唤醒时，将自动获得`unique_lock`并退出`wait`
                        - 若此函数通过异常退出，当前线程也会获得`unique_lock` `(until C++14)`
                - 谓词`pred`为可选的，用于在特定条件为`true`时忽略伪唤醒
                    - 如提供，则等价于`while (!pred) { cv.wait(unique_lock); }`
                - 若此函数不能满足后置条件`unique_lock.owns_lock()`、且调用方线程持有`unique_locl.mutex()`，则调用`std::terminate()` `(since C++14)`
                - 注解
                    - 若当前线程未获得`unique_lock.mutex()`，则调用此函数是 *未定义行为* 
                    - 若`unique_lock.mutex()`与所有当前等待在此条件变量上的线程所用`std::mutex`不是同一个，则 *行为未定义* 
            - [`cv.wait_for`](https://en.cppreference.com/w/cpp/thread/condition_variable/wait_for)
                - `cv.wait_for(unique_lock, duration)`
                    - 返回`enum std::cv_status {no_timeout, timeout}`。
                        - 若经过`duration`所指定的关联时限，则为 `std::cv_status::timeout`
                        - 否则为`std::cv_status::no_timeout`
                - `cv.wait_for(unique_lock, duration, pred)`
                    - 返回`bool`
                        - 如未超时，则为`true`
                        - 否则，为经过`duration`时限后谓词`pred`的值
                    - 等价于`return wait_until(lock, std::chrono::steady_clock::now() + rel_time, std::move(pred));`
            - [`cv.wait_until`](https://en.cppreference.com/w/cpp/thread/condition_variable/wait_until)
                - `cv.wait_until(unique_lock, time_point)`
                    - 返回`enum std::cv_status {no_timeout, timeout}`
                        - 若经过`time_point`所指定的关联时限，则为 `std::cv_status::timeout`
                        - 否则为`std::cv_status::no_timeout`
                - `cv.wait_until(unique_locl, time_point, pred)`
                    - 返回`bool`
                        - 如未超时，则为`true`
                        - 否则，为经过时限后谓词`pred`的值
                    - 等价于
                    ```
                    while (!pred()) 
                    {
                        if (wait_until(lock, timeout_time) == std::cv_status::timeout) 
                        {
                            return pred();
                        }
                    }
                    return true;
                    ```
    - [`std::condition_variable_any`](https://en.cppreference.com/w/cpp/thread/condition_variable_any)
        - `std::condition_variable`的泛化
        - `std::condition_variable_any`能与`std::shared_lock`一同使用，从而实现在`std::shared_mutex`上以 *共享* 模式等待
    - [`std::notify_all_at_thread_exit`](https://en.cppreference.com/w/cpp/thread/notify_all_at_thread_exit)
- *信号量* （semaphore），定义于`<semaphore>`
    - [`std::counting_semaphore`，`std::binary_semaphore`](https://en.cppreference.com/w/cpp/thread/counting_semaphore) `(since C++20)`
- *闩与屏障* （Latches and Barriers），定义于`<latch>`
    - [`std::latch`](https://en.cppreference.com/w/cpp/thread/latch) `(since C++20)`
    - [`std::barrier`](https://en.cppreference.com/w/cpp/thread/barrier) `(since C++20)`
- *线程间通讯* ，定义于`<future>`
    - `C++`线程支持库还提供`std::promise -> std::future`传递链，用于进程间共享信息
        - 没有这一功能时
            - 信息只能通过指针和动态存储区中的`volatile`变量传递
            - 而且还必须等待异步线程结束后，才能安全地获得这些信息
        - 当`std::promise`写入信息时，信息立即可访问，不必等待该线程结束
        - 这些信息在 *共享状态* （shared state）中传递
            - 其中异步任务可以写入信息或存储异常
            - *共享状态* 可以与数个`std::future`或`std::shared_future`实例关联，从而被它们所在的线程 *检验* 、 *等待* 或 *修改* 
    - 操作，定义于`<future>`
        - [`std::promise`](https://en.cppreference.com/w/cpp/thread/promise)
            - 签名
            ```
            template <class R> class promise;          (1) 空模板
            template <class R> class promise<R &>;     (2) 非 void 特化，用于在线程间交流对象
            template <>        class promise<void>;    (3) void 特化，用于交流无状态事件
            ```
            - 特性
                - 类模板`std::promise`提供存储值或异常的设施，之后通过`std::promise`对象所创建的`std::future`对象异步获得结果
                    - 注意`std::promise`只应当使用一次
                - 每个`promise`与 *共享状态* 关联
                    - 共享状态含有一些状态信息和可能仍未求值的结果
                    - 它求值为值（可能为`void`）或求值为异常
                - `promise`可以对共享状态做三件事：
                    - *就绪* ：`promise`存储结果或异常于共享状态。标记共享状态为就绪，并解除阻塞任何等待于与该共享状态关联的`future`上的线程
                    - *释放* ：`promise`放弃其对共享状态的引用。若这是最后一个这种引用，则销毁共享状态。除非这是`std::async`所创建的未就绪的共享状态，否则此操作**不**阻塞
                    - *抛弃* ：`promise`存储以`std::future_errc::broken_promise`为`error_code`的`std::future_error`类型异常，令共享状态为就绪，然后释放它
            - 构造和赋值
                - `std::promise<T> p;`： *默认构造* 一个共享状态为空的`std::promise`
                - `std::promise<T> p1(p2)`，`p1 = p2`：移动构造和移动赋值
            - 操作
                - `p1.swap(p2)`，`std::swap(p1, p2)`：交换二个`std::promise`对象 
                - `std::future<T> f = p.get_future();`：返回与`std::promise<T> p`的结果关联的`std::future<T>`对象。若无共享状态或已调用过`get_future`，则抛出异常。对此函数的调用与对`set_value`、`set_exception`、`set_value_at_thread_exit`或 `set_exception_at_thread_exit`的调用**不**造成数据竞争（但它们不必彼此同步）
                - `p.set_value(val)`：原子地存储`val`到共享状态，并令状态就绪
                - `p.set_value()`：仅对`std::promise<void>`特化成员，使状态就绪
                - `p.set_value_at_thread_exit(val)`：原子地存储`val`到共享状态，而不立即令状态就绪。在当前线程退出时，销毁所有拥有线程局域存储期的对象后，再令状态就绪。若无共享状态或共享状态已存储值或异常，则抛出异常
                - `p.set_value_at_thread_exit(ptr)`：存储`std::exception_ptr ptr`到共享状态中，并令状态就绪
                    - 异常指针
                    ```
                    std::exception_ptr eptr;
                    
                    try 
                    {
                        // throw something...
                    } 
                    catch(...)
                    {
                        eptr = std::current_exception();  // 捕获
                    }
                    ```
                - `p.set_exception_at_thread_exit(ptr)`：存储`std::exception_ptr ptr`到共享状态中，而不立即使状态就绪。在当前线程退出时，销毁所有拥有线程局域存储期的变量后，再零状态就绪
            - 示例
            ```
            void asyncFunc(std::promise<int> & prom)
            {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                prom.set_value(200);
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            
            int main()
            {
                std::promise<int> prom;
                std::future<int> fut = prom.get_future();
                std::thread t1 {asyncFunc, std::ref(prom)};
                std::cout << fut.get() << std::endl;         
                t1.join();                                   // 2s later, print 200
                return 0;                                    // another 2s later, thread end
            }
            ```
        - [`std::packaged_task`](https://en.cppreference.com/w/cpp/thread/packaged_task)
        - [`std::future`](https://en.cppreference.com/w/cpp/thread/future)
            - 签名
            ```
            template <class T> class future;          (1)
            template <class T> class future<T &>;     (2)
            template <>        class future<void>;    (3)
            ```
            - 特性
                - 类模板`std::future`提供访问异步操作结果的机制
                    - 通过`std::async`、`std::packaged_task`或`std::promise`创建的异步操作能提供一个`std::future`对象给该异步操作的创建者 
                    - 然后，异步操作的创建者能用各种方法查询、等待或从`std::future`提取值。若异步操作仍未提供值，则这些方法可能阻塞
                    - 异步操作准备好发送结果给创建者时，它能通过修改链接到创建者的`std::future`的共享状态（例如`std::promise::set_value`）进行
                - 注意，`std::future`所引用的共享状态不与另一异步返回对象共享（与`std::shared_future`相反） 
                - 可 *移动* ，**不可** *复制* 
            - 构造和赋值
                - `std::future<T> fut;`：默认构造
                - `std::future<T> f1(f2);`，`f1 = f2`：默认构造
            - 操作
                - `std::shared_future<T> sf = f.share();`：将`f`的共享状态转移至`shared_future`中。多个`td::shared_future `对象可引用同一共享对象，这对于`std::future`不可能。在`std::future`上调用`share`后`valid() == false`
                - `T t = f.get();`：等待直至`future`拥有合法结果并获取它。它等效地调用`wait()`等待结果。 若调用此函数前`valid()`为`false`则 *行为未定义* 
                - `f.get();`：仅对`std::future<void>`。释放任何共享状态。调用此方法后`valid()`为`false`
                - `f.valid()`：返回是否有合法结果
                - `f.wait()`：阻塞直至结果可用
                - `f.wait_for(duration)`：阻塞一段时间至结果可用或超时。返回`enum class future_status {ready, timeout, deferred}`
                - `f.wait_until(time_point)`：阻塞至结果可用或超时。返回`enum class future_status {ready, timeout, deferred}`
        - [`std::shared_future`](https://en.cppreference.com/w/cpp/thread/shared_future)
            - 提供的操作接口与`std::future`一样
            - 类模板`std::shared_future`提供访问异步操作结果的机制，类似`std::future`，除了允许多个线程等候同一共享状态
            - 不同于仅可移动的`std::future`（故只有一个实例能指代任何特定的异步结果），`std::shared_future`可复制而且多个`shared_future` 对象能指代同一共享状态
            - 若每个线程通过其自身的`shared_future`对象副本访问，则从多个线程访问同一共享状态是安全的
        - [`std::async`](https://en.cppreference.com/w/cpp/thread/async)：异步运行一个函数（有可能在新线程中执行），并返回保有其结果的 `std::future`
            - 签名
            ```
            template <class Function, class ... Args>
            std::future<std::invoke_result_t<std::decay_t<Function>, std::decay_t<Args> ...>>
            async(Function && f, Args && ... args);

            template <class Function, class ... Args>
            std::future<std::invoke_result_t<std::decay_t<Function>, std::decay_t<Args> ...>>
            async(std::launch policy, Function && f, Args && ... args);
            ```
            - [`std::launch`](https://en.cppreference.com/w/cpp/thread/launch)类型对象
                - `std::launch::async`：运行新线程，以异步执行任务
                - `std::launch::deferred`：调用方线程上首次请求其结果时执行任务（惰性求值） 
            - `例1`
            ```
            void asyncFunc()
            {
                std::cout << "async thread id# " << std::this_thread::get_id() << std::endl;
            }
            
            int main()
            {
                std::cout << "main thread id# " << std::this_thread::get_id() << std::endl;
                std::future<void> fut = std::async(std::launch::async, asyncFunc);
                return 0;
            }
            ```
            - `例2`
            ```
            void asyncFunc(int val)
            {
                std::cout << "async thread id# " << std::this_thread::get_id() << std::endl;
                return val + 100;
            }
            
            int main()
            {
                std::cout << "main thread id# " << std::this_thread::get_id() << std::endl;
                std::future<void> fut = std::async(std::launch::async, asyncFunc, 200);
                
                if (fut.valid())
                {
                    std::cout << fut.get() << std::endl;
                }
            }
            ```
    - 线程异常
        - [`std::future_error`](https://en.cppreference.com/w/cpp/thread/future_error)：继承自`std::logic_error`
        - [`std::future_category`](https://en.cppreference.com/w/cpp/thread/future_category)
        - [`std::future_errc`](https://en.cppreference.com/w/cpp/thread/future_errc)

#### [文件系统库](https://en.cppreference.com/w/cpp/filesystem)（Filesystem Library） `(since C++17)`

- *文件系统库* 提供在文件系统与其组件，例如路径、常规文件与目录上进行操作的设施
    - 文件系统库原作为`boost.filesystem`开发，并最终从`C++17`开始并入`ISO C++`
    - 定义于头文件`<filesystem>`、命名空间`std::filesystem`
        - `ubuntu 18.04 LTS`默认的`gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0`自然还不支持这东西
            - 实测直接用`boost`的`<boost/filesystem.hpp>`仍旧可以，但怎么用`boost`就是另外一个故事了
            - 具体配置见 [`CMakeList.txts`](https://github.com/AXIHIXA/Memo/blob/master/code/CMakeList/Boost/CMakeLists.txt)
        - `ubuntu 20.04 LTS`默认的`gcc (Ubuntu 9.3.0-10ubuntu2) 9.3.0`自然就可以了
    - 使用此库可能要求额外的 *编译器/链接器选项* 
        - `GNU < 9.1`实现要求用`-lstdc++fs`链接
        - `LLVM < 9.0`实现要求用`-lc++fs`链接 
    - 兼容性
        - 若层级文件系统不能为实现所访问，或若它不提供必要的兼容性，则文件系统库设施可能不可用
        - 若底层文件系统不支持，则一些特性可能不可用（例如`FAT`文件系统缺少 *符号链接* 并禁止 *多重硬链接* ）
        - 若对此库的函数的调用引入文件系统竞争，即多个线程、进程或计算机交错地访问并修改文件系统中的同一对象，则 *行为未定义*  
- 定义
    - *文件* （file）
        - 持有数据的文件系统对象，能被写入或读取，或二者皆可。文件拥有 *文件名* 及 *文件属性* 
        - *文件类型* （file type）是 *文件属性* 之一，可以是如下 *四种* 之一 
            - *目录* （directory）
            - *硬链接* （hard link）
            - *符号链接* （symbolic link）
            - *常规文件* （regular file）
    - *文件名* （file name）
        - 命名一个文件的字符串。容许字符、大小写区别、最大长度以及被禁止名称 *由实现定义* 
        - `"."`和`".."`有特殊含义
    - *路径* （path）
        - 标识一个文件的元素序列
            - 以可选的 *根名* （root name，例如`Windows`上的`"C:"`或`"//server"`）开始
            - 后随可选的 *根目录* （root directory，例如`Unix`上的`/`）
            - 后随零或更多个 *文件名* （file name，除了最后一个都必须是 *目录* 或 *到目录的链接* ）的序列 
        - 路径可以分为以下 *三种*
            - *绝对路径* ：无歧义地标识一个文件位置的路径
            - *规范路径* ：**不**含 *符号链接* 、`"."`或`".."`元素的绝对路径
            - *相对路径* ：标识相对于文件系统中某位置的文件位置的路径。特殊路径名`"."`（当前目录）和`".."` （父目录）是相对路径 
- 类
    - [`path`](https://en.cppreference.com/w/cpp/filesystem/path)：表示一个路径。 *核心类* 
    - [`filesystem_error`](https://en.cppreference.com/w/cpp/filesystem/filesystem_error)：文件系统错误时抛出的异常
    - [`directory_entry`](https://en.cppreference.com/w/cpp/filesystem/directory_entry)：目录条目
    - [`directory_iterator`](https://en.cppreference.com/w/cpp/filesystem/directory_iterator)：指向目录内容的迭代器
    - [`recursive_directory_iterator`](https://en.cppreference.com/w/cpp/filesystem/recursive_directory_iterator)：指向一个目录及其子目录的内容的迭代器
    - [`file_status`](https://en.cppreference.com/w/cpp/filesystem/file_status)：表示文件类型及权限
    - [`space_info`](https://en.cppreference.com/w/cpp/filesystem/space_info)：关于文件系统上空闲及可用空间的信息
- 枚举
    - [`file_type`](https://en.cppreference.com/w/cpp/filesystem/file_type)：文件的类型
    - [`perms`](https://en.cppreference.com/w/cpp/filesystem/perms)：标识文件系统权限
    - [`perm_options`](https://en.cppreference.com/w/cpp/filesystem/perm_options)：指定权限操作的语义
    - [`copy_options`](https://en.cppreference.com/w/cpp/filesystem/copy_options)：指定复制操作的语义
    - [`directory_options`](https://en.cppreference.com/w/cpp/filesystem/directory_options)：用于迭代目录内容的选项
- `typedef`
    - [`file_time_type`](https://en.cppreference.com/w/cpp/filesystem/file_time_type)：表示文件时间值
- 非成员函数
    - [`absolute`](https://en.cppreference.com/w/cpp/filesystem/absolute)：组成一个绝对路径
    - [`canonical`, `weakly_canonical`](https://en.cppreference.com/w/cpp/filesystem/canonical)：组成一个规范路径
    - [`relative`, `proximate`](https://en.cppreference.com/w/cpp/filesystem/relative)：组成一个相对路径
    - [`copy`](https://en.cppreference.com/w/cpp/filesystem/copy)：复制文件或目录
    - [`copy_file`](https://en.cppreference.com/w/cpp/filesystem/copy_file)：复制文件内容
    - [`copy_symlink`](https://en.cppreference.com/w/cpp/filesystem/copy_symlink)：复制一个符号链接
    - [`create_directory`, `create_directories`](https://en.cppreference.com/w/cpp/filesystem/create_directory)：创建新目录
    - [`create_hard_link`](https://en.cppreference.com/w/cpp/filesystem/create_hard_link)：创建一个硬链接
    - [`create_symlink`, `create_directory_symlink`](https://en.cppreference.com/w/cpp/filesystem/create_symlink)：创建一个符号链接
    - [`current_path`](https://en.cppreference.com/w/cpp/filesystem/current_path)：返回或设置当前工作目录
    - [`exists`](https://en.cppreference.com/w/cpp/filesystem/exists)：检查路径是否指代既存的文件系统对象
    - [`equivalent`](https://en.cppreference.com/w/cpp/filesystem/equivalent)：检查两个路径是否指代同一文件系统对象
    - [`file_size`](https://en.cppreference.com/w/cpp/filesystem/file_size)：返回文件的大小
    - [`hard_link_count`](https://en.cppreference.com/w/cpp/filesystem/hard_link_count)：返回指代特定文件的硬链接数
    - [`last_write_time`](https://en.cppreference.com/w/cpp/filesystem/last_write_time)：获取或设置最近一次数据修改的时间
    - [`permissions`](https://en.cppreference.com/w/cpp/filesystem/permissions)：修改文件访问权限
    - [`read_symlink`](https://en.cppreference.com/w/cpp/filesystem/read_symlink)：获得符号链接的目标
    - [`remove`](https://en.cppreference.com/w/cpp/filesystem/remove)：移除一个文件或空目录
    - [`remove_all`](https://en.cppreference.com/w/cpp/filesystem/remove)：移除一个文件或递归地移除一个目录及其所有内容
    - [`rename`](https://en.cppreference.com/w/cpp/filesystem/rename)：移动或重命名一个文件或目录
    - [`resize_file`](https://en.cppreference.com/w/cpp/filesystem/resize_file)：以截断或填充零更改一个常规文件的大小
    - [`space`](https://en.cppreference.com/w/cpp/filesystem/space)：确定文件系统上的可用空闲空间
    - [`status`](https://en.cppreference.com/w/cpp/filesystem/status)：确定文件属性
    - [`symlink_status`](https://en.cppreference.com/w/cpp/filesystem/status)：确定文件属性，检查符号链接目标
    - [`temp_directory_path`](https://en.cppreference.com/w/cpp/filesystem/temp_directory_path)：返回一个适用于临时文件的目录
- 文件类型判断
    - [`is_block_file`](https://en.cppreference.com/w/cpp/filesystem/is_block_file)：检查给定的路径是否表示块设备
    - [`is_character_file`](https://en.cppreference.com/w/cpp/filesystem/is_character_file)：检查给定的路径是否表示字符设备
    - [`is_directory`](https://en.cppreference.com/w/cpp/filesystem/is_directory)：检查给定的路径是否表示一个目录
    - [`is_empty`](https://en.cppreference.com/w/cpp/filesystem/is_empty)：检查给定的路径是否表示一个空文件或空目录
    - [`is_fifo`](https://en.cppreference.com/w/cpp/filesystem/is_fifo)：检查给定的路径是否表示一个命名管道
    - [`is_other`](https://en.cppreference.com/w/cpp/filesystem/is_other)：检查参数是否表示一个其他文件
    - [`is_regular_file`](https://en.cppreference.com/w/cpp/filesystem/is_regular_file)：检查参数是否表示一个常规文件
    - [`is_socket`](https://en.cppreference.com/w/cpp/filesystem/is_socket)：检查参数是否表示一个具名`IPC socket`
    - [`is_symlink`](https://en.cppreference.com/w/cpp/filesystem/)：检查参数是否表示一个符号链接
    - [`status_known`](https://en.cppreference.com/w/cpp/filesystem/status_known)：检查文件状态是否已知



### 🌱 [Chap 18] 用于大型工程的工具

#### [属性说明符](https://en.cppreference.com/w/cpp/language/attributes)（attribute specifier）

- 为类型、对象、代码等引入由实现定义的 *属性* 
    - 几乎可以出现于任何地方
    - `[[`只能是属性说明符，`a[[] () { return 0; }]`会报错
- `C++`标准属性说明符
```
[[noreturn]]

[[carries_dependency]]

[[deprecated]]                   (since C++14)
[[deprecated("reason")]]         (since C++14)

[[fallthrough]]                  (since C++17)

[[nodiscard]]                    (since C++17)
[[nodiscard(string_literal)]]    (since C++20)

[[maybe_unused]]                 (since C++17)

[[likely]]                       (since C++20)  // 用于分支条件，提示编译器优化
[[unlikely]]                     (since C++20)  // 用于分支条件，提示编译器优化

[[no_unique_address]]            (since C++20)
```

#### [异常处理](https://en.cppreference.com/w/cpp/error)（exception handling）

- *异常类*
    - `C++`标准异常类
        - [`std::exception`](https://en.cppreference.com/w/cpp/error/exception)：标准错误。只报告异常的发生，不提供任何额外信息
            - [`std::logic_error`](https://en.cppreference.com/w/cpp/error/logic_error)：标准逻辑错误
                - [`std::invalid_argument`](https://en.cppreference.com/w/cpp/error/invalid_argument)
                - [`std::domain_error`](https://en.cppreference.com/w/cpp/error/domain_error)：参数对应的结果值不存在
                - [`std::invalid_argument`](https://en.cppreference.com/w/cpp/error/invalid_argument)：无效参数
                - [`std::length_error`](https://en.cppreference.com/w/cpp/error/length_error)：试图创建一个超出该类型最大长度的对象
                - [`std::out_of_range`](https://en.cppreference.com/w/cpp/error/out_of_range)：使用了一个超出有效范围的值
                - [`std::future_error`](https://en.cppreference.com/w/cpp/thread/future_error)
            - [`std::bad_optional_access`](https://en.cppreference.com/w/cpp/utility/optional/bad_optional_access) `(since C++17)`
            - [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error)：标准运行错误
                - [`std::range_error`](https://en.cppreference.com/w/cpp/error/range_error)：生成的结果超出了有意义的值域范围
                - [`std::overflow_error`](https://en.cppreference.com/w/cpp/error/overflow_error)：计算溢出
                - [`std::underflow_error`](https://en.cppreference.com/w/cpp/error/underflow_error)：计算溢出
                - [`std::regex_error`](https://en.cppreference.com/w/cpp/regex/regex_error)：正则表达式语法非法
                - [`std::system_error`](https://en.cppreference.com/w/cpp/error/system_error)
                    - [`std::ios_base::failure`](https://en.cppreference.com/w/cpp/io/ios_base/failure)
                    - [`std::filesystem::filesystem_error`](https://en.cppreference.com/w/cpp/filesystem/filesystem_error) `(since C++17)`
                - [`std::nonexist_local_time`](https://en.cppreference.com/w/cpp/chrono/nonexistent_local_time) `(since C++20)`
                - [`std::ambiguous_local_time`](https://en.cppreference.com/w/cpp/chrono/ambiguous_local_time) `(since C++20)`
                - [`std::format_error`](https://en.cppreference.com/w/cpp/utility/format/format_error) `(since C++20)`
            - [`std::bad_typeid`](https://en.cppreference.com/w/cpp/types/bad_typeid)：`typeid(*p)`运行时解引用了非法多态指针 => 19.2
            - [`std::bad_cast`](https://en.cppreference.com/w/cpp/types/bad_cast)：非法的`dynamic_cast` => 19.2
                - [`std::bad_any_cast`](https://en.cppreference.com/w/cpp/utility/any/bad_any_cast)
            - [`std::bad_weak_ptr`](https://en.cppreference.com/w/cpp/memory/bad_weak_ptr)
            - [`std::bad_function_call`](https://en.cppreference.com/w/cpp/utility/functional/bad_function_call)
            - [`std::bad_alloc`](https://en.cppreference.com/w/cpp/memory/new/bad_alloc)：分配内存空间失败 => 12.1.2
                - [`std::bad_array_new_length`](https://en.cppreference.com/w/cpp/memory/new/bad_array_new_length)
            - [`std::bad_exception`](https://en.cppreference.com/w/cpp/error/bad_exception)
            - [`std::bad_variant_access`](https://en.cppreference.com/w/cpp/utility/variant/bad_variant_access) `(since C++17)`
        - 异常类型都定义了一个名为`what`的成员函数，返回`C`风格字符串`const char *`，提供异常的文本信息
            - 如果此异常传入了初始参数，则返回之
            - 否则，返回值 *由实现决定* 
        - `std::exception`仅仅定义了
            - *默认构造函数*
            - *拷贝构造函数* 
            - *拷贝赋值运算符* 
            - *虚析构函数* 
            - 成员函数[`virtual const char * what() noexcept`](https://en.cppreference.com/w/cpp/error/exception/what)
        - `std::exception`、`std::bad_cast`和`std::bad_alloc`定义了 *默认构造函数* 
        - `std::runtime_error`以及`std::logic_error`**没有** *默认构造函数* 
            - 也就是说这俩要抛出、初始化时必须传参（`C`风格字符串或`std::string`）
            - 由于`what`是虚函数，因此对`what`的调用将执行与异常对象动态类型相对应的版本
    - 自定义异常类
    ```c++
    // hypothetical exception classes for a bookstore application
    class out_of_stock : public std::runtime_error 
    {
    public:
        explicit out_of_stock(const std::string & s) : std::runtime_error(s) 
        { 
        
        }
    };
    
    class isbn_mismatch : public std::logic_error 
    {
    public:
        explicit isbn_mismatch(const std::string & s) : std::logic_error(s) 
        { 
        
        }
        
        isbn_mismatch(const std::string & s, const std::string & lhs, const std::string & rhs) 
                : std::logic_error(s), left(lhs), right(rhs) 
        { 
        
        }
        
        const std::string left; 
        const std::string right;
    };
    
    // throws an exception if both objects do not refer to the same book
    Sales_data & Sales_data::operator+=(const Sales_data & rhs)
    {
        if (isbn() != rhs.isbn())
            throw isbn_mismatch("wrong isbns", isbn(), rhs.isbn());
        units_sold += rhs.units_sold;
        revenue += rhs.revenue;
        return *this;
    }
    
    // use the hypothetical bookstore exceptions
    Sales_data item1, item2, sum;
    
    while (std::cin >> item1 >> item2) 
    { 
        // read two transactions
        try 
        {
            sum = item1 + item2;  // calculate their sum
            // use sum
        } 
        catch (const isbn_mismatch & e) 
        {
            std::cerr << e.what() 
                      << ": left isbn(" << e.left << ") right isbn(" << e.right << ")" 
                      << std::endl;
        }
    }
    ```
- *抛出* 异常
    - `C++`通过 *抛出* （throwing）一条表达式来 *引发* （raising）一个异常
        - 被抛出的表达式的类型以及当前的调用链共同决定了哪段 *处理代码* （handler）将被用来处理该异常
        - 被选中的处理代码是在调用链中与抛出对象类型匹配的最近的处理代码
    - 当执行一个`throw`时，跟在这个`throw`后面的语句都**不会**被执行
        - 相反，程序的控制权从`throw`转移到与之匹配的 *`catch`模块* 
        - 该`catch`可能是同一个函数中的局部`catch`，也可能位于直接或间接调用了发生异常的函数的另一个函数中
        - 控制权转移意味着
            - 沿着调用链的函数可能提早推出
            - 一旦程序开始执行异常处理代码，则沿着调用链创建的对象将被销毁
        - 因为跟在`throw`后面的语句将不再执行，所以`throw`语句的用法有点类似于`return`语句
            - 它通常作为条件语句的一部分
            - 或者作为某个函数的最后（或唯一）一条语句
    - *栈展开* （stack unwinding / unwound）
        - 当抛出一个异常后，程序暂停当前函数的执行，并立即开始寻找与异常匹配的 *`catch`子句* 
        - 当`throw`或 *对抛出异常的函数的调用* 出现在一个 *`try`语句块* （`try` block）内时，检查与该`try`块相关联的`catch`子句
        - 如果找到了匹配的`catch`，就用该`catch`处理异常；处理完毕后，找到与`try`块关联的最后一个`catch`子句之后的点，并从这里继续执行
        - 如果没找到匹配的`catch`，但该`try`语句块嵌套在其他`try`块中，则继续检查与外层`try`相匹配的`catch`子句
        - 如果还是找不到匹配的`catch`，则退出当前函数，在调用当前函数的外层函数中继续寻找
        - 如果最终还是没有找到`catch`，退出了 *主函数* ，程序将调用`std::terminate()`终止整个程序的运行
    - 栈展开过程中对象被自动销毁
        - 栈展开会导致退出语句块，则在之前创建的对象都应该被销毁
        - 尤其对于数组或标准库容器的构造过程，如果异常发生时已经构造了一部分元素，则应该确保这部分元素被正确地销毁
    - 析构函数与异常
        - 析构函数总会被执行，但函数中负责释放资源的代码却可能被跳过
            - 例如，一个块分配了资源，并且负责释放这些资源的代码前面发生了异常，则释放资源的代码将**不会**被执行
        - 另一方面， *类对象分配的资源* 将由类的 *析构函数* 负责释放
            - 使用类来控制资源的分配，就能确保不论发生异常与否，资源都能被正确释放
        - 析构函数在 *栈展开* 过程中被执行
            - 栈展开过程中，一个异常已经被抛出，但尚未被处理
            - 如果此时又出现了新的异常，又未能被捕获，则程序将调用`std::terminate()`
            - 因此，析构函数**不应**抛出不能被它自己处理的异常
                - 换句话说：如果析构函数将要执行某个可能抛出异常的操作，则该操作应该被放置在一个`try`块内，并在析构函数内部得到处理
        - 在实际编程过程中，因为析构函数仅仅是释放资源，所以它不大可能抛出异常
            - 所有标准库类型都能确保它们的析构函数**不会**引发异常
    - *异常对象* （exception object）
        - *异常对象* 是一种特殊的对象，由`throw`语句对其进行 *拷贝初始化* 
            - 因此，`throw`语句中的表达式必须拥有完全类型
            - 而且，如果该表达式时类类型的话，则相应的类必须含有一个可访问的析构函数和一个可访问的拷贝或移动构造函数
            - 如果表达式是数组类型或函数类型，则表达式将被转换成与之对应的指针类型
        - 异常对象位于 *由编译器管理的空间* 中
            - 编译器确保不论最终调用的是哪个`catch`子句，都能访问该空间
            - 当异常处理完毕后，异常对象被销毁
            - 栈展开过程中会逐层退出块，销毁该块内的局部对象
                - 因此，`throw` *指向局部对象的指针* 是**错误**的，因为执行到`catch`之前局部对象就已经被销毁了
                - `throw`指针要求在任何对应的`catch`子句所在的地方，指针所指的对象都必须存在
                - 类似地，函数也不能返回指向局部对象的指针或引用
        - `throw`表达式时，该表达式的 *静态编译时类型* 决定了异常对象的类型
            - 即：`throw` *解引用多态指针* 也是**错误**的。解引用指向派生类对象的基类指针会导致被抛出对象 *被截断* 
        - `Clang-Tidy`要求只能`throw`在`throw`子句中临时创建的匿名`std::exception`类及其派生类对象
- *捕获* 异常
    - *`catch`子句* （catch clause）中的 *异常声明* （exception declaration）看起来像是只包含一个形参的函数形参列表
        - 像在函数形参列表中一样，如果`catch`无需访问抛出的表达式的话，则我们可以忽略捕获形参的名字
        - 声明的类型决定了此`catch`子句所能捕获并处理的异常的类型
        - 声明的类型可以是 *左值引用* ，但**不能**是 *右值引用* 
    - 当进入`catch`子句后，通过异常对象初始化异常声明中的参数
        - 和函数形参类似
            - 如果`catch`的形参类型是 *非引用类型* 
                - 则该形参是异常对象的一个 *副本* 
                - 在`catch`中改变异常对象实际上改变的是局部副本而**不是**异常对象本身
            - 如果`catch`的形参类型是 *左值引用类型* 
                - 则在`catch`中改变异常对象实际上改变的就是异常对象本身
            - 如果`catch`形参类型是 *基类类型* 
                - 则可以使用派生类类型的异常对象对其初始化
                - 只是这样会截断一部分内容
            - `catch`形参**不是**多态的
                - 即：如果`catch`形参类型是基类类型的引用
                - 该参数将以 *常规方式* 绑定到异常对象上
        - 注意：异常声明的静态类型决定了`catch`子句所能执行的操作
            - 如果`catch`的是基类类型，则`catch`子句无法使用派生类特有的任何成员
        - 通常情况下，如果`catch`接受的异常与某个继承体系有关，则通常将其捕获形参定义为引用类型
    - 查找匹配的异常处理代码
        - 在搜寻`catch`子句的过程中，我们最终找到的`catch`**未必**是异常的最佳匹配
            - 相反，挑选出来的是 *第一个* 匹配的
            - 因此，越是专门的`catch`，就越应该置于整个`catch`列表的前端
        - 与函数匹配规则相比，异常匹配规则受到很多限制
            - 绝大多数 *类型转换* 都**不**被允许。除了以下 *三种* 以外，要求异常类型与`catch`形参类型 *精确匹配* 
                - 允许从 *非常量* 向 *常量* 的转换
                    - 即：一条非常量对象的`throw`可以匹配捕获常量的`catch`子句
                - 允许从 *派生类* 向 *基类* 的转换
                - 数组被转换成指向数组元素类型的指针，函数被转换成指向该函数类型的指针
        - 如果在多个`catch`语句的类型之间存在着继承关系，则我们应该把继承链最底端的类（most derived type）放在前面，而将继承链最顶端的类（least derived type）放在后面
    - *重新抛出* （rethrowing）
        - 有时一个单独的`catch`子句不能完整地处理某个异常，在执行了某些校正操作之后，当前的`catch`可能会决定由调用链更上一层的函数接着处理异常
        - 一条`catch`语句通过 *重新抛出* 的操作，将 *当前的异常传递* 给 *其他的* `catch`语句
            - 重新抛出任然是一条`throw`语句，但**不**包含任何表达式，形如
            ```
            throw;
            ```
            - 空的`throw`语句只能出现在`catch`语句或`catch`语句直接或间接调用的函数之内
            - 如果异常处理代码之外的区域遇到了空的`throw`语句，编译器将调用`std::terminate()`
        - 重新抛出后，新的接收者可能是更上一层的`catch`语句，也可能是同层更靠后的`catch`语句
        - 如果`catch`语句改变了参数内容，则只有当参数是左值引用类型时，改变才会被保留并继续传播
        ```
        catch (my_error & eObj)                 // specifier is a reference type
        { 
            eObj.status = errCodes::severeErr;  // modifies the exception object
            throw;                              // the status member of the exception object is severeErr
        } 
        catch (other_error eObj)                // specifier is a nonreference type
        { 
            eObj.status = errCodes::badErr;     // modifies the local copy only
            throw;                              // the status member of the exception object is unchanged
        }
        ```
    - *捕获所有异常* （catch-all）的处理代码
        - 一条`catch`语句通过 *省略号异常声明* `(...)`来 *捕获所有异常* 
            - 形如
            ```
            catch (...)
            ```
            - 一条捕获所有异常的`catch (...)`语句可以与任意类型的异常匹配
                - 虽然这是非常推荐的，但并不是所有的异常都必须继承自`std::exception`
                - 因此`catch (...)`要比`catch (std::exception & e)`更万金油一些
        - `catch (...)`通常与 *重新抛出* 语句一起使用，其中`catch`执行当前局部能完成的工作，随后抛出异常
        ```
        void manip() 
        {
            try 
            {
                // actions that cause an exception to be thrown
            }
            catch (...) 
            {
                // work to partially handle the exception
                throw;
            }
        }
        ```
        - `catch (...)`既能 *单独出现* ，又能与其他几个`catch`语句 *一同出现* 
            - 一同出现时，`catch (...)`自然必须放在最后
            - 不然你让别人怎么玩儿
- *函数`try`语句块* 与构造函数
    - 初始化列表
        - 通常情况下，程序执行的任何时刻都可能发生异常，特别是异常可能发生于处理构造函数初始值的过程中
        - 构造函数在进入其函数体之前首先执行初始值列表
        - 因为在初始值列表抛出异常时，构造函数体内的`try`语句块还未生效，所以构造函数体内的`catch`语句**无法**处理构造函数初始化列表抛出的异常
    - [*函数`try`语句块*](https://en.cppreference.com/w/cpp/language/function-try-block) （function try blocks）
        - 另一译名为 *函数测试块* 
        - 函数`try`语句块使得一组`catch`语句既能处理构造 *函数体* （或析构函数体），也能处理构造函数的 *初始化过程* （或析构函数的 *析构过程* ）
        - 关键字`try`出现在表示构造函数初始值列表的 *冒号* （如有），或表示构造函数体的 *花括号* *之前* ，例如
        ```
        template <typename T>
        Blob<T>::Blob(std::initializer_list<T> il) try : data(std::make_shared<std::vector<T>>(il)) 
        {
            /* empty constructor body */
        } 
        catch (const std::bad_alloc & e) 
        { 
            handle_out_of_memory(e); 
        }
        ```
        - 与上例中的`try`关联的`catch`既能处理构造函数体抛出的异常，又能处理成员初始化列表抛出的异常
    - 用实参 *初始化构造函数的形参* 时发生的异常**不**属于函数`try`语句块的一部分
        - 函数`try`语句块只能处理构造函数开始执行之后发生的异常
        - 和其他函数调用一样，如果在参数初始化过程中发生了异常，则该异常属于调用表达式的一部分，并将在调用者的上下文中处理
    - 处理构造函数初始值异常的 *唯一方法* 就是将构造函数写成函数`try`语句块
- *`noexcept`说明符* （`noexcept` specification）
    - 对于用户以及编译器来说，预先知道某个函数不会抛出异常显然大有裨益
        - 首先，有益于简化调用该函数的代码
        - 其次，如果编译器确认函数不会抛出异常，它就能执行某些不适用于可能出错的代码的特殊优化操作
    - 提供 *`noexcept`异常说明* 可以指定某个函数**不会**抛出异常
        - 关键字`noexcept`跟在函数的 *形参列表后面* 
            - *尾置返回类型之前* （如有）
            - 成员函数：`const`及 *引用限定之后* 、 *`final`、`override`或纯虚函数的`= 0`之前* 
        ```
        void recoup(int) noexcept;  // won't throw
        void alloc(int);            // may throw
        
        auto fun(int) noexcept -> void;
        
        void Base::virtual_fun() noexcept const & = 0;
        void Derived::virtual_fun() noexcept const & override;
        
        ```
        - 我们说`recoup`做了 *不抛出说明* （nothrowing specialization）
        - 对于一个函数来说，`noexcept`说明要么出现在该函数的 *所有* 声明语句和定义语句中，要么就 *一次也不* 出现
    - 违反异常说明
        - 通常情况下，编译器**无法、也不会**在编译时检查`noexcept`说明
            - 实际上会抛出异常的`noexcept`函数也能通过编译
        - `noexcept`函数如果在运行时抛出了异常，则程序会立即调用`std::terminate()`终止程序，来保证 *不在运行时抛出异常* 的承诺
            - 此时是否进行栈展开未定义
    - `noexcept`可以用于两种情况
        1. 我们确定此函数不会抛出异常
        2. 我们根本不知道如何处理异常
    - 向后兼容：异常说明
        - `throw (exception_list)`说明符，位置与`noexcept`相同
        - 函数可以指定关键字`throw`，后跟可能抛出的异常类型的列表
            - 空列表代表不会抛出异常
    ```
    void recoup(int) noexcept;        // recoup won't throw
    void recoup(int) throw();         // equivalent
    ```
    - 异常说明的实参
        - `noexcept`接受一个 *可选* 的实参
        - 必须能够转化成`bool`
        - 实参为`true`代表**不会**抛出异常，为`false`则代表可能抛出异常
    ```
    void recoup(int) noexcept(true);  // recoup won't throw
    void alloc(int) noexcept(false);  // alloc can throw
    ```
    - *`noexcept`运算符* （`noexcept` operator）
        - `noexcept`运算符是一个 *一元运算符* ，返回值时一个`bool`类型的 *右值常量表达式* ，表示运算对象是否会抛出异常
            - 和`sizeof`类似，`noexcept`也**不会**对运算对象求值
            - 调用格式
            ```
            noexcept(e)
            ```
            - 当`e`调用的所有函数都做了`noexcept`说明且`e`本身不含有`throw`语句时，上述表达式为`true`；否则，为`false`
            - 例如
            ```
            noexcept(recoup(i))  // true if calling recoup can't throw, false otherwise
            ```
        - `noexcept`说明符的实参常常与 *`noexcept`运算符* 混合使用
        ```
        void f() noexcept(noexcept(g()));  // f has same exception specifier as g
        ```
    - 异常说明与指针、虚函数和拷贝控制
        - *函数指针* 与该指针 *所指的函数* 
            - 如果我们为某个指针做了`noexcept`声明，则该指针将只能指向`noexcept`的函数
            - 相反，如果我们显式或隐式指明了指针可能抛出异常，则该指针可以指向任何函数，包括`noexcept`的、以及不`noexcept`的
        ```
        void (*pf1)(int) noexcept = recoup;  // ok: both recoup and pf1 promise not to throw
        void (*pf2)(int) = recoup;           // ok: recoup won't throw; it doesn't matter that pf2 might
        pf1 = alloc;                         // error: alloc might throw but pf1 said it wouldn't
        pf2 = alloc;                         // ok: both pf2 and alloc might throw
        ```
        - *虚函数* 与派生类中的`override`
            - 如果我们为基类中某个虚函数做了`noexcept`声明，则派生类中的`override`也必须是`noexcept`的
            - 相反，如果我们显式或隐式指明了这个虚函数可能抛出异常，则派生类中的`override`不论是否`noexcept`都可以
        ```
        class Base 
        {
        public:
            virtual double f1(double) noexcept;  // doesn't throw
            virtual int f2() noexcept(false);    // can throw
            virtual void f3();                   // can throw
        };
        
        class Derived : public Base 
        {
        public:
            double f1(double);                   // error: Base::f1 promises not to throw
            int f2() noexcept(false);            // ok: same specification as Base::f2
            void f3() noexcept;                  // ok: Derived f3 is more restrictive
        };
        ```
        - *拷贝控制成员*
            - 当编译器合成拷贝控制成员时，同时也生成一个异常说明
            - 如果对所有成员和基类的所有操作都承诺了`noexcept`，则合成的成员也是`noexcept`的
            - 如果合成成员调用的任意一个函数可能抛出异常，则合成的成员是`noexcept(false)`
            - 而且，如果我们定义了一个析构函数但没有提供异常说明，则编辑器将合成一个异常说明
            - 合成的异常说明将于假设由编译器合成析构函数时所得的异常说明一致

#### [命名空间](https://en.cppreference.com/w/cpp/language/namespace)（Namespaces）

- *命名空间污染* （namespace pollution）
    - 多个库将名字放置在 *全局命名空间* 中导致名字冲突的情况
        - 传统解决方法：将名字定义得很长，例如`cplusplus_primer_fun1`
            - 此方法仍旧用于 *宏定义* （macro），例如头文件的保护头
                - 宏定义处理发生在预处理阶段
                - 命名空间解析发生于之后的编译阶段，管不到这东西
        - 实名diss一下`OpenCV`里那个叫`debug`的宏，还有`<cmath>`里那个叫`y1`的函数，缺德死了
    - *命名空间* 为防止名字冲突提供了更加可控的机制
        - 命名空间分割了全局命名空间，其中每个命名空间都是一个独立的作用域
        - 通过在某个命名空间中定义库的名字，库的作者以及用户就可以有效避免全局名字固有的限制
- 全部语法速览
    ```
    namespace ns_name { declarations }                                        (1)
    inline namespace ns_name { declarations }                                 (2)
    namespace { declarations }                                                (3)
    ns_name::name                                                             (4)
    using namespace ns_name;                                                  (5)
    using ns_name::name;                                                      (6)
    namespace name = qualified-namespace;                                     (7)
    namespace ns_name::inline(since C++20)(optional) name { declarations }    (8)    (since C++17)
    ```
    1. *具名命名空间* 定义
    2. *内联命名空间* 定义
        - 命名空间`ns_name`内的声明在其外层命名空间中亦可见
    3. *无名命名空间* 定义
        - 其成员的作用域从声明点开始，到翻译单元结尾为止
        - 其成员具有 *内部链接* 
    4. *命名空间名* （还有 *类名* ）可以出现在 *域运算符左侧* ，作为 *限定名字查找* 的一部分
    5. *`using`指示* （`using`-directive）
        - 从这条`using`指示开始、到其作用域结束为止，进行 *非限定名字查找* 时，来自命名空间`ns_name`的任何名字均可见
            - 如同它们被声明于同时含有这条`using`指示以及`ns_name`这两者的更外一层的命名空间作用域中
    6. *`using`声明* （`using`-declaration）
        - 从这条`using`声明开始、到其作用域结束为止，进行 *非限定名字查找* 时，来自命名空间`ns_name`的名字`name`可见
            - 如同它被声明于包含这条`using`声明的相同的类作用域、块作用域或命名空间作用域中
    7. *命名空间别名* （namespace alias）定义
    8. *嵌套命名空间定义* （nested namespace definition） `(since C++17)`
        - `namespace A::B::C { ... }`等价于`namespace A { namespace B { namespace C { ... } } }`
        - *嵌套`inline`*  `(since C++20)`
            - `inline`可出现于除第一个之外的任何一个命名空间名之前
            - `namespace A::B::inline C { ... }`等价于`namespace A::B { inline namespace C { ... } }`
            - `namespace A::inline B::C {}`等价于`namespace A { inline namespace B { namespace C {} } }` 
- 命名空间定义
    - 定义规则
        - 命名空间**不能**定义于 *函数内部* 
        - 命名空间作用域后面**无需**分号
    - 每个命名空间都是一个作用域
    - 命名空间可以是不连续的
        - 命名空间的一部分成员的作用是定义类，以及声明作为类接口的函数及对象，则这些成员应该置于头文件中，这些头文件将被包含在使用了这些成员的文件中
        - 命名空间成员的定义部分则置于另外的源文件中
        - 定义多个类型不相关的命名空间应该使用单独的文件分别表示每个类型（或关联类型构成的集合）
    - 通常**不**把`#include`放在命名空间 *内部* 
        - 这样做就是把头文件的所有名字定义成该命名空间的成员
        - 例如，如下代码就是把命名空间`std`嵌套在命名空间`cplusplus_primer`中，程序将报错
        ```
        namespace cplusplus_primer
        {
        #include <string>
        }
        ```
    - 命名空间与模板特例化
        - 模板特例化 *必须* 定义在 *原始模板所属的命名空间中* 
        - 和其他命名空间名字类似，只要我们在命名空间中声明了特例化，就可以在命名空间外定义它了
        ```
        // we must declare the specialization as a member of std
        namespace std 
        {
        template <> 
        struct hash<Sales_data>;
        }
        
        // having added the declaration for the specialization to std
        // we can define the specialization outside the std namespace
        template <> 
        struct std::hash<Sales_data>
        {
            size_t operator()(const Sales_data & s) const
            { 
                return hash<string>()(s.bookNo) ^ hash<unsigned>()(s.units_sold) ^ hash<double>()(s.revenue); 
            }
            
            // other members as before
        };
        ```
    - *全局命名空间* （global namespace）
        - *全局作用域* （即，在所有类、函数以及命名空间之外）中定义的名字归属于 *全局命名空间* 
            - 全局命名空间以 *隐式* 的方式声明，并存在于所有程序之中
            - 全局作用域的定义被 *隐式* 地添加于全局命名空间中
        - *域运算符* `::`同样可以作用于全局作用域的成员
            - 因为全局作用域是隐式的，所以它并没有名字
            - 全局命名空间中的名字也可以用如下语法显式地访问
            ```
            // in global scope
            // i.e. out of scope of any class, function or namespace
            member_name      // (global_ns)::member_name
            
            // in any scope
            ::member_name    // (global_ns)::member_name
            ```
    - *嵌套命名空间* （nested namespaces）
        - 指定义在其他命名空间内部的命名空间
        ```
        namespace cplusplus_primer 
        {
            // first nested namespace: defines the Query portion of the library
            namespace QueryLib 
            {
                class Query { /* ... */ };
                Query operator&(const Query &, const Query &);
                // ...
            }
            
            // second nested namespace: defines the Sales_data portion of the library
            namespace Bookstore 
            {
                class Quote { /* ... */ };
                class Disc_quote : public Quote { /* ... */ };
                // ...
            }
        }
        ```
        - 嵌套的命名空间同时是一个嵌套的作用域，嵌套在外层命名空间的作用域中
        - 内层的名字将覆盖外层的同名实体
        - 内层的名字只在内层直接可见；外层想访问内层的名字，必须加 [*限定标识符*](https://en.cppreference.com/w/cpp/language/identifiers#Qualified_identifiers)（Qualified identifiers）
        ```
        // outside of namespace QueryLib
        cplusplus_primer::QueryLib::Query
        ```
    - *内联命名空间* （inline namespaces）
        - 在`namespace`前加关键字`inline`可以将命名空间定义为 *内联的* 
            - 关键字`inline`必须出现在命名空间 *第一次定义* 的地方
            - 后续打开命名空间时，可以加`inline`，也可以不加
        - 内联命名空间中的名字在 *外层* 命名空间中 *直接可见* ，不必加 *限定标识符*
        ```
        inline namespace FifthEd 
        {
        // namespace for the code from the Primer Fifth Edition
        }
        
        namespace FifthEd  // implicitly inline
        { 
        class Query_base { /* ... * /};
        // other Query-related declarations
        }
        ```
        - 当应用程序的代码在版本更新时发生了改变，常常会用到内联命名空间
            - 例如，把本书当前版本代码都放在一个内联命名空间中，而旧版本的代码都放在非内联的命名空间中
            ```
            namespace FourthEd 
            {
            class Item_base { /* ... */};
            class Query_base { /* ... */};
            // other code from the Fourth Edition
            }
            ```
            - 命名空间`cplusplus_primer`将同时使用这两个命名空间
            - 假定每个命名空间都定义在同名的头文件中，则可以把命名空间`cplusplus_primer`定义为如下格式
            ```
            namespace cplusplus_primer 
            {
            #include "FifthEd.h"
            #include "FourthEd.h"
            }
            ```
            - 由于`FifthEd`是内联的，所以形如`cplusplus_primer::`的代码就可以直接访问到`FifthEd`的成员
            - 而如果想要使用旧版本的代码，则必须像其他嵌套命名空间一样，加上完整的限定说明符，例如
            ```
            cplusplus_primer::FourthEd::Query_base
            ```
    - *无名命名空间* （unnamed namespaces）
        - 指关键字`namespace`后紧跟花括号括起来的一系列声明语句
        - *无名命名空间* 中定义的变量自动具有 *内部链接* 和 *静态存储期* 
            - 即它们的使用方法和性质就像在外层命名空间（例如全局命名空间）中定义的`static`变量一样
        - 无名命名空间可以不连续，但**不能**跨越多个文件
            - 每个文件定义自己的无名命名空间，如果两个文件都含有无名命名空间，则这两个命名空间**无关**
            - 这两个无名命名空间中可以定义相同的名字，且这些定义表示的是不同的实体
            - 如果一个 *头文件* 包含了未命名的命名空间，则该命名空间中定义的名字将在每个包含了该头文件的文件中对应不同的实体
        - 无名命名空间中的名字可以直接使用，且**不能**使用域运算符
        - 无名命名空间中定义的名字的作用域与该命名空间所在的作用域相同
            - 如果无名命名空间定义在文件最外层作用域中，则该命名空间中的名字一定要与全局作用域中的名字有所区别
            ```
            int i; // global declaration for i
            
            namespace 
            {
            int i;
            }
            
            // ambiguous: defined globally and in an unnested, unnamed namespace
            i = 10;
            ```
            - 其他情况下，无名命名空间中的成员都属于正确的程序实体
        - 和所有命名空间类似，一个无名命名空间也能嵌套在其他命名空间中
            - 此时，无名命名空间中的成员可以通过外层命名空间的名字来访问
            ```
            namespace local 
            {
                namespace 
                {
                    int i;
                }
            }
            
            // ok: i defined in a nested unnamed namespace is distinct from global i
            local::i = 42;
            ```
        - 无名命名空间取代 *文件内`static`声明* 
            - `C`程序中将名字声明为`static`使其对且只对这整个文件有效
            - `C++`程序应当使用无名命名空间取代`C`风格的文件内`static`声明
- 使用命名空间成员
    - *命名空间别名* （namespace alias）
        - 通过 *命名空间别名* 简化很长的名字
        - 格式
        ```
        namespace ns_name = name_of_a_much_longer_ns;
        ```
        - **不能**在命名空间还未定义时就声明别名
        - 一个命名空间可以有好几个别名，所有别名都与原先的命名空间等价
    - *`using`声明* （`using`-declaration）
        - 格式
        ```
        using ns_name::member_name;
        ```
        - 一次只引入某命名空间的一个 *名字*
            - 从这条`using`声明开始、到其作用域结束为止，进行 *非限定名字查找* 时，来自命名空间`ns_name`的名字`name`可见
                - 如同它被声明于包含这条`using`声明的相同的类作用域、块作用域或命名空间作用域中
                - 有效作用域结束后，想访问这一变量，就必须使用完整的 *限定标识符* ，进行 *限定名字查找*
        - 可以出现于 *全局作用域* 、 *局部作用域* 、 *命名空间作用域* 以及 *类作用域* 中 
            - 在 *类作用域* 中，`using`声明 *只能指向基类成员* 
        - `using`声明声明的是一个 *名字* ，而**不是**函数
        ```
        using NS::print(int);  // error: cannot specify a parameter list
        using NS::print;       // ok: using declarations specify names only
        ```
    - *`using`指示* （`using`-directive）
        - 格式
        ```
        using namespace ns_name;
        ```
        - 一次引入一整个命名空间
            - 从这条`using`指示开始、到其作用域结束为止，进行 *非限定名字查找* 时，来自命名空间`ns_name`的任何名字均可见
                - 如同它们被声明于同时含有这条`using`指示以及`ns_name`这两者的 *更外一层* 的命名空间作用域中
        - 位置限制
            - 可以出现于 *全局作用域* 、 *局部作用域* 、 *命名空间作用域* 中 
            - **不可以** 出现在 *类作用域* 中
        - `using`指示**不等于**一大堆`using`声明
            1. `using`声明只将一个名字的作用域提升至其本身所在的作用域；而`using`指示提升的作用域是其本身所在作用域 *的更外一层* 
                - 这是因为`using`指示提升的是一整个命名空间的全部成员
                - 而命名空间中通常包含一些**不能**出现在局部作用域中的定义
                - 因此`using`指示一般被看做是出现在更外一层的作用域中
            2. `using`声明如果造成二义性，会立即产生编译错误；而`using`指示如果造成二义性，并**不会立即**导致 *编译错误* 
                - 对于`using`指示引入的二义性，只有到程序真正直接使用了有二义性的名字时，才会报错
                - 这一点会造成隐藏问题，难以调试
        - `例18.1`
            - 代码
            ```
            // namespace A and function f are defined at global scope
            namespace A
            {
            int i = 1;
            int j = 2;
            }

            int i = 3;
            int j = 4;

            void f1()
            {
                using namespace A;                      // injects A::i and A::j into GLOBAL scope
                std::cout << i * j << std::endl;        // error: reference to i and j is ambiguous
            }

            void f2()
            {
                using namespace A;                      // injects A::i and A::j into GLOBAL scope
                std::cout << A::i * A::j << std::endl;  // ok: uses A::i and A::j
                std::cout << ::i * ::j << std::endl;    // ok: uses ::i and ::j
            }

            void f3()
            {
                using A::i;                             // injects A::i into f3's scope
                using A::j;                             // injects A::j into f3's scope
                std::cout << i * j << std::endl;        // ok: uses A::i and A::j
                std::cout << A::i * A::j << std::endl;  // ok: uses A::i and A::j
                std::cout << ::i * ::j << std::endl;    // ok: uses ::i and ::j
            }
            ```
            - `f1`和`f2`
                - `using namespace A`会将`A::i`和`A::j`提升至 *全局作用域* ，而**不是**函数作用域
                - 因此在`f1`和`f2`中，全局命名空间中就同时有了 *两个* `i`和`j`，但着**并不立即**造成冲突
                - `f1`中直接使用`i`和`j`，造成了二义性冲突
                - 而`f2`中并未使用有二义性的名字，所以程序相安无事，炸弹被隐藏起来了
            - `f3`
                - `using A::i`和`using A::j`只将`A::i`和`A::j`提升至函数作用域
                - 因此直接使用`i`和`j`时，函数作用域覆盖了外层作用域的同名实体，没有问题
        - 头文件与`using`声明或指示
            - 头文件如果在其顶层作用域中含有`using`指示或`using`声明，则会将名字注入到所有包含了该头文件的文件中
            - 通常情况下，头文件应该只负责定义接口部分的名字，而不定义实现部分的名字
            - 因此，*头文件* 最多只能在它的 *函数或命名空间内* 使用 *`using`指示* 或 *`using`声明* 
        - 应尽量**避免**使用 *`using`指示* 
            - 造成的坏处
                - 会造成 *全局命名空间污染* 
                - `using`指示引起的二义性错误直到使用到二义性的名字时才会被发现，此时距`using`指示的引入可能已经很久了，导致程序难以调试
            - 相比于`using`指示，对命名空间的每个成员分别使用`using`声明效果更好
                - 这么做可以减少注入到命名空间中的名字的数量
                - `using`声明引起的二义性错误 *在声明处就能发现* ，更利于程序调试
            - `using`指示也不是一无是处，例如在命名空间本身的实现文件中，就可以使用`using`指示
- 类、命名空间与作用域
    - 对命名空间内部名字的查找遵循常规的查找规则
        - 即由内向外依次查找每个外层作用域
            - 外层作用域也可能是一个或多个嵌套的命名空间
            - 直到最外层的全局命名空间查找过程终止
        - 只有位于开放的块中、且在使用点之前声明的名字才被考虑
        ```
        namespace A 
        {
            int i;
            
            namespace B 
            {
                int i;         // hides A::i within B
                int j;
                
                int f1()
                {
                    int j;     // j is local to f1 and hides A::B::j
                    return i;  // returns B::i
                }
            } // namespace B is closed and names in it are no longer visible
                
            int f2()
            {
                return j;      // error: j is not defined
            }
            
            int j = i;         // initialized from A::i
        }
        ```
        - 对于命名空间中的 *类* 来说，常规的查找规则仍然适用
            - 当成员函数使用某个名字时
                - 首先在该成员中进行查找
                - 然后在类中查找（包括基类）
                - 接着在外层作用域中查找
                    - 这时一个或几个外层作用域可能就是命名空间
            - 除了类内部出现的成员函数定义之外，总是向上查找作用域
            - 可以从函数的 *限定标识符* 推断出名字查找是检查作用域的次序
                - 限定名以 *相反* 次序指出被查找的作用域
            ```
            namespace A 
            {
            int i;
            int k;
            
            class C1 
            {
            public:
                C1() : i(0), j(0) { }      // ok: initializes C1::i and C1::j
                
                int f1() { return k; }     // returns A::k
                int f2() { return h; }     // error: h is not defined
                int f3();
                
            private:
                int i;                     // hides A::i within C1
                int j;
            };
            
            int h = i;                     // initialized from A::i
            }
            
            // member f3 is defined outside class C1 and outside namespace A
            int A::C1::f3() { return h; }  // ok: returns A::h
            ```
        - 实参相关的查找与类类型形参
            - 给 *函数* 传递 *类类型实参* 时，在常规的名字查找之后，还会额外查找 *实参类及其基类所属的命名空间* 
                - 这一规则对 *类的引用* 或 *类的指针* 类型的 *函数实参* 同样有效
            - 例如如下程序
            ```
            int a = 1;
            std::cout << a;
            ```
            - `std::cout << a;`其实是调用的是`std::operator<<(std::cout, a);`
            - 由于其接受了类类型实参，因此名字查找时会在普通查找无果后额外查找`namespace std`
                - 这就是我们不用特别加`std::`或`using std::operator<<;`的原因
            - 查找规则的这个例外允许概念上作为类接口一部分的非成员函数无需单独的`using`声明就能被程序使用
                - 假如此例外不存在，就不得不
                    - 为`<<`单独提供一个`using`声明`using std::operator<<;`，或
                    - 显式调用函数：`std::operator<<(std::cout, a);`
                - 然而躲得过初一躲不过十五，该来的总会来的 
                    - *`std::chrono_literals`字面量* 的操作符，例如`operator""h`就只能提供`using`声明或者显式调用了
                    ```
                    {
                        using std::chrono_literals::operator""min;
                        std::chrono::minutes _1_hour {60min};
                        std::cout << "1 hour is " << _1_hour.count() << " minute(s)" << std::endl;
                    }

                    {
                        std::chrono::minutes _1_hour = std::chrono_literals::operator""min<'6', '0'>();
                        std::cout << "1 hour is " << _1_hour.count() << " minute(s)" << std::endl;
                    }

                    ```
                    - *`std::complex`字面量* 的操作符也没好到哪儿去
        - 名字查找与`std::move`和`std::forward`
            - 通常情况下，如果程序中定义了一个标准库中已有的名字，则将出现以下两种情况中的一种
                - 根据一般的重载规则确定某次调用应该执行函数的某个版本
                - 应用程序根本不会执行标准库版本
            - `std::move`和`std::forward`都是标准库模板函数，都接受一个模板类型参数的右值引用类型形参
                - 复习一下`gcc`对这俩货的实现
                ```
                /// <type_traits>
                /// remove_reference
                template <typename T> struct remove_reference       { typedef T type; };
                template <typename T> struct remove_reference<T &>  { typedef T type; };
                template <typename T> struct remove_reference<T &&> { typedef T type; };
                
                /// <move.h>
                /// @brief     Convert a value to an rvalue.
                /// @param  t  A thing of arbitrary type.
                /// @return    The parameter cast to an rvalue-reference to allow moving it.
                template <typename T>
                constexpr typename std::remove_reference<T>::type &&
                move(T && t) noexcept
                { 
                    return static_cast<typename std::remove_reference<T>::type &&>(t); 
                }
                
                /// <move.h>
                /// @brief     Forward an lvalue.
                /// @return    The parameter cast to the specified type.
                /// This function is used to implement "perfect forwarding".
                template <typename T>
                constexpr T &&
                forward(typename std::remove_reference<T>::type & t) noexcept
                { 
                    return static_cast<T &&>(t); 
                }
                
                /// <move.h>
                /// @brief     Forward an rvalue.
                /// @return    The parameter cast to the specified type.
                /// This function is used to implement "perfect forwarding".
                template <typename T>
                constexpr T &&
                forward(typename std::remove_reference<T>::type && t) noexcept
                {
                    static_assert(!std::is_lvalue_reference<T>::value, 
                                  "template argument substituting T is an lvalue reference type");
                    return static_cast<T &&>(t);
                }
                ```
                - *模板类型参数的右值引用类型的形参* 事实上可以匹配 *任何类型以及任何值类别的实参* 
                    - 如果我们的应用程序也定义了一个接受单一形参的`move`函数，则不管该形参是什么类型，都会与`std::move`冲突
                      - 如下写法将`std::move`提升到了当前作用域。 *重载确定/重载决议* (overload resolution) 流程首先进行 *名字查找* ，会在当前作用域中同时找到`std::move`和自定义的`move`两个 *候选函数* ，而它们也都是 *可行函数* ，一同进入最佳匹配决策环节。由于自行定义的一般是更特化的版本，根据模板函数重载匹配规则 *匹配特化完犊子* ，一旦形参和实参的类型恰好又是 *精确匹配* ，最终编译器就会调用自行编写的版本。即亦，此时如下写法就将永远**无法**调用到`std::move`
                        ```
                        using std::move;
                        move(...);
                        ```
                        - 这也就解释了为什么调用`std::move`时一定要用如下写法
                        ```
                        std::move(...);
                        ```
                        - `forward`也一样
                        - 友元声明与实参相关的查找
                            - 当类声明了一个友元时，该友元声明并未使友元本身可见
                                - 即使这个友元声明是个定义，也还是不可见
                            - 然而，一个另外的未声明的类或函数如果第一次出现在友元声明中，则我们认为它是上一层命名空间的成员
                            - 这条规则与实参相关的查找规则结合在一起将产生意想不到的效果
                                - 如以下代码
                                ```
                                namespace A
                                {
                                class C
                                {
                                public:
                                    // two friends; neither is declared apart from a friend declaration
                                    // these functions implicitly are members of namespace A
                                    friend void f(const C &) {}  // can be found when being called outside A
                                                                 // by argument-dependent lookup

                                    friend void f2() {}          // won't be found when being called outside A
                                                                 // unless otherwise declared
                                };
                                }
                                ```
                                - 此时`f`和`f2`都是`namespace A`的成员，即使`f`**不**存在其它声明，我们也能在`namespace A`之外通过实参相关的查找规则调用`f`
                                ```
                                int main()
                                {
                                    A::C o;
                    
                                    f(o);                        // ok: finds A::f through the friend declaration in A::C
                                    f2();                        // error: f2 not declared
                    
                                    A::f(o);                     // error: A::f not declared
                                    A::f2();                     // error: A::f2 not declared
                                }
                                ```
                                - 因为`f`接受一个 *类类型实参* ，所以名字查找流程会额外搜寻`C`所属的`namespace A`，就会找到`f`的 *隐式声明* 
                                    - 由`C`中对`f`的友元声明，编译器认为`f`是`namespace A`的成员，因而产生一个隐式声明
                                - 相反，因为`f2`没有 *类类型实参* ，因此它就不能被找到了
                                - 加入如下声明令`f`和`f2`可见，则可让上面代码直接没事
                                ```
                                namespace A
                                {
                                void f(const C &);
                                voif f2();
                                }
                                ```
- 重载与命名空间
    - 与实参相关的查找与重载
        - 给 *函数* 传递 *类类型实参* 时，在常规的名字查找之后，还会额外查找 *实参类及其基类所属的命名空间* 
        - 这会影响 *候选函数集* 的确定
        - 我们将在每个实参类及其基类所在的命名空间中搜寻候选函数，这些命名空间中所有同名函数都被加入候选函数集，即使其中某些函数在调用语句处不可见
        ```
        namespace NS 
        {
        class Quote { /* ... */ };
        void display(const Quote &) { /* ... */ }
        }
        
        // Bulk_item's base class is declared in namespace NS
        class Bulk_item : public NS::Quote { /* ... */ };
        
        int main() 
        {
            Bulk_item book1;
            display(book1);  // NS::display is not visible here; still, it is added to candidate function set
            return 0;
        }
        ```
    - 重载与`using`声明
        - `using`声明声明的是一个 *名字* ，而**不是**函数
        ```
        using NS::print(int);  // error: cannot specify a parameter list
        using NS::print;       // ok: using declarations specify names only
        ```
        - 使用`using`声明将把该函数在该命名空间中的所有版本都注入到当前作用域中
        ```
        namespace A
        {
        void f(int) {}
        void f(const std::string &) {}
        }

        namespace B
        {
        void f(const std::vector<int> &) {}
        }
        
        void f(const std::list<int> &) {}
        
        int main
        {
            using A::f;
        
            f(1);                           // ok. calls A::f(1)
            f("hehe");                      // ok. calls A::f(hehe)
            
            f(std::list<int> {0, 1, 2});    // error: no matching function call to f
                                            // current scope: main
                                            // void f(const std::list<int> &) is in global scope
                                            // name-lookup finds name f in current scope and stops
                                            // thus can't find void f(const std::list<int> &)
                                            
            f(std::vector<int> {0, 1, 2});  // error: no matching function call to f
                                            // B::f is not visible at all
        }
        ```
        - 一个`using`声明囊括了重载函数在该命名空间内的所有版本以确保不违反命名空间的接口
            - 如果库的作者为了某项任务提供了好几个不同的函数、并允许用户选择性地忽略重载函数中的一部分但不是全部，将可能导致意想不到的程序行为
        - 一个`using`声明引入的函数将重载该声明语句所属作用域中已有的其他同名函数
        - 如果`using`声明出现在局部作用域中，则引入的名字将隐藏外部作用域的相关声明
        - 如果`using`声明所在的作用域中已经有一个函数与新引入的函数同名且形参列表相同，则该`using`声明将引发 *重定义错误* 
        - `using`声明将为引入的名字添加额外的重载实例，并最终扩充 *候选函数集* 的规模
    - 重载与`using`指示
        - `using`指示将命名空间的全部成员提升到当前作用域的上一层作用域中
        - 如果此命名空间的某个函数与该命名空间所属的作用域的函数重名，则此命名空间的函数将被添加到 *重载集合* 中
        ```
        namespace libs_R_us 
        {
        extern void print(int);
        extern void print(double);
        }
        
        // ordinary declaration
        void print(const std::string &);
        
        // this using directive adds names to the candidate set for calls to print:
        using namespace libs_R_us;
        
        // the candidates for calls to print at this point in the program are:
        // print(int) from libs_R_us
        // print(double) from libs_R_us
        // print(const std::string &) declared explicitly
        
        void fooBar(int ival)
        {
            print("Value: ");  // calls global print(const string &)
            print(ival);       // calls libs_R_us::print(int)
        }
        ```
        - 与`using`声明不同的是，对于`using`指示来说，引入一个与已有函数形参列表完全相同的函数并**不会**产生错误
            - 此时，只要我们指明调用的是命名空间中的版本还是当前作用域的版本即可
    - 跨越多个`using`指示的重载
        - 如果存在多个`using`指示，则来自每个命名空间的名字都会成为候选函数集的一部分
        ```
        namespace AW 
        {
        int print(int);
        }
        
        namespace Primer 
        {
        double print(double);
        }
        
        // using directives create an overload set of functions from different namespaces
        using namespace AW;
        using namespace Primer;
        
        long double print(long double);
        
        int main() 
        {
            print(1);    // calls AW::print(int)
            print(3.1);  // calls Primer::print(double)
            return 0;
        }
        ``` 
        - 在全局作用域中，函数`print`的 *重载集合* 包括
            - `AW::print(int)`
            - `Primer::print(double)`
            - `::print(long double)`
        - 在 *主函数* 中，当前作用域没有候选函数，然后在上一层全局作用域中找到了如上 *候选函数集* 

#### 多重继承与虚继承（multiple inheritance and virtual inheritance）

- *多重继承* （multiple inheritance）
    - *多重继承* 是指从多个直接基类中产生派生类的能力，多重派生类继承了所有父类的属性
        - 在派生类的派生列表中可以有多个基类
            - 每个基类包含一个可选的访问说明符
            - 如果访问说明符被忽略了，则`class`默认`private`，`struct`默认`public`
            - 和单重继承一样，多重继承的派生列表也只能包含已经定义过的类，而且这些类不能是`final`的
            - 直接基类的个数不受限，但同一个基类只能出现一次
        - 体系举例
            - 抽象基类`ZooAnimal`，保存动物园中动物共有的信息，提供公共接口
            - 其他辅助类：负责封装不同的抽象，例如`Panda`由`Bear`和`Endangered`共同派生得来
        ```
        class Bear : public ZooAnimal { /* ... */ };
        class Panda : public Bear, public Endangered { /* ... */ };
        ```
    - 派生类构造函数初始化所有基类
        - 构造一个派生类的对象将同时构造并初始化它的所有基类子对象
        - 与单重继承一样，
            - 多重继承的 *派生类的构造函数* 需要 *自行* 在初始化列表中调用基类构造函数 *初始化基类部分* 
            - 如果没有显式调用基类的构造函数，则此基类对应部分将被 *默认初始化* ，产生 *未定义的值* 
            - 基类的构造顺序与 *派生列表中基类的出现顺序* 保持一致，与初始化列表中基类的顺序**无关**
        ```
        // explicitly initialize both base classes
        Panda::Panda(std::string name, bool onExhibit)
                : Bear(name, onExhibit, "Panda"), 
                  Endangered(Endangered::critical) 
        { 
        
        }
        
        // implicitly uses the Bear default constructor to initialize the Bear subobject
        Panda::Panda()
                : Endangered(Endangered::critical) 
        { 
        
        }
        ```
        - 例如，`ZooAnimal`是整个体系的最终基类，`Bear`是`Panda`的直接基类，`ZooAnimal`是`Bear`的基类。因此一个`Panda`对象将按如下次序进行初始化
            - 首先初始化`ZooAnimal`
            - 接下来初始化`Panda`的第一个直接基类`Bear`
            - 然后初始化`Panda`的第二个直接基类`Endangered`
            - 最后初始化`Panda`
    - 继承的构造函数与多重继承
        - 允许派生类从一个或几个基类中继承构造函数；但如果从多个基类中继承了相同的构造函数（即形参列表完全相同），则将产生错误
        ```
        struct Base1 
        {
            Base1() = default;
            Base1(const std::string &);
            Base1(std::shared_ptr<int>);
        };
        
        struct Base2 
        {
            Base2() = default;
            Base2(const std::string &);
            Base2(int);
        };
        
        // error: D1 attempts to inherit D1::D1 (const string &) from both base classes
        struct D1: public Base1, public Base2 
        {
            using Base1::Base1;  // inherit constructors from Base1
            using Base2::Base2;  // inherit constructors from Base2
        };
        ```
        - 如果一个类从它的多个基类中继承了相同的构造函数，则这个类必须为该构造函数定义它自己的版本
        ```
        struct D2: public Base1, public Base2 
        {
            using Base1::Base1;  // inherit constructors from Base1
            using Base2::Base2;  // inherit constructors from Base2
            
            // D2 must define its own constructor that takes a string
            D2(const string & s) : Base1(s), Base2(s) {}
            D2() = default;      // needed once D2 defines its own constructor
        }
        ```
    - 析构函数与多重继承
        - 和往常一样
            - 派生类的 *析构函数* 只需要负责清除派生类本身分配的资源
            - 派生类的成员会被自动销毁
            - 基类由编译器自动调用基类析构函数进行销毁
        - 析构函数调用顺序正好与构造函数相反
            - 对于`Panda`，构造函数调用顺序为`ZooAnimal -> Bear -> Endangered -> Panda`
            - 析构函数调用顺序则为`~Panda -> ~Endangered -> ~Bear -> ~ZooAnimal`
    - 多重继承的派生类拷贝与移动操作
        - 与单重继承一样
            - 多重继承的派生类如果定义了自己的拷贝、移动构造函数或赋值运算符，则必须 *自行负责* 拷贝、移动或赋值 *完整的对象* 
            - 只有当派生类使用的是 *合成版本* 的拷贝、移动或赋值成员时，才会自动对其基类部分执行这些操作
            - 在合成的拷贝控制成员中，每个基类分别使用自己的对应成员隐式地完成构造、赋值或销毁等工作
            - 例如
                - 假设`Panda`使用合成版本的成员`ling_ling`的初始化过程
                ```
                Panda ying_yang("ying_yang");
                Panda ling_ling = ying_yang;  // uses the copy constructor
                ```
                - 将调用`Bear`的拷贝构造函数，后者又在执行自己的拷贝任务之前先调用`ZooAnimal`的拷贝构造函数
                - 一旦`ling_ling`的`Bear`部分构造完成，接着就会调用`Endangered`的拷贝构造函数来创建对象的相应部分
                - 最后，执行`Panda`的拷贝构造函数
            - 合成的移动构造函数的工作机制与之类似
            - 合成的拷贝赋值运算符
                - 首先赋值`Bear`部分（并通过`Bear`赋值`ZooAnimal`部分）
                - 然后赋值`Endangered`部分
                - 最后赋值`Panda`部分
            - 合成的移动赋值运算符与之类似
- 类型转换与多个基类
    - 和单重继承时一样
        - 多重继承的派生类的指针或引用一样可以被隐式转换成可访问基类的指针或引用，且不会导致实际指向的对象被截断
        - 可以令某个可访问基类的指针或引用直接指向一个派生类对象
        - 例如，可以将`ZooAnimal`、`Bear`或`Endangered`类型的指针或引用绑定到`Panda`对象上
        ```
        // operations that take references to base classes of type Panda
        void print(const Bear &);
        void highlight(const Endangered &);
        ostream & operator<<(ostream &, const ZooAnimal &);
        
        Panda ying_yang("ying_yang");
        print(ying_yang);                     // passes Panda to a reference to Bear
        highlight(ying_yang);                 // passes Panda to a reference to Endangered
        std::cout << ying_yang << std::endl;  // passes Panda to a reference to ZooAnimal
        ```
        - 编译器**不会**在派生类向基类的几种转换中进行比较和选择，因为在它看来转换到任意的一种基类都一样好
        - 例如，如下调用会引发 *二义性错误*
        ```
        void print(const Bear &);
        void print(const Endangered &);
        
        Panda ying_yang("ying_yang");
        print(ying_yang);                     // error: ambiguous
        ```
    - 基于指针类型或引用类型的查找
        - 和单重继承时一样
            - 多重继承的基类的指针或引用的 *静态类型* 决定了哪些成员可见
            - 当然，对于虚函数是动态绑定的，这一点不会变
- 多重继承下的类作用域
    - 单重继承
        - 派生类的作用域嵌套于直接基类和间接基类的作用域中
        - 查找过程沿着继承体系自底向上进行，直到找到所需的名字
        - 派生类的名字将隐藏基类的同名成员
    - 多重继承
        - 相同的查找过程在所有直接基类中同步进行
        - 如果名字在多个基类中都被找到，则此使用将引发 *二义性错误* 
            - 继承含有先沟通名字的多个基类本身是合法的
            - 此时只需要显式指明 *限定标识符* 
- *虚继承* （virtual inheritance）
    - 尽管派生类的派生列表中，同一个基类最多只能出现一次，但实际上派生类可以多次继承同一个基类
        - 可以通过两个直接基类分别继承同一个间接基类
        - 也可以直接继承某个基类，然后通过另一个基类再一次间接继承该类
        - 标准库中的例子：`std::iostream`继承自`std::istream`和`std::ostream`，后两者又都继承了`std::ios_base`，也就是说`std::iostream`继承了`std::ios_base`两次
    - 默认情况下，派生类中含有继承链上每个类对应的子部分
        - 如果某个类在派生过程中出现了多次，则派生类中将包含该类的多个子对象
        - 这显然不是希望看到的
            - 至少对于`std::iostream`，一个流对象肯定希望在同一个缓冲区中进行读写操作，也会要求条件状态能同时反映输入和输出操作的情况
            - 假如在`std::iostream`对象中真的包含了`std::ios_base`的两份拷贝，则上述的共享行为就无法实现了
    - *虚继承* 机制用于解决上述问题
        - 虚继承的目的是，令某个类作出声明，承诺愿意 *共享它的基类* 
        - 其中，共享的基类子对象被称作 *虚基类* （virtual base class）
        - 在这种机制下，不论虚基类在继承体系中出现了多少次，在派生类中都只包含唯一一个共享的虚基类子对象
    - 必须在虚派生的真实 *需求出现前* 就已经 *完成虚派生* 的操作
        - 例如，如果定义`std::iostream`时才出现了对虚派生的需求，但是如果`std::istream`和`std::ostream`**不是**从`std::ios_base`虚派生来的，那就没救了
        - 在实际的编程过程中，位于中间层次的基类将其继承声明为虚继承一般不会带来什么问题
            - 通常情况下，使用虚继承的类层次是由一个人或一个项目组一次性设计完成的
            - 对于一个独立开发的类来说，很少需要基类中某一个是虚基类，况且新基类的开发者也无法改变已有的继承体系
    - 虚派生只影响从制定了虚基类的派生类中进一步派生出的类，**不会**影响派生类本身
    - 使用虚基类
        - 指定虚基类的方式时在派生列表中添加关键字`virtual`
            - `public`和`virtual`的相互顺序随意
            ```
            // the order of the keywords public and virtual is not significant
            class Raccoon : public virtual ZooAnimal { /* ... */ };
            class Bear : virtual public ZooAnimal { /* ... */ };
            ```
        - `virtual`说明符表达了一种愿望，即在后续的派生类当中共享虚基类的同一份实例
            - 至于什么样的类能够作为虚基类，并没有特殊规定
        - 如果某个类指定了虚基类，则该类的派生仍按常规方式进行
            - 例如下面`Panda`类 *只有* `ZooAnimal`一个虚基类部分
            ```
            class Panda : public Bear, public Raccoon, public Endangered 
            {
                // ...
            };
            ```
    - 支持向基类的常规类型转换
        - 不论基类是不是虚基类，派生类对象都能被可访问基类的指针或引用操作
        - 例如，如下从`Panda`向基类类型的转换都是合法的
        ```
        void dance(const Bear &);
        void rummage(const Raccoon &);
        ostream & operator<<(ostream &, const ZooAnimal &);
        
        Panda ying_yang;
        dance(ying_yang);        // ok: passes Panda object as a Bear
        rummage(ying_yang);      // ok: passes Panda object as a Raccoon
        std::cout << ying_yang;  // ok: passes Panda object as a ZooAnimal
        ```
    - 虚基类成员的可见性
        - 因为在每个共享的虚基类中只有唯一一个共享的子对象，所以该基类的成员可以被 *直接访问* ，并且不会产生二义性
        - 此外，如果虚基类的成员 *只被一条派生路径* *覆盖* ，则我们仍然 *可以直接访问* 这个被覆盖的成员
        - 但是如果成员被 *多于一个基类* *覆盖* ，则一般情况下派生类 *必须* 为该成员 *自定义一个新的* 版本
        - 例如
            - 假定
                - `class B`定义了一个成员`B::x`
                - `class D1`和`class D2`均继承自`B`
                - `class D`多重继承自`D1`和`D2`
            - 则，在`D`的作用域中，`x`通过`D`的两个基类都是可见的
            - 此时，如果我们通过`D`的实例使用`x`，则有如下 *三种* 可能性
                - 如果在`D1`和`D2`中都**没有**`x`的定义，则`x`将被解析为`B`的成员，此时**不**存在二义性
                - 如果在`D1`和`D2`中有且只有一个有`x`的定义，则同样**没有**二义性，派生类的`x`比共享虚基类`B`的`x`优先级更高
                - 如果在`D1`和`D2`中都有`x`的定义，则此时直接访问`x`存在二义性
        - 与非虚的多重继承体系一样，解决这种二义性问题的最好方法就是在派生类中为成员自定义一个新的实例
- 构造函数与虚继承
    - 在 *虚派生* 中，虚基类是由 *最低层派生类* 独自初始化的
        - 例如创建`Panda`对象时，`Panda`的构造函数 *独自控制* `ZooAnimal`的初始化过程
    - 这一规则的原因
        - 假设以普通规则处理初始化任务
        - 则虚基类会被派生路径上的多个类重复初始化
        - 此例中，`ZooAnimal`将被`Bear`和`Raccoon`两个类重复初始化
    - *每个虚派生类* 都 *必须在构造函数中初始化它的虚基类* 
        - 这是因为继承体系中每个类都可能在某个时刻成为 *最底层派生类* 
        - 例如之前的动物继承体系，创建`Bear`或`Raccoon`对象时，它就已经位于派生的最低层，因此`Bear`或`Raccoon`的构造函数将直接初始化其`ZooAnimal`部分
        ```
        Bear::Bear(std::string name, bool onExhibit)
                : ZooAnimal(name, onExhibit, "Bear") 
        {
        
        }
        
        Raccoon::Raccoon(std::string name, bool onExhibit)
                : ZooAnimal(name, onExhibit, "Raccoon") 
        {
        
        }
        ```
        - 而当创建一个`Panda`对象时，`Panda`位于派生的最低层，因此由它负责初始化共享的`ZooAnimal`虚基类部分
            - 即使`ZooAnimal`**不是**`Panda`的直接基类，`Panda`的构造函数也可以初始化`ZooAnimal`
        ```
        Panda::Panda(std::string name, bool onExhibit)
                : ZooAnimal(name, onExhibit, "Panda"),
                  Bear(name, onExhibit),
                  Raccoon(name, onExhibit),
                  Endangered(Endangered::critical),
                  sleeping flag(false) 
        {
        
        }
        ```
    - 虚继承的对象的构造方式
        - 首先使用提供给最低层派生类构造函数的初始值初始化该对象的虚基类子部分，然后按照直接基类在派生列表中出现的顺序依次对该直接基类进行初始化
            - 虚基类总是先于非虚基类被构造，与它们在继承体系中的位置和次序**无关**
        - 例如创建`Panda`对象时
            - 首先使用`Panda`的构造函数初始值列表中提供的初始值构造虚基类`ZooAnimal`部分
                - 如果`Panda`**没有**显式地初始化`ZooAnimal`基类，则`ZooAnimal`的默认构造函数将被调用
                - 如果`ZooAnimal`又**没有**默认构造函数，则程序报错
            - 接下来构造`Bear`部分
            - 然后构造`Raccoon`部分
            - 然后构造`Endangered`部分
            - 最后构造`Panda`自己的部分
    - 构造函数与析构函数的次序
        - 一个类可以有许多个虚基类
            - 此时这些虚的子对象会按照它们出现在派生列表中的顺序依次被初始化
            - 之后再正常初始化非虚子对象
        - 例如
        ```
        class Character { /* ... */ };
        class BookCharacter : public Character { /* ... */ };
        class ToyAnimal { /* ... */ };
        class TeddyBear : public BookCharacter, public Bear, public virtual ToyAnimal { /* ... */ };
        ```
        - 编译器按照直接基类的声明顺序对其依次进行检查，以确定其中是否含有虚基类
        - 如果有，则先构造虚基类，然后按照声明逐一构造其它非虚基类
        - 因此，想要创建一个`TeddyBear`对象，需要按照如下次序调用这些构造函数
        ```
        ZooAnimal();      // Bear's virtual base class
        ToyAnimal();      // direct virtual base class
        Character();      // indirect base class of first nonvirtual base class
        BookCharacter();  // first direct nonvirtual base class
        Bear();           // second direct nonvirtual base class
        TeddyBear();      // most derived class
        ```
        - 合成的拷贝和移动构造函数按照完全相同的顺序执行
        - 合成的拷贝赋值运算符中的成员也按照该顺序赋值
        - 和往常一样，对象的销毁顺序与构造顺序正好相反
            - 即，首先销毁`TeddyBear`部分，最后销毁`ZooAnimal`部分






### 🌱 [Chap 19] 特殊工具与技术

#### 控制内存分配（Controlling Memory Allocation）

- 重载`new`和`delete`表达式
    - `new`和`delete`表达式的工作机理
        - 使用一条`new`表达式时
        ```
        // new expressions
        std::string * sp = new std::string("a value");  // allocate and initialize a string
        std::string * arr = new std::string[10];        // allocate ten default-initialized strings
        ```
        - 实际上执行了 *三步* 操作
            1. `new`表达式调用标准库函数`operator new`或`operator new[]`
                - 该函数分配一块足够大的、原始的、未命名的内存空间，用于存储特定类型的对象或对象的数组
            2. 编译器运行相应的构造函数以构造这些对象，并为其传入初始值
            3. 对象被分配了空间并构造完成，返回一个指向该对象的指针
        - 当我们使用一条`delete`表达式时
        ```
        delete sp;      // destroy *sp and free the memory to which sp points
        delete [] arr;  // destroy the elements in the array and free the memory
        ```
        - 实际执行了 *两步* 操作
            1. 对`sp`所指对象或者`arr`所指的数组中的元素执行对应的析构函数
            2. 编译器调用名为`operator delete`或`operator delete[]`的标准库函数释放内存空间
    - 如果应用程序希望控制内存分配的过程，则其需要定义自己的`operator new`和`operator delete`函数
        - 即使在标准库中已经存在这两个函数的定义，我们仍旧可以定义自己的版本
        - 编译器**不会**对这种重复的定义提出异议；相反，编译器将使用我们自定义的版本 *替换* 标准库定义的版本
        - 当自定义了全局的`operator new`和`operator delete`函数后，我们就负担起了控制动态内存分配的职责
        - 这两个函数必须是正确的，因为它们是程序处理过程中至关重要的一部分
    - 应用程序可以在 *全局作用域* 定义`operator new`和`operator delete`函数，也可以将它们声明为 *成员函数* 
        - 当编译器发现一条`new`或`delete`表达式后，将在程序中查找可用的`operator`函数
        - 如果被分配或释放的对象是 *类类型* ，则编译器首先在类及其基类的作用域中查找
        - 此时如果该类含有`operator new`或`operator delete`成员函数，则相应的表达式将调用这些成员
        - 否则，编译器在全局作用域查找匹配的函数
        - 此时如果编译器找到了用户自定义的版本，则使用该版本执行`new`表达式或`delete`表达式
        - 如果没找到，则使用标准库定义的版本
    - 我们可以使用 *域运算符* `::`令`new`表达式或`delete`表达式忽略定义在类中的函数，直接执行全局作用域中的版本
        - 例如，`::new`只在全局作用域中查找匹配的`operator new`函数
        - `::delete`与之类似
    - *`operator new`接口* 和 *`operator delete`接口* 
        - 标准库定义了`operator new`函数和`operator delete`函数的如下重载版本
        ```
        // replaceable (de)allocation functions 
        void * operator new     (size_t);
        void * operator new[]   (size_t);
        void   operator delete  (void *) noexcept;
        void   operator delete[](void *) noexcept;
        void   operator delete  (void *, size_t) noexcept;  (since C++14)
        void   operator delete[](void *, size_t) noexcept;  (since C++14)
        
        // replaceable non-throwing (de)allocation functions 
        void * operator new     (size_t, std::nothrow_t &) noexcept;
        void * operator new[]   (size_t, std::nothrow_t &) noexcept;
        void   operator delete  (void *, std::nothrow_t &) noexcept;
        void   operator delete[](void *, std::nothrow_t &) noexcept;
        
        // non-allocating placement allocation functions
        void * operator new     (size_t, void *) noexcept;
        void * operator new[]   (size_t, void *) noexcept;
        void   operator delete  (void *, void *);	
        void   operator delete[](void *, void *);
            
        // user-defined placement (de)allocation functions
        void * operator new     (size_t, args ...);
        void * operator new[]   (size_t, args ...);
        void   operator delete  (void *, args ...);	
        void   operator delete[](void *, args ...);
        
        // class-specific (de)allocation functions
        void * T::operator new     (size_t);
        void * T::operator new[]   (size_t);           
        void   T::operator delete  (void *);
        void   T::operator delete[](void *);
        void   T::operator delete  (void *, size_t);
        void   T::operator delete[](void *, size_t);
        
        // class-specific placement (de)allocation functions
        void * T::operator new     (size_t, args ...);	
        void * T::operator new[]   (size_t, args ...);	
        void   T::operator delete  (void *, args ...);
        void   T::operator delete[](void *, args ...);
        ```
        - 其中`std::throw_t`是一个空的`struct`，还有一个常量实例`std::nothrow`
        ```
        // if allocation fails, new returns a null pointer
        int * p1 = new int;                 // if allocation fails, new throws std::bad_alloc
        int * p2 = new (std::nothrow) int;  // if allocation fails, new returns a null pointer
        ```
        - 应用程序可以自定义上面函数版本中的任意一个
            - 前提是自定义的版本必须位于 *全局作用域* 或者 *类作用域* 中
            - 当我们将上述运算符定义为类的成员时，它们是 *隐式静态* 的
                - 我们无需显式声明`static`，当然这么做也不会引发错误
                - 因为`operator new`用在对象构造之前，而`operator delete`用在对象析构之后，所以这两个成员必须是 *静态* 的，而且他们**不能**操纵类的任何数据成员
            - 对于`operator new`或`operator new[]`来说，它的返回类型必须是`void *`。第一个形参必须是`size_t`类型，且**不能**有默认实参
                - 当我们动态分配单个对象时，使用`operator new`；动态分配数组时，使用`operator new[]`
                - 当编译器调用`operator new`时，把存储 *指定类型对象* 所需的字节数传给`size_t`形参
                - 当编译器调用`operator new`时，把存储 *该数组所有对象* 所需的字节数传给`size_t`形参
                - 如果我们想要自定义`operator new`函数，则可以提供 *额外形参* 
                    - 此时，用到这些自定义函数的`new`表达式必须使用 *定位形式* （placement version），将实参传递给新增的形参
                    - 尽管在一般情况下我们可以自定义具有任何形参的`operator new`，但下面这个函数不论如何**不允许**被重载，只能由标准库使用
                    ```
                    void * operator new(size_t, void *);  // this version may NOT be redefined
                    ```
            - 对于`operator delete`函数或者`operator delete[]`函数来说，它们的返回类型必须是`void`，第一个形参必须是`void *`类型
                - 执行一条`delete`表达式将调用相应的`operator`函数，并用指向待释放内存的指针来初始化`void *`形参
                - 当我们将`operator delete`函数或者`operator delete[]`函数定义成类的成员时，该函数可以包含另外一个类型为`size_t`的形参
                    - 此时，该形参的初始值时第一个形参所指对象的字节数
                    - `size_t`形参用于删除继承体系中的对象
                        - 如果基类有一个 *虚析构函数* ，则传递给`operator delete`的字节数将因待删除指针所指对象的动态类型不同而有所区别
                        - 而且，实际运行的`operator delete`函数版本也由对象的动态类型决定
    - `术语`：`new`表达式和`operator new`函数
        - 标准库函数`operator new`和`operator delete`的名字容易让人误解
        - 和其他`operator`函数**不同**，这两个函数并**没有** *重载* `new`运算符和`delete`运算符
        - 实际上，我们根本无法自定义`new`表达式或`delete`表达式的行为
            - 一条`new`表达式的执行过程是固定的，总是先调用`operator new`函数以获取内存空间，然后在得到的内存空间中构造对象
            - 一条`delete`表达式的执行过程也是固定的，总是先销毁对象，再调用`operator delete`函数释放对象所占的空间
        - 我们提供新的`operator new`和`operator delete`函数的目的在于改变内存的分配方式
        - 但不管怎样，我们都不能改变`new`运算符和`delete`运算符的基本含义
    - `malloc`函数与`free`函数
        - 继承自`C`语言
        - 编写`operator new`和`operator delete`的一种简单方式
        ```
        void * operator new(size_t size) 
        {
            if (void * mem = malloc(size))
            {
                return mem;
            }
            else
            {
                throw std::bad_alloc();
            }  
        }
        
        void operator delete(void * mem) noexcept 
        { 
            free(mem); 
        }
        ```
- *定位`new`表达式* （placement `new` expression）
    - 调用格式
    ```
    new (place_address) type
    new (place_address) type (initializers)
    new (place_address) type [size]
    new (place_address) type [size] { braced initializer list }
    ```
    - 当只传入一个指针类型的实参时，定位`new`表达式构造对象但是不分配内存
        - 此时 *定位`new`* 调用`operator new(size_t, void *)`
        - 这是一个我们**无法**自定义的`operator new`版本
        - 该函数**不**分配任何内存，它只是简单地返回指针实参
        - 然后由`new`表达式负责在指定的地址初始化对象以完成整个工作
        - 事实上，定位`new`允许我们在一个特定的、预先分配的内存地址上构造对象
    - 传给定位`new`的指针甚至不必须指向动态内存
    - 显式的析构函数调用
        - 例子
        ```
        std::string * sp = new std::string("a value");  // allocate and initialize a string
        sp->~string();
        ```
        - 调用析构函数会销毁对象，但**不会**释放内存

#### [运行时类型识别](https://en.cppreference.com/w/cpp/types)（Run-time Type Identification，`RTTI`）

- 概述
    - *运行时类型实别* 的功能由如下 *两个* 运算符实现
        - [`dynamic_cast`](https://en.cppreference.com/w/cpp/language/dynamic_cast)
        - [`typeid`](https://en.cppreference.com/w/cpp/language/typeid)
    - 当我们把这两个运算符用于某种类型的 *指针或引用* ，并且该类型含有 *虚函数* 时，运算符将使用指针或引用所绑定对象的 *动态类型* 
    - 这两个运算符特别适用于以下情况
        - 我们想使用基类对象的指针或引用执行某个派生类操作并且该操作**不是**虚函数
        - 一般来说，只要有可能，我们都应该尽量使用虚函数
            - 当操作被定义成虚函数时，编译器将根据对象的动态类型自动地选择正确的函数版本
        - 然而，并非任何时候都能定义一个虚函数
        - 假设我们无法使用虚函数，则可以使用`RTTI`运算符
        - 另一方面，与虚成员函数相比，使用`RTTI`运算符蕴含着更多的潜在风险
            - 程序员必须清楚地知道转换的目标类型，并且必须检查类型转换是否被成功执行
    - 使用`RTTI`运算符必须倍加小心。在可能的情况下，最好定义虚函数而非直接接管类型管理的责任
- [`dynamic_cast`](https://en.cppreference.com/w/cpp/language/dynamic_cast)
- [`typeid`](https://en.cppreference.com/w/cpp/language/typeid)
    - 使用形式
    ```
    typeid(e)
    ```
    - 其中，`e`可以是任意类型的表达式或类型的名字
    - `typeid`返回值类型为`const std::type_info &`，或`std::type_info`的公有派生类型的常引用
        - 顶层`const`将被忽略
        - 对于引用，返回值代表其所绑定到的对象的类型
        - 对于数组或函数，**不会**执行向指针的隐式类型转换，例如`int a[10]`，则`typeid(a)`是数组类型而**不是**指针
    - 当且仅当`e`是 *多态类类型的引用左值或解引用指针* 时，`typeid`返回`e`实际指向的对象的 *动态类型* ；否则，返回其本身的 *静态类型* 
        - 解引用指针的结果的类型是 *左值引用* ，其值类别一定是 *左值* 
        - 指针本身也是对象，如果不解引用指针，则判断的就是指针本身而**不是**其指向的对象了
            - 此时当然就只是此指针的静态类型了，一般不是我们所希望的
    - `注意`
        - `typeid`是否需要执行运行时检查决定了表达式 *是否会被求值* 
            - 只有当类型是多态的（含有虚函数）时，编译器才会对表达式求值
            - 反之，则`typeid`返回表达式的静态类型
                - 编译器**不需**对表达式求值就能知道表达式的静态类型
        - 如果表达式的动态类型和静态类型不同，则必须在运行时对表达式求值以确定返回的类型
            - 适用于`typeid(*ptr)`的情况
            - 如果`ptr`的静态类型不含有虚函数，则`ptr`不必是有效指针
            - 否则，`*ptr`将在运行时被求值，此时`ptr`就必须是一个有效的指针了
            - 如果`ptr`是空指针或野指针，则`typeid(*ptr)`将抛出`std::bad_typeid`异常
    ```
    Derived * dp = new Derived();
    Base * bp = dp;                // both pointers point to a Derived object
    
    // compare the type of two objects at run time
    if (typeid(*bp) == typeid(*dp)) 
    {
        // bp and dp point to objects of the same type
    }
    
    // test whether the run-time type is a specific type
    if (typeid(*bp) == typeid(Derived)) 
    {
        // bp actually points to a Derived
    }
    
    // test always fails: the type of bp is pointer to Base
    if (typeid(bp) == typeid(Derived)) 
    {
        // code never executed
    }
    ```
- 使用`RTTI`的一个例子：动态类型敏感的对象判等
    - 类的层次关系
    ```
    class Base 
    {
    public:
        friend bool operator==(const Base &, const Base &);
    
        // interface members for Base
        
    protected:
        virtual bool equal(const Base &) const;
        
        // data and other implementation members of Base
    };
    
    class Derived : public Base 
    {
    public:
        // other interface members for Derived
        
    protected:
        bool equal(const Base &) const;
        
        // data and other implementation members of Derived
    };
    ```
    - 动态类型敏感的`operator ==`
    ```
    bool operator==(const Base & lhs, const Base & rhs)
    {
        // returns false if typeids are different; otherwise makes a virtual call to equal
        return typeid(lhs) == typeid(rhs) && lhs.equal(rhs);
    }
    ```
    - 虚`equal`函数
    ```
    bool Derived::equal(const Base & rhs) const
    {
        // as this function is called only by operator== and only when typeid(lhs) == typeid(rhs)
        // we know the types are equal, so the cast won't throw
        auto r = dynamic_cast<const Derived &>(rhs);
        
        // do the work to compare two Derived objects and return the result
        return ...
    }
    ```
- [`std::type_info`](https://en.cppreference.com/w/cpp/types/type_info)
    - `C++`标准只规定此类必须定义于头文件`<typeinfo>`、并具有如下接口，其他内容均 *由实现定义* 
        - `t1 == t2`：如果`t1`和`t2`表示同一种类型，则返回`true`
        - `t1 != t2`：如果`t1`和`t2`表示不同种类型，则返回`true`
        - `t.name()`：返回一个`C`风格字符串，表示类型名字的可打印形式，具体内容 *由实现定义* 
        - `t1.before(t2)`：返回一个`bool`值，表示`t1`是否位于`t2` *之前* 。 *之前* 具体是什么 *由实现定义* 
    - 除此之外，因为`std::type_info`一般作为基类出现，所以它还应该提供一个公有的虚析构函数。当编译器希望提供额外的类型信息时，通常在`std::type_info`的派生类中完成
    - `std::type_info`的默认构造函数、拷贝构造函数、移动构造函数和赋值运算符均是`= delete;`的
        - 因此，无法定义或拷贝`std::type_info`类的对象，也不能对其赋值
        - 唯一获取途径就是`typeid`运算符
    - `demangle`：`gcc`的实现中，`std::type_info::name`是经过特殊编码的，需要 *还原* （demangle）才能使人可读

#### 枚举（enumeration）

- 将一组常量组织在一起
- 和类一样，每个枚举类型分别定义了一种新的类型
- 枚举属于 *字面值常量* 类型
- `C++`包含 *两种* 枚举
    - *限定作用域枚举* （scoped enumeration）
        - 使用关键字`enum class`或`enum struct`
        - 随后是枚举名字
        - 然后是用 *花括号* 括起来的 *枚举成员列表* （enumerator list）
        - 最后是一个 *分号* 
        ```
        enum class open_modes 
        {
            input, 
            output, 
            append
        };
        ```
    - *非限定作用域枚举* （unscoped enumeration）
        - 省略掉`class`或`struct`
        - 枚举类型的名字是可选的
        ```
        // unscoped enumeration
        enum color 
        {
            red, 
            yellow, 
            green
        }; 
        
        // unnamed, unscoped enum
        enum 
        {
            floatPrec         = 6, 
            doublePrec        = 10, 
            double_doublePrec = 10
        };
        ```
        - 如果`enum`是 *匿名* 的，则只能在定义时定义它的对象
        - 和类的定义类似，我们需要在`enum`定义的右侧花括号和最后的分号之间提供逗号分隔的声明列表
- *枚举成员* （enumerator）
    - 枚举成员的作用域
        - 在 *限定作用域枚举* 的 *枚举成员* 的名字遵循常规的作用域准则，并且在枚举类型的作用域外是不可访问的
        - 与之相反，在 *非限定作用域枚举* 的 *枚举成员* 的作用域与 *枚举本身的作用域* 相同
        ```
        enum color {red, yellow, green};          // unscoped enumeration
        enum stoplight {red, yellow, green};      // error: redefines enumerators
        enum class peppers {red, yellow, green};  // ok: enumerators are hidden
        
        color eyes = green;                       // ok: enumerators are in scope for an unscoped enumeration
        peppers p = green;                        // error: enumerators from peppers are not in scope
                                                  // color::green is in scope but has the wrong type
                                                  
        color hair = color::red;                  // ok: we can explicitly access the enumerators
        peppers p2 = peppers::red;                // ok: using red from peppers
        ```
    - 枚举值
        - 默认情况下，枚举值从`0`开始，依次比上一项的值多`1`
        - 也能为一个或几个枚举成员指定专门的值
            - 此时未指定专门值的枚举成员的值遵循默认规则
        - 枚举值**不一定**唯一
        ```
        enum TypeSize
        {
            TEST_0,               // 0
            TEST_1,               // 1
            BOOL         = 1,     // 1
            CHAR         = 1,     // 1
            WCHAR_T      = 4,     // 4
            INT          = 4,     // 4
            FLOAT        = 4,     // 4
            LONG         = 8,     // 8
            LONG_LONG    = 8,     // 8
            DOUBLE       = 8,     // 8
            LONG_DOUBLE  = 16,    // 16
            TEST_17               // 17
        };
        ```
    - 枚举成员是`const`
        - 也就是说，每个枚举成员本身就是一条常量表达式
        - 因此初始化枚举成员的值必须是 *常量表达式* 
        - 可以在任何需要常量表达式的地方使用枚举成员
        ```
        constexpr intTypes charbits = intTypes::charTyp;
        ```
        - 类似地，也可以将一个`enum`作为`switch`语句的条件，将枚举值作为`case`标签
        - 出于同样的原因，还可以将枚举类型作为非类型模板形参使用，或在类中初始化枚举类型的静态数据成员
    - *非限定作用域枚举* 的对象或枚举成员可以被 *隐式转换成`int`* 
        - 因此我们可以在任何需要`int`的地方使用它们
        - *限定作用域枚举* 是**没有**这种好事或坏事的
        ```
        int i = color::red;    // ok: unscoped enumerator implicitly converted to int
        int j = peppers::red;  // error: scoped enumerations are NOT implicitly converted
        ```
- 和类一样，枚举也定义新的类型
    - 只要`enum`有名字，就能定义并初始化该类型的成员
    - 要想初始化`enum`对象或者为`enum`对象赋值， *必须* 使用该类型的一个 *枚举成员* 或者该类型的 *另一个对象* 
        - 即使这是能自动转`int`的 *非限定作用域枚举* 也一样
    ```
    TypeSize ts = 16;                   // error: 16 is not of type TypeSize
    TypeSize ts = LONG_DOUBLE;          // ok: input is an enumerator of TypeSize
    
    open_modes om = 2;                  // error: 2 is not of type open_modes
    open_modes om = open_modes::input;  // ok: input is an enumerator of open_modes
    ```
- 指定`enum`的大小
    - 尽管每个`enum`都定义了唯一的类型，但实际上`enum`是由某种 *整数类型* 表示的
    - 可以在`enum`的名字后面加上 *冒号* `:`以及我们想在该`enum`中使用的类型
    ```
    enum intValues : unsigned long long 
    {
        charTyp      = 255, 
        shortTyp     = 65535, 
        intTyp       = 65535,
        longTyp      = 4294967295UL,
        long_longTyp = 18446744073709551615ULL
    };
    ```
    - 如果我们**没有**显式指定`enum`的潜在类型，则默认情况下
        - *非限定作用域枚举* **不**存在默认类型，只知道其足够容纳枚举值
        - *限定作用域枚举* 默认`int`
    - 一旦指定了潜在类型（包括对 *限定作用域枚举* 的 *隐式指定* ），则一旦枚举成员的值爆表，则将报 *编译错误* 
- 枚举类型的前置声明
    - 前置声明枚举类型 *必须* （显式或隐式）指定其大小
    ```
    // forward declaration of unscoped enum named intValues
    enum intValues : unsigned long long;  // unscoped, must specify a type
    enum class open_modes;                // scoped enums can use int by default
    ```
    - 和其他声明一样，`enum`的声明和定义必须匹配
        - 该`enum`的所有声明和定义中成员的大小必须一致
        - **不能**在同一个上下文中先声明一个 *非限定作用域枚举* ，再声明一个同名的 *限定作用域枚举* 
        ```
        // error: declarations and definition must agree whether the enum is scoped or unscoped
        enum class intValues;
        enum intValues;                   // error: intValues previously declared as scoped enum
        enum intValues : long;            // error: intValues previously declared as int
        ```
- 形参匹配与枚举类型
    - 即使某个整型值恰好和枚举成员的值相等，它也**不能**作为`enum`类型形参的实参传入
        - 要想初始化`enum`对象或者为`enum`对象赋值， *必须* 使用该类型的一个 *枚举成员* 或者该类型的 *另一个对象* 
    ```
    // unscoped enumeration; the underlying type is machine dependent
    enum Tokens 
    {
        INLINE = 128, 
        VIRTUAL = 129
    };
    
    void ff(Tokens);
    void ff(int);
    
    int main() 
    {
        Tokens curTok = INLINE;
        
        ff(128);                 // exactly matches ff(int)
        ff(INLINE);              // exactly matches ff(Tokens)
        ff(curTok);              // exactly matches ff(Tokens)
        
        return 0;
    }
    ```
    - 尽管**不能**直接将整型值传给`enum`形参，但可以将 *非限定作用域枚举* 类型的对象或枚举成员传给整形对象
        - 此时`enum`的值 *提升* 成`int`或更大的类型，实际提升效果由枚举类型的潜在类型定义
        - 特别地：枚举类型永远**不会**被提升成`unsigned char`，即使枚举值可以用`unsigned char`存储也不行
    ```
    void newf(unsigned char);
    void newf(int);
    
    unsigned char uc = VIRTUAL;
    
    newf(VIRTUAL);               // calls newf(int)
    newf(uc);                    // calls newf(unsigned char)
    ```

#### 类成员指针（Pointer to Class Member）

- *成员指针函数表* （Pointer-to-Member Function Tables）
    - 对于普通函数指针和成员函数指针来说，一种常见的用法是将其存入一个 *函数表* 当中
        - 如果一个类含有几个相同类型的成员，则这样一张表可以帮助我们从这些成员中选择一个
    ```
    class Screen 
    {
    public:
        // other interface and implementation members as before
        
        // Action is a pointer that can be assigned any of the cursor movement members
        using Action = Screen & (Screen::*)();
        
        // specify which direction to move
        enum Directions 
        { 
            HOME, 
            FORWARD, 
            BACK, 
            UP, 
            DOWN 
        };
    
    public:
        Screen & move(Directions cm)
        {
            // run the element indexed by cm on this object
            return (this->*Menu[cm])();  // Menu[cm] points to a member function
        }
    
    private:
        // cursor movement functions
        Screen & home();     
        Screen & forward();
        Screen & back();
        Screen & up();
        Screen & down();
        
    private: 
        // function table
        static Action Menu[];
    };
    
    Screen::Action Screen::Menu[] = 
    { 
        &Screen::home,
        &Screen::forward,
        &Screen::back,
        &Screen::up,
        &Screen::down,
    };
    
    Screen myScreen;
    myScreen.move(Screen::HOME);  // invokes myScreen.home
    myScreen.move(Screen::DOWN);  // invokes myScreen.down
    ```
- 将成员函数用作 *可调用对象* 
    - 成员指针**不是** *可调用对象* 
        - 要想通过一个指向成员函数的指针进行函数调用，必须首先利用 *成员指针访问运算符* `.*` `->*`将该指针绑定到特定对象上
        - 因此，与普通函数指针不同，成员指针不是可调用对象，**不**支持函数调用运算符
    - 使用[`std::function`](https://en.cppreference.com/w/cpp/utility/functional/function)生成可调用对象
        - 示例
        ```
        std::vector<std::string> svec {"", "s1", "s2"};
        std::function<bool (const std::string &)> fcn = &std::string::empty;
        std::find_if(svec.begin(), svec.end(), fcn);
        ```
        - 当一个`std::function`对象中封装了 *成员函数指针* 时
            - `std::function`类将使用正确的 *成员指针访问运算符* 来执行函数调用
            - 通常情况下，执行成员函数的对象被传给隐式的`this`形参
            - 即：`std::function<ret, (obj, ...)> fcn = &Class::fun;`，则
                - `fcn(obj, ...)`将实际调用`obj.*fun(...)`
                - `fcn(ptr, ...)`将实际调用`ptr->*fun(...)`
            - 例如如下代码
            ```
            struct S
            {
                S() = default;
                S(int _a) : a(_a) {}

                void add(int b) const { std::cout << a + b << '\n'; }

                int a {0};
            };
            
            std::vector<S> sv {0, 1, 2};
            std::function<void (const S &, int)> fcn = &S::add;

            for (auto it = sv.begin(), end = sv.end(); it!= end; ++it)
            {
                fcn(*it, 10);
            }
            ```
            - 再例如，对于上面的`std::find_if`例子
                - 标准库算法中本来含有类似于如下形式的代码
                ```
                // assuming it is the iterator inside find_if, so *it is an object in the given range
                if (fcn(*it))      // assuming fcn is the name of the callable inside find_if
                ```
                - 其中，`std::function`将使用正确的 *成员指针访问运算符* ，即：将函数调用转化为了如下形式
                ```
                // assuming it is the iterator inside find_if, so *it is an object in the given range
                if (((*it).*p)())  // assuming p is the pointer to member function inside fcn
                ```
        - `std::function`必须明确知道可调用对象的调用签名，包括返回值以及接受的参数
            - 在此例中，就是对象是否是以 *引用* 或 *指针* 的形式传入的
                - 对于`const`成员函数的指针， 则传入对象最好设为 *常量引用* 或 *常量指针* 
                - 因为常量对象能够调用`const`成员函数，但却无法调用非`const`的
                - 传入对象设为常量可以保证常量对象和非常量对象都能使用这个`std::function`
            - 对于前面的例子，解引用`std::vector<T>::iterator`迭代器的结果将是`T &`
            - 对于下面的例子，由于`std::vector`中保存的是`std::string &`，就必须定义`std::function`接受指针
            ```
            std::vector<std::string *> pvec;
            std::function<bool (const std::string *)> fp = &std::string::empty;
            
            // fp takes a pointer to string and uses the ->* to call empty
            std::find_if(pvec.begin(), pvec.end(), fp);
            ```

#### 嵌套类（Nested Class）

- 概述
    - *嵌套类* 是指定义在 *另一个类内部* 的类，常用于定义作为实现部分的类，又称 *嵌套类型* （Nested Type）
    - 嵌套类是一个独立的类，与外部的类基本没有关系
        - 特别是，外层类的对象和嵌套类的对象是相互独立的
        - 嵌套类中的对象**不**包含任何外层类定义的成员，反之亦然
    - 嵌套类的名字在外层类作用域中可见，在外层类作用域之外**不**可见
        - 和其他嵌套的名字一样，嵌套类的名字**不会**和别的作用域中的同一个名字冲突
    - 嵌套类中成员的种类与非嵌套类是一样的
        - 与其他类类似，嵌套类也使用 *访问限定符* 来控制外接对其成员的访问权限
        - 外层类对嵌套类的成员**没有**特殊的访问权限，反之亦然
    - 嵌套类在其外层类中定义了一个 *类型成员* 
        - 和其他成员类似，该类型的访问权限是由外层类决定的
        - 位于外层类`public`部分的嵌套类实际上定义了一种可以随处访问的类型
        - 位于外层类`protected`部分的嵌套类定义的类型只能被外层类自己、它的友元以及它的派生类访问
        - 位于外层类`private`部分的嵌套类实际上定类型只能被外层类自己以及它的友元访问
- 声明嵌套类
    - 嵌套类和成员函数一样
        - 必须声明在类的内部
        - 但可以定义在类内或类外
    - 如需在外层类外定义嵌套类，则仍需先在外层类内声明此嵌套类，而后再使用
    ```
    class TextQuery 
    {
    public:
        // nested class to be defined later
        class QueryResult;  
        
        // other members as in § 12.3.2
    };
    ```
- 在外层类之外定义嵌套类
    - 在嵌套类在其外层类之外完成真正的定义之前，它都是一个 *不完全类型* 
    - 在外层类之外定义嵌套类时，必须以外层类的名字限定嵌套类的名字
    ```
    // we're defining the QueryResult class that is a member of class TextQuery
    class TextQuery::QueryResult 
    {
    public:
        // in class scope, we don't have to qualify the name of the QueryResult parameters
        friend std::ostream & print(std::ostream &, const QueryResult &);
    
        // no need to define QueryResult::line_no; a nested class can use a member
        // of its enclosing class without needing to qualify the member's name
        QueryResult(std::string, std::shared_ptr<std::set<line_no>>, std::shared_ptr<std::vector<std::string>>);
        
        // other members as in § 12.3.2
    };
    ```
    - 定义嵌套类的成员
    ```
    // defining the member named QueryResult for the class named QueryResult
    // that is nested inside the class TextQuery
    TextQuery::QueryResult::QueryResult(std::string s, 
                                        std::shared_ptr<std::set<line_no>> p, 
                                        std::shared_ptr<std::vector<std::string>> f)
            : sought(s), lines(p), file(f) 
    { 

    }
    ```
    - 嵌套类 *静态成员* 的定义
    ```
    // defines an int static member of QueryResult
    // which is a class nested inside TextQuery
    int TextQuery::QueryResult::static_mem = 1024;
    ```
- 嵌套类作用域中的名字查找
    - 嵌套类的作用域嵌套在了其外层类的作用域之中
    - 名字查找的一般规则在此同样适用

#### 联合体（union）

- *联合体* 是一种特殊的类
    - 可以有多个数据成员，但任意时刻只有一个数据成员可以有值
        - 给`union`的某个成员赋值之后，该`union`的其他成员就变成 *未定义* 的状态了
        - 分配给一个`union`对象的存储空间至少要容纳它的最大的数据成员
    - 和其他类一样，一个`union`也定义了一种新的类型
    - `union`的数据成员
        - `union`**不能**含有 *引用类型* 的成员
        - 含有构造函数或析构函数的类类型也可以作为`union`成员类型
    - `union`的成员函数
        - `union`可以定义包括构造函数和析构函数在内的成员函数
        - `union`中**不能**含有 *虚函数*
            - 这是因为`union`既不能继承自其他类，也不能作为基类使用
    - `union`的访问控制
        - `union`可以为其成员指定`public`、`protected`或`private`等访问控制标记
        - `union`成员默认 *公有* ，和`struct`一样
- 定义`union`
```
// objects of type Token have a single member, which could be of any of the listed types
union Token 
{
    // members are public by default
    char   cval;
    int    ival;
    double dval;
};
```
- 使用`union`
    - `union`的名字是一个 *类型名* 
    - 和其他内置类型一样，默认情况下`union`是未初始化的
    - 我们可以像显式地初始化聚合类一样使用一对花括号内的初始值显式地初始化一个`union`
    ```
    Token first_token = {'a'};  // initializes the cval member
    Token last_token;           // uninitialized Token object
    Token * pt = new Token;     // pointer to an uninitialized Token object
    ```
    - 如果提供了初始值，则该初始值被用于初始化 *第一个* 成员
    - 因此，`first_token`的初始化过程实际上是给`cval`成员赋了一个初值
    - 我们使用通用的 *成员访问运算符* `.` `->`访问一个`union`对象的成员
    ```
    last_token.cval = 'z';
    pt->ival = 42;
    ```
    - 为`union`的一个数据成员赋值会令其他数据成员变成 *未定义状态* 
        - 因此，当我们使用`union`时，必须清楚地知道当前存储在`union`中的值到底是什么类型
        - 如果我们使用错误的数据成员或者为错误的数据成员赋值，则程序可能崩溃或出现异常行为，具体的情况根据成员的类型而有所不同
- *匿名`union`* （anonymous union）
    - 未命名的`union`，并且在右花括号和分号之间没有任何声明
    - 一旦我们定义了一个匿名`union`，编译器就自动地为该`union`创建一个匿名对象
    ```
    union             // anonymous union
    { 
        char   cval;
        int    ival;
        double dval;
    };                // defines an unnamed object, whose members we can access directly
    
    cval = 'c';       // assigns a new value to the unnamed, anonymous union object
    ival = 42;        // that object now holds the value 42
    ```
    - 在匿名`union`**不能**包含 *受保护* 的成员或 *私有* 成员，也**不能**定义 *成员函数* 
- 使用类和 *判别式* （discriminant）管理含有 *类类型成员* 的`union`
    - `union`可以含有定义了构造函数或拷贝控制成员的类类型成员
        - 但此时编译器会将`union`的对应拷贝控制成员合成为 *删除的* 
        - 这类成员在构造和析构时必须显式调用构造函数或析构函数
    - 通常把含有类成员的`union`内嵌在另一个类中
        - 这个类可以管理并控制与`union`的类类型有关的状态转换
        - 例如，为匿名`union`添加`std::string`成员，并将此匿名`union`作为`Token`类的成员
    - 为了追踪`union`中到底存储了什么类型的值，通常定义一个独立的对象，该对象被称作`union`的 *判别式* 
        - 我们可以使用判别式辨认`union`存储的值
        - 为了保持`union`与其判别式同步，我们将判别式也作为`Token`的成员
        - 我们的类将定义一个 *枚举类型* 的成员来追踪其`union`成员的状态
```
class Token
{
public:
    // copy control needed because our class has a union with a string member
    // defining the move constructor and move-assignment operator is left as an exercise
    Token() : tok{INT}, ival{0}
    {

    }

    Token(const Token & t) : tok(t.tok)
    {
        copyUnion(t);
    }

    // if the union holds a string, we must destroy it
    ~Token()
    {
        if (tok == STR)
        {
            using std::string;
            sval.~string();
        }
    }

    // copy assignment
    Token & operator=(const Token & t)
    {
        // if this object holds a string and t doesn't, we have to free the old string
        if (tok == STR && t.tok != STR)
        {
            using std::string;
            sval.~string();
        }

        if (tok == STR && t.tok == STR)
        {
            sval = t.sval;   // no need to construct a new string
        }
        else
        {
            copyUnion(t);    // will construct a string if t.tok is STR
        }

        tok = t.tok;
        return *this;
    }

    // assignment operators to set the differing members of the union
    Token & operator=(const std::string & s)
    {
        if (tok == STR)      // if we already hold a string, just do an assignment
        {
            sval = s;
        }
        else                 // otherwise construct a string
        {
            new (&sval) std::string(s);
        }

        tok = STR;           // update the discriminant
        return *this;
    }

    Token & operator=(char c)
    {
        if (tok == STR)      // if we have a string, free it
        {
            using std::string;
            sval.~string();
        }

        cval = c;            // assign to the appropriate member
        tok = CHAR;          // update the discriminant
        return *this;
    }

    Token & operator=(int i)
    {
        if (tok == STR)      // if we have a string, free it
        {
            using std::string;
            sval.~string();
        }

        ival = i;            // assign to the appropriate member
        tok = INT;           // update the discriminant
        return *this;
    }

    Token & operator=(double d)
    {
        if (tok == STR)      // if we have a string, free it
        {
            using std::string;
            sval.~string();
        }

        dval = d;            // assign to the appropriate member
        tok = DBL;           // update the discriminant
        return *this;
    }

private:
    // check the discriminant and copy the union member as appropriate
    void copyUnion(const Token & t)
    {
        switch (t.tok)
        {
        case Token::INT:
            ival = t.ival;
            break;
        case Token::CHAR:
            cval = t.cval;
            break;
        case Token::DBL:
            dval = t.dval;
            break;
        case Token::STR:
            // to copy a string, construct it using placement new
            new (&sval) std::string(t.sval);
            break;
        }
    }

private:
    enum {INT, CHAR, DBL, STR} tok; // discriminant

    // anonymous union
    // each Token object has an unnamed member of this unnamed union type
    union
    {
        char        cval;
        int         ival;
        double      dval;
        std::string sval;
    };
};
```

#### 局部类（Local Class）

- 类可以定义在 *函数内部* ，这样的类称为 *局部类* 
- 局部类只在定义它的作用域内可见
- 局部类的成员受到严格控制
    - 局部类的 *所有成员* （包括 *成员函数* 在内）都必须完整地定义在类的内部
    - 因此，自然也就**无法**定义 *静态成员* 
- 局部类对其外部作用域的名字的访问权限受到很多限制
    - 局部类中**不能**使用其外层函数的普通局部变量
    ```
    void foo(int val)
    {
        static int si;
        enum Loc { a = 1024, b };
        
        // Bar is local to foo
        struct Bar 
        {
            Loc locVal;  // ok: uses a local type name
            int barVal;
            
            void fooBar(Loc l = a)  // ok: default argument is Loc::a
            {
                barVal = val;       // error: val is local to foo
                barVal = ::val;     // ok: uses a global object
                barVal = si;        // ok: uses a static local object
                locVal = b;         // ok: uses an enumerator
            }
        };
        
        // . . .
    }
    ```
- 常规的访问保护规则对局部类同样适用
    - 外层函数对于局部类的私有成员**没有**任何访问特权
    - 局部类可以将外层函数声明为 *友元* 
    - 局部类已经只对它自己的外层函数可见了，再封装也没什么意义，一般就直接全部公有了
- 局部类中的名字查找
    - 局部类内部的名字查找次序与其他类类似
    - 在声明类的成员时，必须确保用到的名字位于作用域中，然后再使用该名字
    - 定义成员时用到的名字可以出现在类的任何位置
    - 如果某个名字**不是**局部类的成员，则继续在外层函数中查找
    - 如果还没有找到，则外外层函数所在的作用域中查找
- *嵌套的局部类* 
    - 可以在局部类内部再嵌套一个类
    - *嵌套的局部类* 的定义可以出现在 *局部类之外* 
    - 不过，局部类的嵌套类必须定义在与局部类相同的作用域中
    ```
    void foo()
    {
        class Bar 
        {
        public:
            // ...
            class Nested;  // declares class Nested
        };
        
        // definition of Nested
        class Bar::Nested 
        {
        // ...
        };
    }
    ```
    - 和往常一样，在类的外部定义成员时，必须指明该成员所属的作用域
        - 因此在上面的例子中，`Bar::Nested`的例子是说`Nested`是定义在`Bar`的作用域内的一个类
    - 局部类的嵌套类也是一个局部类，必须遵循局部类的各种规定
        - 嵌套类的所有成员都必须定义在嵌套类内部

#### [位域](https://en.cppreference.com/w/cpp/language/bit_field)（Bit Field）

- 位域
    - 声明具有以 *位* （bit，比特）为单位的明确大小的类数据成员
        - 设定成员变量的 *最大宽度* 
            - 用 *范围外的值* *赋值或初始化* 位域是 *未定义行为* 
            - 对位域进行 *自增越过其范围* 是 *未定义行为* 
            - *超越类型极限* 的位域仍 *只容许类型能容纳的最大值* ，剩下的空间就是 *白吃白占* 
                - `C`语言中干脆规定位域的宽度不能超过底层类型的宽度
        - 整个结构的实际大小
            - 位域的实际大小和在内存中的分布是 *由实现定义* 的
            - `16`、`32`、`64`位的处理器一般按照`2`、`4`、`8`字节 *对齐* 
            - 实际大小可能比位域总宽度要大
        - *相邻* 的位域成员一般 *按定义顺序打包* ，可以 *共享跨过字节*
            - 具体行为依赖平台的定义
                - 在某些平台上，位域不跨过字节，其他平台上会跨过
                - 在某些平台上，位域从左到右打包，其他为从右到左 
    - 因为位域不必然始于一个字节的开始，故**不能**取位域的地址
        - **不能定义** 指向位域的 *指针* 和 *非常量引用* 
        - 从位域初始化 *常量引用* 时，将绑定到一个 *临时副本* 上
    - 位域的类型只能是 *整型* 或 *枚举类型* 
        - 最好将位域设为 *无符号类型* ，使用存储在 *带符号类型* 中的位域是 *未定义行为* 
    - 位域**不能是** *静态数据成员* 
    - **没有**位域 *纯右值* 。左值到右值转换始终生成位域底层类型的对象
    - 位域 *类内初始值*
        - `C++20`之前：**不能设置** 
        - `C++20`开始：用提供的 *花括号或等号初始化器* 初始化
    ```
    struct S                              // 64 位处理器一般按 8 字节（ 64 bit ）对齐
    {
        unsigned char c : 16;             // 16 bit 的无符号字符位域，但仍只允许允许值 0...255
                                          // 剩下的 8 bit 那就是白吃白占
        
        unsigned int b1 : 3,              // 3 bit 的无符号整数位域，允许值为 0...7
                        : 2;              // 2 bit 的无名位域，空着 
                        
        unsigned int    : 0;              // 0 bit 的无名位域，空着
                                          // 但为了钦定 b2 对齐下一个字节，实际白吃白占 3 bit 
                   
        unsigned int b2 : 6,              // 6 bit 的无符号整数位域，允许值为 0...63
                     b3 : 2;              // 2 bit 的无符号整数位域，允许值为 0...3
                                          
                                          // 到此位域总宽度一共是 32 bit
                                          // 但整个结构体要按 8 字节（ 64 bit ）对齐
                                          // 所以这儿再次白吃白占 32 bit
    };

    S s;
    std::cout << sizeof(S) << std::endl;  // 64 位处理器上会占用 8 字节（ 64 bit ）
    s.b1 = 7;
    ++s.b1;                               // 值 8 不适合此位域
    std::cout << s.b1 << std::endl;       // 未定义行为，可能是 0
    ```
- 位域的声明
    - 使用下列声明符的类数据成员声明（`[]`代表 *可选* ）
        - `[identifier] [attr] : size`  
        - `[identifier] [attr] : size brace-or-equal-initializer` `(since C++20)`  
    - 位域的 *类型* 由声明语法的 *声明说明符序列* 引入
        - *标识符* ：被声明的位域名
            - 名字是可选的， *无名位域* 引入指定数量的填充位
        - [*属性说明符序列*](https://en.cppreference.com/w/cpp/language/attributes) ：可选的任何数量属性的序列
        - *大小* ： *非负整型* 常量表达式
            - 大于零时，这是位域将占有的位数
            - *只有* *无名位域* 的大小能等于零，用于钦定自己 *后面* 的那个位域 *对齐下一个字节*
        - *花括号或等号初始化器* ：此位域所使用的默认成员初始化器
            - 自然，**不支持** *括号初始化器*
        ```
        int a;
        const int b = 0;
        
        struct S
        {
            // simple cases
            // even these cases are undefined behavior before C++20
            int x1 : 8 = 42;               // OK; "= 42" is brace-or-equal-initializer
            int x2 : 8 { 42 };             // OK; "{ 42 }" is brace-or-equal-initializer
            
            // ambiguities
            int y1 : true ? 8 : a = 42;    // OK; brace-or-equal-initializer is absent
            int y2 : true ? 8 : b = 42;    // error: cannot assign to const int
            int y3 : (true ? 8 : b) = 42;  // OK; "= 42" is brace-or-equal-initializer
            int z : 1 || new int { 0 };    // OK; brace-or-equal-initializer is absent
        };
        ```

#### [`cv`限定](https://en.cppreference.com/w/cpp/language/cv)（`cv` type qualifiers）

- `volatile`限定符
    - `volatile`**不**跨平台
        - `volatile`的确切含义与机器相关，只能通过阅读编译器文档来理解
        - 想要让使用了`volatile`的程序在移植到新机器或新编译器后仍然有效，通常需要对该程序做出某些改变
    - 当对象的值可能在程序的控制或检测之外被改变时，应该将对象声明为`volatile`
        - `volatile`告诉编译器：**不应**对此对象进行优化
    - `volatile`和`const`在使用上很相似
    ```
    volatile int display_register;  // int value that might change
    volatile Task * curr_task;      // curr_task points to a volatile object
    volatile int iax[max_size];     // each element in iax is volatile
    volatile Screen bitmapBuf;      // each member of bitmapBuf is volatile
    ```
    - `volatile`也可以用于修饰类的 *成员函数* 
        - 类似于`const`成员函数
            - `volatile`对象实例将只能调用`volatile`成员函数
    - `volatile`可以用于修饰 *指针* 或 *引用*
        - 类似于`const`
            - `volatile`指针一样分顶层和底层
            - `volatile`对象将只能由`volatile`指针或引用指向
    ```
    volatile int v;                 // v is a volatile int
    int * volatile vip;             // vip is a volatile pointer to int
    volatile int * ivp;             // ivp is a pointer to volatile int
    
    volatile int * volatile vivp;   // vivp is a volatile pointer to volatile int
    int * ip = &v;                  // error: must use a pointer to volatile
    *ivp = &v;                      // ok: ivp is a pointer to volatile
    vivp = &v;                      // ok: vivp is a volatile pointer to volatile
    ```
    - *合成的拷贝* 对`volatile`**无效**
        - `const`和`volatile`的一个重要区别就是我们**不能**使用 *合成的拷贝、移动构造函数及赋值运算符* 初始化`volatile`对象或从其赋值
        - 合成的成员接受的形参类型是**非**`volatile`的常量引用，显然**不能**将非`volatile`引用绑定到`volatile`对象上
        - 如果一个类希望拷贝、移动或赋值其`volatile`对象，则该类必须 *自定义拷贝或移动操作* 
        ```
        class Foo 
        {
        public:
            // copy from a volatile object
            Foo(const volatile Foo &); 
            
            // assign from a volatile object to a nonvolatile object
            Foo & operator=(volatile const Foo &);
            
            // assign from a volatile object to a volatile object
            Foo & operator=(volatile const Foo &) volatile;
            
            // remainder of class Foo
        };
        ```
- *`cv`限定* 可出现于任何类型说明符中，以指定被声明对象或被命名类型的 *常量性* （constness）或 *易变性* （volatility）
    1. `const`对象
        - 包含
            - `const`限定的对象
            - `const`对象的非`mutable`子对象
        - **不能**修改
            - 直接这么做是编译时错误
            - 间接这么做（例如通过到非`const`类型的引用或指针修改`const`对象）是 *未定义行为* 
    2. `volatile`对象
        - 包含
            - `volatile`限定的对象
            - `volatile`对象的子对象
            - `const volatile`对象的`mutable`子对象
        - 通过`volatile`限定的类型的 *泛左值表达式* 的每次访问（读或写操作、成员函数调用等）都**不能被优化掉**
            - 即在单个执行线程内，`volatile`访问不能被优化掉，或者与另一按顺序早于或按顺序晚于该`volatile`访问的可见副作用进行重排序
            - 这使得`volatile `对象适用于与信号处理函数之间的交流，但不适于与另一执行线程交流
        - 试图通过非`volatile`泛左值涉指`volatile`对象（例如，通过到非`volatile`类型的引用或指针）是 *未定义行为* 
    3. `const volatile`对象
        - 包含
            - `const volatile`限定的对象
            - `const volatile`对象的非`mutable`子对象
            - `volatile`对象的`const`子对象
            - `const`对象的非`mutable volatile`子对象
        - 同时表现为`const`对象与`volatile`对象 

#### 链接指示（Linkage Directives）

- `C++`程序调用其他语言（包括`C`语言）的函数时，一样需要先声明再使用，且声明必须制定返回类型和形参列表
    - 编译器检查其调用的方式与普通`C++`函数的方式相同，但生成的代码有区别
        - 具体到`C/C++`，由于`C++`函数可以重载，因此生成的名字比`C`要复杂一点
    - 要想把`C++`代码和其他语言（包括`C`语言）编写的代码放在一起使用，要求我们必须有权访问该语言的编译器，且该编译器与当前的`C++`编译器兼容
        - 当然了，比如在`ubuntu 20.04 LTS`上，`C`和`C++`的默认编译器压根就是同一个（`gcc (Ubuntu 9.3.0-10ubuntu2) 9.3.0`），所以上一条自然是满足的
- *链接指示* 用于声明非`C++`函数
    - 链接指示可以有 *两种* 形式： *单个* 的和 *复合* 的
        - 链接指示**不能**出现在 *类定义* 或 *函数定义* 的 *内部* 
        - `举例`：`<cstring>`头文件中某些`C`函数是如何声明的
        ```
        // illustrative linkage directives that might appear in the C++ header <cstring>
        
        // single-statement linkage directive
        extern "C" 
        size_t strlen(const char *);
        
        // compound-statement linkage directive
        extern "C" 
        {
        int strcmp(const char *, const char *);
        char * strcat(char *, const char *);
        }
        ```
        - `extern`后面的字符串字面值常量指出了编写函数所用的语言
            - 编译器应当支持对`C`语言的链接指示
            - 可能还支持其他的，例如`extern "Ada"`、`extern "FORTRAN"`等
- *链接指示* 与头文件
    - 可以令链接指示后面跟上 *花括号* `{}`括起来的若干函数的声明，从而一次性建立多个链接
    - 花括号的作用是将适用于该链接指示的多个声明聚合在一起
    - 否则，花括号就会被忽略，花括号中生命的函数的名字就是可见的，就好像是在花括号之外声明的一样
    - *多重声明* 的形式可以应用于整个头文件，例如，`C++`的`<cstring>`头文件就可能形如
    ```
    // compound-statement linkage directive
    extern "C" 
    {
    #include <string.h>  // C functions that manipulate C-style strings
    }
    ```
    - 当一个`#include`指示被放置在复合链接指示的花括号中时，头文件中的所有普通函数声明都被认为是由链接指示的语言编写的
    - 链接指示 *可以嵌套* 
        - 因此如果头文件包含 *自带链接指示的函数* ，则该函数的链接**不**受影响
    - `C++`从`C`语言继承的标准库函数可以定义成`C`函数，但并非必须
        - 具体使用`C`还是`C++`实现`C`标准库，是 *由实现定义* 的 
- 指向`extern "C"`函数的指针
    - 编写函数所用的语言是 *函数类型* 的一部分
        - 因此，对于使用链接指示定义的函数来说，它的每个声明都必须使用相同的链接指示
        - 而且，指向其他语言编写的 *函数的指针* 必须与函数本身 *使用相同的链接指示* 
        ```
        // pf points to a C function that returns void and takes an int
        extern "C" void (*pf)(int);
        ```
        - 当我们使用`pf`调用函数时，编译器认定当前调用的是一个`C`函数
        - 指向`C`函数的指针与指向`C++`函数的指针是**不一样**的类型
            - 指向`C`函数的指针**不能**指向`C++`函数，反之亦然
            - 就像其他类型不匹配的问题一样，对不同链接指示的指针之间进行赋值将引发 *编译错误* 
            ```
            void (*pf1)(int);             // points to a C++ function
            extern "C" void (*pf2)(int);  // points to a C function
            pf1 = pf2;                    // error: pf1 and pf2 have different types
            ```
            - 虽然有的编译器允许这种赋值，但这是村规，按照`C++`标准这是 *非法行为* 
- *链接指示* 对整个声明都有效
    - 当我们使用链接指示时，它不仅对函数有效，而且对作为返回值类型或形参类型的函数指针也有效
    ```
    // f1 is a C function; its parameter is a pointer to a C function
    extern "C" void f1(void(*)(int));
    ```
    - 这条声明语句指出`f1`是一个不返回任何值的`C`函数
        - 它有一个类型为`extern "C" void(*)(int)`的`C`函数指针形参
    - 因为链接指示同时作用于声明语句中的所有函数，所以如果我们希望给`C++`函数传入一个指向`C`函数的指针，则必须使用 *类型别名* 
    ```
    // FC is a pointer to a C function
    extern "C" typedef void FC(int);
    // f2 is a C++ function with a parameter that is a pointer to a C function
    void f2(FC *);
    ```
- 导出`C++`函数到其他语言
    - 通过使用链接指示对函数进行定义，我们可以令一个`C++`函数在其他语言编写的程序中可用
    ```
    // the calc function can be called from C programs
    extern "C" double calc(double dparm) { /* ... */ }
    ```
    - 编译器将生成适合于指定语言的代码
    - 值得注意的是，可被多种语言共享的函数的返回类型或形参类型受到很多限制
        - 例如，不大可能把一个`C++`类的对象传给`C`程序
    - 预处理器对`extern "C"`的特殊支持
        - 有时需要在`C`和`C++`中编译同一个源文件
        - 编译`C++`版本的程序时，预处理器定义`__cplusplus`宏
        - 利用这个宏，可以在编译程序时有条件地包含代码
        ```
        #ifdef __cplusplus
        // ok: we're compiling C++
        extern "C"
        #endif
        int strcmp(const char *, const char *);
        ```
- 重载函数与 *链接指示* 
    - 链接指示与重载函数的相互作用依赖于目标语言
        - 如果目标语言支持重载函数，则为该语言实现链接指示的编译器很可能也支持重载这些`C++`的函数
    - `C`语言**不**支持重载函数，因此一个`C`链接指示只能用于说明一组重载函数中的某一个
    ```
    // error: two extern "C" functions with the same name
    extern "C" void print(const char *);
    extern "C" void print(int);
    ```
    - 如果一组重载函数中有一个是`C`函数，则其余的必定都是`C++`函数
    ```
    class SmallInt { /* . . . */ };
    class BigNum { /* . . . */ };

    // the C function can be called from C and C++ programs
    // the C++ functions overload that function and are callable from C++
    extern "C" double calc(double);
    extern SmallInt calc(const SmallInt &);
    extern BigNum calc(const BigNum &);
    ```
    - `C`版本的`calc`函数可以在`C`或`C++`程序中调用，而使用了类类型形参的`C++`函数只能在`C++`程序中调用。
    - 上述性质与 *声明的顺序* **无关**


