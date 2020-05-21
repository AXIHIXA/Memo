# `C++ Primer 5th Edition` Notes






- 记录一些对`C++`理解得不到位的地方
- 基于`C++11`的内容提示，例如`(since C++11)`，一般不再明确标注
- `(until C++11)`、即`C++11`中已经移除的内容，不予收录
- 这玩意收录好多[`cppreference`](https://en.cppreference.com)上的内容，该部分内容是打算当字典看的，总体来讲似乎比`C++ Primer`还不适合初学者看了






### 🌱 一句话

- 常见规则
    - 先声明（或定义）再使用。 *第一次实际使用前* 再声明（定义）
    - **严禁**混用有符号类型和无符号类型（比如：该用`size_t`就用，别啥玩意都整成`int`）
    - 整数和浮点数字面值的后缀一律使用 *大写* 版本，避免`l`和`1`混淆
    - 如果函数有可能用到某个全局变量，则**不宜**再定义同名的局部变量
    - `const`常量不论是声明还是定义都添加`extern`修饰符
    - 想要`auto`推导出引用或者常量的话，直接写清楚是坠吼的（`const auto & a = b`），别折腾顶层`const`什么的
    - 认定应为常量表达式的变量应当声明为`constexpr`类型
    - 凡是不修改类数据成员的成员函数函数一律定义成常成员函数
    - `constexpr`函数、静态`constexpr`成员、`inline`函数（包括类的`inline`成员函数）以及模板的**定义和实现都应**写进头文件
    - `using`声明（`using std::string`、`using namespace std`、`using intptr = int *`等）**不应**写进头文件
    - `for each`循环内以及使用迭代器时**不能**改变被遍历的容器的大小
    - 现代`C++`应使用标准库类型配合迭代器，而**不是**`C`风格的数组和指针。数组也是一种迭代器
    - 现代`C++`**不应**使用旧式的强制类型转换，应当明确调用对应的`xx_cast<T>(expr)`
    - 除非必须，**不要**使用自增自减运算符的后置版本（会造成性能浪费）
    - **不在**内部作用域声明函数（内部作用域生命的东西会覆盖外部作用域的同名东西，可能会影响函数重载的使用）
    - 构造函数**不应**该覆盖掉类内初始值，除非新值与原值不同；不使用类内初始值时，则每个构造函数**都应显式初始化**每一个类内成员
    - 希望类的所有成员都是`public`时，**应**使用`struct`；只有希望使用`private`成员时才用`class`
    - 在类定义开始或结束的地方**集中声明**友元；使用友元，仍另需有一个**单独的函数声明**
    - 类的类型成员（`typedef`以及`using`声明）应该放在类定义**刚开始**的地方的`public`区域
    - 最好令构造函数初始化列表的顺序与成员声明的顺序**保持一致**；**避免**用某些成员初始化其他成员，用构造函数的参数作为初始值
    - 应把静态数据成员的定义与其他非内联函数的定义放在**同一个文件**中
    - 即使一个`constexpr`静态成员在类内部被初始化了，也应该在类外定义一下该成员（此时**不能**再指定初始值）
    - 不需要写访问时，应当使用`const_iterator`
    - 改变容器 *大小* 之后，则 *所有* 指向此容器的迭代器、引用和指针都 *可能* 失效，所以一律更新一波才是 *坠吼的* 。此外，永远**不要缓存**尾后迭代器（这玩意常年变来变去），现用现制，用后即弃
    - 泛型编程要求：**应当**统一使用非成员版本的`swap`，即`std::swap(c1, c2);`
    - 调用泛型算法时，在不需要使用返回的迭代器修改容器的情况下，传参应为`const_iterator`
    - 通常**不对**关联容器使用泛型算法（或不能用或性能很差）
    - `lambda`表达式应尽量**避免**捕获指针或引用。如捕获引用，必须保证在`lambda`执行时变量 *仍存在* 
    - 出于性能考虑，`std::list`和`std::forward_list`应当优先使用 *成员函数版本* 的算法，而**不是**通用算法  
    - 如果将`std::shared_ptr`存放于容器中，而后不再需要全部元素，要使用`c.erase`删除不再需要的元素
    - 如果两个对象 *共享底层数据* ，则某个对象被销毁时，**不能**单方面地销毁底层数据
    - 坚持使用 *智能指针* ，避免所有动态内存管理的破事。智能指针使用规范
        1. **不**使用相同的内置指针初始化（或`reset`）多个智能指针，否则是 *未定义行为*
        2. **不**`delete`从智能指针`get()`到的内置指针
        3. **不**使用智能指针的`get()`初始化（或`reset`） *另一个* 智能指针
        4. 使用智能指针的`get()`返回的内置指针时，记住当最后一个对应的智能指针被销毁后，这个内置指针就 *无效* 了
        5. 使用内置指针管理的资源而不是`new`出来的内存时，记住传递给它一个 *删除器*
- 一些小知识
    - 如果两个字符串字面值位置紧邻且仅由 *空格* 、 *缩进* 以及 *换行符* 分隔，则它们是 *一个整体* 
    - `C++11`规定整数除法商一律向0取整（即：**直接切除小数部分**）
    - 指针解引用的结果是其指向对象的**左值**引用
    - *悬垂引用* （Dangling reference）就是 *野指针* 的引用版，可能出现于一些右值引用数据成员上
    - `*iter++`等价于`*(iter++)` => 优先级：`++` > `*`
    - `p->ele`等价于`(*p).ele` => 优先级：`.` < `*`
    - 非`extern`全局`const`对象和`static`对象被设定为**仅在文件内有效**
    - `std::endl`有刷新缓冲区的效果。最好带上
    - 如果一个函数是永远也不会用到的，那么它可以只有声明而没有定义 => 15.3
    - 引用从来都是作为被引用对象的同义词出现（比如`auto`就不能自动推断出引用），唯一例外是`decltype`。它会原样保留引用以及顶层`const`
    - `main`函数不能递归调用、不能重载
    - 定义在类内部的函数是隐式的`inline`函数
    - 使用`struct`或`class`定义类的**唯一区别**就是默认访问权限：`struct`中默认 *公有* ，而`class`默认 *私有*
    - 每个类定义了**唯一**的类型；两个类即使内容完全一样，它们也是不同的类型，**不能**自动相互转化
    - 如果一个构造函数为每一个参数都提供了默认实参，则它实际上也定义了默认构造函数
    - 能通过一个实参调用的构造函数定义了一条从构造函数的参数类型向类类型隐式转换的规则
    - 非`constexpr`静态成员只能在**类外**初始化；在类外部定义静态成员时，**不能**重复`static`关键字
    - 那些只接受一个单一迭代器来表示第二个序列的算法，都假定 *第二个序列至少与第一个序列一样长*
    - 向目的位置迭代器写数据的算法都假定 *目的位置足够大* ，能容纳要写入的元素
    - 如果容器为空，则`begin`和`end`返回的是**同一个**迭代器，都是尾后迭代器
    - 只有当其元素类型也定义了相应的比较运算符时，才可以使用 *关系运算符* 来比较两个容器
    - 对`std::map`使用下标操作的行为和对数组或者`std::vector`使用时很不相同，使用一个 *不在容器中的键* 作为下标将会 *添加* 一个具有此键的元素到容器中
    - 泛型算法**不能（直接）添加或删除**元素。具体来说，调用`std::unique()`之前要有`std::sort()`，之后还要调用`c.erase()`来实际释放空间
    - 只有迭代器指向的容器支持 *随机访问* 时，才能调用迭代器算术运算（Iterator Arithmetic）
    - 如果要人工转换 *反向迭代器* ，一定记得**反向的`begin`要喂正向的`end`**，且模板参数是 *容器的迭代器类型* ，**不是容器自身类型**
    - 普通迭代器指向的元素和用它转换成的反向迭代器指向的**不是**相同元素，而是相邻元素；反之亦然
    - 标准库中使用的顺序一般默认是 *非降序* ，二元比较谓词一般等价于`<`
    - `std::sort`如何使用谓词：`后 < 前 == true`或`<(后, 前) == true`就 *对换*    
- 读代码标准操作
    - 判断复杂类型`auto`变量的类型：先扒掉引用，再扒掉被引用者的顶层`const`
    - [如何理解`C`声明](https://en.cppreference.com/w/cpp/language/declarations#https://en.cppreference.com/w/cpp/language/declarations#Understanding_C_Declarations)
        - `C`声明遵循以下规则
            - 优先级从高到低
                1. 用于 *分组* 的括号（Parentheses grouping together a part of the declaration）
                2. 后缀操作符（例如`()`表示函数，`[]`表示数组）
                3. 前缀操作符（例如`*`表示指针）
            - `cv`限定如出现于`*`之前，则作用于指向的类型；如出现于`*`之后，则作用于指针本身
        - 如何理解复杂声明
            1. 从名字`p`开始，说`declare p as...`，之后按照如上优先级解读名字周边的内容
            2. 如果`p`右边是`[n]`，则说`array n of...`
            3. 如果`p`右边是表示函数的括号`(param_list)`（例如`()`，`(float, int)`），则说`function (param_list) returning...
            4. 如果`p`左边是`*`（可能还有`cv`限定），则说`xx pointer to...`（例如`int const * const`说成`const pointer to const int`）
            5. 跳出这一层 *分组* 括号（如有），重复`(2) - (5)`
        - 举例：`int (*(*pf)(int, int (*(*)(int))[20]))[10]`：
            - 按顺序翻译为
            ```
            declare pf as pointer to function (int, pointer to function (int) returning pointer to array 20 of int) 
                                     returning pointer to array 10 of int
            ```
        - 大宝贝：[cdecl](https://cdecl.org/) ，帮你干这些破事儿，安装：`sudo apt install cdecl`







### 🌱 字面值（literal）

- *整数* 
    - 可以写作十进制、八进制（以`0`开头）或十六进制（以`0x`或`0X`开头）形式
    ```
    20              // dec，int
    -42             // dec，int
    42ULL           // dec，unsigned long long
    024             // oct，int
    0x1a            // hex，int
    0X1A            // hex，int
    ```
    - 默认（无后缀）情况下
        - 十进制字面值为`int`、`long`和`long long`中能容纳该数值的尺寸最小者
            - 十进制字面值**不会**是负数。例如`-42`中的`-`并不算在字面值里面，其作用只是对字面值`42`取相反数
        - 八进制和十六进制字面值为`int`、`unsigned int`、`long`、`unsigned long`、`long long`和`unsigned long long`中能容纳该数值的尺寸最小者 
    - 可选后缀
        - `u`，`U`： *最小* 匹配`unsigned`。可与`L`、`LL`搭配使用
        - `l`，`L`： *最小* 匹配`long`
        - `ll`，`LL`： *最小* 匹配`long long`
        - 这些后缀都该使用 *大写* 版本，因为小写的`l`太容易和`1`混了
- *浮点数* 
    - 写作一个小数或以科学计数法表示的指数，其中指数部分用`e`或`E`标识
    ```
    3.14159         // double
    3.14159E0       // double
    3.14159L        // long double
    0.              // double
    0e0             // double
    1E-3F           // float
    .001            // double
    ```
    - 默认（无后缀）情况下为`double`
    - 可选后缀
        - `f`，`F`：匹配`float`
        - `l`，`L`：匹配`long double`
- *字符* 和 *字符串* 
    - 由单引号`''`括起来的是字符，双引号`""`括起来的是字符串
    ```
    'a'             // char
    "Hello World!"  // string literal (const char [])
    L'a'            // wchar_t
    u8"hi!"         // UTF-8 string literal
    ```
    - 可选前缀
        - `u`：`Unicode 16`字符，匹配`char16_t`
        - `U`：`Unicode 32`字符，匹配`char32_t`
        - `L`：宽字符，匹配`wchar_t`
        - `u8`：`UTF-8`字符（仅用于字符串字面值），匹配`char`
    - 字符串字面值实际是 *常字符数组* ，编译器 *自动在末尾添加空字符* `'\0'`作为结尾
    ```
    strlen("12")   == 3
    strlen("12\0") == 3
    ```
    - 如果两个字符串字面值位置紧邻且仅由 *空格* 、 *缩进* 以及 *换行符* 分隔，则它们是 *一个整体* 
        - 书写的字符串字面值较长时，不妨分开书写
    ```
    std::cout << "a really, really long string literal "
                 "that spans two lines"                  << std::endl;
    ```
    - 转义序列
        - 转义序列均以`\`开始
            - `'\n'`：换行符
            - `'\r'`：回车符
            - `'\t'`：横向制表符
            - `'\v'`：纵向制表符
            - `'\b'`：退格符
            - `'\f'`：进纸符
            - `'\a'`：报警（响铃）符
            - `'\"'`：双引号
            - `'\''`：单引号
            - `'\\'`：反斜杠
            - `'\?'`：问号
        - 泛化的转义序列
            - `'\0'`：空字符
            - `'\7'`：响铃
            - `'\12'`：换行
            - `'\40'`：空格
            - `'\115'`，`'\x4d'`：`'M'`






### 🌱 [声明和定义](https://en.cppreference.com/w/cpp/language/definition)（Declarations and definitions）

- *定义* 就是完整定义了被引入的实体的 *声明* 
- 所有的 *声明* 都是定义，除以下情况**例外**
    - *无函数体* 的函数声明 
    ```
    int f(int);                       // 声明但不定义 f
    ```
    - 带有 *存储类说明符* `extern`或者 *语言连接说明符* （诸如`extern "C"`）而 *无初始化器* 的所有声明 
    ```
    extern const int a;               // 声明但不定义 a
    extern const int b = 1;           // 定义 b
    ```
    - 在类的定义中的`非 inline (since C++17)` *静态数据成员* 的声明 
    ```
    struct S 
    {
        int n;                        // 定义 S::n
        static int i;                 // 声明但不定义 S::i
        inline static int x;          // 定义 S::x
    };                                // 定义 S
    
    int S::i;                         // 定义 S::i
    ```
    - `(deprecated since C++17)` 已经在类中用`constexpr`说明符定义过的 *静态数据成员* ，在 *命名空间作用域* 中的声明 
    ```
    struct S 
    {
        static constexpr int x = 42;  // 隐含为 inline，定义 S::x
    };
    
    constexpr int S::x;               // 声明 S::x ，不是重复定义
    ```
    - （通过 *前置声明* 或通过在其他声明中使用 *详细类型说明符* ）对类名字进行的声明 
    ```
    struct S;                         // 声明但不定义 S
    class Y f(class T p);             // 声明但不定义 Y 和 T（以及 f 和 p）
    ```
    - *枚举* 的不可见声明 
    ```
    enum Color : int;                 // 声明但不定义 Color
    ```
    - 模板形参的声明 
    ```
    template <typename T>             // 声明但不定义 T
    ```
    - 并非定义的函数声明中的形参声明 
    ```
    int f(int x);                     // 声明但不定义 f 和 x
    int f(int x)                      // 定义 f 和 x
    { 
         return x + a;
    }
    ```
    - `typedef`声明 
    ```
    typedef S S2;                     // 声明但不定义 S2（S 可以是不完整类型）
    ```
    - 别名声明 
    ```
    using S2 = S;                     // 声明但不定义 S2（S 可以是不完整类型）
    ```
    - `using`声明 
    ```
    using N::d;                       // 声明但不定义 d
    ```
    - `(since C++17)` 推导指引的声明（不定义任何实体）
    - `static_assert`声明（不定义任何实体）
    - 特性声明（不定义任何实体） 
    - 空声明（不定义任何实体）
    - `using`指令（不定义任何实体） 
    - 显式实例化声明（ *`extern`模板* ） 
    ```
    extern template f<int, char>;     // 声明但不定义 f<int, char>
    ```
    - 不是定义的显式特化声明 
    ```
    template<> struct A<int>;         // 声明但不定义 A<int>
    ```






### 🌱 [存储期和链接](https://en.cppreference.com/w/cpp/language/storage_duration)（Storage duration and linkage）

#### [存储期](https://en.cppreference.com/w/cpp/language/storage_duration#Storage_duration)（Storage duration）

所有对象都具有下列四种 *存储期* （ *生存期* ）之一：

1. *自动存储期*  （Automatic storage duration）
    - 包含
        - **未声明**为`static`、`extern`或`thread_local`的所有 *局部对象* 
    - 存储方式
        - 对象的存储在外围代码块开始时分配，而在结束时解分配
        - 存储于 *自动存储区* （程序的 *栈* 内存）
        - 被称作 *自动对象* 
2. *静态存储期*  （Static storage duration）
    - 包含
        - 所有声明于 *命名空间作用域* （包含 *全局命名空间* ）的对象
        - 声明带有`static`或`extern`的对象
    - 存储方式
        - 对象的存储在程序开始时分配，而在程序结束时解分配
        - 该对象的定义语句只在程序第一次执行到时会被执行，之后会被程序跳过
            - 也就是说该对象永远 *只存在一个实例*
        - 存储于 *静态存储区* （程序的 *静态* 内存）
3. *线程局部存储期*  （Thread local storage duration）
    - 包含
        - *所有* 声明带有`thread_local`的对象
    - 存储方式
        - 对象的存储在线程开始时分配，而在线程结束时解分配
        - `thread_local`对象的实例每个线程各有一份
        - `thread_local`能与`static`或`extern`一同出现，以调整链接
4. *动态存储期*  （Dynamic storage duration）
    - 包含
        - [`new`表达式](https://en.cppreference.com/w/cpp/language/new)搞出来的东西
        - 其他内存申请操作（例如`malloc`）搞出来的东西
    - 存储方式
        - 对象的存储是通过使用 *动态内存分配函数* 来按请求进行分配和解分配的
        - 显式请求分配时分配，显式请求释放时 *才会* 释放
        - 存储于 *动态存储区* （程序的 *堆* 内存）

#### [链接](https://en.cppreference.com/w/cpp/language/storage_duration#Linkage)（Linkage）

所有变量都具有如下四种 *链接* 之一，用于调节变量在不同文件（ *翻译单元* ）之间的可见性：

1. *无链接* （No linkage）
    - 名字只能从 *其所在的作用域* 使用
    - 声明于 *块作用域* 的下列任何名字均 *无链接* 
        1. **未显式声明**为`extern`的变量（无关乎`static`修饰符）
        2. *局部类* 及 *其成员函数* 
        3. *其他名字* ，`typedef`、 *枚举* 及 *枚举项* 
        4. **未指定**为拥有 *外部* 、 `模块 (since C++20)` 或 *内部* 链接的名字亦 *无链接* ，无关乎其声明所处在的作用域
2. *内部链接* （Internal linkage）
    - 名字可从 *当前翻译单元* 中的 *所有作用域* 使用
    - 声明于 *命名空间作用域* 的下列任何名字均具有 *内部链接* 
        1. 声明为`static`的 *变量*  、 `变量模板 (since C++14)`、 *函数* 和 *函数模板* 
        2. `const`限定的 *变量* （包含`constexpr`），但以下情况**除外**
            - `extern`的
            - 先前声明为外部链接的
            - `volatile`
            - 模板的 `(since C++14)`
            - `inline`的 `(since C++17)`
        3. *匿名联合体* 的数据成员
        4. 声明于 *无名命名空间* 或 *无名命名空间内的命名空间* 中名字，即使是显式声明为`extern`者，均拥有 *内部链接* 
    - `const`常量不论是声明还是定义都添加`extern`修饰符
        - 默认状态下，`const`对象为文件作用域（即：仅在文件内有效）
        - 如果想在多个文件之间共享`const`对象，则必须在定义的对象之前添加`extern`关键字
        ```
        extern const int BUF_SIZE = fcn();  // globals.cpp
        extern const int BUF_SIZE;          // globals.h
        extern const int BUF_SIZE;          // sth.h （其他要用到`BUF_SIZE`的头文件）
        ```
        - 编译器在编译过程中会把所有的`const`变量都替换成相应的字面值（这一步骤实际上是 [*默认初始化*](https://en.cppreference.com/w/cpp/language/initialization) 过程中的 [*常量初始化*](https://en.cppreference.com/w/cpp/language/constant_initialization)（Constant initialization））。为了执行上述替换，编译器必须知道变量的初始值。如果程序包含多个文件，则每个用了`const`对象的文件都必须得能访问到它的初始值才行。要做到这一点，就必须在每一个用到变量的文件之中都有它的定义。为了支持这一用法，同时避免对同一变量的重复定义，默认情况下，`const`对象被设定为仅在文件内有效。当多个文件中出现了同名的`const`变量时，其实等同于在不同文件中分别定义了**独立的**变量。如果希望`const`对象只在一个文件中定义一次，而在多个文件中声明并使用它，则需采用上述操作。
3. *外部链接* （External linkage）
    - 名字能从 *其他翻译单元* 中的作用域使用
        - 具有 *外部链接* 的变量和函数亦具有 [*语言链接*](https://en.cppreference.com/w/cpp/language/language_linkage)（Language linkage），这使得可以链接到以 *不同编程语言* 编写的 *翻译单元* 
            - `C++`的默认外联语言当然是`C++`自己
            - `C++`外联`C`是坠常见的，举个栗子
            ```
            #ifdef __cplusplus
            extern "C" int foo(int, int);  // C++ compiler sees this
            #else
            int foo(int, int);             // C compiler sees this
            #endif
            ```
            - 啊，熟悉的[*`python C/C++`拓展*](https://docs.python.org/3.7/extending/extending.html)，你来了
    - *首次* 声明于 *块作用域* 的下列名字均具有 *外部链接* 
        - 声明为`extern`的 *变量名* 
        - *函数名*  
    - 声明于 *命名空间作用域* 的下列名字均具有 *外部链接* ，除下一条的两个例外
        - 所有声明为`extern`的 *变量* 
        - **未声明**为`static`的 *函数* 和 *函数模板* 
        - *命名空间作用域* 内**未声明**为`static`的非`const`变量
        - *枚举*
        - *类名* ， *成员函数*  ， *静态数据成员* （不论是否`const`）， *嵌套类* ， *嵌套枚举* ， *首次引入* 的 *友元函数* 
        - **未声明**为`static`的 *函数模板* 
    - 上一条的两个**例外**
        - 名字声明于 *无名命名空间或内嵌于无名命名空间的命名空间* ，则该名字拥有 *内部链接* 
        - 声明于 *命名空间作用域* 中的 *具名模块* 且 *不被导出* ，且无内部链接，则该名字拥有 *模块链接* `(since C++20)`
    ```
    int a;             // 这其实是声明并定义了变量a
    extern int a;      // 这才是仅仅声明而不定义
    extern int a = 1;  // 这是声明并定义了变量a并初始化为1。
                       // 任何包含显式初始化的声明即成为定义，如有extern则其作用会被抵消
    ```
4. *模块链接* （Module linkage） `(since C++20)`
    - 名字 *只能* 从 *同一模块单元* 或 *同一具名模块中的其他翻译单元* 的作用域指代
    - 声明于 *命名空间作用域* 中的 *具名模块* 且 *不被导出* ，且无内部链接，则该名字拥有 *模块链接*






### 🌱 [作用域](https://en.cppreference.com/w/cpp/language/scope)（scope）

- `C++`程序中出现的每个名字，只在某些可能不连续的源码部分中 *有效* （即，编译器能知道这玩意儿是啥、在哪儿声明的），这些部分被称为其 *作用域* 
    - 编译器通过[名字查找](https://en.cppreference.com/w/cpp/language/lookup)（Name lookup）实现名字和声明的关联
        - 如果名字已经指明了 *命名空间* ，则进行[限定名字查找](https://en.cppreference.com/w/cpp/language/qualified_lookup)（Qualified name lookup）
        - 否则，进行[无限定名字查找](https://en.cppreference.com/w/cpp/language/unqualified_lookup)（Unqualified name lookup）
- 根据变量的 *定义位置* 和 *生命周期* ，`C++`的变量具有不同的 *作用域* ，共分为以下几类 
    - [*块作用域*](https://en.cppreference.com/w/cpp/language/scope#Block_scope)（Block scope）
    - [*函数形参作用域*](https://en.cppreference.com/w/cpp/language/scope#Function_parameter_scope)（Function parameter scope）
    - [*函数作用域*](https://en.cppreference.com/w/cpp/language/scope#Function_scope)（Function scope）
    - [*命名空间作用域*](https://en.cppreference.com/w/cpp/language/scope#Namespace_scope)（Namespace scope）
        - 包含 *全局命名空间作用域* （Global namespace scope），即所谓的 *全局作用域* 
    - [*类作用域*](https://en.cppreference.com/w/cpp/language/scope#Class_scope)（Class scope）
    - [*枚举作用域*](https://en.cppreference.com/w/cpp/language/scope#Enumeration_scope)（Enumeration scope）
    - [*模板形参作用域*](https://en.cppreference.com/w/cpp/language/scope#Template_parameter_scope)（Template parameter scope）
- 作用域始于 *声明点* ，内部作用域（inner scope）的变量会 *覆盖* 外部作用域（outer scope）的 *同名变量* 

#### [块作用域](https://en.cppreference.com/w/cpp/language/scope#Block_scope)（Block scope）

- *块* （ *复合语句* ）中的声明所引入的变量拥有 *块作用域*
- *块作用域* 开始于其声明点，并终止于该块末尾
- 如果 *内嵌块* 引入了相同的相同名字的声明，则会 *覆盖* 掉外层作用域的同名变量
    - 一句闲话：`python`的语言标准`PEP 8`干脆规定内部作用域里不能声明外部作用域的同名变量
```
int main()
{
    int a = 0;                            // 第一个 a 的作用域开始
    ++a;                                  // 名字 a 在作用域中并指代第一个 a
    {
        int a = 1;                        // 第二个 a 的作用域开始
                                          // 第一个 a 的作用域间断
        a = 42;                           // a 在作用域中并指代第二个 a
    }                                     // 块结束，第二个 a 的作用域结束
                                          //         第一个 a 的作用域恢复
}                                         // 块结束，第一个 a 的作用域结束

int b = a;                                // 错误：名字 a 不在作用域中
```
- 声明于 *异常处理块* 中的名字的作用域开始于其声明点，并在 *该异常处理块* 结束时结束
    - 对 *其他异常处理块* 或 *外围块* **不可见**
```
try 
{   
    f();
} 
catch (const std::runtime_error & re)     // re 的作用域开始
{ 
    int n = 1;                            // n 的作用开始
    std::cout << re.what() << std::endl;  // re 在作用域中
}                                         // re 的作用域结束， n 的作用域结束
catch (std::exception & e) 
{
    std::cout << re.what() << std::endl;  // 错误： re 不在作用域中
    ++n;                                  // 错误： n 不在作用域中
}
```
- 在`for`循环的初始化语句中，在`for`循环的条件中，在范围`for`循环的范围声明中，在`if`语句或`switch`语句的初始化语句中，在`if`语句、`while`循环或`switch`语句的条件中，声明的名字的作用域，开始于其声明点，并结束于控制语句的末尾
```
Base * bp = new Derived;
if (Derived * dp = dynamic_cast<Derived *>(bp))
{
    dp->f();                              // dp 在作用域中
}                                         // dp 的作用域结束
 
for(int n = 0;                            // n 的作用域开始
    n < 10;                               // n 在作用域中
    ++n)                                  // n 在作用域中
{
    std::cout << n << ' ';                // n 在作用域中
}                                         // n 的作用域结束
```
- 特别地：`switch`语句中定义的变量的作用域是**整个`switch`语句**，而不仅是某个单独的`case`！
    - 如果某处一个**带有初值**的变量位于作用域之外，在另一处该变量位于作用域之内，则从前一处跳转至后一处的行为是**非法**的
```
switch (num)
{
case 0:
    // 因为程序的执行流程可能绕开下面的初始化语句，所以此 switch 语句不合法
    std::string filename;                 // 错误：控制流绕过一个隐式初始化的变量
    int i = 0;                            // 错误：控制流绕过一个显式初始化的变量
    int j;                                // 正确：j 没有初始化
    j = 1;                                // 正确：可以给 j 赋值，这样就不是初始化了
    break;

case 1:
{
    std::string filename;                 // 正确：隐式初始化的变量作用域只限于这个 case 下面的块
    int i = 0;                            // 正确：显式初始化的变量作用域只限于这个 case 下面的块
    break;
}
    
case 2:
    // 正确：虽然j在作用域之内，但它没有被初始化
    j = nextNum();                        // 正确：给 j 赋值
    
    if (filename.empty())                 // filename 在作用域内，但没有被初始化
    {
        // do something...
    }
    
    break;
    
default:
    j = 2;
    std::cout <<j << std::endl;
    break;
}
```

#### [函数形参作用域](https://en.cppreference.com/w/cpp/language/scope#Function_parameter_scope)（Function parameter scope）

- 函数 *形参* （包括`lambda`表达式的形参）或函数 *局部预定义变量* 的作用域开始于其声明点
    - 如果是函数 *声明* ，则终止于 *声明符末尾* 
    - 如果是函数 *定义* ，则终止于 *函数体末尾* ，或使用 *函数`try`块* 时，终止于 *最后一个异常处理块末尾* 
```
const int n = 3;
 
int f1(int n,                             // 全局 n 的作用域间断
                                          // 参数 n 的作用域开始
       int y = n);                        // 错误：默认实参涉指了形参
 
int (*(*f2)(int n))[n];                   // OK ：函数形参 n 的作用域终止于其函数声明符的末尾
                                          // 数组声明符中，全局 n 在作用域中
                                          // （这声明了返回 int 的 3 元素数组的指针的函数的指针）

auto (*f3)(int n)->int (*)[n];            // 错误：以参数 n 为数组边界
 
 
int f(int n = 2)                          // n 的作用域开始
try                                       // 函数 try 块
{                                         // 函数体开始
    ++n;                                  // n 在作用域中并指代函数形参
    
    {
        int n = 2;                        // 局部变量 n 的作用域开始
                                          // 函数参数 n 的作用域中断
        ++n;                              // n 在此块中指代局部变量
    }                                     // 局部变量 n 的作用域结束
                                          // 函数参数 n 的作用域恢复
} 
catch (...) 
{
    ++n;                                  // n 在作用域中并指代函数形参
    throw;
}                                         // 最后异常处理块结束，函数形参 n 的作用域结束

int a = n;                                // OK ：名称 n 在作用域中
```

#### [函数作用域](https://en.cppreference.com/w/cpp/language/scope#Function_scope)（Function scope）

- 声明于函数内的`label`（且 *只有* `label`），在 *该函数* 和 *其所有内嵌代码块* 的 *任何位置* 都在作用域中，无论在其自身声明的前后
    - E. Dijkstra: Go To Statement Considered Harmful. *Communications of the ACM (CACM)* (1968) 
```
void f()
{
   {   
       goto label;                        // label 在作用域中，尽管之后才声明
label:;
   }
   
   goto label;                            // label 忽略块作用域
}
 
void g()
{
    goto label;                           // 错误： g() 中 label 不在作用域中
}
```

#### [命名空间作用域](https://en.cppreference.com/w/cpp/language/scope#Namespace_scope)（Namespace scope）

- [*命名空间*](https://en.cppreference.com/w/cpp/language/namespace) 中声明的任何实体的作用域均开始于其声明，并包含
    - 其后所有 *同名命名空间* 
    - 使用了`using`命令 *引入了此实体或整个这个命名空间的域* 
    - 这个命名空间的 *剩余部分* 
- *翻译单元* （文件）的顶层作用域（即所谓的 *文件作用域* 或 *全局作用域* ）亦为命名空间，而被正式称作 *全局命名空间作用域* 
    - 任何声明于 *全局命名空间作用域* 的实体的作用域均开始于其声明，并持续到 *翻译单元的结尾* 
- 声明于 *无名命名空间* 或 *内联命名空间* 的实体的作用域 *包括外围命名空间* 
```
namespace N
{                                         // N 的作用域开始（作为全局命名空间的成员）
    int i;                                // i 的作用域开始
    int g(int a) { return a; }            // g 的作用域开始
    int j();                              // j 的作用域开始
    void q();                             // q 的作用域开始
    
    namespace 
    {
        int x;                            // x 的作用域开始
    }                                     // x 的作用域不结束
    
    inline namespace inl 
    {                                     // inl 的作用域开始
        int y;                            // y 的作用域开始
    }                                     // y 的作用域不结束
}                                         // i, g, j, q, inl, x, y 的作用域间断
 
namespace 
{
    int l = 1;                            // l 的作用域开始
}                                         // l 的作用域不结束（它是无名命名空间的成员）
 
namespace N 
{                                         // i, g, j, q, inl, x, y 的作用域持续
    int g(char a) 
    {                                     // 重载 N::g(int)
        return l + a;                     // 来自无名命名空间的 l 在作用域中
    }
    
    int i;                                // 错误：重复定义（ i 已在作用域中）
    extern int i;                         // OK ：允许重复的变量声明
    int j();                              // OK ：允许重复的函数声明
    
    int j() 
    {                                     // OK ：先前声明的 N::j() 的定义
        return g(i);                      // 调用 N::g(int)
    }
    
    int q();                              // 错误： q 已在作用域中并有不同的返回类型
}                                         // i, g, j, q, inl, x, y 的作用域间断

int main() 
{
    using namespace N;                    // i, g, j, q, inl, x, y 的作用域恢复
    i = 1;                                // N::i 在作用域中
    x = 1;                                // N::(anonymous)::x 在作用域中
    y = 1;                                // N::inl::y 在作用域中
    inl::y = 2;                           // N::inl 亦在作用域中
}                                         // i, g, j, q, inl, x, y 的作用域间断
```

#### [类作用域](https://en.cppreference.com/w/cpp/language/scope#Class_scope)（Class scope）

- 类中声明的名字的作用域开始于其声明点，并包含
    - 类体的 *剩余部分* 
    - 所有 *成员函数体* （无论是否定义于类定义外或在该名字的声明之前）及其 *默认实参* 、 *异常规定* 
    - 类内 *花括号或等号初始化器* 
    - 递归地包括 *嵌套类* 中的所有这些内容
```
class X 
{
    int f(int a = n) 
    {                                     // X::n 在默认实参中在作用域
         return a*n;                      // X::n 在函数体内在作用域中
    }
    
    using r = int;
    r g();
    int i = n*2;                          // X::n 在初始化器内在作用域中
 
    int x[n];                             // 错误： n 在类体内不在作用域中
    static const int n = 1;
    int x[n];                             // OK ： n 现在在类体内在作用域中
};
 
r X::g() 
{
    return n;                             // 错误： r 在类外成员函数的作用域外
}
                                      
auto X::g() -> r                          // OK ：尾随返回类型 X::r 在作用域中
{ 
    return n;                             // X::n 在类外成员函数体的作用域中
}                                 
```
- 任何类成员名只能用于四种语境中
    - 在其自身的 *类作用域* 或在派生类的类作用域之中
    - 在对其类或其派生类的类型的表达式运用 *`.`运算符* 之后
    - 在对其类或其派生的类的指针类型的表达式运用 *`->`运算符* 之后
    - 在对其类或其派生类的名字运用 *`::`运算符* 之后
        - 参见 [*限定标识符*](https://en.cppreference.com/w/cpp/language/identifiers#Qualified_identifiers)（Qualified identifiers）

#### [枚举作用域](https://en.cppreference.com/w/cpp/language/scope#Enumeration_scope)（Enumeration scope）

- *有作用域枚举* （即`enum class T`）中引入的枚举项的名字的作用域开始于其声明点，并 *终止于* `enum` *说明符末尾* 
    - *有作用域枚举* 使得枚举类型必须带着枚举作用域，避免混淆
- *无作用域枚举* （即传统的`enum T`）中引入的枚举项的名字的作用域在`enum` *说明符结尾后仍在作用域中* 
```
enum e1_t 
{                                         // 无作用域枚举
    A,
    B = A * 2
};                                        // A 与 B 的作用域不结束
 
enum class e2_t 
{                                         // 有作用域枚举
    SA,
    SB = SA * 2                           // SA 在作用域中
};                                        // SA 与 SB 的作用域结束
 
e1_t e1 = B;                              // OK ： B 在作用域中
e2_t e2 = SB;                             // 错误： SB 不在作用域中
e2_t e2 = e2_t::SB;                       // OK
```

#### [模板形参作用域](https://en.cppreference.com/w/cpp/language/scope#Template_parameter_scope)（Template parameter scope）

- *模板形参名* 的作用域直接开始于其声明点，并持续到引入它的模板中的 *最小* 的那个的末尾
    - 具体而言，模板形参能用于 *其后的模板形参* 的声明，及 *基类* 的指定，但**不能**用于 *其前的模板形参* 的声明
```
template <typename T,                     // T 的作用域开始
          T * p,                          // T 能用用于非类型形参
          class U = T>                    // T 能用作默认类型
class X : public Array<T>                 // T 能用于基类名
{
                                          // T 还能在体内使用
};                                        // T 与 U 的作用域结束， X 的作用域持续
```
- *模板模板形参的形参名* 的作用域，是该名字出现于其中的 *最小* 模板形参列表
```
template <template <                      // 模板模板形参
                    typename Y,           // Y 的作用域开始
                    typename G = Y        // Y 在作用域中
                   >                      // Y 与 G 的作用域结束
          class T,
          typename U = Y                  // 错误： Y 不在作用域中
          typename U
         >
class X
{
};                                        // T 与 U 的作用域结束
```
- 与其他嵌套作用域类似，模板形参名在其自身的持续期间 *覆盖* 来自外层作用域的相同名字
```
typedef int N;
template <N X,                            // int 类型的非类型模板形参
          typename N,                     // 此 N 的作用域开始，打断 ::N 的作用域
          template<N Y> class T>          // 此处的 N 是模板形参，非 int
struct A;
```

#### [*限定标识符*](https://en.cppreference.com/w/cpp/language/identifiers#Qualified_identifiers)（Qualified identifiers）

- *限定标识表达式* 
    - 在 *无限定标识表达式* 前面带上 *作用域解析运算符*（scope resolution operator）`::`
    - 以及 *可选地* 带上一系列`::`分隔的 *枚举* 、 *类* 或 *命名空间* 名，或`decltype`表达式
    - *全局命名空间* 默认指定，不用特别强调时可以不写
    - 例如
        - `std::string::npos`：指名命名空间`std`中的，类`string`中的， *静态数据成员* `npos`
        - `::tolower`：指名 *全局命名空间* 中的 *函数* `tolower`
        - `::std::cout`：指名 *全局命名空间* 中的，命名空间`std`中的， *全局变量* `cout`
        - `boost::signals2::connection`指名命名空间`boost`中的，命名空间`signals2`中的，类`connection`
- *限定标识符* 中，可能会需要以关键词`template`来消除待决模板名歧义
- 使用示例：如何强行 *访问被覆盖的外层同名变量* 
```
// note: this code is for illustration purposes only and reflects bad practice
// it is generally a bad idea to use the same name for a parameter and a member
size_t shit = 2;

struct Item
{
    void print1(size_t shit) const
    {
        // shit:       function parameter
        // this->shit: class member
        // ::shit:     global one
        printf("%zu %zu %zu\n", shit, this->shit, ::shit);
    }

    size_t shit = 1;
};

Item t;
t.print1(0);  // 0 1 2
```

#### 从作用域和存储期看变量

- *全局非静态变量* 
    - 包括
        - 定义于所有函数之外的非静态变量
    - 具有 *命名空间作用域* （可以通过 *链接* 跨文件），存储于 *静态存储区*
        - 只需在一个源文件中定义，就可以作用于所有的源文件
        - 当其他不包含全局变量定义的源文件中，使用前需用`extern`再次声明这个全局变量
- *全局静态变量* 
    - 包括
        - 定义于所有函数之外的 *静态变量* 
        - 定义于所有函数之外的 *非外连常量* 
    - 具有 *命名空间作用域* （但**不能**跨文件），存储于 *静态存储区*
        - 如果程序包含多个文件，则 *仅作用于定义它的文件* ，**不能**作用于其它文件
        - 天坑：如果两个不同的源文件都定义了相同名字的 *全局静态变量* 或者 *全局非外连常量*， 那么它们是**不同的变量**    
- *局部非静态变量* 
    - 包括
        - 函数体内定义的非静态变量
        - 块语句内定义的非静态变量
        - 函数形参
    - 具有 *块作用域* ，存储于 *自动存储区*
        - 是 *自动对象* （automatic object）
            - 每当函数控制路径经过变量定义语句时创建该对象并初始化，当到达定义所在块末尾时销毁之
            - 自然，只存在于块执行期间
- *局部静态变量* 
    - 具有 *块作用域* ，存储于 *静态存储区*
        - 是 *局部静态对象* （local static object）
            - 在程序的执行路径第一次经过对象定义语句时初始化，并且直到整个程序终止时才被销毁。
            - 在此期间，对象所在函数执行完毕也不会对它有影响。
            - 如没有显式初始化，则会执行 *默认初始化* （内置类型隐式初始化为`0`）
        - 和 *全局变量* 的区别
            - 全局变量对所有的函数可见的
            - 静态局部变量只对定义自己的函数可见
    ```
    size_t countCalls()
    {
        static size_t ctr = 0;  // 调用结束后这个值依然有效，且初始化会且只会在第一次调用时执行一次。
                                // 内置类型的局部静态对象隐式初始化为0，即：这里的显式初始化为0其实是不必要的。
        return ++ctr;
    }
    ```
- *静态函数* 
    - 函数的返回值类型前加上`static`关键字
    - 只在声明它的文件当中可见，**不能**被其它文件使用






### 🌱 [初始化](https://en.cppreference.com/w/cpp/language/initialization)（Initialization）

#### 初始化器（Initializer）

- 任何对象在被创建时都会被 *初始化* 
    1. 构造对象时，可以提供一个 *初始值* 进行 *初始化* ，也可以不提供（此时应用 *默认初始化* 规则）
    2. 在函数调用时也会发生，函数的 *形参* 及 *返回值* 亦会被初始化
- *初始值* 由以下三种 *初始化器* 提供
    1. *括号初始化器* `(expression-list)`：括号包裹、逗号分隔的、由 *表达式* 或 *花括号初始化器* 组成的列表
    2. *等号初始化器* `= expression`：等号后面跟着一个表达式
    3. *花括号初始化器* `{initializer-list}`：花括号包裹、逗号分隔的、由 *表达式* 或 *花括号初始化器* 组成的列表， *可以为空* 
        - `C++11`引入，使用花括号初始化器进行的初始化被称作 *列表初始化* 。如损失精度，则报 *编译错误* 
        ```
        int a = 1;       // ok. 
        int b(1);        // ok. 
        int c = {1};     // ok. list-initialization
        int d{1};        // ok. list-initialization
        int e = {3.14};  // error: type 'double' can not be narrowed down to 'int' in initializer list
        int f{3.14};     // error: type 'double' can not be narrowed down to 'int' in initializer list
        ```
- 根据上下文， *初始化器* 具体可能进行
    1. [*值初始化*](https://en.cppreference.com/w/cpp/language/value_initialization)，例如`std::string s{};`
    2. [*直接初始化*](https://en.cppreference.com/w/cpp/language/direct_initialization)，例如`std::string s("hello");`
    3. [*复制初始化*](https://en.cppreference.com/w/cpp/language/copy_initialization)，例如`std::string s = "hello";`
    4. [*列表初始化*](https://en.cppreference.com/w/cpp/language/list_initialization)，例如`std::string s{'a', 'b', 'c'};`
    5. [*聚合初始化*](https://en.cppreference.com/w/cpp/language/aggregate_initialization)，例如`char a[3] = {'a', 'b'};`
    6. [*引用初始化*](https://en.cppreference.com/w/cpp/language/reference_initialization)，例如`char & c = a[0];`，`std::string s = std::move("hello");`
    7. [*默认初始化*](https://en.cppreference.com/w/cpp/language/default_initialization)，例如`T object;`
- *内置类型* 变量的 *隐式初始化* 
    - *全局* 变量和 *局部静态* 变量： *零初始化* 
    - *局部非静态* 变量：**无**， *值未定义* 
- 这章下面几节是当字典用的，看看就得了

#### [值初始化](https://en.cppreference.com/w/cpp/language/value_initialization)

- 在变量以 *空初始化器* 构造时进行的初始化
```
T()                                     (1)  // 匿名临时量
new T ()                                (2)     
Class::Class(...) : member() { ... }    (3)     
T object {};                            (4)
T{}                                     (5)
new T {}                                (6)  // 匿名临时量   
Class::Class(...) : member{} { ... }    (7)
```
- 初始化流程
    1. 若`T`是 *聚合体* 且使用的是花括号，执行 *聚合初始化* 
    2. 若`T`是**没有** *默认构造函数* ，或拥有 *用户提供的或被删除的默认构造函数* 的类类型，则 *默认初始化* 对象
    3. 若`T`是拥有 *默认构造函数* 的类类型，而 *默认构造函数既非用户提供亦未被删除* （即 *隐式定义* 的或`= default;`的）
        1. *零初始化* 对象的数据成员
        2. 然后，若数据成员拥有 *非平凡的默认构造函数* ，则 *默认初始化* 
    4. 若`T`是 *数组类型* ，则 *值初始化* 数组的 *每个元素* 
    5. 否则， *零初始化* 对象
- 注意事项
    - 若 *构造函数* 是 *用户声明* 的，且**未在**其 *首个声明上显式`= default;`* ，则它是 *用户提供* 的 
    - 语法`T object();`声明的是 *函数* 
    - **不能**值初始化 *引用*
    - 语法`T() (1)`对于 *数组* **禁止**，但允许`T{} (5)`
    - 所有 *标准容器* 在以单个`size_type`实参进行构造、或由对`resize()`的调用而增长时， *值初始化* 其各个元素
    - 对没有用户提供的构造函数而拥有类类型成员的类进行值初始化，其中成员的类拥有用户提供的构造函数，会在调用成员的构造函数前对成员 *清零* 

#### [直接初始化](https://en.cppreference.com/w/cpp/language/direct_initialization)

- 从 *明确的构造函数实参的集合* 初始化对象
```
T object(arg);
T object(arg1, arg2, ...);                                    (1)  // 以括号初始化器初始化
T object {arg};                                               (2)  // 以花括号初始化器初始化
T(other)
T(arg1, arg2, ...)                                            (3)  // 用函数式转型或以带括号的表达式列表初始化  
static_cast< T >( other )                                     (4)     
new T(args, ...)                                              (5)     
Class::Class() : member(args, ...) { ... }                    (6)     
[arg](){ ... }                                                (7)  // lambda表达式中用复制捕获的变量初始化闭包对象的成员
```

#### [复制初始化](https://en.cppreference.com/w/cpp/language/copy_initialization)

- 从 *另一对象* 初始化对象
```
T object = other;                                             (1)     
T object = {other} ;                                          (2)
function(other)                                               (3)  // 函数非引用形参    
return other;                                                 (4)     
throw object;
catch (T object)                                              (5)     
T array[N] = {other};                                         (6)  // 聚合初始化中以初始化提供了初始化器的每个元素   
```

#### [列表初始化](https://en.cppreference.com/w/cpp/language/list_initialization)

- 从 *花括号初始化器* 初始化对象 
    - *直接列表初始化* （考虑`explicit`和非`explicit`构造函数）
    ```
    T object {arg1, arg2, ...};                               (1)     
    T {arg1, arg2, ...}                                       (2)     
    new T{arg1, arg2, ...}                                    (3)     
    Class { T member { arg1, arg2, ... }; };                  (4)   // 在不使用等号的非静态数据成员初始化器中    
    Class::Class() : member{arg1, arg2, ...} {...             (5)   // 构造函数的成员初始化列表中使用花括号初始化器列表
    ```
    - *复制列表初始化* （考虑`explicit`和非`explicit`构造函数，但只调用非`explicit`构造函数） 
    ```
    T object = {arg1, arg2, ...};                             (6)   
    function({arg1, arg2, ...})                               (7)   
    return {arg1, arg2, ... } ;                               (8)   
    object[{arg1, arg2, ... }]                                (9)   
    object = {arg1, arg2, ... }                               (10)  // 赋值表达式中以列表初始化对重载的运算符的形参初始化
    U({arg1, arg2, ... })                                     (11)  // 函数式强制转换表达式或其他构造函数调用
    Class { T member = { arg1, arg2, ... }; };                (12)  // 在使用等号的非静态数据成员初始化器中
    ```

#### [聚合初始化](https://en.cppreference.com/w/cpp/language/aggregate_initialization)

- 从 *花括号初始化器* 初始化 
```
T object = {arg1, arg2, ...};                                 (1)   
T object {arg1, arg2, ...};                                   (2)
```
- 注意事项
    - 当聚合初始化 *联合体* 时，只初始化其 *首个非静态数据成员*
    - 内容不足时后面的元素一律 *零初始化*
    - *聚合体* 是下列类型之一
        - *数组* 类型
        - 符合以下条件的类类型（常为`struct`或`union`） 
            - **无** *私有或受保护的`直接 (since C++17)`非静态数据成员*  
            - 构造函数满足
                - **无**用户提供的构造函数（允许显式预置或弃置的构造函数） `(C++11 ~ C++17)`
                - **无**用户提供、继承或`explicit`构造函数（允许显式预置或弃置的构造函数） `(C++17 ~ C++20)`
                - **无**用户声明或继承的构造函数 `(since C++20)`
            - **无** *`虚、私有或受保护 (since C++17)`基类*
            - **无** *虚成员函数* 
            - **无** *默认成员初始化器* `(C++11 ~ C++14)`


#### [引用初始化](https://en.cppreference.com/w/cpp/language/reference_initialization)

- *绑定引用* 到对象 
```
T & ref = object ;
T & ref = {arg1, arg2, ...};
T & ref(object) ;
T & ref {arg1, arg2, ...} ;                                   (1)   
T && ref = object ;
T && ref = {arg1, arg2, ...};
T && ref (object) ;
T && ref {arg1, arg2, ...} ;                                  (2) 
given R fn(T & arg); or R fn(T && arg);
fn(object)
fn({arg1, arg2, ...})                                         (3)  // 有引用形参的函数的调用表达式
inside T & fn() or T && fn()
return object;                                                (4)  // 当函数返回引用类型时
given T & ref; or T && ref; inside the definition of Class
Class::Class(...) : ref(object) {...}                         (5)  // 以成员初始化器初始化引用类型的非静态数据成员时 
```
- *临时量* 生存期：
    - 一旦引用被绑定到临时量或其子对象，临时量的生存期就 *被延续* 以匹配引用的生存期，但有下列例外
        1. `return`语句中绑定到函数返回值的 *临时量* 不被延续：它立即于返回表达式的末尾销毁。这种函数始终返回悬垂引用 
            - 特例： *右值引用* 及 *常量左值引用* 所绑定对象的 *生存期被延长* 到该引用的作用域结尾，因此返回`std::move(temp)`右值引用是可以的
        2. 在构造函数初始化器列表中绑定到引用成员的 *临时量* ，只持续到构造函数退出前，而非对象存在期间 `(until C++14)`
        3. 在函数调用中绑定到函数形参的 *临时量* ，存在到含这次函数调用的全表达式结尾为止：若函数返回一个引用，而其生命长于全表达式，则它将成为悬垂引用 
        4. 绑定到`new`表达式中所用的初始化器中的引用的 *临时量* ，存在到含该`new`表达式的全表达式结尾为止，而非被初始化对象的存在期间。若被初始化对象的生命长于全表达式，则其引用成员将成为悬垂引用
        5. 绑定到用直接初始化语法（括号），而非列表初始化语法（花括号）初始化的聚合体的引用元素中的引用的 *临时量* ，存在直至含该初始化器的全表达式末尾为止 `(since C++20)`
            ```
            struct A 
            {
                int && r;
            };
            
            A a1{7};   // OK：延续生存期
            A a2(7);   // 良构，但有悬垂引用
            ```
    - 总而言之，临时量的生存期不能以进一步“传递”来延续：从绑定了该临时量的引用初始化的第二引用不影响临时量的生存期。 
- 注意事项
    - 仅在 *函数形参声明* ， *函数返回类型声明* ， *类成员声明* ，以及 *带`extern`说明符* 时， *引用* 可以 *不与初始化器一同出现*  

#### [默认初始化](https://en.cppreference.com/w/cpp/language/default_initialization)

- **不使用** *初始化器* 构造变量时执行的初始化
```
T object;                                                     (1)    
new T                                                         (2)    
```
- 初始化流程
    1. 位于 *静态存储区* 和 *线程局部存储区* 的对象，首先进行 *零初始化* 
    2. 若`T`是 *类类型* ，则考虑各构造函数并实施针对空实参列表的 [*重载决议*](https://en.cppreference.com/w/cpp/language/overload_resolution)（Overload resolution）。调用所选的构造函数（默认构造函数之一），以提供新对象的初始值
    3. 若`T`是 *数组类型* ，则每个数组元素都被 *默认初始化* 
    4. 否则，不做任何事。 *自动对象（及其子对象）* 被初始化为 *不确定值*  
- 注意事项
    - 若`T`是`const`限定类型，则它必须是 *具有用户提供的默认构造函数* 的 *类类型* 
    - **不能**默认初始化 *引用*   






### 🌱 [`cv`限定](https://en.cppreference.com/w/cpp/language/cv)（`cv` (`const` and `volatile`) type qualifiers）

可出现于任何类型说明符中，以指定被声明对象或被命名类型的 *常量性* （constness）或 *易变性* （volatility）。

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






### 🌱 [成员访问操作符](https://en.cppreference.com/w/cpp/language/operator_member_access)（Member access operators）

- 用于访问其操作数的成员，包括
    - *下标* （subscript）：提供对其指针或数组操作数所指向的对象的访问
        - 语法：`a[b]`
        - 可重载
        - 类内定义：`R & T::operator[](S b);`
        - **不可**类外定义
    - *间接寻址* （indirection）：提供对其指针操作数所指向的对象或函数的访问 
        - 语法：`*a`
        - 可重载
        - 类内定义：`R & T::operator*();`
        - 类外定义：`R & operator*(T a);`
    - *取地址* （address-of）：创建指向其对象或函数操作数的指针
        - 语法：`&a`
        - 可重载
        - 类内定义：`R * T::operator&();`
        - 类外定义：`R * operator&(T a);`
    - *对象的成员* （member of object）：提供对其对象操作数的数据成员或成员函数的访问
        - 语法：`a.b`
        - **不可**重载
        - **不可**类内定义
        - **不可**类外定义
    - *对象的成员指针* （pointer to member of object）：提供对其对象操作数的数据成员或成员函数的访问
        - 语法：`a.*b`
        - **不可**重载
        - **不可**类内定义
        - **不可**类外定义
    - *指针的成员* （member of pointer）：提供对其指针操作数所指向的类的数据成员或成员函数的访问
        - 语法：`a->b`
        - 可重载
        - 类内定义：`R * T::operator->();`
        - **不可**类外定义
    - *指针的成员指针* （pointer to member of pointer）：提供对其指针操作数所指向的类的数据成员或成员函数的访问
        - 语法：`a->*b`
        - 可重载
        - 类内定义：`R & T::operator->*(S b);`
        - 类外定义：`R & operator*(T a, S b);`
- 重载 *成员访问操作符* ，返回类型应当与内建运算符所提供的 *返回类型相匹配* ，以便用户定义的运算符可以和内建运算符以相同方式使用
    - 这一点与重载 *运算符* 不同：任何类型都可以作为运算符的返回类型（包括`void`）
    - 一个例外是`operator->`，它必须返回一个 *指针* 或者另一个 *带有重载的`operator->`的类* ，以使其真正可用






### 🌱 [指针声明](https://en.cppreference.com/w/cpp/language/pointer)

- 一网打尽各种指针声明

#### 指针声明

1. `T    * d;`： *指针声明符* ，`d`为指向`T`类型数据的指针
2. `T C::* d;`： *成员指针声明符* ，`d`为指向`C`的`T`类型 *非静态成员* 的指针
```
decl-specifier-seq                       * attr(optional) cv(optional) declarator  (1)
decl-specifier-seq nested-name-specifier * attr(optional) cv(optional) declarator  (2) 
```
- *声明说明符序列* （declarator specifier sequence）：用于说明指针指向的类型
- *嵌套名说明符* （nested name specifier）：由 *名字* 和 *作用域解析运算符* `::`组成的序列
- `attr`： *属性列表* ，可选
- `cv`：应用到 *被声明指针* 的`cv`限定，可选
    - *被指向类型* 的`cv`限定是 *声明说明符序列* 的一部分
- *声明符* （declarator）：除 *引用* 声明符**之外**的任意声明符，可以是另一 *指针* 声明符
    - **无**指向 *引用* 或 *位域* 的指针
    - 允许指向 *指针* 的指针
    - 提到 *指针* 一词时，除特别提及，通常**不**包含指向 *非静态成员* 的指针

#### 指针类型的的值

- 指针类型的值是一定是下列四种情况之一
    - 指向 *对象* 或 *函数* 的指针
        - 该情况下说该指针指向函数或对象
        - 对象的地址为内存中对象所 *占用的首字节* 的地址
    - 对象 *尾后* 指针
        - 为内存中对象所 *占用的存储之后的首字节* 的地址
    - 某类型的 *空* 指针值 
        - `NULL`（就是`0`）或`nullptr`
    - *无效* 指针值 
        - *无效指针值* 的 *任何用法* 均是 *未定义行为* ，尤其点名`diss`如下作大死者
            - 通过 *无效指针值* 间接寻址者
            - 将 *无效指针值* 传递给解分配函数者
- **注意**：两个表示同一地址的指针可能拥有 *不同* 的值
```
struct C 
{
    int x;
    int y;
};

C c;
 
int * px = &c.x;                        // px  的值为 指向 c.x 的指针
int * pxe= px + 1;                      // pxe 的值为  c.x 的尾后指针
int * py = &c.y;                        // py  的值为 指向 c.y 的指针

assert(pxe == py);                      // 测试两个指针是否表示相同地址
                                        // 这条 assert 可能被触发，也可能不被触发
                                        // 至少 g++ version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04) 上实测没有触发
 
*pxe = 1;                               // 即使上面的 assert 未被触发，亦为未定义行为
```

#### 对象指针

- 对 *任何* 对象类型（包含 *指针* 类型）使用 *取址运算符* `&`获得的地址都能用于初始化 *对象指针*
    - 可以直接用 *数组头* 初始化指向 *数组首元素* 的指针
        - *数组* 可隐式转换为 *指针* 
        ```
        int a[2];
        int * p1 = a;                   // 指向数组 a 首元素 a[0]（一个 int）的指针
         
        int b[6][3][8];
        int (*p2)[3][8] = b;            // 指向数组 b 首元素 b[0] 的指针，
                                        // 被指者为 int 的 8 元素数组的 3 元素数组
        ```
        - 数组指针的 *类型别名* ，可用于简化定义
        ```
        int arr[10];

        typedef int (*int_arr_10_ptr_t1)[10];         // 指向长度为 10 的 int 数组的指针类型的别名
        typedef decltype(arr) * int_arr_10_ptr_t2;    // 等价类型别名

        using int_arr_10_ptr_t3 = int[10];            // 等价类型别名
        using int_arr_10_ptr_t4 = decltype(arr) *;    // 等价类型别名
        ```
    - 可以直接用 *派生类的地址* 初始化指向 *基类* 的指针
        - *派生类指针* 可隐式转换 *基类指针* 
        - 若`Derived`是 *多态* 的，则这种指针可用于进行 *虚函数调用* 
    ```
    struct Base {};
    struct Derived : Base {};
     
    Derived d;
    Base * p = &d;
    ```
```
int n;
int * np = &n;                         // int 的指针
int * const * npp = &np;               // 非 const int 的 const 指针的非 const 指针
 
int a[2];
int (*ap)[2] = &a;                     // int 的数组的指针
 
struct S { int n; };
S s = {1};
int * sp = &s.n;                       // 指向作为 s 的成员的 int 的指针
```
- 指针可作为 *内建间接寻址运算符* `*`的操作数，返回指代被指向对象的 *左值* 表达式
```
int n;
int * p = &n;                          // 指向 n 的指针
int & r = *p;                          // 绑定到指代 n 的左值表达式的引用
r = 7;                                 // 存储 int 7 于 n
std::cout << *p << std::endl;          // 左值到右值隐式转换从 n 读取值
```
- 指向类对象的指针亦可作为 [*成员访问运算符*](https://en.cppreference.com/w/cpp/language/operator_member_access#Built-in_pointer-to-member_access_operators) `->`、`->*`的左操作数
- 某些 *加法* 、 *减法* 、 *自增* 和 *自减* 运算符对于指向数组元素的指针有定义
    - 这种指针满足`LegacyRandomAccessIterator`要求，使得`C++` *标准库算法可以用于内建数组* 上
- *比较运算符* 对指针也有定义
    - 数组元素的地址按照下标递增
    - 类的非静态数据成员的地址按照声明顺序递增

#### `void`指针

- 指向 *任意类型* 对象的指针
    - 用于传递 *未知类型* 对象，这在`C`接口中常见
        - `std::malloc`返回`void *`
        - `std::qsort`期待接受两个`const void *`参数的用户提供回调
        - `pthread_create`期待接受并返回`void *`的用户提供的回调
        - 所有情况下， *调用方负责* 在使用前将指针 *转型* 到正确类型
- 普通指针可 *隐式转换* 成`void`指针（`cv`限定可选），**不**改变其值
    - 若原指针指向某 *多态* 类型对象中的 *基类* 子对象，则可用`dynamic_cast`获得指向最终派生类型的完整对象的`void *`
- `void`指针转回原类型
    - *必须* `static_cast`、`reinterpret_cast`或 *显式强转* ，生成其原指针值
```
int n = 1;
int * p1 = &n;
void * pv = p1;
int * p2 = static_cast<int *>(pv);
std::cout << *p2 << std::endl;         // 1
```

#### 函数指针

- 以 *非成员函数* 或 *静态成员函数* 的地址初始化
    - 由于 *函数* 到 *函数指针* 隐式转换的原因，取址运算符是 *可选* 的
    ```
    void f(int);
    void (*p1)(int) = &f;
    void (*p2)(int) = f;               // 与 &f 相同
    ```
    - 与 *数组指针* 的辨析
        - *函数名* 可以 *隐式转化* 成 *函数指针*
        - *数组头* 会 *隐式转化* 成指向 *数组元素类型* 的指针，而**不是**指向 *数组类型* 的指针
    ```
    bool le(int, int);
    bool (*pf1)(int, int) = le;        // 正确，指向函数
    bool (*pf2)(int, int) = &le;       // 正确，指向函数
    
    int arr[10];
    int * p1 = arr;                    // 正确，指向 int
    int * p2 = &arr;                   // 错误
    int *(p3)[10] = &arr;              // 正确，指向数组
    ```
- 不同于 *函数* 或 *函数的引用* ， 函数指针是 *对象* ，从而能 *存储于数组* 、 *被复制* 、 *被赋值* 等
    - 这些玩意儿的解释方法和文法参见开篇章节中的[如何理解`C`声明](https://en.cppreference.com/w/cpp/language/declarations#https://en.cppreference.com/w/cpp/language/declarations#Understanding_C_Declarations)一块儿
```
int (*f)()                  f as pointer to function () returning int
int (*f())()                f as function () returning pointer to function () returning int
int * f()                   f as function returning pointer to int
int (*a[])()                a as array of pointer to function returning int
int (*f())[]                f as function () returning pointer to array of int
int (f[])()                 ARRAY OF FUNCTION IS NOT ALLOWED!!!
                            f as array of function () returning int, which, again, is NOT ALLOWED
int * const *(*g)(float)    g as pointer to function (float) returning pointer to const pointer to int
```
- 函数指针可用作 *函数调用运算符* 的左操作数，这会调用被指向的函数
```
int f(int n)
{
    std::cout << n << std::endl;
    return n * n;
}
 
int (*p)(int) = f;
int x = p(7);                          // 49
```
- *解引用* 函数指针生成标识被指向函数的 *左值*
```
int f();
int (*p)() = f;                        // 指针 p 指向 f
int (&r)() = *p;                       // 将标识 f 的左值绑定到引用

f();                                   // 直接调用函数 f
r();                                   // 通过左值引用调用函数 f
p();                                   // 直接通过指针调用函数 f
(*p)();                                // 通过函数左值调用函数 f
```
- 若 *只有一个重载匹配* 指针类型的话，函数指针可以从可包含函数、函数模板特化及函数模板的一个重载集进行初始化
```
template <typename T> T f(T n) { return n; }
double f(double n) { return n; }
 
int (*p)(int) = f;                     // 实例化并选择 f<int>
```
```
void ff(int*);
void ff(unsigned int);
void (*pf1)(unsigned int) = 0;         // pf1 points to nothing
void (*pf2)(unsigned int) = ff;        // pf1 points to ff(unsigned int)

void (*pf3)(int) = ff;                 // error: no ff with a matching parameter list
double (*pf4)(int*) = ff;              // error: return type of ff and pf4 don't match
```
- *相等比较* 运算符对于函数指针有定义（若指向同一函数则它们比较相等）  

#### （类的）数据成员指针

- 指向类`C`的 *非静态数据成员* `m`的指针，以`&C::m`初始化
    - 这是 *类* 的一个 *附属* ，跟具体的某个对象没关系
    - `C`的 *成员函数* 中，`&(C::m)`、`&m`等**不再是**数据成员指针
- 能用作 [*成员指针访问运算符*](https://en.cppreference.com/w/cpp/language/operator_member_access) `operator.*`、`operator->*`的右操作数
    - 使得每个该类的对象都能用这个 *类的数据成员指针* 访问到自己的数据成员
```
struct C { int m; };

int C::* p = &C::m;                    // pointer to data member m of class C
C c = {7};
std::cout << c.*p << std::endl;        // prints 7

C * cp = &c;
cp->m = 10;
std::cout << cp->*p << std::endl;      // prints 10
```
- 无二义 *非虚基类的数据成员指针* 可以 *隐式转化* 为 *派生类的数据成员指针*
```
struct Base { int m; };
struct Derived : Base {};
 
int Base::* bp = &Base::m;
int Derived::* dp = bp;
Derived d;
d.m = 1;
std::cout << d.*dp << ' ' << d.*bp << std::endl;   // 打印 1 1
```
- *派生类的数据成员指针* 转回无二义 *非虚基类的数据成员指针*
    - *必须* `static_cast` 或 *显式强转* 
    - 即使 *基类* *并无该成员* （但当用该指针访问时，最终派生类中有）亦可
        - 此时用基类对象访问此指针是 *未定义行为*
```
struct Base {};
struct Derived : Base { int m; };
 
int Derived::* dp = &Derived::m;
int Base::* bp = static_cast<int Base::*>(dp);

Derived d;
d.m = 7;
std::cout << d.*bp << std::endl;       // OK：打印 7

Base b;
std::cout << b.*bp << std::endl;       // 未定义行为
```
- 套娃
    - 成员指针的被指向类型也可以是成员指针自身
    - 成员指针可为多级，而且在每级可以有不同的`cv`限定
    - 亦允许指针和成员指针的混合多级组合
```
struct A
{
    int m;
    int A::* const p;                  // const pointer to non-const member
};
 
const A a = {1, &A::m};
 
// non-const pointer to data member which is a const pointer to non-const member
int A::* const A::* p1 = &A::p;
std::cout << a.*(a.*p1) << std::endl;  // prints 1

// regular non-const pointer to a const pointer-to-member
int A::* const* p2 = &a.p;
std::cout << a.**p2 << 'std::endl;     // prints 1
```

#### （类的）成员函数指针

- 指向类`C`的 *非静态成员函数* `f`的指针，以`&C::f`初始化。在 C 的成员函数内，如 &(C::f) 或 &f 这样的表达式不构成成员函数指针。
    - 这是 *类* 的一个 *附属* ，跟具体的某个对象没关系
    - `C`的 *成员函数* 中，`&(C::f)`、`&f`等**不再是**成员函数指针
- 能用作 [*成员指针访问运算符*](https://en.cppreference.com/w/cpp/language/operator_member_access) `operator.*`、`operator->*`的右操作数
    - 使得每个该类的对象都能用这个 *类的数据成员指针* 访问到自己的数据成员
    - 结果表达式 *只能用作* 函数调用运算符的 *左操作数* 
```
struct C
{
    void f(int n) { std::cout << n << '\n'; }
};
 
void (C::* p)(int) = &C::f;            // 指向类 C 的成员函数 f 的指针

C c;
(c.*p)(1);                             // 打印 1

C* cp = &c;
(cp->*p)(2);                           // 打印 2
```
- *基类的成员函数指针* 可以 *隐式转换* 为 *派生类的成员函数指针*
    - 指向同一函数
    - 如果函数是 *多态* 的，则派生类对象调用基类或者派生类成员函数指针都会调用到派生类的
```
struct Base
{
    virtual void f() { std::cout << "Base::f()" << std::endl; }
};


struct Derived: public Base
{
    void f() override { std::cout << "Derived::f()" << std::endl; }
};

void (Base::* bp)() = &Base::f;
void (Derived::* dp)() = bp;

Derived d;
(d.*bp)();                             // Derived::f()
(d.*dp)();                             // Derived::f()
```
- *派生类的成员函数指针* 转回无二义 *基类的成员函数指针*
    - *必须* `static_cast` 或 *显式强转* 
    - 即使 *基类* *并无该成员* （但当用该指针访问时，最终派生类中有）亦可
        - 此时用基类对象访问此指针是 *未定义行为*
```
struct Base {};
struct Derived : Base
{
    void f(int n) { std::cout << n << std::endl; }
};
 
void (Derived::* dp)(int) = &Derived::f;
void (Base::* bp)(int) = static_cast<void (Base::*)(int)>(dp);

Derived d;
(d.*bp)(1);                           // OK：打印 1

Base b;
(b.*bp)(2);                           // 未定义行为
```
- *成员函数指针* 可用作 *回调* 或 *函数对象* 
    - 通常在应用`std::mem_fn`或`std::bind`之后
```
std::vector<std::string> v{"a", "ab", "abc"};
std::vector<std::size_t> l;
std::transform(v.begin(), v.end(), std::back_inserter(l), std::mem_fn(&std::string::size));
for (std::size_t n : l) std::cout << n << std::endl;  // 1 2 3
```

#### 空指针

- 每个类型的指针都拥有一个特殊值，称为该类型的 *空指针值*（null pointer value）
    - 值为 *空* 的指针**不**指向对象或函数
        - 解引用空指针是 *未定义行为* 
        - 与所有 *同类型空指针比较相等* 
    - *空指针* 可用于 
        - 指示对象不存在（例如`function::target()`）
        - 作为其他错误条件的指示器（例如`dynamic_cast`）
        - 通常，接受指针实参的函数始终 *需要检查值是否为空* ，并以不同方式处理该情况（例如，`delete`表达式在传递空指针时不做任何事） 
- 为将指针初始化为 *空* 或赋 *空值* 给既存指针，可以使用下面值
    - *空指针字面量* `nullptr`
    - *空指针常量* `NULL`
    - 从整数值`​0​`的 *隐式转换* 
-  *零初始化* 和 *值初始化* 亦将指针初始化为其 *空* 值

#### 常量性

- 若指针声明中`cv`在`*` *之前* 出现，则它是 *声明说明符序列* 的一部分，并应用到 *被指向的对象* 
- 若指针声明中`cv`在`*` *之后* 出现，则它是 *声明符* 的一部分，并应用到 *所声明的指针自身*  
```
const T *        // pointer to const T
T const *        // pointer to const T
T * const        // const pointet to T
const T * const  // const pointer to const T
T const * const  // const pointet to const T
```
```
// pc is a non-const pointer to const int
// cpc is a const pointer to const int
// ppc is a non-const pointer to non-const pointer to const int
const int ci = 10, *pc = &ci, *const cpc = pc, **ppc;
// p is a non-const pointer to non-const int
// cp is a const pointer to non-const int
int i, *p, *const cp = &i;
 
i = ci;    // okay: value of const int copied into non-const int
*cp = ci;  // okay: non-const int (pointed-to by const pointer) can be changed
pc++;      // okay: non-const pointer (to const int) can be changed
pc = cpc;  // okay: non-const pointer (to const int) can be changed
pc = p;    // okay: non-const pointer (to const int) can be changed
ppc = &pc; // okay: address of pointer to const int is pointer to pointer to const int
 
ci = 1;    // error: const int cannot be changed
ci++;      // error: const int cannot be changed
*pc = 2;   // error: pointed-to const int cannot be changed
cp = &ci;  // error: const pointer (to non-const int) cannot be changed
cpc++;     // error: const pointer (to const int) cannot be changed
p = pc;    // error: pointer to non-const int cannot point to const int
ppc = &p;  // error: pointer to pointer to const int cannot point to
           // pointer to non-const int
```






### 🌱 复合类型（指针和引用）

- 指针`*`以及引用`&`只从属于某个声明符，而不是基本数据类型的一部分
- 指针解引用的结果是被引用对象的左值引用
- 指针或引用的类型必须与其所引用的对象的**类型严格一致**（除下一条的2个例外），即
    - 指针只能用同类型的其他指针（包括字面量或对象强制转换成的指针，以及取值符获取的地址），或者`NULL`、`nullptr`赋值
    - `double`字面量或变量都不能强转成指针
    - 引用只能绑定到同类型的对象上
    - 对于常量，只能绑定常量指针或常引用，不能绑定普通指针或普通引用
- 上一条有2个例外
    - 指针：常量指针指向非常量对象；基类指针指向派生类对象
    - 引用：常引用绑定到任何能转化为本类型常引用的对象（包括字面值）上；基类引用绑定到派生类对象上
        - *常引用* 可以绑定在 *其它类型* 的 *右值* 上。尤其，允许为一个常引用绑定
            - 非常量的对象
            - 非常量的对象
            - 字面值
            - 一般表达式
```
double i = 4.2;
const int & r1 = i;             // 正确：we can bind a const int& to a plain int object
const int & r2 = 4.2;           // 正确：r1 is a reference to const
const int & r3 = i * 2;         // 正确：r3 is a reference to const
int & r4 = i * 2;               // 错误：r4 is a plain reference, not const reference

// 注：执行如下代码时：

double pi = 3.1415926;  
const int & a = pi;             // ok

// 实际上编译器干了这么件事：

int tmp = pi;
const int & a = tmp;

// 如果不是常量引用，改的就不是pi而是临时量tmp，容易造成人祸，因`C++直接规定非常量引用不能绑定给临时量。
```
- *常量指针* （指针指向常量）和 *指针常量* （指针本身是常量）不一样
```
int num = 1;  
const int * p1 = &num;          // 指向`const int`的指针。不能用p1修改num的值，但可以让p1指向别的`(const) int`变量。
int * const p2 = &num;          // 指向`int`的常指针。不能让p1指向别的`int`变量，但可以用p1修改num的值。
const int * const p2 = &num;    // 指向`const int`的常指针。既不能用p1修改num的值，也不可以让p1指向别的`int`变量。
```
- *顶层* `const`和 *底层* `const`
    - *顶层* `const`（Top-level `const`）：任意的对象是常量
    - *底层* `const`（Low-level `const`）：仅限指针或引用。指向的那个对象本身是常量
        - 注意，引用一旦绑定就永远不能改了，因此普通引用以及常引用本身都自带 *顶层* `const`
        - 常量引用永远都是 *底层* `const`
        - 对常量对象取地址获取的指针是 *底层* `const`
        - 非常量引用**不能**绑定到常量对象上
    ```
    int i = 0;
    int * const p1 = &i;        // 顶层const
    const int ci = 42;          // 顶层const
    const int * p2 = &ci;       // 底层const
    const int * const p3 = p2;  // 第一个const为底层，第二个为顶层
    const int & r = ci;         // 常量引用永远都是底层const
    ```






### 🌱 处理类型

#### 类型别名

```
typedef int * intptr;
using intptr2 = int *;

int a = 1;
const intptr p = &a;             // "const (int *)", i.e. `int * const`. NOT `const int *`!!!
const intptr2 p2 = &a, p3 = &a;  // 注意这里p3已经是指针了，不需要再加*
```

#### [`auto`类型说明符](https://en.cppreference.com/w/cpp/language/auto)

- `auto`定义的变量必须有初始值
    - 编译器通过初始值来推算类型
- `auto`一句话定义多个变量时，所有变量类型必须一样
```
auto a = 1, *b = &a;     // 正确，a为int, b为int *
auto sz = 0, pi = 3.14;  // 错误，sz和pi类型不同
```
- 复合类型、常量和`auto`
    - 对于引用，`auto`推导为被引用对象的类型（使用引用实际上是使用被引用的对象，特别是引用被用作初始值时，参与初始化的是被引用对象的值）
    ```
    int a = 0, &r = i;
    auto b = r;                  // b为int，而不是int &
    ```
    - 对于`const`：`auto`会忽略顶层`const`
    ```
    int i = 1;
    const int ci = i, &cr = ci;
    auto b = ci;                 // b为int（ci为顶层const）
    auto c = cr;                 // c为int（cr为ci的别名, ci本身是顶层const）
    auto d = &i;                 // d为int *（&i为const int *，）
    auto e = &ci;                // e为const int *（对常量对象取地址是底层const）
    ```
    - 如果希望`auto`推断出引用或者顶层常量，则声明`auto`时必须加上相应的描述符
    ```
    const auto f = ci;           // f为const int
    auto & g = ci;               // g为const int &
    auto & h = 42;               // 错误：不能为非常量引用绑定字面值
    const auto & j = 42;         // 正确：可以为常量引用绑定字面值
    ```
    - `auto`一句话定义多个变量时，所有变量类型必须一样。注意`*`和`*`是从属于声明符的，而不是基本数据类型的一部分
    ```
    auto k = ci, &l = ci;        // k为int，l为int &
    auto & m = ci, *p2 = &ci;    // m为const int &，p2为const int *
    auto & n = i, *p2 = &ci;     // 错误：i的类型为int，而&ci的类型为const int
    ```

#### [`decltype`类型指示符](https://en.cppreference.com/w/cpp/language/decltype)（`decltype` specifier）

- `decltype(expr)`在不对`expr`进行求值的情况下分析并返回`expr`的数据类型
```
decltype(f()) a = b;         // a的类型就是函数f的返回值类型。同时，这句话并不会调用f()
```
- `decltype(expr)`会 *原样保留* *引用* 以及 *顶层`const`*
    - 引用从来都是作为被引用对象的同义词出现（比如`auto`就不能自动推断出引用），唯一例外是`decltype`
    - 这很符合`decltype`一词在自然语言中的语义，必须原样转发人家本来是什么
```
const int ci = 0, &cj = ci;
decltype(ci) x = 0;          // x为const int
decltype(cj) y = x;          // y为const int &
decltype(cj) z;              // 错误：z为const int &，必须被初始化
```
- `decltype((...))`（双层括号）的结果永远是引用，而`decltype(...)`（单层括号）当且仅当`...`是引用类型时才是引用






### 🌱 [`sizeof`运算符](https://en.cppreference.com/w/cpp/language/sizeof)

`sizeof`运算符返回一条表达式或者一个类型名字所占的字节数，返回类型为`size_t`类型的**常量表达式**。   
`sizeof`运算符满足右结合律。    
`sizeof`并**不实际计算其运算对象的值**。
有两种形式：
```
sizeof(Type)  // 返回类型大小
sizeof expr   // 返回表达式 结果类型 大小
```
`sizeof`运算符的结果部分地依赖于其作用的类型
- 对`char`，或者`char`类型的表达式，执行结果为`1`
- 对 *引用* ，执行结果为**被引用对象所占空间**的大小
- 对 *解引用指针* ，执行结果为**指针指向对象所占空间**大小，指针**不需**有效
    - 无效指针是安全的，因为`sizeof`**不计算其运算对象的值**
- 对 *数组头* ，执行结果为**整个数组所占空间**的大小，等价于对数组中所有元素各自执行一次`sizeof`后再求和。`sizeof`**不会**把数组头转换为指针处理
- 对`std::string`、`std::vector`对象，执行结果为该类型**固定部分**大小，**不会**计算对象中的元素具体占用多大空间






### 🌱 [位域](https://en.cppreference.com/w/cpp/language/bit_field)（Bit Field）

- 位域
    - 声明具有以 *位* （bit，比特）为单位的明确大小的类数据成员
        - 设定成员变量的 *最大宽度* 
            - 用 *范围外的值* *赋值或初始化* 位域是 *未定义行为* 
            - 对位域进行 *自增越过其范围* 是 *未定义行为* 
            - *超越类型极限* 的位域仍 *只容许类型能容纳的最大值* ，剩下的空间就是 *白吃白占* 
                - `C`语言中干脆规定位域的宽度不能超过底层类型的宽度
        - 整个结构的实际大小
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
        - [*属性说明符序列*](https://zh.cppreference.com/w/cpp/language/attributes) ：可选的任何数量属性的序列
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





### 🌱 [值类别](https://en.cppreference.com/w/cpp/language/value_category)（Value Categories）

#### 基本值类别

每个表达式 *只属于* 三种 *基本值类别* 中的一种： 

- *左值* `lvalue`（left value）
    - 包括（看看就得了，当字典用的）
        1. 变量、函数、`模板形参对象 (since C++20)`或数据成员之名，不论其类型，例如`std::cin`或`std::endl`
            - 即使变量的类型是 *右值引用* ，由其名字构成的表达式仍是左值表达式
        2. 返回类型为左值引用的函数调用或重载运算符表达式，例如`std::getline(std::cin, str)`、`std::cout << 1`、`str1 = str2`或`++it`
        3. `a = b`，`a += b`，`a %= b`，以及所有其他内建的赋值及复合赋值表达式
        4. `++a`和`--a`，内建的前置自增与前置自减表达式
        5. `*p`，内建的间接寻址表达式
        6. `a[n]`和`n[a]`，内建的下标表达式，但`a[n]`中的一个操作数应为 *数组左值* 
        7. `a.m`，对象成员表达式，以下两种情况**除外**
            - `m`为 *成员枚举项* 或 *非静态成员函数* （这玩意儿是 *纯右值* 中的一大奇葩）
            - `a`为 *右值* ，而`m`为 *非引用类型的非静态数据成员* （这玩意儿是 *将亡值* ）
        8. `p->m`，内建的指针成员表达式，**除去**`m`为 *成员枚举项* 或 *非静态成员函数* 的情况（ *纯右值* ）
        9. `a.*mp`，对象的成员指针表达式，其中`a`是 *左值* 且`mp`是 *数据成员指针*
        10. `p->*mp`，内建的指针的成员指针表达式，其中`mp`是 *数据成员指针*
        11. `a, b`，内建的逗号表达式，其中`b`是 *左值*
        12. `a ? b : c`，对某些`b`和`c`的三元条件表达式（例如，当它们都是 *同类型左值时* ，但细节见其定义）
        13. 字符串字面量，例如`"Hello, world!"`
        14. 转换为左值引用类型的转型表达式，例如`static_cast<int &>(x)` 
        15. 返回类型是到函数的右值引用的函数调用表达式或重载的运算符表达式
        16. 转换为函数的右值引用类型的转型表达式，如`static_cast<void (&&)(int)>(x)`
    - 性质
        1. 与 *泛左值* 相同（见下文）
        2. 可以由内建的取址运算符 *取地址* ，例如`&++i[1]`,`&std::endl`是合法表达式
        3. 可以 *赋值* 
        4. 可用于 *初始化左值引用* ，这会将一个新名字关联给该表达式所标识的对象
- *将亡值* `xvalue`（expiring value）
    - 包括（看看就得了，当字典用的）
        1. 返回类型为对象的右值引用的函数调用或重载运算符表达式，例如`std::move(x)`
        2. `a[b]`，内建的下标表达式，`a`或`b`是 *数组右值* 
        3. `a.m`，对象成员表达式，`a`是 *右值* ，且`m`是 *非引用类型* 的 *非静态数据成员* 
        4. `a.*mp`，对象的成员指针表达式，`a`为 *右值* ，`mp`为 *数据成员指针* 
        5. `a ? b : c`，对某些`b`和`c`的三元条件表达式（细节见其定义）
        6. *强转右值引用* ，例如`static_cast<T &&>(x)`
        7. *临时量实质化* 后，任何指代 *临时对象* 的表达式 `(since C++17)`
    - 性质
        1. 与 *泛左值* 相同（见下文）
            - 特别是， *亡值* 可以是 *多态* 的，而且 *非类* 的亡值可以有`cv`限定
        2. 与 *右值* 相同（见下文）
            - 特别是， *亡值* 可以绑定到 *右值引用* 上 
- *纯右值* `prvalue`（pure right value）
    - 包括（看看就得了，当字典用的）
        1. （除了 *字符串字面量* **之外**的，这是 *左值* ）字面量，例如`42`、`true`或`nullptr`
        2. 返回类型是非引用的函数调用或重载运算符表达式，例如`str.substr(1, 2)`、`str1 + str2`或`it++`
        3. `a++`和`a--`，内建的后置自增与后置自减表达式
        4. `a + b`、`a % b`、`a & b`、`a << b`，以及其他所有内建的算术表达式
        5. `a && b`、`a || b`、`!a`，内建的逻辑表达式
        6. `a < b`、`a == b`、`a >= b`以及其他所有内建的比较表达式
        7. `&a`，内建的取地址表达式
        8. `a.m`，对象成员表达式，`m`为 *成员枚举项* 或 *非静态成员函数* 
        9. `p->m`，内建的指针成员表达式，其中`m`为 *成员枚举项* 或 *非静态成员函数*
        10. `p->*mp`，内建的指针的成员指针表达式，其中`mp`是 *成员函数指针*
        11. `a, b`，内建的逗号表达式，其中`b`是 *右值*
        12. `a ? b : c`，对某些`b`和`c`的三元条件表达式（细节见其定义）
        13. 转换为非引用类型的转型表达式，例如`static_cast<double>(x)`，`std::string{}`或`(int)42`
        14. `this`指针
        15. 枚举项
        16. 非类型模板形参，除非其类型为`类或 (since C++20)`左值引用类型
        17. `lambda`表达式，例如`[](int x){ return x * x; }` 
        18. `requires`表达式，例如`requires (T i) { typename T::type; }` `(since C++20)`
        19. 概念的特化，例如`std::equality_comparable<int>` `(since C++20)`
    - 性质
        1. 与 *右值* 相同（见下文）
        2. 不能为 *多态* 
            - 它所标识的对象的动态类型 *必须* 为该表达式的类型
        3. *非类非数组的纯右值* **不能**有 *`cv`限定* 
            - 注意：函数调用或转型表达式可能生成非类的`cv`限定类型的纯右值，但其`cv`限定符被立即剥除
        4. **不能**具有 *不完整类型* （除了类型`void`（见下文），或在`decltype`说明符中使用之外）
        5. **不能**具有 *抽象类类型或其数组类型* 

#### 生活中常见的两类复合值类别

- *泛左值* `glvalue`（generalized left value）
    - 包括
        1. *左值* 
        2. *将亡值* 
    - 性质
        1. 可以通过以下三种方式 *隐式转换成纯右值*
            - *左值* 到 *右值* 
            - *数组* 到 *指针* 
            - *函数* 到 *指针* 
        2. 可以是 *多态* 
            - 其所标识的对象的 *动态类型* 不必是该表达式的静态类型
        3. 可以具有 *不完整类型* 
            - 前提是该表达式中容许
    - 一个奇葩
        - 位域（Bit fields）
            - 代表 *位域* 的表达式（例如`a.m`，其中`a`是类型`struct A { int m: 3; }`的 *左值* ）是 *泛左值* 
            - 可用作 *赋值运算符的左操作数* 
            - **不能** *取地址* 
                - **不能**绑定于 *非常量左值引用* 上
                - *常量左值引用* 或 *右值引用* 可以从位域泛左值初始化，但不会直接绑定到位域，而是绑定到一个 *临时副本* 上
- *右值* `rvalue`（right value，如此称呼的历史原因是，右值可以出现于赋值表达式的右边）
    - 包括
        1. *将亡值* 
        2. *纯右值* 
    - 性质
        1. **不能** 由内建的取址运算符 *取地址* ，例如这些表达式非法：`&int()`，`&i++[3]`，`&42`，`&std::move(x)`
        2. **不能** *被赋值* （ *自定义赋值运算符* **除外**）
        3. 可以用于 *初始化常量（左值）引用* 
            - 该右值所标识的对象的 *生存期被延长* 到该引用的作用域结尾 
        4. 可以用于 *初始化右值引用* 
            - 该右值所标识的对象的 *生存期被延长* 到该引用的作用域结尾
            - 当被 *用作函数实参* ，且该函数有同时有（ *右值引用形参* 、 *常量左值引用形参* ）两个版本时，将传入 *右值引用* 
                - 将调用其 *移动构造函数* 
            - *复制* 和 *移动赋值运算符* 与上一条类似
    - 两个奇葩
        - 未决成员函数调用（Pending member function call）
            - 调用 *非静态成员函数* （`a.mf`，`p->mf`，`a.*pmf`，`p->*pmf`，其中`mf`是 *非静态成员函数* ，`pmf`是 [*成员函数指针*](https://en.cppreference.com/w/cpp/language/pointer#Pointers_to_member_functions) ）是 *纯右值* 
            - 但 *只能* 用作 *函数调用运算符的左操作数* ，例如 `(p->*pmf)(args)`
            - **不能** 用作 *任何其他用途* 
        - `void`表达式（`void` expressions）
            - void 表达式**没有** *结果对象* `(since C++17)`
            - 返回`void`的函数调用表达式，强转`void`的转型表达式以及`throw`表达式都是 *纯右值* 
            - 但**不能**用来 *初始化引用* 或者 *用作函数实参* 
            - 可以用在舍弃值的语境，例如
                - 自成一行、作为逗号运算符的左操作数等
                - 返回`void`的函数中的`return`语句
                - `throw`表达式可用作条件运算符`a ? b : c;`的第二个和第三个操作数






### 🌱 类型转换（Conversions）

如果`T`是引用类型，则转换结果为**左**值。

#### [`static_cast`](https://en.cppreference.com/w/cpp/language/static_cast)

- `static_cast<T>(expr)`
    - 用于任何具有明确定义的不包含底层`const`的强制类型转换。结果的值和被转换对象的值可能不同。例如
        - `double`强转`int`
            - 即强制截取整数部分，有精度损失
        - `void *`强转`T *`
            - 其实这一条其实也可以用`reinterpret_cast`，因为`void *`强转`T *`的语义就是强行按照`T *`解释那块内存

#### [`dynamic_cast`](https://en.cppreference.com/w/cpp/language/dynamic_cast)

- `dynamic_cast<T>(expr)`
    - 支持运行时的类型识别 => 19.2

#### [`const_cast`](https://en.cppreference.com/w/cpp/language/const_cast)

- `const_cast<T>(expr)`
    - 用于且只有它能用于改变运算对象的**底层**`const`（cast away the `const`）
        - 即：只能用于指针或引用
        ```
        int b = 2;
        const int c0 = const_cast<const int>(b);                 // 错误：const int类型不是指针或引用
        const int & c1 = const_cast<const int &>(b);             // 正确
        const int & c2 = static_cast<const int &>(b);            // 正确
        const int & c3 = b;                                      // 正确
        ```
    - 只能用于更改`const`属性，不能更改类型
    - 如果`expr`指向的对象**本身不是常量**，则通过`const_cast`获取写权限是合法行为；但如果对象本身是常量，则结果 *未定义* 
    ```
    const char * pc;
    char * p = const_cast<char *>(pc);                           // 正确，但通过p写值是未定义的行为
    char * q = static_cast<char *>(cp);                          // 错误，static_cast不能用于去除const
    static_cast<std::string>(pc);                                // 正确，字符串字面值转换为std::string
    const_cast<std::string>(pc);                                 // 错误，const_cast只能用于去除const
    ```

#### [`reinterpret_cast`](https://en.cppreference.com/w/cpp/language/reinterpret_cast)

- `reinterpret_cast<T>(expr)`
    - 强制编译器按照`T`类型重新解读一块内存
    ```
    int * a = new int(1);
    char * pc = reinterpret_cast<char *>(a);                     // 正确
    std::string s(pc);                                           // 可能会RE，（取决于从a开始多久出现0？）
    ```
    - 需要使用`reinterpret_cast`的典型场景（这些场景不能用`static_cast`）
        - 将指针强转成指针
            - 解析二进制数据流
            ```
            uint8_t dat[12] = {0};                               // 假设这是小端机上的二进制数据流
            dat[0] = 1U;
            dat[4] = 2U;
            dat[8] = 3U;
            uint32_t * arr = reinterpret_cast<uint32_t *>(dat);  // 正确
            uint32_t * arr2 = static_cast<uint32_t *>(dat);      // 错误：uint8_t *转换为uint32_t *是没有明确定义的

            for (size_t i = 0; i < 3; ++i)
            {
                printf("%p %u\n", arr + i, arr[i]);              // 输出：1, 2, 3
            }
            ```
            - 探究数据在内存中的二进制存储
            ```
            float pi = 3.14159;
            int * p1 = reinterpret_cast<int *>(&pi);     
            printf("0x%x\n", *p1);                               // 0x40490fd0
            
            uint32_t r = 0x40490fd0;   
            float * p2 = reinterpret_cast<float *>(&r);
            printf("%f\n", *p2);                                 // 3.141590
            ```
        - 将指针强转成数字（获取具体的地址）
        ```
        int a = 1, 
        int * p = &a;
        size_t b = (size_t) p;                                   // 正确：人见人爱的C风格强转
        size_t b2 = static_cast<size_t>(p);                      // 错误：int *转换为size_t是没有明确定义的
        size_t b3 = reinterpret_cast<size_t>(p);                 // 正确
        ```

#### [显式强制类型转换](https://en.cppreference.com/w/cpp/language/explicit_cast)

- *显式强制类型转换* 使用`C`风格写法和函数式写法，用显式和隐式转换的组合进行类型之间的转换
```
(new_type) expression                   (1)     
new_type(expression)                    (2)     
new_type(expressions)                   (3)     
new_type()                              (4)     
new_type{expression-list(optional)}     (5)  (since C++11)
template-name(expressions(optional))    (6)  (since C++17)
template-name{expressions(optional)}    (7)  (since C++17)
```
- 根据具体位置不同，旧式的强制类型转换的效果与`static_cast`、`const_cast`或`reinterpret_cast`相同。具体来讲，定义为以下各项中第一个成功的
    - `const_cast`
    - `static_cast` (though ignoring access restrictions)
    - `static_cast` (though ignoring access restrictions), then `const_cast`
    - `reinterpret_cast`
    - `reinterpret_cast`, then `const_cast`
```
int * ip;
char * cp = (char *) ip;  // 相当于reinterpret_cast<char *>(ip);
```

#### [用户定义类型转换](https://en.cppreference.com/w/cpp/language/cast_operator)

```
operator conversion-type-id             (1)  // 声明用户定义的转换函数，它参与所有隐式和显式转换
explicit operator conversion-type-id    (2)  // 声明用户定义的转换函数，它仅参与直接初始化和显式转换 (since C++11)

struct X 
{
    operator int() const { return 7; }                  // 隐式转换
    explicit operator int*() const { return nullptr; }  // 显式转换
    operator int(*)[3]() const { return nullptr; }      // 错误：转换类型标识中不允许出现数组运算符
    using arr_t = int[3];
    operator arr_t*() const { return nullptr; }         // 若通过 typedef 进行则 OK
    operator arr_t () const;                            // 错误：不允许任何情况下转换到数组
};
 
X x;

int n = static_cast<int>(x);                            // OK：设 n 为 7
int m = x;                                              // OK：设 m 为 7

int* p = static_cast<int*>(x);                          // OK：设 p 为 null
int* q = x;                                             // 错误：无隐式转换
```

#### [标准转换](https://en.cppreference.com/w/cpp/language/implicit_conversion)（隐式类型转换）

- *标准转换* 就是从一个类型到另一类型的 *隐式转换* ，由 *编译器自动完成* 
- 凡是在语境中使用了某种表达式类型`T1`，但语境不接受该类型，而接受另一类型`T2`的时候，会进行 *隐式转换* ，具体是
    1. 调用以`T2`为形参声明的函数时，以该表达式为实参
    2. 运算符期待`T2`，而以该表达式为操作数
    3. 初始化`T2`类型的新对象，包括在返回`T2`的函数中的`return`语句
    4. 将表达式用于`switch`语句（`T2`为整型类型）
    5. 将表达式用于`if`语句或循环（`T2`为`bool`）
- 仅当存在一个从`T1`到`T2`的无歧义 *隐式转换序列* 时，程序 *良构* （能编译）
- *隐式转换序列* 由下列内容依照这个顺序所构成
    1. 零或一个 *标准转换序列* 
        1. 下列三者中的零或一个：
            - 左值到右值转换
            - 数组到指针转换
            - 函数到指针转换
        2. 零或一个数值提升或数值转换
        3. 零或一个函数指针转换 `(since C++17)`
        4. 零或一个限定调整
    2. 零或一个 *用户定义转换* 
        - 零或一个非`explicit`单实参构造函数或非`explicit`转换函数的调用构成
    3. 零或一个 *标准转换序列* 
        - 当考虑构造函数或用户定义转换函数的实参时， *只允许一个* 标准转换序列（否则将实际上可以将用户定义转换串连起来）
        - 从一个内建类型转换到另一内建类型时， *只允许一个* 标准转换序列






### 🌱 调试帮助

#### `assert`

- 定义`#define NDEBUG`可以关闭`assert`宏检查
- 以下宏可用于细化调试信息
````
__func__
__FILE__
__LINE__
__TIME__
__DATE__
````






### 🌱 [Chap 6] 函数

#### 函数返回值

```
std::string foo(const std::string & word)
{
    return word;  // 生成一个word的副本（copy），返回之。这里有一次拷贝的性能损失 => 使用右值引用可以避免
}
```

- 不要返回局部对象的引用或者指针
- 调用一个返回引用的函数获得左值，否则获得右值
- 列表初始化返回值
    - 如果返回的是内置类型，则大括号内只能有1个值；如果是类，由类定义初始值如何被使用
```
std::vector<std::string> process()
{
    return {"func", "success"};
}
```
- 返回数组指针的若干骚操作
    - 正常写
    ```
    int (*fun(int i))[10];
    ```
    - 使用 *尾置返回值* 
    ```
    auto fun(int i) -> int (*)[10];
    ```
    - 使用`decltype`：在已知要返回的是谁的情况下
    ```
    int odd[] = {1, 3, 5, 7, 9};
    int even[] = {0, 2, 4, 6, 8};
    
    decltype(odd) *arrPtr(int i)        // decltype不负责把数组类型转化成指针。因为不能返回数组，所以要加一个*，返回数组指针
    {
        return (i % 2) ? &odd : &even;  
    }
    ```

#### 函数重载

- 顶层`const`不影响传入的对象，因此以下定义不合法
```
Record lookup(Phone);
Record lookup(const Phone);      // redeclares Record lookup(Phone)

Record lookup(Phone *);
Record lookup(Phone * const);    // redeclares Record lookup(Phone *)
```
- 可以基于底层`const`重载函数
```
// functions taking const and nonconst references or pointers have different parameters
// declarations for four independent, overloaded functions
Record lookup(Account &);        // function that takes a reference to Account
Record lookup(const Account &);  // new function that takes a const reference
Record lookup(Account *);        // new function, takes a pointer to Account
Record lookup(const Account *);  // new function, takes a pointer to const
```
- 类成员函数基于`const`的重载
    - 通过区分成员函数是否为`const`的，我们可以对其进行重载；
    - 原理：编译器可以根据`this`指针参数的底层`const`区分参数类型
- 不允许两个函数除了返回值其余都相同：
```
Record lookup(const Account&);
bool lookup(const Account&);     // error: only the return type is different
```

#### 数组形参

- 数组的两个特殊性质
    - 不允许拷贝数组
    - 使用数组头时（通常）会将其转换成指向数组0号元素的指针
        - 包括`auto`
        - 不包括`decltype`

```
// 尽管形式不同，以下三个函数声明等价，都有一个const int *类型形参
void print(const int *);    
void print(const int[]);
void print(const int[10]);  // 此处长度没有意义。可以传入长度不为10的数组，是合法的
```

#### 可变形参

- 初始化列表（`initializer-list`）：用于所有实参类型相同的函数
```
#include <initializer-list>

std::initializer-list<T> lst;              // 默认初始化。T类型元素的空列表
std::initializer-list<T> lst{a, b, c...};  // lst元素数量和初始值一样多；
                                           // lst的元素是对应初始值的拷贝（copies）；
                                           // 列表中的元素永远、均为常量，不能改变

lst2(lst);                                 // 拷贝或赋值一个初始化列表对象不会拷贝列表中的元素；
lst3 = lst;                                // 拷贝后，原始列表和副本共享元素

lst.size();                                // 列表中的元素数量
lst.begin();                               // 返回指向lst中首元素的指针
lst.end();                                 // 返回指向lst中尾元素下一位置的指针
```

```
void error_msg(std::initializer_list<std::string> il)
{
    for (const std::string & s = il.begin(); s != il.end(); ++s)
    {
        std::cout << *s << " ";
    }
    
    std::cout << std::endl;
}

expected == actual ? error_msg({"functionX", "okay"}) : error_msg({"functionX", expected, actual});
```
- 省略符形参：仅用于`C`和`C++`通用的类型，只能作为函数的最后一个参数
```
#include <cstdarg>

// 三种声明格式：
void foo(parm_list, ...);
void foo(parm_list...);
void foo(...);
```

```
#include <iostream>
#include <cstdarg>
#include <cmath>
 
double sample_stddev(int count, ...) 
{
    double sum = 0;
    std::va_list args1;
    va_start(args1, count);                      // 第二个参数是va_list前面的具名参数的名字
    std::va_list args2;
    va_copy(args2, args1);
    
    for (int i = 0; i < count; ++i) 
    {
        double num = va_arg(args1, double);
        sum += num;
    }
    
    va_end(args1);                               // 调用va_start之后不调用va_end，行为未定义
    double mean = sum / count;
 
    double sum_sq_diff = 0;
    
    for (int i = 0; i < count; ++i) 
    {
        double num = va_arg(args2, double);
        sum_sq_diff += (num-mean) * (num-mean);  
    }
    
    va_end(args2);                               // 调用va_copy之后不调用va_end，行为未定义
    return std::sqrt(sum_sq_diff / count);
}
 
std::cout << sample_stddev(4, 25.0, 27.3, 26.9, 25.7) << std::endl;
```
- 可变参数模板 => 16.4

#### 函数指针形参

- 声明
```
// third parameter is a function type and is automatically treated as a pointer to function
void 
useBigger(const string & s1, 
          const string & s2,
          bool pf(const string &, const string &));
          
// equivalent declaration: explicitly define the parameter as a pointer to function
void 
useBigger(const string & s1, 
          const string & s2,
          bool (*pf)(const string &, const string &));
```
- 使用：传入函数名、函数名手动取地址或者已有的指针均可
```          
// automatically converts the function lengthCompare to a pointer to function
useBigger(s1, s2, lengthCompare);
useBigger(s1, s2, &lengthCompare);
useBigger(s1, s2, pf);
```






### 🌱 [Chap 7] 类的基础概念

#### 合成的默认构造函数（Synthesized default constructor）

- 按如下规则初始化类成员
    - 存在类内初始值，则以其初始化对应成员
        - 类内初始值可接受的语法
        ```
        int a1 = 0;    // 正确
        int a2 = {0};  // 正确
        int a3{0};     // 正确
        int a4(0);     // 错误！
        ```
    - 否则，执行 *默认初始化* 或者 *值初始化* 
- 生成条件：
    - 只有类**没有声明任何构造函数**时，编译器才会自动生成默认构造函数
    - 如果类中包含其他类类型成员，且它没有默认构造函数，则这个类**不能**生成默认构造函数
    - => 13.1.6
- 如果类内包含内置类型或复合类型的变量，则只有当这些成员全部被赋予了类内初始值时，这个类才适合于使用默认构造函数
    - 注意：类成员变量从属于内部作用域，默认初始化是 *未定义* 的，不能指望！
- 类必须包含默认构造函数以便在上述情况下使用。实际应用中，如果提供了其它构造函数，最好也提供一个默认构造函数
    - `= default;`
        - 用于既定义了自己的构造函数，又需要默认构造函数的情况
        - 作为声明写在类内部，则构造函数默认`inline`；或作为定义写在类外部，则构造函数不`inline`

#### [成员初始化器列表](https://en.cppreference.com/w/cpp/language/constructor)（Member initializer lists）

- 初始化器列表接受的初始化语法
    1. `Constructor() : x(?), ... { }`
    2. `Constructor() : x{?}, ... { }`
- 如果成员是`const`、 *引用* 或者 *没有默认构造函数的类类型* ，如没有类内初始值，则 *必须* 在成员初始化器列表中初始化，而**不能**等到函数体中赋值
- 初始化的顺序是按照类成员被声明的顺序，与其在列表中的顺序无关
    - 最好令构造函数初始化列表的顺序与成员声明的顺序保持一致
    - 尽量避免用某些成员初始化其他成员，最好用构造函数的参数作为初始值
- 如果一个构造函数为每一个参数都提供了 *默认实参* ，则它实际上也定义了 *默认构造函数* 
- 某个数据成员被初始化器列表忽略时，则先被 *默认初始化* ，之后再按照构造函数体中的规则进行 *二次赋值*

#### 委托构造函数（delegating constructor）

- 一个委托构造函数使用它所属类的其它构造函数执行它自己的初始化过程，
  或者说它把自己的一些（或全部）职责委托给了其他构造函数
```
struct Item
{
    Item() : Item(0, "")  {}  // delegating constructor
    Item(const int & k, const std::string & v) : key(k), value(v)  {}

    int key;
    std::string value;
}
```

#### 转换构造函数（converting constructor）

- 如果构造函数只接受一个实参，那么它实际上定义了转换为此类类型的 *隐式转换* 机制，有时我们将这种构造函数称作 *转换构造函数* => 14.9
- 能通过一个实参调用的构造函数定义了一条从构造函数的参数类型向类类型隐式转换的规则
- 编译器**只允许一步**隐式类型转换，且转换结果是**临时右值**对象
```
// 错误：需要用户定义的两种转换：
// (1) 把"9-999-99999-9"转换成std::string
// (2) 再把这个（临时的）std::string转换成SalesData
item.combine("9-999-99999-9");

// 正确：显式地转换成std::string，再隐式地转换成SalesData
item.combine(std::string("9-999-99999-9")); 

// 正确：隐式地转换成std::string，再显式地转换成SalesData
item.combine(SalesData("9-999-99999-9")); 
```

#### 显式构造函数（`explicit` constructor）

- 我们可以通过将构造函数声明为`explicit`来抑制构造函数定义的隐式转换
```
class SalesData 
{
public:
    SalesData() = default;
    SalesData(const std::string & s, unsigned n, double p) : 
        bookNo(s), units_sold(n), revenue(p * n)  {}
    explicit SalesData(const std::string & s): bookNo(s)  {}
    explicit SalesData(std::istream &);
};

// 此时，没有任何构造函数能用于隐式地创建`SalesData`对象。
item.combine(nullBook);                                  // 错误，对应构造函数是explicit的
item.combine(std::cin);                                  // 错误，对应构造函数是explicit的
```
- `explicit`**只能在类内声明**，只对一个实参的构造函数有意义
- `explicit`构造函数**只能用于直接初始化**
    - 执行拷贝形式的初始化（使用`=`）时，实际发生了隐式类型转换。此时只能直接初始化，而不能使用`explicit`构造函数
    ```
    SalesData item1(nullBook);                           // 正确
    SalesData item2 = nullBook;                          // 错误
    ```
    - 为转换显式地使用构造函数
    ```
    SalesData item2 = SalesData(nullBook);               // 正确：显式构造的对象
    SalesData item3 = static_cast<SalesData>(std::cin);  // 正确：static_cast可以使用explicit构造函数
    ```
- 标准库中含有显式构造函数的类
    - 接受一个单参数`const char *`的`std::string`构造函数 *不是* `explicit`的
    - 接受容量参数的`std::vector`构造函数**是**`explicit`的

#### 其他构造函数

- => 13
- => 15.7
- => 18.1.3

#### 友元

- 友元不是类的成员，不受`public`、`private`以及`protected`这些访问限制的约束
- 友元**不具有**传递性。每个类**单独**负责控制自己的友元类或友元函数
    - `B`有友元`A`，`C`有友元`B`，则`A`能访问`B`的私有成员，但不能访问`C`的私有成员
- 在类定义开始或结束的地方**集中声明**友元
- *友元函数*
    - 友元函数的声明仅仅是指定访问权限，并不是真正的函数声明。想要使用友元，仍**另需一单独的函数声明**
    - 对于重载函数，必须对特定的函数（特有的参数列表）单独声明
- *友元类*
    - 令一个类成为友元
- *友元成员函数*
    - 令一个类的某个成员函数成为友元
- *友元声明和作用域*
    - 关于这段代码最重要的是：理解友元声明的作用是**影响访问权限**，它本身**并非**普通意义上的函数声明
    - 并不是所有编译器都强制执行关于友元的这一规定
    ```
    struct X
    {
        friend void f()
        { 
            // friend functions can be defined in the class
            // this does NOT serve as declaration, even though this is already a defination
            // to use this function, another declaration is REQUIRED
        }

        X()
        {
            f();     // ERROR: no declaration for f
        } 
        
        void g();
        void h();
    };

    void X::g()
    {
        return f();  // ERROR: f hasn't been declared
    } 

    void f();        // declares the function defined inside X

    void X::h()
    {
        return f();  // OK: declaration for f is now in scope
    } 
    ```

#### 类的类型成员

- 类中的`typedef`和`using`必须先定义后使用
- 一般放在类定义刚开始的地方的`public`区域

#### 常对象、类的常数据成员和类的常成员函数

- 常对象
    ```
    const MyClass inst1;
    MyClass const inst2;
    ```
    - 对象的数据成员的值在对象被调用时不能被改变
    - 必须进行初始化，且能被更新
    - 常对象只能调用常成员函数，不能调用普通成员函数
- 常成员变量
    - 使用`const`关键字修饰的成员变量就是常成员变量
    ```
    const T c;
    T const c;
    ```
    - 注意事项
        - 任何函数都**不可以**对其进行赋值或修改
        - 必须而且只能在构造函数初始化列表中进行初始化
        - 假如类有多个构造函数，必须在 *所有的* 构造函数中都对其进行初始化
- 常成员函数
    ```
    const ret func() const;
    ```
    - `const`是函数类型的一部分，在声明函数和定义函数时都要有`const`关键字，在调用时不必加`const`
    - 一般成员函数可以引用本类中的数据成员，也可以修改非`const`数据成员
    - 常成员函数则只能引用本类中的数据成员，而**不能**修改除 *可变数据成员* 以外任何成员
    - 凡是不修改类数据成员的函数一律定义成常成员函数
    
数据成员              | 普通成员函数          | `const`成员函数
--------------------|---------------------|---------------------
普通数据成员          | 可引用，可修改        | 可引用，**不可**修改 
常数据成员            | 可引用，**不可**修改 | 可引用，**不可**修改 
*常对象* 的数据成员   | 不允许                | 可引用，**不可**修改 


#### 可变数据成员（mutable data member）

- 可用于更改`const`对象的成员
    - `const`对象只能调用`const`成员函数
    - 可变数据成员永远不会是`const`，即使它是`const`对象的成员
    - 任何成员函数，包括`const`成员函数，都可以改变可变数据成员
```
class Screen 
{
public:
    inline const size_t & some_member() const;
    
private:
    mutable size_t access_ctr = 0;                 // may change even in a const object
};

inline const size_t & Screen::some_member() const
{
    return ++access_ctr;                           // keep a count of the calls to any member function
}

const Screen s1;
printf("%zu\n", s1.some_member());                 // 1
```

#### 类的前向声明

- 只声明不定义一个类：`class Item;`
    - 在定义之前，`Item`是 *不完全类型*
    - 不完全类型使用受限
        - 可以定义指向这种类型的指针或引用
        - 可以声明（**不能**定义）以这种类型为参数，或返回值类型的函数
        - **不能**创建这种类型的对象
        - **不能**用引用或指针访问其成员
        - *非静态数据成员* **不能**被声明为这种类型
            - *静态数据成员* 可以！
- 特别地：类可以包含指向 *自身类型* 的引用或指针


#### 类作用域

```
// note: this code is for illustration purposes only and reflects bad practice
// it is generally a bad idea to use the same name for a parameter and a member
size_t shit = 2;

struct Item
{
    void print1(size_t shit) const
    {
        // shit:       function parameter
        // this->shit: class member
        // ::shit:     global one
        printf("%zu %zu %zu\n", shit, this->shit, ::shit);
    }

    size_t shit = 1;
};

Item t;
t.print1(0);  // 0 1 2
```

#### 聚合类（aggregate class）

- *聚合类* 使得用户可以直接访问其成员，并且具有特殊的初始化语法形式
- 当一个类满足如下条件（常为`struct`或`union`）时，我们说它是 *聚合的* 
    - **无** *私有或受保护的`直接 (since C++17)`非静态数据成员*  
    - 构造函数满足
        - **无**用户提供的构造函数（允许显式预置或弃置的构造函数） `(C++11 ~ C++17)`
        - **无**用户提供、继承或`explicit`构造函数（允许显式预置或弃置的构造函数） `(C++17 ~ C++20)`
        - **无**用户声明或继承的构造函数 `(since C++20)`
    - **无** *`虚、私有或受保护 (since C++17)`基类*
    - **无** *虚成员函数* 
    - **无** *默认成员初始化器* `(C++11 ~ C++14)`
- 我们可以提供一个花括号括起来的成员初始值列表，并用它初始化聚合类的成员
```
Entry e = {0, "Anna"};
```
- 与初始化数组元素的规则一样，如果初始值列表中的元素个数少于类的成员数量，则靠后的成员被 *值初始化*
- 初始化列表的元素个数不能超过类的成员数量
- 显示初始化类的对象成员存在 *三个* 明显的缺点
    - 要求类的所有成员都是`public`的；
    - 将正确初始化每个对象的每个成员的责任交给了用户，容易出错
    - 添加或删除一个成员之后，所有的初始化语句都需要更新

#### 字面值常量类

- 常量表达式（const expression）
    - 字面值
        - 算数类型
        - 引用和指针
        - 字面值常量类
        - 19.3
    - 常量表达式：值**不会改变**、并且在**编译过程中就能得到**计算结果的表达式
        - 字面值和用常量表达式初始化的`const`对象也是常量表达式
- `constexpr`变量
    - 允许将变量声明为`constexpr`类型，以便由编译器来验证变量的值是否是一个常量表达式
    - 声明为`constexpr`的变量一定是一个常量，而且必须用常量表达式来初始化
    ```
    constexpr int mf = 20;         // 正确
    constexpr int limit = mf + 1;  // 正确
    constexpr int sz = size();     // 当且仅当size是constexpr函数时，才正确
    ```              
    - 尽管不能使用普通函数作为`constexpr`变量的初始值，但可以使用`constexpr`函数
    - `constexpr`引用和指针
        - 只能绑定到固定地址的变量上（例如 *全局对象* ， *局部静态对象* ）
        - **不能**指向局部非静态对象
        - `constexpr`指针和变量的初始值必须是`nullptr`、`0`或者存储于某个固定地址的对象
        - `constexpr`指针为**顶层**`const`
- `constexpr`函数
    - `constexpr`函数是指能用于常量表达式的函数
    - 定义`constexpr`函数需要遵守
        - 函数的返回类型和所有形参的类型都是字面值类型
        - 函数体中只包含运行时**不执行任何操作**的语句，例如
            - 空语句
            - 类型别名
            - `using`声明
        - 函数体中如有可执行语句，只能是**一条**`return`语句
    ```
    constexpr int new_sz()  { return 42; }
    constexpr int foo = new_sz();           // 正确：foo是常量表达式

    // 如果arg是常量表达式，那么scale(arg)也是常量表达式
    constexpr size_t scale(size_t cnt)  { return new_sz() * cnt; }

    int arr[scale(2)];                      // 正确
    int i = 2;                              // i不是常量表达式
    int a2[scale(i)];                       // 错误：i不是常量表达式，scale(i)也不是
    ```
    - 执行初始化时，编译器把对`constexpr`函数的调用替换成其结果值
    - `constexpr`函数是隐式的`inline`函数；
    - `constexpr`函数、`inline`函数以及模板的**定义和实现都应**写进头文件
- 字面值常量类
    - `constexpr`函数的参数和返回值都必须是字面值类型
      除了算数类型、引用和指针以外，**字面值常量类**也是字面值类型
      和其他类不同，字面值常量类可能含有`constexpr`函数成员
      这样的成员必须符合`constexpr`函数的所有要求，是隐式`const`的
    - 以下类是字面值常量类
        - 数据成员都是字面值类型的聚合类
        - 满足以下要求的非聚合类
            - 数据成员都必须是字面值类型
            - 类必须至少有一个`constexpr`构造函数
            - 如果一个数据成员含有类内初始值，则内置类型成员的初始值必须是常量表达式
              或者如果成员属于某种类类型，则初始值必须使用成员自己的`constexpr`构造函数
            - 类必须使用析构函数的默认定义，该成员负责销毁类的对象
    - `constexpr`构造函数
        - 字面值常量类的构造函数可以是`constexpr`，且必须有至少一个`constexpr`构造函数
        - `constexpr`构造函数可以声明成`= default;`的或者`= delete;`的
        - `constexpr`构造函数的函数体是**空的**
            - 既要满足构造函数的要求（不能有返回语句）
            - 又要满足`constexpr`函数的要求（函数体中如有可执行语句，只能是**一条**`return`语句）
        - `constexpr`构造函数必须初始化**所有**数据成员，初始值或者使用`constexpr`构造函数，或者是一条常量表达式
        - `constexpr`构造函数用于生成`constexpr`对象以及`constexpr`函数的参数或返回类型
        ```
        class Debug 
        {
        public:
            constexpr Debug(bool b = true): hw(b), io(b), other(b) {}
            constexpr Debug(bool h, bool i, bool o): hw(h), io(i), other(o) {}
            
            constexpr bool any() { return hw || io || other; }
            
            void set_io(bool b) { io = b; }
            void set_hw(bool b) { hw = b; }
            void set_other(bool b) { hw = b; }
            
        private:
            bool hw;                                 // hardware errors other than IO errors
            bool io;                                 // IO errors
            bool other;                              // other errors
        };
        
        constexpr Debug io_sub(false, true, false);  // debugging IO
        
        if (io_sub.any())                            // equivalent to if(true)
        {
            std::cerr << "print appropriate error messages" << std::endl;
        }    
            
        constexpr Debug prod(false);                 // no debugging during production
        
        if (prod.any())                              // equivalent to if(false)
        {
            std::cerr << "print an error message" << std::endl;
        }
        ```

#### 类的静态成员

- 声明
    - 通过在成员声明之前加上`static`使得其与类关联在一起
    - 静态成员可以是`public`、`private`或者`protected`的
    - 静态数据成员的类型可以是常量、引用、指针、类类型等等
- 定义
    - 和其他成员函数一样，既可以在类内部，也可以在类外部定义静态成员函数
    - `static`关键字只能出现在类内部的声明语句，在类外部定义静态成员时，**不能**重复
    - 静态成员只能在**类外**定义并初始化，且只能被定义**一次**；除`constexpr`静态成员外，**不能**在类内初始化，**不**由构造函数初始化
        - 要想确保只定义一次，应把静态数据成员的定义与其他非内联函数的定义放在**同一个文件**中
        - 从类名开始，就已经是类的作用域之内了，所以
            - `initRate()`**不用**再加类名
            - 可以访问 *私有成员* 
        ```
        double Account::interestRate = initRate();
        ```
        - 即使一个`constexpr`静态成员在类内部被初始化了，也应该在类外定义一下该成员（此时**不能**再指定初始值）
        ```
        // Account.h 
        
        class Account 
        {
        public:
            static double rate() { return interestRate; }
            static void rate(double);
            
        private:
            static constexpr int period = 30;  // period is a constant expression
            double daily_tbl[period];
        };
        
        // Account.cpp
        
        // definition of a static member with no initializer
        constexpr int Account::period;         // initializer provided in the class definition
        ```
    - 类似于 *全局变量* ，静态数据成员定义在任何函数之外；一旦被定义，就会 *一直存在* 
- 效果
    - 类的静态成员存在于对象之外，对象中不包含任何与静态数据成员相关的数据
    - 类的静态成员被其所有实例共享
    - 静态成员函数也不与对象绑定，**不能**使用`this`指针，**不能**声明成`const`的
- 使用
    - 使用作用域运算度直接访问静态成员
    ```
    double r = Account::rate();
    ```
    - 可以使用类的对象、引用或者指针来访问静态成员
    ```
    Account ac1;
    Account * ac2 = &ac1;
    // equivalent ways to call the static member rate function
    r = ac1.rate();   // through an Account object or reference
    r = ac2->rate();  // through a pointer to an Account object
    ```
    - *成员函数* 不通过作用域运算符就能直接使用静态成员
    ```
    class Account 
    {
    public:
        void calculate() { amount += amount * interestRate; }
        
    private:
        static double interestRate;
    };
    ```
- 静态成员能用于某些场景，而普通成员不能
    - 静态数据成员可以是 *不完全类型* 
        - 特别地，静态数据成员可以是 *其所属的类类型* 
        - 非静态成员只能是 *其所属的类类型* 的 *指针* 或 *引用* 
    ```
    class Bar 
    {
    public:
        // ...
        
    private:
        static Bar mem1;  // ok: static member can have incomplete type
        Bar * mem2;       // ok: pointer member can have incomplete type
        Bar mem3;         // error: data members must have complete type
    }
    ```
    - 可以使用静态成员作为 *默认实参* 
        - 非静态成员不能作为默认实参，因为它的值本身属于对象的一部分
          结果就是无法真正提供一个对象以便从中获取成员的值，从而引发错误
    ```
    class Screen 
    {
    public:
        // bkground refers to the static member
        // declared later in the class definition
        Screen & clear(char = bkground);
        
    private:
        static const char bkground;
    };
    ```






### 🌱 [Chap 8] `I/O`库

- 这章挺没意思的，全篇在讲`<iostream>`，还是[`C`风格`I/O`](https://en.cppreference.com/w/cpp/io/c)用着舒服
    - [`printf`](https://en.cppreference.com/w/c/io/fprintf)
    - [`std::printf`](https://en.cppreference.com/w/cpp/io/c/fprintf)






### 🌱 [Chap 9] [顺序容器](https://en.cppreference.com/w/cpp/container)（Sequential Container）

#### 顺序容器

- [`std::vector`](https://en.cppreference.com/w/cpp/container/vector)：
  可变大小数组。支持快速随机访问。在尾部之外的位置插入删除元素可能很慢
- [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string)：
  与`std::vector`相似，专门用于保存字符。随机访问快。在尾部插入删除速度快
- [`std::deque`](https://en.cppreference.com/w/cpp/container/deque)：
  双端队列。支持快速随机访问。在头尾插入删除元素很快
- [`std::list`](https://en.cppreference.com/w/cpp/container/list)：
  双向链表。只支持双向**顺序**访问。在任何位置插入删除元素都很快
- [`std::foward_list`](https://en.cppreference.com/w/cpp/container/forward_list)：
  单向链表。只支持双向**顺序**访问。在任何位置插入删除元素都很快
- [`std::array`](https://en.cppreference.com/w/cpp/container/array)：
  *固定大小* 数组。支持快速随机访问。**不能**添加删除元素。**支持拷贝赋值**（内置数组不行）
```
std::array<int, 10> ia1; // ten default-initialized ints
std::array<int, 10> ia2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};     // list initialization
std::array<int, 10> ia3 = {42};                               // ia3[0] is 42, remaining elements are 0

int digs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
int cpy[10] = digs;                                           // error: no copy or assignment for built-in arrays
std::array<int, 10> digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
std::array<int, 10> copy = digits;                            // ok: so long as array types match
```
- 注意事项
    - 除`std::array`外，其他容器均提供高效灵活的内存管理
    - 除`std::foward_list`没有`size()`操作（为了达到与手写的单向链表一样的效率）外，其余容器均为常数复杂度
    - 顺序容器构造函数的一个版本接受容器大小参数，它使用了元素类型的**默认**构造函数
      对于没有默认构造函数的类型的容器，构造时还需传递 [*元素初始化器*](https://en.cppreference.com/w/cpp/named_req/Allocator)（Allocator）
    ```
    std::vector<noDefault> v1(10, init);  // 正确：提供了元素初始化器
    std::vector<noDefault> v2(10);        // 错误：必须提供一个元素初始化器
    ```

#### 容器适配器

- [`std::stack`](https://en.cppreference.com/w/cpp/container/stack)（位于头文件`<stack>`中）
    - 默认基于`std::deque`实现
    ```
    // copies elements from deq into stk
    std::stack<int> stk(deq);
    ```
    - 也可以接受除`std::array`以及`std::forward_list`之外的任一顺序容器，封装成一个栈
    ```
    // empty stack implemented on top of vector
    std::stack<std::string, std::vector<std::string>> str_stk;
    // str_stk2 is implemented on top of vector and initially holds a copy of svec
    std::stack<std::string, std::vector<std::string>> str_stk2(svec);
    ```
    - 特有操作
        - `s.pop()`：弹出栈顶元素，但不返回该元素的值
        - `s.push(item)`：压栈一个值为`item`的元素
        - `s.emplace(args)`：压栈一个用`args` *构造* 的元素
        - `s.top()`：返回栈顶元素，但不将其弹出
- [`std::queue`](https://en.cppreference.com/w/cpp/container/queue)（位于头文件`<queue>`中）
    - 默认基于`std::deque`实现，也可以接受除`std::array`、`std::forward_list`以及`std::vector`之外的任一顺序容器
    - 默认 *先入先出* （`FIFO`），队尾入队，队首出队
    - 特有操作
        - `q.pop()`：弹出首元素，但不返回该元素的值
        - `q.front()`：返回首元素，但不将其弹出
        - `q.back()`：返回尾元素，但不将其弹出
        - `q.push(item)`：入队（在队列末尾）一个值为`item`的元素
        - `q.emplace(args)`：入队（在队列末尾）一个用`args` *构造* 的元素
- [`std::priority_queue`](https://en.cppreference.com/w/cpp/container/priority_queue)（位于头文件`<queue>`中）
    - 默认基于`std::vector`实现，也可以接受`std::deque`
    ```
    template <class T,
              class Container = std::vector<T>,
              class Compare = std::less<typename Container::value_type>> 
    class priority_queue;
    ```
    - 标准库的实现默认是 *小顶堆* 。即：若`a < b`，则`a`的优先级比`b`高。若想制造大顶堆，可以：
        - 重载`<`运算符 => 11.2.2
        - 插入相反数
        - 传入谓词
    - 特有操作
        - `q.pop()`：弹出最高优先级元素，但不返回该元素的值
        - `q.top()`：返回最高优先级元素，但不将其弹出
        - `q.push(item)`：（在适当位置）插入一个值为`item`的元素
        - `q.emplace(args)`：（在适当位置）插入一个用`args` *构造* 的元素
- *所有* 容器适配器 *都支持* 的操作和类型
    - `size_type`
    - `value_type`
    - `container_type`：实现此适配器的底层容器的类型
    - `A a`
    - `A a(c)`
    - *关系运算符* ：`==`，`!=`，`<`，`<=`，`>`，`>=`
    - `a.empty()`
    - `a.size()`
    - `std::swap(a, b)`，`a.swap(b)`

#### 容器操作

- 类型别名（均为容器类 *静态* 成员）
    - `iterator`：此类型容器的迭代器类型
        - 对于容器常量，只能获取常量迭代器
    - `const_iterator`：可以读取元素，但不能修改元素的迭代器类型
    - [`reverse_iterator`](https://en.cppreference.com/w/cpp/iterator/reverse_iterator)：按逆序寻址（颠倒了`++`和`--`）的迭代器，**不支持**`std::foward_list`。具体实现为`std::reverse_iterator<Container::iterator>`
    - `const_reverse_iterator`：不能修改的逆序迭代器，**不支持**`std::foward_list`
    - `size_type`：`size_t` aka `unsigned long`，足够保存此种容器类型最大可能容器的大小
    - `difference_type`：`ptrdiff_t` aka `long int`，足够保存两个该容器类型的 *迭代器之间* 的距离
    - `value_type`：元素类型
    - `reference`：元素的左值引用类型，等价于`value_type &`
    - `const_reference`：元素的常引用类型，等价于`const value_type &`
- 构造函数
    - `C c`：默认构造函数，构造空容器
    - `C c1(c2)`：拷贝构造，将`c2`中所有元素拷贝到`c1`
    - `C c(b, e)`：构造`c`，将迭代器`b`和`e`指定的范围内的元素拷贝到`c`。**不支持**`std::array`
        - 将容器初始化为另一容器的拷贝时，两个容器的容器类型和元素类型都必须相同
        - 如果用迭代器指定范围，则仅元素类型相同即可
    - `C c{a, b, c...}`或`C c = {a, b, c...};`：列表初始化
    - 大小相关构造函数
        - 只有顺序容器才接受大小参数； *关联容器* **不支持**
- 赋值与`swap`
    - `c1 = c2`：将`c1`中的元素全部替换为`c2`中的元素
    - `c1 = {a, b, c...}`：将`c1`中的元素替换为列表中的元素。**不支持**`std::array`
    - `a.swap(b)`：交换`a`与`b`中的元素
    - `std::swap(a, b)`：交换`a`与`b`中的元素
- 大小
    - `c.size()`：`c`中元素的数目。常数复杂度。**不支持**`std::foward_list`
    - `c.max_size()`：`c`可保存的最大元素数目
    - `c.empty()`：若`c`中存储了元素，返回`false`，否则返回`true`
- 添加删除元素（**不支持**`std::array`）
    - `c.insert(args)`：将`args`中的元素拷贝进`c`
    - `c.emplace(init)`：使用`inits`构造`c`中的一个元素
    - `c.erase(args)`：删除`args`指定的元素
    - `c.clear()`：删除`c`中所有元素，返回`void`
- 关系运算符
    - `==`，`!=`：所有容器都支持相等和不等运算符
    - `<`，`<=`，`>`，`>=`：关系运算符。**不支持**无序关联容器
- 获取迭代器
    - `c.begin()`，`c.end()`：返回指向`c`的首元素和尾哨兵的迭代器（“尾后迭代器”，off-the-end iterator）
    - `c.cbegin()`，`c.cend()`：返回`const_iterator`
    - `c.rbegin()`，`c.rend()`：返回指向`c`的尾元素和头哨兵的迭代器。**不支持**`std::foward_list`
    - `c.crbegin()`，`c.crend()`：返回`const_reverse_iterator`。**不支持**`std::foward_list`

#### 容器操作可能导致迭代器、引用和指针失效

- 总则
    - 总而言之，容器大小一旦动了，则 *所有* 指向此容器的迭代器、引用和指针都 *可能* 失效，所以一律更新一波才是 *坠吼的* 
    - 此外，永远**不要缓存**尾后迭代器（这玩意常年变来变去），现用现制，用后即弃，`end()`的实现都是很快的
- 辨析
    - 插入元素
        - 对于`std::vector`和`std::string`，如果存储空间 *重新分配* ，则 *所有* 指向此容器的迭代器、引用和指针**都会失效**。如果 *未重新分配* ，指向 *插入位置之前* 的元素的迭代器、 引用和指针 *仍有效* ，但指向 *插入位置之后* 的元素的迭代器、 引用和指针**都会失效**
        - 对于`std::deque`，插入到首尾位置之外的任何位置都会导致 *所有* 指向此容器的迭代器、引用和指针**失效**。如果在首尾位置添加元素，则 *迭代器* 会**失效**，但指向 *存在的元素* 的 *引用* 和 *指针* *仍有效*
        - 对于`std::list`和`std::foward_list`，指向容器的迭代器（包括首前和尾后迭代器）、指针和引用 *仍有效*
    - 删除元素
        - 对于`std::vector`和`std::string`，指向 *被删除元素之前* 的元素的迭代器、 引用和指针 *仍有效* ，但指向 *插入位置之后* 的元素的迭代器、 引用和指针**都会失效**。 *尾后迭代器* **一定会失效**
        - 对于`std::deque`，在首尾位置之外的任何位置删除元素都会导致 *所有* 指向此容器的迭代器、引用和指针**失效**。如果删除 *尾元素* ，则 *尾后迭代器* 也会**失效**，但 *其他* *迭代器* 、 *引用* 和 *指针* *不受影响* ；如果删除 *首元素* ，这些也 *不受影响*
        - 对于`std::list`和`std::foward_list`，指向容器的迭代器（包括首前和尾后迭代器）、指针和引用 *仍有效*
    - `resize()`
        - 如果`resize()`缩小容器，则指向 *被删除元素* 的 *迭代器* 、 *引用* 和 *指针* **失效**
        - 对于`resize()`导致存储空间重新分配（对于`std::vector`，`std::string`以及`std::deque`），则 *所有* 指向此容器的 *迭代器* 、 *引用* 和 *指针* **都会失效**

#### 容器定义和初始化

- `C c`：默认构造函数。如果`C`是一个`std::array`，则`c`中元素按默认方式初始化；否则`c`为空
- `C c1(c2)`，`C c1 = c2`：`c1`初始化为`c2`的拷贝。
                             `c1`和`c2`必须是**相同类型**
                             （即：相同的容器类型和元素类型，对于`std::array`还有相同大小）
- `C c{a, b, c...}`，`C c = {a, b, c...}`：`c`初始化为初始化列表中元素的拷贝。
                                             列表中元素类型必须与`C`兼容。
                                             对于`std::array`，列表中元素数目不大于`array`大小，
                                             遗漏元素一律 *值初始化* 
- `C c(b, e)`：`c`初始化为迭代器`b`和`e`指定范围中的元素的拷贝。
                范围中元素的类型必须与`C`兼容（`std::array`**不适用**）
```
// each container has three elements, initialized from the given initializers
std::list<std::string> authors = {"Milton", "Shakespeare", "Austen"};
std::vector<const char *> articles = {"a", "an", "the"};

std::list<std::string> list2(authors);                                   // ok: types match
std::deque<std::string> authList(authors);                               // error: container types don't match
std::vector<std::string> words(articles);                                // error: element types must match
// ok: converts const char * elements to std::string
std::forward_list<std::string> words(articles.begin(), articles.end());

// copies up to but not including the element denoted by it
std::list<std::string>::iterator it = authors.end();
--it;
std::deque<std::string> authList(authors.begin(), it);
```                   
只有 *顺序容器* （**不包括**`std::array`）的构造函数才能接受大小参数
- `C seq(n)`：`seq`包含`n`个元素，这些元素进行了 *值初始化* 。此构造函数是`explicit`的。**不适用**于`std::string`
- `C seq(n, t)`：`seq`包含`n`个初始化为值`t`的元素
```
std::vector<int> ivec(10, -1);      // 10 int elements, each initialized to -1
list<std::string> svec(10, "hi!");  // 10 strings; each element is "hi!"
std::forward_list<int> ivec(10);    // 10 elements, each initialized to 0
std::deque<std::string> svec(10);   // 10 elements, each an empty string
```

#### 赋值和`swap`

- 以下赋值运算符可用于所有容器
    - `c1 = c2`：将`c1`中的元素替换为`c2`中元素的拷贝。`c1`和`c2`必须具有相同的类型
    - `c = {a, b, c...}`：将`c1`中元素替换为初始化列表中元素的拷贝（`std::array`**不适用**）
    ```
    std::array<int, 10> a1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<int, 10> a2 = {0};                             // OK: elements all have value 0
    a1 = a2;                                                  // OK: replaces elements in a1
    a2 = {0};                                                 // error: cannot assign to an array 
                                                              //        from a braced list
    ```
    - `std::swap(c1, c2);`，`c1.swap(c2);`：交换`c1`和`c2`中的元素。`c1`和`c2`必须具有**相同类型**
        - 除`std::array`**以外**，`swap`不对任何元素进行拷贝，删除或插入操作，因此可以保证 *常数复杂度* 
        - 元素不被移动意味着除`std::array`和`std::string`**以外**，原先的迭代器、指针和引用 *不会失效* 
        - 泛型编程要求：**应当**统一使用 *非成员版本* 的`swap`，即`std::swap(c1, c2);`
    ```
    std::vector<std::string> svec1(10);  // vector with 10 elements
    std::vector<std::string> svec2(24);  // vector with 24 elements
    std::swap(svec1, svec2);
    ```
- `assign`操作**不适用于** *关联容器* 以及`std::array`
    - `seq.assign(b, e);`：将`seq`中的元素替换为迭代器`b`和`e`所表示范围中的元素。迭代器`b`和`e`**不能**指向`seq`中的元素
        - 由于旧元素被 *替换* ，因此传递给`assign`的迭代器**不能**指向调用`assign`的容器
    ```
    std::list<std::string> names;
    std::vector<const char *> oldstyle;
    names = oldstyle;                                  // error: container types don't match
    names.assign(oldstyle.cbegin(), oldstyle.cend());  // ok: can convert from const char * to string
    ```
    - `seq.assign({a, b, c...})`：将`seq`中的元素替换为初始化列表中的元素
    - `seq.assign(n, t)`：将`seq`中的元素替换为`n`个值为`t`的元素
    ```
    // equivalent to: slist1.clear(); 
    //                slist1.insert(slist1.begin(), 10, "Hiya!");
    std::list<std::string> slist1(1); // one element, which is the empty string
    slist1.assign(10, "Hiya!");       // ten elements; each one is Hiya !
    ```
- 赋值运算会导致指向左边容器内部的迭代器、引用和指针**失效**。而`swap`操作将容器内容交换，**不会**导致指向容器的迭代器、引用和指针失效
    - `std::array`和`std::string`**除外**

#### 关系运算符

- 每个容器类型都支持 *相等运算符* `==`和`!=`
- **除** *无序关联容器* **外**，所有容器都支持 *关系运算符* `>`、`>=`、`<`和`<=`
    - *关系运算符* 左右的容器必须为**相同类型**
    - 比较两个容器的方式与`std::string`类似，为 *逐元素字典序* 
    - 只有当其元素类型也定义了相应的比较运算符时，才可以使用 *关系运算符* 来比较两个容器
    ```
    std::vector<int> v1 = {1, 3, 5, 7, 9, 12};
    std::vector<int> v2 = {1, 3, 9};
    std::vector<int> v3 = {1, 3, 5, 7};
    std::vector<int> v4 = {1, 3, 5, 7, 9, 12};
    v1 < v2   // true; v1 and v2 differ at element [2]: v1[2] is less than v2[2]
    v1 < v3   // false; all elements are equal, but v3 has fewer of them;
    v1 == v4  // true; each element is equal and v1 and v4 have the same size()
    v1 == v2  // false; v2 has fewer elements than v1
    ```
    
#### 顺序容器操作

- 插入元素
    - `c.push_back(t)`，`c.emplace_back(args)`：在`c`的尾部创建一个值为`t`或由`args`创建的元素。返回`void`
        - `clang`建议：使用`emplace_back`，**而不是**`push_back`
    - `c.push_front(t)`，`c.emplace_front(args)`：在`c`的头部创建一个值为`t`或由`args`创建的元素。返回`void`
    - `c.insert(p, t)`，`c.emplace(p, args)`：在迭代器`p`指向的元素 *之前* 创建一个值为`t`或由`args`创建的元素。返回指向新添加的元素的迭代器
        - `emplace`函数在容器中直接 *构建* 新元素。传递给`emplace`函数的参数必须与元素类型的构造函数参数相匹配
        ```
        struct Entry 
        {
            Entry() = default;
            Entry(int _k, std::string _v) : k(_k), v(_v) {}
            int k{0};
            std::string v{""};
        };
        
        std::vector<Entry> v;
        
        // 以下等价
        v.push_back(Entry(1, "str1"));
        v.emplace_back(1, "str1");
        
        v.insert(v.end(), Entry(1, "str1"));
        v.emplace(v.end(), 1, "str1");
        ```
    - `c.insert(p, n, t)`：在迭代器`p`指向的元素 *之前* 创建`n`个值为`t`或由`args`创建的元素。返回指向新添加的第一个元素的迭代器
    - `c.insert(p, b, e)`：将迭代器`b`和`e`指定的范围内的元素插入到迭代器`p`指向的元素 *之前* 。`b`和`e`**不能**指向`c`中的元素。返回指向新添加第一个元素的迭代器；若范围为空，则返回`p`
    - `c.insert(p, {a, b, c...})`：将列表`{a, b, c...}`中的 *元素* 插入到迭代器`p`指向的元素 *之前* 。返回指向新添加的第一个元素的迭代器；若列表为空，返回`p`
    - 注意事项
        - 向`std::vector`、`std::string`或`std::deque`插入元素会使 *很多* 指向该容器的 *迭代器* 、 *引用* 和 *指针* **失效**
        - `std::forward_list`有自己专属版本的`insert`和`emplace`，**不支持**`push_back`以及`emplace_back`
        - `std::vector`**不支持**`push_front`以及`emplace_front`
        - `std::string`**不支持**`push_front`以及`emplace_front`
        - `std::array`**不支持**以上全部
- 访问元素
    - `c.back()`：返回`c`中尾元素的 *引用* 。若`c`为空，函数行为 *未定义* 
    - `c.front()`：返回`c`中首元素的 *引用* 。若`c`为空，函数行为 *未定义* 
    - `c[i]`：返回`c`中下标为`i`的元素的 *引用* 。若下标越界，函数行为 *未定义* 
    - `c.at(i)`：返回`c`中下标为`i`的元素的 *引用* 。若下标越界，则抛出`out_of_range`异常
    - 注意事项
        - 对 *空容器* 使用`front()`或者`back()`就像下标越界一样，是一种严重的程序设计**错误**
        - `at()`和 *下标* 操作**只**适用于`std::string`，`std::vector`，`std::deque`以及`std::array`
        - `back()`**不**适用于`std::foward_list`
- 删除元素
    - `c.pop_back()`：删除`c`的尾元素。若`c`为空，则函数行为 *未定义* 。返回`void`
    - `c.pop_front()`：删除`c`的首元素。若`c`为空，则函数行为 *未定义* 。返回`void`
    - `c.erase(p)`：删除迭代器`p`指定的元素，返回指向被删除元素之后元素的迭代器。若`p`指向尾元素，则返回尾后迭代器。若`p`是尾后迭代器，函数行为 *未定义* 
    - `c.erase(b, e)`：删除迭代器`b`和`e`所指定范围内的元素，返回指向最后一个被删除元素之后元素的迭代器。若`e`本身就是尾后迭代器，也返回尾后迭代器
    - `c.clear()`：删除`c`中所有元素。等价于`c.erase(c.begin(), c.end())`。返回`void`
    - 注意事项
        - 删除`std::deque`中 *除首尾之外* 的任何元素都会使 *所有* 指向该容器的 *迭代器* 、 *引用* 和 *指针* **失效**
        - 删除`std::vector`或`std::string`中的任何元素都会使 *指向删除点之后* 的 *迭代器* 、 *引用* 和 *指针* **失效**
        - 上述函数 *并不检查参数* 。必须确保被删除**元素真实存在**
        - `std::forward_list`有自己专属版本的`erase`，**不支持**`pop_back`
        - `std::vector`**不支持**`pop_front`
        - `std::string`**不支持**`pop_front`
        - `std::array`**不支持**以上全部
    ```
    list<int> lst = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto it = lst.begin();
    
    while (it != lst.end())
    {
        if (*it % 2)                        // if the element is odd
        {
            it = lst.erase(it);             // erase this element
        }  
        else
        {
            ++it;
        }  
    }
    ```
- `std::foward_list`的特殊操作
    - `lst.before_begin()`，`lst.cbefore_begin()`：返回头哨兵元素的迭代器和`const_iterator`。**不能**解引用
    - `lst.insert_after(p, t)`：在迭代器`p` *之后* 的位置插入`t`。返回指向最后一个插入元素的迭代器
    - `lst.insert_after(p, n, t)`：在迭代器`p` *之后* 的位置插入`n`个`t`。返回指向最后一个插入元素的迭代器
    - `lst.insert_after(p, b, e)`：在迭代器`p` *之后* 的位置插入迭代器`b`和`e`所指定范围之间的元素。`b`和`e`**不能**指向`lst`内。如果范围为空，则返回`p`。如果`p`是尾后迭代器，则函数行为 *未定义* 
    - `lst.insert_after(p, {a, b, c...})`：在迭代器`p` *之后* 的位置插入列表`{a, b, c...}`中的 *元素*
    - `lst.emplace_after(p, args)`：使用`args`在`p`指定位置 *之后* *创建* 一个元素。返回指向这个新元素的迭代器。如果`p`是尾后迭代器，则函数行为 *未定义* 
    - `lst.erase_after(p)`：删除迭代器`p`指向位置之后的 *一个* 元素
    - `lst.erase_after(b, e)`：删除迭代器`b`和`e`所指定范围之间（不包括`e`）的元素。返回指向被删元素下一个元素的迭代器。如果`p`指向尾元素或者是一个尾后迭代器，则函数行为 *未定义* 
    ```
    forward_list<int> flst = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto prev = flst.before_begin();        // denotes element "off the start" of flst
    auto curr = flst.begin();               // denotes the first element in flst
    
    while (curr != flst.end())              // while there are still elements to process
    { 
        if (*curr % 2)                      // if the element is odd
        {
            curr = flst.erase_after(prev);  // erase it and move curr
        }
        else 
        {
            prev = curr;                    // move the iterators to denote the next
            ++curr;                         // element and one before the next element
        }
    }
    ```
- 内存管理
    - `c.resize(n)`：调整`c`的大小为`n`个元素，若`n < c.size()`，则多出的元素被**丢弃**。若必须添加新元素，则新元素进行 *值初始化* 
    - `c.resize(n, t)`：调整`c`的大小为`n`个元素，若`n < c.size()`，则多出的元素被**丢弃**。若必须添加新元素，则新元素值初始化为`t`
    - `c.shrink_to_fit()`：将`capacity()`减小为与`size()`相同大小
    - `c.capacity()`：不重新分配内存的话，`c`最多可以保存多少元素
    - `c.reserve(n)`：分配至少能容纳`n`个元素的内存空间
        - 并不改变容器中元素的数量，它仅影响`std::vector`预先分配多大空间
        - 当所需空间超过当前容量时，才会改变容量
        - 如果需求大于当前容量，至少分配与需求一样大的空间；反之，什么也不做
    - 注意事项
        - 如果`resize()`缩小容器，则指向 *被删除元素* 的 *迭代器* 、 *引用* 和 *指针* **失效**
        - 对于`resize()`导致存储空间重新分配（对于`std::vector`，`std::string`以及`std::deque`），则 *所有* 指向此容器的 *迭代器* 、 *引用* 和 *指针* **都会失效**
        - `shrink_to_fit()` *只适用于* `std::vector`，`std::string`和`std::deque`
        - `capacity()`和`reserve(n)` *只适用于* `std::vector`和`std::string`
        - `std::array`**不支持**以上全部

#### `std::string`的特殊操作

- 额外构造方法
    - `std::string s(cp, n)`：`s`是`cp`指向的字符数组中前`n`个字符的拷贝，此数组应至少包含`n`个字符，且**必须**以`'\0'`结尾
    - `std::string s(s2, pos2)`：`s`是`std::string s2`从下标`pos2`开始的字符的拷贝。若`pos2 > s2.size()`，构造函数行为 *未定义* 
    - `std::string s(s2, pos2, len2)`：`s`是`std::string s2`从下标`pos2`开始`len2`个字符的拷贝。若`pos2 > s2.size()`，构造函数行为 *未定义* 。不管`len2`的值是多少，构造函数至多拷贝`s2.size() - len2`个字符
    ```
    const char *cp = "Hello World!!!";       // null-terminated array
    char noNull[] = {'H', 'i'};              // not null terminated
    std::string s1(cp);                      // copy up to the null in cp; s1 == "Hello World!!!"
    std::string s2(noNull, 2);               // copy two characters from no_null; s2 == "Hi"
    std::string s3(noNull);                  // undefined: noNull not null terminated
    std::string s4(cp + 6, 5);               // copy 5 characters starting at cp[6]; s4 == "World"
    std::string s5(s1, 6, 5);                // copy 5 characters starting at s1[6]; s5 == "World"
    std::string s6(s1, 6);                   // copy from s1 [6] to end of s1; s6 == "World!!!"
    std::string s7(s1, 6, 20);               // ok, copies only to end of s1; s7 == "World!!!"
    std::string s8(s1, 16);                  // throws an out_of_range exception
    ```
- 取子串
    - `s.substr(idx, n)`：返回一个`std::string`，包含`s`中从下标`idx`开始 *最多* `n`个字符的拷贝。`idx`的默认值是`0`。`n`的默认值是`s.size() - idx`，即拷贝从`idx`开始的全部内容。如果`idx > s.size()`，则抛出`out_of_range`异常；如果`idx + n > s.size()`，则`substr()`会调整数值为`s.size()`
    ```
    std::string s("hello world");
    std::string s2 = s.substr(0, 5);         // s2 = hello
    std::string s3 = s.substr(6);            // s3 = world
    std::string s4 = s.substr(6, 11);        // s3 = world
    std::string s5 = s.substr(12);           // RE: out_of_range exception
    ```
- 额外修改方法
    - `s.insert(idx, args)`：在下标`idx` *之前* 插入`args`指定的字符。返回一个指向`s`的引用
    - `s.insert(iter, args)`：在迭代器`iter`指向位置 *之前* 插入`args`指定的字符。返回指向第一个字符的迭代器
    - `s.erase(idx, len)`：删除从下标`idx`开始的`len`个字符。如果`len`被省略，则一路删到`s`末尾。返回一个指向`s`的引用
    - `s.assign(args)`：将`s`中的字符替换为`args`指定的字符。返回一个指向`s`的引用
    - `s.append(args)`：将`args`追加到`s`。返回一个指向`s`的引用
    - `s.replace(idx, len, args)`：把`s`中从下标`idx`开始的`len`个字符替换为`args`指定的字符。返回一个指向`s`的引用
    - `s.replace(b, e, args)`：把`s`中迭代器`b`和`e`指定的范围内的字符替换为`args`指定的字符。返回一个指向`s`的引用
    - 注意事项
        - `args` *可能可以* 是以下选项之一，具体那个函数不能用谁还是指望`IDE`的语法提示罢
            - `str`：`std::string`，不能与`s`相同 
            - `str, pos, len`：`str`从下标`pos`开始 *最多* `len`个字符的拷贝
            - `cp`：指向以`'\0'`结尾的字符数组
            - `cp, len`：`cp`的 *最多* 前`len`个字符
            - `n, c`：`n`个字符`c`
            - `b, e`：迭代器`b`和`e`指定的范围内的字符
            - `{'a', 'b', 'c'...}`：字符组成的初始化列表
- 字符串搜索
    - `s.find(args)`：查找`args`第一次出现的位置
    - `s.rfind(args)`：查找`args`最后一次出现的位置
    - `s.find_first_of(args)`：查找`args`中 *任何一个字符* 第一次出现的位置
    - `s.find_last_of(args)`：查找`args`中 *任何一个字符* 最后一次出现的位置
    - `s.find_first_not_of(args)`：查找第一个 *不在* `args`中的字符
    - `s.find_last_not_of(args)`：查找最后一个 *不在* `args`中的字符
    - 注意事项
        - `args`**必须**是以下形式之一
            - `c, pos`：从`s`中位置`pos`开始查找字符`c`。`pos`默认值为`0`
            - `s2, pos`：从`s`中位置`pos`开始查找字符串`s2`。`pos`默认值为`0`
            - `cp, pos`：从`s`中位置`pos`开始查找字符指针`cp`指向的以`'\0'`结尾的字符数组。`pos`默认值为`0`
            - `cp, pos, n`：从`s`中位置`pos`开始查找字符指针`cp`指向的以`'\0'`结尾的字符数组的前`n`个字符。`pos`和`n`**没有默认值**
        - 搜索函数返回值
            - 成功，返回`std::string::size_type`（`size_t` aka `unsigned long`），表示匹配发生位置的 *下标* 
            - 失败，返回`std::string::npos` *静态成员* ，类型也为`std::string::size_type`，初始化为值`-1`
    ```
    std::string numbers("0123456789"), name("r2d2");
    std::string::size_type pos = 0;
    
    // each iteration finds the next number in name
    while ((pos = name.find_first_of(numbers, pos)) != std::string::npos) 
    {
        std::cout << "found number at index: " << pos << ", element is " << name[pos] << std::endl;
        ++pos;  // move to the next character
    }
    ```
- 字符串 *字典序* 比较
    - `s.compare(s2)`：比较`s`和`s2`
    - `s.compare(pos1, n1, s2)`：比较`s`中`pos1`开始的`n1`个字符和`s2`
    - `s.compare(pos1, n1, s2, pos2, n2)`：比较`s`中`pos1`开始的`n1`个字符和`s2`中`pos2`开始的`n2`个字符
    - `s.compare(cp)`：比较`s`和`cp`指向的以`'\0'`结尾的字符数组
    - `s.compare(pos1, n1, cp)`：比较`s`中`pos1`开始的`n1`个字符和`cp`指向的以`'\0'`结尾的字符数组
    - `s.compare(pos1, n1, cp, n2)`：比较`s`中`pos1`开始的`n1`个字符和`cp + n2`指向的以`'\0'`结尾的字符数组
- 数值转换
    - `C++`风格字符串转数值定义于`<string>`中
        - 签名集锦
        ```
        int                std::stoi(const std::string & str, std::size_t * pos = 0, int base = 10)
        long               std::stol(const std::string & str, std::size_t * pos = 0, int base = 10)
        long long          std::stoll(const std::string & str, std::size_t * pos = 0, int base = 10)
        unsigned long      std::stoul(const std::string & str, std::size_t * pos = 0, int base = 10)
        unsigned long long std::stoull(const std::string & str, std::size_t * pos = 0, int base = 10)
        float              std::stof(const std::string & str, std::size_t * pos = 0)
        double             std::stod(const std::string & str, std::size_t * pos = 0)
        long double        std::stold(const std::string & str, std::size_t * pos = 0)
        ```
        - 从`str` *第一个* *非空字符* 开始分析
        - 转换成整数
            - `str`有效部分可以包含
                - 正负号（`+`，`-`），可选
                - 八进制符号（`0`），只对`base = 8`或`base = 0`时有效，可选
                - 十六进制符号（`0x`，`0X`），只对`base = 16`或`base = 0`时有效，可选
                - 数字（`[0-9]`）
            - `base`：有效进制为`{0...36}`，其中`0`表示自动检测，结果为`8`，`16`或`10`进制
            - `pos`是 *输出参数* ，用来保存`str`有效部分开始第一个非数值字符的位置。默认值为`0`，即：函数不保存位置
        - 转换成浮点数
            - 从`str` *第一个* *非空字符* 开始分析
            - `str`有效部分可以包含
                - 十进制
                    - 正负号（`+`，`-`），可选
                    - 数字（`[0-9]`）
                    - 小数点（`.`）
                    - 指数（`e`、`E`），底数为`10`，可选
                - 十六进制
                    - 正负号（`+`，`-`），可选
                    - 数字（`[0-9]`）
                    - 十六进制符号（`0x`，`0X`）
                    - 小数点（`.`）
                    - 指数（`p`、`P`），底数为`2`，可选
                - 无穷
                    - 正负号（`+`，`-`），可选
                    - `INF`或`INFINITY`，大小写不敏感
                - `NaN`
                    - 正负号（`+`，`-`），可选
                    - `NAN`或`NAN[a-zA-Z0-9_]*`，大小写不敏感
            - `pos`是 *输出参数* ，用来保存`str`有效部分开始第一个非数值字符的位置。默认值为`0`，即：函数不保存位置
        - 异常
            - 如果`str`参数 *不能转换* 成数值，则抛出`std::invalid_argument`异常
            - 如果转换得到的 *数值溢出* ，则抛出`std::out_of_range`异常
        - 例程
        ```
        size_t idx;
        double res = std::stod("+3.14159pi", &idx);
        printf("%lf %zu\n", res, idx);               // 3.141590 8
        ```
    - 数值转字符串（`<string>`）
    ```
    std::string std::to_string(int value)                 (1)  // std::sprintf(buf, "%d", value)
    std::string std::to_string(long value)                (2)  // std::sprintf(buf, "%ld", value)
    std::string std::to_string(long long value)           (3)  // std::sprintf(buf, "%lld", value)
    std::string std::to_string(unsigned value)            (4)  // std::sprintf(buf, "%u", value)
    std::string std::to_string(unsigned long value)       (5)  // std::sprintf(buf, "%lu", value)
    std::string std::to_string(unsigned long long value)  (6)  // std::sprintf(buf, "%llu", value)
    std::string std::to_string(float value)               (7)  // std::sprintf(buf, "%f", value)
    std::string std::to_string(double value)              (8)  // std::sprintf(buf, "%f", value)
    std::string std::to_string(long double value)         (9)  // std::sprintf(buf, "%Lf", value)
    ```

#### 用于`std::list`和`std::forward_list`的特定算法（容器成员函数）

- 出于性能考虑，[`std::list`](https://en.cppreference.com/w/cpp/container/list)和[`std::forward_list`](https://en.cppreference.com/w/cpp/container/forward_list)应当优先使用 *成员函数版本* 的算法，而**不是**标准库泛型算法
    - 这些算法并**不真正拷贝或移动**元素，只会 *更改指针*
- 列表整合算法，以下算法均返回`void`
    - [`lst.merge(lst2)`](https://en.cppreference.com/w/cpp/container/list/merge)：将来自 *有序列表* `lst2`的元素归并入 *有序列表* `lst`。归并之后`lst2` *变为空* 。排序后满足`*it <= *(it + n) == true`
    - [`lst.merge(lst2, comp)`](https://en.cppreference.com/w/cpp/container/list/merge)：将来自 *有序列表* `lst2`的元素归并入 *有序列表* `lst`。归并之后`lst2` *变为空* 。排序后满足`comp(*it, *(it + n)) == true`
    - [`lst.remove(val)`](https://en.cppreference.com/w/cpp/container/list/remove)：调用`erase`删除掉值为`val`的元素
    - [`lst.remove_if(pred)`](https://en.cppreference.com/w/cpp/container/list/remove)：调用`erase`删除掉满足`pred(*it) == true`的元素
    - [`lst.reverse()`](https://en.cppreference.com/w/cpp/container/list/reverse)：反转列表
    - [`lst.sort()`](https://en.cppreference.com/w/cpp/container/list/sort)：排序，排序后满足`*it <= *(it + n) == true`
    - [`lst.sort(comp)`](https://en.cppreference.com/w/cpp/container/list/sort)：排序，排序后满足`comp(*it, *(it + n)) == true`
    - [`lst.unique()`](https://en.cppreference.com/w/cpp/container/list/unique)：调用`erase`删除同一个值的 *连续* 拷贝，判定标准：`*it1 == *it2`
    - [`lst.unique(pred)`](https://en.cppreference.com/w/cpp/container/list/unique)：调用`erase`删除同一个值的 *连续* 拷贝，判定标准：`pred(*it1, *it2) == true`
- [`std::list<T, Allocator>::splice`](https://en.cppreference.com/w/cpp/container/list/splice)
    - `lst.splice(p, lst2)`：把整个`lst2`中元素 *移动至* `lst`中`p` *之前* 的位置。`O(1)`
    - `lst.splice(p, lst2, p2)`：把`lst2`中`p2`指向的元素 *移动至* `lst`中`p` *之前* 的位置。`O(1)`
    - `lst.splice(p, lst2, b, e)`：把`lst2`中`[b, e)`之间的元素 *移动至* `lst`中`p` *之前* 的位置。如果`lst2`和`lst`是同一列表，`O(1)`；否则，`O(n)`
- [`std::forward_list<T, Allocator>::splice_after`](https://en.cppreference.com/w/cpp/container/forward_list/splice_after)
    - `flst.splice_after(p, lst2)`：把整个`lst2`中元素 *移动至* `flst`中`p` *之后* 的位置。`O(n)`
    - `flst.splice_after(p, lst2, p2)`：把`lst2`中`p2`指向的元素 *移动至* `flst`中`p` *之后* 的位置。`O(1)`
    - `flst.splice_after(p, lst2, b, e)`：把`lst2`中`[b, e)`之间的元素 *移动至* `flst`中`p` *之后* 的位置。如果`lst2`和`flst`是同一列表，`O(1)`；否则，`O(n)`
- 与其他非成员版本的泛型算法不同，列表这些成员函数版本的算法会 *改变底层容器* 
    - 比如`lst.unique`就会真正地 *删除* 连续的重复元素，而`std::unique()`相当于只是个排序算法
    - 比如`lst.merge`会将源列表的元素 *移动* 至目标，也就是说源列表已经空了






### 🌱 [Chap 10.3.2] [`lambda`表达式](https://en.cppreference.com/w/cpp/language/lambda)

#### 概述

- 可以理解为未命名的`inline`函数
- 向函数传递`lambda`时，`lambda`会 *立即执行*
- 编译器实现：当定义`lambda`时
    - 编译器生成一个与此`lambda`对应的新的未命名类类型，与一个该类型的未命名实例（函数对象） => 14.8.1
    - 匿名`lambda`用于传参时，传递的就是现生成的该类的一个临时实例（的拷贝）
    - 用`auto`定义一个用`lambda`初始化的变量时，则定义了一个从`lambda`生成的该类型对象实例
    - 默认情况下，从`lambda`生成的类都包含 *对应所捕获变量* 的 *数据成员* 
    - `lambda`的数据成员和普通的类一样，也在对象被创建时初始化
    ```
    // The lambda expression is a prvalue expression of unique unnamed non-union non-aggregate class type, 
    // known as closure type, 
    // which is declared (for the purposes of ADL) in the smallest block scope, class scope, or namespace scope 
    // that contains the lambda expression. 
    
    // the keyword mutable was not used
    ret ClosureType::operator()(params) const { body }  
    
    // the keyword mutable was used
    ret ClosureType::operator()(params) { body }        
    
    // generic lambda (since C++14)
    template <template-params>
    ret ClosureType::operator()(params) const { body }  
    
    // generic lambda, the keyword mutable was used (since C++14)
    template <template-params>
    ret ClosureType::operator()(params) { body }   
    ```

#### 定义格式

```
auto f1 = [capture_list] (paramater_list) -> return_type { function_body; };

// equivalent type casts
return_type                               (*f2)(paramater_list) = f1;
std::function<return_type (paramater_list)> f3                  = f1;
```

#### 内容物

- 捕获列表
    - 把`lambda`表达式 *所在的函数中的局部非静态变量* 声明在捕获列表里，就可以在`lambda`表达式函数体使用该变量
    - 对于局部静态变量或者全局变量，则**不需捕获**即可使用
    - 捕获方式：与参数传递方式类似，可以是
        - *值捕获* ：捕获被创建时变量的 *拷贝* 
            - *可变 `lambda`*
                - 不加`mutable`参数，则此`lambda`被设置为`Closure`类的 *常成员函数* ，**不能修改**被捕获的变量
                - 如果使用了`mutable`参数，则**不能省略**参数列表
            ```
            size_t v1 = 42;
            auto f1 = [v1]            { return ++v1; };  // error: increment of read-only variable ‘v1’
            auto f2 = [v1] mutable    { return ++v1; };  // error: lambda requires '()' before 'mutable'
            auto f3 = [v1] () mutable { return ++v1; };  // ok
            ```
            - **不能**拷贝`std::ostream`对象，因此捕获它们只能靠引用
        - *引用捕获* ：捕获被创建时变量的 *引用* 
            - 自然，`lambda`中使用的就是被捕获的对象本身，地址是一样的
            - 被引用捕获的变量能否 *修改* 取决于那个变量原先 *是不是常量* 
                - 如果是**常量**，那么捕获的引用就是常引用，自然**不能改**
            - 引用捕获与返回捕获有相同的限制，即：必须确保`lambda`被调用时被引用的对象依然 *存在* 
                - 如果`lambda`在函数结束后被调用，则它引用捕获的变量自然已经不存在了，行为 *未定义*
                - 如果可能，尽量**避免**捕获指针或引用 
        ```
        size_t v1 = 42;
        printf("v1 = %zu @ %p\n", v1, &v1);                                       // v1 = 42 @ 0x7ffcc11095e0

        auto f1 = [v1]  { printf("f1 v1 = %zu @ %p\n", v1, &v1); return v1;   };  
        auto f2 = [&v1] { printf("f2 v1 = %zu @ %p\n", v1, &v1); return ++v1; };  

        v1 = 0;
        size_t j1 = f1();                                                         // f1 v1 = 42 @ 0x7ffcc11095e8
        printf("after f1: v1 = %zu, j1 = %zu\n", v1, j1);                         // after f1: v1 = 0, j1 = 42

        v1 = 0;
        size_t j2 = f2();                                                         // f2 v1 = 0 @ 0x7ffcc11095e0
        printf("after f2: v1 = %zu, j2 = %zu\n", v1, j2);                         // after f2: v1 = 1, j2 = 1
        ```
    - 捕获的声明
        - `[]`：空捕获列表。`lambda`不能使用所在函数中的变量
        - `[identifier_list]`：`identifier_list`是一个逗号分隔的名字列表。默认情况下，捕获列表中的变量都采用值捕获，即被拷贝。名字前如使用`&`，则显示指明该变量采用引用捕获
        - `[&]`： 隐式引用捕获列表。编译器自动引用捕获`lambda`函数体中使用的局部变量
        - `[=]`： 隐式值捕获列表。编译器自动值捕获`lambda`函数体中使用的局部变量
        - `[&, identifier_list]`：混合式引用捕获列表。`identifier_list`是一个逗号分隔的名字列表，包含0至多个变量，变量名前**不能**有`&`。这些变量采用值捕获方式，而其他被隐式捕获的变量则一律采用引用捕获
        - `[=, identifier_list]`：混合式值捕获列表。`identifier_list`是一个逗号分隔的名字列表，包含0至多个变量，**不能**包含`this`，变量名前 *必须* 有`&`。这些变量采用引用捕获方式，而其他被隐式捕获的变量则一律采用值捕获
- 参数列表
    - 对于非可变`lambda`，可以连同括号一起忽略。如忽略，则等价于指定 *空的* 参数列表
    - **不能**有 *默认参数*
- 返回值类型
    - 可以忽略，此时返回值类型由返回的表达式的类型推断而来
        - 如果`lambda`的函数体包含任何单一`return`语句之外的内容，且未指定返回值类型，则返回`void`
    - 如不忽略，则必须使用 *尾置返回* 
    ```
    // ok. refers returning int
    std::transform(vec.begin(), vec.end(), vec.begin(), [] (int i)  
    {
        return i < 0 ? -i : i;  
    });
    
    // error. refers returning void but returns int -- from C++ Primer 5th Edition
    // note: at least on g++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0 this one runs correctly with -std=c++11
    std::transform(vec.begin(), vec.end(), vec.begin(), [] (int i)  
    {
        if (i < 0) return -i; else return i;
    });
    
    // ok. returns int
    std::transform(vec.begin(), vec.end(), vec.begin(), [] (int i)  -> int
    {
        if (i < 0) return -i; else return i;
    });
        ```
- 函数体：必要组成部分
```
auto f = [] { return 42; };
std::cout << f() << std::endl;  // 42

std::stable_sort(vec.begin(), vec.end(), [] (const string & a, const string & b) 
{ 
    return a.size() < b.size(); 
});
```

#### [参数绑定](https://en.cppreference.com/w/cpp/utility/functional/bind)

- 头文件`<functional>`中定义了[`std::bind`](https://en.cppreference.com/w/cpp/utility/functional/bind)
```
auto newCallable = std::bind(callable, arg_list);
```
- `arg_list`
    - 是逗号分隔的参数列表，和`callable`的参数列表一一对应
        - 自然，长度和`callable`的参数列表相同
    - `arg_list`可以包含以下三类东西，代表`callable`在对应位置参数绑定为
        - [*占位符*](https://en.cppreference.com/w/cpp/utility/functional/placeholders) `std::placeholders::_n`（`n`为正整数）：
          `newCallable`被调用时接受的第`n`个参数
        - 普通变量或字面量：该变量的拷贝（即绑定死为 *调用`std::bind()`时* 该变量的值）
        - [`std::ref(obj)`](https://en.cppreference.com/w/cpp/utility/functional/ref)，
          [`std::cref(obj)`](https://en.cppreference.com/w/cpp/utility/functional/ref)：
          该对象的 *引用* 或 *常量引用*
    ```
    #include <functional>
    #include <iostream>
    
    void f(int & n1, int & n2, const int & n3)
    {
        printf("%d %d %d\n", n1, n2, n3);
        ++n1;     // increments the copy of n1 stored in the function object
        ++n2;     // increments the main()'s n2
        // ++n3;  // compile error
    }
    
    int main()
    {
        int n1 = 1, n2 = 2, n3 = 3;
        std::function<void ()> bound_f = std::bind(f, n1, std::ref(n2), std::cref(n3));
        n1 = 10, n2 = 11, n3 = 12;
        printf("%d %d %d\n", n1, n2, n3);  // 10 11 12
        bound_f();                         // 1 11 12                                                     
        printf("%d %d %d\n", n1, n2, n3);  // 10 12 12
    }
    ```
- `newCallable`是一个返回值与`callable`相同、参数个数为`arg_list`中占位符 *最大标号* 数值的函数对象
- 调用`newCallable`时，`newCallable`会调用`callable`
    - `callable`接受的参数为`arg_list`中对应位置的变量   
    - `newCallable`接受的参数 *不一定全部* 被传递给`callable`
        - `n`个占位符标号 *可以不是* `1 ~ n`，可以有空缺
```
void f1(T1 a1, T2 a2, T3 a3, T4 a4);

// signature of f2: void f2(, T2, T2);
auto f2 = std::bind(f1, std::placeholders::_2, std::placeholders::_2, 6, std::placeholders::_3);

// equivalent:
f2(1, 2, 3);
f1(2, 2, 6, 3);

// another example
auto g = std::bind(f, a, b, std::placeholders::_2, c, std::placeholders::_1);

// equivalent:
g(X, Y);
f(a, b, Y, c, X);
```
- 用途：用函数代替列表为空的`lambda`
    - 对于要多次使用的操作，应当编写函数并复用，而不是编写一堆重复的`lambda`
```
bool checkSize(const std::string & s, const std::string::size_type &sz)
{
    return s.size() >= sz;
}

// 此std::bind()调用只有一个占位符，表示`check6`只接受单一参数。
// 占位符出现在`arg_list`的第一个位置，表明`check6`的此参数对应`check_size()`的第一个参数，即const std::string & s。
auto check6 = std::bind(checkSize, _1, 6);

// 相当于bool b1 = checkSize(s, 6);
bool b1 = check6(s);

// 以下调用等价
size_t sz = 6;
auto wc = std::find_if(words.begin(), words.end(), [sz] (const std::string & s) 
{
    return s.size >= sz;
});

auto wc2 = std::find_if(words.begin(), words.end(), std::bind(checkSize, _1, 6));
```  






### 🌱 [Chap 10.4] [迭代器](https://en.cppreference.com/w/cpp/iterator)（Iterator）

- 所有标准库容器都支持迭代器，但只有少数几种才同时支持下标运算符
- 再次强调：`for each`循环内以及使用迭代器时**不能**改变被遍历的容器的大小

#### 迭代器运算符

- `*iter`：返回迭代器`iter`所知元素的**左值**引用
    - 解引用 *非法* 迭代器或者 *尾后* 迭代器是 *未定义行为*
- `iter->mem`：解引用`iter`并获取该元素名为`mem`的成员，等价于`(*iter).mem`
- `++iter`，`iter++`：令`iter`指向容器中的下一个元素
    - 尾后迭代器并不实际指向元素，**不能**递增或递减
    - 至少`g++`允许自减尾后迭代器`--c.end()`获取尾元素
- `--iter`，`iter--`：令`iter`指向容器中的上一个元素
- `iter1 == iter2`，`iter1 != iter2`：判断两个迭代器是否相等（不相等）。
                                      如果两个迭代器指向的是同一个元素，或者它们是同一个容器的尾后迭代器，
                                      则相等；反之，不相等。
- 自然，只有迭代器指向的容器支持相应操作时，才能调用上述操作

#### 迭代器算术运算（Iterator Arithmetic）

- `iter + n`： *多步递进* ，结果仍为迭代器，或指向容器中元素，或指向尾后
- `iter - n`： *多步递进* ，结果仍为迭代器，或指向容器中元素，或指向尾后
- `iter += n`： *多步递进* ，结果仍为迭代器，或指向容器中元素，或指向尾后
- `iter[n]`，`*(iter + n)`： *下标* 运算
- `iter1 - iter2`：两个迭代器之间的距离（`difference_type`），
                   即：将`iter2`向前移动`iter1 - iter2`个元素，将得到`iter1`；
- `<`，`<=`，`>`，`>=`：关系运算符。参与运算的两个迭代器必须是合法的（或指向容器中元素，或指向尾后）。
                        如果前者指向的容器位置在后者指向的容器位置之前，则前者小于后者
- 只有迭代器指向的容器支持 *随机访问* 时，才能调用上述操作
    - 比如：`std::list`、`std::forward_list`的内存都不连续，不能随机访问，因此**不支持**迭代器算术运算

#### 范围访问（Range Access）

- 这些全局函数支持 *容器* 、 *内置数组* 和`std::initializer_list`
- [`std::begin()`](https://en.cppreference.com/w/cpp/iterator/begin)，
  [`std::cbegin()`](https://en.cppreference.com/w/cpp/iterator/begin)，
  [`std::end()`](https://en.cppreference.com/w/cpp/iterator/end)，
  [`std::cend()`](https://en.cppreference.com/w/cpp/iterator/end),
  [`std::rbegin()`](https://en.cppreference.com/w/cpp/iterator/rbegin)，
  [`std::crbegin()`](https://en.cppreference.com/w/cpp/iterator/rbegin)，
  [`std::rend()`](https://en.cppreference.com/w/cpp/iterator/rend)，
  [`std::crend()`](https://en.cppreference.com/w/cpp/iterator/rend)
    - 用于 *容器* ，返回 *迭代器* ；用于 *数组* ，返回 *指针*
    - 带`c`的返回 *常迭代器* 或 *常指针* ，带`r`的返回 *反向迭代器* 
    - 如果容器为空，则`std::begin`和`std::end`返回的是**同一个**迭代器，都是 *尾后迭代器* 
    - 自定义 *构成范围* 的迭代器`begin`和`end`**必须满足**的要求
        - 它们或指向同一容器中的元素，或指向同一容器的尾后
        - `begin <= end`，即：`end`不在`begin`之前
```
std::vector<int> vec{0, 1, 2, 3};
std::vector<int>::iterator iter_beg = std::cbegin(vec);
std::vector<int>::iterator iter_end = std::cend(vec);
std::for_each(iter_beg, iter_end, [] (const int & n) { printf("%d ", i); });

int arr[] = {0, 1, 2, 3};
int * ptr_beg = std::cbegin(arr);
int * ptr_end = std::cend(arr);
std::for_each(ptr_beg, iter_end, [] (const int & n) { printf("%d ", i); });
```
- [`std::size()`](https://en.cppreference.com/w/cpp/iterator/size)，
  [`std::ssize()`](https://en.cppreference.com/w/cpp/iterator/size)，
  [`std::empty()`](https://en.cppreference.com/w/cpp/iterator/empty)
    - 顾名思义，`ssize`是`signed size`，返回的是`ptrdiff_t aka long int`而不是`size_t aka unsigned long`
- [`std::data()`](https://en.cppreference.com/w/cpp/iterator/data)
    - 签名
    ```
    template <class C>
    constexpr auto data(C & c) -> decltype(c.data());

    template <class C>
    constexpr auto data(const C & c) -> decltype(c.data());

    template <class T, std::size_t N>
    constexpr T * data(T (&array)[N]) noexcept;

    template <class E>
    constexpr const E * data(std::initializer_list<E> il) noexcept;
    ```
    - 返回：指向数据块的指针
        - 具体：`return c.data()`或`return array`或`return il.begin()`
    ```
    std::string s("Hello");
    char buf[20] {0};
    strcpy(buf, std::data(s));
    printf("%s\n", buf);        // Hello
    ```

#### 泛型算法约定的几类迭代器

这块《`C++ Primer 5th Edition`》和[`cppreference`](https://en.cppreference.com/w/cpp/iterator)不一样，
就直接从[`cppreference`](https://en.cppreference.com/w/cpp/iterator)上摘抄了。

- 输入迭代器
    - 标准库算法共约定使用以下五类迭代器 
        1. [`LegacyInputIterator`](https://en.cppreference.com/w/cpp/named_req/InputIterator)
        2. [`LegacyForwardIterator`](https://en.cppreference.com/w/cpp/named_req/ForwardIterator)
        3. [`LegacyBidirectionalIterator`](https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator)
        4. [`LegacyRandomAccessIterator`](https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator)
        5. [`LegacyContiguousIterator`](https://en.cppreference.com/w/cpp/named_req/ContiguousIterator)
    - 第`n`类输入迭代器需支持下列到`n + 1`级为止（含）的全部操作
        1. 读（read）
        2. 单步递增（increment (without multiple passes)） 
        3. 多步递增（increment (with multiple passes)）
        4. 递减（decrement）
        5. 随机访问（random access）
        6. 连续存储（contiguous storage）
- 输出迭代器
    - [`LegacyOutputIterator`](https://en.cppreference.com/w/cpp/named_req/OutputIterator)需支持如下操作
        1. 写（write）
        2. 单步递增（increment (without multiple passes)） 
- 同时满足[`LegacyInputIterator`](https://en.cppreference.com/w/cpp/named_req/InputIterator)
  和[`LegacyOutputIterator`](https://en.cppreference.com/w/cpp/named_req/OutputIterator)
  的要求的迭代器称作 *可变迭代器* （mutable iterators）

#### 泛型迭代器操作函数

- [`std::advance()`](https://en.cppreference.com/w/cpp/iterator/advance)
    - 签名
    ```
    template <class InputIt, class Distance>
    void 
    advance(InputIt & it, 
            Distance n);
    ```
    - 将`it`后移`n`步。如果`n < 0`，则进行前移（如果迭代器不支持双向移动，则行为 *未定义* ）
    ```
    std::vector<int> v {3, 1, 4};
    std::vector<int>::iterator vi = v.begin();
    std::advance(vi, 2);
    std::cout << *vi << std::endl;              // 4
    ```
    - 复杂度：`O(n)`，如支持 *随机访问* ，则`O(1)`
- [`std::distance()`](https://en.cppreference.com/w/cpp/iterator/distance)
    - 签名
    ```
    template <class InputIt>
    typename std::iterator_traits<InputIt>::difference_type
    distance(InputIt first, 
             InputIt last);
    ```
    - 返回`first`到`last`的步数（`ptrdiff_t aka long int`）
        - 如果迭代器 *不支持双向访问* ，则如果`first`不能自增到`last`， *行为未定义* 
        - 如果迭代器 *支持双向访问* ，则如果`first`不能自增到`last`、且`last`不能自增到`first`， *行为未定义* 
    ```
    std::vector<int> v {3, 1, 4};
    std::cout << std::distance(v.begin(), v.end()) << std::endl;  // 3
    std::cout << std::distance(v.end(), v.begin()) << std::endl;  // -3
    ```
    - 复杂度：`O(n)`，如支持 *随机访问* ，则`O(1)`
- [`std::next()`](https://en.cppreference.com/w/cpp/iterator/next)
    - 签名
    ```
    template <class ForwardIt>
    ForwardIt 
    next(ForwardIt it,
         typename std::iterator_traits<ForwardIt>::difference_type n = 1)
    {
        return std::advance(it, n);
    }
    ```
    - 返回：`it`的第`n`个后继
    ```
    int a[] {0, 1, 2, 3};
    int * p = std::next(a);
    printf("%d\n", *p);
    ```
    - 复杂度：`O(n)`，如支持 *随机访问* ，则`O(1)`
- [`std::prev()`](https://en.cppreference.com/w/cpp/iterator/prev)
    - 签名
    ```
    template <class ForwardIt>
    ForwardIt 
    prev(ForwardIt it,
         typename std::iterator_traits<ForwardIt>::difference_type n = 1)
    {
        return std::advance(it, -n);
    }
    ```
    - 返回：`it`的第`n`个前驱
    - 复杂度：`O(n)`，如支持 *随机访问* ，则`O(1)`

#### 迭代器适配器（Iterator Adaptors）

- *插入迭代器* （insert iterator）
    - 通过插入器进行赋值时，插入器调用容器操作向指定容器的指定位置插入元素
    - 支持操作
        - `it = t`：在`it`所指位置 *之前* 插入值元素`t`
            - 根据具体容器，可能会调用`c.push_back(t)`，`c.push_front(t)`或`c.insert(t, iter)`
            - 自然，只有在 *容器支持该操作* 的情况下，才能使用对应的插入器
        - `*it`，`++it`，`it++`：这些操作**并不做任何事，一律返回`it`自己！！！**
    - 插入器有如下三种
        - [`std::back_insert_iterator`](https://en.cppreference.com/w/cpp/iterator/back_insert_iterator)
            - 生成：[`std::back_inserter()`](https://en.cppreference.com/w/cpp/iterator/back_inserter)
            ```
            template <class Container>
            std::back_insert_iterator<Container> back_inserter(Container & c)
            {
                return std::back_insert_iterator<Container>(c);
            }
            ```
            - 通过此迭代器赋值时，赋值运算符调用`c.push_back()`将一个具有给定值的元素添加到容器中
            ```
            std::vector<int> vec;                         // empty vector
            std::back_insert_iterator<std::vector<int>> it = std::back_inserter(vec);
            *it = 42;                                     // actually calls: vec.push_back(42);
            ```
            - 常常使用`std::back_inserter()`创建迭代器，作为算法的 *目的位置* 使用
            ```
            std::vector<int> vec;                         // empty vector
            std::fill_n(vec.end(), 10, 0);                // warning: fill_n on empty container is undefined
            std::fill_n(std::back_inserter(vec), 10, 0);  // correct: insert 10 elements to vec
            ```
        - [`std::front_insert_iterator`](https://en.cppreference.com/w/cpp/iterator/front_insert_iterator)
            - 生成：[`std::front_inserter()`](https://en.cppreference.com/w/cpp/iterator/front_inserter)
            ```
            template <class Container>
            std::front_insert_iterator<Container> front_inserter(Container & c)
            {
                return std::front_insert_iterator<Container>(c);
            }
            ```
            - 插入位置 *固定* 是容器第一个位置。即：`std::front_insert_iterator`指向元素会 *随着赋值操作移动*
        - [`std::insert_iterator`](https://en.cppreference.com/w/cpp/iterator/insert_iterator)
            - 生成：[`std::inserter()`](https://en.cppreference.com/w/cpp/iterator/inserter)
            ```
            template <class Container>
            std::insert_iterator<Container> inserter(Container & c, typename Container::iterator i)
            {
                return std::insert_iterator<Container>(c, i);
            }
            ```
            - 插入位置为指向位置 *之前* ，`*insert_iter = t`相当于`c.insert(t, iter)`
            - 使用：经常配合`std::set`使用
            ```
            std::multiset<int> s {1, 2, 3};
            std::fill_n(std::inserter(s, s.end()), 5, 2);                           // 1 2 2 2 2 2 2 3 

            std::vector<int> d {100, 200, 300};
            std::vector<int> l {1, 2, 3, 4, 5};
         
            // when inserting in a sequence container, insertion point advances
            // because each std::insert_iterator::operator= updates the target iterator
            std::copy(d.begin(), d.end(), std::inserter(l, std::next(l.begin())));  // 1 100 200 300 2 3 4 5
            ```
- *流迭代器* （stream iterator）
    - 没意思，不看了
- [*反向迭代器*](https://en.cppreference.com/w/cpp/iterator/reverse_iterator)（reverse iterator）
    - 从容器尾元素向首元素移动的迭代器，递增递减语义颠倒
        - `++rit`会移动至前一个元素，`--rit`会移动至后一个
        - *容器的反向迭代器类型* 实际上是这个模板类型的特化，例如`std::vector<int>`
        ```
        typedef std::reverse_iterator<std::vector<int>> std::vector<int>::reverse_iterator;
        ```
        - 只能在支持 *双向迭代的容器* 或 *双向迭代器* 定义反向迭代器
            - 除`std::forward_list`以外的顺序容器都支持反向迭代器
        - 反向迭代器的目的是表示 *元素范围* 的左开右闭区间，而这是 *不对称* 的
            - 普通迭代器指向的元素和用它转换成的反向迭代器指向的**不是**相同元素，而是相邻元素；反之亦然
    - 生成
        - 用容器自带的`c.rbegin()`或者`std::rbegin(c)`等等
            - 比如容器自带的`c.rbegin()`的内部实现如下
        ```
        template <class Container>
        std::reverse_iterator<typename Container::iterator>
        Container::rbegin()
        { 
            return std::reverse_iterator<Container::iterator>(end()); 
        }
        ```
        - 正反转化
            - 正转反：[`std::make_reverse_iterator()`](https://en.cppreference.com/w/cpp/iterator/make_reverse_iterator) `(since C++17)`
                - 如果要人工转换反向迭代器，一定记得**反向的`begin`要喂正向的`end`**！！！
                - 注意模板参数是容器的迭代器类型，**不是容器**！！！
            ```
            template <class Iter>
            std::reverse_iterator<Iter> make_reverse_iterator(Iter i)
            {
                return std::reverse_iterator<Iter>(i);
            }
            ```
            - 反转正：`r.base()`
            ```
            std::string s1("FIRST,MIDDLE,LAST");
            std::string::reverse_iterator rc = std::find(s1.rbegin(), s1.rend(), ',');
            std::string s2(rc.base(), s1.end());
            std::cout << s2 << std::endl;  // LAST
            ```
    - 反向迭代器主要还是用于让泛型算法无缝衔接反向处理容器
        - 比如不传谓词的情况下用`std::sort()`进行非增序排序
    ```
    std::vector<int> v {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    // std::make_reverse_iterator()用法示例
    // 人工转换反向迭代器，记得反向的begin要喂正向的end
    // 顺便骚一下流迭代器，你说非得写成这德行是何苦呢
    std::reverse_iterator<std::vector<int>::iterator> rbeg = std::make_reverse_iterator(v.end());
    std::reverse_iterator<std::vector<int>::iterator> rend = std::make_reverse_iterator(v.begin());
    std::copy(rbeg, rend, std::ostream_iterator<int>(std::cout, ", "));          // 9, 8, 7, 6, 5, 4, 3, 2, 1
    
    // 当然还能直接调用容器方法获得反向迭代器，最直观
    // 反向迭代器主要还是用于让泛型算法无缝衔接反向处理容器
    // 比如不传谓词的情况下进行逆向排序
    std::vector<int>::reverse_iterator rbeg2 = v.rbegin();
    std::vector<int>::reverse_iterator rend2 = v.rend();
    std::sort(rbeg, rend);
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, ", "));  // 9, 8, 7, 6, 5, 4, 3, 2, 1    
    ```
- [*移动迭代器*](https://en.cppreference.com/w/cpp/iterator/move_iterator)（move iterator）
    - 生成：[`std::make_move_iterator()`](https://en.cppreference.com/w/cpp/iterator/make_move_iterator)
    ```
    template <class Iter>
    constexpr std::move_iterator<Iter> make_move_iterator( Iter i )
    {
        return std::move_iterator<Iter>(std::move(i));
    }
    ```
    - 使用
    ```
    std::list<std::string> s {"one", "two", "three"};
 
    std::vector<std::string> v1(s.begin(), s.end()); // copy
 
    std::vector<std::string> v2(std::make_move_iterator(s.begin()), std::make_move_iterator(s.end())); // move
 
    std::cout << "v1 now holds: ";
    for (const std::string & str : v1)
        std::cout << "\"" << str << "\" ";       // v1 now holds: "one" "two" "three"
    std::cout << "\nv2 now holds: ";
    for (const std::string & str : v2)
        std::cout << "\"" << str << "\" ";       // v2 now holds: "one" "two" "three"
    std::cout << "\noriginal list now holds: ";
    for (const std::string & str : s)
        std::cout << "\"" << str << "\" ";       // original list now holds: "" "" ""
    std::cout << '\n';
    ```






### 🌱 [Chap 10] [泛型算法](https://en.cppreference.com/w/cpp/algorithm)（Generic Algorithms）

#### 概述

- 大部分泛型算法定义于头文件`<algorithm>`中
- 标准库还在头文件`<numeric>`中定义了一组数值泛型算法
- 大部分标准库算法的形参满足以下格式
    - `alg(beg, end, [predicate])`
    - `alg(beg, end, dest, [predicate])`
    - `alg(beg, end, beg2, [predicate])`
    - `alg(beg, end, beg2, end2, [predicate])`  
- 且约定
    - 那些只接受一个单一迭代器来表示第二个序列的算法，都假定 *第二个序列至少与第一个序列一样长*
    - 向目的位置迭代器写数据的算法都假定 *目的位置足够大* ，能容纳要写入的元素
    - 标准库中使用的顺序一般默认是 *非降序* ，二元比较谓词一般等价于`<`
    
#### 使用原则和注意事项

- 泛型算法永远**不会**直接操作容器，但仍旧依赖于元素类型的操作
    - 泛型算法只会运行于 *迭代器* 之上，不会执行特定容器的操作，甚至不需在意自己遍历的是不是容器
        - 因此，**泛型算法不能（直接）添加或删除元素**
    - 调用泛型算法时，在不需要使用返回的迭代器修改容器的情况下，传参应为`const_iterator`
    
#### 谓词（Predicate）

- 大多数算法提供接口，允许我们用 *谓词* 代替默认的运算符
    - 比如排序算法中如何定义 *非增序* ，或是查找算法中如何定义 *相等* ，等等
- 谓词是可调用的表达式。具体传参可以用
    - *函数头*
    - [*函数指针*](https://en.cppreference.com/w/cpp/language/pointer#Pointers_to_functions)
    - [*函数对象*](https://en.cppreference.com/w/cpp/utility/functional) => 14.8
    - [*`lambda`表达式*](https://en.cppreference.com/w/cpp/language/lambda) => 10.3.2
- 标准库算法使用以下两类谓词
    - *一元谓词* （unary predicate）
        - 接受单一参数
        - 一般为迭代器指向元素类型的常引用
            - 不是强制要求，但泛型算法都要求谓词**不能**改变传入元素的值
    - *二元谓词* （binary predicate）
        - 接受两个参数
        - 一般均为迭代器指向元素类型的常引用
            - 不是强制要求，但泛型算法都要求谓词**不能**改变传入元素的值
- 典型二元谓词举例：[`Compare`](https://en.cppreference.com/w/cpp/named_req/Compare)
    - `bool comp(const T & a, const T & b);`
        - 参数类型：常引用不是强制的，但**不能更改传入的对象**
        - 返回值：`bool`亦不是强制的，但要求可以 *隐式转化* 为`bool`
        - 要求：满足 *严格偏序* （Strict partial order）关系
            1. *反自反性* （irreflexivity）：`comp(a, a) == false`
            2. *非对称性* （asymmetry）：`comp(a, b) == true -> comp(b, a) == false`
            3. *传递性* （transitivity）：`comp(a, b) == true AND comp(b, c) == true -> comp(a, c) == true`
    - `bool equiv(const T & a, const T & b);`
        - 参数类型：常引用不是强制的，但**不能更改传入的对象**
        - 返回值：`bool`亦不是强制的，但要求可以 *隐式转化* 为`bool`
        - 要求：满足 *严格偏序* （Strict partial order）关系
            1. *自反性* （reflexivity）：`equiv(a, a) == true`
            2. *对称性* （symmetry）：`equiv(a, b) == true -> equiv(b, a) == true`
            3. *传递性* （transitivity）：`equiv(a, b) == true AND equiv(b, c) == true -> equiv(a, c) == true` 
- 标准库提供以下预定义好的 [*函数对象*](https://en.cppreference.com/w/cpp/utility/functional)（模板类，用时给一个`Type`并创建对象即可）
    - 算术操作（Arithmetic operations）
        - [`plus`](https://en.cppreference.com/w/cpp/utility/functional/plus)：`x + y`
        - [`minus`](https://en.cppreference.com/w/cpp/utility/functional/minus)：`x - y`
        - [`multiplies`](https://en.cppreference.com/w/cpp/utility/functional/multiplies)：`x * y`
        - [`divides`](https://en.cppreference.com/w/cpp/utility/functional/divides)：`x / y`
        - [`modulus`](https://en.cppreference.com/w/cpp/utility/functional/modulus)：`x % y`
        - [`negate`](https://en.cppreference.com/w/cpp/utility/functional/negate)：`-x`
    - 比较（Comparisons）
        - [`equal_to`](https://en.cppreference.com/w/cpp/utility/functional/equal_to)：`x == y`
        - [`not_equal_to`](https://en.cppreference.com/w/cpp/utility/functional/not_equal_to)：`x != y`
        - [`greater`](https://en.cppreference.com/w/cpp/utility/functional/greater)：`x > y`
        - [`less`](https://en.cppreference.com/w/cpp/utility/functional/less)：`x < y`
        - [`greater_equal`](https://en.cppreference.com/w/cpp/utility/functional/greater_equal)：`x >= y`
        - [`less_equal`](https://en.cppreference.com/w/cpp/utility/functional/less_equal)：`x <= y`
        ```
        std::vector<int> v {0, 1, 1, 2};
        std::sort(v.begin(), v.end(), std::greater<>());
        std::for_each(v.begin(), v.end(), [] (const int & i) { printf("%d ", i); });  // 2 1 1 0
        ```
    - 逻辑操作（Logical operations）
        - [`logical_and`](https://en.cppreference.com/w/cpp/utility/functional/logical_and)：`x && y`
        - [`logical_or`](https://en.cppreference.com/w/cpp/utility/functional/logical_or)：`x || y`
        - [`logical_not`](https://en.cppreference.com/w/cpp/utility/functional/logical_not)：`!x`
    - 位操作（Bitwise operations）
        - [`bit_and`](https://en.cppreference.com/w/cpp/utility/functional/bit_and)：`x & y`
        - [`bit_or`](https://en.cppreference.com/w/cpp/utility/functional/bit_or)：`x | y`
        - [`bit_xor`](https://en.cppreference.com/w/cpp/utility/functional/bit_xor)：`x ^ y`
        - [`bit_not`](https://en.cppreference.com/w/cpp/utility/functional/bit_not)：`~x`

### 🌱 [Appendix A] [标准库](https://en.cppreference.com/w/cpp/algorithm)

#### 只读算法（Non-modifying sequence operations）
    
- [`std::all_of`, `std::any_of`, `std::none_of`](https://en.cppreference.com/w/cpp/algorithm/all_any_none_of)
    - 签名
    ```
    template <class InputIt, class UnaryPredicate>
    bool 
    all_of(InputIt        first, 
           InputIt        last, 
           UnaryPredicate p);
                
    template <class InputIt, class UnaryPredicate>
    bool 
    any_of(InputIt        first, 
           InputIt        last, 
           UnaryPredicate p);     

    template <class InputIt, class UnaryPredicate>
    bool 
    none_of(InputIt        first, 
            InputIt        last, 
            UnaryPredicate p);
    ```
    - 检查`p(*it) == true`是否在` [first, last)`上
        - `all_of`：一直成立。如果区间为 *空* ，返回`true`
        - `any_of`：至少有一个实例成立。如果区间为 *空* ，返回`false`
        - `none_of`：一直**不**成立。如果区间为 *空* ，返回`true`
    - 复杂度：`O(last - first)`次谓词调用
- [`std::for_each`](https://en.cppreference.com/w/cpp/algorithm/for_each)
    - 可能的实现
    ```
    template <class InputIt, class UnaryFunction>
    UnaryFunction 
    for_each(InputIt       first, 
             InputIt       last, 
             UnaryFunction f)
    {
        for (; first != last; ++first) 
        {
            f(*first);
        }
        return f;  // implicit move since C++11
    }
    ```
    - 依次对区间`[first, last)`内每个元素调用`f(*iter)`
        - 如果`InputIt`不是常迭代器，则`f`可以修改元素。
        - `f`如有返回值，则直接被丢弃
        - **不能**复制序列中的元素
    - `f`
        - Function object, to be applied to the result of dereferencing every iterator in the range `[first, last)`
        - Signature of the function should be equivalent to the following: `void fun(const Type & a);`
            - The signature does not need to have `const &`
            - `Type` must be such that an object of type `InputIt` can be dereferenced and then implicitly converted to `Type`
    - 返回：传入的`f`经过迭代之后的 *右值引用* 
        - 想要获得经历过迭代的`f`，则 *只能依靠返回值* ，传入的`f`在`for_each`结束后 *未定义* 
    - 复杂度：`Omega(last - first)`次`f`调用
    ```
    struct Sum
    {
        void operator()(int n) { sum += n; }
        int sum{0};
    };
     
    std::vector<int> nums{3, 4, 2, 8, 15, 267};
    std::for_each(nums.begin(), nums.end(), [](int &n){ n++; });  // nums chamges to: 4 5 3 9 16 268
    Sum tmp;
    Sum sum = std::for_each(nums.begin(), nums.end(), tmp);       // tmp.sum UNDEFINED!!!
                                                                  // sum.sum == 305
    ```
- [`std::for_each_n`](https://en.cppreference.com/w/cpp/algorithm/for_each_n) `(since C++17)`
    - 可能的实现
    ```
    template <class InputIt, class Size, class UnaryFunction>
    InputIt 
    for_each_n(InputIt       first, 
               Size          n, 
               UnaryFunction f);
    {
        for (Size i = 0; i < n; ++first, (void) ++i) 
        {
            f(*first);
        }
        return first;
    }
    ```    
    - 依次对区间`[first, first + n)`内每个元素调用`f(*iter)`
    - 返回：`first + n`
    - 复杂度：`Omega(n)`次`f`调用
- [`std::count`, `std::count_if`](https://en.cppreference.com/w/cpp/algorithm/count)
    - 签名
    ```
    template <class InputIt, class T>
    typename iterator_traits<InputIt>::difference_type
    count(InputIt   first, 
          InputIt   last, 
          const T & value);
          
    template <class InputIt, class UnaryPredicate>
    typename iterator_traits<InputIt>::difference_type
    count_if(InputIt        first, 
             InputIt        last, 
             UnaryPredicate p);
    ```
    - 返回：`ptrdiff_t` aka `long int`，区间`[first, last)`之内等于`value`或者满足`p(*iter) == true`的值的个数
    - 复杂度：`Omega(last - first)`次谓词调用
- [`std::mismatch`](https://en.cppreference.com/w/cpp/algorithm/mismatch)
    - 原型
    ```
    template <class InputIt1, class InputIt2>
    std::pair<InputIt1, InputIt2>
    mismatch(InputIt1 first1, 
             InputIt1 last1,
             InputIt2 first2);
              
    template <class InputIt1, class InputIt2, class BinaryPredicate>
    std::pair<InputIt1, InputIt2>
    mismatch(InputIt1        first1, 
             InputIt1        last1,
             InputIt2        first2,
             BinaryPredicate p);
              
    template <class InputIt1, class InputIt2>
    std::pair<InputIt1, InputIt2>
    mismatch(InputIt1 first1, 
             InputIt1 last1,
             InputIt2 first2, 
             InputIt2 last2);
              
    template <class InputIt1, class InputIt2, class BinaryPredicate>
    std::pair<InputIt1, InputIt2>
    mismatch(InputIt1        first1, 
             InputIt1        last1,
             InputIt2        first2, 
             InputIt2        last2,
             BinaryPredicate p);
    ```
    - 返回：指向 *序列1* 与 *序列2* 中第一对在相对应位置 *不匹配* 元素的迭代器
        - *不匹配* 定义为：`*iter1 != *iter2`或`p(*iter1, *iter2) == false`
        - 如果没有不匹配发生，则返回`last1`和其在 *序列2* 中对应的迭代器
        - 如果 *序列1* 比 *序列2* 长，行为 *未定义*
    - 复杂度：`O(min(last1 - first1, last2 - first2))`次比较或谓词调用
- [`std::find`, `std::find_if`, `std::find_if_not`](https://en.cppreference.com/w/cpp/algorithm/find)
    - 签名
    ```
    template <class InputIt, class T>
    InputIt 
    find(InputIt   first, 
         InputIt   last, 
         const T & value);
         
    template <class InputIt, class UnaryPredicate>
    InputIt 
    find_if(InputIt        first, 
            InputIt        last,
            UnaryPredicate p);
            
    template <class InputIt, class UnaryPredicate>
    InputIt 
    find_if_not(InputIt        first, 
                InputIt        last,
                UnaryPredicate q);
    ```
    - 返回
        - `find`：第一个在区间`[first, last)`之内的值为`value`的迭代器，如不存在则返回`last`
        - `find`：值满足`p(*iter) == true`
        - `find_if_not`：值满足`q(*iter) == false`
    ```
    std::vector<int> vec{0, 1, 2, 3, 4, 5, 6...};
    int val = 3;
    std::vector<int>::const_iterator res = std::find(vec.cbegin(), vec.cend(), val);
    std::cout << "The value " << val << (res == vec.cend()) ? " is NOT present" ： “ is present” << std::endl;
    ```
    - 复杂度：`O(last - first)`次谓词调用
    - 指针就是一种迭代器，因此`std::find()`可用于内置数组
    ```
    int arr[]{0, 1, 2, 3, 4, 5, 6...};
    int val = 3;
    int * res_1 = std::find(std::begin(arr), std::end(arr), val);
    int * res_2 = std::find(arr + 1, arr + 4, val);
    ```
- [`std::find_end`](https://en.cppreference.com/w/cpp/algorithm/find_end)
    - 签名
    ```
    template <class ForwardIt1, class ForwardIt2>
    ForwardIt1 
    find_end(ForwardIt1 first, 
             ForwardIt1 last,
             ForwardIt2 s_first, 
             ForwardIt2 s_last);
                     
    template <class ForwardIt1, class ForwardIt2, class BinaryPredicate>
    ForwardIt1 
    find_end(ForwardIt1      first, 
             ForwardIt1      last,
             ForwardIt2      s_first,
             ForwardIt2      s_last,
             BinaryPredicate p);
    ```
    - 返回：`[first, last)`内、指向被搜索序列`[s_first, s_last)` *最后一次* 出现位置的迭代器
        - 相等元素定义为`v1 == v2`或`p(v1, v2) == true`
        - 如果 *没有出现* 或者`[s_first, s_last)`为 *空* ，返回`last`
    - 复杂度：`O(S * (N - S + 1))`次比较或谓词调用，`S = std::distance(s_first, s_last)`，`N = std::distance(first, last)`
- [`std::find_first_of`](https://en.cppreference.com/w/cpp/algorithm/find_first_of)
    - 签名
    ```
    template <class InputIt, class ForwardIt>
    InputIt 
    find_first_of(InputIt   first, 
                  InputIt   last,
                  ForwardIt s_first, 
                  ForwardIt s_last);
                  
    template <class InputIt, class ForwardIt, class BinaryPredicate>
    InputIt 
    find_first_of(InputIt         first, 
                  InputIt         last,
                  ForwardIt       s_first, 
                  ForwardIt       s_last, 
                  BinaryPredicate p);
    ```
    - 返回：`[first, last)`内、指向被搜索序列`[s_first, s_last)`中的 *任一元素* *第一次* 出现位置的迭代器
        - 相等元素定义为`v1 == v2`或`p(v1, v2) == true`
        - 如果 *没有出现* ，返回`last`
    - 复杂度：`O(S * N)`次比较或谓词调用，`S = std::distance(s_first, s_last)`，`N = std::distance(first, last)`
- [`std::adjacent_find`](https://en.cppreference.com/w/cpp/algorithm/adjacent_find)
    - 签名
    ```
    template <class ForwardIt>
    ForwardIt 
    adjacent_find(ForwardIt first, 
                  ForwardIt last);

    template <class ForwardIt, class BinaryPredicate>
    ForwardIt 
    adjacent_find(ForwardIt       first, 
                  ForwardIt       last, 
                  BinaryPredicate p);
    ```
    - 返回：在`[first, last)`内 *第一个* 满足`*it == *(it + 1)`或`p(*it, *(it + 1)) == true`的迭代器`it`
    - 复杂度：`Omega(min((result - first) + 1, (last - first) - 1)`次谓词调用，`result`为返回值
- [`std::search`](https://en.cppreference.com/w/cpp/algorithm/search)
    - 签名
    ```
    template <class ForwardIt1, class ForwardIt2>
    ForwardIt1 
    search(ForwardIt1 first, 
           ForwardIt1 last,
           ForwardIt2 s_first, 
           ForwardIt2 s_last);
                   
    template <class ForwardIt1, class ForwardIt2, class BinaryPredicate>
    ForwardIt1 
    search(ForwardIt1      first,
           ForwardIt1      last,
           ForwardIt2      s_first, 
           ForwardIt2      s_last, 
           BinaryPredicate p);
    ```
    - 返回：`[first, last)`内、指向被搜索序列`[s_first, s_last)` *第一次* 出现位置的迭代器
        - 相等元素定义为`v1 == v2`或`p(v1, v2) == true`
        - 如果 *没有出现* ，返回`last`
        - 如果`[s_first, s_last)`为 *空* ，返回`first`
    - 复杂度：`O(S * N)`次比较或谓词调用，`S = std::distance(s_first, s_last)`，`N = std::distance(first, last)`
- [`std::search_n`](https://en.cppreference.com/w/cpp/algorithm/search_n)
    - 签名
    ```
    template <class ForwardIt, class Size, class T>
    ForwardIt 
    search_n(ForwardIt first, 
             ForwardIt last, 
             Size      count, 
             const T & value);


    template <class ForwardIt, class Size, class T, class BinaryPredicate >
    ForwardIt 
    search_n(ForwardIt       first, 
             ForwardIt       last, 
             Size            count, 
             const T &       value,
             BinaryPredicate p);
    ```
    - 返回：`[first, last)`内、指向连续`count`个`value`组成的序列 *第一次* 出现位置的迭代器
        - 相等元素定义为`v1 == v2`或`p(v1, v2) == true`
        - 如果 *没有出现* ，返回`last`
        - 如果`count <= 0`，返回`first`
    - 复杂度：`O(last - first)`次比较或谓词调用

#### 写算法（Modifying sequence operations）

- [`std::copy`, `std::copy_if`](https://en.cppreference.com/w/cpp/algorithm/copy)
    - 可能的实现
    ```
    template <class InputIt, class OutputIt>
    OutputIt 
    copy(InputIt  first, 
         InputIt  last, 
         OutputIt d_first)
    {    
        while (first != last) 
        {
            *d_first++ = *first++;
        }
        
        return d_first;
    }
         
    template <class InputIt, class OutputIt, class UnaryPredicate>
    OutputIt 
    copy_if(InputIt first, 
            InputIt last,
            OutputIt d_first,
            UnaryPredicate pred)
    {    
        while (first != last) 
        {
            if (pred(*first))
            {
                *d_first++ = *first;
            }
                
            first++;
        }
        
        return d_first;
    }
    ```
    - 将区间`[first, last)`之内所有元素拷贝至以`d_first`开始的一片内存中，
        - `copy_if`：只拷贝满足`pred(*iter) == true`的元素
        - 需保证写`d_first`开始的这一片内存是合法行为
    - 返回：拷贝生成的序列的尾后迭代器
    - 复杂度：`Omega(last - first)`次赋值
    ```
    int a1[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; 
    int a2[sizeof(a1) / sizeof(*a1)]; 
    int * res = std::copy(std::begin(a1), std::end(a1), a2); 
    ```
- [`std::copy_n`](https://en.cppreference.com/w/cpp/algorithm/copy_n)
    - 可能的实现
    ```
    template <class InputIt, class Size, class OutputIt>
    OutputIt 
    copy_n(InputIt  first, 
           Size     count, 
           OutputIt result)
    {
        if (count > 0) 
        {
            *result++ = *first;
            
            for (Size i = 1; i < count; ++i) 
            {
                *result++ = *++first;
            }
        }
        
        return result;
    }
    ```
    - 将区间`[first, first + count)`之内的`count`个值复制到区间`[result, result + count)`
        - 需保证写`[result, result + count)`开始的这一片内存是合法行为
    - 返回：拷贝生成的序列的尾后迭代器`result + count`，如`count <= 0`，则返回`result`
    - 复杂度：如果`count > 0`，`Omega(count)`次赋值；否则`O(1)`
- [`std::copy_backward`](https://en.cppreference.com/w/cpp/algorithm/copy_backward)
    - 可能的实现
    ```
    template <class BidirIt1, class BidirIt2>
    BidirIt2 
    copy_backward(BidirIt1 first, 
                  BidirIt1 last, 
                  BidirIt2 d_last)
    {
        while (first != last) 
        {
            *(--d_last) = *(--last);
        }
        
        return d_last;
    }
    ```
    - 将`[first, last)`内的元素拷贝至到`d_last` *为止* 的一片内存内
        - 拷贝顺序为 *逆序* ，后面的元素先拷贝，但元素的相对顺序不变
    - 返回：迭代器`d_last - (last - first)`
    - 复杂度：`Omega(last - first)`次赋值
    ```
    std::vector<int> vec {1, 2, 3, 4, 5};
    int buf[10] {0};
    int * ret = std::copy_backward(vec.cbegin(), vec.cend(), std::end(buf));
    printf("%d\n", ret == std::end(buf) - (vec.cend() - vec.cbegin()));  // 1
    ```
- [`std::move`](https://en.cppreference.com/w/cpp/algorithm/move)
    - 可能的实现
    ```
    template <class InputIt, class OutputIt>
    OutputIt 
    move(InputIt  first, 
         InputIt  last, 
         OutputIt d_first)
    {
        while (first != last) 
        {
            *d_first++ = std::move(*first++);
        }
        
        return d_first;
    }
    ```
    - 将`[first, last)`内的元素 *移动* 至以`d_first`开始的一片内存内
        - 移动完成后`[first, last)`内元素 *仍是合法的该类型元素* ，但 *具体数值不一等和移动前相同*
    - 返回：移动生成的序列的尾后迭代器
    - 复杂度：`Omega(last - first)`次 *移动赋值* 
        - 注意别的泛型算法都是 *拷贝赋值* 
- [`std::move_backward`](https://en.cppreference.com/w/cpp/algorithm/move_backward)
    - 可能的实现
    ```
    template <class BidirIt1, class BidirIt2>
    BidirIt2 
    move_backward(BidirIt1 first,
                  BidirIt1 last,
                  BidirIt2 d_last)
    {
        while (first != last) 
        {
            *(--d_last) = std::move(*(--last));
        }
        
        return d_last;
    }
    ```
    - 将`[first, last)`内的元素 *移动* 至到`d_last` *为止* 的一片内存内
        - 移动顺序为 *逆序* ，后面的元素先移动，但元素的相对顺序不变
    - 返回：迭代器`d_last - (last - first)`
    - 复杂度：`Omega(last - first)`次 *移动赋值* 
        - 注意别的泛型算法都是 *拷贝赋值* 
- [`std::fill`](https://en.cppreference.com/w/cpp/algorithm/fill)
    - 可能的实现
    ```
    template <class ForwardIt, class T>
    void 
    fill(ForwardIt first, 
         ForwardIt last, 
         const T & value)
    {
        for (; first != last; ++first) 
        {
            *first = value;
        }
    }
    ```
    - 将区间`[first, last)`之内所有元素都赋值为`value` 
    - 返回：`void`
    - 复杂度：`Omega(last - first)`次赋值
    ```
    std::fill(vec.begin(), vec.end(), 0));
    std::fill(vec.begin(), vec.begin() + vec.size() / 2, 0));
    ```
- [`std::fill_n`](https://en.cppreference.com/w/cpp/algorithm/fill_n)
    - 可能的实现
    ```
    template <class OutputIt, class Size, class T>
    OutputIt 
    fill_n(OutputIt  first, 
           Size      count, 
           const T & value)
    {
        for (Size i = 0; i < count; i++) 
        {
            *first++ = value;
        }
        
        return first;
    }
    ```
    - 将区间`[first, first + count)`之内所有元素都赋值为`value` 
        - `std::fill_n()`**不**检查写区间`[first, first + count)`是否合法，这是程序员的责任
        - 在 *空容器* 上调用`std::fill_n()`或其它写算法是 *未定义* 行为。对于空容器应当使用`std::back_insert_iterator` => 10.4.1
    - 返回：迭代器`first + count`
    - 复杂度：如果`count > 0`，`Omega(count)`次赋值；否则`O(1)`
- [`std::transform`](https://en.cppreference.com/w/cpp/algorithm/transform)
    - 可能的实现
    ```
    template <class InputIt, class OutputIt, class UnaryOperation>
    OutputIt 
    transform(InputIt        first1, 
              InputIt        last1, 
              OutputIt       d_first,
              UnaryOperation unary_op)
        {
            while (first1 != last1) 
            {
                *d_first++ = unary_op(*first1++);
            }
            retu
            rn d_first;
        }
    
    template <class InputIt1, class InputIt2, class OutputIt, class BinaryOperation>
    OutputIt 
    transform(InputIt1        first1, 
              InputIt1        last1, 
              InputIt2        first2,
              OutputIt        d_first, 
              BinaryOperation binary_op)
        {
            while (first1 != last1) 
            {
                *d_first++ = binary_op(*first1++, *first2++);
            }
            
            return d_first;
        }
    ```
    - 将 *对应函数* 应用于 *一片区间* 内，并将结果存储于`d_first`开始的一片区域中
        1. 将`unary_op`应用于`[first, last)`上的每个元素，取其返回值
        2. 将`binary_op`应用如下定义的一对元素上：一个定义在`[first, last)`上，另一个取自从`first2`开始的对应位置，取其返回值
    - 输出对象可以是 *自己* 
    - 返回：拷贝生成的序列的尾后迭代器
    - 复杂度：`Omega(last1 - first1)`次谓词调用
    ```
    // turn all elements of a int vector into their absolute values
    std::transform(vec.begin(), vec.end(), vec.begin(), [] (const int & i)
    {
        return i < 0 ? -i : i;
    });
    ```
- [`std::generate`](https://en.cppreference.com/w/cpp/algorithm/generate)
    - 可能的实现
    ```
    template <class ForwardIt, class Generator>
    void 
    generate(ForwardIt first, 
             ForwardIt last, 
             Generator g)
    {
        while (first != last) 
        {
            *first++ = g();
        }
    }
    ```
    - 将`[first, last)`内元素 *依次* 赋值为`g()`
    - `g`签名：`ret g();`
    - 复杂度：`Omega(last - first)`次`g`调用
    ```
    int g()
    {
        static int i = 0;
        return i++;
    }
    
    std::vector<int> vec(5);
    std::generate(vec.begin(), vec.end(), g);
    std::for_each(vec.cbegin(), vec.cend(), [] (const int & i) -> void { printf("%d ", i); });  // 0 1 2 3 4
    ```
- [`std::generate_n`](https://en.cppreference.com/w/cpp/algorithm/generate_n)
    - 可能的实现
    ```
    template <class OutputIt, class Size, class Generator>
    OutputIt 
    generate_n(OutputIt  first, 
               Size      count, 
               Generator g)
    {
        for (Size i = 0; i < count; ++i) 
        {
            *first++ = g();
        }
        
        return first;
    }
    ```
    - 将`[first, first + cpunt)`内元素 *依次* 赋值为`g()`
    - `g`签名：`ret g();`
    - 复杂度：`Omega(count)`次`g`调用（`count > 0`）
- [`std::remove`, `std::remove_if`](https://en.cppreference.com/w/cpp/algorithm/remove)
    - 可能的实现
    ```
    template <class ForwardIt, class T>
    ForwardIt 
    remove(ForwardIt first, 
           ForwardIt last, 
           const T & value)
    {
        first = std::find(first, last, value);
        
        if (first != last)
        {
            for (ForwardIt i = first; ++i != last;)
            {
                if (!(*i == value))
                {
                    *first++ = std::move(*i);
                }
            }
        }
        
        return first;
    }

    template <class ForwardIt, class UnaryPredicate>
    ForwardIt 
    remove_if(ForwardIt      first, 
              ForwardIt      last, 
              UnaryPredicate p)
    {
        first = std::find_if(first, last, p);
        
        if (first != last)
        {
            for (ForwardIt i = first; ++i != last;)
            {
                if (!p(*i))
                {
                    *first++ = std::move(*i);
                }
            }
        }
        
        return first;
    }
    ```
    - *移除* `[first, last)`中全部满足`*it == val`或`p(*it) == true`的元素
        - *移除* ：用被清除元素后面的元素覆盖被清除元素，**并不**改变容器大小
        - 这也就是为什么这其实是一个排序算法
    - 返回：移除完成后的逻辑区间的尾后迭代器（past-the-end iterator for the new logical end of the range）
        - 此迭代器后面的元素仍可被解引用访问，但值 *未定义*    
    - 复杂度：`Omega(last - first)`次谓词调用
- [`std::remove_copy`, `std::remove_copy_if`](https://en.cppreference.com/w/cpp/algorithm/remove_copy)
    - 可能的实现
    ```
    template <class InputIt, class OutputIt, class T>
    OutputIt 
    remove_copy(InputIt   first, 
                InputIt   last,
                OutputIt  d_first, 
                const T & value)
    {
        for (; first != last; ++first)
        {
            if (!(*first == value))
            {
                *d_first++ = *first;
            }
        }
        
        return d_first;
    }

    template <class InputIt, class OutputIt, class UnaryPredicate>
    OutputIt
    remove_copy_if(InputIt        first,
                   InputIt        last,
                   OutputIt       d_first,
                   UnaryPredicate p)
    {
        for (; first != last; ++first)
        {
            if (!p(*first))
            {
                *d_first++ = *first;
            }
        }
        
        return d_first;
    }
    ```
    - 将`[first, last)`中全部满足`*it != val`或`p(*it) != true`的元素拷贝至以`d_first`开始的一片内存中
    - 返回：拷贝完成后区间的尾后迭代器
    - 复杂度：`Omega(last - first)`次谓词调用
- [`std::replace`, `std::replace_if`](https://en.cppreference.com/w/cpp/algorithm/replace)
    - 签名
    ```
    template <class ForwardIt, class T>
    void
    replace(ForwardIt first,
            ForwardIt last,
            const T & old_value,
            const T & new_value)
    {
        for (; first != last; ++first)
        {
            if (*first == old_value)
            {
                *first = new_value;
            }
        }
    }

    template <class ForwardIt, class UnaryPredicate, class T>
    void
    replace_if(ForwardIt      first,
               ForwardIt      last,
               UnaryPredicate p,
               const T &      new_value)
    {
        for (; first != last; ++first)
        {
            if (p(*first))
            {
                *first = new_value;
            }
        }
    }
    ```
    - 将区间`[first, last)`之内所有满足条件的元素修改为`new_value`
        - `replace`：值为`old_value`的元素
        - `replace_if`：满足`p(*iter) == true`的元素
    - 返回：`void`
    - 复杂度：`Omega(last - first)`次谓词调用
- [`std::replace_copy`, `std::replace_copy_if`](https://en.cppreference.com/w/cpp/algorithm/replace_copy)
    - 签名
    ```
    template <class InputIt, class OutputIt, class T>
    OutputIt replace_copy(InputIt   first,
                          InputIt   last,
                          OutputIt  d_first,
                          const T & old_value,
                          const T & new_value)
    {
        for (; first != last; ++first)
        {
            *d_first++ = (*first == old_value) ? new_value : *first;
        }
        
        return d_first;
    }

    template <class InputIt, class OutputIt, class UnaryPredicate, class T>
    OutputIt
    replace_copy_if(InputIt        first,
                    InputIt        last,
                    OutputIt       d_first,
                    UnaryPredicate p,
                    const T &      new_value)
    {
        for (; first != last; ++first)
        {
            *d_first++ = p(*first) ? new_value : *first;
        }
        
        return d_first;
    }    
    ```
    - 将对应 *替换规则* 应用于区间`[first, last)`内，并将结果存储于`d_first`开始的一片区域中
        - `replace_copy`：其中所有值为`old_value`元素都被修改为`new_value`
        - `replace_copy_if`：只替换满足`p(*iter) == true`的元素
    - 返回：拷贝生成的序列的尾后迭代器
    - 复杂度：`Omega(last - first)`次谓词调用
    ```
    // 此调用后，ilst不变，ivec包含ilst的一份拷贝，且原来的0全部被替换为42
    std::replace_copy(ilst.begin(), ilst.end(), std::back_inserter(ivec), 0, 42);
    ```
- [`std::swap`](https://en.cppreference.com/w/cpp/algorithm/swap)
    - 签名
    ```
    template <class T>
    void 
    swap(T & a, 
         T & b);

    template <class T2, std::size_t N>
    void 
    swap(T2 (&a)[N], 
         T2 (&b)[N]);
    ```
    - *互换* 类或数组的内容。数组版实际调用`std::swap_ranges(a, a + N, b);`
    - 返回：`void`
    - 复杂度：类版`O(1)`，数组版`O(N)`
- [`std::swap_ranges`](https://en.cppreference.com/w/cpp/algorithm/swap_ranges)
    - 可能的实现
    ```
    template <class ForwardIt1, class ForwardIt2>
    ForwardIt2 
    swap_ranges(ForwardIt1 first1, 
                ForwardIt1 last1, 
                ForwardIt2 first2)
    {
        while (first1 != last1) 
        {
            std::iter_swap(first1++, first2++);
        }
        
        return first2;
    }
    ```
    - *互换* `[first, last)`和`[first2, first2 + last1 - first1)`的内容
    - 返回：尾后迭代器`first2 + last1 - first1`
    - 复杂度：`O(last - first)`
- [`std::iter_swap`](https://en.cppreference.com/w/cpp/algorithm/iter_swap)
    - 可能的实现
    ```
    template <class ForwardIt1, class ForwardIt2>
    void 
    iter_swap(ForwardIt1 a, 
              ForwardIt2 b)
    {
        std::swap(*a, *b);
    }
    ```
    - 互换两个迭代器指向元素的值
    - 返回：`void`
    - 复杂度：`O(1)`
- [`std::reverse`](https://en.cppreference.com/w/cpp/algorithm/reverse)
    - 可能的实现
    ```
    template <class BidirIt>
    void 
    reverse(BidirIt first, 
            BidirIt last)
    {
        while ((first != last) && (first != --last)) 
        {
            std::iter_swap(first++, last);
        }
    }
    ```
    - 反转区间`[first, last)`。首尾齐发往中间对换
    - 复杂度：`Omega((last - first) / 2)`次对换
- [`std::reverse_copy`](https://en.cppreference.com/w/cpp/algorithm/reverse_copy)
    - 可能的实现
    ```
    template <class BidirIt, class OutputIt>
    OutputIt 
    reverse_copy(BidirIt  first, 
                 BidirIt  last, 
                 OutputIt d_first)
    {
        while (first != last) 
        {
            *(d_first++) = *(--last);
        }
        
        return d_first;
    }
    ```
    - 反转区间`[first, last)`。 *就地* 首尾齐发往中间对换，结果存储于以`d_first`开始的一片内存之中
    - 复杂度：`O(last - first)`
- [`std::rotate`](https://en.cppreference.com/w/cpp/algorithm/rotate)
    - 可能的实现
    ```
    template <class ForwardIt>
    ForwardIt 
    rotate(ForwardIt first, 
           ForwardIt n_first, 
           ForwardIt last)
    {
        if (first == n_first)  return last;
        if (n_first == last)  return first;

        ForwardIt read      = n_first;
        ForwardIt write     = first;
        ForwardIt next_read = first;                   // read position for when "read" hits "last"

        while (read != last) 
        {
            if (write == next_read) next_read = read;  // track where "first" went
            std::iter_swap(write++, read++);
        }

        // rotate the remaining sequence into place
        (rotate)(write, next_read, last);
        return write;
    }
    ```
    - 将`[first, n_first - 1] [n_first, last - 1]` *就地* 旋转为`[n_first, last - 1] [first, n_first - 1]`
    - 返回：指向原先`first`指向元素在旋转之后所在位置的迭代器，`first + (last - n_first)`
    - 复杂度：`O(last - first)`
- [`std::rotate_copy`](https://en.cppreference.com/w/cpp/algorithm/rotate_copy)
    - 可能的实现
    ```
    template <class ForwardIt, class OutputIt>
    OutputIt 
    rotate_copy(ForwardIt first, 
                ForwardIt n_first,
                ForwardIt last, 
                OutputIt  d_first)
    {
        d_first = std::copy(n_first, last, d_first);
        return std::copy(first, n_first, d_first);
    }
    ```
    - 将`[first, n_first - 1] [n_first, last - 1]`旋转为`[n_first, last - 1] [first, n_first - 1]`，结果存储于`d_first`开始的一片内存中
    - 返回：指向原先`first`指向元素在旋转、拷贝之后所在位置的迭代器，`d_first + (last - n_first)`
    - 复杂度：`O(last - first)`
- [`std::shuffle`](https://en.cppreference.com/w/cpp/algorithm/random_shuffle)
    - `std::random_shuffle`已于`C++14`被废弃，`C++17`被移除。略去
    - 可能的实现
    ```
    template <class RandomIt, class URBG>
    void 
    shuffle(RandomIt first, 
            RandomIt last, 
            URBG &&  g)
    {
        typedef typename std::iterator_traits<RandomIt>::difference_type diff_t;
        typedef std::uniform_int_distribution<diff_t> distr_t;
        typedef typename distr_t::param_type param_t;
     
        distr_t D;
        diff_t n = last - first;
        
        for (diff_t i = n - 1; i > 0; --i) 
        {
            std::swap(first[i], first[D(g, param_t(0, i))]);
        }
    }
    ```
    - 随机打乱，每种排列都相同概率出现
    - `g`：`UniformRandomBitGenerator` whose result type is convertible to `std::iterator_traits<RandomIt>::difference_type`
    - 复杂度：`O(last - first)`
    ```
    std::vector<int> v {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    std::for_each(v.cbegin(), v.cend(), [] (const int & i) { printf("%d ", i); });  // 这可真是只有鬼知道是啥了
    ```
- [`std::sample`](https://en.cppreference.com/w/cpp/algorithm/sample) `(since C++17)`
    - 签名
    ```
    template <class PopulationIterator, class SampleIterator, class Distance, class URBG>
    SampleIterator 
    sample(PopulationIterator first, 
           PopulationIterator last,
           SampleIterator     out, 
           Distance           n,
           URBG &&            g);
    ```
    - 随机采样`std::min(n, last - first)`个，每个元素相同概率被采样
    - `out`：采样结果写入输出迭代器`out`之中。`out`所指范围**不能**在`[first, last)`之内
    - `g`：`UniformRandomBitGenerator` whose result type is convertible to `std::iterator_traits<RandomIt>::difference_type`
    - 返回值：a copy of out after the last sample that was output, that is, end of the sample range
    - 复杂度：`O(last - first)`
    ```
    std::string in = "hgfedcba", out;
    std::sample(in.begin(), in.end(), std::back_inserter(out), 5, std::mt19937{std::random_device{}()});
    std::cout << "five random letters out of " << in << " : " << out << std::endl;  // five random letters out of hgfedcba : gfcba
    ```
- [`std::unique`](https://en.cppreference.com/w/cpp/algorithm/unique)
    - 可能的实现
    ```
    template <class ForwardIt>
    ForwardIt 
    unique(ForwardIt first, 
           ForwardIt last)
    {
        if (first == last)
        {
            return last;
        }
        
        ForwardIt result = first;
        
        while (++first != last) 
        {
            if (!(*result == *first) && ++result != first) 
            {
                *result = std::move(*first);
            }
        }
        
        return ++result;
    }

    template <class ForwardIt, class BinaryPredicate>
    ForwardIt 
    unique(ForwardIt       first, 
           ForwardIt       last, 
           BinaryPredicate p)
    {
        if (first == last)
        {
            return last;
        }
            
        ForwardIt result = first;
        
        while (++first != last) 
        {
            if (!p(*result, *first) && ++result != first) 
            {
                *result = std::move(*first);
            }
        }
        
        return ++result;
    }
    ```
    - 对区间`[first, last)`中每一组 *连续的* *相等* 元素，只保留第一个， *移除* 其余元素
        - *移除* ：用被清除元素后面的元素覆盖被清除元素，**并不**改变容器大小
        - *相等* ：`*iter1 == *iter2`或`p(*iter1, *iter2) == true`
        - 这也就是为什么这其实是一个排序算法
    - 返回：移除完成后的逻辑区间的尾后迭代器（past-the-end iterator for the new logical end of the range）
        - 此迭代器后面的元素仍可被解引用访问，但值 *未定义*
    - 使用前应该**先调用**`std::sort()`，之后**再调用**容器的`erase()`方法
        - *标准库算法* 操作的 *均是* 迭代器而不是容器，因此，**标准库算法不能（直接）添加或删除元素**
    ```
    std::vector<int> v {1, 2, 1, 1, 3, 3, 3, 4, 5, 4};
 
    auto last = std::unique(v.begin(), v.end());  // remove consecutive (adjacent) duplicates
    v.erase(last, v.end());  // v now holds {1 2 1 3 4 5 4 x x x}, where 'x' is indeterminate
 
    // sort followed by unique, to remove all duplicates
    std::sort(v.begin(), v.end());  // {1 1 2 3 4 4 5}
    last = std::unique(v.begin(), v.end());  // v now holds {1 2 3 4 5 x x}, where 'x' is indeterminate
    v.erase(last, v.end());  // v now holds {1, 2, 3, 4, 5}
    ```  
- [`std::unique_copy`](https://en.cppreference.com/w/cpp/algorithm/unique_copy)    
    - 签名
    ```
    template <class InputIt, class OutputIt>
    OutputIt 
    unique_copy(InputIt  first, 
                InputIt  last,
                OutputIt d_first);
                      
    template <class InputIt, class OutputIt, class BinaryPredicate>
    OutputIt 
    unique_copy(InputIt         first, 
                InputIt         last,
                OutputIt        d_first, 
                BinaryPredicate p);
    ```
    - 将区间`[first, last)`内的元素拷贝到以`d_first`开始的一片内存中，且 *不保留连续的重复元素* 。
        - 即：对每组连续的重复元素，只有第一个被拷贝
    - 返回：拷贝生成的序列的尾后迭代器
    - 复杂度：`Omega(last - first + 1)`次谓词调用

#### 划分（Partitioning operations）

- [`std::is_partitioned`](https://en.cppreference.com/w/cpp/algorithm/is_partitioned)
    - 可能的实现
    ```
    template <class InputIt, class UnaryPredicate>
    bool 
    is_partitioned(InputIt        first, 
                   InputIt        last, 
                   UnaryPredicate p)
    {
        for (; first != last; ++first)
            if (!p(*first))
                break;
        for (; first != last; ++first)
            if (p(*first))
                return false;
        return true;
    }
    ```
    - 返回：如果`[first, last)`是 *按`p`划分好的* ，或区间为 *空* ，则返回`true`
        - 区间中所有满足谓词`p`的元素 *都在* 不满足`p`的元素 *之前* 
    - 复杂度：`O(last - first)`次谓词调用
- [`std::partition`](https://en.cppreference.com/w/cpp/algorithm/partition)
    - 可能的实现
    ```
    template <class ForwardIt, class UnaryPredicate>
    ForwardIt 
    partition(ForwardIt      first, 
              ForwardIt      last, 
              UnaryPredicate p)
    {
        first = std::find_if_not(first, last, p);
        if (first == last) return first;
     
        for (ForwardIt i = std::next(first); i != last; ++i) 
        {
            if (p(*i)) 
            {
                std::iter_swap(i, first);
                ++first;
            }
        }
        
        return first;
    }
    ```
    - *按`p`划分* 区间`[first, last)`
        - 区间中所有满足谓词`p`的元素 *都在* 不满足`p`的元素 *之前* 
        - **非稳定**划分，即元素之前的相对位置**不被保护**
    - 返回：指向第二组第一个元素的迭代器
    - 复杂度：`O(last - first)`次谓词调用。如果`ForwardIt`支持递减，则还有`O((last - first) / 2)`次对换；否则，`O(last - first)`次
- [`std::partition_copy`](https://en.cppreference.com/w/cpp/algorithm/partition_copy)
    - 可能的实现
    ```
    template <class InputIt, class OutputIt1, class OutputIt2, class UnaryPredicate>
    std::pair<OutputIt1, OutputIt2>
    partition_copy(InputIt        first, 
                   InputIt        last,
                   OutputIt1      d_first_true, 
                   OutputIt2      d_first_false,
                   UnaryPredicate p)
    {
        while (first != last) 
        {
            if (p(*first)) 
            {
                *d_first_true = *first;
                ++d_first_true;
            } 
            else 
            {
                *d_first_false = *first;
                ++d_first_false;
            }
            
            ++first;
        }
        
        return std::pair<OutputIt1, OutputIt2>(d_first_true, d_first_false);
    }
    ```
    - *按`p`划分* 区间`[first, last)`，两部分分别存入`d_first_true`和`d_first_false`
        - 区间中所有满足谓词`p`的元素 *都在* 不满足`p`的元素 *之前* 
        - 如果输入输出区间 *重叠* ， *行为未定义* 
    - 返回：拷贝生成的`true`、`false`两个序列的尾后迭代器
    - 复杂度：`Omega(last - first)`次谓词调用
- [`std::stable_partition`](https://en.cppreference.com/w/cpp/algorithm/stable_partition)
    - 签名
    ```
    template <class BidirIt, class UnaryPredicate>
    BidirIt 
    stable_partition(BidirIt        first, 
                     BidirIt        last, 
                     UnaryPredicate p);
    ```
    - *按`p`稳定划分* 区间`[first, last)`
        - 区间中所有满足谓词`p`的元素 *都在* 不满足`p`的元素 *之前* 
        - 元素之前的相对位置 *被保护*
    - 返回：指向第二组第一个元素的迭代器
    - 复杂度：`O(last - first)`次谓词调用。如果内存足够，则还有`O(last - first)`次对换；否则，`O(N log N)`次
- [`std::partition_point`](https://en.cppreference.com/w/cpp/algorithm/partition_point)
    - 签名
    ```
    template <class ForwardIt, class UnaryPredicate>
    ForwardIt 
    partition_point(ForwardIt      first, 
                    ForwardIt      last, 
                    UnaryPredicate p);
    ```
    - 返回： *已按`p`划分好的* 区间`[first, last)`的分界点，即
        - 第一个不满足谓词`p`的元素
        - 如果区间为 *空* ，返回`last`

#### 排序（Sorting operations）    

- [`std::is_sorted`](https://en.cppreference.com/w/cpp/algorithm/is_sorted)
    - 可能的实现
    ```
    template <class ForwardIt>
    bool 
    is_sorted(ForwardIt first, 
              ForwardIt last)
    {
        return std::is_sorted_until(first, last) == last;
    }

    template <class ForwardIt, class Compare>
    bool 
    is_sorted(ForwardIt first, 
              ForwardIt last, 
              Compare comp)
    {
        return std::is_sorted_until(first, last, comp) == last;
    }
    ```
    - 返回：序列是否按照 *非降序* 排列好了
        - *逆序* 的定义：`*(it + n) < *it`或`comp(*(it + n), *it) == true`
    - 复杂度：`O(last - first)`
- [`std::is_sorted_until`](https://en.cppreference.com/w/cpp/algorithm/is_sorted_until)
    - 可能的实现
    ```
    template <class ForwardIt>
    ForwardIt 
    is_sorted_until(ForwardIt first, 
                    ForwardIt last)
    {
        return is_sorted_until(first, last, std::less<>());
    }

    template <class ForwardIt, class Compare>
    ForwardIt 
    is_sorted_until(ForwardIt first, 
                    ForwardIt last, 
                    Compare   comp) 
    {
        if (first != last) 
        {
            ForwardIt next = first;
            
            while (++next != last) 
            {
                if (comp(*next, *first))
                {
                    return next;
                }
                    
                first = next;
            }
        }
        
        return last;
    }
    ```
    - 返回：指向第一个与其前驱构成 *逆序* 的元素的迭代器，如序列为 *非降序* 则返回`last`
        - *逆序* 的定义：`*(it + n) < *it`或`comp(*(it + n), *it) == true`
    - 复杂度：`O(last - first)`
- [`std::sort`](https://en.cppreference.com/w/cpp/algorithm/sort)
    - 签名
    ```
    template <class RandomIt>
    void 
    sort(RandomIt first, 
         RandomIt last);
    
    template <class RandomIt, class Compare>
    void 
    sort(RandomIt first, 
         RandomIt last, 
         Compare  comp);
    ```
    - 把区间`[first, last)`内元素按照 *非降序* （non-descending order）排序
        - **不是**稳定排序，即不保证排序前后相等元素的相对顺序保持不变
        - *非降序* ：如果`v1`、`v2`满足`v1 < v2`或`comp(v1, v2) == true`，则`v1`应在`v2` *前面* 
            - 粗略理解为`comp(前, 后) ≈ true`，例外在相等元素`comp(a, a) == false`
            - 两元素相等，则谁前谁后无所谓，都不违反上面的定义
            - 一句话，你想按前面比后面大就传`>`，不然就传`<`
        - `gcc`实现：对任何迭代器`it`，和任何自然数`n`
            - 如果两个元素满足`*(it + n) < *it`或`comp(*(it + n), *it) == true`，则它们会被 *对换* 
            - `gcc`实现如何使用谓词一句话：`后 < 前 == true`或`<(后, 前) == true`就 *对换* 
        - 想要 *非增序* （non-ascending order）排序，可以
            1. 直接喂一个`std::greater`模板对象作为谓词
                - 注意**不能**喂`std::greater_equal`，必须是 *严格偏序* （也就是说相等元素要返回`false`，不然死循环了）
            ```
            std::vector<int> v {0, 1, 1, 2};
            std::sort(v.begin(), v.end(), std::greater<>());
            std::for_each(v.begin(), v.end(), [] (const int & i) { printf("%d ", i); });  // 2 1 1 0
            ```
            2. 喂其他的自定义谓词
            ```
            std::vector<int> v {0, 1, 1, 2};
            std::sort(v.begin(), v.end(), [] (const int & v1, const int & v2) { return v1 > v2; });
            std::for_each(v.begin(), v.end(), [] (const int & i) { printf("%d ", i); });  // 2 1 1 0
            ```
            3. 如果喂两个 [*反向迭代器*](https://en.cppreference.com/w/cpp/iterator/reverse_iterator) ，就连谓词也给省了 => 10.4
            ```
            std::vector<int> v {0, 1, 1, 2};
            std::sort(v.rbegin(), v.rend());
            std::for_each(v.begin(), v.end(), [] (const int & i) { printf("%d ", i); });  // 2 1 1 0
            ```
            4. 颠倒重载`<`运算符：那可真是没事儿闲的了，堪比`#define true false`或者`#define < >`
    - 谓词`comp`需满足[`Compare`](https://en.cppreference.com/w/cpp/named_req/Compare)标准规定的条件 => 10.3
        - 签名：`bool comp(const T & a, const T & b);`
        - 参数类型：常引用不是强制的，但**不能更改传入的对象**
        - 返回值：`bool`亦不是强制的，但要求可以 *隐式转化* 为`bool`
        - 要求：满足 *严格偏序* （Strict partial order）关系
            1. *反自反性* （irreflexivity）：`comp(a, a) == false`
            2. *非对称性* （asymmetry）：`comp(a, b) == true -> comp(b, a) == false`
            3. *传递性* （transitivity）：`comp(a, b) == true AND comp(b, c) == true -> comp(a, c) == true`
    - 复杂度
        - `O(N·log(N))`, where `N = std::distance(first, last)` comparisons *on average* `(until C++11)`
        - `O(N·log(N))`, where `N = std::distance(first, last)` comparisons `(since C++11)`
- [`std::partial_sort`](https://en.cppreference.com/w/cpp/algorithm/partial_sort)
    - 签名
    ```
    template <class RandomIt>
    void 
    partial_sort(RandomIt first, 
                 RandomIt middle, 
                 RandomIt last);

    template <class RandomIt, class Compare>
    void 
    partial_sort(RandomIt first, 
                 RandomIt middle, 
                 RandomIt last,
                 Compare  comp);
    ```
    - 重排序列，使得完工后`[first, middle)`包含原序列中 *最小的* `middle - first` 个元素
        - **非稳定**排序
        - 完工后`[middle, last)`内元素顺序 *未定义* 
    - 复杂度：`O((last-first) log (middle-first))`次谓词调用
- [`std::partial_sort_copy`](https://en.cppreference.com/w/cpp/algorithm/partial_sort_copy)
    - 签名
    ```
    template <class InputIt, class RandomIt>
    RandomIt 
    partial_sort_copy(InputIt  first, 
                      InputIt  last,
                      RandomIt d_first, 
                      RandomIt d_last);
                            
    template <class InputIt, class RandomIt, class Compare>
    RandomIt 
    partial_sort_copy(InputIt  first, 
                      InputIt  last,
                      RandomIt d_first, 
                      RandomIt d_last,
                      Compare  comp);
    ```
    - 按照 *非降序* 排序序列`[first, last)`并将结果存储于`[d_first, d_last)`之中
        - 实际拷贝的元素个数为两个序列长度的最小值
        - **非稳定**排序
    - 返回：完工后`d_first`序列的尾后迭代器，即`d_first + std::min(last - first, d_last - d_first)`
    - 复杂度：`O(N log(min(D, N))`，`N = last - first`, `D = d_last - d_first`次谓词调用
- [`std::stable_sort`](https://en.cppreference.com/w/cpp/algorithm/stable_sort)
    - 签名
    ```
    template <class RandomIt>
    void 
    stable_sort(RandomIt first, 
                RandomIt last);

    template <class RandomIt, class Compare>
    void 
    stable_sort(RandomIt first, 
                RandomIt last, 
                Compare  comp);
    ```
    - *稳定非降序排序* 
    - 复杂度：`O(N log(N)^2)`，`N = last -  first`。如果有额外空间，则`O(N log(N))`
- [`std::nth_element`](https://en.cppreference.com/w/cpp/algorithm/nth_element)
    - 签名
    ```
    template <class RandomIt>
    void 
    nth_element(RandomIt first, 
                RandomIt nth, 
                RandomIt last);

    template <class RandomIt, class Compare>
    void 
    nth_element(RandomIt first, 
                RandomIt nth, 
                RandomIt last,
                Compare  comp);
    ```
    - 重排序列使得完工后
        1. `*nth`为原序列中第`n`大的元素（即已排好序的原序列中应该在这儿的元素）
        2. `[first, nth)`之内的元素一律不比`*nth`大
    - 复杂度：`O(last - first)`

#### 有序序列二分查找（Binary search operations (on sorted ranges)）

- [`std::lower_bound`](https://en.cppreference.com/w/cpp/algorithm/lower_bound)
    - 可能的实现
    ```
    template <class ForwardIt, class T>
    ForwardIt 
    lower_bound(ForwardIt first, 
                ForwardIt last, 
                const T & value)
    {
        ForwardIt it;
        typename std::iterator_traits<ForwardIt>::difference_type count, step;
        count = std::distance(first, last);
     
        while (count > 0) 
        {
            it = first; 
            step = count / 2; 
            std::advance(it, step);
            if (*it < value) 
            {
                first = ++it; 
                count -= step + 1; 
            }
            else
                count = step;
        }
        return first;
    }

    template <class ForwardIt, class T, class Compare>
    ForwardIt 
    lower_bound(ForwardIt first, 
                ForwardIt last, 
                const T & value, 
                Compare   comp)
    {
        ForwardIt it;
        typename std::iterator_traits<ForwardIt>::difference_type count, step;
        count = std::distance(first, last);
     
        while (count > 0) 
        {
            it = first;
            step = count / 2;
            std::advance(it, step);
            if (comp(*it, value)) 
            {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }
        return first;
    }
    ```
    - 返回：指向 *按`element < value`划分好的* 序列`[first, last)`中 *第一个* *大于等于* `val`的元素的迭代器，如没有则返回`last`
        - 区间中所有满足`element < value`，或`comp(element, value) == true`的元素 *都在* 不满足的元素 *之前* 
        - 按照非降序排序过的序列自然是划分好的
    - 复杂度：`O(log(last - first))`次比较或谓词调用；如果不是随机访问迭代器，则是`O(last - first)`
- [`std::upper_bound`](https://en.cppreference.com/w/cpp/algorithm/upper_bound)
    - 可能的实现
    ```
    template <class ForwardIt, class T>
    ForwardIt 
    upper_bound(ForwardIt first, 
                ForwardIt last, 
                const T & value)
    {
        ForwardIt it;
        typename std::iterator_traits<ForwardIt>::difference_type count, step;
        count = std::distance(first, last);
     
        while (count > 0) 
        {
            it = first; 
            step = count / 2; 
            std::advance(it, step);
            if (!(value < *it)) 
            {
                first = ++it;
                count -= step + 1;
            } 
            else
                count = step;
        }
        return first;
    }
    
    template <class ForwardIt, class T, class Compare>
    ForwardIt 
    upper_bound(ForwardIt first, 
                ForwardIt last, 
                const T & value, 
                Compare   comp)
    {
        ForwardIt it;
        typename std::iterator_traits<ForwardIt>::difference_type count, step;
        count = std::distance(first, last);
     
        while (count > 0) 
        {
            it = first; 
            step = count / 2;
            std::advance(it, step);
            if (!comp(value, *it)) 
            {
                first = ++it;
                count -= step + 1;
            } 
            else
                count = step;
        }
        return first;
    }
    ```
    - 返回：指向 *按`element <= value`划分好的* 序列`[first, last)`中 *第一个* *大于* `val`的元素的迭代器，如没有则返回`last`
        - 区间中所有满足`!(value < element)`（即`element <= value`），或`!comp(value, element) == true`的元素 *都在* 不满足的元素 *之前* 
        - 按照非降序排序过的序列自然是划分好的
    - 复杂度：`O(log(last - first))`次比较或谓词调用，如果不是随机访问迭代器，则是`O(last - first)`
- [`std::binary_search`](https://en.cppreference.com/w/cpp/algorithm/binary_search)
    - 可能的实现
    ```
    template <class ForwardIt, class T>
    bool 
    binary_search(ForwardIt first, 
                  ForwardIt last, 
                  const T & value)
    {
        first = std::lower_bound(first, last, value);
        return (!(first == last) && !(value < *first));
    }

    template <class ForwardIt, class T, class Compare>
    bool 
    binary_search(ForwardIt first, 
                  ForwardIt last, 
                  const T & value, 
                  Compare   comp)
    {
        first = std::lower_bound(first, last, value, comp);
        return (!(first == last) && !(comp(value, *first)));
    }
    ```
    - 二分查找`value`是否在 *按如下条件划分好的* 序列中出现
    - 序列必须 *同时满足* 如下三条性质
        1. 已按`element < value`或`comp(element, value)`划分好，即：满足条件的元素全部位于不满足条件的元素之前
        2. 已按`!(value < element)`（即`element <= value`）或`!comp(value, element)`划分好，即：满足条件的元素全部位于不满足条件的元素之前
        3. 对序列中任意元素，如果`element < value`或`comp(element, value)`成立，则`!(value < element)`或`!comp(value, element)`亦成立
    - 一个按照非降序完全排序过得序列自然满足上述条件
    - 复杂度：`O(log(last - first))`次比较或谓词调用；如果不是随机访问迭代器，则是`O(last - first)`
- [`std::equal_range`](https://en.cppreference.com/w/cpp/algorithm/equal_range)
    - 可能的实现
    ```
    template <class ForwardIt, class T>
    std::pair<ForwardIt, ForwardIt> 
    equal_range(ForwardIt first, 
                ForwardIt last,
                const T & value)
    {
        return std::make_pair(std::lower_bound(first, last, value),
                              std::upper_bound(first, last, value));
    }

    template <class ForwardIt, class T, class Compare>
    std::pair<ForwardIt, ForwardIt> 
    equal_range(ForwardIt first, 
                ForwardIt last,
                const T & value, 
                Compare   comp)
    {
        return std::make_pair(std::lower_bound(first, last, value, comp),
                              std::upper_bound(first, last, value, comp));
    }
    ```
    - 返回：值为`value`的区间
    - 序列必须 *同时满足* 如下三条性质
        1. 已按`element < value`或`comp(element, value)`划分好，即：满足条件的元素全部位于不满足条件的元素之前
        2. 已按`!(value < element)`（即`element <= value`）或`!comp(value, element)`划分好，即：满足条件的元素全部位于不满足条件的元素之前
        3. 对序列中任意元素，如果`element < value`或`comp(element, value)`成立，则`!(value < element)`或`!comp(value, element)`亦成立
    - 一个按照非降序完全排序过得序列自然满足上述条件
    - 复杂度：`O(2 * log(last - first))`次比较或谓词调用；如果不是随机访问迭代器，则是`O(last - first)`

#### 有序序列二路归并（Merge operations on sorted ranges）

- [`std::merge`](https://en.cppreference.com/w/cpp/algorithm/merge)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt 
    merge(InputIt1 first1, 
          InputIt1 last1,
          InputIt2 first2, 
          InputIt2 last2,
          OutputIt d_first)
    {
        for (; first1 != last1; ++d_first) 
        {
            if (first2 == last2) 
            {
                return std::copy(first1, last1, d_first);
            }
            if (*first2 < *first1) 
            {
                *d_first = *first2;
                ++first2;
            } 
            else 
            {
                *d_first = *first1;
                ++first1;
            }
        }
        return std::copy(first2, last2, d_first);
    }
    
    template <class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt 
    merge(InputIt1 first1, 
          InputIt1 last1,
          InputIt2 first2, 
          InputIt2 last2,
          OutputIt d_first, 
          Compare  comp)
    {
        for (; first1 != last1; ++d_first) 
        {
            if (first2 == last2) 
            {
                return std::copy(first1, last1, d_first);
            }
            if (comp(*first2, *first1)) 
            {
                *d_first = *first2;
                ++first2;
            } 
            else 
            {
                *d_first = *first1;
                ++first1;
            }
        }
        return std::copy(first2, last2, d_first);
    }
    ```
    - 归并两个 *非降序列* `[first1, last1)`和`[first2, last2)`至以`d_first`开始的一片内存中
    - 复杂度：`O(std::distance(first1, last1) + std::distance(first2, last2))`
- [`std::inplace_merge`](https://en.cppreference.com/w/cpp/algorithm/inplace_merge)
    - 签名
    ```
    template <class BidirIt>
    void 
    inplace_merge(BidirIt first, 
                  BidirIt middle, 
                  BidirIt last);

    template <class BidirIt, class Compare>
    void 
    inplace_merge(BidirIt first, 
                  BidirIt middle, 
                  BidirIt last, 
                  Compare comp);
    ```
    - 就地归并两个 *连续的* *非降序列* `[first, middle)`和`[middle, last)`成为`[first, last)`
    - 复杂度：`Omega(N - 1)`次比较（内存足够）否则，`O(N log N)`次比较，`N = std::distance(first, last)`

#### 有序序列集合操作（Set operations (on sorted ranges)）

- [`std::includes`](https://en.cppreference.com/w/cpp/algorithm/includes)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2>
    bool 
    includes(InputIt1 first1, 
             InputIt1 last1,
             InputIt2 first2, 
             InputIt2 last2)
    {
        for (; first2 != last2; ++first1)
        {
            if (first1 == last1 || *first2 < *first1)
                return false;
            if (!(*first1 < *first2))
                ++first2;
        }
        return true;
    }

    template <class InputIt1, class InputIt2, class Compare>
    bool 
    includes(InputIt1 first1, 
             InputIt1 last1,
             InputIt2 first2, 
             InputIt2 last2, 
             Compare  comp)
    {
        for (; first2 != last2; ++first1)
        {
            if (first1 == last1 || comp(*first2, *first1))
                return false;
            if (!comp(*first1, *first2))
                ++first2;
        }
        return true;
    }
    ```
    - 检查 *有序序列* `[first2, last2)`是否是 *有序序列* `[first1, last1)`的 *子列*
        - *子列* *不一定连续*
    - 复杂度：`O(2 * (N1 + N2 - 1))`次比较，`N1 = std::distance(first1, last1)`，`N2 = std::distance(first2, last2)`     
- [`std::set_difference`](https://en.cppreference.com/w/cpp/algorithm/set_difference)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt 
    set_difference(InputIt1 first1, 
                   InputIt1 last1,
                   InputIt2 first2, 
                   InputIt2 last2,
                   OutputIt d_first)
    {
        while (first1 != last1) 
        {
            if (first2 == last2) return std::copy(first1, last1, d_first);
     
            if (*first1 < *first2) 
            {
                *d_first++ = *first1++;
            } 
            else 
            {
                if (!(*first2 < *first1)) 
                {
                    ++first1;
                }
                ++first2;
            }
        }
        return d_first;
    }

    template <class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt 
    set_difference(InputIt1 first1, 
                   InputIt1 last1,
                   InputIt2 first2, 
                   InputIt2 last2,
                   OutputIt d_first, 
                   Compare  comp)
    {
        while (first1 != last1) 
        {
            if (first2 == last2) return std::copy(first1, last1, d_first);
     
            if (comp(*first1, *first2)) 
            {
                *d_first++ = *first1++;
            } 
            else 
            {
                if (!comp(*first2, *first1)) 
                {
                    ++first1;
                }
                ++first2;
            }
        }
        return d_first;
    }
    ```
    - 将 *有序序列* `[first1, last1)`中 *没有出现* 在 *有序序列* `[first2, last2)`中的元素拷贝至以`d_first`开始的一片内存中
        - 结果也是 *有序序列* 
        - 相等元素被平等看待，即如果`v`出现在序列1中`m`次、2中`n`次，则结果中会含有`std::max(0, m - n)`个`v`
    - 返回：构造序列的尾后迭代器
    - 复杂度：`O(2 * (N1 + N2 - 1))`次比较，`N1 = std::distance(first1, last1)`，`N2 = std::distance(first2, last2)`     
- [`std::set_intersection`](https://en.cppreference.com/w/cpp/algorithm/set_intersection)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt 
    set_intersection(InputIt1 first1, 
                     InputIt1 last1,
                     InputIt2 first2, 
                     InputIt2 last2,
                     OutputIt d_first)
    {
        while (first1 != last1 && first2 != last2) 
        {
            if (*first1 < *first2) 
            {
                ++first1;
            } 
            else  
            {
                if (!(*first2 < *first1)) 
                {
                    *d_first++ = *first1++;
                }
                ++first2;
            }
        }
        return d_first;
    }
    
    template <class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt 
    set_intersection(InputIt1 first1, 
                     InputIt1 last1,
                     InputIt2 first2, 
                     InputIt2 last2,
                     OutputIt d_first, 
                     Compare  comp)
    {
        while (first1 != last1 && first2 != last2) 
        {
            if (comp(*first1, *first2)) 
            {
                ++first1;
            } 
            else 
            {
                if (!comp(*first2, *first1)) 
                {
                    *d_first++ = *first1++;
                }
                ++first2;
            }
        }
        return d_first;
    }
    ```
    - 将 *同时出现* 在 *有序序列* `[first1, last1)`中和 *有序序列* `[first2, last2)`中的元素拷贝至以`d_first`开始的一片内存中
        - 结果也是 *有序序列* 
        - 相等元素被平等看待，即如果`v`出现在序列1中`m`次、2中`n`次，则结果中会含有`std::min(m, n)`个`v`
    - 返回：构造序列的尾后迭代器
    - 复杂度：`O(2 * (N1 + N2 - 1))`次比较，`N1 = std::distance(first1, last1)`，`N2 = std::distance(first2, last2)`    
- [`std::set_symmetric_difference`](https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt 
    set_symmetric_difference(InputIt1 first1, 
                             InputIt1 last1,
                             InputIt2 first2, 
                             InputIt2 last2,
                             OutputIt d_first)
    {
        while (first1 != last1) 
        {
            if (first2 == last2) return std::copy(first1, last1, d_first);
     
            if (*first1 < *first2) 
            {
                *d_first++ = *first1++;
            } 
            else 
            {
                if (*first2 < *first1) 
                {
                    *d_first++ = *first2;
                } 
                else 
                {
                    ++first1;
                }
                ++first2;
            }
        }
        return std::copy(first2, last2, d_first);
    }

    template <class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt 
    set_symmetric_difference(InputIt1 first1, 
                             InputIt1 last1,
                             InputIt2 first2, 
                             InputIt2 last2,
                             OutputIt d_first, 
                             Compare  comp)
    {
        while (first1 != last1) 
        {
            if (first2 == last2) return std::copy(first1, last1, d_first);
     
            if (comp(*first1, *first2)) 
            {
                *d_first++ = *first1++;
            } 
            else 
            {
                if (comp(*first2, *first1)) 
                {
                    *d_first++ = *first2;
                } 
                else 
                {
                    ++first1;
                }
                ++first2;
            }
        }
        return std::copy(first2, last2, d_first);
    }
    ```
    - 将 *只在* *有序序列* `[first1, last1)`中出现或 *只在* *有序序列* `[first2, last2)`中出现的元素拷贝至以`d_first`开始的一片内存中
        - 结果也是 *有序序列* 
        - 相等元素被平等看待，即如果`v`出现在序列1中`m`次、2中`n`次，则结果中会含有`std::abs(m - n)`个`v`
    - 返回：构造序列的尾后迭代器
    - 复杂度：`O(2 * (N1 + N2 - 1))`次比较，`N1 = std::distance(first1, last1)`，`N2 = std::distance(first2, last2)`  
- [`std::set_union`](https://en.cppreference.com/w/cpp/algorithm/set_union)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt 
    set_union(InputIt1 first1, 
              InputIt1 last1,
              InputIt2 first2, 
              InputIt2 last2,
              OutputIt d_first)
    {
        for (; first1 != last1; ++d_first) 
        {
            if (first2 == last2)
                return std::copy(first1, last1, d_first);
            if (*first2 < *first1) 
            {
                *d_first = *first2++;
            } 
            else 
            {
                *d_first = *first1;
                if (!(*first1 < *first2))
                    ++first2;
                ++first1;
            }
        }
        return std::copy(first2, last2, d_first);
    }

    template <class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt 
    set_union(InputIt1 first1, 
              InputIt1 last1,
              InputIt2 first2, 
              InputIt2 last2,
              OutputIt d_first, 
              Compare  comp)
    {
        for (; first1 != last1; ++d_first) 
        {
            if (first2 == last2)
                return std::copy(first1, last1, d_first);
            if (comp(*first2, *first1)) 
            {
                *d_first = *first2++;
            } 
            else 
            {
                *d_first = *first1;
                if (!comp(*first1, *first2))
                    ++first2;
                ++first1;
            }
        }
        return std::copy(first2, last2, d_first);
    }
    ```
    - 将在 *有序序列* `[first1, last1)`中出现 *或* 在 *有序序列* `[first2, last2)`中出现的元素拷贝至以`d_first`开始的一片内存中
        - 结果也是 *有序序列* 
        - 相等元素被平等看待，即如果`v`出现在序列1中`m`次、2中`n`次，则结果中会含有`std::max(m, n)`个`v`
            - 先把序列1中的`m`个全部 *依次* 拷贝过去，之后 *依次* 拷贝序列2中的前`std::max(n - m, 0)`个
    - 返回：构造序列的尾后迭代器
    - 复杂度：`O(2 * (N1 + N2 - 1))`次比较，`N1 = std::distance(first1, last1)`，`N2 = std::distance(first2, last2)`  

#### 堆操作（Heap operations）

- [`std::is_heap`](https://en.cppreference.com/w/cpp/algorithm/is_heap)
    - 签名
    ```
    template <class RandomIt>
    bool 
    is_heap(RandomIt first, 
            RandomIt last);

    template <class RandomIt, class Compare>
    bool 
    is_heap(RandomIt first, 
            RandomIt last, 
            Compare  comp);
    ```
    - 判断序列`[first, last)`是否是一个 *大顶堆* （max heap）
        - *大顶堆* 是满足如下性质的区间`[f, l)`
            1. 记`N = l - f`，则对任意`0 < i < N`，`f[floor((i - 1) / 2)] >= f[i]`（即，逻辑二叉树结构中的爹大于等于儿子）
            2. 可用[`std::push_heap`](https://en.cppreference.com/w/cpp/algorithm/push_heap)插入元素
            3. 可用[`std::pop_heap`](https://en.cppreference.com/w/cpp/algorithm/pop_heap)弹出堆顶元素
    - 复杂度：`O(2 * log(last - first))`
- [`std::is_heap_until`](https://en.cppreference.com/w/cpp/algorithm/is_heap_until)
    - 签名
    ```
    template <class RandomIt>
    RandomIt 
    is_heap_until(RandomIt first, 
                  RandomIt last);

    template <class RandomIt, class Compare>
    RandomIt 
    is_heap_until(RandomIt first,
                  RandomIt last, 
                  Compare  comp);
    ```
    - 返回：满足`[first, it)`是 *大顶堆* 的 *最靠后* 的迭代器`it`
    - 复杂度：`O(last - first)`
- [`std::make_heap`](https://en.cppreference.com/w/cpp/algorithm/make_heap)
    - 签名
    ```
    template <class RandomIt>
    void 
    make_heap(RandomIt first, 
              RandomIt last);

    template <class RandomIt, class Compare>
    void 
    make_heap(RandomIt first, 
              RandomIt last,
              Compare  comp);
    ```
    - 在`[first, last)`内建 *大顶堆*
    - 复杂度：`O(3 * std::distance(first, last))`次比较
- [`std::push_heap`](https://en.cppreference.com/w/cpp/algorithm/push_heap)
    - 签名
    ```
    template <class RandomIt>
    void 
    push_heap(RandomIt first, 
              RandomIt last);

    template<class RandomIt, class Compare>
    void 
    push_heap(RandomIt first, 
              RandomIt last,
              Compare  comp);
    ```
    - 将`last - 1`所指元素插入 *大顶堆* `[first, last - 1)`
    - 复杂度：`O(log(std::distance(first, last)))`次比较
- [`std::pop_heap`](https://en.cppreference.com/w/cpp/algorithm/pop_heap)
    - 签名
    ```
    template <class RandomIt>
    void 
    pop_heap(RandomIt first, 
             RandomIt last);

    template <class RandomIt, class Compare>
    void 
    pop_heap(RandomIt first, 
             RandomIt last, 
             Compare  comp);
    ```
    - 将`first`弹出 *大顶堆* `[first, last)`
        - *对换* `first`和`last - 1`，然后使区间`[first, last - 1)`重新成为堆（下沉`last - 1`即可）
    - 复杂度：`O(2 * log(last - first))`次比较
- [`std::sort_heap`](https://en.cppreference.com/w/cpp/algorithm/sort_heap)
    - 签名
    ```
    template <class RandomIt>
    void 
    sort_heap(RandomIt first, 
              RandomIt last);
    
    template <class RandomIt, class Compare>
    void 
    sort_heap(RandomIt first, 
              RandomIt last, 
              Compare  comp);
    ```
    - 将 *大顶堆* `[first, last)`转换成 *非降序* 序列
        - 新序列自然不再是 *大顶堆* 了
    - 复杂度：`O(2 * N * log(N))`次比较，`N = std::distance(first, last)`

#### 最值（Minimum/maximum operations）

- [`std::max`](https://en.cppreference.com/w/cpp/algorithm/max)
    - 可能的实现
    ```
    template <class T> 
    const T & 
    max(const T & a, 
        const T & b)
    {
        return (a < b) ? b : a;
    }

    template <class T, class Compare> 
    const T & 
    max(const T & a, 
        const T & b, 
        Compare   comp)
    {
        return (comp(a, b)) ? b : a;
    }

    template <class T>
    T 
    max(std::initializer_list<T> ilist)
    {
        return *std::max_element(ilist.begin(), ilist.end());
    }

    template <class T, class Compare>
    T 
    max(std::initializer_list<T> ilist, 
        Compare                  comp)
    {
        return *std::max_element(ilist.begin(), ilist.end(), comp);
    }
    ```
    - 返回：2个元素的最大值或整个初始化列表中的最大值
    - 复杂度：两元素版`O(1)`，初始化列表版`Omega(ilist.size() - 1)`
- [`std::max_element`](https://en.cppreference.com/w/cpp/algorithm/max_element)
    - 可能的实现
    ```
    template <class ForwardIt>
    ForwardIt 
    max_element(ForwardIt first, 
                ForwardIt last)
    {
        if (first == last) return last;
     
        ForwardIt largest = first;
        ++first;
        for (; first != last; ++first) 
        {
            if (*largest < *first) 
            {
                largest = first;
            }
        }
        return largest;
    }

    template <class ForwardIt, class Compare>
    ForwardIt 
    max_element(ForwardIt first, 
                ForwardIt last, 
                Compare   comp)
    {
        if (first == last) return last;
     
        ForwardIt largest = first;
        ++first;
        for (; first != last; ++first) 
        {
            if (comp(*largest, *first)) 
            {
                largest = first;
            }
        }
        return largest;
    }
    ```
    - 返回：`[first, last)`中 *最大* 的元素的迭代器
    - 复杂度：`Omega(max(N - 1,0))`次比较，`N = std::distance(first, last)` 
- [`std::min`](https://en.cppreference.com/w/cpp/algorithm/min)
- [`std::min_element`](https://en.cppreference.com/w/cpp/algorithm/min_element)
- [`std::minmax`](https://en.cppreference.com/w/cpp/algorithm/minmax)
    - 返回最大值和最小值组成的`std::pair`
- [`std::minmax_element`](https://en.cppreference.com/w/cpp/algorithm/minmax_element)
    - 返回最大值和最小值的迭代器组成的`std::pair`
- [`std::clamp`](https://en.cppreference.com/w/cpp/algorithm/clamp) `(since C++17)`
    - 可能的实现
    ```
    template <class T>
    constexpr const T & 
    clamp(const T & v,
          const T & lo, 
          const T & hi)
    {
        assert(!(hi < lo));
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }
    
    template <class T, class Compare>
    constexpr const T & 
    clamp(const T & v, 
          const T & lo, 
          const T & hi, 
          Compare   comp)
    {
        assert(!comp(hi, lo));
        return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
    }
    ```
    - 返回：如果`lo < v < hi`，返回`v`；否则`lo`和`hi`中返回和`v`同侧的那个， *掐头去尾*
    - 复杂度：`O(1)`

#### 比较（Comparison operations）

- [`std::equal`](https://en.cppreference.com/w/cpp/algorithm/equal)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2>
    bool 
    equal(InputIt1 first1, 
          InputIt1 last1, 
          InputIt2 first2)
    {
        for (; first1 != last1; ++first1, ++first2)
        {
            if (!(*first1 == *first2)) 
            {
                return false;
            }
        }
        return true;
    }
    
    template <class InputIt1, class InputIt2>
    bool 
    equal(InputIt1 first1, 
          InputIt1 last1, 
          InputIt2 first2, 
          InputIt2 last2);

    template <class InputIt1, class InputIt2, class BinaryPredicate>
    bool 
    equal(InputIt1        first1, 
          InputIt1        last1, 
          InputIt2        first2,
          BinaryPredicate p)
    {
        for (; first1 != last1; ++first1, ++first2) 
        {
            if (!p(*first1, *first2)) 
            {
                return false;
            }
        }
        return true;
    }
    
    template <class InputIt1, class InputIt2, class BinaryPredicate>
    bool 
    equal(InputIt1        first1, 
          InputIt1        last1, 
          InputIt2        first2, 
          InputIt2        last2,
          BinaryPredicate p);
    ```
    - 返回：如果 *序列1* 中所有元素都与 *序列2* 中对应位置元素满足`*iter1 == *iter2`或`p(*iter1, *iter2) == true`，则返回`true`，反之返回`false`
    - 复杂度：`O(N)`
- [`std::lexicographical_compare`](https://en.cppreference.com/w/cpp/algorithm/lexicographical_compare)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2>
    bool 
    lexicographical_compare(InputIt1 first1, 
                            InputIt1 last1,
                            InputIt2 first2, 
                            InputIt2 last2)
    {
        for ( ; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2 ) 
        {
            if (*first1 < *first2) return true;
            if (*first2 < *first1) return false;
        }
        return (first1 == last1) && (first2 != last2);
    }

    template <class InputIt1, class InputIt2, class Compare>
    bool 
    lexicographical_compare(InputIt1 first1, 
                            InputIt1 last1,
                            InputIt2 first2, 
                            InputIt2 last2,
                            Compare  comp)
    {
        for ( ; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2 ) 
        {
            if (comp(*first1, *first2)) return true;
            if (comp(*first2, *first1)) return false;
        }
        return (first1 == last1) && (first2 != last2);
    }
    ```
    - 检查 *序列1* 是否按 *字典序* *小于* *序列2*
        - *字典序比较* 是拥有下列属性的操作
            1. 逐元素比较二个范围
            2. 首个不匹配元素定义范围是否按字典序小于或大于另一个
            3. 若一个范围是另一个的前缀，则较短的范围小于另一个
            4. 若二个范围拥有等价元素和相同长度，则范围按字典序相等
            5. 空范围按字典序小于任何非空范围
            6. 二个空范围按字典序相等
    - 复杂度：`O(2 * min(N1, N2)`次比较，`N1 = std::distance(first1, last1)`，`N2 = std::distance(first2, last2)` 

#### 排列算法（Permutation operations）

- [`std::is_permutation`](https://en.cppreference.com/w/cpp/algorithm/is_permutation)
    - 签名
    ```
    template <class ForwardIt1, class ForwardIt2>
    bool 
    is_permutation(ForwardIt1 first, 
                   ForwardIt1 last,
                   ForwardIt2 d_first)；
                   
    template <class ForwardIt1, class ForwardIt2, class BinaryPredicate>
    bool 
    is_permutation(ForwardIt1      first1, 
                   ForwardIt1      last1,
                   ForwardIt2      first2, 
                   BinaryPredicate p);
                   
    template <class ForwardIt1, class ForwardIt2>
    bool 
    is_permutation(ForwardIt1 first1, 
                   ForwardIt1 last1,
                   ForwardIt2 first2, 
                   ForwardIt2 last2);
                     
    template <class ForwardIt1, class ForwardIt2, class BinaryPredicate>
    bool 
    is_permutation(ForwardIt1      first1, 
                   ForwardIt1      last1,
                   ForwardIt2      first2, 
                   ForwardIt2      last2,
                   BinaryPredicate p);
    ```
    - 检查 *序列1* 是否是 *序列2* 的一个排列
    - 复杂度：`O(N^2)`次谓词调用。若两序列相等，`O(N)`，`N = std::distance(first1, last1)`
        - 如果迭代器支持 *随机访问* ，且`std::distance(first1, last1) != std::distance(first2, last2)`，则没有谓词调用 
- [`std::next_permutation`](https://en.cppreference.com/w/cpp/algorithm/next_permutation)
    - 签名
    ```
    template <class BidirIt>
    bool 
    next_permutation(BidirIt first, 
                     BidirIt last);

    template <class BidirIt, class Compare>
    bool 
    next_permutation(BidirIt first, 
                     BidirIt last, 
                     Compare comp);
    ```
    - 将序列重排为下一个全排列，如果存在返回`true`，否则将其重排为第一个全排列`std::sort(first, last)`并返回`false`
    - 复杂度：`O(N / 2)`次对换，`N = std::distance(first, last)`
- [`std::prev_permutation`](https://en.cppreference.com/w/cpp/algorithm/prev_permutation)
    - 签名
    ```
    template <class BidirIt>
    bool 
    prev_permutation(BidirIt first, 
                     BidirIt last);

    template <class BidirIt, class Compare>
    bool 
    prev_permutation(BidirIt first, 
                     BidirIt last, 
                     Compare comp);
    ```
    - 将序列重排为上一个全排列，如果存在返回`true`，否则将其重排为最后一个全排列`std::sort(first, last); std::sort(first, last); std::reverse(first, last);`并返回`false`
    - 复杂度：`O(N / 2)`次对换，`N = std::distance(first, last)`

#### 数值操作（Numeric operations）

- [`std::iota`](https://en.cppreference.com/w/cpp/algorithm/iota)
    - 可能的实现
    ```
    template <class ForwardIt, class T>
    void 
    iota(ForwardIt first, 
         ForwardIt last, 
         T         value)
    {
        while (first != last) 
        {
            *first++ = value;
            ++value;
        }
    }
    ```
    - 以始于`value`并重复地求值`++value`的顺序递增值填充范围`[first, last)`
    - `O(last - first)`
- [`std::accumulate`](https://en.cppreference.com/w/cpp/algorithm/accumulate)
    - 可能的实现
    ```
    template <class InputIt, class T>
    constexpr T 
    accumulate(InputIt first, 
               InputIt last, 
               T       init)
    {
        for (; first != last; ++first) 
        {
            init = std::move(init) + *first;     // std::move since C++20
        }
        
        return init;
    }

    template <class InputIt, class T, class BinaryOperation>
    constexpr T 
    accumulate(InputIt         first, 
               InputIt         last, 
               T               init, 
               BinaryOperation op)
    {
        for (; first != last; ++first) 
        {
            init = op(std::move(init), *first);  // std::move since C++20
        }
        
        return init;
    }
    ```
    - 返回：区间`[first, last)`之内所有元素以及`init`的 *基于* `op` 的 *总和* 
    - 复杂度：`O(last - first)`次谓词调用
- [`std::inner_product`](https://en.cppreference.com/w/cpp/algorithm/inner_product)
    - 可能的实现
    ```
    template <class InputIt1, class InputIt2, class T>
    constexpr T 
    inner_product(InputIt1 first1, 
                  InputIt1 last1,
                  InputIt2 first2, 
                  T        init)
    {
        while (first1 != last1) 
        {
             init = std::move(init) + *first1 * *first2;          // std::move since C++20
             ++first1;
             ++first2;
        }
        
        return init;
    }

    template <class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2>
    constexpr T 
    inner_product(InputIt1         first1, 
                  InputIt1         last1,
                  InputIt2         first2, 
                  T                init,
                  BinaryOperation1 op1
                  BinaryOperation2 op2)
    {
        while (first1 != last1) 
        {
             init = op1(std::move(init), op2(*first1, *first2));  // std::move since C++20
             ++first1;
             ++first2;
        }
        
        return init;
    }
    ```
    - 顾名思义
- [`std::adjacent_difference`](https://en.cppreference.com/w/cpp/algorithm/adjacent_difference)
    - 可能的实现
    ```
    template <class InputIt, class OutputIt>
    constexpr OutputIt 
    adjacent_difference(InputIt  first, 
                        InputIt  last, 
                        OutputIt d_first)
    {
        if (first == last) return d_first;
     
        typedef typename std::iterator_traits<InputIt>::value_type value_t;
        value_t acc = *first;
        *d_first = acc;
        
        while (++first != last) 
        {
            value_t val = *first;
            *++d_first = val - std::move(acc); // std::move since C++20
            acc = std::move(val);
        }
        return ++d_first;
    }

    template <class InputIt, class OutputIt, class BinaryOperation>
    constexpr OutputIt 
    adjacent_difference(InputIt         first, 
                        InputIt         last, 
                        OutputIt        d_first, 
                        BinaryOperation op)
    {
        if (first == last) return d_first;
     
        typedef typename std::iterator_traits<InputIt>::value_type value_t;
        value_t acc = *first;
        *d_first = acc;
        while (++first != last) {
            value_t val = *first;
            *++d_first = op(val, std::move(acc)); // std::move since C++20
            acc = std::move(val);
        }
        return ++d_first;
    }
    ```
    - 等价于
    ```
    *(d_first)   = *first;
    *(d_first+1) = *(first+1) - *(first);
    *(d_first+2) = *(first+2) - *(first+1);
    *(d_first+3) = *(first+3) - *(first+2);
    ...
    ```
    - 返回：生成序列的尾后迭代器
    - 复杂度：`O((last - first) - 1)`
- [`std::partial_sum`](https://en.cppreference.com/w/cpp/algorithm/partial_sum)
    - 可能的实现
    ```
    template <class InputIt, class OutputIt>
    constexpr OutputIt 
    partial_sum(InputIt  first, 
                InputIt  last, 
                OutputIt d_first)
    {
        if (first == last) return d_first;
     
        typename std::iterator_traits<InputIt>::value_type sum = *first;
        *d_first = sum;
     
        while (++first != last) 
        {
           sum = std::move(sum) + *first;    // std::move since C++20
           *++d_first = sum;
        }
        return ++d_first;
     
        // or, since C++14:
        // return std::partial_sum(first, last, d_first, std::plus<>());
    }

    template<class InputIt, class OutputIt, class BinaryOperation>
    constexpr OutputIt 
    partial_sum(InputIt         first, 
                InputIt         last, 
                OutputIt        d_first, 
                BinaryOperation op)
    {
        if (first == last) return d_first;
     
        typename std::iterator_traits<InputIt>::value_type sum = *first;
        *d_first = sum;
     
        while (++first != last) 
        {
           sum = op(std::move(sum), *first); // std::move since C++20
           *++d_first = sum;
        }
        return ++d_first;
    }
    ```
    - 等价于
    ```
    *(d_first)   = *first;
    *(d_first+1) = *first + *(first+1);
    *(d_first+2) = *first + *(first+1) + *(first+2);
    *(d_first+3) = *first + *(first+1) + *(first+2) + *(first+3);
    ...
    ```
    - 返回：生成序列的尾后迭代器
    - 复杂度：`O((last - first) - 1)`






### 🌱 [Chap 11] [关联容器](https://en.cppreference.com/w/cpp/container)（Associative Container）

#### 关联容器概述

- 关联容器类型
    - *有序关联容器* ，按 *键*（key，关键字） *有序* 保存元素，使用 *红黑树* （red-black tree）实现
        - [`std::map`](https://en.cppreference.com/w/cpp/container/map)： *关联数组* （associative array），保存 *键-值词条* （entry，`<key, value>`）
        - [`std::set`](https://en.cppreference.com/w/cpp/container/set)：只保存键
        - [`std::multimap`](https://en.cppreference.com/w/cpp/container/multimap)：键可重复出现的`std::map`
        - [`std::multiset`](https://en.cppreference.com/w/cpp/container/multiset)：键可重复出现的`std::set`
    - *无序关联容器* （unordered associative container），使用 *散列表* （hash table）实现
        - [`std::unordered_map`](https://en.cppreference.com/w/cpp/container/unordered_map)：散列组织的`std::map`
        - [`std::unordered_set`](https://en.cppreference.com/w/cpp/container/unordered_set)：散列组织的`std::set`
        - [`std::unordered_multimap`](https://en.cppreference.com/w/cpp/container/unordered_multimap)：散列组织的`std::map`，键可重复出现
        - [`std::unordered_multiset`](https://en.cppreference.com/w/cpp/container/unordered_multiset)：散列组织的`std::set`，键可重复出现
- 使用举例
    - 使用`std::map`（计数）
    ```
    std::map<std::string, size_t> word_count;
    std::string word;
    
    while (std::cin >> word)
    {
        ++word_count[word]; 
    }
        
    for (const std::pair<std::string, size_t> & w : word_count)
    {
        printf("\"%s\" occurs %zu time(s)\n", w.first.c_str(), w.second);
    }
    ```
    - 使用`std::set`（去重）
    ```
    std::map<std::string, size_t> word_count;
    std::set<std::string> exclude = {"The", "But", "And", "Or", "An", "A", "the", "but", "and", "or", "an", "a"};
    std::string word;
    
    while (std::cin >> word)
    {
        if (exclude.find(word) == exclude.end())
        {
            ++word_count[word];
        }    
    }
    ```
- 定义（初始化）`std::map`和`std::set`
    - 每个关联容器都定义了默认构造函数，用于创建指定类型的空容器
    - 也可以将关联容器创建为其他关联容器的拷贝
    - 或者从一个值范围来初始化关联容器
        - 对于`std::map`，必须提供键值对`<key, value>`
    ```
    // empty
    map<std::string, size_t> word_count; 
    
    // list initialization
    std::set<string> exclude = {"the", "but", "and", "or", "an", "a", "The", "But", "And", "Or", "An", "A"};
    
    // three elements; authors maps last name to first
    std::map<std::string, string> authors = {{"Joyce", "James"}, {"Austen", "Jane"}, {"Dickens", "Charles"}};
    ```
- 定义（初始化）`std::multi_map`和`std::multi_set`
    - `std::map`和`std::set`的键必须是唯一的
        - 插入重复的键或词条会被 *忽略*
    - `std::multi_map`和`std::multi_set`允许键重复
        - 插入重复的键或词条会被保留
    ```
    // define a vector with 20 elements, holding two copies of each number from 0 to 9
    std::vector<int> ivec;
    
    for (int i = 0; i != 10; ++i) 
    {
        ivec.push_back(i);
        ivec.push_back(i);        // duplicate copies of each number
    }
    
    cout << ivec.size() << endl;  // prints 20
    
    // iset holds unique elements from ivec; 
    std::set<int> iset(ivec.cbegin(), ivec.cend());
    cout << iset.size() << endl;  // prints 10
    
    // miset holds all 20 elements
    std::multiset<int> miset(ivec.cbegin(), ivec.cend());
    cout << miset.size() << endl; // prints 20
    ```
- 键类型要求
    - 必须定义了 *严格弱序* （stirct weak ordering，例：`<=`运算符）
        1. `a <= b`和`b <= a`有且仅能有一个成立
        2. `a <= b`且`b <= c`，则`a <= c`
        3. `!(a <= b) && !(b <= a)`意味着`a == b`
    - 在实际编程中重要的是，如果类型定义了 *行为正常* 的`<`运算符，则可以用它做键
    - 对于没有重载运算符的自定义类型，`std::multiset`允许传入 *谓词* 
    ```
    inline bool compareIsbn(const Sales_data & lhs, const Sales_data & rhs)
    {
        return lhs.isbn() < rhs.isbn();
    }

    // bookstore can have several transactions with the same ISBN
    // elements in bookstore will be in ISBN order
    // 3 equivalent declarations
    std::multiset<Sales_data, decltype(compareIsbn) *>                                      bookstore1(compareIsbn);
    std::multiset<Sales_data, bool (*)(const Sales_data &, const Sales_data &)>             bookstore2(compareIsbn);
    std::multiset<Sales_data, std::function<bool (const Sales_data &, const Sales_data &)>> bookstore3(compareIsbn);
    ```
- [`std::pair`](https://en.cppreference.com/w/cpp/utility/pair)
    - 定义于`<utility>`中
    - 默认构造函数对成员进行 *值初始化*
        - `std::string`，`std::vector`被初始化成空容器，`size_t`被初始化为`0`
    - 可以显式调用构造函数传参初始化，或进行 *列表初始化*
    - 数据成员`first`、`second`为 *公有*
    ```
    std::pair<std::string, std::string>      anon;        // holds two strings
    std::pair<std::string, size_t>           word_count;  // holds a string and an size_t
    std::pair<std::string, std::vector<int>> line;        // holds string and vector<int>
    
    std::pair<std::string, std::string>      author{"James", "Joyce"};
    
    // print the results
    std::cout << w.first << " occurs " << w.second << ((w.second > 1) ? " times" : " time") << std::endl;
    ```
    - 生成`std::pair`
        - `std::pair<T1, T2> p;`： *默认初始化* ，创建`std::pair`，成员进行值初始化
        - `std::pair<T1, T2> p(v1, v2);`： *显式构造* ，创建`std::pair`，成员初始化为给定值
        - `std::pair<T1, T2> p = {v1, v2};`： *列表初始化* ，创建`std::pair`，成员初始化为给定值
        - `std::make_pair(v1, v2);`：创建`std::pair`，元素类型由`v1`和`v2`自动推断。成员初始化为给定值
    - 访问`std::pair` 
        - `p.first`
        - `p.second`
        - `p1 rel_op p2`： *关系运算符* `<`，`>`，`<=`，`>=`，按 *字典序* 定义
        - `p1 == p2`
        - `p1 != p2`
    - 返回`std::pair`
    ```
    std::pair <string, int> process(std::vector<std::string> & v)
    {
        if (!v.empty())
            return {v.back(), v.back().size()};    // list initialize
        else
            return std::pair<std::string, int>();  // explicitly constructed return value
    }
    ```

#### 关联容器操作

- 类型别名
    - `key_type`：此容器类型的键类型
    - `mapped_type`：四种`map`类型中、每个键关联的类型
    - `value_type`：对于`std::set`，与`key_type`相同；对于`std::map`，为`std::pair<const key_type, mapped_type>`
        - 不能改变元素的键，因为键是 *常量* 
```
std::set<std::string>::value_type v1;        // std::string
std::set<std::string>::key_type v2;          // std::string
std::map<std::string, int>::value_type v3;   // std::pair<const std::string, int>
std::map<std::string, int>::key_type v4;     // std::string
std::map<std::string, int>::mapped_type v5;  // int
```
- 迭代器
    - 解引用关联容器迭代器时，会得到类型为容器类型的`value_type`的值的 *引用* 
        - 对`std::map`而言，是`std::pair<const key_type, mapped_type>`类型
        - 必须记住，一个`std::map`的`value_type`是一个`std::pair`，对一个词条，值可以变，**键不能变**
    ```
    // get an iterator to an element in word_count
    std::map<std::string, size_t>::iterator map_it = word_count.begin();
    // *map_it is a reference to a pair<const string, size_t> object
    std::cout << map_it->first;          // prints the key for this element
    std::cout << " " << map_it->second;  // prints the value of the element
    map_it->first = "new key";           // error: key is const
    ++map_it->second;                    // ok: we can change the value through an iterator
    ```
    - `std::set`的迭代器 *全部是* *常迭代器*
        - `std::set`的`iterator`和`const_iterator` *全部是* *常迭代器*
        - 键可以读，但不能改
    ```
    std::set<int> iset = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::set<int>::iterator set_it = iset.begin();
    
    if (set_it != iset.end()) 
    {
        *set_it = 42;                       // error: keys in a set are read-only
        std::cout << *set_it << std::endl;  // ok: can read the key
    }
    ```
    - 遍历关联容器
        - `std::map`和`std::set`都支持`begin()`、`end()`等操作
        - *有序关联容器* 迭代器按照 *键升序* 遍历元素
    ```
    std::map<std::string, size_t>::iterator map_it = word_count.cbegin();

    while (map_it != word_count.cend()) 
    {
        cout << map_it->first << " occurs " << map_it->second << " times" << endl;
        ++map_it; 
    }
    ```
    - 关联容器和泛型算法
        - 通常**不对**关联容器使用 *泛型算法* => 10
            - 因为泛型算法经常 *写值* ，但关联容器的键是 *常量* 
        - 关联容器 *只可用于* 只读算法
            - 只读算法经常 *搜索* ，但关联容器迭代器**不支持随机访问**，性能会很差
        - 实际编程中，关联容器要么作为源序列，要么作为目的位置。例如
            1. 调用`std::copy`将一个关联容器的元素拷贝到另一个序列中
            2. 将 *插入迭代器* 绑定到关联容器上，将关联容器作为目的位置
- 添加元素
    - 关联容器的`insert`操作
        - `c.insert(v)`，`c.emplace(args)`：返回`std::pair<iterator, bool>`，包含指向具有指定键的元素，以及指示插入是否成功的`bool`。对于`std::multi_set`和`std::multi_map`，总会插入给定元素并返回一个指向新元素的迭代器
        - `c.insert(b, e)`，`c.insert({a, b, c...})`：返回`void`。插入区间或初始化列表中的所有元素，对于非`multi`容器，键不能重复
        - `c.insert(p, v)`，`c.emplace(p, args)`：迭代器`p`用于 *提示* *开始搜索新元素应该存储的位置* ，返回指向具有给定键的元素的迭代器
    ```
    // ways to add word to word_count
    word_count.insert({word, 1});
    word_count.insert(std::make_pair(word, 1));
    word_count.insert(std::pair<std::string, size_t>(word, 1));
    word_count.insert(std::map<std::string, size_t>::value_type(word, 1));
    
    word_count.emplace(word, 1);
    ```
    - 检测`insert`的返回值
    ```
    std::map<std::string, size_t> word_count;
    std::string word;
    
    while (std::cin >> word) 
    {
        std::pair<std::map<std::string, size_t>::iterator, bool> ret = word_count.insert({word, 1});
        
        if (!ret.second)
        {
            ++ret.first->second;
        }
    }
    ```
    - 向`std::multi_set`或`std::multi_map`添加元素
    ```
    std::multimap<std::string, std::string> authors;
    // adds the first element with the key Barth, John
    authors.insert({"Barth, John", "Sot-Weed Factor"});
    // ok: adds the second element with the key Barth, John
    authors.insert({"Barth, John", "Lost in the Funhouse"});
    ```
- 删除元素
    - `c.erase(k)`：从`c`中删除 *每个* 键为`k`的元素，返回`size_type`值，代表删除的元素的数量
    - `c.erase(p)`：从`c`中删除迭代器`p`指向的元素。`p`必须指向 *`c`中真实存在* 的元素，且**不能**等于`c.end()`。返回指向`p`元素之后一个元素的迭代器；若`p`指向`c`中的尾元素，则返回`c.end()`
    - `c.erase(b, e)`：删除区间`[b, e)`内的元素，返回`e`
    ```
    // erase on a key returns the number of elements removed
    if (word_count.erase(removal_word))
        std::cout << "ok: " << removal_word << " removed" << std::endl;
    else 
        std::cout << "oops: " << removal_word << " not found!" << std::endl;
    
    // for std::multi_map
    std::multimap<std::string, std::string> authors;
    authors.insert({"Barth, John", "Sot-Weed Factor"});
    authors.insert({"Barth, John", "Lost in the Funhouse"});
    size_t cnt = authors.erase("Barth, John");  // cnt == 2
    ```
- `std::map`的 *下标* 操作
    - `std::map`和`std::multi_map`提供了 *下标运算符* 和对应的`c.at()`函数
        - `c[k]`：返回键为`k`的元素的 *引用* ；如果`k` *不在`c`中* ，则 *添加* 一个键为`k`的元素，并值初始化之
            - 对`std::map`使用下标操作的行为和对数组或者`std::vector`使用时很不相同，使用一个 *不在容器中的键* 作为下标将会 *添加* 一个具有此键的元素到容器中
            - 如果不希望添加元素，则应该使用`c.find(k)`
        - `c.at(k)`：返回键为`k`的元素的 *引用* ；如果`k` *不在`c`中* ，则抛出`out_of_range`异常
    ```
    std::map <std::string, size_t> word_count;  // empty map
    word_count["Anna"] = 1;                     // insert a value-initialized element with key "Anna"
                                                // then assign 1 to its value
    ```
    - 使用下标操作的返回值
        - 对于`std::map`，下标返回`mapped_type`类型，而解引用迭代器返回`value_type aka std::pair<const key_type, mapped_type>`
        - 返回的是 *左值* ，可以读写
    ```
    std::cout << word_count["Anna"];            // fetches the element indexed by Anna; prints 1
    ++word_count["Anna"];                       // fetches the element and add 1 to it
    std::cout << word_count["Anna"];            9// fetches the element and print it; prints 2
    ```
- 查找元素
    - `c.find(k)`：返回指向 *第一个* 键为`k`的元素的迭代器，如不存在则返回尾后迭代器
    - `c.count(k)`：返回键为`k`的元素的数量。对于非`multi`容器，返回值永远是`0`或`1`
    - `c.lower_bound(k)`：返回指向 *第一个* 键 *大于等于* `k`的元素的迭代器
    - `c.upper_bound(k)`：返回指向 *第一个* 键 *大于* `k`的元素的迭代器
    - `c.equal_range(k)`：返回`std::pair<iterator, iterator>(first, last)`，区间`[first, last)`内元素键全部等于`k`

#### 无序关联容器        

- 无序容器提供与有序容器相同的操作
- 无序关联容器在存储上组织为一组 *桶* （bucket），每个桶保存零或多个元素
    - 无序关联容器用 *散列函数* （hash function）将元素映射到桶
    - 访问时，先计算键的散列值，去对应的桶查找元素
    - 性能依赖于散列函数的质量，以及桶的数量和大小
    - *迭代时顺序是乱的*
- 桶管理
    - 桶接口
        - `c.bucket_count()`：正在使用的桶的数目
        - `c.max_bucket_count()`：容器能容纳的最多的桶的数量
        - `c.bucket_size(n)`：第`n`个桶中元素数量
        - `c.bucket(k)`：键为`k`的元素在哪个桶中
    - 桶迭代
        - `local_iterator`：用于访问桶中元素的迭代器类型
        - `const_local_iterator`：用于访问桶中元素的常迭代器类型
        - `c.begin(n)`，`c.end(n)`：桶`n`的首元素迭代器和尾后迭代器
        - `c.cbegin(n)`，`c.cend(n)`：桶`n`的首元素常迭代器和尾后常迭代器
    - 散列策略
        - `c.load_factor()`：每个桶的平均元素数量，返回`float`
        - `c.max_load_factor()`：`c`试图维护的平均桶大小，返回`float`值。`c`会在需要的时候添加新的桶，以使得`load_factor <= max_load_factor`
        - `c.rehash(n)`：重新散列，使得`bucker_count >= n`且`bucket_count > size / max_load_factor`
        - `c.reserve(n)`：重组存储，在 *不重新散列* 的情况下使得`c`可以保存`n`个元素
- 无序关联容器对键类型的要求
    - 默认情况下，无序关联容器使用键类型的`==`运算符来比较元素
    - 它还使用`hash<key_type>`类型的对象来生成每个元素的散列值
    - 标准库为 *内置类型* （包括 *指针* ）提供了 *`hash`模板* ，还为包括和 *智能指针* 定义了`hash`
    - 因此，可以直接定义键是 *内置类型* （包括 *指针* ）、`std::string`和 *智能指针* 类型的无序关联容器
    - 但**不能直接定义**键是 *自定义类型* 的无序关联容器
        - 必须人工提供 *`hash`模板* => 16.5
    - 还可以人工提供充当`==`运算符的函数以及散列函数
        - 如果类已经有`==`运算符了，则只用重载散列函数
    ```
    size_t hasher(const Sales_data & sd)
    {
        return hash<string>()(sd.isbn());
    }
    
    bool eqOp(const Sales_data & lhs, const Sales_data & rhs)
    {
        return lhs.isbn() == rhs.isbn();
    }
    
    using SD_multiset = std::unordered_multiset<Sales_data, decltype(hasher)*, decltype(eqOp)*>;

    // arguments are the bucket size and pointers to the hash function and equality operator
    SD_multiset bookstore(42, hasher, eqOp);
    
    // use FooHash to generate the hash code; Foo must have an == operator
    std::unordered_set<Foo, decltype(FooHash)*> fooSet(10, FooHash);
    ```






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
    - 使用`new`直接管理内存，初始化可以选择
        - *默认初始化* 
            - *不提供* 初始化器 
            - 对象的值 *未定义* 
        ```
        int * pi = new int;
        std::string * ps = new std::string;
        ```
        - *直接初始化* 
            - 提供 *非空* 的初始化器 
            - 显式指定对象初值，可以使用 *括号* 或 *花括号* 初始化器
        ```
        int * pi = new int(1024);
        std::string * ps = new std::string(10, '9');
        std::vector<int> * pv = new std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        ```
        - *值初始化* 
            - 提供 *空的* 初始化器 
            - 如类类型没有合成的默认构造函数，则值初始化进行的也是默认初始化，没有意义
            - 对于内置类型，值初始化的效果则是 *零初始化* 
        ```
        std::string * ps1 = new std::string;   // default initialized to the empty string
        std::string * ps = new std::string();  // value initialized to the empty string
        int * pi1 = new int;                   // default initialized; *pi1 is undefined
        int * pi2 = new int();                 // value initialized to 0; *pi2 is 0
        ```
    - 使用`auto`
        - 需提供 *初始化器* ，且初始化器中 *只能有一个值* 
            - 编译器需要从初始化器中推断类型
    ```
    auto p1 = new auto(obj);      // p points to an object of the type of obj
                                  // that object is initialized from obj
    auto p2 = new auto{a, b, c};  // error: must use parentheses for the initializer
    ```
    - 动态分配`const`对象
        - 用`new`分配`const`对象是合法的，返回指向`const`的指针
        - 类似于其他`const`对象，动态分配的`const`对象亦必须进行初始化
            - 对于有 *默认构造函数* 的类类型，可以默认初始化
            - 否则，必须直接初始化
    ```
    // allocate and direct-initialize a const int
    const int * pci = new const int(1024);

    // allocate a default-initialized const empty string
    const std::string * pcs = new const std::string;
    ```
    - 内存耗尽
        - 无内存可用时，`new`会抛出`std::bad_alloc`异常，返回 *空指针*
        - 可以使用 *定位`new`* 表达式`new (std::nothrow)`（placement new）阻止抛出异常 => 19.1.2
    ```
    // if allocation fails, new returns a null pointer
    int * p1 = new int;            // if allocation fails, new throws std::bad_alloc
    int * p2 = new (nothrow) int;  // if allocation fails, new returns a null pointer
    ```
    - 动态释放内存
        - `delete`表达式
            - 传递给`delete`的指针必须是 *指向被动态分配的对象* 的指针或者 *空指针* 
            - 将同一个对象反复释放多次是 *未定义行为*
            - *`const`对象* 虽然不能更改，但却 *可以销毁* 
            - `delete`之后指针成为了 *空悬指针* （dangling pointer）
                - *你就是一个没有对象的野指针*
        ```
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
    ```
    std::shared_ptr<std::string> p1;
    if (p1 && p1->empty()) *p1 = "hi";
    ```
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
        - `p.unique()`：`return p.use_count() = 1;`
        - `p.use_count()`：返回`p`的 *引用计数* （与`p`共享对象的智能指针的数量）。 *可能很慢，主要用于调试* 
        - `p.reset()`：若`p`是唯一指向其对象的`std::shared_ptr`，则释放此对象，将`p` *置空*
        - `p.reset(q)`：若`p`是唯一指向其对象的`std::shared_ptr`，则释放此对象，令`p` *指向内置指针* `q`
        - `p.reset(q, d)`：若`p`是唯一指向其对象的`std::shared_ptr`，则 *调用`d`* 释放此对象，将`p` *置空*
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
    - `std::unique_ptr`独有的操作
        - `std::unique_ptr<T> u1`：定义一个 *空的* `std::unique_ptr<T>`，使用默认删除器`delete`
        - `std::unique_ptr<T, D> u2`：定义一个 *空的* `std::unique_ptr<T, D>`，`D` *删除器* 的类型
        - `std::unique_ptr<T, D> u(d)`：定义一个 *空的* `std::unique_ptr<T, D>`，`D` *删除器* 的类型，`d`为指定的 *删除器* 
        - `u = nullptr`：释放`u`指向的对象，将`u` *置空* 
        - `u.release()`：`u` *放弃* 对指针的控制权，返回内置指针，并将`u` *置空* 
        - `u.reset()`：释放指向`u`的对象，将`u` *置空*
        - `u.reset(q)`：释放指向`u`的对象，令`u` *指向内置指针* `q`
        - `u.resert(nullptr)`：释放指向`u`的对象，将`u` *置空*
- `std::shared_ptr`
    - `std::make_shared`函数
        - 最安全的分配和使用动态内存的方法
        - 在动态内存中分配一个对象并 *用其参数构造对象* ，返回指向该对象的`shared_ptr`
            - 就类似与顺序容器的`c.emplace(args)`
            - 不提供任何参数就是 *值初始化* 对象
        ```
        std::shared_ptr<int>         p3 = std::make_shared<int>(42);                     // int 42
        std::shared_ptr<std::string> p4 = std::make_shared<std::string>(10, '9');        // std::string "9999999999"
        std::shared_ptr<int>         p5 = std::make_shared<int>();                       // int 0 (value initialized)
        auto                         p6 = std::make_shared<std::vector<std::string>>();  // 空 std::vector<std::string>
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
                                             
        auto r = std:: make_shared<int>(42);  // int to which r points has one user assign to r, 
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
    - 同一时刻只能有一个`std::unique_ptr`指向一个给定对象
    - 
- `std::weak_ptr`

#### 动态数组（Dynamic arrays）

- `new`和数组
- `allocator`类







### 🌱 [Chap 13] 拷贝控制

- 






### 🌱 [Chap 14] 操作重载与类型转换

- 






### 🌱 [Chap 15] OOP

- 






### 🌱 [Chap 16] 模板与泛型编程

- 






### 🌱 [Chap 17] 标准库特殊设施

- 






### 🌱 [Chap 18] 用于大型工程的工具

#### 异常处理

- `C++`标准异常
    - `<exception>`
        - `std::exception`：只报告异常的发生，不提供任何额外信息。 *只能* *默认初始化* ，**不能**传参
    - `<stdexcept>`
        - `std::runtime_error`：所有运行错误
            - `std::range_error`：运行错误，生成的结果超出了有意义的值域范围
            - `std::overflow_error`：运行错误，计算溢出
            - `std::underflow_error`：运行错误，计算溢出
        - `std::logic_error`：所有逻辑错误
            - `std::domain_error`：逻辑错误，参数对应的结果值不存在
            - `std::invalid_argument`：逻辑错误，无效参数
            - `std::length_error`：逻辑错误，试图创建一个超出该类型最大长度的对象
            - `std::out_of_range`：逻辑错误，使用了一个超出有效范围的值
    - `<new>`
        - `std::bad_alloc`异常类。 *只能* *默认初始化* ，**不能**传参 => 12.1.2
    - `<typeinfo>`
        - `std::bad_cast`异常类 => 19.2
- 以上异常除特别说明的，都 *必须* 传参（`C`风格字符串）
- 异常类型之定义了一个名为`what`的成员函数，返回`C`风格字符串`const char *`，提供异常的文本信息。
  如果此异常传入了初始参数，则返回之；否则返回值由编译器决定。






### 🌱 [Chap 19] 特殊工具与技术

- 












    
















