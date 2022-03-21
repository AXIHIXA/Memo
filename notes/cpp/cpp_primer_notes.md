# *C++ Primer* Notes


### 🌱 一句话

- 常见规则
    - 先声明（或定义）再使用。 *第一次实际使用前* 再声明（定义）
    - **严禁**混用有符号类型和无符号类型（比如：该用`size_t`就用，别啥玩意都整成`int`）
    - 下标遍历型`for`循环的唯一指定写法，例如`for (std::string::size_type i = 0; i != 10; ++i) {}`， *括号内声明* `i`（避免冲突）、`std::string::size_type`（精确匹配类型，别啥玩意都整成`int`），`!=`（为了性能和一般至少都会实现`==`和`!=`，而`<`则不一定了）和`++i`（为了性能，严禁乱用`i++`）这四条你要细品
    - 范围遍历型`for`循环的科学写法：`for (const auto & item : iteratable) {}`。`auto`可以帮助减少非必要的拷贝（例如`std::map<K, V>`只读范围遍历时的元素类型应该是`const std::pair<const K, V> &`，如果用`const std::pair<K, V> &`遍历则实际上会进行隐式类型转换和拷贝，性能一样不好）
    - 整数和浮点数字面值的 *后缀* 一律使用 *大写* 版本，避免`l`和`1`混淆
    - 函数的返回值类型**永远不要**设置成引用，左值右值都不行，统统是悬垂的
    - 如果函数有可能用到某个全局变量，则**不宜**再定义同名的局部变量
    - `const`常量不论是声明还是定义都添加`extern`修饰符
    - 想要`auto`推导出引用或者常量的话，直接写清楚是坠吼的（`const auto & a = b`），别折腾顶层`const`什么的
    - 认定应为常量表达式的变量应当声明为`constexpr`类型
    - 凡是不修改类数据成员的成员函数函数一律定义成常成员函数
    - `constexpr`函数、静态`constexpr`成员、`inline`函数（包括类的`inline`成员函数）以及 *模板* （ *函数* 和 *类* ）的**定义和实现都应**写进头文件
    - `using`声明（`using std::string`、`using namespace std`、`using ns_name = long_namespace`、`using intptr = int *`等）**不应**写进头文件
    - `for each`循环内以及使用迭代器时**不能**改变被遍历的容器的大小
    - 现代`C++`应使用标准库类型配合迭代器，而**不是**`C`风格的数组和指针。指针也是一种迭代器
    - 现代`C++`**不应**使用旧式的强制类型转换，应当明确调用对应的`xx_cast<T>(expr)`
    - 除非必须，**不要**使用自增自减运算符的后置版本（会造成性能浪费）
    - **不在**内部作用域声明函数（内部作用域的名字会覆盖外部作用域的同名实体，名字查找先于类型匹配，可能会影响函数重载的使用）
    - *交互式* 系统通常应该 *关联输入流和输出流* ，这意味着所有输出，包括用户提示信息，都会在读操作之前被打印出来
    - 在对诸如`std::string`、`std::vector`等`C++`容器进行 *索引* 操作时，正确的类型是该容器的静态类型成员`std::vector::size_type`，而该类型通常是`size_t`的别名
    - 改变容器 *大小* 之后，则 *所有* 指向此容器的迭代器、引用和指针都 *可能* 失效，所以 *一律更新* 一波才是坠吼的
    - 永远**不要缓存**尾后迭代器（这玩意常年变来变去），现用现制，用后即弃
    - 不需要写访问时，应当使用`const_iterator`
    - 调用泛型算法时，在不需要使用返回的迭代器修改容器的情况下，传参应为`const_iterator`
    - 通常**不**对关联容器使用泛型算法（或不能用或性能很差）
    - **应该**使用`auto`接收`lambda`表达式，乱用`std::function`或 *函数指针* 接收`lambda`表达式会导致类型转换和其他严重的性能损失！
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
    - 大多数应用都应该使用 *标准库容器* 而**不是**动态分配的数组。使用容器更为简单，更不容易出现内存管理错误，并且可能有更好的性能
    - *接受右值引用形参的函数模板* 使用守则
        - 通常应用于如下场景
            1. *转发* （forwarding） => 16.2.7
            2. *模板重载* （template overloading）=> 16.3
        - 通常使用如下方式重载 => 13.6.3
        ```
        template <typename T> void f(T &&);       // binds to nonconst rvalues
        template <typename T> void f(const T &);  // lvalues and const rvalues
        ```
    - 将 *左值* `static_cast`成 *右值引用* 是允许的，但实际使用时应当使用封装好的`std::move`而**不是**`static_cast`
    - 在定义任何函数之前，记得 *声明所有重载的函数版本* ，这样就不必担心编译器由于未遇到希望调用的版本而实例化并非所需的函数模板
    - 一个特定文件所需要的 *所有模板声明* 通常 *一起放置在文件开始* 位置，出现于任何使用这些模板的代码之前 => 16.3
    - *模板及其特例化版本* 应该 *声明在同一个头文件* 中，所有同名模板的声明应该放在前面，然后是这些模板的特例化版本
    - `C++`程序**不应**使用`C`库函数`rand`，而应使用`std::default_random_engine`和恰当的分布类对象
    - 析构函数**不应**抛出不能被它自己处理的异常，即：如果析构函数将要执行某个可能抛出异常的操作，则该操作应该被放置在一个`try`块内，并在析构函数内部得到处理
    - `throw` *指向局部对象的指针* 是**错误**的，因为执行到`catch`之前局部对象就已经被销毁了
    - `throw` *解引用多态指针* 也是**错误**的。解引用指向派生类对象的基类指针会导致被抛出对象 *被截断* 
    - `throw`指针要求在任何对应的`catch`子句所在的地方，指针所指的对象都必须存在
    - `Clang-Tidy`要求只能`throw`在`throw`子句中临时创建的匿名`std::exception`类及其派生类对象
    - 通常情况下，如果`catch`接受的异常与某个继承体系有关，则通常将其捕获形参定义为引用类型
    - 越是专门的`catch`，就越应该置于整个`catch`列表的前端。如果在多个`catch`语句的类型之间存在着继承关系，则我们应该把继承链最底端的类（most derived type）放在前面，而将继承链最顶端的类（least derived type）放在后面。因为挑选规则是第一个能匹配的，而不是最佳匹配
    - `C++`应使用 *无名命名空间* 代替 *文件内`static`声明*
    - 应尽量**避免**使用 *`using`指示* 
    - *头文件* 最多只能在它的 *函数或命名空间内* 使用 *`using`指示* 或 *`using`声明* 
- 跟类有关的一箩筐规则
    - 构造函数**不应**该覆盖掉 *类内初始值* ，除非新值与原值不同；不使用 *类内初始值* 时，则每个构造函数**都应显式初始化**每一个类内成员。此外，构造函数如果有默认实参，则此默认实参的值**不应**与对应成员的类内初始值相左
    - `Clang-Tidy`直接规定只有一个实参的构造函数必须是`explicit`的
    - `Clang-Tidy`规定构造函数的非平凡形参应该是 *传值加`std::move`* ，而**不是** *传常引用* 
        - 对平凡形参，直接传值，搞引用反而麻烦
    - *构造函数初始化器列表* 的顺序**必须**按照类成员被声明的顺序。 *构造函数初始化器列表* 中执行初始化的 *顺序是按照类成员被声明的顺序* ，与其在列表中的顺序**无关**
    - 希望类的所有成员都是`public`时，**应**使用`struct`；只有希望使用`private`成员时才用`class`
    - `class`中的 *私有成员* 之前**应该**显式写出`private`，仅依靠默认会混淆继承关系，容易产生误会
    - 在类定义开始或结束的地方**集中声明**友元；使用友元，仍另需有一个**单独的函数声明**
    - 类的类型成员（`typedef`以及`using`声明）应该放在类定义**刚开始**的地方的`public`区域
    - 最好令构造函数初始化列表的顺序与成员声明的顺序**保持一致**；**避免**用某些成员初始化其他成员，用构造函数的参数作为初始值
    - 应把 *静态数据成员* 的定义与其他 *非内联函数* 的定义放在类外、和类的定义**同一个头文件**中
    - 即使一个`constexpr`静态成员在类内部被初始化了，也应该在类外定义一下该成员（此时**不能**再指定初始值）
    - *拷贝赋值运算符* 的要求
        - 赋值运算符应该返回一个指向其左侧运算对象的 *引用* 
        - 必须正确处理 *自赋值* （ *拷贝并交换赋值运算符* 则自动能处理自赋值）
        - 大多数拷贝赋值运算符组合了 *析构函数* 和 *拷贝构造函数* 二者的工作
            - 公共的工作应放到 *私有的工具函数* 中完成
    - 千言万语汇聚成一句话， *三五法则* ，五个拷贝控制成员要定义就 *都定义全* ，就没这么多破事儿了
    - 还有一句： *拷贝并交换赋值运算符* 好哇，天生异常安全、不怕自赋值，还同时能充当拷贝和移动两种运算符
    - 一般情况下**不应该**重载、 *逻辑与* 、 *逻辑或* 、 *逗号* 和 *取地址* 运算符
    - `operator->()` 一般**不执行任何操作**，而是调用`operator*()`并返回其结果的 *地址* （即返回 *类的指针* ）
    - `operator bool()`一般定义成`explicit`的
    - 基类除了只需要虚析构函数时以外，亦**必须遵守** *三五法则* 
    - 派生类覆盖基类虚函数时必须保证函数签名与基类版本完全一致、**必须**显式加上`override`或`final`（此时不必再加`virtual`）
    - 如果虚函数使用 *默认实参* ，**必须**和基类中的定义一致
    - 除了继承来的 *虚函数* ，派生类**不应该**重用那些定义在其基类中的名字
    - 基类的拷贝控制成员中， *析构函数* **必须**是虚函数（即使这个函数不执行任何操作也是如此）；除 *析构函数之外* ，均**不应**定义为虚函数
        - `delete`非虚析构函数基类的指针是 *未定义行为*
        - 解引用基类指针并赋值是**不好**的
    - 派生类 *拷贝控制成员* 应首先 *首先调用基类对应成员* 处理基类部分，再处理自己的部分
        - 派生类构造函数应 *首先调用基类构造函数* 初始化 *基类部分* ， *之后* 再按照 *声明的顺序* 依次初始化 *派生类成员* ， *析构函数* **除外**：顺序和构造是反的，且编译器会自动调用基类析构函数
        - 派生类拷贝或移动构造函数**必须**显式调用基类对应构造函数，否则基类部分将被 *默认初始化* ，产生 *未定义值* 
    - 位于继承体系中间层级的类 *可以* 选择 *虚继承* ，必须在虚派生的真实 *需求出现前* 就已经 *完成虚派生* 的操作
    - 虽然虚基类的初始化只由最低层派生类独自负责，但 *每个虚派生类* 仍旧都 *必须在构造函数中初始化它的虚基类* 
- 一些小知识
    - 类成员函数的各色说明符限定符相对顺序完整版
    ```
    struct Base                  { virtual auto fun(int) const && noexcept -> void          = 0; }
    struct Derived : public Base {         auto fun(int) const &&          -> void override {}   }
    ```
    - 概念辨析一
        - *多态* （polymorphism）指类定义了 *虚函数* （virtual function）
        - *抽象基类* （abstract base class）指定义了 *纯虚函数* （pure virtual function）的基类
        - *虚基类* （virtual base class）指被 *虚继承* （virtual inheritance）的基类
    - 概念辨析二
        - `int && a = 1;`
        - *类型* （type）是对象的值的性质，例如上例说的就是`int &&`
        - *值类别* （value category）说的是`a`本身，任何一个对象都一定是 *左值* `lvalue`、 *纯右值* `prvalue`或 *将亡值* `xvalue`三者之一
    - 给`char a`和`unsigned char b`加上 *加号* `+a`，`+b`就把它们提升成了`int`和`unsigned int`，可以用于`std::cout`
    - 如果两个字符串字面值位置紧邻且仅由 *空格* 、 *缩进* 以及 *换行符* 分隔，则它们是 *一个整体* 
    - `C++11`规定整数除法商一律向0取整（即：**直接切除小数部分**）
    - 指针解引用的结果是其指向对象的**左值**引用
    - *悬垂引用* （Dangling reference）就是 *悬垂指针* （Dangling pointer， *野指针* ）的引用版
    - 程序 *良构* （well-formed）指的就是程序能过编译， *非良构* （ill-formed）就是过不了编译的意思
    - `*iter++`等价于`*(iter++)` => 优先级：`++` > `*`
    - `p->ele`等价于`(*p).ele` => 优先级：`.` < `*`
    - 非`extern`全局`const`对象和`static`对象被设定为**仅在文件内有效**
    - `std::endl`有刷新缓冲区的效果。最好带上
    - 如果一个函数是永远也不会用到的，那么它可以只有声明而没有定义 => 15.3
    - 如果一个函数形参是没有用到的，那么在函数定义中也不必为之具名 => 14.6
    - 像在函数形参列表中一样，如果`catch`无需访问抛出的表达式的话，则我们可以忽略捕获形参的名字 => 18.1.2
    - 引用从来都是作为被引用对象的同义词出现（比如`auto`就不能自动推断出引用），唯一例外是`decltype`。它会原样保留引用以及顶层`const`
    - `main`函数不能递归调用、不能重载
    - *名字查找* 先于 *类型匹配* ，因此不同的作用域中**无法**重载函数
    - 定义在类内部的函数是隐式的`inline`函数
    - 使用`struct`或`class`定义类的**唯一区别**就是默认访问权限
        - `struct`成员默认 *公有* ，继承时默认 *公有继承*
        - `class`成员默认 *私有* ，继承时默认 *私有继承*
    - 每个类定义了**唯一**的类型；两个类即使内容完全一样，它们也是不同的类型，**不能**自动相互转化
    - 如果一个构造函数为每一个参数都提供了默认实参，则它实际上也定义了默认构造函数
    - 能通过一个实参调用的构造函数定义了一条从构造函数的参数类型向类类型隐式转换的规则
    - *非`constexpr`静态成员* 只能在**类外**初始化
    - *类静态成员* *只能在**类外*** 定义、定义时**不能**重复`static`关键字
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
    - 重载的运算符如果是 *成员函数* ，则第一个（左侧）运算对象绑定到隐式的`this`指针上，只需指定右侧运算符（如有）。成员运算符函数的（显式）参数数量比运算符的运算对象总数 *少一个* 
    - 重载的运算符要么是 *类成员* ，要么含有 *至少一个类类型参数*
    - 重载的 *箭头* 运算符必须返回 *类的指针* 或者 *自定义了箭头运算符的某个类的对象* 
    - 表达式被用作 *条件* 时， *类型转换运算符* 即使是`explicit`的，仍会被 *隐式应用* 
    - 对于 *用户代码* 中某个节点来说，当且仅当 *基类公有成员可访问* 时， *派生类向基类的类型转换可用* 
    - 友元**不能**传递、**不能**继承
    - 默认情况下，`C++`语言假定通过作用域运算符访问的名字**不是**类型。希望使用模板类型参数的 *类型成员* 时，必须 *显示指明`typename`*
    - 引用类或函数模板的 *一个特定实例* 之前 *必须前向声明模板自身* ；如果引用的是 *全部实例* ，则 *不需前向声明* 
    - 无论何时使用一个类模板，都必须在模板名后面接上尖括号`<>`。对于全默认实参的类模板，也要带一个空尖括号
    - 函数模板的匹配规则一句话：形参匹配，特例化（非模板才是最特例化的），完犊子
    - 所有标准库类型都能确保它们的析构函数**不会**引发异常
    - 当且仅当`e`是 *多态类类型的引用左值或解引用指针* 时，`typeid(e)`的 *动态类型* ；否则，返回 *静态类型* 
- 读代码标准操作
    - 判断复杂类型`auto`变量的类型：先扒掉引用，再扒掉被引用者的顶层`const`
    - 如何理解`C`声明
        - 参考了`Expert C Programming - Deep C Secrets`一书`pp.76`的神图[`Magic Decoder Ring for C Declarations`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/fig_3_3.pdf)
        - `C`声明遵循以下规则
            - 优先级从高到低
                1. 用于 *分组* 的括号（Parentheses grouping together a part of the declaration）
                2. 后缀操作符（`()`表示函数，`[]`表示数组）
                3. 前缀操作符（`*`表示指针）
            - `cv`限定如出现于`*`之前，则作用于指向的类型；如出现于`*`之后，则作用于指针本身
        - 如何理解复杂声明
            1. 从 *最左侧* 的 *标识符* （名字）`p`开始，说`declare p as...`，之后按照如上优先级解读名字周边的内容
            2. 如果`p`右边是`[n]`，则说`array n of...`
            3. 如果`p`右边是表示函数的括号`(param_list)`（例如`()`，`(float, int)`），则说`function (param_list) returning...`
            4. 如果`p`左边是`*`（可能还有`cv`限定），则说`xx pointer to...`（例如`const int * const`说成`const pointer to const int`）
            5. 跳出这一层 *分组* 括号（如有），重复`(b) - (e)`
        - 举例：`int (*(*pf)(int, int (*(*)(int))[20]))[10]`：
            - 按顺序翻译为
            ```
            declare pf as pointer to function (int, pointer to function (int) returning pointer to array 20 of int) 
                                     returning pointer to array 10 of int
            ```
        - 大宝贝：[cdecl](https://cdecl.org/) 
            - 自动帮你干这些破事儿
            - `ubuntu`下一键安装：`sudo apt install cdecl`







### 🌱 字面量（literal）

- *整数* 
    - 可以写作十进制、八进制（以`0`开头）、十六进制（以`0x`或`0X`开头）`或二进制 (since C++14)`形式
    ```
    20              // dec，int
    -42             // dec，int
    42ULL           // dec，unsigned long long
    024             // oct，int
    0x1a            // hex，int
    0X1A            // hex，int
    0b11            // bin, int
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
    - *原始字符串字面量* （raw string literal）
        - 其中所有字符均被理解为 *未转义字符* （unescaped characters）
            - 格式
            ```
            prefix(optional) R"delimiter( raw_characters )delimiter" (since C++11)
            ```
            - 由如下部分组成
                - `prefix`为前面四格可选前缀
                - `R"`：必须原样保留
                - `delimiter`为一对分隔符，是**除**括号、反斜杠和空格**以外**的任何源字符所构成的字符序列（可为空，长度至多`16`个字符） 
                - `(`：必须原样保留
                - 字符串，其字符一律不转义
                - `)"`：必须原样保留
        - 举例
        ```
        // A Normal string 
        std::cout << "Geeks.\nFor.\nGeeks.\n";  
        
        // OUTPUT: 
        Geeks.
        For.
        Geeks.

        // A Raw string 
        std::cout << R"(Geeks.\nFor.\nGeeks.\n)" << std::endl;  
        
        // OUTPUT:
        // Geeks.\nFor.\nGeeks.\n
        ```
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
- *`std::chrono::duration`字面量* `(since C++14)`
    - => 17






### 🌱 [声明和定义](https://en.cppreference.com/w/cpp/language/definition)（Declarations and definitions）

- *定义* 就是完整定义了被引入的实体的 *声明* 
- **除以下情况以外**，所有的 *声明* 都是定义
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
        - 因为这是 *声明* ，类内又只能定义非静态数据成员，所以必须在类外单独再定义一次
    ```
    struct S 
    {
        int n;                        // 定义 S::n
        static int i;                 // 声明但不定义 S::i
        inline static int x;          // 定义 S::x
    };                                // 定义 S
    
    int S::i;                         // 定义 S::i
    ```
    - 已经在类中用`constexpr`说明符定义过的 *静态数据成员* ，在 *命名空间作用域* 中的声明 `(deprecated since C++17)` 
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
    template <class T>                // 模板形参列表中 typename 和 class 等价
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
    - 推导指引的声明（不定义任何实体） `(since C++17)` 
    - `static_assert`声明（不定义任何实体）
    - 特性声明（不定义任何实体） 
    - 空声明（不定义任何实体）
    - `using`指令（不定义任何实体） 
    - 模板的显式实例化声明（ *`extern`模板* ） 
    ```
    extern template f<int, char>;     // 声明但不定义 f<int, char>
    ```
    - 不是定义的模板特例化声明 
    ```
    template <> 
    struct A<int>;                    // 声明但不定义 A<int>
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

所有变量都具有如下四种 *链接* 之一，用于调节变量在不同 *翻译单元* （translation unit）之间的可见性：

A <u>_translation unit_</u> is the source code giving rise to a single object file, say, `foo.o`. 
It’s basically a single source file, plus all of its `#include` files. 

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
        - 为什么`const`需要默认内部链接
            - 编译器在编译过程中会把所有的`const`变量都替换成相应的字面值
                - 这一步骤实际上是 [*默认初始化*](https://en.cppreference.com/w/cpp/language/initialization) 过程中的 [*常量初始化*](https://en.cppreference.com/w/cpp/language/constant_initialization)（Constant initialization））
            - 为了执行上述替换，编译器必须知道变量的初始值
                - 如果程序包含多个文件，则每个用了`const`对象的文件都必须得能访问到它的初始值才行
                - 要做到这一点，就必须在每一个用到变量的文件之中都有它的定义
            - 在此基础上，为了避免出现对同一变量的重复定义，`const`对象被设定为仅在文件内有效
                - 当多个文件中出现了同名的`const`变量时，其实等同于在不同文件中分别定义了**独立的**变量
                - 如果希望`const`对象只在一个文件中定义一次，而在多个文件中声明并使用它，则需采用上述操作。
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
    int a;             // 这其实是声明并定义了变量 a
    extern int a;      // 这才是仅仅声明而不定义
    extern int a = 1;  // 这是声明并定义了变量 a 并初始化为 1
                       // 任何包含显式初始化的声明即成为定义，如有 extern 则其作用会被抵消
    ```
4. *模块链接* （Module linkage） `(since C++20)`
    - 名字 *只能* 从 *同一模块单元* 或 *同一具名模块中的其他翻译单元* 的作用域指代
    - 声明于 *命名空间作用域* 中的 *具名模块* 且 *不被导出* ，且无内部链接，则该名字拥有 *模块链接*

#### 从作用域和存储期看变量

- *全局非静态变量* 
    - 包括
        - 定义于所有函数之外的非静态变量
    - 具有 *（全局）命名空间作用域* 
    - 默认具有 *内部链接* ，但可以显式添加 *外部链接* 从而实现跨文件
        - 只需在一个源文件中定义，就可以作用于所有的源文件
        - 当其他不包含全局变量定义的源文件中，使用前需用`extern`再次声明这个全局变量
    - 存储于 *静态存储区*
- *全局静态变量* 
    - 包括
        - 定义于所有函数之外的 *静态变量* 
        - 定义于所有函数之外的 *非`extern`常量* 
    - 具有 *（全局）命名空间作用域* 
    - 具有 *内部链接* （因为`static`和`extern`是冲突的，所以这些肯定是定死了内部链接、这辈子是跨不了文件了）
        - 如果程序包含多个文件，则 *仅作用于定义它的文件* ，**不能**作用于其他文件
        - 天坑：如果两个不同的源文件都定义了相同名字的 *全局静态变量* 或者 *全局非外连常量*， 那么它们是**不同的变量**   
    - 存储于 *静态存储区* 
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
    - 具有 *块作用域* ，但存储于 *静态存储区* 
        - 是 *局部静态对象* （local static object）
            - 在程序的控制流第一次经过对象定义语句时初始化，并且直到整个程序终止时才被销毁。
            - 在此期间，对象所在函数执行完毕也不会对它有影响。
            - 如没有显式初始化，则会执行 *默认初始化* （内置类型隐式初始化为`0`）
            - `C++11`开始保证初始化是线程安全的，在一个线程开始局部静态对象的初始化后到完成初始化前，其他线程执行到这个局部静态对象的初始化语句就会等待，直到该对象初始化完成。
                - 可用于优雅地构造`Meyers`单例
                ```
                // singleton.h
                class Singleton
                {
                public:
                    Singleton(const Singleton &) = delete;
                    Singleton(Singleton &&) = delete;
                    
                    Singleton & operator=(const Singleton &) = delete;
                    Singleton & operator=(Singleton &&) = delete;
                
                    static Singleton & getInstance() noexcept
                    {
                        static Singleton instance;
                        return instance;
                    }
                
                private:
                    Singleton() = default;
                    ~Singleton() = default;
                };
                
                // blah.cpp
                Singleton & singleton {Singleton::getInstance()};
                ```
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
    - 
- *静态函数* 
    - 函数的返回值类型前加上`static`关键字
    - 只在声明它的文件当中可见，**不能**被其他文件使用






### 🌱 [作用域](https://en.cppreference.com/w/cpp/language/scope)（scope）

- `C++`程序中出现的每个名字，只在某些可能不连续的源码部分中 *有效* （即，编译器能知道这玩意儿是啥、在哪儿声明的），这些部分被称为其 *作用域* 
    - 编译器通过[名字查找](https://en.cppreference.com/w/cpp/language/lookup)（Name lookup）实现名字和声明的关联
        - 如果名字已经指明了 *命名空间* ，则进行[限定名字查找](https://en.cppreference.com/w/cpp/language/qualified_lookup)（Qualified name lookup）
        - 否则，进行[无限定名字查找](https://en.cppreference.com/w/cpp/language/unqualified_lookup)（Unqualified name lookup）
        - [实参依赖查找](https://en.cppreference.com/w/cpp/language/adl)（Argument-dependent Lookup，ADL）
            - 对于函数调用（包括对重载运算符的隐式调用），在非限定名字查找范围内的作用域和命名空间之外，还查找实参类型所在命名空间和类
                - 名字查找一旦发现如下三种情况，ADL不会发生：
                    - 类成员名；
                    - 块作用域里的函数声明（不包括using声明）；
                    - 任何不是函数或函数模板的名字（例如，函数对象，普通变量，等等）
            - ADL应用举例：
                - ADL可以找到定义在类体内的友元运算符（对于没有参数的友元函数，那是找不到的）；
                - `std::cout << '\n'`：`operator<<`并不在全局命名空间里，之所以能找到是因为`std::cout`的类型`std::ostream`在`std`里
- 根据变量的 *定义位置* 和 *生命周期* ，`C++`的变量具有不同的 *作用域* ，共分为以下几类 
    - [*块作用域*](https://en.cppreference.com/w/cpp/language/scope#Block_scope)（Block scope）
    - [*函数形参作用域*](https://en.cppreference.com/w/cpp/language/scope#Function_parameter_scope)（Function parameter scope）
    - [*函数作用域*](https://en.cppreference.com/w/cpp/language/scope#Function_scope)（Function scope）
    - [*命名空间作用域*](https://en.cppreference.com/w/cpp/language/scope#Namespace_scope)（Namespace scope）
        - 包含 *全局命名空间作用域* （Global namespace scope），即所谓的 *全局作用域* 
    - [*类作用域*](https://en.cppreference.com/w/cpp/language/scope#Class_scope)（Class scope）
    - [*枚举作用域*](https://en.cppreference.com/w/cpp/language/scope#Enumeration_scope)（Enumeration scope）
    - [*模板形参作用域*](https://en.cppreference.com/w/cpp/language/scope#Template_parameter_scope)（Template parameter scope）
- 作用域始于 *声明点* ，内部作用域（inner scope）的 *名字* 会 *覆盖* 外部作用域（outer scope）的 *同名实体* （变量、函数，等等）

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
    - E. Dijkstra: Go To Statement Considered Harmful. *Communications of the ACM* (1968) 
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
    1. 这个命名空间的 *剩余部分* 
    2. 其后所有 *同名命名空间* 
    3. 使用了 *`using`指令* 或 *`using`声明* *引入了此实体或整个这个命名空间的域* 
- *翻译单元* 的顶层作用域（即所谓的 *文件作用域* 或 *全局作用域* ）亦为命名空间，被正式称作 *全局命名空间作用域* 
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
    using namespace N;                    // i, g, j, q, inl, x, y 的作用域恢复（至全局作用域）
    i = 1;                                // N::i 在作用域中
    x = 1;                                // N::(anonymous)::x 在作用域中
    y = 1;                                // N::inl::y 在作用域中
    inl::y = 2;                           // N::inl 亦在作用域中
}                                         // i, g, j, q, inl, x, y 的作用域间断
```

#### [类作用域](https://en.cppreference.com/w/cpp/language/scope#Class_scope)（Class scope）

- 类中声明的名字的作用域开始于其声明点，并包含
    1. 类体的 *剩余部分* 
    2. 所有 *成员函数体* （无论是否定义于类定义外或在该名字的声明之前）及其 *默认实参* 、 *异常规定* 
    3. 类内 *花括号或等号初始化器* （即 *类内初始值* ）
    4. 递归地包括 *嵌套类* 中的所有这些内容
```
class X 
{
    int f(int a = n) 
    {                                     // X::n 在默认实参中在作用域
         return a * n;                    // X::n 在函数体内在作用域中
    }
    
    using r = int;
    r g();
    int i = n * 2;                        // X::n 在初始化器内在作用域中
 
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

- *限定作用域枚举* （scoped enumeration，`enum class T`或`enum struct T`）中引入的 *枚举成员* （enumerator）的名字的作用域开始于其声明点，并 *终止于* `enum` *说明符末尾* 
    - *限定作用域枚举* 使得枚举类型必须带着枚举作用域，避免混淆
- *非限定作用域枚举* （unscoped enumeration，`enum T`）中引入的 *枚举成员* 的名字的作用域在`enum` *说明符结尾后仍在作用域中* 
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

#### [限定标识符](https://en.cppreference.com/w/cpp/language/identifiers#Qualified_identifiers)（Qualified identifiers）

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
    3. [*拷贝初始化*](https://en.cppreference.com/w/cpp/language/copy_initialization)，例如`std::string s = "hello";`
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
    2. 若`T`是**没有** *合成的默认构造函数* 的类类型，则 *默认初始化* 对象
    3. 若`T`是拥有 *合成默认构造函数* 的类类型
        1. *零初始化* 对象的数据成员
        2. 然后，若数据成员拥有 *非合成的默认构造函数* ，则在上一条的基础上再执行一次 *默认初始化* 
    4. 若`T`是 *数组类型* ，则 *值初始化* 数组的 *每个元素* 
    5. 否则， *零初始化* 对象
- 注意事项
    - *合成的默认构造函数* 指 *隐式定义* （用户**未定义**）或 *显式合成* （在首个用户定义处`= default;`）的 ***没有**形参* 的构造函数
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
T object{arg};                                                (2)  // 以花括号初始化器初始化
T(other)
T(arg1, arg2, ...)                                            (3)  // 用函数式转型或以带括号的表达式列表初始化  
static_cast< T >( other )                                     (4)     
new T(args, ...)                                              (5)     
Class::Class() : member(args, ...) { ... }                    (6)     
[arg](){ ... }                                                (7)  // lambda表达式中用复制捕获的变量初始化闭包对象的成员
```

#### [拷贝初始化](https://en.cppreference.com/w/cpp/language/copy_initialization)

- 从 *另一对象* 初始化对象，但只考虑非`explicit`构造函数
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
- [临时量生存期](https://en.cppreference.com/w/cpp/language/lifetime)：
    - 一旦 *常左值引用* 或 *右值引用* 被绑定到 *临时量* 或其子对象，临时量的生存期就 *被延续* 以匹配 *该引用* 的生存期
        - 总而言之，临时量的生存期**不能**以 *进一步传递* 来延续
            - 从绑定了该临时量的引用初始化的第二引用**不影响**临时量的生存期，例如
        ```
        // 此函数将返回悬垂引用
        // 默认实参 临时量A() 绑定给了 常左值引用形参 a ，那么 A() 的生存期就到 a 生命结束，即 return a; 末尾为止
        // 至于说 a 又被 绑定给了 第二引用 常引用返回值，那就跟 A() 的生存期半毛钱关系都没有了
        const A & foo(const A & a = A()) 
        {
            return a;
        }
        ```
    - 上一条有下列 *五条* **例外**
        1. *函数返回语句* 中绑定到 *引用类型返回值* 的 *临时量* 
            - 函数中的 *任何临时量都* 只存在到返回表达式的末尾。这种函数始终返回 *悬垂引用* （Dangling reference） 
            - 函数的返回值类型**永远不要**设置成引用，左值引用，右值引用都不行，统统是悬垂的
        2. *构造函数初始化器列表* 中绑定到 *引用类型数据成员* 的 *临时量* 
            - 只存在到构造函数退出前，而非对象存在期间 `(until C++14)`
        3. *函数调用* 时绑定到 *引用类型形参* 的 *临时量*     
            - 只存在到含这次函数调用的全表达式结尾为止。若函数返回一个引用，而其生命长于全表达式，则它将成为 *悬垂引用*  
        4. *`new`表达式的初始化器* 中绑定到 *引用类型初值* 上的 *临时量* 
            - 只存在到含该`new`表达式的全表达式结尾为止，而非被初始化对象的存在期间。若被初始化对象的生命长于全表达式，则其引用成员将成为 *悬垂引用* 
        5. *聚合体直接初始化语法（括号）* 中绑定 *引用类型元素* 上的 *临时量* 
            - 只存在到含该初始化器的全表达式末尾为止 `(since C++20)`
            - *聚合体列表初始化语法（花括号）* 则 *会被延续* 
            ```
            struct A 
            {
                int && r;
            };
            
            A a1{7};   // OK：延续生存期
            A a2(7);   // 良构（程序能编译），但有悬垂引用
            ```
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
    2. 若`T`是 *类类型* ，则考虑各构造函数并实施针对空实参列表的 [*重载确定/重载决议*](https://en.cppreference.com/w/cpp/language/overload_resolution)（overload resolution）。调用所选的构造函数（默认构造函数之一），以提供新对象的初始值
    3. 若`T`是 *数组类型* ，则每个数组元素都被 *默认初始化* 
    4. 否则，不做任何事。 *自动对象（及其子对象）* 被初始化为 *不确定值*  
- 注意事项
    - 若`T`是`const`限定类型，则它必须是 *具有用户提供的默认构造函数* 的 *类类型* 
    - **不能**默认初始化 *引用*   






### 🌱 [成员访问操作符](https://en.cppreference.com/w/cpp/language/operator_member_access)（Member access operators）

- 用于访问其操作数的成员，包括
    - *下标* （subscript）：提供对其指针或数组操作数所指向的对象的访问
        - 语法：`a[b]`
        - 可重载
        - 类内定义：`R & T::operator[](S b);`
        - **不可**类外定义
    - *间接寻址* （indirection），又称 *解引用* （dereference）：提供对其指针操作数所指向的对象或函数的访问 
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
    - 对象 *尾后* (off-the-end) 指针
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
 
int * px  = &c.x;                       // px  的值为 指向 c.x 的指针
int * pxe = px + 1;                     // pxe 的值为  c.x 的尾后指针
int * py  = &c.y;                       // py  的值为 指向 c.y 的指针

assert(pxe == py);                      // 测试两个指针是否表示相同地址
                                        // 这条 assert 可能被触发，也可能不被触发
                                        // 至少 gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04) 上实测没有触发
 
*pxe = 1;                               // 即使上面的 assert 未被触发，亦为未定义行为
```
- **注意**：理论上指向同一对象的两个多态指针也可能有不同的值
    - 一般发生于指向 *虚多重继承* （multiple virtual inheritence）的类的多态指针上
    - 和虚多重继承的类内部的结构有关，按标准属于 *未定义行为* 
```c++
struct A
{
    virtual void foo() {}
    unsigned long long a {0x1112131415161718};
};

struct B : virtual public A
{
    void foo() override {}
    unsigned long long b {0x2122232425262728};
};

struct C : virtual public A
{
    void foo() override {}
    unsigned long long c {0x3132333435363738};
};

struct D : public B, public C
{
    void foo() override {}
    unsigned long long d {0x4142434445464748};
};

std::shared_ptr<D> pd(new D);
A * pa = pd.get();
B * pb = pd.get();
C * pc = pd.get();

// 16 32 32 56
std::cout << sizeof(A) << ' '
          << sizeof(B) << ' '
          << sizeof(C) << ' '
          << sizeof(D) << '\n';

// Hacked G++ Memory Layout of Classes Involving Multiple Virtual Inheritance.
// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0 
// The exact layout varies from implementations and relying on that is undefined behavior! 
// Address (Byte) Low -> High. 
// 0      7 8     15 16    23 24    31 32    39 40    47 48    55
// [      ] [ B::b ] [      ] [ C::c ] [ D::d ] [      ] [ A::a ]
// |<----- B ----->| |<----- C ----->|          |<----- A ----->|
// |<--------------------------- D ---------------------------->|
auto p = reinterpret_cast<unsigned char *>(pd.get());

for (std::size_t i = 0; i != sizeof(D); ++i)
{
    std::printf("%p 0x%02x\n", p + i, *(p + i));
}

// 0x5612b6d78ed8 0x5612b6d78eb0
// 0x5612b6d78eb0 0x5612b6d78eb0
// 0x5612b6d78ec0 0x5612b6d78eb0
// 0x5612b6d78eb0 0x5612b6d78eb0
std::cout << pa << ' ' << dynamic_cast<const void *>(pa) << '\n'
          << pb << ' ' << dynamic_cast<const void *>(pb) << '\n'
          << pc << ' ' << dynamic_cast<const void *>(pc) << '\n'
          << pd << ' ' << dynamic_cast<const void *>(pd.get()) << '\n';
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
        - 数组指针的`typedef`或 *类型别名* ，可用于简化定义
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
    - 这些玩意儿的解释方法和文法参见开篇章节中的如何理解`C`声明
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

#### （类的）数据成员指针 (Class Data Member Pointer)

- `Class::*`表示 *成员指针* (Member Pointer, Pointer to Member)
    - `Class::*` is just a vexing C++ syntax which is consistant to the vexing C-style declaration syntax. 
      See the following example: 
    ```c++
    struct C
    {
        explicit C(int v_ = 0, C * cp_ = nullptr) : v(v_), cp(cp_) {}

        void foo() { std::cout << "void C::foo()\n"; }

        int v;
        C * cp;
    };

    int C::* pointerToMemberOfCWhoseTypeIsInt = &C::v;
    C * C::* pointerToMemberOfCWhoseTypeIsPointerToC = &C::cp;
    void (C::* pointerToMemberOfCWhoseTypeIsFunctionTakingNoArgumentAndReturningVoid)() = &C::foo;

    auto cp = std::make_shared<C>();

    std::cout << (*cp).*pointerToMemberOfCWhoseTypeIsInt << '\n';                          // 0
    std::cout << ((*cp).*pointerToMemberOfCWhoseTypeIsPointerToC == nullptr) << '\n';      // 1
    ((*cp).*pointerToMemberOfCWhoseTypeIsFunctionTakingNoArgumentAndReturningVoid)();      // void C::foo()

    std::cout << cp.get()->*pointerToMemberOfCWhoseTypeIsInt << '\n';                      // 0
    std::cout << (cp.get()->*pointerToMemberOfCWhoseTypeIsPointerToC == nullptr) << '\n';  // 1
    (cp.get()->*pointerToMemberOfCWhoseTypeIsFunctionTakingNoArgumentAndReturningVoid)();  // void C::foo()
    ```
    - 至于是数据成员指针，还是成员函数指针，那就跟普通函数指针与普通对象指针的区别一样，只看括号了
- 指向类`C`的 *非静态数据成员* `m`的指针，以`&C::m`初始化
    - 这是 *类* 的一个 *附属* ，跟具体的某个对象没关系
    - `C`的 *成员函数* 中，`&(C::m)`、`&m`等**不再是**数据成员指针
- 能用作 [*成员指针访问运算符*](https://en.cppreference.com/w/cpp/language/operator_member_access) `operator.*`、`operator->*`的右操作数
    - 使得每个 *该类的对象* 都能用这个 *类的数据成员指针* 访问到自己的数据成员
```c++
struct C { int m; };

int C::* p = &C::m;                    // pointer to data member m of class C
C c = {7};
std::cout << c.*p << std::endl;        // prints 7

C * cp = &c;
cp->m = 10;
std::cout << cp->*p << std::endl;      // prints 10
```
- 无二义 *非虚基类的数据成员指针* 可以 *隐式转化* 为 *派生类的数据成员指针*
```c++
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
```c++
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

#### （类的）成员函数指针 (Class Member Function Pointer)

- 指向类`C`的 *非静态成员函数* `f`的指针，以`&C::f`初始化
    - 这是 *类* 的一个 *附属* ，跟具体的某个对象没关系
    - `C`的 *成员函数* 中，`&(C::f)`、`&f`等**不再是**成员函数指针
- 能用作 [*成员指针访问运算符*](https://en.cppreference.com/w/cpp/language/operator_member_access) `operator.*`、`operator->*`的右操作数
    - 使得每个 *该类的对象* 都能用这个 *类的数据成员指针* 访问到自己的数据成员
    - 结果表达式 *只能用作* 函数调用运算符的 *左操作数* 
```c++
struct C
{
    void f(int n) { std::cout << n << '\n'; }
};
 
void (C::* p)(int) = &C::f;            // 指向类 C 的成员函数 f 的指针

C c;
(c.*p)(1);                             // 打印 1

C * cp = &c;
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






### 🌱 [引用声明](https://en.cppreference.com/w/cpp/language/reference)

- 一网打尽各种引用声明

#### 引用声明

1. `T  & d = ...`： *左值引用声明符* ，`d`为`T`类型的 *左值引用* 
2. `T && d = ...`： *右值引用声明符* ，`d`为`T`类型的 *右值引用* 
```
decl-specifier-seq  & attr(optional) declarator    (1)
decl-specifier-seq && attr(optional) declarator    (2)
```
- *声明说明符序列* （declarator specifier sequence）：用于说明引用绑定的类型
    - *被绑定类型* 的`cv`限定是 *声明说明符序列* 的一部分
- `attr`： *属性列表* ，可选
- *声明符* （declarator）：除 *引用* 声明符**之外**的任意声明符，可以是另一 *指针* 声明符
    - 引用必须被 *初始化* 为指代一个有效的对象或函数
    - **不**存在 *`void`的引用* 以及 *引用的引用* 
    - 引用类型无法在顶层被`cv`限定
        - 声明中没有为此而设的语法
        - 若将限定性添加到`typedef`名、`decltype`说明符或 *类型模板形参* ，则忽略它
- 引用**不是**对象；它们不必占用内存，尽管若需要，编译器会分配内存空间
    - 例如，引用类型的非静态数据成员通常会增加类的大小，量为存储内存地址所需
- 因为引用**不是**对象，故
    - **不**存在 *引用的数组*
    - **不**存在 *指向引用的指针* 
    - **不**存在 *引用的引用* 
```
int & a[3];  // error
int &* p;    // error
int & &r;    // error
```
- *引用坍缩* （Reference collapsing） => 16.2.5
    - 容许通过 *模板* 或 *`typedef`中的类型操作* 构成 *引用的引用* 
        - 这种情况下适用引用坍缩（reference collapsing）规则
        - *右值引用的右值引用* 坍缩成 *右值引用* ， *所有其他组合* 均坍缩成 *左值引用* 
    - 这条规则，和将`T &&`用于 *函数模板* 时的 *模板实参推导* 的特殊规则一起，组成使得`std::forward`可行的规则
```
typedef int &  lref;
typedef int && rref;
int n;
lref &  r1 = n;  // type of r1 is int &
lref && r2 = n;  // type of r2 is int &
rref &  r3 = n;  // type of r3 is int &
rref && r4 = 1;  // type of r4 is int &&
```

#### 左值引用（lvalue references）

- 左值引用可用于建立既存对象的 *别名* （可选地拥有不同的`cv`限定） 
```
std::string s = "Ex";
std::string & r1 = s;
const std::string & r2 = s;

r1 += "ample";                 // modifies s
r2 += "!";                     // error: cannot modify through reference to const
std::cout << r2 << std::endl;  // prints s, which now holds "Example"
```
- 它们亦可用于在函数调用中实现 *按引用传递* 
```
void double_string(std::string & s) 
{
    s += s;  // 's' is the same object as main()'s 'str'
}
 
std::string str = "Test";
double_string(str);
std::cout << str << std::endl;
``` 
- 当函数的返回值是左值引用时，函数调用表达式成为左值表达式
```
char & char_number(std::string & s, std::size_t n) 
{
    return s.at(n);         // string::at() returns a reference to char
}
 
std::string str = "Test";
char_number(str, 1) = 'a';  // the function call is lvalue, can be assigned to
std::cout << str << std::endl;
```

#### 右值引用（lvalue references） => 13.6.1

- 右值引用可用于 *为临时对象延长生存期* 
    - 注意， *常左值引用* 亦能延长临时对象生存期，但不能通过常左值引用修改它们
```
std::string s1 = "Test";
std::string && r1 = s1;            // error: can't bind to lvalue

const std::string & r2 = s1 + s1;  // okay: lvalue reference to const extends lifetime
r2 += "Test";                      // error: can't modify through reference to const

std::string && r3 = s1 + s1;       // okay: rvalue reference extends lifetime
r3 += "Test";                      // okay: can modify through reference to non-const
std::cout << r3 << std::endl;;
```
- 当函数同时具有 *右值引用* 和 *左值引用* 的 *重载* 时
    - *右值引用重载* 绑定到 *右值* （包含 *纯右值* 和 *将亡值* ）
    - *左值引用重载* 绑定到 *左值* 
- 这允许在适当时机自动选择 *移动构造函数* 、 *移动赋值运算符* 和其他 *具移动能力的函数* 
    - 例如`std::vector::push_back()`
```
void f(int & x) 
{
    printf("lvalue reference overload f(%d)\n", x);
}
 
void f(const int & x) 
{
    printf("lvalue reference to const overload f(%d)\n", x);
}
 
void f(int && x)
{
    printf("rvalue reference overload f(%d)\n", x);    
}
 
int i = 1;
const int ci = 2;
f(i);             // calls f(int &)
f(ci);            // calls f(const int &)
f(3);             // calls f(int &&)
                  // would call f(const int &) if f(int &&) overload wasn't provided
f(std::move(i));  // calls f(int && )

// rvalue reference variables are lvalues when used in expressions
int && x = 1;
f(x);             // calls f(int & x)
f(std::move(x));  // calls f(int && x)
```
- 因为 *右值引用* 能绑定到 *将亡值* ，故它们能指代 *非临时对象* 
    - 这使得可以将作用域中 *不再需要的对象* *移动出去* 
```
int i2 = 42;
int && rri = std::move(i2);         // binds directly to i2

std::vector<int> v{1, 2, 3, 4, 5};
std::vector<int> v2(std::move(v));  // binds an rvalue reference to v
assert(v.empty());
```

#### 转发引用（Forwarding Reference） => 16.2.7

- 转发引用是一种特殊的引用，它保持函数实参的 *值类别* ，使得能利用`std::forward`转发实参
- 转发引用是下列之一 
    1. *函数模板的函数形参* ，其被声明为同一函数模板的类型模板形参的 *无`cv`限定的右值引用* 
        - 这玩意如果不特意规定，那妥妥的就是 *二义性调用*
        - 所以强制规定，按照 *实参* 的 *值类别* ，`T`分别匹配成 *左值引用* 或 *右值引用*
        - 然后和形参处的 *右值引用* 放一块儿，来一波 *引用坍缩* ，美滋滋
        - 老冤家：[`std::move`](https://en.cppreference.com/w/cpp/utility/move)，这玩意就是个转发引用，可别认成普通的右值引用了
    ```
    template <class T>
    int g(const T && x);               // x is not a forwarding reference
                                       // T is const-qualified
    
    template <class T>
    int f(T && x)                      // x is a forwarding reference
    {                    
        return g(std::forward<T>(x));  // and so can be forwarded
    }
     
    int i;
    f(i);   // argument is lvalue, calls f<int &>(int &), std::forward<int &>(x) is lvalue
    f(0);   // argument is rvalue, calls f<int>(int &&), std::forward<int>(x) is rvalue
    
    template <class T> 
    struct A 
    {
        template <class U>
        A(T && x, U && y, int * p);    // x is not a forwarding reference
                                       // T is not a type template parameter of the constructor
                                       // y is a forwarding reference
    };
    
    template <class T>
    typename std::remove_reference<T>::type && move(T && t) noexcept;  // t is a forwarding reference
    ```
    2. `auto &&`，但当其 *从花括号初始化器列表推导* 时则**不是**
    ```
    auto && vec = foo();       // foo() may be lvalue or rvalue, vec is a forwarding reference
    auto i = std::begin(vec);  // works either way
    (*i)++;                    // works either way
    g(std::forward<decltype(vec)>(vec)); // forwards, preserving value category
     
    for (auto && x: f()) 
    {
        // x is a forwarding reference; this is the safest way to use range for loops
    }
     
    auto && z = {1, 2, 3};     // NOT a forwarding reference (special case for initializer lists)
    ```

#### 悬垂引用（Dangling References）

- 但有可能创建一个程序，被指代对象的生存期结束，但引用仍保持可访问（ *悬垂* ）
    - 访问悬垂引用是 *未定义行为* 
    - 一个常见例子是 *返回自动变量的非常量左值引用的函数* 
        - 比如函数内定义的临时量
```
std::string & f()
{
    std::string s = "Example";
    return s;                   // exits the scope of s:
                                // its destructor is called and its storage deallocated
                                // dangling reference
}

const std::string & g()
{
    std::string s = "Example";
    return s;                   // dangling reference
}

std::string && h()
{
    std::string s = "Example";
    return std::move(s);        // dangling reference
}
```






### 🌱 复合类型

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
        - *常引用* 可以绑定在 *其他类型* 的 *右值* 上。尤其，允许为一个常引用绑定
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

#### `typedef`和类型别名

```
typedef int * intptr;
using intptr2 = int *;

int a = 1;
const intptr p = &a;             // "const (int *)", i.e. `int * const`. NOT `const int *`!!!
const intptr2 p2 = &a, p3 = &a;  // 注意这里 p3 已经是指针了，不需要再加 *
```

#### [`auto`](https://en.cppreference.com/w/cpp/language/auto)

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
    auto b = r;                  // b 为 int ，而不是 int &
    ```
    - 对于`const`：`auto`会忽略顶层`const`
    ```
    int i = 1;
    const int ci = i, &cr = ci;
    auto b = ci;                 // b为int（ ci 为顶层 const ）
    auto c = cr;                 // c为int（ cr 为 ci 的别名,  ci 本身是顶层 const ）
    auto d = &i;                 // d为int *（&i 为 const int * ）
    auto e = &ci;                // e为const int *（对常量对象取地址是底层 const ）
    ```
    - 如果希望`auto`推断出引用或者顶层常量，则声明`auto`时必须加上相应的描述符
    ```
    const auto f = ci;           // f 为 const int
    auto & g = ci;               // g 为 const int &
    auto & h = 42;               // 错误：不能为非常量引用绑定字面值
    const auto & j = 42;         // 正确：可以为常量引用绑定字面值
    ```
    - `auto`一句话定义多个变量时，所有变量类型必须一样。注意`*`和`*`是从属于声明符的，而不是基本数据类型的一部分
    ```
    auto k = ci, &l = ci;        // k 为 int，l 为 int &
    auto & m = ci, *p2 = &ci;    // m 为 const int & ，p2 为 const int *
    auto & n = i, *p2 = &ci;     // 错误： i 的类型为 int ，而 &ci 的类型为 const int
    ```

#### [`decltype`](https://en.cppreference.com/w/cpp/language/decltype)

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
- `typeof`是`gcc`自己搞的村规，**不是**`C++`标准
    - `decltype`是`C++11`才有的
    - `typeof`特性跟`decltype`不一样，而且不能跨平台，**不要用**
    - 另外，别把它和`RTTI`运算符`typeid`给搞混了 => 19.2.2






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






### 🌱 [值类别](https://en.cppreference.com/w/cpp/language/value_category)（Value Category）

#### History

With the introduction of move semantics in C++11,
value categories were redefined to characterize two independent properties of expressions:

- **has identity**:
  it's possible to determine whether the expression refers to the same entity as another expression,
  such as by comparing addresses of the objects or the functions they identify (obtained directly or indirectly);
- **can be moved from**:
  move constructor, move assignment operator, or another function overload
  that implements move semantics can bind to the expression.

In C++11, expressions that:

- have identity and cannot be moved from are called `lvalue` expressions;
- have identity and can be moved from are called `xvalue` expressions;
- do not have identity and can be moved from are called `prvalue` expressions;
- do not have identity and cannot be moved from are **not** used.

The expressions that have identity are called `glvalue` expressions.
Both lvalues and `xvalues` are `glvalue` expressions.


The expressions that can be moved from are called `rvalue` expressions.
Both `prvalues` and `xvalues` are `rvalue` expressions.

#### 基本值类别

每个表达式 *只属于* 三种 *基本值类别* 中的一种： 

- *左值* `lvalue`
    - 包括（看看就得了，当字典用的）
        1. 变量、函数、`模板形参对象 (since C++20)`或数据成员之名，不论其类型，例如`std::cin`或`std::endl`
            - 即使变量的类型是 *右值引用* ，由其名字构成的表达式仍是左值表达式
        2. 返回类型为左值引用的函数调用或重载运算符表达式，例如`std::getline(std::cin, str)`、`std::cout << 1`、`str1 = str2`或`++it`
        3. `a = b`，`a += b`，`a %= b`，以及所有其他内建的赋值及复合赋值表达式
        4. `++a`和`--a`，内建的前置自增与前置自减表达式
        5. `*p`，解引用指针
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
            - 特别是， *将亡值* 可以是 *多态* 的，而且 *非类* 的 *将亡值* 可以有`cv`限定
        2. 与 *右值* 相同（见下文）
            - 特别是， *将亡值* 可以绑定到 *右值引用* 上 
- *纯右值* `prvalue`（pure rvalue）
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

#### 复合值类别

- *泛左值* `glvalue`（generalized lvalue）
    - 哪些值类别是 *泛左值* 
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
        - *位域* （Bit fields）
            - 代表 *位域* 的表达式（例如`a.m`，其中`a`是类型`struct A { int m: 3; }`的 *左值* ）是 *泛左值* 
            - 可用作 *赋值运算符的左操作数* 
            - **不能** *取地址* 
                - **不能**绑定于 *非常量左值引用* 上
                - *常量左值引用* 或 *右值引用* 可以从位域泛左值初始化，但不会直接绑定到位域，而是绑定到一个 *临时副本* 上
- *右值* `rvalue`（rvalue，如此称呼的历史原因是，右值可以出现于赋值表达式的右边）
    - 哪些值类别是 *右值* 
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






### 🌱 类型转换（Type Conversions, Type cast）


- 所有`cast<T>`的结果的 *值类别* （value category）是
    - *左值* `lvalue`，如果`T`为 *左值引用* 或 *函数类型的右值引用*  
    - *将亡值* `xvalue`，如果`T`为 *对象类型的右值引用*
    - *纯右值* `prvalue`，其他情况。此时生成转换结果需要一次 *拷贝构造* 
- See [Effective Cpp Notes Item 27: Minimize casting](./effective_cpp_notes_01_effective_cpp.md#-item-27-minimize-casting) for common pitfalls. 


#### [`static_cast`](https://en.cppreference.com/w/cpp/language/static_cast)

```c++
static_cast<T>(expr)
```
Only the following conversions can be done with `static_cast`, 
**except** when such conversions would cast away const-ness or volatility.
1. If `T` is a reference or pointer to class `D` 
   and `expr` is <`lvalue` reference or `prvalue` pointer> of `D`'s non-virtual base `B`: 
   - `static_cast` performs a _downcast_. 
   - This downcast is ill-formed if `B` is ambiguous, inaccessible, or virtual base (or a base of a virtual base) of `D`. 
   - Such a downcast makes **no** runtime checks to ensure that the object's runtime type is actually `D`, 
     and may only be used safely if this precondition is guaranteed by other means, 
     such as when implementing [static polymorphism](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern#Static_polymorphism). 
   - Safe _downcast_ may be done with `dynamic_cast`. 
2. If `T` is an `rvalue` reference type:   
   - `static_cast` converts the value of `glvalue`, class `prvalue`, or any `lvalue` expression 
     to `xvalue` referring to the same object as the expression, 
     or to its base sub-object (depending on `T`). 
     - If the target type is an inaccessible or ambiguous base of the type of the expression, the program is ill-formed. 
     - If the expression is a bit field `lvalue`, it is first converted to `prvalue` of the underlying type. 
     - This type of `static_cast` is used to implement move semantics in `std::move`. 
3. If there is an implicit conversion sequence from `expr` to `T`, 
   or if overload resolution for a direct initialization of an object or reference of type `T` from `expr` 
   would find at least one viable function: 
   - `static_cast<T>(expr)` returns the imaginary variable `tep` initialized as if by `T tmp(expr);`, 
     which may involve implicit conversions, a call to the constructor of `T`, or a call to a user-defined conversion operator. 
     - For non-reference `T`, the result object of the `static_cast` `prvalue` expression is what's direct-initialized. 
4. If `T` is the type `void` (possibly `cv`-qualified): 
   - `static_cast` discards the value of `expr` after evaluating it. 
5. If a standard conversion sequence from `T` to the type of `expr` exists, 
   that does **not** include `lvalue`-to-`rvalue`, 
   array-to-pointer, function-to-pointer, null pointer, null member pointer, function pointer or boolean conversion:  
   - `static_cast` can perform the inverse of that implicit conversion. 
6. If conversion of `expr` to `T` involves `lvalue`-to-`rvalue`, array-to-pointer, or function-to-pointer conversion, 
   it can be performed explicitly by `static_cast`.
7. Scoped enumeration type can be converted to an integer or floating-point type: 
   - The result is the same as [implicit conversion](https://en.cppreference.com/w/cpp/language/implicit_conversion) 
     from the enum's underlying type to the destination type. 
8. Numeric type to complete enumeration type:
   - A value of integer or enumeration type can be converted to any complete enumeration type. 
     - If the underlying type of the enumeration is **not** fixed, 
       the behavior is _undefined_ 
       if the value of expression is out of range 
       (the range is all values possible for the smallest bit field
       large enough to hold all enumerators of the target enumeration). 
     - If the underlying type of the enumeration is fixed,
       the result is the same as converting the original value 
       first to the underlying type of the enumeration
       and then to the enumeration type.
   - A value of a floating-point type can also be converted to any complete enumeration type.
     - The result is the same as converting the original value 
       first to the underlying type of the enumeration, 
       and then to the enumeration type.
9. A pointer to member of some class `D` can be _upcast_ to 
   a pointer to member of its unambiguous, accessible base class `B`. 
   - This `static_cast` makes **no** checks to ensure 
     the member actually exists in the runtime type of the pointed-to object. 
10. A `prvalue` of type `void *` (possibly `cv`-qualified) can be converted to pointer to any object type. 
    - If the original pointer value represents an address of a byte in memory 
      that does **not** satisfy the alignment requirement of the target type, 
      then the resulting pointer value is unspecified. 
    - Otherwise:
      - If the original pointer value points to an object `a`, 
        and there is an object `b` of the target type (ignoring `cv`-qualification) 
        that is _pointer-interconvertible_ with `a`, the result is a pointer to `b`. 
      - Otherwise, the pointer value is unchanged. 
    - Conversion of any pointer to `void *` and back to pointer to the original (or more `cv`-qualified) type 
      preserves its original value.


As with all cast expressions, the result is:
- An `lvalue` if `T` is an `lvalue` reference type, or an `rvalue` reference to function type;
- An `xvalue` if `T` is an `rvalue` reference to object type;
- A `prvalue` otherwise. In this case, a copy is made. 


Two objects `a` and `b` are _pointer-interconvertible_ if one of these applies:
- They are the same object;
- One is a union object and the other is a non-static data member of that object;
- One is a [standard-layout](https://en.cppreference.com/w/cpp/language/data_members#Standard_layout) class object 
  and the other is the first non-static data member of that object.  
  Or, if the object has **no** non-static data members, any base class sub-object of that object;
- There exists an object `c` such that `a` and `c` are pointer-interconvertible, 
  and `c` and `b` are pointer-interconvertible. 


**Notes**: 
`static_cast` may also be used to _disambiguate_ function overloads 
by performing a function-to-pointer conversion to specific type 
```c++
std::for_each(files.begin(), 
              files.end(),
              static_cast<std::ostream & (*)(std::ostream &)>(std::flush));
```

#### [`dynamic_cast`](https://en.cppreference.com/w/cpp/language/dynamic_cast)

- `dynamic_cast<T>(expr)`
    - Safely converts pointers and references to classes up, down, and sideways along the inheritance hierarchy.
    - `dynamic_cast`s are usually expensive, might involve multiple `std::strcmp`s along the inheritance hierarchy. 
    - Supports _Runtime Type Identification (RTTI)_

If the cast is successful, `dynamic_cast` returns a value of type `T`.   
If the cast fails and `T` is a pointer type, it returns a null pointer of that type.   
If the cast fails and `T` is a reference type, it throws an exception that matches a handler of type `std::bad_cast`.

Only the following conversions can be done with `dynamic_cast`, 
**except** when such conversions would cast away constness or volatility: 
1. If the type of `expr` is exactly `T` or a less cv-qualified version of `T`, 
   the result is the value of `expr`, with type T. 
   (In other words, dynamic_cast can be used to add constness. 
   An implicit conversion and `static_cast` can perform this conversion as well.)
2. If the value of `expr` is the null pointer value, 
   the result is the null pointer value of type `T`.
3. If `T` is a pointer or reference to `Base`, 
   and the type of `expr` is a pointer or reference to `Derived`, 
   where `Base` is a unique, accessible base class of `Derived`, 
   the result is a pointer or reference to the `Base` class sub-object within the `Derived` object pointed or identified by `expr`. 
   (Note: an implicit conversion and `static_cast` can perform this conversion as well.)
4. If `expr` is a pointer to a polymorphic type, 
   and `T` is a pointer to `void`, 
   the result is a pointer to the most derived object pointed or referenced by `expr`.
5. If `expr` is a pointer or reference to a polymorphic type `Base`, 
   and `T` is a pointer or reference to the type `Derived`, 
   a run-time check is performed:
   1. The most derived object pointed/identified by `expr` is examined. 
      If, in that object, `expr` points/refers to a public base of `Derived`, 
      and if only one object of `Derived` type is derived from the sub-object pointed/identified by `expr`, 
      then the result of the cast points/refers to that `Derived` object. (This is known as a _downcast_.)
   2. Otherwise, if `expr` points/refers to a public base of the most derived object, 
      and, simultaneously, the most derived object has an unambiguous public base class of type `Derived`, 
      the result of the cast points/refers to that `Derived` (This is known as a _sidecast_.)
   3. Otherwise, the runtime check fails. 
      If the `dynamic_cast` is used on pointers, the null pointer value of type `T` is returned. 
      If it was used on references, the exception `std::bad_cast` is thrown.
6. When `dynamic_cast` is used in a constructor or a destructor (directly or indirectly), 
   and `expr` refers to the object that's currently under construction/destruction, 
   the object is considered to be the most derived object. 
   If T is not a pointer or reference to the constructor's/destructor's own class or one of its bases, 
   the behavior is undefined.


A _downcast_ can also be performed with `static_cast`, 
which avoids the cost of the runtime check, 
but it's only safe if the program can guarantee (through some other logic) 
that the object pointed to by expression is definitely `Derived`. 


Similar to other cast expressions, the result is:
- An `lvalue` if `T` is an `lvalue` reference type (`expr` must be an lvalue)
- An `xvalue` if `T` is an `rvalue` reference type
  (`expr` must be a `glvalue` (`prvalues` are materialized `since C++17`) of a complete class type)
- A `prvalue` if `T` is a pointer type


Some forms of `dynamic_cast` rely on _Runtime Type Identification (RTTI)_, 
that is, information about each polymorphic class in the compiled program. 
Compilers typically have options to disable the inclusion of this information.

#### [`const_cast`](https://en.cppreference.com/w/cpp/language/const_cast)

```c++
const_cast<T>(expr)
```
Only the following conversions can be done with `const_cast`. 
In particular, only `const_cast` may be used to _cast away_ (remove) const-ness or volatility. 
1. Two possibly multilevel pointers to the same type may be converted between each other, 
   **regardless of** cv-qualifiers at each level;
2. To reference types:
   - A `lvalue` reference of any type `T` may be converted to 
     a more or less cv-qualifie `lvalue` or `rvalue` reference to the same type `T`.
   - A `prvalue` of class type or an `xvalue` of any type may be converted 
     to a more or less cv-qualified `rvalue` reference. 
   - The result of a reference `const_cast` refers to the original object if `expr` is a `glvalue`, 
     and to the materialized temporary otherwise.
3. Same rules apply to possibly multilevel pointers to data members 
   and possibly multilevel pointers to arrays of known and unknown bound 
   (arrays to cv-qualified elements are considered to be cv-qualified themselves). 
4. Null pointer value may be converted to the null pointer value of `T`. 


As with all cast expressions, the result is:
- An `lvalue` if `T` is an `lvalue` reference type, or an `rvalue` reference to function type;
- An `xvalue` if `T` is an `rvalue` reference to object type;
- A `prvalue` otherwise. In this case, a copy is made.


**Notes**.
- Pointers to functions and pointers to member functions are **not** subject to `const_cast`.
- `const_cast` makes it possible to form 
  a reference or pointer to non-const type that is actually referring to a const object, 
  or a reference or pointer to non-volatile type that is actually referring to a volatile object. 
  Modifying a const object through a non-const access path 
  and referring to a volatile object through a non-volatile `glvalue` 
  results in _undefined behavior_.


#### [`reinterpret_cast`](https://en.cppreference.com/w/cpp/language/reinterpret_cast)


```c++
reinterpret_cast<T>(expr)
```
Unlike `static_cast`, but like `const_cast`, 
the `reinterpret_cast` expression does **not** compile to any CPU instructions 
(**except** when converting between integers and pointers 
or on obscure architectures where pointer representation depends on its type). 
It is purely a compile-time directive which instructs the compiler 
to treat `expr` as if it had the type `T`. 


Only the following conversions can be done with `reinterpret_cast`, 
**except** when such conversions would cast away const-ness or volatility.
1. `expr` of integral, enumeration, pointer, or pointer-to-member type can be converted to its own type. 
   The resulting value is the same as the value of `expr`. 
2. A pointer can be converted to any integral type large enough to hold all values of its type (e.g. to `std::uintptr_t`). 
3. A value of any integral or enumeration type can be converted to a pointer type. 
   - A pointer converted to an integer of sufficient size and back to the same pointer type is guaranteed to have its original value.
   - The round-trip conversion in the opposite direction is **not** guaranteed,  
     because the same pointer may have multiple integer representations. 
   - The null pointer constant `NULL` or integer zero is also **not** guaranteed 
     to yield the null pointer value of the target type. 
     `static_cast` or implicit conversion should be used for this purpose. 
4. Any value of type `std::nullptr_t`, including `nullptr`, 
   can be converted to any integral type as if it were `(void *) 0`, 
   but **no** value, not even `nullptr` can be converted to `std::nullptr_t`. 
   - The null pointer constant `NULL` or integer zero is also **not** guaranteed
     to yield the null pointer value of the target type.
   - `static_cast` or implicit conversion should be used for this purpose.
5. Any object pointer type `T1 *` can be converted to another object pointer type cv `T2 *`. 
   - This is exactly equivalent to `static_cast<cv T2 *>(static_cast<cv void *>(expr))`,  
     which implies that if `T2`'s alignment requirement is **not** stricter than `T1`'s, 
     the value of the pointer does **not** change 
     and conversion of the resulting pointer back to its original type yields the original value. 
   - In any case, the resulting pointer may only be de-referenced safely if allowed by the _type aliasing rules_. 
6. An `lvalue` reference of type `T1` can be converted to reference to another type `T2`. 
   - The result is an `lvalue` or `xvalue` referring to the same object as the original lvalue, but with a different type. 
   - **No** temporary is created, **no** copy is made, **no** constructors or conversion functions are called. 
   - The resulting reference can only be accessed safely if allowed by the _type aliasing rules_. 
7. Any pointer to function can be converted to a pointer to a different function type. 
   - Calling the function through a pointer to a different function type is _undefined_, 
     but converting such pointer back to pointer to the original function type 
     yields the pointer to the original function. 
8. On some implementations (in particular, on any POSIX compatible system as required by dlsym), 
   a function pointer can be converted to `void *` or any other object pointer, or vice versa. 
   - If the implementation supports conversion in both directions, 
     conversion to the original type yields the original value.
9. The null pointer value of any pointer type can be converted to any other pointer type, 
   resulting in the null pointer value of that type. 
   - Note that the null pointer constant `nullptr` or any other value of type `std::nullptr_t` 
     can **not** be converted to a pointer with `reinterpret_cast`. 
     Implicit conversion or `static_cast` should be used for this purpose. 
10. A pointer to member function can be converted to pointer to a different member function of a different type. 
    - Conversion back to the original type yields the original value.
11. A pointer to data member of some class `T1` can be converted to a pointer to another data member of another class `T2`. 
    - If `T2`'s alignment is **not** stricter than `T1`'s, 
      conversion back to the original type `T1` yields the original value.


As with all cast expressions, the result is:
- An `lvalue` if `T` is an `lvalue` reference type, or an `rvalue` reference to function type;
- An `xvalue` if `T` is an `rvalue` reference to object type;
- A `prvalue` otherwise. In this case, a copy is made.


**Type Aliasing**.  
Whenever an attempt is made to read or modify the stored value of an object of type `DynamicType` 
through a `glvalue` of type `AliasedType`, 
the behavior is _undefined_ unless one of the following is true:
- `AliasedType` and `DynamicType` are _similar_. 
  - Two types are _similar_ if, ignoring top-level cv-qualification:
    - They are the same type;
    - They are both pointers, and the pointed-to types are similar; 
    - They are both pointers to member of the same class, 
      and the types of the pointed-to members are similar; 
    - They are both arrays of the same size or at least one of them is array of unknown bound, 
      and the array element types are similar.
- `AliasedType` is the (possibly cv-qualified) signed or unsigned variant of `DynamicType`. 
- `AliasedType` is [`std::byte`](https://en.cppreference.com/w/cpp/types/byte), `char`, or `unsigned char`: 
  this permits examination of the object representation of any object as an array of bytes. 


#### [旧式强制类型转换](https://en.cppreference.com/w/cpp/language/explicit_cast)

- *旧式强制类型转换* 使用`C`风格写法和函数式写法，用显式和隐式转换的组合进行类型之间的转换
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

#### [用户定义转换](https://en.cppreference.com/w/cpp/language/cast_operator) => 14.9

- 表达式被用作 *条件* 时， *类型转换运算符* 即使是`explicit`的，仍会被 *隐式应用* => 14.9.1
```
operator conversion-type-id             (1)  // 声明用户定义的类型转换运算符，它参与所有隐式和显式转换
explicit operator conversion-type-id    (2)  // 声明用户定义的类型转换运算符，它仅参与直接初始化和显式转换 (since C++11)

struct X 
{
    explicit operator bool() const { return true; }     // 显式转换，但用作条件时仍可隐式调用
    
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

int * p = static_cast<int *>(x);                        // OK：设 p 为 null
int * q = x;                                            // 错误：无隐式转换

if (x) printf("x\n");                                   // OK
```

#### [隐式类型转换](https://en.cppreference.com/w/cpp/language/implicit_conversion)（Implicit conversion）

- *隐式类型转换* 就是由 *编译器自动完成* 的类型转换
- 凡是在语境中使用了某种表达式类型`T1`，但语境不接受该类型，而接受另一类型`T2`的时候，会进行 *隐式类型转换* ，具体是
    1. 调用以`T2`为形参声明的函数时，以该表达式为实参
    2. 运算符期待`T2`，而以该表达式为操作数
    3. 初始化`T2`类型的新对象，包括在返回`T2`的函数中的`return`语句
    4. 将表达式用于`switch`语句（`T2`为整型类型）
    5. 将表达式用于`if`语句或循环（`T2`为`bool`）
- 仅当存在一个从`T1`到`T2`的无歧义 *隐式转换序列* 时，程序 *良构* （能编译）
- *隐式转换序列* 由下列内容依照这个顺序所构成 (implicit conversion sequence)
    1. 零或一个 *标准转换序列* (standard conversion sequence)
        1. 下列三者中的零或一个：
            - *左值到右值* 转换
            - *数组头到指针* 转换
            - *函数头到指针* 转换
        2. 零或一个 *数值提升* 或 *数值转换* 
        3. 零或一个 *函数指针转换* `(since C++17)`
        4. 零或一个 *限定调整* 
    2. 零或一个 *用户定义转换* (user-defined conversion)
        - 零或一个非`explicit` *单实参构造函数* 或非`explicit` *类型转换运算符* 的调用构成
        - 表达式被用作 *条件* 时， *类型转换运算符* 即使是`explicit`的，仍会被 *隐式应用* => 14.9.1
    3. 零或一个 *标准转换序列* 
        - 当考虑用户定义的 *构造函数* 或 *类型转换运算符* 时， *只允许一个* 标准转换序列
            - 否则将实际上可以将用户定义转换串连起来
        - 从一个 *内建类型* 转换到另一内建类型时， *只允许一个* 标准转换序列






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


### 🌱 [`C++17`引入的大宝贝](https://en.cppreference.com/w/cpp/17) (C++17 New Stuff)

#### [Fold Expression](https://en.cppreference.com/w/cpp/language/fold)

- Syntax
  - `( pack op ... )`: unary right fold
  - `( ... op pack )`: unary left fold
  - `( pack op ... op init )`: binary right fold
  - `( init op ... op pack )`: binary left fold
  - `op`: any of the following 32 binary operators. In a binary fold, both ops must be the same.
  - `pack`: an expression that contains an unexpanded parameter pack and does not contain a cast-expression. 
  - `init`:	an expression that does not contain an unexpanded parameter pack and does not contain a cast-expression.
  - Note that the opening and closing parentheses are a required part of the fold expression. 
- Explanation
  1. Unary right fold `(E op ...)` becomes `(E_1 op (... op (E_{N-1} op E_N)))`
  2. Unary left fold `(... op E)` becomes `(((E_1 op E_2) op ...) op E_N)`
  3. Binary right fold `(E op ... op I)` becomes `(E_1 op (... op (E_{N−1} op (E_N op I))))`
  4. Binary left fold `(I op ... op E)` becomes `((((I op E_1) op E_2) op ...) op E_N)`
- When a unary fold is used with a pack expansion of length zero, only the following operators are allowed:
  1. Logical AND (`&&`). The value for the empty pack is `true`
  2. Logical OR (`||`). The value for the empty pack is `false`
  3. The comma operator (`,`). The value for the empty pack is `void()`
- Note: 
  - If the expression used as init or as pack has an operator with precedence below cast at the top level, 
    it must be parenthesized:
  ```c++
  template <typename ... Args>
  int sum(Args ... args)
  {
      // return (args + ... + 1 * 2);  // Error: operator with precedence below cast
      return (args + ... + (1 * 2));   // OK
  }
  ```
- Example
```c++
#include <iostream>
#include <vector>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>
 
template <typename ... Args>
void printer(Args && ... args)
{
    (std::cout << ... << args) << '\n';
}
 
template <typename T, typename ... Args>
void push_back_vec(std::vector<T> & v, Args && ... args)
{
    static_assert((std::is_constructible_v<T, Args &&> && ...));
    (v.push_back(std::forward<Args>(args)), ...);
}
 
// compile-time endianness swap based on http://stackoverflow.com/a/36937049 
template <class T, std::size_t ... N>
constexpr T bswap_impl(T i, std::index_sequence<N...>)
{
    return (((i >> N * CHAR_BIT & std::uint8_t(-1)) << (sizeof(T) - 1 - N) * CHAR_BIT) | ...);
}
 
template <class T, class U = std::make_unsigned_t<T>>
constexpr U bswap(T i)
{
    return bswap_impl<U>(i, std::make_index_sequence<sizeof(T)>{});
}
 
int main()
{
    printer(1, 2, 3, "abc");
 
    std::vector<int> v;
    push_back_vec(v, 6, 2, 45, 12);
    push_back_vec(v, 1, 2, 9);
    for (const auto i : v) std::cout << i << ' ';
 
    static_assert(bswap<std::uint16_t>(0x1234u) == 0x3412u);
    static_assert(bswap<std::uint64_t>(0x0123456789abcdefULL) == 0xefcdab8967452301ULL);
    
    return EXIT_SUCCESS;
}

// Output: 
// 123abc
// 6 2 45 12 1 2 9
```

#### [`if`](https://en.cppreference.com/w/cpp/language/if)

- `if` syntax
```
attr(optional) if constexpr(optional) ( init-statement(optional) condition )
    statement-true 
else 
    statement-false
```
- `if` statements with initializer (also for [`switch`](https://en.cppreference.com/w/cpp/language/switch))
  - If `init-statement` is used, the `if` statement is equivalent to
  ```c++
  {
      init_statement
      attr(optional) if constexpr(optional) ( condition )
          statement-true
      else
          statement-false
  }
  ```
  - Except that names declared by the `init-statement` or `condition` 
    are in the same scope of `statement-true` and `statement-false`.
```c++
std::map<int, std::string> m;
std::mutex mx;
extern bool shared_flag;  // guarded by mx

if (auto it = m.find(10); it != m.end()) { return it->second.size(); }
if (char buf[10]; std::fgets(buf, 10, stdin)) { m[0] += buf; }
if (std::lock_guard lock(mx); shared_flag) { unsafe_ping(); shared_flag = false; }
if (int s; int count = ReadBytesWithSignal(&s)) { publish(count); raise(s); }

if (const auto keywords = {"if", "for", "while"};
    std::ranges::any_of(keywords, [&tok](const char* kw) { return tok == kw; })) 
{
    std::cerr << "Token must not be a keyword\n";
}
```
- `constexpr if`
  - The statement that begins with `if constexpr` is known as the `constexpr if` statement.
  - In a `constexpr if` statement, the value of `condition` must be 
    an expression contextually converted to `bool`, where the conversion is a constant expression `(since C++23)`. 
    If the `value` is `true`, then `statement-false` is discarded (if present), 
    otherwise, `statement-true` is discarded.
  - The `return` statements in a discarded `statement` 
    do **not** participate in function return type deduction:
  ```c++
  // Something non-achievable without dispatchers previously
  template <typename T>
  auto get_value(T t) 
  {
      if constexpr (std::is_pointer_v<T>)
          return *t;  // deduces return type to int for T = int*
      else
          return t;   // deduces return type to int for T = int
  }
  ```
  - The discarded statement can [odr-use](https://en.cppreference.com/w/cpp/language/definition#One_Definition_Rule) 
    a variable that is `not` defined
  ```c++
  extern int x; // no definition of x required
  int f() 
  {
      if constexpr (true)
          return 0;
      else if (x)
          return x;
      else
          return -x;
  }
  ```

#### [Structured Bindings](https://en.cppreference.com/w/cpp/language/structured_binding)

- Binding to arrays, tuple-like types and class data members
```c++
int a[2] = {1, 2};
 
auto [x, y] = a;      // creates e[2], copies a into e, then x refers to e[0], y refers to e[1]
auto & [xr, yr] = a;  // xr refers to a[0], yr refers to a[1]

float x{};
char  y{};
int   z{};
 
std::tuple<float &, char &&, int> tup(x, std::move(y), z);

// a names a structured binding that refers to x; decltype(a) is float &
// b names a structured binding that refers to y; decltype(b) is char &&
// c names a structured binding that refers to the 3rd element of tup; decltype(c) is const int
const auto & [a, b, c] = tup;

struct S 
{
    mutable int x1 : 2;
    volatile double y1;
};

S f() { return S{1, 2.3}; }

const auto [x, y] = f();             // x is an int lvalue identifying the 2-bit bit field
                                     // y is a const volatile double lvalue
std::cout << x << ' ' << y << '\n';  // 1 2.3
x = -2;                              // OK
//  y = -2.;                         // Error: y is const-qualified
std::cout << x << ' ' << y << '\n';  // -2 2.3
```

#### [`inline` Specifier](https://en.cppreference.com/w/cpp/language/inline)

- Inline Functions
  - The `inline` specifier, when used in a function's `decl-specifier-seq`, 
    declares the function to be an `inline` function.
  - A function defined entirely inside a `class`/`struct`/`union` definition, 
    whether it's a member function or a non-member friend function, 
    is implicitly an `inline` function if it is attached to the global module `(since C++20)`.
  - A function declared `constexpr` is implicitly an `inline` function.
  - A `delete`d function is implicitly an `inline` function: 
    its (`delete`d) definition can appear in more than one translation unit. 
- Inline Variables
  - The `inline` specifier, when used in a `decl-specifier-seq` of a variable with `static` storage duration 
    (`static` `class` member or namespace-scope variable), 
    declares the variable to be an `inline` variable.
  - A static member variable (but **not** a namespace-scope variable) declared `constexpr` 
    is implicitly an `inline` variable.
- Explanation
  - An `inline` function or variable has the following properties:
    1. The definition of an `inline` function or variable must be reachable 
       in the translation unit where it is accessed (not necessarily before the point of access). 
    2. An `inline` function or variable with `external` linkage (e.g. **not** `static`) 
       has the following additional properties:
       1. There may be more than one definition of an `inline` function or variable in the program 
          as long as each definition appears in a different translation unit, 
          and for non-`static` inline functions and variables, all definitions are identical. 
          For example, an `inline` function or variable 
          may be defined in a header file that is `#include`'d in multiple source files. 
       2. It must be declared `inline` in every translation unit. 
       3. It has the same address in every translation unit. 
  - In an `inline` function,
    - Function-local `static` objects in all function definitions 
      are shared across all translation units 
      (they all refer to the same object defined in one translation unit); 
    - Types defined in all function definitions are also the same in all translation units.
  - `inline` `const` variables at namespace scope have external linkage by default 
    (unlike the non-`inline` non-`cv`-qualified variables). 
  - `inline` variables eliminate the main obstacle to packaging C++ code as header-only libraries. 
```c++
/// "example.h"
#ifndef EXAMPLE_H
#define EXAMPLE_H
 
#include <atomic>
 
// function included in multiple source files must be inline
inline int sum(int a, int b)
{
    return a + b;
}
 
// variable with external linkage included in multiple source files must be inline
inline std::atomic<int> counter(0);
 
#endif

/// "1.cpp"
#include "example.h"

int a()
{
    ++counter;
    return sum(1, 2);
}

/// "2.cpp"
#include "example.h"

int a() // yet another function with name `a`
{
    ++counter;
    return sum(3, 4);
}

int b()
{
    ++counter;
    return sum(5, 6);
}
```

- [Execution Policies](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)

Tested on ubuntu 20.04 with gcc9. Needs TBB to work. 
```c++
#include <execution>

std::vector<int> v;
v.reserve(1000);
for (int i = 0; i != 1000; ++i) v.emplace_back(i);
std::mutex mutex;

std::for_each(std::execution::par, v.begin(), v.end(), [&mutex = mutex](int & i)
{
    i += 1000;
    std::lock_guard g(mutex);
    std::cout << i << '\n';
});

tbb::parallel_for(v.begin(), v.end(), [&mutex = mutex](int & i)
{
    i += 1000;
    std::lock_guard g(mutex);
    std::cout << i << '\n';
});
```






### 🌱 [Chap 6] 函数

#### 实参和形参

- *实参* （argument）
- *形参* （parameter）：
    - 如果实参是左值，则形参由实参 *拷贝构造* 
    - 如果实参是右值，则形参由实参 *移动构造* 

#### 函数返回值

```
std::string foo(const std::string & word)
{
    return word;  // 生成一个word的副本（copy），返回之。这里有一次拷贝的性能损失
}
```

- **不要**返回局部对象的引用（左值右值都不行）或者指针
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

#### 函数重载（Overloaded functions）

- 同一作用域中几个函数名字相同而形参列表不同，我们称之为 *重载函数* 
    - 重载函数一般形参类型不同，但执行的操作非常相似
    - `main`函数**不能**重载
```
void print(const char * cp);
void print(const int * beg, const int * end);
void print(const int ia[], size_t size);
```
- 调用时，编译器会根据 *传递的实参的类型* 推断调用哪个版本
    - 可能允许 *隐式类型转换* ，比如把非常量转化成常量
```
int j[2]{0, 1};
print("Hello World");                     // calls print(const char *)
print(j, std::end(j) - std::begin(j));    // calls print(const int *, size_t)
print(std::begin(j), std::end(j));        // calls print(const int *, const int *)
```
- 重载规则
    - 重载的函数的 *形参列表* *必须* 有所不同
        - 形参总数
        - 某位置的形参类型
    - **不允许** 两个函数除了返回值类型以外的所有要素都相同
    ```
    Record lookup(const Account&);
    bool lookup(const Account&);          // error: only the return type is different
    ```
- 如何判断形参类型是否相异
    - 省略形参名字的
    ```
    // each pair declares the same function
    Record lookup(const Account & acct);
    Record lookup(const Account &);       // parameter names are ignored
    ```
    - `typedef`或类型别名撞车的
    ```
    typedef Phone Telno;
    Record lookup(const Phone &);
    Record lookup(const Telno &);         // Telno and Phone are the same type
    ```
    - 值传递和引用传递同时存在的
        - 重载本身合法，但高危存在 *二义性调用*
        - 要是 *值传递* 、 *左值引用传递* 和 *右值引用传递* 同时存在，那可真是死绝了
    ```
    void f(int a)
    {
        printf("f(int)\n");
    }

    void f(int & a)
    {
        printf("f(int &)\n");
    }
    
    void f(int && a)
    {
        printf("f(int &&)\n");
    }
    
    int i = 0;
    f(i);             // error: ambiguous call
    f(0);             // error: ambiguous call
    f(std::move(a));  // error: ambiguous call
    ```
    - 顶层`const`**不影响**传入的对象，因此以下定义不合法
    ```
    Record lookup(Phone);
    Record lookup(const Phone);           // redeclares Record lookup(Phone)
    
    Record lookup(Phone *);
    Record lookup(Phone * const);         // redeclares Record lookup(Phone *)
    ```
    - 可以基于底层`const`重载函数
        - 可以通过 *实参* 是否是常量来决定调用哪个版本
            - *常量* **不能**隐式转换成 *非常量* ，因此只能调用普通引用版本
            - *非常量* 可以隐式转化成 *常量* ，因此理论上哪个都行
                - 不过编译器会 *优先选用非常量形参的版本* 
    ```
    // functions taking const and nonconst references or pointers have different parameters
    // declarations for four independent, overloaded functions
    Record lookup(Account &);        // function that takes a reference to Account
    Record lookup(const Account &);  // new function that takes a const reference
    
    Record lookup(Account *);        // new function, takes a pointer to Account
    Record lookup(const Account *);  // new function, takes a pointer to const
    ```
    - 类成员函数的 *`const`限定* 和 *引用限定* 均可用于区分重载函数 => 13.6.3
        - 原理：编译器可以根据`this`指针参数的底层`const`，或者 *值类别* 区分参数类型
- 重载和作用域
    - 在不同的作用域中**无法**重载函数
    - **不要**在 *块作用域* 内声明或定义函数，容易覆盖外层作用域中的同名 *实体* （ *对象* ， *函数* 等等都可能覆盖）
        - `C++`中， *名字查找* 先于 *类型匹配* 
            - 编译器首先查找对 *名字* 的声明，一旦找到，就会 *忽略掉* 外部作用域的 *同名实体*
            - 之后才会看声明的类型与调用时的实际类型是否匹配，如果不匹配， *直接报错* 
    ```
    string read();
    void print(const string &);
    void print(double);          // overloads the print function
    
    void fooBar(int ival)
    {
        bool read = false;       // new scope: hides the outer declaration of read
        string s = read();       // error: read is a bool variable, not a function
                                 // bad practice: usually it's a bad idea to declare functions at local scope
        
        void print(int);         // new scope: hides previous instances of print
        print("Value: ");        // error: print(const string &) is hidden
        print(ival);             // ok: print(int) is visible
        print(3.14);             // ok: calls print(int); print(double) is hidden
    }
    
    void print(int);             // another overloaded instance
    
    void fooBar2(int ival)
    {
        print("Value: ");        // calls print(const string &)
        print(ival);             // calls print(int)
        print(3.14);             // calls print(double)
    }
    ```

#### [重载确定/重载决议](https://en.cppreference.com/w/cpp/language/overload_resolution)（overload resolution）

- 又称 *函数匹配* （function matching）
- 编译器首先将调用的实参与重载集合中每一个函数的形参进行比较，然后根据结果确定调用版本
- 函数匹配过程的三种可能结果
    1. 找到 *最佳匹配* （best match）
    2. *无匹配* （no match）
    3. 多个函数都可调用且无明显最佳选择： *二义性调用* （ambiguous call）
- 函数匹配流程
    1. 确定 *候选函数* （candidate functions）
        - 确定本次调用对应的 *重载函数集* ，集合中的函数就是 *候选函数* ，具备两个特征
            1. 与被调用的函数 *同名* 
            2. 声明在调用点 *可见* 
    2. 选出 *可行函数* （viable functions）
        - 考察调用提供的 *实参* ，从 *候选函数* 中选出能被这组 *实参* 调用的函数（ *可行函数* ），具备两个特征
            1. 形参 *数量* 与本次调用提供的实参数量 *相等* 
                - 如果函数有 *默认实参* ，则传入的实参 *数量* 可能 *少于* 实际使用的实参数量
            2. 每个实参的 *类型* 都与对应的形参类型 *相同* ，或 *能隐式转换* 成该类型
        - 如果没有 *可行函数* ： *无匹配* 
    3. 寻找 *最佳匹配* （best match）（best viable function）
        - 如果有且仅有一个函数同时满足下列两个条件，则匹配成功
            1. 该函数的每个实参的匹配都不劣于其他可行函数需要的匹配
            2. 该函数的至少有一个实参的匹配优于其他可行函数需要的匹配
        - 如果没有 *最佳匹配* ： *二义性调用* 
        - 调用重载函数时应当尽量 *避免强制类型转换* ，如果确实需要，这说明形参设计不合理
        - 实参类型转换
            - 为确定最佳匹配，编译器将实参类型到形参类型的转换划分成以下几个等级，具体排序如下
                1. 精确匹配，包括
                    - 实参和形参类型相同
                    - 实参为数组或函数，转指针
                    - 给实参添加或删除 *顶层`const`* 
                2. 通过`const_cast`（添加或删除 *底层`const`* ）实现匹配
                3. 通过 *类型提升* 实现匹配
                4. 通过 *算数类型转换* 或 *指针转换* 实现匹配
                5. 通过 *用户定义转换* (user-defined conversion) 实现匹配 => 14.9
            - 函数匹配和`const`实参
                - 实参的 *顶层`const`* *被忽略*
                - 如果重载函数的区别在于 *底层`const`* 
                    - *底层`const`* 
                        - 引用类型的形参是否引用了`const`
                        - 指针类型的形参是否指向`const`
                        - 底层`const`直接就是二义性调用
                    - 通过 *实参是否是常量* 来选择
                ```
                Record lookup(Account &);        // function that takes a reference to Account
                Record lookup(const Account &);  // new function that takes a const reference
                const Account a;
                Account b;
                lookup(a);                       // calls lookup(const Account &)
                lookup(b);                       // calls lookup(Account &)
                ```
            - 左右值引用：和底层`const`同理

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

#### [合成的默认构造函数](https://en.cppreference.com/w/cpp/language/default_constructor)（Synthesized default constructor）

- 什么是 *默认构造函数* 
    - 没有任何形参的构造函数
- 什么是 *合成的默认构造函数* 
    - 如果用户没有显式定义任何构造函数，则编译器在满足 *生成条件* 时，会自动 *隐式定义* 一个 *合成的默认构造函数* 
    - 如果用户定义了构造函数，可以使用`= default;`显式要求编译器生成一个  *合成的默认构造函数* 
        - `= default;`用于既定义了自己的构造函数，又需要默认构造函数的情况
        - 作为声明写在类内部，则构造函数默认`inline`；或作为定义写在类外部，则构造函数不`inline`
- 合成的默认构造函数按如下规则初始化类成员
    - 存在类内初始值，则以其初始化对应成员
        - 类内初始值可接受的语法
        ```
        int a1 = 0;    // 正确
        int a2 = {0};  // 正确
        int a3{0};     // 正确
        int a4(0);     // 错误！
        ```
    - 否则，执行 *默认初始化* 或者 *值初始化* 
- 合成的默认构造函数的生成条件
    - 只有类**没有声明任何构造函数**时，编译器才会自动生成默认构造函数
    - 如果类中包含其他类类型成员，且它没有默认构造函数，则这个类**不能**生成默认构造函数
    - => 13.1.6
- 如果类内包含内置类型或复合类型的变量，则只有当这些成员全部被赋予了类内初始值时，这个类才适合于使用默认构造函数
    - 注意：类成员变量从属于内部作用域，默认初始化是 *未定义* 的，不能指望！
- 类必须包含默认构造函数以便在上述情况下使用。实际应用中，如果提供了其他构造函数，最好也提供一个默认构造函数
    - 比如 *值初始化* 时，对于有合成的默认构造函数的类类型，会首先对所有数据成员零初始化
    - 如果没有合成的默认构造函数，则直接 *默认初始化* ，很多情况下就是什么也不做，这将产生 *未定义的值* 

#### [初始化列表](https://en.cppreference.com/w/cpp/language/constructor)（Member initializer lists）

- 又称 *成员初始化器列表* 
- 初始化器列表接受的初始化语法
    1. `Constructor() : x(?), ... { }`
    2. `Constructor() : x{?}, ... { }`
- 如果成员是`const`、 *引用* 或者 *没有默认构造函数的类类型* ，如没有类内初始值，则 *必须* 在初始化列表中初始化，而**不能**等到函数体中赋值
- 初始化的 *顺序是按照类成员被声明的顺序* ，与其在列表中的顺序**无关**
    - 最好令构造函数初始化列表的顺序与成员声明的顺序保持一致
    - 尽量避免用某些成员初始化其他成员，最好用构造函数的参数作为初始值
- 如果一个构造函数为每一个参数都提供了 *默认实参* ，则它实际上也定义了 *默认构造函数* 
- 某个数据成员被初始化器列表忽略时，则先被 *默认初始化* ，之后再按照构造函数体中的规则进行 *二次赋值*

#### 委托构造函数（delegating constructor）

- 一个委托构造函数使用它所属类的其他构造函数执行它自己的初始化过程，
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

#### 显式构造函数（[`explicit`](https://en.cppreference.com/w/cpp/language/explicit) constructor）

- 我们可以通过将构造函数声明为`explicit`来抑制构造函数定义的隐式转换
    - `Clang Tidy`直接规定只有一个实参的构造函数必须是`explicit`的
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

- *拷贝构造函数* 、 *移动构造函数* => 13
- *继承的构造函数* => 15.7
- *函数`try`语句块* => 18.1.3

#### 友元

- 友元**不是**类的成员，**不**受`public`、`private`以及`protected`这些访问限制的约束
    - 也就是说，在类的`public`、`protected`或`private`区域声明友元效果完全**没有**区别
- 友元**不具有**传递性。每个类**单独**负责控制自己的友元类或友元函数
    - `C`有友元`B`、`B`有友元`A`，则`A`能访问`B`的私有成员，但不能访问`C`的私有成员
- 在类定义开始或结束的地方**集中声明**友元
- *友元函数*
    - 友元函数的声明仅仅是指定访问权限，并不是真正的函数声明。想要使用友元，仍**另需一单独的函数声明**（重载运算符友元除外，**不需**单独声明）
    - 对于重载函数，必须对特定的函数（特有的参数列表）单独声明
- *友元类*
    - 令一个类成为友元
- *友元成员函数*
    - 令一个类的某个成员函数成为友元
      - *友元声明和作用域*
          - 关于这段代码最重要的是：理解友元声明的作用是**影响访问权限**，它本身**并非**普通意义上的函数声明（仍旧是声明，但不自带可见性）
            - [cppreference关于这一点的具体解释](https://en.cppreference.com/w/cpp/language/namespace)：
              Names introduced by friend declarations within a non-local class X become members of the innermost enclosing namespace of X, 
              but they do not become visible to ordinary name lookup (neither unqualified nor qualified) 
              unless a matching declaration is provided at namespace scope, 
              either before or after the class definition. 
              Such name may be found through ADL which considers both namespaces and classes.
- 并不是所有编译器都强制执行关于友元的这一规定
```c++
struct X
{
    friend void f()
    {
        // Friend functions can be defined in the class. 
        // This declaration provides NO visibility to regular name lookup, 
        // even though this is already a definition. 
        // To use this function, another declaration is REQUIRED
    }

    friend X operator+(const X & x1, const X & x2)
    {
        // This declaration also provides NO visibility to regular name lookup, 
        // but can be found via ADL
        return {x1.v + x2.v};
    }

    void foo()
    {
        f();                    // ERROR: no declaration for f found
        X tmp = X {1} + X {2};  // CORRECT  
    }

    void g();
    void h();

    int v;
};

void X::g()
{
    return f();  // ERROR: no declaration for f found
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
    
| 数据成员            | 普通成员函数          | `const`成员函数      |
|--------------------|--------------------|----------------------|
| 普通数据成员        | 可引用，可修改         | 可引用，**不可**修改  | 
| 常数据成员          | 可引用，**不可**修改   | 可引用，**不可**修改  |
| *常对象* 的数据成员  | 不允许               | 可引用，**不可**修改   | 


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
    - 在定义之前，`Item`是 *不完整类型* （incomplete type）
    - *不完整类型* 使用受限
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
        - *枚举类型* => 19.3
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
            - `typedef`或类型别名
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
    - `static`关键字 *只能出现在类内部的声明语句* ，在类外部定义静态成员时，**不能**重复
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
    - 可以使用类的对象、引用或者指针来访问 *静态成员* 
        - **不能**访问 *非静态成员* 
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

#### `I/O`类

- `I/O`库头文件和类
    - [`<iostream>`](https://en.cppreference.com/w/cpp/header/iostream)
        - [`std::istream`，`std::wistream`](https://en.cppreference.com/w/cpp/io/basic_istream)：从流读取数据
        - [`std::ostream`，`std::wostream`](https://en.cppreference.com/w/cpp/io/basic_ostream)：向流写入数据
        - [`std::iostream`，`std::wiostream`](https://en.cppreference.com/w/cpp/io/basic_iostream)：读写流
    - [`<fstream>`](https://en.cppreference.com/w/cpp/header/fstream)
        - [`std::ifstream`，`std::wifstream`](https://en.cppreference.com/w/cpp/io/basic_ifstream)：从文件读取数据
        - [`std::ofstream`，`std::wofstream`](https://en.cppreference.com/w/cpp/io/basic_ofstream)：向文件写入数据
        - [`std::fstream`，`std::wfstream`](https://en.cppreference.com/w/cpp/io/basic_fstream)：读写文件
    - [`<sstream>`](https://en.cppreference.com/w/cpp/header/sstream)
        - [`std::istringstream`、`std::wistringstream`](https://en.cppreference.com/w/cpp/io/basic_istringstream)：从`std::string`读取数据
        - [`std::ostringstream`、`std::wostringstream`](https://en.cppreference.com/w/cpp/io/basic_ostringstream)：从`std::string`写入数据
        - [`std::stringstream`、`std::wstringstream`](https://en.cppreference.com/w/cpp/io/basic_stringstream)：读写`std::string`
    - 继承关系
        - ![](http://upload.cppreference.com/mwiki/images/0/06/std-io-complete-inheritance.svg)
- 为了支持使用 *宽字符* 的语言，标准库定义了一组类型和对象来操纵`wchar_t`类型的数据
    - 宽字符版本的类型和函数的名字以一个`w`开始
    - 例如：`std::wcin`、`std::wcout`和`std::wcerr`分别是`std::cin`、`std::cout`和`std::cerr`的宽字符版对象
    - 宽字符版本的类型和函数与其对应的普通`char`版本的定义于同一个文件中
- `I/O`对象**无** *拷贝* 或 *赋值*
    - **不能** *拷贝* 或 *赋值* `I/O`对象
    - `I/O`对象**不能**被设为函数 *形参类型* 或 *返回值类型* 
```
std::ofstream out1, out2;
out1 = out2;                    // error: cannot assign stream objects
std::ofstream print(ofstream);  // error: can't initialize the ofstream parameter
out2 = print(out2);             // error: cannot copy stream objects
```
- *条件状态* （conditional states）
    - `C++ I/O`库定义了`4`个与机器无关的`stream::iostate`类型的`constexpr`值，代表流对象的 *条件状态* 
        - `gcc`实现
        ```
        // <bits/ios_base.h>
        // class ios_base
        
        public: 
            enum _Ios_Iostate
            { 
                _S_goodbit         = 0,
                _S_badbit          = 1L << 0,
                _S_eofbit          = 1L << 1,
                _S_failbit         = 1L << 2,
                _S_ios_iostate_end = 1L << 16,
                _S_ios_iostate_max = __INT_MAX__,
                _S_ios_iostate_min = ~__INT_MAX__
            };
            
            typedef _Ios_Iostate iostate;
            static const iostate badbit  = _S_badbit;
            static const iostate eofbit  = _S_eofbit;
            static const iostate failbit = _S_failbit;
            static const iostate goodbit = _S_goodbit;
        
        protected: 
            iostate _M_exception;
            iostate _M_streambuf_state;
        ```
        - 其中
            - `stream::iostate`：`stream`是一种`I/O`类型，`iostate`是一种机器相关的类型，提供了表达条件状态的完整功能
            - `stream::badbit`： *系统级错误* ，如不可恢复的读写错误
                - 通常情况下，一旦`badbit`被置位，就意味着流已经彻底崩溃、无法再使用了
            - `stream::failbit`： *可恢复的错误* ，如期望读取数值缺读到了字符，或已到达文件末尾
                - 通常可以修正，流还可以继续使用
            - `stream::eofbit`： *已到达文件末尾* ，此时`eofbit`和`failbit`都会被置位
            - `stream::goodbit`： *流有效* ，`badbit`、`failbit`和`eofbit`都**未**被置位
    - 查询条件状态
        - `s.rdstate()`：返回流`s`当前的条件状态，返回值类型为`stream::iostate`
        - `s.good()`：若流`s`处于 *有效状态* ，即所有错误位均**未**置位时，则返回`true`
        - `s.eof()`：若流`s`的`eofbit`置位，则返回`true`
        - `s.fail()`：若流`s`处于 *无效状态* ，即`badbit`或`failbit`置位时，则返回`true`
        - `s.bad()`：若流`s`的`badbit`置位，则返回`true`
        - 把流作为 *条件* 使用时，如果`badbit`、`failbit`都**未**被置位，则返回`true`；否则，返回`false`
            - 表达式被用作 *条件* 时， *类型转换运算符* 即使是`explicit`的，仍会被 *隐式应用* => 14.9.1
        ```
        // <bits/basic_ios.h>
        // class basic_ios : public ios_base
        
        public:
            explicit operator bool() const { return !this->fail(); }
            bool     operator!()     const { return this->fail(); }
            
            iostate rdstate() const { return _M_streambuf_state; }
            bool    good()    const { return this->rdstate() == 0; }
            bool    eof()     const { return (this->rdstate() & eofbit) != 0; }
            bool    fail()    const { return (this->rdstate() & (badbit | failbit)) != 0; }
            bool    bad()     const { return (this->rdstate() & badbit) != 0; }
        ```
    - 管理条件状态
        - `s.clear()`：将流`s`中所有条件状态位 *复位* ，将流的状态设置为有效，返回`void`
        - `s.clear(flags)`：将流`s`中对应条件状态位设置为`flags`。`flags`的类型为`stream::iostate`
        ```
        void clear(std::ios_base::iostate flags = std::ios_base::goodbit);
        {
            // By default, assigns std::ios_base::goodbit 
            // which has the effect of clearing all error state flags. 
        
            if (this->rdbuf())
            {
                _M_streambuf_state = flags;
            } 
            else
            {
                // there is no associated stream buffer
                _M_streambuf_state = flags | badbit;
            }
                
            if (this->exceptions() & this->rdstate())
            {
                __throw_ios_failure("basic_ios::clear");
            }
        }
        ```
        - `s.setstate(flags)`：根据给定的`flags`中的`1`，将流`s`中对应条件状态位 *置位* 。`flags`的类型为`stream::iostate`
        ```
        void setstate(iostate flags) { this->clear(this->rdstate() | flags); }
        ```
        - 使用示例
        ```
        // remember the current state of cin
        std::istream::iostate old_state = std::cin.rdstate();  // remember the current state of std::cin
        std::cin.clear();                                      // make std::cin valid
        process_input(std::cin);                               // use std::cin
        std::cin.setstate(old_state);                          // now reset std::cin to its old state

        // turns off failbit and badbit but all other bits unchanged
        std::cin.clear(std::cin.rdstate() & ~std::cin.failbit & ~std::cin.badbit);
        ```
    - `I/O`错误案例分析
        - 错误是任何语言任何`I/O`操作的特色，不能不品尝。考虑如下代码
        ```
        int ival;
        std::cin >> ival;
        ```
        - 在标准输入中键入`Boo`，读操作就会失败
            - `std::cin::operator>>`期待一个`int`，却得到了`char 'B'`
            - 此时`std::cin`就会进入 *错误状态* 
            - 类似地，键入 *文件结束标识* ，也会导致`std::cin`进入错误状态
        - 一旦一个流发生错误，其上后续的所有`I/O`操作都会失败
            - 只有当一个流处于 *无错状态* 时，才可以从它读取或写入数据
            - 由于流可能处于错误状态，因此代码应该在使用流之前检查它是否处于良好状态
            - 确认一个流对象状态的最好办法就是将它作为一个 *条件* 来使用
                - 如果`I/O`操作成功，则流保持有效状态，则条件为`true`
            ```
            while (std::cin >> word)
            {
                // ok: read operation successful...
            }
            ```
- *输出缓冲* （Output Buffer）
    - 每个输出流都管理一个 *缓冲区* （buffer），用来保存程序读写的数据
        - 例如，执行`std::cout << "Hello World!\n";`时，文本串
            - 可能 *立即被打印* 出来
            - 也可能被操作系统保存于 *缓冲区* 中， *随后再打印* 
        - 便于操作系统可以将程序的多个输出组合成单一的系统级`I/O`操作，可以提升性能
    - 导致 *缓冲刷新* （buffer flushing，即数据真正被写到输出设备或文件）的原因有很多
        - 程序正常结束
            - 作为主函数的返回操作的一部分，缓冲区被刷新
        - 缓冲区满
            - 需要刷新缓冲区，之后新数据才能继续被写入缓冲区
        - 显式刷新
            - 使用`std::endl`
            - 每个输出操作之后，我们可以使用操作符`unitbuf`设置流的内部状态，来清空缓冲区
                - 默认情况下，对`std::cerr`的设置时`unitbuf`的
                - 即：通过`std::cerr`写入到`stderr`的内容都是立即刷新的
        - 一个输出流可能 *被关联到* 另一个流。此时，读写 *被关联* 的流时， *关联到* 的流的缓冲区会被刷新
            - 例如，默认情况下，`std::cin`和`std::cerr`都被关联到`std::cout`
            - 此时，读`std::cin`或写`std::cerr`都会导致`std::cout`的缓冲区被刷新
    - *警告* ：程序崩溃时，输出缓冲区**不会**刷新
    - 刷新输出缓冲区
        - `std::endl`：输出一个 *换行符* `'\n'`，并刷新缓冲区
        - `std::flush`：**不**输出任何额外字符，刷新缓冲区
        - `std::ends`：输出一个 *空字符* ，并刷新缓冲区
    ```
    std::cout << "hi!" << std::endl;   // writes hi and a newline, then flushes the buffer
    std::cout << "hi!" << std::flush;  // writes hi, then flushes the buffer; adds no data
    std::cout << "hi!" << std::ends;   // writes hi and a null, then flushes the buffer
    ```
    - `unitbuf`和`nounitbuf`操纵符（`unitbuf` & `nounitbuf`Manipulator）
        - 如果想每次输出操作之后都立即刷新缓冲区，可以使用`unitbuf`操纵符
            - 它告诉流对象：接下来每次写操作之后都进行一次`flush`操作
        - 而`nounitbuf`则重置流对象，使其恢复使用正常的系统管理的缓冲区刷新机制
    ```
    std::cout << unitbuf;              // all writes will be flushed immediately
    // any output is flushed immediately, no buffering
    std::cout << nounitbuf;            // returns to normal buffering
    ```
    - *关联* 输入和输出流（Tying Input and Output Streams Together）
        - 当一个 *输入或输出流* *被关联到* 一个 *输出流* 时，任何试图 *读写被关联的流* 操作都会 *先刷新关联的输出流* 
            - 既可以将一个`std::istream`对象关联到另一个`std::ostream`上
            - 也可以将一个`std::ostream`对象关联到另一个`std::ostream`上
        - 标准库将`std::cin` *关联到* `std::cout`上
            - 即`std::cin >> ival;`将导致`std::cout`
        - *交互式* 系统通常应该 *关联输入流和输出流* ，这意味着所有输出，包括用户提示信息，都会在读操作之前被打印出来
        - `s.tie`有 *两个* 重载的版本
            1. 不带参数，返回指向输出流的指针
                - 如果本对象当前被关联到一个输出流，则返回的就是指向这个流的指针
                - 如果未关联到流，则返回 *空指针* 
            2. 接受一个指向`std::ostream`的指针，将自己关联到此`ostream`上
                - 即`x.tie(&o)`将流`x`关联到输出流`o`上
        ```
        // illustration only: 
        // the library ties std::cin and std::cout for us
        std::cin.tie(&std::cout);                       
        
        // old_tie points to the stream (if any) currently tied to std::cin
        std::ostream * old_tie = std::cin.tie(nullptr);  // std::cin is no longer tied
        
        // ties cin and cerr; 
        // not a good idea because cin should be tied to cout
        std::cin.tie(&std::cerr);                        // reading std::cin flushes cerr, not std::cout
        std::cin.tie(old_tie);                           // reestablish normal tie between std::cin and std::cout
        ```

#### 文件`I/O`

- 头文件`<fstream>`定义了三个 *文件流* 类型来支持文件`I/O`
    - `std::ifstream`从一个给定文件读取数据
    - `std::ofstream`向一个给定文件写入数据
    - `std::fstream`可以读写给定文件
- `std::fstream`特有的操作
    - `std::fstream fs;`：创建一个 *未绑定* 的 *文件流对象*
    - `std::fstream fs(file);`：创建文件流对象`fs`并绑定到文件`file`上。`file`可以是`std::string`或`C`风格字符串。默认 *文件模式* 依赖于`std::fstream`的类型。此构造函数为`explicit`的
    - `std::fstream fs(file, mode);`：创建文件流对象`fs`并以`mode`指定的方式打开文件`file`。`file`可以是`std::string`或`C`风格字符串。此构造函数为`explicit`的
    - `fs.open(file);`：打开文件`file`并将之与`fs`绑定。`file`可以是`std::string`或`C`风格字符串。默认 *文件模式* 依赖于`std::fstream`的类型。返回`void`
    - `fs.close();`：关闭与`fs`绑定的文件，返回`void`
    - `fs.is_open();`：返回一个`bool`，指出与`fs`关联的文件是否成功打开且尚未关闭
- 使用 *文件流对象* 
    - `std::fstream`继承自`std::iostream`，派生类对象可以作为基类使用
    - 创建文件流对象时，可以提供 *文件名* （可选）
        - 如提供，则`open`自动被调用
        ```
        std::ifstream fin(ifile);    // construct an ifstream and open the given file
        std::ofstream fout;          // output file stream that is not associated with any file
        ```
        - 如不提供，可以随后调用`fs.open(file)`将它与文件关联起来
        ```
        std::ifstream fin(ifile);    // construct an ifstreamand open the given file
        std::ofstream fout;          // output file stream that is not associated with any file
        fout.open(ifile + ".copy");  // open the specified file
        ```
        - 如果`open` *成功* ，则流状态会被设置为`fs.good() == true`
        - 如果`open`调用 *失败* ，则`failbit`会被置位
            - 随后的读写操作均会失败
            - 由于调用`open`可能会失败，应该进行检测
            ```
            if (fout)  // check that the open succeeded
            {
                // the open succeeded, so we can use the file
            }
            ```
    - 一旦一个文件流已经打开，它就保持与对应文件的关联
        - 为了将文件流关联到另外一个文件，必须首先关闭已经关联的文件
        - 一旦文件成功关闭，我们可以打开新文件
        ```
        fin.close();                 // close the file
        fin.open(ifile + "2");       // open another file
        ```
    - `std::fstream`对象会被 *自动析构* ，析构时会 *自动调用`close`* 
- *文件模式* （file mode）
    - 每个流都有一个关联的 *文件模式* ，用来指出如何使用文件
        - 实现
        ```
        // <bits/ios_base.h>
        // class ios_base
        
        public:
            enum _Ios_Openmode 
            { 
                _S_app              = 1L << 0,
                _S_ate              = 1L << 1,
                _S_bin              = 1L << 2,
                _S_in               = 1L << 3,
                _S_out              = 1L << 4,
                _S_trunc            = 1L << 5,
                _S_ios_openmode_end = 1L << 16,
                _S_ios_openmode_max = __INT_MAX__,
                _S_ios_openmode_min = ~__INT_MAX__
            };
        
        // <fstream>
        // class fstream
        
        public:
            typedef _Ios_Openmode openmode;

            /// Seek to end before each write.
            static const openmode app    = _S_app;

            /// Open and seek to end immediately after opening.
            static const openmode ate    = _S_ate;

            /// Perform input and output in binary mode (as opposed to text mode).
            /// This is probably not what you think it is; see
            /// https://gcc.gnu.org/onlinedocs/libstdc++/manual/fstreams.html#std.io.filestreams.binary
            static const openmode binary = _S_bin;

            /// Open for input.  Default for @c ifstream and fstream.
            static const openmode in     = _S_in;

            /// Open for output.  Default for @c ofstream and fstream.
            static const openmode out    = _S_out;

            /// Open for input.  Default for @c ofstream.
            static const openmode trunc  = _S_trunc;
        ```
        - 其中
            1. `std::fstream::app`： *每次写操作* 前均 *定位到文件末尾* 
            2. `std::fstream::ate`： *打开文件* 后立即 *定位到文件末尾* 
            3. `std::fstream::binary`：以 *二进制 式* 进行`I/O`
            4. `std::fstream::in`：以 *读* 方式打开。`std::ifstream`和`std::fstream`的默认选项
            5. `std::fstream::out`：以 *写* 方式打开。`std::ofstream`和`std::fstream`的默认选项
            6. `std::fstream::trunc`： *截断* 文件。`std::fstream`的默认选项
    - 不论用哪种方式打开文件，我们都可以指定文件模式
        - `fs.open(file, mode);`：显式打开文件
        - `std::fstream fs(file, mode);`：隐式打开文件
    - 指定文件模式有如下限制
        - 只有`std::ofstream`和`std::fstream`对象可以被设定为`out`模式
        - 只有`std::ifstream`和`std::fstream`对象可以被设定为`in`模式
        - 只有当`out`被设定时才可设定`trunc`模式
        - 只要`trunc`没有被设定，就可以设定`app`模式。在`app`模式下，即使没有显示指定`out`，文件也总是以 *写方式* 被打开
        - 默认情况下，即使我们没有指定`trunc`，以`out`方式打开的文件也会 *被截断* 
            - 为了保留以`out`模式打开的文件的内容，我们 *必须* 
                - *同时指定* `app`模式，这样只会将数据追加写到文件末尾；或
                - *同时指定* 指定`in`模式，即打开文件同时进行 *读写* 操作 => 17.5.3
        - `ate`和`binary`模式可用于 *任何* 类型的文件流对象，且可以与其他任何文件模式组合使用
    - 以`out`模式打开文件会丢弃已有数据
        - 默认情况下，当我们打开一个`std::ostream`时，文件的内容会被 *丢弃* 
        - 阻止一个`std::ostream`清空给定文件内容的方法是同时指定`app`模式
    ```
    // file1 is truncated in each of these cases
    std::ofstream fout1("file1");                                   // out and trunc are implicit
    std::ofstream fout2("file1", ofstream::out);                    // trunc is implicit
    std::ofstream fout3("file1", ofstream::out | ofstream::trunc);
    
    // to preserve the file's contents, we must explicitly specify app mode
    std::ofstream fapp1("file2", ofstream::app);                    // out is implicit
    std::ofstream fapp2("file2", ofstream::out | ofstream::app);
    ```
    - 每次调用`open`时都会确定文件模式
        - 可能是显式设置的，也可能是隐式的默认值
        - 对于一个给定流，每当打开文件时，都可以改变其文件模式
        ```
        std::ofstream fout;                         // no file mode is set
        fout.open("scratchpad");                    // mode implicitly out and trunc
        fout.close();                               // close out so we can use it for a different file
        fout.open("precious", std::ofstream::app);  // mode is out and app
        fout.close();
        ```

#### 内存`I/O`

- 头文件`<sstream>`中定义了三个类型来支持 *内存`I/O`* 
    - 这些类型可以读写`std::string`，就像是`std::string`的一个输入输出流一样
    - `std::istringstream`：从`std::string`读取数据
    - `std::ostringstream`：向`std::string`写入数据
    - `std::stringstream`：读写`std::string`
- `std::stringstream`特有的操作
    - `std::stringstream ss;`：`ss`是一个 *未绑定* 的`std::stringstream`对象
    - `std::stringstream ss(s);`：`ss`是一个`std::stringstream`对象，保存`std::string s`的一个拷贝。此构造函数是`explicit`的
    - `ss.str()`：返回`ss`所保存的`std::string`的拷贝
    - `ss.str(s)`：将`std::string s`拷贝到`ss`中。返回`void`
- 使用`std::istringstream`
```
// read file line by line
std::istringstream sin;
sin.str("1\n2\n3\n4\n5\n6\n7\n");
int sum = 0;
std::string line;
while (std::getline(sin, line, '\n')) 
{
    sum += std::stoi(line);
}
std::cout << "\nThe sum is: " << sum << std::endl;  // The sum is 28

```
- 使用`std::ostream`
```
for (const PersonInfo & entry : people) 
{ 
    // for each entry in people
    std::ostringstream formatted, badNums;  // objects created on each loop

    for (const std::string & nums : entry.phones) 
    { 
        // for each number
        if (!valid(nums)) 
        {
            // string in badNums
            badNums << " " << nums;  
        } 
        else
        {
            // "writes" to formatted's string
            formatted << " " << format(nums);
        }
    }

    if (badNums.str().empty())  // there were no bad numbers
    {
        std::cout << entry.name << " "              // print the name
                  << formatted.str() << std::endl;  // and reformatted numbers
    } 
    else  // otherwise, print the name and bad numbers
    {
        std::cerr << "input error: " << entry.name
                  << " invalid number(s) " << badNums.str() << std::endl;
    }
}
```

#### 格式化`I/O`（formatted `I/O`）

- 除了 *条件状态* 之外，每个`std::iostream`对象还维护一个 *格式标记* （format flags）来控制`I/O`如何控制格式化的细节
    - `gcc`实现为[`std::ios_base::fmtflags`](https://en.cppreference.com/w/cpp/io/ios_base/fmtflags)
    ```
    // <bits/ios_base.h>
    // class ios_base
    
    public: 
        enum _Ios_Fmtflags 
        { 
            _S_boolalpha        = 1L << 0,
            _S_dec              = 1L << 1,
            _S_fixed            = 1L << 2,
            _S_hex              = 1L << 3,
            _S_internal         = 1L << 4,
            _S_left             = 1L << 5,
            _S_oct              = 1L << 6,
            _S_right            = 1L << 7,
            _S_scientific       = 1L << 8,
            _S_showbase         = 1L << 9,
            _S_showpoint        = 1L << 10,
            _S_showpos          = 1L << 11,
            _S_skipws           = 1L << 12,
            _S_unitbuf          = 1L << 13,
            _S_uppercase        = 1L << 14,
            _S_adjustfield      = _S_left | _S_right | _S_internal,
            _S_basefield        = _S_dec | _S_oct | _S_hex,
            _S_floatfield       = _S_scientific | _S_fixed,
            _S_ios_fmtflags_end = 1L << 16,
            _S_ios_fmtflags_max = __INT_MAX__,
            _S_ios_fmtflags_min = ~__INT_MAX__
        };
        
        typedef _Ios_Fmtflags fmtflags;
        static const fmtflags boolalpha   = _S_boolalpha;
        static const fmtflags dec         = _S_dec;
        static const fmtflags fixed       = _S_fixed;
        static const fmtflags hex         = _S_hex;
        static const fmtflags internal    = _S_internal;
        static const fmtflags left        = _S_left;
        static const fmtflags oct         = _S_oct;
        static const fmtflags right       = _S_right;
        static const fmtflags scientific  = _S_scientific;
        static const fmtflags showbase    = _S_showbase;
        static const fmtflags showpoint   = _S_showpoint;
        static const fmtflags showpos     = _S_showpos;
        static const fmtflags skipws      = _S_skipws;
        static const fmtflags unitbuf     = _S_unitbuf;
        static const fmtflags uppercase   = _S_uppercase;
        static const fmtflags adjustfield = _S_adjustfield;
        static const fmtflags basefield   = _S_basefield;
        static const fmtflags floatfield  = _S_floatfield;
        
    protected:
        fmtflags _M_flags;
    ```
    - 可以使用`std::ios::fmtflags old_fmt_flags = stream.flags();`来获取流`stream`的 *格式标记* 
- 标准库定义了一组 *操纵符* （manipulator），来 *修改* 流的 *格式标记* 
    - *基础操纵符* ，定义于头文件`<iostream>`中
        - 独立操纵符（打开）
            - `std::boolalpha`：将`bool`输出为`true`和`false`
            - `std::showbase`：对整形值，输出表示进制的前缀
            - `std::showpoint`：对浮点值，总是输出小数点
            - `std::showpos`：对非负数，总是输出`+`
            - `std::uppercase`：在十六进制值中输出`0X`，在科学计数法中输出`E`
        - 独立操纵符（关闭）
            - `std::noboolalpha`：将`bool`输出为`1`和`0`。 *默认状态* 
            - `std::noshowbase`：对整形值， *不* 输出表示进制的前缀。 *默认状态* 
            - `std::noshowpoint`：对浮点值，当且仅当 *包含小数部分时，才输出* 小数点。 *默认状态* 
            - `std::noshowpos`：对非负数， *不* 输出`+`。 *默认状态* 
            - `std::nouppercase`：在十六进制值中输出`0x`，在科学计数法中输出`e`。 *默认状态*  
        - 整形进制操纵符
            - `std::dec`：整型值输出为十进制。 *默认状态*  
            - `std::hex`：整型值输出为十六进制
            - `std::oct`：整型值输出为八进制
        - 浮点格式操纵符
            - `std::fixed`：浮点值输出为定点十进制
            - `std::scitenific`：浮点值输出为科学计数法
            - `std::hexfloat`：浮点值输出为十六进制
            - `std::defaultfloat`： *重置* 浮点值输出为定点十进制
        - 对齐格式操纵符
            - `std::left`：左对齐输出。在值的右侧添加填充字符
            - `std::right`：右对齐输出。在值的左侧添加填充字符。 *默认格式* 
            - `std::internal`：两侧对齐输出，符号左对齐、值右对齐，在中间添加填充字符
        - 输入操纵符
            - `std::skipws`：输入运算符跳过空白符， *默认状态*
            - `std::noskipws`：输入运算符**不**跳过空白符， *默认状态*
        - 输出操纵符
            - `std::unitbuf`：每次输出操作后立即刷新缓冲区
            - `std::nounitbuf`：恢复正常的缓冲区刷新方式， *默认状态*
            - `std::flush`：刷新`std::ostream`缓冲区。**不**改变流的状态， *只影响下一次输出* 
            - `std::ends`：插入 *空字符* ，并刷新`std::ostream`缓冲区。**不**改变流的状态， *只影响下一次输出* 
            - `std::endl`：插入 *换行符* `'\n'`，并刷新`std::ostream`缓冲区。**不**改变流的状态， *只影响下一次输出* 
    - *带参数操纵符* ，定义于头文件`<iomanip>`中
        - `std::setiosflags(mask)`：将`std::ios_base::fmtflags mask`中的`1`在流对象格式标志中的对应位置全部置`1`
            - 注意这里要用`std::ios_base::fmtflags`本身，而**不是**上面那些的操纵符
            - `std::ios`继承了`std::ios_base`，域指定为`std::ios`能短点儿
        - `std::resetiosflags(mask)`：将`std::ios_base::fmtflags mask`中的`1`在流对象格式标志中的对应位置全部置`0`
            - 注意这里要用`std::ios_base::fmtflags`本身，而**不是**上面那些的操纵符
            - `std::ios`继承了`std::ios_base`，域指定为`std::ios`能短点儿，例如
            ```
            std::cout << std::hex;
            std::cout << std::setiosflags(std::ios::showbase | std::ios::uppercase);
            std::cout << 100 << std::endl;  // 0X64
            std::cout << std::resetiosflags(std::hex | std::ios::showbase | std::ios::uppercase);
            std::cout << 100 << std::endl;  // 100
            ```
        - `std::setbase(b)`：将整数输出为`b`进制
        - `std::setprecision(n)`：将输出精度设置为`n`
        - `std::setw(w)`：下一次`I/O`读或写值的宽度为`w`个字符。**不**改变流的状态， *只影响下一次`I/O`* 
        - `std::setfill(c)`：下一次输出用`c`填充空白。默认是 *空格* ` `。**不**改变流的状态， *只影响下一次输出* 
- 很多操纵符改变格式状态
    - 操纵符用于两大类输出控制
        - 数值的输出形式
        - 补白的数量和位置
    - 大多数改变格式状态的操纵符都是设置、复原成对的
        - 一个用于设置新格式
        - 一个用于复原成正常的默认格式
    - 当操纵符改变流的格式状态时， *通常* 改变后的状态对 *所有* 后续`I/O`都生效
        - 当然有一些就不改变流的状态，只对下一次`I/O`生效，比如
            - `std::flush`
            - `std::ends`
            - `std::endl`
            - `std::setw(w)`
            - `std::setfill(c)`
        - 最好在不再需要特殊格式时将流恢复到默认状态
            - 显式调用相反的操纵符固然可以
            - 也可以直接调用`std::setioflags(old_fmt_flags)`操纵符
            ```
            std::cout << 1024UL << std::endl;                               // 1024
            std::ostream::fmtflags old_fmt_flags {std::cout.flags()};
            std::cout << std::hex << std::showbase << 1024UL << std::endl;  // 0x400
            std::cout << std::setiosflags(old_fmt_flags);
            std::cout << 1024UL << std::endl;                               // 1024
            ```
- 控制`bool`的格式
```
// default bool values: 1 0
std::cout << "default bool values: " << true << " " << false << std::endl

// alpha bool values: true false
std::cout << "alpha bool values: " << boolalpha << true << " " << false << std::endl;

bool bool_val = get_status();
std::cout << boolalpha     // sets the internal state of cout
          << bool_val
          << noboolalpha;  // resets the internal state to default formatting
```
- 指定整形的进制
    - 操纵符`std::oct`、`std::hex`和`std::dec` *只* 影响整形运算对象，浮点值的表示**不**受影响
```
std::cout << "default: " << 20 << " " << 1024 << std::endl;                 // default: 20 1024
std::cout << "octal: " << std::oct << 20 << " " << 1024 << std::endl;       // octal: 24 2000
std::cout << "hex: " << std::hex << 20 << " " << 1024 << std::endl;         // hex: 14 400
std::cout << "decimal: " << std::dec << 20 << " " << 1024 << std::endl;     // decimal: 20 1024
```
- 输出中指出进制
```
std::cout << std::showbase;    // show the base when printing integral values
std::cout << "default: " << 20 << " " << 1024 << std::endl;                 // default: 20 1024
std::cout << "in octal: " << std::oct << 20 << " " << 1024 << std::endl;    // in octal: 024 02000
std::cout << "in hex: " << std::hex << 20 << " " << 1024 << std::endl;      // in hex: 0x14 0x400
std::cout << "in decimal: " << std::dec << 20 << " " << 1024 << std::endl;  // in decimal: 20 1024
std::cout << std::noshowbase;  // reset the state of the stream

std::cout << std::uppercase << std::showbase << std::hex 
          << "printed in hexadecimal: " << 20 << " " << 1024                // printed in hexadecimal: 0X14 0X400
          << std::nouppercase << std::noshowbase << std::dec << std::endl;
```
- 控制浮点数格式
    - 可以控制的三种格式
        1. 以多高精度（多少位有效数字）输出
        2. 输出为十六进制、十进制还是科学计数法形式
        3. 对于没有小数部分的浮点值是否还输出小数点
    - 默认情况下
        1. 六位精度
        2. 非常大或小的数输出为科学计数法形式，其他输出为十进制形式
        3. 对于没有小数部分的浮点值，**不**输出小数点
    - 指定精度
        1. 使用`std::cout::precision(n)`
            - `precision`是重载的，一个不接受参数，返回当前精度；另一个接受要设定的精度
        2. 使用`std::cout << std::setprecision(n);`
    ```
    // std::cout.precision reports the current precision value
    // Precision: 6, Value: 1.41421
    std::cout << "Precision: " << std::cout.precision() << ", Value: " << std::sqrt(2.0) << endl;
    
    // cout.precision(12) asks that 12 digits of precision be printed
    // Precision: 12, Value: 1.41421356237
    std::cout.precision(12);
    std::cout << "Precision: " << std::cout.precision() << ", Value: " << std::sqrt(2.0) << endl;
    
    // alternative way to set precision using the setprecision manipulator
    // Precision: 3, Value: 1.41
    std::cout << std::setprecision(3);
    std::cout << "Precision: " << std::cout.precision() << ", Value: " << std::sqrt(2.0) << endl;
    ```
    - 指定计数法
        - 除非真正需要显式控制浮点数的计数法（例如按列打印数据或打印表示金额或百分比的数据），否则由标准库选择计数法时坠吼的
    ```
    std::cout << "default format: " 
              << 100 * std::sqrt(2.0) << std::endl;     // default format: 141.421
    std::cout << "scientific: " << std::scientific 
              << 100 * std::sqrt(2.0) << std::endl;     // scientific: 1.414214e+002
    std::cout << "fixed decimal: " << std::fixed 
              << 100 * std::sqrt(2.0) << std::endl;     // fixed decimal: 141.421356
    std::cout << "hexadecimal: " << std::hexfloat 
              << 100 * std::sqrt(2.0) << std::endl;     // hexadecimal: 0x1.1ad7bcp+7
    std::cout << "use defaults: " << std::defaultfloat 
              << 100 * std::sqrt(2.0) << std::endl;     // use defaults: 141.421
    ```
    - 打印小数点
    ```
    std::cout << 10.0 << std::endl;             // 10
    std::cout << std::showpoint << 10.0         // 10.0000
              << std::noshowpoint << std::endl; // revert to default format for the decimal point
    ```
- 数值输出补白（对齐格式）
    - `std::setw(w)`：指定 *下一个* 数字或字符串的 *最小空间* 
        - **不**改变流的状态， *只影响下一次`I/O`* 
    - `std::left`：左对齐输出
    - `std::right`：右对齐输出， *默认格式* 
    - `std::internal`：控制负数的符号的位置，它是左对齐符号、右对齐值，用空格填满所有中间空间
```
int i = -16;
double d = 3.14159;

// pad the first column to use a minimum of 12 positions in the output
std::cout << "i: " << std::setw(12) << i << "next col" << std::endl   // i:          -16next col
          << "d: " << std::setw(12) << d << "next col" << std::endl;  // d:      3.14159next col

// pad the first column and left-justify all columns
std::cout << std::left
          << "i: " << std::setw(12) << i << "next col" << std::endl   // i: -16         next col
          << "d: " << std::setw(12) << d << "next col" << std::endl   // d: 3.14159     next col
          << std::right;  // restore normal justification

// pad the first column and right-justify (as default) all columns 
std::cout << std::right
          << "i: " << std::setw(12) << i << "next col" << std::endl   // i:          -16next col
          << "d: " << std::setw(12) << d << "next col" << std::endl;  // d:      3.14159next col

// pad the first column but put the padding internal to the field
std::cout << std::internal
          << "i: " << std::setw(12) << i << "next col" << std::endl   // i: -         16next col
          << "d: " << std::setw(12) << d << "next col" << std::endl;  // d:      3.14159next col

// pad the first column, using # as the pad character
std::cout << std::setfill('#')
          << "i: " << std::setw(12) << i << "next col" << std::endl   // i: -#########16next col
          << "d: " << std::setw(12) << d << "next col" << std::endl   // d: #####3.14159next col
          << std::setfill(' ');  // restore the normal pad character
```
- 控制输入格式
    - 默认情况下，输入运算符会 *忽略空白字符* （ *空格* 、 *制表符* 、 *换行* 、 *换纸* 和 *回车* ）
        - 例如，如下循环当给定如下序列时，循环会执行 *四次* ，读取字符`'a'`、`'b'`、`'c'`和`'d'`，跳过中间的空格以及可能的制表符和换行符，输出`abcd`
        ```
        // INPUT: 
        // a b c
        // d
        
        for (char ch; std::cin >> ch;)
        {
            std::cout << ch;
        }
        
        // OUTPUT:
        // abcd
        ```
    - 操纵符`std::noskipws`会令输入运算符读取空白符
        - 例如，如下循环还是给定上面的序列时，循环会执行 *七次* 
        ```
        std::cin >> noskipws;  // set cin so that it reads whitespace
        
        for (char ch; std::cin >> ch;)
        {
            std::cout << ch;
        }
        
        std::cin >> skipws;    // reset cin to the default state so that it discards whitespace
        ```

#### 非格式化`I/O`（unformatted `I/O`）

- 单字节低层`I/O`操作
    - `is.get(ch);`：从`std::istream is`读取 *下一个字节* 存入字符`ch`中，返回`is`的引用
    - `os.put(ch);`：将字符`ch`输出到`std::ostream os`中，返回`os`的引用
    - `is.get();`：将`is`的 *下一个字节* 作为`int`返回
    - `is.putback(ch);`：将刚才读取的 *最近一个字符* （必须和`ch`相同）放回`is`，返回`is`的引用
    - `is.unget();`：将`is` *回退一个字节* ，返回`is`的引用
    - `is.peek();`：将`is`的 *下一个字节* 作为`int`返回，但**不**将其从流中删除
- 将字符放回输入流
    - `is.peek()`：返回输入流中下一个字符的副本，但不会将其从流中删除；`is.peek()`返回的值仍旧留在流中
    - `is.unget()`：使得流向后移动，从而最后读取的值又回到流中。即使我们不知道最后从流中读取什么值，仍然可以调用`is.unget()`
    - `is.putback(ch)`：更特殊版本的`is.unget()`。它退回从流中读取的最后一个值，但它接受一个参数，此参数必须与最后读取的值相同
    - 标准库只保证在读取下一个值之前能回退 *一个值* 
        - 即，标准库**不**保证在中间不进行读取操作的情况下能连续调用`is.putback()`或`is.unget()`
- 从输入操作返回的`int`值
    - `is.peek()`和无参数版本的`is.get()`都以`int`类型从输入流返回一个字符
        - 返回`int`而不是`char`的原因是，可以返回 *文件尾标记* 
        - 使用`char`范围中的每个值来表示一个真实字符，因此`char`没有额外的值用于表示 *文件尾* 
    - 返回`int`的函数将它们要返回的字符首先转换为`unsigned char`，然后再将结果 *提升* 为`int`
        - 因此，即使字符集中有字符映射到负值，这些操作返回的`int`也是 *正值* 
        - 而标准库用 *负值* 表示 *文件尾* ，这就可以保证与任何合法字符的值都不同
        - 我们可以用`EOF`来检测从`is.get()`返回的值是否是文件尾，而不必记忆文件尾的实际数值
        ```
        // <cstdio>
        
        #ifndef EOF
        #define EOF (-1)
        #endif
        ```
    - *警告* ： *必须* 使用`int`，而**不是**`char`来接收返回值
        - 正确
        ```
        int ch;   // use an int, not a char to hold the return from get()
        
        // loop to read and write all the data in the input
        while ((ch = std::cin.get()) != EOF)
        {
            std::cout.put(ch);
        }
        ```
        - 死循环：在`char`被实现为`unsigned char`的机器上，接收到的`EOF`会被转换成`unsigned char`，和`EOF`不再相等
        ```
        char ch;  // using a char here invites disaster!
        
        // the return from cin.get is converted to char and then compared to an int
        // result of comparasion of constant (-1) with expression of type unsigned char is always false! 
        while ((ch = std::cin.get()) != EOF)
        {
            std::cout.put(ch);
        }
        ```
- 多字节低层`I/O`操作
    - `is.get(sink, size, delim);`：从`std::istream is`中读取最多`size`个字节，并存入字符数组`sink`中。读取过程直至遇到字符`delim`或读取了`size`个字节或遇到文件尾为止。如果遇到了`delim`，则将其留在流中，**不**读取出来放入`sink`
        - 一个常见错误是本想从流中删除分隔符，但却忘了做
    - `is.getline(sink, size, delim);`：与前者相似，但会读取并丢弃`delim`。`delim`默认值为`'\n'`
    - `is.read(sink, size);`：读取最多`size`个字节，存入字符数组`sink`中，返回`is`的引用
    - `is.gcount();`：返回上一个未格式化读取操作从`is`读取的字节数
    - `is.ignore(size, delim);`：读取并忽略最多`size`个字符，包括`delim`。`size`默认值为`1`，`delim`默认值为 *文件尾* 
    - `os.write(source, size);`：将字符数组`source`中的`size`个字节写入`std::ostream os`，返回`os`的引用
    - `std::getline(is, buf, delim)`：定义于`<string>`中的按行读入的函数。`delim`默认值为`'\n'`。将读取到的每行存入存入`std::string buf`中，会读取并丢弃`delim`。**不**影响`is.gcount()`。返回`is`的引用
- 确定读取了多少个字符
    - 应在任何后续的非格式化输入操作之前调用`is.gcount`，特别地，`is.peek()`、`is.putback(ch)`或`is.unget()`也是非格式化输入操作
    - 如果在`is.gcount()`之前调用了`is.peek()`、`is.putback(ch)`或`is.unget()`，则`is.gcount()`返回`0`
- 处理空白字符
    - 对于 *空白分隔* 的读入操作（例如`int n; std::cin >> n;`）时，任何后随的 *空白符* ，包括 *换行符* 都会被留在流中
    - 然后当切换到面向行的输入时，以`getline`取得的首行只会是 *该空白符* 
    - 多数情况下这是不想要的行为，可能的解法包括
        1. 对`getline`的显式的额外初始调用
        2. 以`std::cin >> std::ws`移除额外的空白符
            - 可以直接`std::getline(std::cin >> std::ws, buf, delim)`
        3. 以`std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');`忽略输入行上剩下的全部字符 

#### `C++`风格按行读文件

- 推荐的方法：`std::getline(is, buf, delim)`。都`C++`了读入的东西还存字符数组里干啥，真要折腾字符数组的话，`fgets`跟`fread`它不香吗
    - `std::getline`会 *读入并丢弃* 行尾的换行符（或指定的分隔符`delim`），比起`is.get`又不知道高到哪儿去了
    - `fstream`对象会自动析构，析构时自动调用`close`，
        - 所以放进`if`里面列表初始化（`if`**不**支持圆括号初始化），这样就既起到了判断流合法的作用，又保证了自动释放加自动关闭文件
        - 有没有点`python`里面`with open("input.txt", "r") as fin`的感觉了？
    - 另外`line`这个临时变量要是写`while`的话还得放外面，作用域大了容易跟别人撞车，写成`for`的循环变量就真是美汁儿汁儿
```
if (std::ifstream fin {"input.txt", std::ifstream::in})
{
    for (std::string line; std::getline(fin, line, '\n');)
    {
        std::cout << line << std::endl;
    }
}
else
{
    // ...
}
```
- 使用`if with initializer`语法可以进一步适配不能自动转换成`bool`的文件流，例如`cv::FileStorage` `(since C++17)`
```
if (cv::FileStorage fin; fin.open("var/out/laplacian.yml", cv::FileStorage::READ))
{
    // do something...
}
else
{
    // ...
}
```

#### `C++`风格读取文件全部内容

```
if (std::ifstream fin {"input.txt", std::ifstream::in})
{
    std::ostringstream sout;
    sout << fin.rdbuf();
    std::string fileBuf = sout.str();
    std::cout << fileBuf << std::endl;
}
else
{
    // ...
}
```

#### `C++`段子：`std::cout`真比`printf`慢吗

- 你要是乱用`std::endl`天天刷缓冲区玩，那肯定真是慢死了
    - 你看看谁家每个`printf`后面都跟个`fflush`的？
- 一般如果不要求线程安全的话，`std::ios::sync_with_stdio(false)`的话还是很快的
    - 但这个要求就**不能**混用`std::cout`和`printf`了
```
#include <cstdio>
#include <iostream>
 
int main()
{
    std::ios::sync_with_stdio(false);
    std::cout << "a\n";
    std::printf("b\n");
    std::cout << "c\n";
}

// POSSIBLE OUTPUT: 
b
a
c
```

#### 流随机访问

- *定位标记*
    - `gcc`实现
    ```
    // <bits/base_ios.h>
    // class base_ios
    
    public:
        enum _Ios_Seekdir 
        { 
            _S_beg = 0,
            _S_cur = _GLIBCXX_STDIO_SEEK_CUR,  // 1
            _S_end = _GLIBCXX_STDIO_SEEK_END,  // 2
            _S_ios_seekdir_end = 1L << 16 
        };
        
        typedef _Ios_Seekdir seekdir;
        /// Request a seek relative to the beginning of the stream.
        static const seekdir beg = _S_beg;
        /// Request a seek relative to the current position within the sequence.
        static const seekdir cur = _S_cur;
        /// Request a seek relative to the current end of the sequence.
        static const seekdir end = _S_end;
    ```
- 重定位流使之跳过一些数据
    - 定位（seek）到流中给定的位置
    - 告诉（tell）我们当前的位置
- 虽然标准库为所有的流类型都定义了`seek`和`tell`函数，但它们是否有意义取决于流绑定到了什么设备
    - 大多数系统中，绑定到`std::cin`、`std::cout`、`std::cerr`以及`std::clog`的流都**不**支持随机访问
        - 对这些流调用`seek`和`tell`会 *运行时错误* ，并将流置于一个 *无效状态* 
    - `std::fstream`和`std::stringstream`对象一般支持随机访问
    - 分别有 *获得* `g`版本（读取数据，用于`std::istream`）以及 *放置* `p`版本（写入数据，用于`std::ostream`）
        - `std::fstream`以及`std::stringstream`自然两个版本都能用
- `seek`和`tell`函数
    - `is.tellg()`：返回`std::istream is`中标记的当前位置
    - `os.tellp()`：返回`std::ostream os`中标记的当前位置
    - `is.seekg(pos)`：在`std::istream is`中将标记重定位到给定的绝对地址，`pos`通常是前一个`tellg`的返回值
    - `os.seekp(pos)`：在`std::ostream os`中将标记重定位到给定的绝对地址，`pos`通常是前一个`tellp`的返回值
    - `is.seekg(off, from)`：在一个`std::istream is`中将标记定位到`from`之前或之后`off`个字符的位置，`from`可以是
        1. `std::ios_base::beg`：偏移量相对于流开始位置
        2. `std::ios_base::cur`：偏移量相对于流当前位置
        3. `std::ios_base::end`：偏移量相对于流结束位置
    - `os.seekp(off, from)`：在一个`std::ostream os`中将标记定位到`from`之前或之后`off`个字符的位置，`from`和前者相同
- 只有一个标记
    - 一个流中只存在单一的标记，并**不**存在独立的读标记和写标记
    - 由于只有单一的标记，因此只要我们在读写操作间切换，就必须通过`seek`操作来重定位标记
- 重定位标记
    - `is.seekg(pos)`，`os.seekp(pos)`中，`pos`类型为`std::istream::pos_type`或`std::ostream::pos_type`，具体机器相关
        - 表示一个文件位置，
    - `is.seekg(off, from)`，`os.seekp(off, from)`：`off`类型为`std::istream::off_type`或`std::ostream::off_type`，具体机器相关
        - 表示距当前位置的偏移量，可正可负，即又能向前偏移，又能向后偏移
- 访问标记
```
// remember the current write position in mark
std::ostringstream writeStr;  // output stringstream
std::ostringstream::pos_type mark = writeStr.tellp();
// ...
if (cancelEntry)
{
    // return to the remembered position
    writeStr.seekp(mark);
}
```
- 读写同一个文件
    - 例：给定要读取的文件，要在此文件的末尾写入新的一行，这一行包含文件中每行的相对起始位置，例如
    ```
    INPUT: 
    abcd
    efg
    hi
    j

    OUTPUT:
    abcd
    efg
    hi
    j
    5 9 12 14
    ```
    - 程序如下
    ```
    int main()
    {
        // open for input and output and preposition file pointers to end-of-file
        fstream inOut("copyOut", fstream::ate | fstream::in | fstream::out);
        if (!inOut) 
        {
            cerr << "Unable to open file!" << endl;
            return EXIT_FAILURE;           // EXIT_FAILURE see § 6.3.2 (p. 227)
        }
        
        // inOut is opened in ate mode, so it starts out positioned at the end
        auto end_mark = inOut.tellg();     // remember original end-of-file position
        inOut.seekg(0, fstream::beg);      // reposition to the start of the file
        size_t cnt = 0;                    // accumulator for the byte count
        string line;                       // hold each line of input
        
        // while we haven't hit an error and are still reading the original data
        while (inOut && inOut.tellg() != end_mark && getline(inOut, line)) 
        { 
            // and can get another line of input
            cnt += line.size() + 1;        // add 1 to account for the newline
            auto mark = inOut.tellg();     // remember the read position
            inOut.seekp(0, fstream::end);  // set the write marker to the end
            inOut << cnt;                  // write the accumulated length
            
            // print a separator if this is not the last line
            if (mark != end_mark) 
            {
                inOut << " ";
            }
            
            inOut.seekg(mark);             // restore the read position
        }
        
        inOut.seekp(0, fstream::end);      // seek to the end
        inOut << "\n";                     // write a newline at end-offile
        
        return 0;
    }
    ```

#### [`C`风格`I/O`](https://en.cppreference.com/w/cpp/io/c) （C-style file input/output）

- `C++`标准库的`C I/O`子集实现`C`风格流输入/输出操作
    - `<cstdio>`头文件提供通用文件支持并提供有窄和多字节字符`I/O`能力的函数
        - 类型
            - `FILE`：对象类型，足以保有控制`C I/O`流所需的全部信息
            - `fpos_t`：完整非数组对象类型，足以唯一指定文件中的位置，包含其多字节解析状态
            - [`size_t`](https://en.cppreference.com/w/cpp/types/size_t)：`typedef unsigned long size_t;`
        - 宏常量
            - `stdin`，`stdout`，`stderr`：`FILE *`类型表达式，分别与 *标准输入流* 、 *标准输出流* 和 *标准错误流* 关联
            ```
            // <stdio.h>
            /* Standard streams.  */
            extern struct _IO_FILE * stdin;     /* Standard input stream.  */
            extern struct _IO_FILE * stdout;    /* Standard output stream.  */
            extern struct _IO_FILE * stderr;    /* Standard error output stream.  */
            /* C89/C99 say they're macros.  Make them happy.  */
            #define stdin  stdin
            #define stdout stdout
            #define stderr stderr
            
            // <FILE.h>
            /* The opaque type of streams.  This is the definition used elsewhere.  */
            typedef struct _IO_FILE FILE;
            ```
            - `EOF`：拥有`int`类型和负值的整数常量表达式 
            ```
            // <libio.h>
            #ifndef EOF
            #define EOF (-1)
            #endif
            ```
            - `FOPEN_MAX`：能同时打开的文件数
            ```
            // <stdio_lim.h>
            #undef  FOPEN_MAX
            #define FOPEN_MAX 16
            ```    
            - `FILENAME_MAX`：要保有最长受支持文件名的字符数组所需的长度 
            ```
            // <stdio_lim.h>
            #define FILENAME_MAX 4096
            ```
            - `BUFSIZ`：`setbuf`所用的缓冲区大小 
            ```
            // <stdio.h>
            /* Default buffer size.  */
            #ifndef BUFSIZ
            # efine BUFSIZ _IO_BUFSIZ
            #endif
            
            // <libio.h>
            #define _IO_BUFSIZ _G_BUFSIZ
            
            // <_G_config.h>
            #define _G_BUFSIZ 8192
            ```
            - `_IOFBF`，`_IOLBF`，`_IONBF`：给`setvbuf`的参数，分别指示 *全缓冲* 、 *行缓冲* 和 *无缓冲* `I/O`
            ```
            // <stdio.h>
            /* The possibilities for the third argument to `setvbuf'.  */
            #define _IOFBF 0    /* Fully buffered.  */
            #define _IOLBF 1    /* Line buffered.  */
            #define _IONBF 2    /* No buffering.  */
            ```
            - `SEEK_SET`，`SEEK_CUR`，`SEEK_END`：给`fseek`的参数，分别指示从 *文件起始* 、 *当前文件位置* 和 *文件尾* 寻位
            ```
            // <stdio.h>
            /* The possibilities for the third argument to `fseek'.
               These values should not be changed.  */
            #define SEEK_SET   0    /* Seek from beginning of file.  */
            #define SEEK_CUR   1    /* Seek from current position.  */
            #define SEEK_END   2    /* Seek from end of file.  */
            #ifdef __USE_GNU
            # define SEEK_DATA 3    /* Seek to next data.  */
            # define SEEK_HOLE 4    /* Seek to next hole.  */
            #endif
            ```
    - `<cwchar>`头文件提供有宽字符`I/O`能力的函数（不搞外语，就不看了）
- `C`流是`FILE`类型对象，只能通过`FILE *`类型指针访问及操作
    - 通过解引用`FILE *`创建`FILE`类型对象的本地拷贝是可以的，但使用这种本地拷贝是 *未定义行为*
    - 每个`C`流与外部物理设备（文件、标准输入流、打印机、序列端口等）关联
    - 除了访问设备所必须的系统限定信息（例如`POSIX`文件描述符），每个`C`流对象保有以下内容
        1. *字符宽度* （Character width）
            - 未设置
            - 窄
            - 宽
        2. *缓冲状态* （Buffering state）
            - 无缓冲
            - 行缓冲
            - 全缓冲
        3. *缓冲区* （buffer），可为外部的用户提供缓冲区所替换
        4. *`I/O`模式* （`I/O` mode）
            - 输入
            - 输出
            - 更新（输入与输出）
        5. *二进制/文本模式指示器* （Binary/text mode indicator）
        6. *文件尾指示器* （End-of-file status indicator）
        7. *错误状态指示器* （Error status indicator）
        8. *文件位置指示器* （File position indicator），是一个`std::fpos_t`类型对象
            - 对于宽字符流，还包含 *解析状态* （parse state，`std::mbstate_t`类型对象）
        9. *再入锁* （Reentrant lock），用于在多个线程读、写、寻位或查询流时避免数据竞争（data races） `(since C++17)`
- *文本模式* 和 *二进制模式* 
    - *文本流* （text stream）
        - 由 *行* 组成，每行都是 *有序字符序列* 加上 *终止符* `'\n'`
            - 最后一行是否需要终止符`'\n'` *由具体实现定义* 
        - 为配合 *操作系统* 的文本格式，`I/O`时可能会添加、切换或删除个别字符
            - 特别地，`Windows`上的`C`流在输出时将`'\n'`转换为`'\r\n'`、输入时将`'\r\n'`转换为`'\n'`
        - 仅当如下条件全部被满足时，从文件流中读入的数据与之前写入文件的数据相同
            - 数据只含有 *可打印字符* 和 *控制字符* （`'\t'`及`'\n'`）
            - `'\n'`的直接前驱字符**不是** *空格* ` `（`'\n'`前的空格在读入时可能消失）
            - *尾字符* 是`'\n'`
    - *二进制流* （binary stream）
        - 有序字符序列，直接记录内部数据
        - 从二进制流读入的数据永远与先前写入的数据相同
            - 只可能在流末尾加入空字符`'\0'`
    - `POSIX`**不辨别**文本与二进制流
        - 无`'\n'`或任何其他字符的特殊映射 
- `C`风格文件访问
    - [`fopen`](https://en.cppreference.com/w/cpp/io/c/fopen)：打开文件
        - 签名
        ```
        FILE * fopen(const char * filename, const char * mode);
        ```
        - 打开`filename`所指示的文件并返回与该文件关联的流，用`mode`确定 *文件访问模式* 
        - 可能的文件访问模式
            - `"r"`： *只读* ，为读取打开文件。若文件已存在，则从起始位置读取；否则，打开失败
            - `"w"`： *只写* ，为写入创建文件。若文件已存在，则截断文件；否则，创建新文件
            - `"a"`： *追加* ，追加到文件。若文件已存在，则追加写入到文件的末尾；否则，创建新文件
                - 在 *追加* 模式中，写入数据到文件尾，忽略文件位置指示器的当前位置
            - `"r+"`： *扩展读* ，为读写打开文件。若文件已存在，则从起始位置读取；否则，打开失败
            - `"w+"`： *扩展写* ，为读写创建文件。若文件已存在，则截断文件；否则，创建新文件
            - `"a+"`： *扩展追加* ，为读写打开文件。若文件已存在，则追加写入到文件的末尾；否则，创建新文件
                - 在 *追加* 模式中，写入数据到文件尾，忽略文件位置指示器的当前位置
            - 注意
                - 文件访问标志`"b"`为 *可选* 的，用于指定用 *二进制模式* 打开文件。此标志在`POSIX`系统上无效果，但例如在`Windows`上，它禁用`'\n'`和`'\x1A'`的特殊处理
                - 文件访问标志`"x"`能 *可选* 地追加到`"w"`或`"w+"`指定符，强制函数在文件已存在时 *失败* ，**而非**截断文件 `(since C++17)`
                - 若模式不是以上字符串之一，则 *行为未定义* 
        - 返回值
            - 成功，则返回指向控制打开的文件流的对象的指针，并清除文件尾和错误位。流为 *完全缓冲* 的，除非`filename`指代交互式设备
            - 错误，返回 *空指针* ，并设置[`errno`](https://en.cppreference.com/w/cpp/error/errno)
        - 使用示例
        ```
        #include <cstdlib>
        #include <cstdio>
        
        int main()
        {
            FILE * fp = fopen("test.txt", "r");
            
            if (!fp) 
            {
                std::perror("File opening failed");
                return EXIT_FAILURE;
            }
         
            int c;  // note: int, not char, required to handle EOF
            
            while ((c = std::fgetc(fp)) != EOF) 
            { 
                // standard C I/O file reading loop
                putchar(c);
            }
         
            if (ferror(fp))
            {
                puts("I/O error when reading");
            }
            else if (feof(fp))
            {
                puts("End of file reached successfully");
            }
                
            fclose(fp);
            
            return EXIT_SUCCESS;
        }
        ```
    - [`freopen`](https://en.cppreference.com/w/cpp/io/c/freopen)：以不同名称打开既存流 
        - 签名
        ```
        FILE * freopen(const char * filename, const char * mode, FILE * stream);
        ```
        - 首先，试图关闭与`stream`关联的文件，忽略任何错误；然后，若`filename`非空，则用`mode`打开`filename`所指定的文件，然后将该文件与`stream`所指向的文件流关联；若`filename`为空指针，则函数试图重打开已与`stream`关联的文件（此情况下是否允许模式改变是实现定义的）
        - 返回值
            - 成功时：`stream`
            - 失败时：`NULL` 
    - [`fclose`](https://en.cppreference.com/w/cpp/io/c/fclose)：关闭文件 
        - 签名
        ```
        int fclose(FILE * stream);
        ```
        - 刷新缓冲区，然后关闭给定的文件流。任何未读取的缓冲数据将被舍弃
            - 无论操作是否成功，流都不再关联到文件。且由`setbuf`或`setvbuf`分配的缓冲区若存在，则亦被解除关联，并且若使用自动分配则被解分配
            - 若在`fclose`返回后使用`stream`，则 *行为未定义*  
        - 返回值
            - 成功时：`0`
            - 失败时：`EOF` 
    - [`fflush`](https://en.cppreference.com/w/cpp/io/c/fflush)：将输出流与实际文件同步 
        - 签名
        ```
        int fflush(FILE * stream);
        ```
        - 刷新输出流`stream`的缓冲区
            - 对于输出流（和最近操作为输出的更新流），将来自`stream`缓冲区的未写入数据写入关联的输出设备
            - 对于输入流（和最近操作为输入的更新流）， *行为未定义* 
            - 若`stream`为 *空指针* ，则刷新 *所有* 输出流（和最近操作为输出的更新流），包含程序不能直接访问的流 
        - 返回值
            - 成功时：`0`
            - 失败时：`EOF`
    - [`setbuf`](https://en.cppreference.com/w/cpp/io/c/setbuf)：为文件流设置缓冲区 
        - 签名
        ```
        void setbuf(FILE * stream, char * buffer);
        ```
        - 为`C`流`stream`上进行的`I/O`操作设置内部缓冲区
            - 若`buffer`**非**空，则等价于`setvbuf(stream, buffer, _IOFBF, BUFSIZ)` 。
            - 若`buffer`为 *空* ，则等价于`setvbuf(stream, NULL, _IONBF, 0) `，这会 *关闭缓冲* 
        - 注意
            - `setbuf`所用的缓冲区大小为宏定义常量`BUFSIZ`。若`BUFSIZ`不是适合的缓冲区大小，则能用`setvbuf`更改它
            - `setbuf`**不**指示成功或失败，因此应当使用`setvbuf` *检测错误* 
            - 此函数仅可在已将`stream`关联到打开的文件后，但要在任何其他操作（除对`setbuf`或`setvbuf`的失败调用以外）前使用
            - 一个 *常见错误* 是设置`stdin`或`stdout`的缓冲区为生存期在程序终止前结束的数组
            ```
            int main() 
            {
                char buf[BUFSIZ];
                std::setbuf(stdin, buf);
            } // buf 的生存期结束，未定义行为
            ```
        - 使用示例
        ```
        #include <chrono>
        #include <cstdio>
        #include <thread>
         
        int main()
        {
            using namespace std::chrono_literals;
         
            std::setbuf(stdout, NULL);        // 无缓冲的 stdout
            std::putchar('a');                // 在无缓冲的流上立即显现
            std::this_thread::sleep_for(1s);
            std::putchar('b');
        }
        ```
    - [`setvbuf`](https://en.cppreference.com/w/cpp/io/c/setvbuf)：为文件流设置缓冲区与其大小
        - 签名
        ```
        int setvbuf(FILE * stream, char * buffer, int mode, size_t size);
        ```
        - 以`mode`所指示值更改给定文件流`stream`的缓冲模式
            - 若`buffer`为 *空指针* ，则 *重设* 内部缓冲区大小为`size`
            - 若`buffer`**不**是空指针，则指示流使用始于`buffer`而大小为`size`的用户提供缓冲区
                - 必须在`buffer`所指向的数组的生存期结束前用`fclose`关闭流
                - 成功调用`setvbuf`后，`buffer`内容 *不确定* ，使用它是 *未定义行为* 
            - `mode`可以是
                1. `_IOFBF`， *全缓冲* ：当缓冲区为空时，从流读入数据。或者当缓冲区满时，向流写入数据
                2. `_IOLBF`， *行缓冲* ：每次从流中读入一行数据或向流中写入一行数据
                3. `_IONBF`， *无缓冲* ：直接从流中读入数据或直接向流中写入数据，缓冲设置无效
        - 注意
            - 此函数仅可在已将`stream`关联到打开的文件后，但要在任何其他操作（除对`setbuf`或`setvbuf`的失败调用以外）前使用
            - 不是所有`size`字节都需要用于缓冲：实际缓冲区大小通常向下取整到`2`的倍数、页面大小的倍数等
            - 多数实现上，行缓冲仅对 *终端输入流* 可用
            - 一个 *常见错误* 是设置`stdin`或`stdout`的缓冲区为生存期在程序终止前结束的数组
            ```
            int main() 
            {
                char buf[BUFSIZ];
                std::setbuf(stdin, buf);
            } // buf 的生存期结束，未定义行为
            ```
        - 返回值
            - 成功时：`0`
            - 失败时：非零
- `C`风格直接`I/O`
    - [`fread`](https://en.cppreference.com/w/cpp/io/c/fread)：从文件读取 
        - 签名
        ```
        size_t fread(void * buffer, size_t size, size_t count, FILE * stream);
        ```
        - 从给定输入流`stream`读取至多`count`个大小为`size`的对象到数组`buffer`中
            - 如同对每个对象调用`size`次`fgetc`，依次将结果存储到`reinterpret_cast`为`unsigned char *`的`buffer`中
            - 流的文件位置指示器将前进读取的字符数
            - 若对象**不** *可平凡复制* （TriviallyCopyable），则 *行为未定义* 
            - 若出现 *错误* ，则流的文件位置指示器的结果值 *不确定* 
                - 若此时只读入了某个元素的部分内容，则元素的值亦不确定 
        - 返回值
            - 成功时：成功读取的对象数
            - 出现错误或文件尾：实际读取的对象数，可能小于`count`
            - 若`size`或`count`为 *零* ，则`fread` *返回零* 且**不**进行其他动作
        - 使用示例
        ```
        #include <cstdio>
        #include <fstream>
        #include <iostream>
        #include <vector>
        
        int main()
        {
            // 准备文件
            std::ofstream("test.txt") << 1 << ' ' << 2 << std::endl;
           
            if (FILE * fp = fopen("test.txt", "r"))
            {
                // char 可平凡复制
                std::vector<char> buf1(4);  // NOTE: SHOULD use buf(4) to value-initialize 4 elements
                                            //       if use buf.reserve(4), buf.size() WON'T get updated!!! 
                std::fread(&buf1[0], sizeof buf1[0], buf1.size(), fp);
                fclose(fp);
            }
         
            for (char n : buf1)
            {
                std::cout << n << std::endl;
            }
         
            // std::string 不可平凡复制
            // 用 fread 读入 std::string 是未定义行为
            std::vector<std::string> buf2(4);  
            fread(&buf2[0], sizeof buf2[0], buf2.size(), fp);
        }
        ```
    - [`fwrite`](https://en.cppreference.com/w/cpp/io/c/fwrite)：写入文件 
        - 签名
        ```
        size_t fwrite(const void * buffer, size_t size, size_t count, FILE * stream);
        ```
        - 向输出流`stream`中写入数组`buffer`中的前`count`个对象
            - 如同将每个对象`reinterpret_cast`为`unsigned char *`，然后依次对每个对象调用`size`次`fputc`
            - 流的文件位置指示器将前进写入的字符数
            - 若对象**不** *可平凡复制* （TriviallyCopyable），则 *行为未定义* 
            - 若出现 *错误* ，则流的文件位置指示器的结果值 *不确定* 
        - 返回值
            - 成功时：成功写入的对象数
            - 出现错误或文件尾：实际写入的对象数，可能小于`count`
            - 若`size`或`count`为 *零* ，则`fwrite` *返回零* 且**不**进行其他动作
        - 使用示例
        ```
        #include <array>
        #include <cstdio>
        #include <vector>
         
        int main ()
        {
            // 写缓冲区到文件
            if (FILE * fp1 = fopen("file.bin", "wb")) 
            {
                std::array<int, 3> v {42, -1, 7};  // std::array 的底层存储为数组
                fwrite(v.data(), sizeof v[0], v.size(), fp1);
                fclose(fp1);
            }
         
            // 读取同一数据并打印它到标准输出
            if (FILE * fp2 = fopen("file.bin", "rb")) 
            {
                std::vector<int> rbuf(10);         // std::vector 的底层存储亦为数组
                size_t sz = fread(rbuf.data(), sizeof rbuf[0], rbuf.size(), fp2);
                fclose(fp2);
                
                for (size_t n = 0; n < sz; ++n) 
                {
                    printf("%d\n", rbuf[n]);
                }
            }
        }
        ```
- `C`风格非格式化`I/O`
    - [`fgetc`，`getc`](https://en.cppreference.com/w/cpp/io/c/fgetc)：从文件流获取字符 
        - 签名
        ```
        int fgetc(FILE * stream);
        int getc(FILE * stream);
        ```
        - 读取来自给定输入流`stream`的下个字符
            - [`getchar`](https://en.cppreference.com/w/cpp/io/c/getchar)：`int getchar();`等价于`getc(stdin);`
            - *警告* ： *必须* 使用`int`，而**不是**`char`来接收返回值
                - 正确
                ```
                int ch;   // use an int, not a char to hold the return from get()
                
                // loop to read and write all the data in the input
                while ((ch = getc()) != EOF)
                {
                    putchar(ch);
                }
                ```
                - 死循环：在`char`被实现为`unsigned char`的机器上，接收到的`EOF`会被转换成`unsigned char`，和`EOF`不再相等
                ```
                char ch;  // using a char here invites disaster!
                
                // the return from getc is converted to char and then compared to an int
                // result of comparasion of constant (-1) with expression of type unsigned char is always false! 
                while ((ch = getc()) != EOF)
                {
                    putchar(ch);
                }
                ```
        - 返回值
            - 成功时，为 *获得的字符* 
            - 失败时，为`EOF`
                - 若 *文件尾条件* 导致失败，则另外设置文件尾指示器（`feof()`）
                - 若 *其他错误* 导致失败，则另外设置错误指示器（`ferror()`）
    - [`fgets`, `gets`](https://en.cppreference.com/w/cpp/io/c/fgets)：从文件流获取字符串 
        - 签名
        ```
        char * fgets(char * str, int count, FILE * stream);
        ```
        - 从给定流`stream`读取最多`count - 1`个字符并将它们存储于`str`所指向的字符数组
            - 若 *文件尾* 出现或发现 *换行符* ，则终止分析
                - 后一情况下`str`将 *包含换行符* 
                - 成功读入且无错误发生时，`str`结尾将被写入`'\0'`结尾
            - 尽管标准规范在`count <= 1`的情况下不明，常见的实现
                - 若`count < 1`，则**不**做任何事并报告错误
                - 若`count == 1`，则 
                    - 某些实现**不**做任何事并报告错误
                    - 其他实现**不**读内容，存储零于`str[0]`并报告成功 
            - [`gets`](https://en.cppreference.com/w/cpp/io/c/gets) `(deprecated in C++11)(removed in C++14)` **不**能避免 *缓冲区溢出* 。应使用`std::fgets`替代 
        - 返回值
            - 成功时，为`str`
            - 失败时，为 *空指针* 
                - 若 *文件尾条件* 导致失败，则另外设置`stream`的 *文件尾指示器* （`feof()`））
                    - 该情况下**不**改变`str`所指向数组的内容（即**不**以 *空字符* 覆写首字节）
                - 若 *其他错误* 导致失败，则另外设置`stream`的 *错误指示器* （`ferror()`）
                    - 此时`str`所指向的数组内容是 *不确定的* （甚至可以**不**是 *空终止* 的）
                    - `POSIX`额外要求若`fgets`遇到异于文件尾条件的失败，则设置`errno`
        - 使用示例
        ```
        if (FILE * tmpf = tmpfile())
        {
            fputs("Alan Turing\n", tmpf);
            fputs("John von Neumann\n", tmpf);
            fputs("Alonzo Church\n", tmpf);
            rewind(tmpf);

            for (char buf[BUFSIZ]; fgets(buf, sizeof buf, tmpf);)
            {
                // fgets will accept '\n' at end of line, rather than drop it
                // fgets will also append a '\0' at buf's end
                // delete '\n' manually
                buf[strlen(buf) - 1] = '\0';
                printf("\"%s\"\n", buf);
            }

            fclose(tmpf);
        }
        ```
    - [`fputc`，`putc`](https://en.cppreference.com/w/cpp/io/c/fputc)：写字符到文件流 
        - 签名
        ```
        int fputc(int ch, FILE * stream);
        int putc(int ch, FILE * stream);
        ```
        - 向输出流`stream`中写入字符`ch`
            - 在内部，在 *写入前将字符转换为`unsigned char`* 
            - `C`中，`putc`可以实现为宏，而这在`C++`中被禁止。从而，`fputc`和`putc`始终拥有相同效果
            - [`putchar`](https://en.cppreference.com/w/cpp/io/c/putchar)：`int putchar(int ch);`等价于`putc(ch, stdin);`
        - 返回值
            - 成功时，返回 *被写入字符* 
                - `fputc`、`putc`以及`putchar`的返回值在 *`ch`为负* 时**不**等于`ch`
                ```
                int ch = -3;
                int ret = putchar(ch);
                printf("%d %d\n", ret, ch);    // 253 -3
                ```
            - 失败时，返回 *EOF* ，并设置 *错误指示器* （`ferror()`） 
    - [`fputs`](https://en.cppreference.com/w/cpp/io/c/fputs)：写字符串到文件流 
        - 签名
        ```
        int fputs(const char * str, FILE * stream);
        ```
        - 向`stream`写入`str`的每个字符
            - 如同重复执行`fputc`
            - **不**写入`str`的终止空字符 
            - 与`puts`（后附新换行符）不同，`fputs`则写入 *不修改的字符串* 
        - 返回值
            - 成功时，返回 *非负值*
                - 不同的实现返回不同的非负数
                    - 一些返回最后写入的字符
                    - 一些返回写入的字符数（或若字符串长于`INT_MAX`则为该值）
                    - 一些简单地非负常量，例如零
            - 失败时，返回`EOF`，并设置 *错误指示器* （`ferror()`） 
        - 使用示例
        ```
        if ((int rc = std::fputs("Hello World", stdout)) == EOF)
        {
            std::perror("fputs()");  // POSIX 要求设置 errno
        }
        ```
    - [`puts`](https://en.cppreference.com/w/cpp/io/c/puts)：写字符串到`stdout`
        - 签名
        ```
        int puts(const char * str);
        ```
        - 向`stdout`写入`str`，附带一个 *换行符* `'\n'`
            - 如同重复执行`putc`
            - **不**写入`str`的终止空字符
            - 与`fputs`（**不**附加 *换行符* ）不同，`puts`则多写入一个 *换行符*
        - 返回值
            - 成功时，返回 *非负值*
                - 不同的实现返回不同的非负数
                    - 一些返回最后写入的字符
                    - 一些返回写入的字符数（或若字符串长于`INT_MAX`则为该值）
                    - 一些简单地非负常量，例如零
            - 失败时，返回`EOF`，并设置 *错误指示器* （`ferror()`） 
                - 在重定向`stdout`到文件时，导致`puts`失败的典型原因是 *用尽了文件系统的空间* 
        - 使用示例
        ```
        #include <cstdio>
 
        int main()
        {
            int rc = std::puts("Hello World");
         
            if (rc == EOF)
            {
                perror("puts()");    // POSIX 要求设置 errno
            }
        }
        ```
    - [`ungetc`](https://en.cppreference.com/w/cpp/io/c/ungetc)：把字符放回文件流 
        - 签名
        ```
        int ungetc(int ch, FILE * stream);
        ```
        - 若`ch != EOF`，则将字符`ch` `reinterpret_cast`为`unsigned char`后，写入到`stream`的输入缓冲区
        - 若`ch == EOF`，则 *操作失败* 而**不**影响流
        - 对`ungetc`的成功调用清除文件尾状态标志`feof`
        - 在 *二进制流* 上对`ungetc`的成功调用将流位置指示器 *减少一* （若流位置指示器为零，则 *行为未定义* ）
        - 在 *文本流* 上对`ungetc`的成功调用以 *未定义方式* 修改流位置指示器，但保证在以读取操作取得所有回退的字符后，流位置指示器等于其在`ungetc`之前的值
        - 返回值
            - 成功时。返回`ch`
            - 失败时。返回`EOF`，而给定的流保持不变 
- `C`风格格式化`I/O`
    - [`scanf`，`fscanf`，`sscanf`](https://en.cppreference.com/w/cpp/io/c/fscanf)：从`stdin`、文件流或缓冲区读取有格式输入
        - 签名
        ```
        ​int scanf(const char * format, ... );                          ​(1)    
        int fscanf(FILE * stream, const char * format, ... );          (2)  
        int sscanf(const char * buffer, const char * format, ... );    (3) 
        ```
        - 从各种源读取数据，按照`format`转译并存储结果于给定位置
        - *格式字符串* （format string）`format`由下列内容组成
            - 除 *百分号* `%`和 *空白字符* 以外的字符：精确匹配输入中的 *这个字符* ，否则失败
            - *空白字符* （whitespace characters）：匹配输入中的 *任意长度连续空白字符* ，注意`format`中`\n`、` `、`\t\t`或 *其他空白字符* 都是等价的 
                - 空白字符由`isspace(c)`定义
                    - 包括
                        1. *空格* （space，`0x20`，`' '`）
                        2. *换页* （form feed，`0x0c`，`'\f'`）
                        3. *换行* （line feed，`0x0a`，`'\n'`）
                        4. *回车* （carriage return，`0x0d`，`'\r'`）
                        5. *水平制表符* （horizontal tab，`0x09`，`'\t'`）
                        6. *垂直制表符* （vertical tab，`0x0b`，`'\v'`）
                    - 如果`ch`的值**不能**表示为`unsigned char`，并且**不等于**`EOF`，则函数 *行为未定义* 
            - *转换说明* （conversion specifications），并可以依序包含如下内容
                - 打头的 *百分号* `%`
                - *赋值抑制字符* （assignment-suppressing character）`*`：丢弃此项对应的结果， *可选*
                - *最大域宽说明符* （max field width specifier）： *正整数* ，进行在当前转换说明所指定的转换时允许处理的最大字符数， *可选*
                    - 注意若不提供宽度，则`%s`和`%[`可能导致 *缓冲区溢出* 
                - *长度修饰符* （length modifier）：具体可选内容见下表，用于指明此处匹配的期待参数类型， *可选*
                - *转换格式说明符* （conversion format specifier）
                    - 包括如下选项
                        - `%`：匹配字面`%`
                        - `c`：匹配一个 *字符* 
                            - *最大域宽说明符* 
                                - 若使用了宽度说明符，则匹配准确的宽度个字符（该参数必须是指向有充足空间的数组的指针）
                                - 不同于`%s`和`%[`，它**不会**在数组后 *附加空字符*  
                            - *长度修饰符* 
                                - `c`：期待参数类型`char *`
                                - `lc`：期待参数类型`wchar_t *`
                        - `s`：匹配一个 *字符串* 
                            - 宽度说明符
                                - 若使用宽度说明符，则至多匹配宽度个字符，或匹配到首个提前出现的空白符前
                                - 总是在匹配的字符后存储一个空字符（故参数数组长度至少比宽度多一）
                            - 长度说明符
                                - `c`：期待参数类型`char *`
                                - `ls`：期待参数类型`wchar_t *`
                        - `[set]`：匹配一个由来自 *集合* `[set]`的字符组成的 *字符串* 
                            - 集合
                                - 若集合的首字符是`^`，则匹配所有**不在**集合中的字符。
                                - 若集合以`]`或`^]`开始，则`]`字符亦被包含入集合。
                                - 在扫描集合的非最初位置的字符`-`是否可以指示范围，如 [0-9] ，是 *实现定义* 的
                            - 最大域宽说明符
                                - 若使用宽度说明符，则最多匹配到宽度
                                - 总是在匹配的字符后存储一个空字符（故参数数组长度至少比宽度多一）
                            - 长度说明符
                                - `c`：期待参数类型`char *`
                                - `lc`：期待参数类型`wchar_t *`
                        - `d`：匹配一个 *十进制整数* 
                            - 长度说明符
                                - `hhd`：期待参数类型`signed char *`或`unsigned char *`
                                - `hd`：期待参数类型`signed short *`或`unsigned short *`
                                - `d`：期待参数类型`signed int *`或`unsigned int *`
                                - `ld`：期待参数类型`signed long *`或`unsigned long *`
                                - `lld`：期待参数类型`signed long long *`或`unsigned long long *`
                                - `jd`：期待参数类型`intmax_t *`或`uintmax_t *`
                                - `zd`：期待参数类型`size_t *`
                                - `td`：期待参数类型`ptrdiff_t *`
                        - `i`：匹配一个 *整数* 
                            - 长度说明符
                                - `hhi`：期待参数类型`signed char *`或`unsigned char *`
                                - `hi`：期待参数类型`signed short *`或`unsigned short *`
                                - `i`：期待参数类型`signed int *`或`unsigned int *`
                                - `li`：期待参数类型`signed long *`或`unsigned long *`
                                - `lli`：期待参数类型`signed long long *`或`unsigned long long *`
                                - `ji`：期待参数类型`intmax_t *`或`uintmax_t *`
                                - `zi`：期待参数类型`size_t *`
                                - `ti`：期待参数类型`ptrdiff_t *`
                        - `u`：匹配一个 *无符号十进制整数* 
                            - 长度说明符
                                - `hhu`：期待参数类型`signed char *`或`unsigned char *`
                                - `hu`：期待参数类型`signed short *`或`unsigned short *`
                                - `u`：期待参数类型`signed int *`或`unsigned int *`
                                - `lu`：期待参数类型`signed long *`或`unsigned long *`
                                - `llu`：期待参数类型`signed long long *`或`unsigned long long *`
                                - `ju`：期待参数类型`intmax_t *`或`uintmax_t *`
                                - `zu`：期待参数类型`size_t *`
                                - `tu`：期待参数类型`ptrdiff_t *`
                        - `o`：匹配一个 *无符号八进制整数* 
                            - 长度说明符
                                - `hho`：期待参数类型`signed char *`或`unsigned char *`
                                - `ho`：期待参数类型`signed short *`或`unsigned short *`
                                - `o`：期待参数类型`signed int *`或`unsigned int *`
                                - `lo`：期待参数类型`signed long *`或`unsigned long *`
                                - `llo`：期待参数类型`signed long long *`或`unsigned long long *`
                                - `jo`：期待参数类型`intmax_t *`或`uintmax_t *`
                                - `zo`：期待参数类型`size_t *`
                                - `to`：期待参数类型`ptrdiff_t *`
                        - `x`，`X`：匹配一个 *无符号十六进制整数* 
                            - 长度说明符
                                - `hhx`：期待参数类型`signed char *`或`unsigned char *`
                                - `hx`：期待参数类型`signed short *`或`unsigned short *`
                                - `x`：期待参数类型`signed int *`或`unsigned int *`
                                - `lx`：期待参数类型`signed long *`或`unsigned long *`
                                - `llx`：期待参数类型`signed long long *`或`unsigned long long *`
                                - `jx`：期待参数类型`intmax_t *`或`uintmax_t *`
                                - `zx`：期待参数类型`size_t *`
                                - `tx`：期待参数类型`ptrdiff_t *`
                        - `n`： *返回* *迄今读取的字符数* 
                            - **不**消耗输出
                            - **不**增加赋值计数
                            - 若此说明符拥有 *赋值抑制运算符* ，则 *行为未定义* 
                            - 长度说明符
                                - `hhn`：期待参数类型`signed char *`或`unsigned char *`
                                - `hn`：期待参数类型`signed short *`或`unsigned short *`
                                - `n`：期待参数类型`signed int *`或`unsigned int *`
                                - `ln`：期待参数类型`signed long *`或`unsigned long *`
                                - `lln`：期待参数类型`signed long long *`或`unsigned long long *`
                                - `jn`：期待参数类型`intmax_t *`或`uintmax_t *`
                                - `zn`：期待参数类型`size_t *`
                                - `tn`：期待参数类型`ptrdiff_t *`
                        - `a`，`A`，`e`，`E`，`f`，`F`，`g`，`G`：匹配一个 *浮点数* 
                            - 长度说明符
                                - `f`：期待参数类型`float *`
                                - `lf`：期待参数类型`double *`
                                - `Lf`：期待参数类型`long double *`
                        - `p`：匹配一个 *指针* ，具体 *格式由实现定义* ，应和`printf("%p")`产生的字符串格式相同
                            - 长度说明符
                                - `p`：期待参数类型`void *`
                    - 注解
                        - 所有异于`[`、`c`和`n`的转换说明符，在尝试分析输入前消耗并 *舍弃所有前导空白字符* 
                        - 这些被消耗的字符**不计入**指定的 *最大域宽*  
        - 返回值
            - 成功赋值的参数的数量（在首个参数赋值前发生匹配失败的情况下可为零）
            - 若在赋值首个接收的参数前输入失败，则为`EOF`
    - [`printf`，`fprintf`，`sprintf`，`snprintf`](https://en.cppreference.com/w/c/io/fprintf)：打印有格式输出到 `stdout`、文件流或缓冲区 
        - 签名
        ```
        int printf(const char * format, ... );                                      (1)     
        int fprintf(FILE * stream, const char * format, ... );                      (2)     
        int sprintf(char * buffer, const char * format, ... );                      (3)     
        int snprintf(char * buffer, size_t buf_size, const char * format, ... );    (4)
        ```
        - 从给定位置加载数据，按`format`指定的格式转换为字符串等价版本，并将结果写入
            - `(1)` `stdout`
            - `(2)` 流`stream`
            - `(3)` 字符数组`buffer`
            - `(4)` 字符数组`buffer`
                - 若`buf_size > 0`：至多写`buf_size - 1`个字符，以`'\0'`结尾
                - 若`buf_size == 0`：不写入任何内容，且`buffer`可以是 *空指针* ，然而依旧计算返回值并返回
        - *格式字符串* （format string）`format`由下列内容组成
            - 除 *百分号* `%`以外的字符：精确匹配输入中的 *这个字符* ，否则失败
            - *转换说明* （conversion specifications），并可以依序包含如下内容
                - 打头的 *百分号* `%`
                - *输出格式说明符* （conversion format specifier），具体包含如下选项， *可选* 
                    - `-`：域内 *左对齐* （默认右对齐）
                    - `+`：对数值始终 *显示符号* （默认只显示负数的负号）
                    - ` `：若有符号转换的结果不以符号开始或为空，则 *前置空格* 于结果。若已存在`+`，则忽略 *空格* ` `
                    - `#`：使用 *替用形式* 。准确的效果见下表，其他情况下 *行为未定义* 
                    - `0`：对于整数和浮点数转换，使用 *前导`0`* 代替 *空格* 字符填充域
                        - 对于整数，若显式指定精度，则忽略此标签
                        - 对于其他转换，使用此标签导致 *未定义行为* 
                        - 若已存在`-`，则忽略`0`
                - *最小域宽说明符* （min field width specifier）： *正整数* 或`*`， *可选*
                    - 域内的对齐以及补白填充，遵照前一项 *输出格式说明符* 
                    - 使用`*`的情况下，以一个 *额外* 的`int`类型 *参数* 指定宽度
                    - 若参数值为负数，则它导致指定 *`-`标签* 和 *正域宽* 
                        - 注意：这是最小宽度：决不会截断值
                - *精度说明符* （precision sepcifier）： *以`.`开头的整数* 或 `*`，表示转换的精度
                    - 使用`*`的情况下，以一个 *额外* 的`int`类型 *参数* 指定精度
                    - 若此参数的值为 *负数* ，则它 *被忽略* 
                    - 若既不使用数字亦不使用`*`，则精度采用`0`
                    - 精度的准确效果见下表
                - *长度修饰符* （length modifier）：具体可选内容见下表，用于指明此处匹配的期待参数类型， *可选*
                - *转换格式说明符* （conversion format specifier）
                    - 包括如下选项
                        - `%`：写字面的`%`。完整转换指定必须是`%%`
                        - `c`：写 *单个字符* 
                            - 长度修饰符
                                - `c`：参数首先被转换成`unsigned char`，期待参数类型`int`
                                - `lc`：参数首先被转换成 *字符串* ，期待参数类型`wint_t`
                        - `s`：写 *字符串*  
                            - 精度
                                - 指定写入最大的字符数
                                - 若未指定精度，则写每个字节首个`'\0'`（不写`'\0'`本身）
                            - 长度修饰符
                                - `s`：期待参数类型`char *`
                                - `ls`：数组会被转换成`char`数组，期待参数类型`wchar_t *`
                        - `d`，`i`：写 *有符号十进制整数* `[-]dddd`
                            - 精度
                                - 精度指定出现的 *最小数位数* 。默认精度是`1`
                                - 若被转换的值和精度都是`​0​`，则转换结果 *无字符*  
                            - 长度修饰符 
                                - `hhd`：期待参数类型`signed char`
                                - `hd`：期待参数类型`short`
                                - `d`：期待参数类型`int`
                                - `ld`：期待参数类型`long`
                                - `lld`：期待参数类型`long long`
                                - `jd`：期待参数类型`intmax_t`
                                - `zd`：期待参数类型 *有符号`size_t *`* 
                                - `td`：期待参数类型`ptrdiff_t`
                        - `u`：写 *无符号十进制整数* `dddd`
                            - 精度
                                - 精度指定出现的 *最小数位数* 。默认精度是`1`
                                - 若被转换的值和精度都是`​0​`，则转换结果 *无字符*                          
                            - 长度修饰符 
                                - `hhu`：期待参数类型`unsigned char`
                                - `hu`：期待参数类型`unsigned short`
                                - `u`：期待参数类型`unsigned int`
                                - `lu`：期待参数类型`unsigned long`
                                - `llu`：期待参数类型`unsigned long long`
                                - `ju`：期待参数类型`uintmax_t`
                                - `zu`：期待参数类型`size_t`
                                - `tu`：期待参数类型 *无符号`ptrdiff_t`* 
                        - `o`：写 *无符号八进制整数* `oooo`
                            - 精度
                                - 默认形式
                                    - 精度指定出现的 *最小数位数* 。默认精度是`1`
                                    - 若被转换的值和精度都是`​0​`，则转换结果 *无字符* 
                                - 替用形式
                                    - 精度按需增加，以写入一个表示八进制的前导`0`                                    
                                    - 若被转换值和精度都是`​0`，则写入单个`​0`​                                
                            - 长度修饰符 
                                - `hho`：期待参数类型`unsigned char`
                                - `ho`：期待参数类型`unsigned short`
                                - `o`：期待参数类型`unsigned int`
                                - `lo`：期待参数类型`unsigned long`
                                - `llo`：期待参数类型`unsigned long long`
                                - `jo`：期待参数类型`uintmax_t`
                                - `zo`：期待参数类型`size_t`
                                - `to`：期待参数类型 *无符号`ptrdiff_t`* 
                        - `x`，`X`：写 *无符号十六进制整数* `hhhh`
                            - 大小写
                                - `x`使用`abcdef`
                                - `X`使用`ABCDEF`
                            - 精度
                                - 默认形式
                                    - 精度指定出现的 *最小数位数* 。默认精度是`1`
                                    - 若被转换的值和精度都是`​0​`，则转换结果 *无字符* 
                                - 替用形式                                 
                                    - 若被转换值非`​0`，结果含有`0x`或`0X`前缀​                                
                            - 长度修饰符 
                                - `hhx`：期待参数类型`unsigned char`
                                - `hx`：期待参数类型`unsigned short`
                                - `x`：期待参数类型`unsigned int`
                                - `lx`：期待参数类型`unsigned long`
                                - `llx`：期待参数类型`unsigned long long`
                                - `jx`：期待参数类型`uintmax_t`
                                - `zx`：期待参数类型`size_t`
                                - `tx`：期待参数类型 *无符号`ptrdiff_t`* 
                        - `f`，`F`：写 *十进制浮点数* `[-]ddd.ddd`
                            - 精度
                                - 默认形式
                                    - 精度指定 *小数点后位数* 。默认精度是`6`
                                - 替用形式
                                    - 即使没有小数点后数位也写小数点
                            - 无穷大和非数的转换样式见注意
                            - 长度修饰符 
                                - `f`：期待参数类型`double`
                                - `lf`：期待参数类型`double`
                                - `Lf`：期待参数类型`long double`
                        - `e`，`E`：写 *十进制科学计数法浮点数* 
                            - 大小写
                                - `e`：使用`[-]d.ddde±dd`
                                - `E`：使用`[-]d.dddE±dd`
                            - 指数
                                - 指数至少含二个数位，仅当所需时使用更多数位
                                - 若值为`0`，则指数亦为`​0​`
                            - 精度
                                - 默认形式
                                    - 精度指定 *小数点后位数* 。默认精度是`6`
                                - 替用形式
                                    - 即使没有小数点后数位也写小数点
                            - 无穷大和非数的转换样式见注意
                            - 长度修饰符 
                                - `e`：期待参数类型`double`
                                - `le`：期待参数类型`double`
                                - `Le`：期待参数类型`long double`
                        - `a`，`A`：写 *十六进制浮点数* 
                            - 大小写
                                - `a`：使用`[-]0xh.hhhp±d`
                                - `A`：使用`[-]0Xh.hhhP±d`
                            - 指数  
                                - 若参数是正规化的浮点值，则首个十六进制数位非`0`
                                - 若值为`0`，则指数亦为`​0`
                            - 精度
                                - 默认形式
                                    - 精度指定 *小数点后位数* 。默认精度足以准确表示该值
                                - 替用形式
                                    - 即使没有小数点后数位也写小数点
                            - 无穷大和非数的转换样式见注意
                            - 长度修饰符 
                                - `a`：期待参数类型`double`
                                - `la`：期待参数类型`double`
                                - `La`：期待参数类型`long double`
                        - `g`，`G`：写 *十进制小数或十进制科学计数法浮点数* ，取决于值和精度
                            - 大小写
                                - `g`：使用`e`，`f`
                                - `G`：使用`E`，`F`
                            - 令`P`等于精度
                                - 若它非零，若精度未指定则为`6`，若精度为`​0`则等于`1`
                                - 若带样式`E`的转换会有指数`X`，则
                                    - 若`P > X >= −4`，则转换带`f`或`F`风格，及精度`P − 1 − X`
                                    - 否则，转换带`e`或`E`风格，及精度`P − 1`
                            - 除非要求替用表示，否则末尾零会被移除，且若不留下小数部分则小数点亦被移除
                            - 无穷大和非数的转换样式见注意
                            - 长度修饰符 
                                - `g`：期待参数类型`double`
                                - `lg`：期待参数类型`double`
                                - `Lg`：期待参数类型`long double`
                        - `n`： *返回* *迄今写入的字符数* 
                            - 结果被 *写入到参数所指向的值* 。该指定**不可**含有任何标签、域宽或精度
                            - 长度说明符
                                - `hhn`：期待参数类型`signed char *`或`unsigned char *`
                                - `hn`：期待参数类型`signed short *`或`unsigned short *`
                                - `n`：期待参数类型`signed int *`或`unsigned int *`
                                - `ln`：期待参数类型`signed long *`或`unsigned long *`
                                - `lln`：期待参数类型`signed long long *`或`unsigned long long *`
                                - `jn`：期待参数类型`intmax_t *`或`uintmax_t *`
                                - `zn`：期待参数类型 *有符号`size_t *`* 
                                - `tn`：期待参数类型`ptrdiff_t *`
                        - `p`：写 *指针* ，具体格式由实现定义，需要与`scanf("%p")`保持一致
                            - 长度说明符
                                - `p`：期待参数类型`void *`
                    - 注释
                        - 浮点转换函数转换无穷大到`inf`或`infinity`。使用哪个是实现定义的
                        - 非数转换成`nan`或`nan(char_sequence)`。使用哪个是实现定义的
                        - 转换`F`、`E`、`G`、`A`替代上面输出`INF`、`INFINITY`、`NAN`
                        - 尽管`%c`期待`int`，传递`char`是安全的，因为在调用变参数函数时发生 *整数提升* 
                        - 内存写入转换指定符`%n`是安全漏洞的常见目标，这里格式字符串依赖用户输入，而有边界检查的`printf_s`系列函数**不**支持此转换指定符
        - 返回值
            - `(1-2)` 若成功则为写入的字符数，若发生错误则为负值
            - `(3)` 若成功则为写入的字符数（不包含空终止字符），若发生错误则为负值
            - `(4)` 若成功则为会写入充分大缓冲区的字符数（不包含`'\0'`），若发生错误则为负值。因此，当且仅当返回值非负且小于`buf_size`，`'\0'`结尾的字符序列才是被完整写完的
                - 以零`buf_size`和对于`buffer`的 *空指针* 调用`snprintf`可以用于确定容纳输出的所需缓冲区大小
                ```
                const char * fmt = "sqrt(2) = %f";
                int sz = std::snprintf(nullptr, 0, fmt, std::sqrt(2));
                std::vector<char> buf(sz + 1);                          // note + 1 for null terminator
                std::snprintf(&buf[0], buf.size(), fmt, std::sqrt(2));
                ```
        - 示例
        ```
        printf(    "[% 15lf, % 15lf]\n", 10.1234, -10.1234);  // [      10.123400,      -10.123400]
        printf("[%# 15.lf, %# 15.lf]\n",    10.0,    -10.0);  // [            10.,            -10.]
        printf(  "[% 15.lf, % 15.lf]\n",    10.0,    -10.0);  // [             10,             -10]
        printf("[% 15.8lf, % 15.8lf]\n", 10.1234, -10.1234);  // [    10.12340000,    -10.12340000]
        printf(  "[% 15.8d, % 15.8d]\n",      10,      -10);  // [       00000010,       -00000010]
        printf("[%+15.8lf, %+15.8lf]\n", 10.1234, -10.1234);  // [   +10.12340000,    -10.12340000]
        printf("[%-15.8lf, %-15.8lf]\n", 10.1234, -10.1234);  // [10.12340000    , -10.12340000   ]
        printf("[%015.8lf, %015.8lf]\n", 10.1234, -10.1234);  // [000010.12340000, -00010.12340000]
        ```
- `C`风格文件寻位
    - [`ftell`](https://en.cppreference.com/w/cpp/io/c/ftell)：返回当前文件位置指示器 
        - 签名
        ```
        long ftell(FILE * stream);
        ```
        - 返回文件流`stream`的文件位置指示器的当前值
            - 若流以 *二进制模式* 打开，则此函数获得的值是 *距文件起始的字节数* 
            - 若流以 *文本模式* 打开，则此函数的返回值是 *未指定* 的，而且仅若作为`fseek`的输入才有意义
        - 返回值
            - 成功时为文件位置指示器，
            - 失败出现则为`-1L`。失败时亦设置`errno` 
    - [`fgetpos`](https://en.cppreference.com/w/cpp/io/c/fgetpos)：获取文件位置指示器 
        - 签名
        ```
        int fgetpos(FILE * stream, fpos_t * pos);
        ```
        - 获得文件流`stream`的文件位置指示器和当前分析状态（若存在），并将它们存储于`pos`所指向的对象。存储的值仅在作为`fsetpos`的输入的情况有意义
        - 返回值
            - 成功时为`​0​`
            - 否则 *非零* 值。失败时还设置`errno` 
    - [`fseek`](https://en.cppreference.com/w/cpp/io/c/fseek)：移动文件位置指示器到文件中的指定位置 
        - 签名
        ```
        int fseek(FILE * stream, long offset, int origin);
        ```
        - 设置文件流`stream`的 *文件位置指示器* 为`origin`后的`offset`个字节
            - `origin`可以选择
                - `SEEK_SET`，从 *文件起始* 寻位
                - `SEEK_CUR`，从 *当前文件位置* 寻位
                - `SEEK_END`：从 *文件尾* 寻位
            ```
            // <stdio.h>
            /* The possibilities for the third argument to `fseek'.
               These values should not be changed.  */
            #define SEEK_SET   0    /* Seek from beginning of file.  */
            #define SEEK_CUR   1    /* Seek from current position.  */
            #define SEEK_END   2    /* Seek from end of file.  */
            ```
            - 若`stream`以 *二进制模式* 打开，则新位置准确地是`origin`后的`offset`个字节
                - 不要求二进制流支持`SEEK_END`，尤其是是否输出附加的空字节。
            - 若`stream`以 *文本模式* 打开，则
                - `offset`值可以零（可用于任何`origin`），或
                - 若`origin == SEEK_SET`，则`offset`还可以是先前在关联到同一个文件的流上调用`ftell`的返回值
        - 除了更改文件位置指示器，`fseek`还撤销`ungetc`的效果并清除文件尾状态
        - 若发生读或写错误，则设置流的 *错误指示器* （`ferror`）而**不**影响文件位置
        - 返回值
            - 成功时为`​0​`
            - 否则为 *非零* 
    - [`fsetpos`](https://en.cppreference.com/w/cpp/io/c/fsetpos)：移动文件位置指示器到文件中的指定位置 
        - 签名
        ```
        int fsetpos(FILE * stream, const fpos_t * pos);
        ```
        - 按照`pos`所指向的值，设置文件流`stream`的 *文件位置指示器* 和 *宽字节分析状态* 
            - 除了建立新的分析状态和位置，调用此函数还会撤销`ungetc`的效果，并清除文件尾状态
            - 若读或写失败，则设置流的 *错误指示器* （`ferror`） 
        - 返回值
            - 成功时为`​0​`
            - 否则 *非零* 值。失败时还设置`errno` 
    - [`rewind`](https://en.cppreference.com/w/cpp/io/c/rewind)：移动文件位置指示器到文件起始 
        - 签名
        ```
        void rewind(FILE * stream);
        ```
        - 移动 *文件位置指示器* 到给定文件流的起始
            - 函数等价于在调用`fseek(stream, 0, SEEK_SET);`的同时还 *清除* *文件尾指示器* 和 *错误指示器* 
            - 此函数丢弃任何来自先前对`ungetc`调用的效果
- `C`风格`I/O`错误处理
    - [`clearerr`](https://en.cppreference.com/w/cpp/io/c/clearerr)：清除错误
        - 签名
        ```
        void clearerr(FILE * stream);
        ```
        - 重置给定文件流的 *错误指示器* 和 *文件尾指示器* 
    - [`feof`](https://en.cppreference.com/w/cpp/io/c/feof)：检查文件尾 
        - 签名
        ``` 
        int feof(FILE * stream);
        ```
        - 检查是否已抵达给定文件流的结尾
        - 返回值
            - 若已抵达文件流尾则为 *非零值* 
            - 否则为`​0​`
        - 用例
        ```
        #include <cstdlib>
        #include <cstdio>
        
        int main()
        {
            FILE * fp = fopen("test.txt", "r");
            
            if (!fp) 
            {
                std::perror("File opening failed");
                return EXIT_FAILURE;
            }
         
            int c;  // note: int, not char, required to handle EOF
            
            while ((c = std::fgetc(fp)) != EOF) 
            { 
                // standard C I/O file reading loop
                putchar(c);
            }
         
            if (ferror(fp))
            {
                puts("I/O error when reading");
            }
            else if (feof(fp))
            {
                puts("End of file reached successfully");
            }
                
            fclose(fp);
            
            return EXIT_SUCCESS;
        }
        ```
    - [`ferror`](https://en.cppreference.com/w/cpp/io/c/ferror)：检查文件错误 
        - 签名
        ```
        int ferror(FILE * stream);
        ```
        - 检查给定的流的错误
        - 返回值
            - 若文件流已出现错误则为 *非零值* 
            - 否则为`​0​`
    - [`perror`](https://en.cppreference.com/w/cpp/io/c/perror)：显示对应当前错误的字符串于`stderr`
        - 签名
        ```
        void perror(const char * s);
        ```
        - 打印当前存储于系统变量`errno`的错误码对应的错误信息到`stderr`
        - 通过连接下列组分构成描述
            - `s`所指向的空终止字节字符串的内容后随`": "`（除非`s`为 *空指针* 或`s`所指向字符为 *空字符* ）
            - 由实现定义的，描述存储于`errno`的错误码的错误消息字符串后随`'\n'`。错误消息字符串等同于`strerror(errno)`的结果
        - 示例
        ```
        double not_a_number = std::log(-1.0);
        
        if (errno == EDOM) 
        {
            perror("log(-1) failed");  // log(-1) failed: Numerical argument out of domain
        }
        ```






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
    ```
    template <class T,
              class Container = std::deque<T>> 
    class stack;
    ```
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
    ```
    template <class T,
              class Container = std::deque<T>> 
    class queue;
    ```
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
        - 重载`<`运算符
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
    - `c.data()`：返回指向作为元素存储工作的底层数组的指针
        - 指针满足范围`[data(), data() + size())`始终是合法范围
        - 若容器为空，则`data()`可能或可能不返回 *空指针* （但不论如何，该情况下`data()`**不可**解引用）
    ```
    // 准备文件
    std::ofstream("test.txt") << 1 << ' ' << 2 << std::endl;
   
    if (FILE * fp = fopen("test.txt", "r"))
    {
        // char 可平凡复制
        std::vector<char> buf1(4); 
        std::fread(&buf1[0], sizeof buf1[0], buf1.size(), fp);
        fclose(fp);
    }
 
    for (char n : buf1)
    {
        std::cout << n << std::endl;
    }
    ```
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
    std::string std::to_string(int value);                 (1)  // std::sprintf(buf, "%d", value)
    std::string std::to_string(long value);                (2)  // std::sprintf(buf, "%ld", value)
    std::string std::to_string(long long value);           (3)  // std::sprintf(buf, "%lld", value)
    std::string std::to_string(unsigned value);            (4)  // std::sprintf(buf, "%u", value)
    std::string std::to_string(unsigned long value);       (5)  // std::sprintf(buf, "%lu", value)
    std::string std::to_string(unsigned long long value);  (6)  // std::sprintf(buf, "%llu", value)
    std::string std::to_string(float value);               (7)  // std::sprintf(buf, "%f", value)
    std::string std::to_string(double value);              (8)  // std::sprintf(buf, "%f", value)
    std::string std::to_string(long double value);         (9)  // std::sprintf(buf, "%Lf", value)
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
    // The lambda expression 
    // is a prvalue expression of unique unnamed non-union non-aggregate class type, 
    // known as closure type, 
    // which is declared (for the purposes of ADL) in the smallest 
    // block scope, class scope, or namespace scope 
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

- 可以使用`auto`，`std::function`或 *函数指针* 接收`lambda`表达式
```
auto f1 = [capture_list] (paramater_list) -> return_type { function_body; };

// type casts, do NOT use these!!!
std::function<return_type (paramater_list)> f2                  = f1;
return_type                               (*f3)(paramater_list) = f1;
```
- `注意`：`lambda`的实际类型就是匿名类型`lambda`，既**不是**`std::function`也**不是**函数指针
```
#include <bits/stdc++.h>
#include <boost/core/demangle.hpp>

int main(int argc, char * argv[])
{
    auto cmp1 = [] (int a, int b) { return a < b; };
    std::function<bool (int, int)> cmp2 = cmp1;
    bool (*cmp3)(int, int) = cmp1;

    std::cout << boost::core::demangle(typeid(cmp1).name()) << std::endl;
    std::cout << boost::core::demangle(typeid(cmp2).name()) << std::endl;
    std::cout << boost::core::demangle(typeid(cmp3).name()) << std::endl;

    return EXIT_SUCCESS;
}

// OUTPUT:
main::{lambda(int, int)#1}
std::function<bool (int, int)>
bool (*)(int, int)
```
- **应该**使用`auto`接收`lambda`表达式
    - 乱用`std::function`或 *函数指针* 接收`lambda`表达式将会导致类型转换和其他严重的性能损失！
    - 例如，使用`std::bind`的时候，乱用`std::function`会导致严重的性能损失
    ```
    using std::placeholders::_1, std::placeholders::_2;
    std::function<bool(T, T)> cmp1 = std::bind(f, _2, 10, _1);  // bad
    auto cmp2 = std::bind(f, _2, 10, _1);                       // good
    auto cmp3 = [] (T a, T b) { return f(b, 10, a); };          // also good

    std::stable_partition(std::begin(x), std::end(x), cmp?);
    ```
    - 对于`cmp2`和`cmp3`，泛型算法能够`inline`谓词调用
    - 但对于`cmp1`
        - 首先，显式创建函数对象有类型转换的性能损失
        - 其次，本来能`inline`的这回也`inline`不了了
        - 最后，`std::function`内部是 *多态* 的，还搭上了 *运行时类型识别* 耗费的时间

#### 内容物

- Capture List `[capture_list]`: 
    - The capture list is a comma-separated list of zero or more `capture`s, optionally beginning with the `capture-default`. 
    - The capture list defines the outside variables that are accessible from within the lambda function body. 
    - Possible `capture-default`s (**not** recommended to use): 
      1. `&` (implicitly capture the used automatic variables by reference)
      2. `=` (implicitly capture the used automatic variables by copy).
    - Possible `capture`s:
      1. `identifier`: simple by-copy capture
      2. `identifier ...`: simple by-copy capture that is a pack expansion
      3. `identifier initializer`: by-copy capture with an initializer
      4. `& identifier`: simple by-reference capture
      5. `& identifier ...`: simple by-reference capture that is a pack expansion
      6. `& identifier initializer`: by-reference capture with an initializer
      7. `this`: simple by-reference capture of the current object
      8. `* this`: simple by-copy capture of the current object
      9. `... identifier initializer`: by-copy capture with an initializer that is a pack expansion
      10. `& ... identifier initializer`:  by-reference capture with an initializer that is a pack expansion
    - 把`lambda`表达式 *所在的函数中的局部非静态变量* 声明在捕获列表里，就可以在`lambda`表达式函数体使用该变量
    - 对于局部静态变量或者全局变量，则**不需捕获**即可使用
    - 捕获方式：与参数传递方式类似，可以是
        - *值捕获* ：捕获被创建时变量的 *拷贝* 
            - *可变 `lambda`* （mutable `lambda`）
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
- 参数列表 Parameter List `(params)`
    - The list of parameters, as in named functions. `auto` type parameter is accepted `(since C++17)`. 
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
    // note: at least on gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0 this one runs correctly with -std=c++11
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
        auto bound_f = std::bind(f, n1, std::ref(n2), std::cref(n3));
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
    - 至少`gcc`允许自减尾后迭代器`--c.end()`获取尾元素
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

- *插入迭代器* （insert iterators）
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
    - 虽然`std::iostream`类型不是容器，但标准库定义了可以用于这些`I/O`类型对象的迭代器
        - [`std::istream_iterator`](https://en.cppreference.com/w/cpp/iterator/istream_iterator)：读取输入流
        - [`std::ostream_iterator`](https://en.cppreference.com/w/cpp/iterator/ostream_iterator)：向输出流写数据
        - [`std::istreambuf_iterator`](https://en.cppreference.com/w/cpp/iterator/istreambuf_iterator)
        - [`std::ostreambuf_iterator`](https://en.cppreference.com/w/cpp/iterator/ostreambuf_iterator)
    - Refer to 
      [Effective STL Notes](./effective_cpp_notes_03_effective_stl.md) 
      Item 29 for details on stream buffer iterators.
        - They are more efficient than regular stream itertors when dealing non-formatted raw character input.  
    - 这些迭代器将它们对应的流当做一个特定类型的元素序列来处理
        - 通过使用 *流迭代器* ，我们可以使用泛型算法从流对象读取数据或向其写入数据
    - `std::istream_iterator`操作
        - `std::istream_iterator<T> in(is);`：`in`从输入流`is`读取类型为`T`的值
        - `std::istream_iterator<T> end;`：默认初始化，生成读取类型为`T`的值的`std::istream_iterator`，表示 *尾后位置* 
        - `in1 == in2`，`in1 != in2`：`in1`和`in2`必须读取相同的类型，如果它们都是 *尾后迭代器* ，或绑定到相同的输入，则它们相等
        - `*in`：返回从流中读取的值
        - `in->mem`：与`(*in).mem`相同
        - `++in`，`in++`：使用元素类型定义的`>>`运算符从输入流中读取下一个值。与以往相同，前置版本返回一个指向递增后迭代器的引用，后置版本返回旧值
    ```
    std::ifstream fin("afile");
    std::istream_iterator<std::string> str_it(fin);  // reads strings from "afile"
    
    
    // build vector from std::cin
    std::istream_iterator<int> in_iter(std::cin);    // read ints from std::cin
    std::istream_iterator<int> eof_iter;             // istream ''end'' iterator
    std::vector<int> vec;

    while (in_iter != eof_iter)                      // while there's valid input to read
    {
        // postfix increment reads the stream and returns the old value of the iterator
        // we dereference that iterator to get the previous value read from the stream
        vec.push_back(*in_iter++);
    }
    
    // equivalent method to build vector from std::cin
    std::istream_iterator<int> in_iter(std::cin);    // read ints from std::cin
    std::istream_iterator<int> eof_iter;             // istream ''end'' iterator
    std::vector<int> vec(in_iter, eof_iter);         // construct vec from an iterator range
    ```
    - 使用算法操作流迭代器
        - 部分泛型算法可以用于流迭代器
    ```
    istream_iterator<int> in_it(std::cin), eof_it;
    std::cout << std::accumulate(in_it, eof_it, 0) << std::endl;
    ```
    - `std::istream_iterator`允许使用 *懒惰求值* 
        - `std::istream_iterator`绑定到输入流对象时，标准库并**不**保证迭代器立即从流读取数据
            - 具体实现可以推迟从流读取数据，直到我们使用迭代器时才真正读取
            - 标准库实现只保证在第一次解引用迭代器之前，从流中读取数据的操作已经完成了
        - 如果创建`std::istream_iterator`但没有使用就销毁了，或从两个不同的对象同步读取同一个流，那么何时读取就很重要了
    - `std::ostream_iterator`操作
        - `std::ostream_iterator<T> out(os);`：`out`将类型为`T`的值写入到`std::ostream os`中
        - `std::ostream_iterator<T> out(os, d);`：`out`将类型为`T`的值写入到`std::ostream os`中，每个值后面都输出一个`d`，`d`指向`C`风格字符串
        - `out = val;`：用`operator<<`将`val`写入`out`所绑定的`std::ostream`中，`val`的类型必须与`out`可写的类型兼容。 *赋值时可以忽略解引用和递增运算*
        - `*out`，`++out`，`out++`：这些运算符存在，但**不**做任何事，直接返回`out`
    ```
    std::ostream_iterator<int> out_iter(std::cout, " ");
    
    for (int e : vec)
        *out_iter++ = e;     // the assignment writes this element to cout
    std::cout << std::endl;

    for (int e : vec)
        out_iter = e;        // the assignment writes this element to cout
    std::cout << std::endl;

    copy(vec.begin(), vec.end(), out_iter);
    std::cout << std::endl;
    ```
    - 使用流迭代器处理类类型
    ```
    std::istream_iterator<Sales_item> item_iter(std::cin), eof;
    std::ostream_iterator<Sales_item> out_iter(std::cout, "\n");
    
    // store the first transaction in sum and read the next record
    Sales_item sum = *item_iter++;
    
    while (item_iter != eof)
    {
        // if the current transaction (which is stored in item_iter) has the same ISBN
        if (item_iter->isbn() == sum.isbn())
        {
            sum += *item_iter++; // add it to sum and read the next transaction
        }
        else 
        {
            out_iter = sum;      // write the current sum
            sum = *item_iter++;  // read the next transaction
        }
    }
    
    out_iter = sum;              // remember to print the last set of records
    ```
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
    - 解引用移动迭代器生成 *右值引用*
        - 普通迭代器生成 *左值引用*
        - 可以传移动迭代器给`std::uninitialized_copy`（问题是人家`C++17`都有`std::uninitialized_move`了啊233）
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
        - 要求：满足 *严格偏序* （strict partial order）关系
            1. *反自反性* （irreflexivity）：`x < x`永不成立；
            2. *反对称性* （asymmetry）：如`x < y`，则`y < x`不成立；
            3. *传递性* （transitivity）：如`x < y`且`y < z`，则`x < z`。
    - `bool equiv(const T & a, const T & b);`
        - 参数类型：常引用不是强制的，但**不能更改传入的对象**
        - 返回值：`bool`亦不是强制的，但要求可以 *隐式转化* 为`bool`
        - 要求：满足 *等价* （equivalence）关系
            1. *自反性* （irreflexivity）：`x == x`恒为真；
            2. *对称性* （asymmetry）：如`x == y`，则`y == x`；
            3. *传递性* （transitivity）：如`x == y`且`y == z`，则`x == z`。
- 标准库提供以下预定义好的 [*函数对象*](https://en.cppreference.com/w/cpp/utility/functional)（模板类，用时给一个 *调用签名* 并创建对象即可）
    - 算术操作（Arithmetic operations）
        - [`std::plus`](https://en.cppreference.com/w/cpp/utility/functional/plus)：`x + y`
        - [`std::minus`](https://en.cppreference.com/w/cpp/utility/functional/minus)：`x - y`
        - [`std::multiplies`](https://en.cppreference.com/w/cpp/utility/functional/multiplies)：`x * y`
        - [`std::divides`](https://en.cppreference.com/w/cpp/utility/functional/divides)：`x / y`
        - [`std::modulus`](https://en.cppreference.com/w/cpp/utility/functional/modulus)：`x % y`
        - [`std::negate`](https://en.cppreference.com/w/cpp/utility/functional/negate)：`-x`
    - 比较（Comparisons）
        - [`std::equal_to`](https://en.cppreference.com/w/cpp/utility/functional/equal_to)：`x == y`
        - [`std::not_equal_to`](https://en.cppreference.com/w/cpp/utility/functional/not_equal_to)：`x != y`
        - [`std::greater`](https://en.cppreference.com/w/cpp/utility/functional/greater)：`x > y`
        - [`std::less`](https://en.cppreference.com/w/cpp/utility/functional/less)：`x < y`
        - [`std::greater_equal`](https://en.cppreference.com/w/cpp/utility/functional/greater_equal)：`x >= y`
        - [`std::less_equal`](https://en.cppreference.com/w/cpp/utility/functional/less_equal)：`x <= y`
        ```
        std::vector<int> v {0, 1, 1, 2};
        std::sort(v.begin(), v.end(), std::greater<>());
        std::for_each(v.begin(), v.end(), [] (const int & i) { printf("%d ", i); });  // 2 1 1 0
        ```
    - 逻辑操作（Logical operations）
        - [`std::logical_and`](https://en.cppreference.com/w/cpp/utility/functional/logical_and)：`x && y`
        - [`std::logical_or`](https://en.cppreference.com/w/cpp/utility/functional/logical_or)：`x || y`
        - [`std::logical_not`](https://en.cppreference.com/w/cpp/utility/functional/logical_not)：`!x`
    - 位操作（Bitwise operations）
        - [`std::bit_and`](https://en.cppreference.com/w/cpp/utility/functional/bit_and)：`x & y`
        - [`std::bit_or`](https://en.cppreference.com/w/cpp/utility/functional/bit_or)：`x | y`
        - [`std::bit_xor`](https://en.cppreference.com/w/cpp/utility/functional/bit_xor)：`x ^ y`
        - [`std::bit_not`](https://en.cppreference.com/w/cpp/utility/functional/bit_not)：`~x`

### 🌱 [Appendix A] [算法标准库`<algorithm>`](https://en.cppreference.com/w/cpp/algorithm)

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
        - Function object, to be applied to the result of de-referencing every iterator in the range `[first, last)`
        - Signature of the function should be equivalent to the following: `void fun(const Type & a);`
            - The signature does not need to have `const &`
            - `Type` must be such that an object of type `InputIt` can be de-referenced and then implicitly converted to `Type`
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
        - 在 *空容器* 上调用`std::fill_n()`或其他写算法是 *未定义* 行为。对于空容器应当使用`std::back_insert_iterator` => 10.4.1
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
         T & b)
    {
        T tmp = std::move(a);
        a = std::moveE(b);
        b = std::move(tmp);
    }

    template <class T2, std::size_t N>
    void 
    swap(T2 (&a)[N], 
         T2 (&b)[N])
    {
        std::swap_ranges(a, a + N, b);
    }
    ```
    - *互换* 类或数组的内容
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
            using std::swap;
            swap(first[i], first[D(g, param_t(0, i))]);
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
    - 必须定义了 *严格弱序* （stirct weak ordering，例：`<`运算符）
        1. *反自反性* （irreflexivity）：`x < x`永不成立；
        2. *反对称性* （asymmetry）：如`x < y`，则`y < x`不成立；
        3. *传递性* （transitivity）：如`x < y`且`y < z`，则`x < z`。以上三条构成 *严格偏序* 关系（strict partial ordering）的充要条件；
        4. *不可比较性的传递性* （transitivity of incomparability）：如`x`和`y` *不可比* ，即`!(x < y) and !(y < x)`，且`y`与`z`也不可比，则`x`与`z`也不可比。这其实就是 *等价关系* 的传递性。
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
        - `p1.swap(p2)`
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
    - *显式实例化* （explicit instantiation）
        - 模板被实例化的相同实例可能出现在多个对象文件中，会造成严重的额外开销
        - *显式实例化* 用于避免这种额外开销
            - 编译器遇到 *显式模板声明* 时，不会再本文件中生成实例化代码
            - 将一个实例化声明为`extern`就意味着承诺在程序的其他位置会有一个非`extern`声明（定义）
            - 对于一个给定的实例化版本，可能会有多个`extern`声明，但必须 *有且仅有一个实例化定义* 
        ```
        extern template declaration;                            // instantiation declaration
        template declaration;                                   // instantiation definition
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
    ```
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
    - 常常用于 *向下转换* （downcasting）
        - 指 *多态基类* （带有虚函数）的引用或指针向其 *派生类* 的引用或指针的类型转换
        - *向下转换* 也能通过`static_cast`实现
            - `static_cast`**不**进行 *运行时类型检查* （runtime type check）
            - 虽然节省时间，但程序员必须自行保证被转换的引用或指针的动态类型必须就是目标类型或其公有派生类型，否则程序不安全
    - 使用形式
    ```
    dynamic_cast<Type *>(e)     (1)
    dynamic_cast<Type &>(e)     (2)
    dynamic_cast<Type &&>(e)    (3)
    ```
    - 其中
        - `Type`必须是一个 *类类型* ，并且通常情况下应当含有 *虚函数* 
        - 在形式`(1)`中，`e`必须是一个 *合法指针* 的 *纯右值* ，转换结果为 *纯右值* 
        - 在形式`(2)`中，`e`必须是一个 *泛左值* ，转换结果为 *左值* 
        - 在形式`(3)`中，`e`必须是一个`左值或右值 (until C++17)` `泛左值 (since C++17)`，转换结果为 *将亡值* 
        - 在上面所有形式中，`e`的 *动态类型* 必须符合以下三个条件中的任意一个
            - `e`的类型是`Type`的 *公有派生类* 
            - `e`的类型就是`Type`
        - 如果符合，则转换可以成功；否则，转换失败
            - 如果目标是 *指针* 类型，则结果为`0`
            - 如果目标是 *引用* 类型，则还会抛出`std::bad_cast`异常
        - 执行 *向下转换* 时，`e`的静态类型必须是多态的，否则会报编译错误
    - 指针类型的`dynamic_cast`
        - 例如
            - `class Base`至少含有一个虚函数
            - `class Derived : public Base`
            - 如果有`Base * bp`，则可在运行时将其转换成`Derived *`
            ```
            if (Derived * dp = dynamic_cast<Derived *>(bp))
            {
                // use the Derived object to which dp points
            } 
            else 
            { 
                // bp points at a Base object
                // use the Base object to which bp points
            }
            ```
            - 如果`bp`实际指向`Derived`对象，则上述类型转换初始化`dp`并令其指向`bp`所指的`Derived`对象
                - 此时，`if`语句内部使用`Derived`操作的代码时安全的
            - 否则，类型转换的结果为`0`，意味着条件失败，此时`else`执行相应的`Base`操作
        - 我们可以对 *空指针* 执行`dynamic_cast`，结果是所需类型的空指针
        - *在条件部分执行`dynamic_cast`* 定义`dp`的好处
            - 是可以在一个操作中同时完成类型转换和条件检查两项任务
            - `dp`在`if`语句外不可访问，一旦转换失败，即使后面的代码忘了做判断，也不会接触到这个野指针，从而确保程序安全
    - 引用类型的`dynamic_cast`
        - 引用类型的`dynamic_cast`和指针类型的`dynamic_cast`在表示错误发生的方式上略有不同
        - 因为不存在所谓的空引用，所以对于引用类型来说**无法**适用于指针类型完全相同的错误报告策略
        - 当对引用类型的类型转换失败时，程序抛出一个`std::bad_cast`异常
        ```
        try 
        {
            const Derived & d = dynamic_cast<const Derived &>(b);
            // use the Derived object to which b referred
        } 
        catch (std::bad_cast & e) 
        {
            // handle the fact that the cast failed
        }
        ```
- [`typeid`](https://en.cppreference.com/w/cpp/language/typeid)
    - 使用形式
    ```
    typeid(e)
    ```
    - 其中，`e`可以是任意类型的表达式或类型的名字
    - `typeid`返回值类型为`const std::type_info &`，或`std::type_info`的公有派生类型的常引用 => 19.2.4
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
    ```
    // typename demangle is needed for gcc
    
    #include <bits/stdc++.h>
    #include <boost/core/demangle.hpp>
              
    struct A            { virtual void fun() {} };
    struct B : public A {                       };
    struct C            {                       };
    struct D : public C {                       };
    
    A * p1 = new B();
    std::cout << boost::core::demangle(typeid( p1).name()) << '\n';  // A*
    std::cout << boost::core::demangle(typeid(*p1).name()) << '\n';  // B
    
    C * p2 = new D();
    std::cout << boost::core::demangle(typeid( p2).name()) << '\n';  // C*
    std::cout << boost::core::demangle(typeid(*p2).name()) << '\n';  // C
    
    auto t0 = std::make_tuple(10, "hehe", 3.14);
    std::cout << boost::core::demangle(typeid(t0).name()) 
              << '\n';  // std::tuple<int, char const*, double>

    auto t1 = std::forward_as_tuple(10, "hehe", 3.14);
    std::cout << boost::core::demangle(typeid(t1).name()) 
              << '\n';  // std::tuple<int&&, char const (&) [5], double&&>
    ```

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

- *成员指针* （pointer to member）是指可以指向类的非静态成员的指针
    - 一般情况下，指针指向对象
    - 成员指针指向的是对象的成员，而不是对象本身
    - 类的静态成员**不**属于任何对象因此无须特殊的指向静态成员的指针，普通指针即可胜任
- 成员指针的类型囊括了类的类型以及成员的类型
    - 初始化一个这样的指针时，我们令其指向类的某个成员，但是不指定该成员所属的对象
    - 直到使用成员指针时，才提供成员所属的对象
- 示例类
```
class Screen 
{
public:
    typedef std::string::size_type pos;
    
    char get_cursor() const { return contents[cursor]; }
    char get() const;
    char get(pos ht, pos wd) const;

private:
    std::string contents;
    pos cursor;
    pos height, width;
};
```
- *数据成员指针* （Pointers to Data Members）
    - 定义
        - 定义指向`const std::string`的`Screen`类数据成员指针
        ```
        const std:string Screen::* pdata;
        ```
        - 初始化或赋值时需指定它所指的成员
        ```
        pdata = &Screen::contents;
        ```
        - 还可以使用`auto`或`decltype`
        ```
        auto pdata = &Screen::contents;
        ```
    - 使用
        - 当初始化成员指针或为其赋值时，它**并未**指向任何数据
            - 成员指针定义了成员而非该成员所属的对象
            - 只有当需要解引用成员指针时，我们才需要提供对象
        - *成员指针访问运算符* `.*` `->*`
        ```
        Screen myScreen, 
        // .* dereferences pdata to fetch the contents member from the object myScreen
        const std::string s1 = myScreen.*pdata;
        
        // ->* dereferences pdata to fetch contents from the object to which pScreen points
        Screen * pScreen = &myScreen;
        const std::string s2 = pScreen->*pdata;
        ```
    - 返回数据成员指针的函数
        - 常规的访问控制规则对成员指针同样有效
            - 例如，先前的`Screen`类的`content`成员为私有的，因此之前对于`pdata`的使用必须位于`Screen`类内部或其友元中，否则将报错
        - 获取私有数据成员指针
            - 因为数据成员一般是私有的，所以我们通常不能直接获取数据成员的指针
            - 如果一个像`Screen`这样的类希望我们可以访问它的`content`成员，最好定义一个函数，令其返回值是指向该成员的指针
            ```
            class Screen 
            {
            public:
                // data is a static member that returns a pointer to member
                static const std::string Screen::* data() { return &Screen::contents; }
                // other members as before
            };
            
            // data() returns a pointer to the contents member of class Screen
            const std::string Screen::* pdata = Screen::data();
            // fetch the contents of the object named myScreen
            std::string s = myScreen.*pdata;
            ```
- *成员函数指针* （Pointers to Member Functions）
    - 定义
        - 使用`auto`或显式接收
            - 如果 *成员函数存在重载* 的问题，则必须显式声明函数类型以明确指出想要使用的是哪个函数
        ```
        // pmf is a pointer that can point to a Screen member function that is const
        // that returns a char and takes no arguments
        auto pmf = &Screen::get_cursor;

        char (Screen::* pmf2)(Screen::pos, Screen::pos) const;
        pmf2 = &Screen::get;
        ```
        - 由于优先级问题，和普通函数指针一样，括号必不可少
        ```
        // error: 
        // nonmember function p(Screen::pos, Screen::pos) returning char Screen::* 
        // cannot have a const qualifier
        char Screen::* p(Screen::pos, Screen::pos) const;
        ```
        - 成员函数和指向该成员的指针之间**不**存在自动转换规则
        ```
        // pmf points to a Screen member that takes no arguments and returns char
        pmf = &Screen::get;  // must explicitly use the address-of operator
        pmf = Screen::get;   // error: no conversion to pointer for member functions
        ```
    - 使用
        - 同样使用 *成员指针访问运算符* `.*` `->*`作用于指向成员函数的指针，以及调用类的成员函数
        ```
        // passes the arguments 0, 0 to the two-parameter version of get on the object myScreen
        Screen myScreen；
        char c1 = (myScreen.*pmf2)(0, 0);
        
        // call the function to which pmf points on the object to which pScreen points
        Screen * pScreen = &myScreen;
        char c2 = (pScreen->*pmf)();
        ```
        - 同样，由于优先级问题，括号必不可少
        ```
        // this one
        myScreen.*pmf()
        
        // is equivalent to
        myScreen.*(pmf())
        ```
        - 因为函数调用优先级比交个哦，所以在声明指向成员的函数指针并使用这样的指针进行函数调用时，括号必不可少
            - `(C::*p)(params)`
            - `(obj.*p)(params)`
    - 成员指针类型别名
        - 使用`typedef`或 *类型别名* 可以让成员指针更容易理解
        ```
        // Action is a type that can point to a member function of Screen
        // that returns a char and takes two pos arguments
        using Action = char (Screen::*)(Screen::pos, Screen::pos) const;
        
        // get points to the get member of Screen
        Action get = &Screen::get;
        
        // action takes a reference to a Screen and a pointer to a Screen member function
        Screen & action(Screen &, Action = &Screen::get);
        
        Screen myScreen;
        
        // equivalent calls:
        action(myScreen);                // uses the default argument
        action(myScreen, get);           // uses the variable get that we previously defined
        action(myScreen, &Screen::get);  // passes the address explicitly
        ```
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
    - 使用[`std::mem_fn`](https://en.cppreference.com/w/cpp/utility/functional/mem_fn)生成可调用对象
        - 使用`std::mem_fn`来从成员函数指针生成函数对象，且成员类型由编译器自动推断，无需用户自行指定
        ```
        std::find_if(svec.begin(), svec.end(), std::mem_fn(&std::string::empty));
        ```
        - `std::mem_fn`生成的可调用对象可以通过对象调用，也可以通过指针调用
        ```
        auto f = mem_fn(&string::empty);  // f takes a std::string or a std::string *
        f(*svec.begin());                 // ok: passes std::string &; f uses .* to call empty
        f(&svec[0]);                      // ok: passes std::string *; f uses .-> to call empty
        ```
        - 实际上，我们可以认为`std::mem_fn`生成的可调用对象含有一对 *重载的函数调用运算符* 
            - 一个接受`std::string &`实参
            - 另一个接受`std::string *`实参
    - 使用[`std::bind`](https://en.cppreference.com/w/cpp/utility/functional/bind)生成可调用对象
        - 还可以使用`std::bind`从成员函数生成可调用对象
        ```
        // bind each string in the range to the implicit first argument to empty
        auto it = std::find_if(svec.begin(), svec.end(), std::bind(&std::string::empty, std::placeholders_1));
        ```
        - 和`std::function`类似的地方是，当我们使用`std::bind`时，必须将函数中用于表示执行对象的隐式形参转换成显式的
        - 和`std::mem_fn`类似的地方是，`std::bind`生成的可调用对象的第一个实参既可以是`std::string &`，又可以是`std::string *`
        ```
        auto f = std::bind(&std::string::empty, std::placeholders::_1);
        f(*svec.begin());  // ok: argument is std::string &, f will use .* to call empty
        f(&svec[0]);       // ok: argument is std::string *, f will use .-> to call empty
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


