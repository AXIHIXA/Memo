# 《C++ Primer 5th Edition》拾遗

记录一些对C++理解得不到位的地方。

### 🌱 一句话

- 常见规则
    - `const`常量不论是声明还是定义都添加`extern`修饰符；
    - 想要`auto`推导出引用或者常量的话，直接写清楚是坠吼的（`const auto & a = b`），别折腾顶层`const`什么的；
    - 认定应为常量表达式的变量应当声明为`constexpr`类型；
    - `constexpr`函数、静态`constexpr`成员、`inline`函数（包括类的`inline`成员函数）以及模板的**定义和实现都应**写进头文件；
    - `using`声明（`using std::string`、`using namespace std`、`using intptr = int *`等）**不应**写进头文件；
    - `for each`循环内以及使用迭代器时**不能**改变被遍历的容器的大小；
    - 现代`C++`应使用标准库类型配合迭代器，而**不是**`C`风格的数组和指针。数组也是一种迭代器；
    - 现代`C++`**不应**使用旧式的强制类型转换，应当明确调用对应的`xx_cast<T>(expr)`；
    - 除非必须，**不要**使用自增自减运算符的后置版本（会造成性能浪费）；
    - **不在**内部作用域声明函数（内部作用域生命的东西会覆盖外部作用域的同名东西，可能会影响函数重载的使用）；
    - 构造函数**不应**该覆盖掉类内初始值，除非新值与原值不同；不使用类内初始值时，则每个构造函数**都应显式初始化**每一个类内成员；
    - 希望类的所有成员都是`public`时，**应**使用`struct`；只有希望使用`private`成员时才用`class`；
    - 在类定义开始或结束的地方**集中声明**友元；使用友元，仍另需有一个**单独的函数声明**；
    - 类的类型成员（`typedef`以及`using`声明）应该放在类定义**刚开始**的地方的`public`区域； 
    - 最好令构造函数初始化列表的顺序与成员声明的顺序**保持一致**；**避免**用某些成员初始化其他成员，用构造函数的参数作为初始值；
    - 应把静态数据成员的定义与其他非内联函数的定义放在**同一个文件**中；
    - 即使一个`constexpr`静态成员在类内部被初始化了，也应该在类外定义一下该成员（此时**不能**再指定初始值）；
    - 不需要写访问时，应当使用`const_iterator`；
    - 改变容器内容之后，一律手动更新迭代器；
    - 泛型编程要求：**应当**统一使用非成员版本的`swap`，即`std::swap(c1, c2);`；
    
- 一些常识
    - `C++11`规定整数除法商一律向0取整（即：**直接切除小数部分**）；
    - 指针解引用的结果是其指向对象的**左值**引用；
    - `*iter++`等价于`*(iter++)` => 优先级：`++` > `*`；
    - `p->ele`等价于`(*p).ele` => 优先级：`.` < `*`；
    - `const`对象被设定为**仅在文件内有效**；
    - `std::endl`有刷新缓冲区的效果。最好带上；
    - 如果一个函数是永远也不会用到的，那么它可以只有声明而没有定义 => 15.3；
    - 引用从来都是作为被引用对象的同义词出现（比如`auto`就不能自动推断出引用），唯一例外是`decltype`。它会原样保留引用以及顶层`const`；
    - `main`函数不能递归调用、不能重载；
    - 定义在类内部的函数是隐式的`inline`函数；
    - 使用`struct`或`class`定义类的**唯一区别**就是默认访问权限：`struct`中默认`public`，而`class`默认`private`；
    - 每个类定义了**唯一**的类型；两个类即使内容完全一样，它们也是不同的类型，**不能**自动相互转化；
    - 如果一个构造函数为每一个参数都提供了默认实参，则它实际上也定义了默认构造函数；
    - 能通过一个实参调用的构造函数定义了一条从构造函数的参数类型向类类型隐式转换的规则；
    - 非`constexpr`静态成员只能在**类外**初始化；在类外部定义静态成员时，**不能**重复`static`关键字；
    - 如果容器为空，则`begin`和`end`返回的是**同一个**迭代器，都是尾后迭代器；
    - 只有当其元素类型也定义了相应的比较运算符时，才可以使用关系运算符来比较两个容器；
    
- 读代码标准操作
    - 对复杂的声明符，从变量名看起，先往右，再往左，碰到一个圆括号就调转阅读的方向；
      括号内分析完就跳出括号，还是按先右后左的顺序，如此循环，直到整个声明分析完；
        - 举例：`int (*(*pf)(int, int (*(*)(int))[20]))[10]`：
            - 按顺序翻译为：declare `pf` as pointer to function (int, pointer to function (int) returning pointer to array 20 of int) returning pointer to array 10 of int；
        - 大宝贝：[cdecl](https://cdecl.org/) ，安装：`sudo apt install cdecl`。
    - 判断复杂类型`auto`变量的类型：先扒掉引用，再扒掉被引用者的顶层`const`；

### 🌱 初始化

#### 显式初始化和隐性初始化

```
// 显式初始化

int a = 1;
int a(1);
int a = {1};     // 列表初始化。如会损失精度，则CE
int a{1};        // 列表初始化。如会损失精度，则CE

// e.g.
int a = {3.14};  // 列表初始化。会损失精度，报CE
int a{3.14};     // 列表初始化。会损失精度，报CE

// 隐式初始化

- 内置变量且在函数体之外，隐式初始化为0
- 函数体内的静态变量，隐式初始化为0
- 内置变量且在函数体之内，**无**隐式初始化 => 未初始化的内置类型局部变量将产生未定义的值
- 自定义对象：由类决定是否允许隐式初始化以及初始化为何值
```

#### 拷贝初始化和直接初始化

- 如果初始化时使用了等号，则是拷贝初始化（生成并直接初始化临时右值对象，再将临时对象拷贝到左值），有性能损失：
```
std::string s = std::string("hehe"); 
// 实际执行时等价于：
std::string tmp("hehe"); 
std::string s = tmp; 
```
- 如不使用等号，则是直接初始化。

### 🌱 `extern`修饰符

#### 声明和定义

```
int a;             // 这其实是声明并定义了变量a
extern int a;      // 这才是仅仅声明而不定义
extern int a = 1;  // 这是声明并定义了变量a并初始化为1。“任何包含显式初始化的声明即成为定义，如有extern则其作用会被抵消”
```

#### `const`常量不论是声明还是定义都添加`extern`修饰符

- 默认状态下，`const`对象仅在文件内有效
- 如果想在多个文件之间共享`const`对象，则必须在定义的对象之前添加`extern`关键字。

```
extern const int BUF_SIZE = fcn();  // globals.cpp
extern const int BUF_SIZE;          // globals.h
extern const int BUF_SIZE;          // sth.h （其他要用到`BUF_SIZE`的头文件）
```

编译器在编译过程中会把所有的`const`变量都替换成相应的字面值。
为了执行上述替换，编译器必须知道变量的初始值。
如果程序包含多个文件，则每个用了`const`对象的文件都必须得能访问到它的初始值才行。
要做到这一点，就必须在每一个用到变量的文件之中都有它的定义。
为了支持这一用法，同时避免对同一变量的重复定义，默认情况下，`const`对象被设定为仅在文件内有效。
当多个文件中出现了同名的`const`变量时，其实等同于在不同文件中分别定义了**独立的**变量。

如果希望`const`对象只在一个文件中定义一次，而在多个文件中声明并使用它，则需采用上述操作。


### 🌱 复合类型（指针、引用）

- 指针`*`以及引用`&`只从属于某个声明符，而不是基本数据类型的一部分

- 指针解引用的结果是被引用对象的左值引用

- 指针或引用的类型必须与其所引用的对象的**类型严格一致**（除下一条的2个例外），即：
    - 指针只能用同类型的其他指针（包括字面量或对象强制转换成的指针，以及取值符获取的地址），或者`NULL`、`nullptr`赋值；
    - `double`字面量或变量都不能强转成指针；
    - 引用只能绑定到同类型的对象上；
    - 对于常量，只能绑定常量指针或常引用，不能绑定普通指针或普通引用。
    
- 上一条有2个例外：
    - 指针：常量指针指向非常量对象；基类指针指向派生类对象
    - 引用：常引用绑定到任何能转化为本类型常引用的对象（包括字面值）上；基类引用绑定到派生类对象上
        - **常引用可以绑定在【其它类型】的【右值】上**。尤其，允许为一个常引用绑定：
            - 非常量的对象
            - 字面值
            - 一般表达式

```
double i = 4.2;
const int & r1 = i;      // 正确：we can bind a const int& to a plain int object
const int & r2 = 4.2;    // 正确：r1 is a reference to const
const int & r3 = i * 2;  // 正确：r3 is a reference to const
int & r4 = i * 2;        // 错误：r4 is a plain reference, not const reference

// 注：执行如下代码时：

double pi = 3.1415926;  
const int & a = pi;      // ok

// 实际上编译器干了这么件事：

int tmp = pi;
const int & a = tmp;

// 如果不是常量引用，改的就不是`pi`而是临时量`tmp`，容易造成人祸，因此`C++`直接规定非常量引用不能绑定给临时量。
```

- `常量指针`（指针指向常量）和`指针常量`（指针本身是常量）不一样：

```
int num = 1;  
const int * p1 = &num;        // 指向`const int`的指针。不能用p1修改num的值，但可以让p1指向别的`(const) int`变量。
int * const p2 = &num;        // 指向`int`的常指针。不能让p1指向别的`int`变量，但可以用p1修改num的值。
const int * const p2 = &num;  // 指向`const int`的常指针。既不能用p1修改num的值，也不可以让p1指向别的`int`变量。
```

- `顶层const`和`底层const`
    - 顶层`const`（Top-level `const`）：任意的对象是常量
    - 底层`const`（Low-level `const`）：仅限指针或引用。指向的那个对象本身是常量
        - 注意，引用一旦绑定就永远不能改了，因此普通引用以及常引用本身都自带顶层`const`；
        - 常量引用永远都是**底层**`const`；
        - 对常量对象取地址是**底层**`const`。
    ```
    int i = 0;
    int * const p1 = &i;        // 顶层const
    const int ci = 42;          // 顶层const
    const int * p2 = &ci;       // 底层const
    const int * const p3 = p2;  // 第一个const为底层，第二个为顶层
    const int & r = ci;         // 常量引用永远都是底层const
    ```
    - 附赠一条知乎高票回答：“顶层”和“底层”的翻译很容易让人误解为就只有两层，实际上当然是不是的。首先我们假设有这样的代码：
    ```
    template<typename T> using Const = const T;
    template<typename T> using Ptr = T *;

    const int *** const shit = nullptr;
    
    // 要怎么看呢？很简单，不要用`const`和`*`，用`Const`和`Ptr`来表达，马上明白：
    Const<Ptr<Ptr<Ptr<Const<int>>>>> shit = nullptr;
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

#### `auto`类型说明符

- `auto`定义的变量必须有初始值，（编译器通过初始值来推算类型）；
- `auto`一句话定义多个变量时，所有变量类型必须一样：
```
auto a = 1, *b = &a;     // 正确，a为int, b为int *
auto sz = 0, pi = 3.14;  // 错误，sz和pi类型不同
```

- 复合类型、常量和`auto`：
    - 对于引用，`auto`推导为被引用对象的类型（使用引用实际上是使用被引用的对象，特别是引用被用作初始之时，参与初始化的是被引用对象的值）
    ```
    int a = 0, &r = i;
    auto b = r;                  // b为int，而不是int &
    ```
    - 对于`const`：`auto`会忽略顶层`const`：
    ```
    int i = 1;
    const int ci = i, &cr = ci;
    auto b = ci;                 // b为int（ci为顶层const）
    auto c = cr;                 // c为int（cr为ci的别名, ci本身是顶层const）
    auto d = &i;                 // d为int *（&i为const int *，）
    auto e = &ci;                // e为const int *（对常量对象取地址是底层const）
    ```
    - 如果希望`auto`推断出引用或者顶层常量，则声明`auto`时必须加上相应的描述符：
    ```
    const auto f = ci;           // f为const int
    auto & g = ci;               // g为const int &
    auto & h = 42;               // 错误：不能为非常量引用绑定字面值
    const auto & j = 42;         // 正确：可以为常量引用绑定字面值
    ```
    - `auto`一句话定义多个变量时，所有变量类型必须一样。注意`*`和`*`是从属于声明符的，而不是基本数据类型的一部分：
    ```
    auto k = ci, &l = ci;        // k为int，l为int &
    auto & m = ci, *p2 = &ci;    // m为const int &，p2为const int *
    auto & n = i, *p2 = &ci;     // 错误：i的类型为int，而&ci的类型为const int
    ```

#### `decltype`类型指示符

- `decltype(expr)`在不对`expr`进行求值的情况下分析并返回`expr`的数据类型：
```
decltype(f()) a = b;         // a的类型就是函数f的返回值类型。同时，这句话并不会调用f()
```
- `decltype(expr)`会原样保留引用以及顶层`const`：
    - 引用从来都是作为被引用对象的同义词出现（比如`auto`就不能自动推断出引用），唯一例外是`decltype`
    - 这很符合`decltype`一词在自然语言中的语义，必须原样转发人家本来是什么
```
const int ci = 0, &cj = ci;
decltype(ci) x = 0;          // x为const int
decltype(cj) y = x;          // y为const int &
decltype(cj) z;              // 错误：z为const int &，必须被初始化
```
- `decltype((...))`（双层括号）的结果永远是引用，而`decltype(...)`（单层括号）当且仅当`...`是引用类型时才是引用。

### 🌱 `左值`和`右值`


- 赋值运算符`a = b`中，`a`需是（非常量）左值，返回结果也是**左**值；
- 取地址符`&a`中，`a`需是左值，返回指向`a`的右值指针；
- 解引用运算符`*a`和下标运算符`a[i]`的返回结果均为**左**值；
- 自增自减运算符`a++`等中，`a`需是左值；前置版本`++a`返回结果亦为**左**值；
- 箭头运算符`p->ele`返回**左**值；点运算符`a.ele`返回值左右类型**和`a`相同**；


### 🌱 `sizeof`运算符

`sizeof`运算符返回一条表达式或者一个类型名字所占的字节数，返回类型为`size_t`类型的**常量表达式**。    
`sizeof`运算符满足右结合律。    
`sizeof`并**不实际计算其运算对象的值**。
有两种形式：

```
sizeof(Type)  // 返回类型大小
sizeof expr   // 返回表达式结果类型大小
```

`sizeof`运算符的结果部分地依赖于其作用的类型：

- 对`char`，或者`char`类型的表达式，执行结果为`1`；
- 对`引用`，执行结果为**被引用对象所占空间**的大小；
- 对`解引用指针`，执行结果为**指针指向对象所占空间**大小，指针**不需**有效
    - 无效指针是安全的，因为`sizeof`**不计算其运算对象的值**；
- 对`数组头`，执行结果为**整个数组所占空间**的大小，等价于对数组中所有元素各自执行一次`sizeof`后再求和。`sizeof`**不会**把数组头转换为指针处理；
- 对`std::string`、`std::vector`对象，执行结果为该类型**固定部分**大小，**不会**计算对象中的元素具体占用多大空间。

### 🌱 强制类型转换

#### 命名的强制类型转换

如果`T`是引用类型，则转换结果为**左**值

- `static_cast<T>(expr)`：
    - 用于任何具有明确定义的不包含底层`const`的强制类型转换。结果的值和被转换对象的值可能不同。例如：
        - `double`强转`int`（有精度损失）；
        - `void *`强转`Type *`（这一条其实也可以用`reinterpret_cast`，因为`void *`强转`Type *`的语义就是强行按照`Type *`解释那块内存）。
- `dynamic_cast<T>(expr)`：
    - 支持运行时的类型识别 => 19.2
- `const_cast<T>(expr)`：
    - 用于且只有它能用于改变运算对象的**底层**`const`（cast away the `const`）。
        - 即：只能用于指针或引用
        ```
        int b = 2;
        const int c0 = const_cast<const int>(b);                 // 错误：const int类型不是指针或引用
        const int & c1 = const_cast<const int &>(b);             // 正确
        const int & c2 = static_cast<const int &>(b);            // 正确
        const int & c3 = b;                                      // 正确
        ```
    - 只能用于更改`const`属性，不能更改类型。
    - 如果`expr`指向的对象**本身不是常量**，则通过`const_cast`获取写权限是合法行为；但如果对象本身是常量，则结果未定义。
    ```
    const char * pc;
    char * p = const_cast<char *>(pc);                           // 正确，但通过p写值是未定义的行为
    char * q = static_cast<char *>(cp);                          // 错误，static_cast不能用于去除const
    static_cast<std::string>(pc);                                // 正确，字符串字面值转换为std::string
    const_cast<std::string>(pc);                                 // 错误，const_cast只能用于去除const
    ```
- `reinterpret_cast<T>(expr)`：
    - 强制编译器按照`T`类型重新解读一块内存。
    
    ```
    int * a = new int(1);
    char * pc = reinterpret_cast<char *>(a);                     // 正确
    std::string s(pc);                                           // 可能会RE，（取决于从a开始多久出现0？）
    ```
    - 需要使用`reinterpret_cast`的场景（不能用`static_cast`的场景，暂时没发现第3种妙用）：
        - 将指针强转成指针：
            - （比如解析二进制数据流）：
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
            - （或吃饱了撑的去探究数据在内存中的二进制存储）：
            ```
            float pi = 3.14159;
            int * p1 = reinterpret_cast<int *>(&pi);     
            printf("0x%x\n", *p1);                               // 0x40490fd0
            
            uint32_t r = 0x40490fd0;   
            float * p2 = reinterpret_cast<float *>(&r);
            printf("%f\n", *p2);                                 // 3.141590
            ```
        - 将指针强转成数字（获取具体的地址）：
        ```
        int a = 1, 
        int * p = &a;
        size_t b = (size_t) p;                                   // 正确：人见人爱的C风格强转
        size_t b2 = static_cast<size_t>(p);                      // 错误：int *转换为size_t是没有明确定义的
        size_t b3 = reinterpret_cast<size_t>(p);                 // 正确
        ```
        
#### 旧式的强制类型转换

- 以下两种语法等价，因为具体行为难以断言且可能隐式进行`reinterpret_cast`，都应避免使用：
```
T t = T(expr);   // 函数式
T t = (T) expr;  // C风格
```

- 根据具体位置不同，旧式的强制类型转换的效果与`static_cast`、`const_cast`或`reinterpret_cast`相同。具体来讲，定义为以下各项中第一个成功的：
    - `const_cast`
    - `static_cast` (though ignoring access restrictions)
    - `static_cast` (though ignoring access restrictions), then `const_cast`
    - `reinterpret_cast`
    - `reinterpret_cast`, then `const_cast`
```
int * ip;
char * cp = (char *) ip;  // 相当于reinterpret_cast<char *>(ip);
```

### 🌱 `switch`

- `switch`语句中定义的变量的作用域是**整个`switch`语句**，而不仅是某个单独的`case`！

如果某处一个**带有初值**的变量位于作用域之外，在另一处该变量位于作用域之内，
则从前一处跳转至后一处的行为是非法的。

```
int num = 2;

switch (num)
{
case 0:
    // 因为程序的执行流程可能绕开下面的初始化语句，所以此switch语句不合法
    std::string filename;  // 错误：控制流绕过一个隐式初始化的变量
    int i = 0;             // 错误：控制流绕过一个显式初始化的变量
    int j;                 // 正确：j没有初始化
    j = 1;                 // 正确：可以给j赋值，这样就不是初始化了
    break;
    
case 1:
    // 正确：虽然j在作用域之内，但它没有被初始化
    j = nextNum();         // 正确：给j赋值
    
    if (filename.empty())  // filename在作用域内，但没有被初始化
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

### 🌱 异常处理

- `C++`标准异常
    - `exception`：最通用的异常类`exception`，只报告异常的发生，不提供任何额外信息。
    - `stdexcept`：几种常用的异常类：
        > - `excpetion`：最常见的问题
        > - `runtime_error`：所有RE
        >     - `range_error`：RE，生成的结果超出了有意义的值域范围
        >     - `overflow_error`：RE，计算溢出
        >     - `underflow_error`：RE，计算溢出
        > - `logic_error`：所有逻辑错误
        >     - `domain_error`：逻辑错误，参数对应的结果值不存在
        >     - `invalid_argument`：逻辑错误，无效参数
        >     - `length_error`：逻辑错误，试图创建一个超出该类型最大长度的对象
        >     - `out_of_range`：逻辑错误，使用了一个超出有效范围的值
    - `new`：`bad_alloc`异常类。12.1.2
    - `type_info`：`bad_cast`异常类。19.2
    
- `excpetion`，`bad_alloc`和`bad_cast`只能默认初始化，不能传参；其余异常必须传参（`C`风格字符串）。
- 异常类型之定义了一个名为`what`的成员函数，返回`C`风格字符串`const char *`，提供异常的文本信息。
  如果此异常传入了初始参数，则返回之；否则返回值由编译器决定。

### 🌱 自动对象 & 局部静态对象

- *自动对象* （automatic object）：
  普通局部变量对应的对象。函数控制路径经过变量定义语句时创建该对象，当到达定义所在块末尾时销毁之，只存在于块执行期间。
  函数的形参是一种自动对象。
- *局部静态对象* （local static object）：
  在程序的执行路径第一次经过对象定义语句时初始化，并且直到整个程序终止时才被销毁。
  在此期间，对象所在函数执行完毕也不会对它有影响。
  局部静态对象如果没有显式初始化，则会执行隐式初始化。内置类型的局部静态对象隐式初始化为`0`。
```
size_t countCalls()
{
    static size_t ctr = 0;  // 调用结束后这个值依然有效，且初始化会且只会在第一次调用时执行一次。
                            // 内置类型的局部静态对象隐式初始化为0，即：这里的显式初始化为0其实是不必要的。
    return ++ctr;
}
```

### 🌱 函数形参

#### 数组形参

- 数组的两个特殊性质：
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

#### 函数重载

- 顶层`const`不影响传入的对象，因此以下定义不合法：
```
Record lookup(Phone);
Record lookup(const Phone);      // redeclares Record lookup(Phone)

Record lookup(Phone *);
Record lookup(Phone * const);    // redeclares Record lookup(Phone *)
```
- 可以基于底层`const`重载函数：
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

- 省略符形参：仅用于`C`和`C++`通用的类型，只能作为函数的最后一个参数。
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


### 🌱 函数返回值

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
- 返回数组指针的若干骚操作：
    - 正常写：`int (*fun(int i))[10];`
    - 使用尾置返回值：`auto fun(int i) -> int (*)[10];`
    - 使用`decltype`：在已知要返回的是谁的情况下：
    ```
    int odd[] = {1, 3, 5, 7, 9};
    int even[] = {0, 2, 4, 6, 8};
    
    decltype(odd) *arrPtr(int i)        // decltype不负责把数组类型转化成指针。因为不能返回数组，所以要加一个*，返回数组指针
    {
        return (i % 2) ? &odd : &even;  
    }
    ```

### 🌱 调试帮助

#### `assert`

- 定义`#define NDEBUG`可以关闭`assert`宏检查
- 以下宏可用于细化调试信息：
````
__func__
__FILE__
__LINE__
__TIME__
__DATE__
````

### 🌱 函数指针

#### 函数指针

- 声明：用指针替代函数名即可。
    - `bool lengthCompare(const std::string &, const stf::string)`的类型是`bool(const std::string &, const stf::string)`
    - 声明指向`bool(const std::string &, const stf::string)`类型函数的指针：
        - `bool (*pf)(const std::string &, const stf::string);      // 未初始化`
- 使用：

```
pf = lengthCompare;                                                 // pf now points to lengthCompare()
pf = &lengthCompare;                                                // equivalent assignment: & is optional

bool b1 = pf("hello", "goodbye");                                   // calls lengthCompare
bool b2 = (*pf)("hello", "goodbye");                                // equivalent call
bool b3 = lengthCompare("hello", "goodbye");                        // equivalent call
```
- 重载函数的指针：函数指针的类型必须与重载函数中的某一个精确匹配
```
void ff(int*);
void ff(unsigned int);
void (*pf1)(unsigned int) = 0;                                      // pf1 points to nothing
void (*pf2)(unsigned int) = ff;                                     // pf1 points to ff(unsigned int)

void (*pf3)(int) = ff;                                              // error: no ff with a matching parameter list
double (*pf4)(int*) = ff;                                           // error: return type of ff and pf4 don't match
```
- 使用类型别名（`typedef`或`using`）可以简化书写：
```
typedef bool Func(const std::string &, const std::string &);        // function type
typedef decltype(lengthCompare) Func2;                              // equivalent type

typedef bool (*FuncP)(const std::string &, const std::string &);    // function pointer type
typedef decltype(lengthCompare) * FuncP2;                           // equivalent type
using FuncP3 = bool (*)(const std::string &, const std::string &);  // equivalent type
using FuncP4 = decltype(lengthCompare) *;                           // equivalent type
```

#### 与数组指针的辨析

- 注意
    - 函数名会转化成函数指针
    - 数组头会转化成指向数组元素类型的指针，而不是指向数组类型的指针
    ```
    bool le(int, int);
    bool (*pf1)(int, int) = le;               // 正确
    bool (*pf2)(int, int) = &le;              // 正确
    
    int arr[10];
    int * p1 = arr;                           // 正确
    int * p2 = &arr;                          // 错误
    int *(p3)[10] = &arr;                     // 正确
    ```
- 数组指针的类型别名：
```
int arr[10];

typedef int (*int_arr_10_ptr_t1)[10];         // 指向长度为10的int数组的指针类型的别名
typedef decltype(arr) * int_arr_10_ptr_t2;    // 等价类型别名

using int_arr_10_ptr_t3 = int[10];            // 等价类型别名
using int_arr_10_ptr_t4 = decltype(arr) *;    // 等价类型别名
```

#### 函数指针形参

- 声明：
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
- 使用：传入函数名、函数名手动取地址或者已有的指针均可：
```          
// automatically converts the function lengthCompare to a pointer to function
useBigger(s1, s2, lengthCompare);
useBigger(s1, s2, &lengthCompare);
useBigger(s1, s2, pf);
```

### 🌱 类

#### 合成的默认构造函数（synthesized default constructor）

- 按如下规则初始化类成员：
    - 存在类内初始值，则以其初始化对应成员；
        - 类内初始值可接受的语法：
        ```
        int a1 = 0;    // 正确
        int a2 = {0};  // 正确
        int a3{0};     // 正确
        int a4(0);     // 错误！
        ```
    - 默认初始化该成员。

- 生成条件：
    - 只有类没有声明任何构造函数时，编译器才会自动生成默认构造函数；
    - 如果类中包含其他类类型成员，且它没有默认构造函数，则这个类也不能生成默认构造函数；
    - 13.1.6

- 如果类内包含内置类型或复合类型的变量，则只有当这些成员全部被赋予了类内初始值时，这个类才适合于使用默认构造函数。
    - 注意：类成员变量从属于内部作用域，默认初始化是未定义的，不能指望！
    
- 作用：当对象被 *默认初始化* 或者 *值初始化* 时，执行默认构造函数。
    - *默认初始化* 在以下情况下发生：
        - 当我们在块作用域内，不使用任何初始值，定义一个非静态变量或者数组时；
        - 当一个类本身含有类类型成员，且使用合成的默认构造函数时；
        - 当类类型成员没有在构造函数初始化列表中显式地初始化时。
    - *值初始化* 在以下情况下发生：
        - 在数组初始化过程中，如果我们提供的初始值数量少于数组的大小时；
        - 当我们不使用初始值，定义一个局部静态变量时；
        - 当我们通过书写形如`T()`的表达式显式地请求值初始化时，其中`T`是类型名
          （`std::vector`的一个构造函数只接受一个实参用于说明其大小，
            它就是使用一个这种形式的实参来对它的元素初始化器进行值初始化）。
    - 类必须包含默认构造函数以便在上述情况下使用。实际应用中，如果提供了其它构造函数，最好也提供一个默认构造函数。
        - `= default;`
            - 用于既定义了自己的构造函数，又需要默认构造函数的情况；
            - 作为声明写在类内部，则构造函数默认`inline`；或作为定义写在类外部，则构造函数不`inline`。

#### 构造函数初始值列表

- 某个数据成员被初始值列表忽略时，则**先被默认初始化**，之后再按照构造函数体中的规则进行**二次赋值**；
- 初始化列表接受的初始化语法：`x(?)`或`x{?}`；
- 如果成员是`const`、引用或者没有默认构造函数的类类型，如没有类内初始值，则**必须**在初始化列表中初始化，而不能在函数体中赋值；
- 初始化的顺序是按照类成员被声明的顺序，与其在列表中的顺序无关；
    - 最好令构造函数初始化列表的顺序与成员声明的顺序保持一致；
    - 尽量避免用某些成员初始化其他成员，最好用构造函数的参数作为初始值。
- 如果一个构造函数为每一个参数都提供了默认实参，则它实际上也定义了默认构造函数。

#### 委托构造函数（delegating constructor）

- 一个委托构造函数使用它所属类的其它构造函数执行它自己的初始化过程，
  或者说它把自己的一些（或全部）职责委托给了其他构造函数。
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

- 如果构造函数只接受一个实参，那么它实际上定义了转换为此类类型的隐式转换机制，有时我们将这种构造函数称作转换构造函数；=> 14.9
- 能通过一个实参调用的构造函数定义了一条从构造函数的参数类型向类类型隐式转换的规则；
- 编译器只允许一步隐式类型转换，且转换结果是**临时右值**对象：
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

- 我们可以通过将构造函数声明为`explicit`来抑制构造函数定义的隐式转换：
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
- `explicit`**只能在类内声明**，只对一个实参的构造函数有意义。
- `explicit`构造函数**只能用于直接初始化**：
    - 执行拷贝形式的初始化（使用`=`）时，实际发生了隐式类型转换。此时只能直接初始化，而不能使用`explicit`构造函数：
    ```
    SalesData item1(nullBook);                           // 正确
    SalesData item2 = nullBook;                          // 错误
    ```
    - 为转换显式地使用构造函数：
    ```
    SalesData item2 = SalesData(nullBook);               // 正确：显式构造的对象
    SalesData item3 = static_cast<SalesData>(std::cin);  // 正确：static_cast可以使用explicit构造函数
    ```
- 标准库中含有显式构造函数的类：
    - 接受一个单参数`const char *`的`std::string`构造函数 *不是* `explicit`的；
    - 接受容量参数的`std::vector`构造函数**是**`explicit`的；

#### Further Topics on Constructors

- 13
- 15.7
- 18.1.3

#### 友元

- 友元不是类的成员，不受`public`、`private`以及`protected`这些访问限制的约束；
- 友元**不具有**传递性。每个类**单独**负责控制自己的友元类或友元函数；
    - `B`有友元`A`，`C`有友元`B`，则`A`能访问`B`的私有成员，但不能访问`C`的私有成员。
- 在类定义开始或结束的地方**集中声明**友元。
- *友元函数*
    - 友元函数的声明仅仅是指定访问权限，并不是真正的函数声明。想要使用友元，仍**另需一单独的函数声明**；
    - 对于重载函数，必须对特定的函数（特有的参数列表）单独声明。
- *友元类*
    - 令一个类成为友元
- *友元成员函数*
    - 令一个类的某个成员函数成为友元
- *友元声明和作用域*
    - 关于这段代码最重要的是：理解友元声明的作用是**影响访问权限**，它本身**并非**普通意义上的函数声明。
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

- 类中的`typedef`和`using`必须先定义后使用；
- 一般放在类定义刚开始的地方的`public`区域。

#### 可变数据成员（mutable data member）

- 可用于更改`const`对象的成员：
    - `const`对象只能调用`const`成员函数；
    - 可变数据成员永远不会是`const`，即使它是`const`对象的成员；
    - 任何成员函数，包括`const`成员函数，都可以改变可变数据成员。
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
    - 在定义之前，`Item`是<u>不完全类型</u>；
    - 不完全类型使用受限：
        - 可以定义指向这种类型的指针或引用；
        - 可以声明（**不能**定义）以这种类型为参数，或返回值类型的函数；
        - **不能**创建这种类型的对象；
        - **不能**用引用或指针访问其成员；
        - *非静态数据成员* **不能**被声明为这种类型；
            - *静态数据成员* 可以！
 
- 特别地：类可以包含指向自身类型的引用或指针。


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

- 聚合类使得用户可以直接访问其成员，并且具有特殊的初始化语法形式。
- 当一个类满足如下条件（就相当于纯`C`风格的`struct`）时，我们说它是聚合的：
    - 所有成员都是`public`的；
    - **没有**定义任何构造函数；
    - **没有**类内初始值；
    - **没有**基类，也没有`virtual`函数。
- 我们可以提供一个花括号括起来的成员初始值列表，并用它初始化聚合类的成员：
```
Entry e = {0, "Anna"};
```
- 与初始化数组元素的规则一样，如果初始值列表中的元素个数少于类的成员数量，则靠后的成员被 *值初始化* ；
- 初始化列表的元素个数不能超过类的成员数量；
- 显示初始化类的对象成员存在三个明显的缺点：
    - 要求类的所有成员都是`public`的；
    - 将正确初始化每个对象的每个成员的责任交给了用户，容易出错；
    - 添加或删除一个成员之后，所有的初始化语句都需要更新。

#### 字面值常量类

- 常量表达式（const expression）
    - 字面值：
        - 算数类型；
        - 引用和指针；
        - 字面值常量类；
        - 19.3
    - 常量表达式：值**不会改变**、并且在**编译过程中就能得到**计算结果的表达式；
        - 字面值和用常量表达式初始化的`const`对象也是常量表达式。
        
- `constexpr`变量
    - 允许将变量声明为`constexpr`类型，以便由编译器来验证变量的值是否是一个常量表达式；
    - 声明为`constexpr`的变量一定是一个常量，而且必须用常量表达式来初始化：
    ```
    constexpr int mf = 20;         // 正确
    constexpr int limit = mf + 1;  // 正确
    constexpr int sz = size();     // 当且仅当size是constexpr函数时，才正确
    ```              
    - 尽管不能使用普通函数作为`constexpr`变量的初始值，但可以使用`constexpr`函数；
    - `constexpr`引用和指针：
        - 只能绑定到固定地址的变量上：
            - 例如全局对象，局部静态对象等；
            - **不能**指向局部非静态对象。
        - `constexpr`指针和变量的初始值必须是`nullptr`、`0`或者存储于某个固定地址的对象；
        - `constexpr`指针为**顶层**`const`。
        
- `constexpr`函数
    - `constexpr`函数是指能用于常量表达式的函数；
    - 定义`constexpr`函数需要遵守：
        - 函数的返回类型和所有形参的类型都是字面值类型；
        - 函数体中只包含运行时不执行任何操作的语句，例如：
            - 空语句；
            - 类型别名；
            - `using`声明。
        - 函数体中如有可执行语句，只能是**一条**`return`语句。
    ```
    constexpr int new_sz()  { return 42; }
    constexpr int foo = new_sz();           // 正确：foo是常量表达式

    // 如果arg是常量表达式，那么scale(arg)也是常量表达式
    constexpr size_t scale(size_t cnt)  { return new_sz() * cnt; }

    int arr[scale(2)];                      // 正确
    int i = 2;                              // i不是常量表达式
    int a2[scale(i)];                       // 错误：i不是常量表达式，scale(i)也不是
    ```
    - 执行初始化时，编译器把对`constexpr`函数的调用替换成其结果值；
    - `constexpr`函数是隐式的`inline`函数； 
    - `constexpr`函数、`inline`函数以及模板的**定义和实现都应**写进头文件。
    
- 字面值常量类
    - `constexpr`函数的参数和返回值都必须是字面值类型。
      除了算数类型、引用和指针以外，**字面值常量类**也是字面值类型。
      和其他类不同，字面值常量类可能含有`constexpr`函数成员。
      这样的成员必须符合`constexpr`函数的所有要求，是隐式`const`的。

    - 以下类是字面值常量类：
        - 数据成员都是字面值类型的聚合类；
        - 满足以下要求的非聚合类：
            - 数据成员都必须是字面值类型；
            - 类必须至少有一个`constexpr`构造函数；
            - 如果一个数据成员含有类内初始值，则内置类型成员的初始值必须是常量表达式；
              或者如果成员属于某种类类型，则初始值必须使用成员自己的`constexpr`构造函数；
            - 类必须使用析构函数的默认定义，该成员负责销毁类的对象。
            
    - `constexpr`构造函数
        - 字面值常量类的构造函数可以是`constexpr`，且必须有至少一个`constexpr`构造函数；
        - `constexpr`构造函数可以声明成`= default;`的或者`= delete;`的；
        - `constexpr`构造函数的函数体是**空的**；
            - 既要满足构造函数的要求（不能有返回语句）
            - 又要满足`constexpr`函数的要求（函数体中如有可执行语句，只能是**一条**`return`语句）
        - `constexpr`构造函数必须初始化**所有**数据成员，初始值或者使用`constexpr`构造函数，或者是一条常量表达式；
        - `constexpr`构造函数用于生成`constexpr`对象以及`constexpr`函数的参数或返回类型：
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

- 声明：
    - 通过在成员声明之前加上`static`使得其与类关联在一起；
    - 静态成员可以是`public`、`private`或者`protected`的；
    - 静态数据成员的类型可以是常量、引用、指针、类类型等等。
- 定义：
    - 和其他成员函数一样，既可以在类内部，也可以在类外部定义静态成员函数；
    - `static`关键字只能出现在类内部的声明语句，在类外部定义静态成员时，**不能**重复；
    - 静态成员只能在**类外**定义并初始化，且只能被定义**一次**；除`constexpr`静态成员外，**不能**在类内初始化，不由构造函数初始化：
        - 要想确保只定义一次，应把静态数据成员的定义与其他非内联函数的定义放在**同一个文件**中；
        - 从类名开始，就已经是类的作用域之内了，所以：
            - `initRate()`**不用**再加类名；
            - 可以访问 *私有成员* ；
        ```
        double Account::interestRate = initRate();
        ```
        - 即使一个`constexpr`静态成员在类内部被初始化了，也应该在类外定义一下该成员（此时**不能**再指定初始值）：
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
    - 类似于 *全局变量* ，静态数据成员定义在任何函数之外；一旦被定义，就会 *一直存在* ；
    
- 效果：
    - 类的静态成员存在于对象之外，对象中不包含任何与静态数据成员相关的数据；
    - 类的静态成员被其所有实例共享；
    - 静态成员函数也不与对象绑定，**不能**使用`this`指针，**不能**声明成`const`的；
- 使用：
    - 使用作用域运算度直接访问静态成员：
    ```
    double r = Account::rate();
    ```
    - 可以使用类的对象、引用或者指针来访问静态成员：
    ```
    Account ac1;
    Account * ac2 = &ac1;
    // equivalent ways to call the static member rate function
    r = ac1.rate();   // through an Account object or reference
    r = ac2->rate();  // through a pointer to an Account object
    ```
    - *成员函数* 不通过作用域运算符就能直接使用静态成员：
    ```
    class Account 
    {
    public:
        void calculate() { amount += amount * interestRate; }
        
    private:
        static double interestRate;
    };
    ```
- 静态成员能用于某些场景，而普通成员不能：
    - 静态数据成员可以是 *不完全类型* ：
        - 特别地，静态数据成员可以是 *其所属的类类型* ；
        - 非静态成员只能是 *其所属的类类型* 的 *指针* 或 *引用* 。
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
    - 可以使用静态成员作为 *默认实参* ：
        - 非静态成员不能作为默认实参，因为它的值本身属于对象的一部分，
          结果就是无法真正提供一个对象以便从中获取成员的值，从而引发错误。
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

### 🌱 顺序容器

#### 类型

- 顺序容器类型
    - `std::vector`：可变大小数组。支持快速随机访问。在尾部之外的位置插入删除元素可能很慢；
    - `std::deque`：双端队列。支持快速随机访问。在头尾插入删除元素很快；
    - `std::list`：双向链表。只支持双向**顺序**访问。在任何位置插入删除元素都很快；
    - `std::foward_list`：单向链表。只支持双向**顺序**访问。在任何位置插入删除元素都很快；
    - `std::array`：固定大小数组。支持快速随机访问。**不能**添加删除元素。**支持拷贝赋值**（内置数组不行）；
    ```
    std::array<int, 10> ia1; // ten default-initialized ints
    std::array<int, 10> ia2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};     // list initialization
    std::array<int, 10> ia3 = {42};                               // ia3[0] is 42, remaining elements are 0

    int digs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int cpy[10] = digs;                                           // error: no copy or assignment for built-in arrays
    std::array<int, 10> digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<int, 10> copy = digits;                            // ok: so long as array types match
    ```
    - `std::string`：与`std::vector`相似，专门用于保存字符。随机访问快。在尾部插入删除速度快。
- 除`std::array`外，其他容器均提供高效灵活的内存管理；
- 除`std::foward_list`没有`size()`操作（为了达到与手写的单向链表一样的效率）外，其余容器均为常数复杂度；
- 顺序容器构造函数的一个版本接受容器大小参数，它使用了元素类型的**默认**构造函数。
  对于没有默认构造函数的类型的容器，构造时还需传递元素初始化器：
```
std::vector<noDefault> v1(10, init);  // 正确：提供了元素初始化器
std::vector<noDefault> v2(10);        // 错误：必须提供一个元素初始化器
```

#### 容器操作

- 类型别名（均为容器类 *静态* 成员）：
    - `iterator`：此类型容器的迭代器类型；
        - 对于容器常量，只能获取常量迭代器。
    - `const_iterator`：可以读取元素，但不能修改元素的迭代器类型；
    - `reverse_iterator`：按逆序寻址元素的迭代器，**不支持**`std::foward_list`；
    - `const_reverse_iterator`：不能修改的逆序迭代器，**不支持**`std::foward_list`；
    - `size_type`：无符号整数类型（`size_t`，aka `unsigned long`），足够保存此种容器类型最大可能容器的大小；
    - `difference_type`：带符号整数类型（`ptrdiff_t`，aka `long`），足够保存两个该容器类型的迭代器之间的距离；
    - `value_type`：元素类型；
    - `reference`：元素的左值引用类型，等价于`value_type &`；
    - `const_reference`：元素的常引用类型，等价于`const value_type &`。
- 构造函数：
    - `C c;`：默认构造函数，构造空容器；
    - `C c1(c2);`：拷贝构造，将`c2`中所有元素拷贝到`c1`；
    - `C c(b, e);`：构造`c`，将迭代器`b`和`e`指定的范围内的元素拷贝到`c`。**不支持**`std::array`；
        - 将容器初始化为另一容器的拷贝时，两个容器的容器类型和元素类型都必须相同；
        - 如果用迭代器指定范围，则仅元素类型相同即可。

    - `C c{a, b, c...};`或`C c = {a, b, c...};`：列表初始化；
    - 大小相关构造函数
        - 只有顺序容器才接受大小参数； *关联容器* **不支持**。
- 赋值与`swap`：
    - `c1 = c2;`：将`c1`中的元素全部替换为`c2`中的元素；
    - `c1 = {a, b, c...};`：将`c1`中的元素替换为列表中的元素。**不支持**`std::array`；
    - `a.swap(b);`：交换`a`与`b`中的元素；
    - `std::swap(a, b);`：交换`a`与`b`中的元素。
- 大小：
    - `c.size();`：`c`中元素的数目。常数复杂度。**不支持**`std::foward_list`；
    - `c.max_size();`：`c`可保存的最大元素数目；
    - `c.empty();`：若`c`中存储了元素，返回`false`，否则返回`true`。
- 添加删除元素（**不支持**`std::array`）：
    - `c.insert(args);`：将`args`中的元素拷贝进`c`；
    - `c.emplace(init);`：使用`inits`构造`c`中的一个元素；
    - `c.erase(args);`：删除`args`指定的元素；
    - `c.clear();`：删除`c`中所有元素，返回`void`；
- 关系运算符：
    - `==`，`!=`：所有容器都支持相等和不等运算符；
    - `<`，`<=`，`>`，`>=`：关系运算符。**不支持**无序关联容器。
- 获取迭代器：
    - `c.begin()`，`c.end()`：返回指向`c`的首元素和尾元素之后位置的迭代器（“尾后迭代器”，off-the-end iterator）；
    - `c.cbegin()`，`c.cend()`：返回`const_iterator`；
    - `c.rbegin()`，`c.rend()`：返回指向`c`的尾元素和首元素之前位置的迭代器。**不支持**`std::foward_list`；
    - `c.crbegin()`，`c.crend()`：返回`const_reverse_iterator`。**不支持**`std::foward_list`；
    - `std::begin()`，`std::end()`：不但能用于容器， *还能用于内置数组* ：
    ```
    std::vector<int> vec{0, 1, 2, 3};
    std::vector<int>::iterator iter_beg = std::begin(vec);
    std::vector<int>::iterator iter_end = std::end(vec);
    
    int arr[] = {0, 1, 2, 3};
    int * ptr_beg = std::begin(arr);
    int * ptr_end = std::end(arr);
    ```

#### 迭代器（iterator）

- 所有标准库容器都支持迭代器，但只有少数几种才同时支持下标运算符；
- 如果容器为空，则`begin`和`end`返回的是**同一个**迭代器，都是尾后迭代器；
- `for each`循环内以及使用迭代器时**不能**改变被遍历的容器的大小；
- 迭代器运算符：
    - `*iter`：返回迭代器`iter`所知元素的**左值**引用；
        - 解引用 *非法* 迭代器或者 *尾后* 迭代器是**未定义行为**。
    - `iter->mem`：解引用`iter`并获取该元素名为`mem`的成员，等价于`(*iter).mem`
    - `++iter`：令`iter`指向容器中的下一个元素；
        - 尾后迭代器并不实际指向元素，因此**不能**递增或递减。
    - `--iter`：令`iter`指向容器中的上一个元素；
    - `iter1 == iter2`，`iter1 != iter2`：判断两个迭代器是否相等（不相等）。
                                          如果两个迭代器指向的是同一个元素，或者它们是同一个容器的尾后迭代器，
                                          则相等；反之，不相等。
- 迭代器运算（iterator arithmetic）：
    - `iter + n`：结果仍为迭代器，或指向容器中元素，或指向尾后；
    - `iter - n`：结果仍为迭代器，或指向容器中元素，或指向尾后；
    - `iter += n`；
    - `iter1 - iter2`：两个迭代器之间的距离（`difference_type`），
                       即：将`iter2`向前移动`iter1 - iter2`个元素，将得到`iter1`；
    - `<`，`<=`，`>`，`>=`：关系运算符。参与运算的两个迭代器必须是合法的（或指向容器中元素，或指向尾后）。
                            如果前者指向的容器位置在后者指向的容器位置之前，则前者小于后者。
                            `std::list`的迭代器**不支持**；
- 自定义 *构成范围* 的迭代器`begin`和`end`**必须满足**的要求：
    - 它们或指向同一容器中的元素，或指向同一容器的尾后；
    - `begin <= end`，即：`end`不在`begin`之前

#### 容器定义和初始化

- `C c;`：默认构造函数。如果`C`是一个`std::array`，则`c`中元素按默认方式初始化；否则`c`为空；
- `C c1(c2);`，`C c1 = c2;`：`c1`初始化为`c2`的拷贝。
                             `c1`和`c2`必须是**相同类型**
                             （即：相同的容器类型和元素类型，对于`std::array`还有相同大小）；
- `C c{a, b, c...};`，`C c = {a, b, c...};`：`c`初始化为初始化列表中元素的拷贝。
                                             列表中元素类型必须与`C`兼容。
                                             对于`std::array`，列表中元素数目不大于`array`大小，
                                             遗漏元素一律 *值初始化* ；
- `C c(b, e);`：`c`初始化为迭代器`b`和`e`指定范围中的元素的拷贝。
                范围中元素的类型必须与`C`兼容（`std::array`**不适用**）；
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
                
只有 *顺序容器* （**不包括**`std::array`）的构造函数才能接受大小参数：
- `C seq(n);`：`seq`包含`n`个元素，这些元素进行了 *值初始化* ；
               此构造函数是`explicit`的。**不适用**于`std::string`；
- `C seq(n, t);`：`seq`包含`n`个初始化为值`t`的元素。
```
std::vector<int> ivec(10, -1);      // 10 int elements, each initialized to -1
list<std::string> svec(10, "hi!");  // 10 strings; each element is "hi!"
std::forward_list<int> ivec(10);    // 10 elements, each initialized to 0
std::deque<std::string> svec(10);   // 10 elements, each an empty string
```


#### 赋值和`swap`

- 以下赋值运算符可用于所有容器：
    - `c1 = c2;`：将`c1`中的元素替换为`c2`中元素的拷贝。`c1`和`c2`必须具有相同的类型；
    - `c = {a, b, c...};`：将`c1`中元素替换为初始化列表中元素的拷贝（`std::array`**不适用**）；
    ```
    std::array<int, 10> a1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<int, 10> a2 = {0};                             // OK: elements all have value 0
    a1 = a2;                                                  // OK: replaces elements in a1
    a2 = {0};                                                 // error: cannot assign to an array 
                                                              //        from a braced list
    ```
    - `std::swap(c1, c2);`，`c1.swap(c2);`：交换`c1`和`c2`中的元素。
                                            `c1`和`c2`必须具有**相同类型**；
        - 除`std::array`**以外**，`swap`不对任何元素进行拷贝，删除或插入操作，因此可以保证 *常数复杂度* ；
        - 元素不被移动意味着除`std::array`和`std::string`**以外**，原先的迭代器、指针和引用 *不会失效* ；
        - 泛型编程要求：**应当**统一使用非成员版本的`swap`，即`std::swap(c1, c2);`。
    ```
    std::vector<std::string> svec1(10);  // vector with 10 elements
    std::vector<std::string> svec2(24);  // vector with 24 elements
    std::swap(svec1, svec2);
    ```
- `assign`操作**不适用于** *关联容器* 以及`std::array`：
    - `seq.assign(b, e);`：将`seq`中的元素替换为迭代器`b`和`e`所表示范围中的元素。
                           迭代器`b`和`e`**不能**指向`seq`中的元素；
        - 由于旧元素被 *替换* ，因此传递给`assign`的迭代器**不能**指向调用`assign`的容器。
    ```
    std::list<std::string> names;
    std::vector<const char *> oldstyle;
    names = oldstyle;                    // error: container types don't match
                                         // ok: can convert from const char * to string
    names.assign(oldstyle.cbegin(), oldstyle.cend());
    ```
    - `seq.assign({a, b, c...});`：将`seq`中的元素替换为初始化列表中的元素；
    - `seq.assign(n, t);`：将`seq`中的元素替换为`n`个值为`t`的元素；
    ```
    // equivalent to: slist1.clear(); 
    //                slist1.insert(slist1.begin(), 10, "Hiya!");
    std::list<std::string> slist1(1); // one element, which is the empty string
    slist1.assign(10, "Hiya!");       // ten elements; each one is Hiya !
    ```
- 赋值运算会导致指向左边容器内部的迭代器、引用和指针**失效**。
  而`swap`操作将容器内容交换，**不会**导致指向容器的迭代器、引用和指针失效
  （`std::array`和`std::string`**除外**）。

#### 关系运算符

- 每个容器类型都支持 *相等运算符* `==`和`!=`；
- **除** *无序关联容器* **外**，所有容器都支持 *关系运算符* `>`、`>=`、`<`和`<=`。
    - *关系运算符* 左右的容器必须为**相同类型**;
    - 比较两个容器的方式与`std::string`类似，为 *逐元素字典序* ;
    - 只有当其元素类型也定义了相应的比较运算符时，才可以使用 *关系运算符* 来比较两个容器。
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

- 插入元素：
    - 1
    - 2
- 访问元素：
    - 1
    - 2
- 删除元素：
    - 1
    - 2
- 特殊的`std::foward_list`操作：
    - 1
    - 2
- 改变容器大小：
    - 1
    - 1
- 容器操作可能使迭代器失效。

#### 额外的`std::string`操作


#### 容器适配器



