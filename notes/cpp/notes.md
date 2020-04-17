# 《C++ Primer 5th Edition》拾遗

记录一些对C++理解得不到位的地方。

### 🌱 一句话

- 常见规则

    - `const`常量不论是声明还是使用都添加`extern`修饰符
    - 想要`auto`推导出引用或者常量的话，直接写清楚是坠吼的（`const auto & a = b`），别折腾顶层`const`什么的
    - `constexpr`函数、`inline`函数以及模板的**定义和实现都应**写进头文件
    - `using`声明（`using std::string`、`using namespace std`、`using intptr = int *`等）**不应**写进头文件
    - `for each`循环内以及使用迭代器时**不能**改变被遍历的容器的大小
    - 现代`C++`应使用标准库类型配合迭代器，而不是`C`风格的数组和指针。数组也是一种迭代器
    - 除非必须，不要使用自增自减运算符的后置版本（会造成性能浪费）
    
- 一些常识

    - `*iter++`等价于`*(iter++)` => 优先级：`++` > `*`
    - `p->ele`等价于`(*p).ele` => 优先级：`.` < `*`
    - 如果一个函数是永远也不会用到的，那么它可以只有声明而没有定义 => 15.3
    
- 读代码标准操作

    - 对复杂的声明符，从右往左看比较好理解
    - 对数组声明，从数组的名字开始由内向外看比较好理解
    - 判断复杂类型`auto`变量的类型：先扒掉引用，再扒掉被引用者的顶层`const`


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

如果初始化时使用了等号，则是拷贝初始化（生成并直接初始化临时右值对象，再将临时对象拷贝到左值），有性能损失：

```
std::string s = std::string("hehe"); 
// 实际执行时等价于：
std::string tmp("hehe"); 
std::string s = tmp; 
```

如不使用等号，则是直接初始化。

### 🌱 `extern`修饰符

#### 声明和定义

```
int a;             // 这其实是声明并定义了变量a
extern int a;      // 这才是仅仅声明而不定义
extern int a = 1;  // 这是声明并定义了变量a并初始化为1。“任何包含显式初始化的声明即成为定义，如有extern则其作用会被抵消”
```

#### `const`常量不论是声明还是使用都添加`extern`修饰符

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


### 🌱 复合类型（指针、引用）和常量

- 指针`*`以及引用`&`只从属于某个声明符，而不是基本数据类型的一部分

- 指针或引用的类型必须与其所引用的对象的**类型严格一致**（除下一条的2个例外），即：
    - 指针只能用同类型的其他指针（包括立即数或对象强制转换成的指针，以及取值符获取的地址），或者`NULL`、`nullptr`赋值
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

- `constexpr`
    - `constexpr`函数默认`inline`。`constexpr`函数必须返回字面值。`constexpr`函数可以用于初始化常量。   
    - `constexpr`函数、`inline`函数以及模板的**定义和实现都应**写进头文件。

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

- `decltype((...))`（双层括号）的结果永远是引用，而`decltype(...)`（单层括号）当且仅当`...`是引用类型时才是引用。
- `auto`和`decltype`就是个坑，别用。


### 🌱 `std::begin()` & `std::end()`

```
int ia[] = {0, 1, 2, 3};
int * pbeg = std::begin(ia);
int * pend = std::end(ia);
```

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
- 对`引用`，执行结果为`被引用对象所占空间`的大小；
- 对`解引用指针`，执行结果为`指针指向对象所占空间`大小，指针**不需**有效；
- 对`数组头`，执行结果为`整个数组所占空间`的大小，等价于对数组中所有元素各自执行一次`sizeof`后再求和。`sizeof`**不会**把数组头转换为指针处理；
- 对`std::string`、`std::vector`对象，执行结果为`该类型固定部分`大小，**不会**计算对象中的元素具体占用多大空间。

### 🌱 强制类型转换

#### 命名的强制类型转换

如果`T`是引用类型，则转换结果为**左**值

- `static_cast<T>(expr)`
    - 用于任何具有明确定义的不包含底层`const`的强制类型转换。结果的值和被转换对象的值可能不同。例如：
        - `double`强转`int`（有精度损失）；
        - `void *`强转`Type *`（这一条其实也可以用`reinterpret_cast`，因为`void *`强转`Type *`的语义就是强行按照`Type *`解释那块内存）。
- `dynamic_cast<T>(expr)`：支持运行时的类型识别 => 19.2
- `const_cast<T>(expr)`：
    - 用于且只有它能用于去除运算对象的底层`const`（cast away the `const`）。
    - 只能用于更改`const`属性，不能更改类型。
    - 如果`expr`指向的对象**本身不是常量**，则通过`const_cast`获取写权限是合法行为。
    - 但如果对象本身是常量，则结果未定义。
```
const char * pc;
char * p = const_cast<char *>(pc);   // 正确，但通过p写值是未定义的行为
char * q = static_cast<char *>(cp);  // 错误，static_cast不能用于去除const
static_cast<std::string>(pc);        // 正确，字符串字面值转换为std::string
const_cast<std::string>(pc);         // 错误，const_cast只能用于去除const
```
- `reinterpret_cast<T>(expr)`：强制编译器按照`T`类型重新解读一块内存。可以用作指针强转（比如解析二进制数据流）。目前没发现其他妙用。
```
int * a = new int(1);
char * pc = reinterpret_cast<char *>(a);             // 正确
std::string s(pc);                                   // 可能会RE，（取决于从a开始多久出现0？）
```
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

- `自动对象（automatic object）`：
  普通局部变量对应的对象。函数控制路径经过变量定义语句时创建该对象，当到达定义所在块末尾时销毁之，只存在于块执行期间。
  函数的`形参`是一种`自动对象`。
- `局部静态对象（local static object）`：
  在程序的执行路径第一次经过对象定义语句时初始化，并且直到整个程序终止时才被销毁。
  在此期间，对象所在函数执行完毕也不会对它有影响。
  `局部静态对象`如果没有显式初始化，则会执行隐式初始化。内置类型的`局部静态对象`隐式初始化为`0`。
```
size_t countCalls()
{
    static size_t ctr = 0;  // 调用结束后这个值依然有效，且初始化会且只会在第一次调用时执行一次。
                            // 内置类型的局部静态对象隐式初始化为0，即：这里的显式初始化为0其实是不必要的。
    return ++ctr;
}
```

### 🌱 数组形参

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
