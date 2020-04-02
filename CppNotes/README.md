# 《C++ Primer 5th Edition》拾遗

记录一些对C++理解得不到位的地方。

### \# 初始化

- 显式初始化

```
int a = 1;
int a(1);
int a = {1};  // 列表初始化。如会损失精度，则CE
int a{1};     // 列表初始化。如会损失精度，则CE

// e.g.
int a = {3.14};  // 列表初始化。会损失精度，报CE
int a{3.14};     // 列表初始化。会损失精度，报CE
```

- 隐式初始化

```
- 内置变量且在函数体之外，隐式初始化为0
- 内置变量且在函数体之内，无隐式初始化
- 自定义对象：由类决定是否允许隐式初始化以及初始化为何值
```

### \# 指针只能用字面量，或者用`&`获取的地址初始化或者赋值

### \# “指针`*`以及引用`&`只从属于某个声明符，而不是基本数据类型的一部分。”

### \# `extern`修饰符

```
int a;             // 这其实是声明并定义了变量a
extern int a;      // 这才是仅仅声明而不定义
extern int a = 1;  // 这是声明并定义了变量a并初始化为1。“任何包含显式初始化的声明即成为定义，如有extern则其作用会被抵消”
```

### \# `const`常量不论是声明还是使用都添加`extern`修饰符

“如果想在多个文件之间共享`const`对象，则必须在定义的对象之前添加`extern`关键字。”

```
extern const int BUF_SIZE = 1024;  // globals.cpp
extern const int BUF_SIZE;         // globals.h
extern const int BUF_SIZE;         // sth.h （其他要用到`BUF_SIZE`的头文件）
```

### \# 对复杂的声明符，从右往左看比较好理解

### \# `const`指针与引用

对于常量，只能绑定常量指针或引用，不能绑定普通版。

指针或引用的类型必须与其所引用的对象的类型一致，但有2个例外：
- 常量引用可以绑定在“存在可接受的转换规则”的对象上【这一条指向常量的指针不行】；    
- 基类的指针或引用可以绑定到派生类上。

尤其，允许为一个常量引用绑定：
- 非常量的对象
- 字面值
- 一般表达式

即：**常量引用可以绑定在（其它类型的）右值上**！

```
double i = 4.2;
const int & r1 = i;      // ok: we can bind a const int& to a plain int object
const int & r2 = 4.2;    // ok: r1 is a reference to const
const int & r3 = i * 2;  // ok: r3 is a reference to const
int & r4 = i * 2;        // error: r4 is a plain, non const reference

// 注：执行如下代码时：

double pi = 3.1415926;  
const int & a = pi;     // ok

// 实际上编译器干了这么件事：

int tmp = pi;
const int & a = tmp;

// 如果不是常量引用，改的就不是`pi`而是临时量`tmp`，容易造成人祸，因此`C++`直接规定非常量引用不能绑定给临时量。
```

“指向常量的指针”和“常指针”不一样：

```
int num = 1;  
const int * p1 = &num;        // 指向`const int`的指针。不能用p1修改num的值，但可以让p1指向别的`(const) int`变量
int * const p2 = &num;        // 指向`int`的常指针。不能让p1指向别的`int`变量，但可以用p1修改num的值
const int * const p2 = &num;  // 指向`const int`的常指针。既不能用p1修改num的值，也不可以让p1指向别的`int`变量
```

### \# `constexpr`

`constexpr`函数默认`inline`。`constexpr`函数必须返回字面值。`constexpr`函数可以用于初始化常量。

### \# 类型别名

```
typedef int * intptr;
using intptr2 = int *;

int a = 1;
const intptr p = &a;             // "const (int *)", i.e. `int * const`. NOT `const int *`!!!
const intptr2 p2 = &a, p3 = &a;  // 注意这里p3已经是指针了，不需要再加*
```

### \# `auto` & `decltype`

- 如果希望`auto`推断出引用或者顶层常量，则声明`auto`时必须加上相应的描述符，例如`const auto & a = ...`。

- `decltype((...))`（双层括号）的结果永远是引用，而`decltype(...)`（单层括号）当且仅当`...`是引用类型时才是引用。





