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

### \# 初始化和对`const`的引用

2.3.1节提及：引用的类型必须与其所引用的对象的类型一致，但有2个例外。（另一个：基类的指针或引用可以绑定到派生类上）    
其中一个：初始化常量引用时，允许用任意表达式作为初始值，只要该表达式的结果能够被转换成引用的类型即可。    
即：常量引用可以绑定在“存在可接受的转换规则”的对象上。
尤其，允许为一个常量引用绑定：

- 非常量的对象
- 字面值
- 一般表达式

即：**常量引用可以绑定在右值上**！

```
int i = 42;
const int & r1 = i;      // ok: we can bind a const int& to a plain int object
const int & r2 = 42;     // ok: r1 is a reference to const
const int & r3 = i * 2;  // ok: r3 is a reference to const
int & r4 = i * 2;        // error: r4 is a plain, non const reference

// 注：执行如下代码时：

double pi = 3.1415926;
const int & a = pi;

// 实际上编译器干了这么件事：

int tmp = pi;
const int & a = tmp;

// 如果不是常量引用，改的就不是`pi`而是临时量`tmp`，容易造成人祸，因此`C++`直接规定非常量引用不能绑定给临时量。
```