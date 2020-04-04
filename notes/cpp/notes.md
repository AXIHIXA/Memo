# 《C++ Primer 5th Edition》拾遗

记录一些对C++理解得不到位的地方。

## 🌱 一句话

- 常见规则

    - `const`常量不论是声明还是使用都添加`extern`修饰符
    - `constexpr`函数、`inline`函数以及模板的**定义和实现都应**写进头文件
    - `using`声明**不应**写进头文件
    - `for each`循环内以及使用迭代器时**不能**改变被遍历的容器的大小
    - 现代`C++`应使用标准库类型配合迭代器，而不是`C`风格的数组和指针。数组也是一种迭代器
    - 除非必须，不要使用自增自减运算符的后置版本（会造成性能浪费）

    + 对复杂的声明符，从右往左看比较好理解
    + 对数组声明，从数组的名字开始由内向外看比较好理解

- 一些常识

    - `*iter++`等价于`*(iter++)` => 优先级：`++` > `*`
    - `p->ele`等价于`(*p).ele` => 优先级：`.` < `*`
    - 指针只能用字面量，或者用`&`获取的地址初始化或者赋值
    - 指针`*`以及引用`&`只从属于某个声明符，而不是基本数据类型的一部分
        - hehe
    
    
## 语法点

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
- 内置变量且在函数体之内，**无**隐式初始化
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

```
int a;             // 这其实是声明并定义了变量a
extern int a;      // 这才是仅仅声明而不定义
extern int a = 1;  // 这是声明并定义了变量a并初始化为1。“任何包含显式初始化的声明即成为定义，如有extern则其作用会被抵消”
```

### 🌱 `const`常量不论是声明还是使用都添加`extern`修饰符

“如果想在多个文件之间共享`const`对象，则必须在定义的对象之前添加`extern`关键字。”

```
extern const int BUF_SIZE = 1024;  // globals.cpp
extern const int BUF_SIZE;         // globals.h
extern const int BUF_SIZE;         // sth.h （其他要用到`BUF_SIZE`的头文件）
```

### 🌱 `const`指针与引用

#### 绑定

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
const int & a = pi;      // ok

// 实际上编译器干了这么件事：

int tmp = pi;
const int & a = tmp;

// 如果不是常量引用，改的就不是`pi`而是临时量`tmp`，容易造成人祸，因此`C++`直接规定非常量引用不能绑定给临时量。
```

“指向常量的指针”和“常指针”不一样：

```
int num = 1;  
const int * p1 = &num;        // 指向`const int`的指针。不能用p1修改num的值，但可以让p1指向别的`(const) int`变量。
int * const p2 = &num;        // 指向`int`的常指针。不能让p1指向别的`int`变量，但可以用p1修改num的值。
const int * const p2 = &num;  // 指向`const int`的常指针。既不能用p1修改num的值，也不可以让p1指向别的`int`变量。
```

#### 顶层`const`和底层`const`

- 顶层`const`：任意的对象是常量
- 底层`const`：指针或引用指向的那个对象本身是常量

```
int i = 0;
int * const p1 = &i;        // we can't change the value of p1; const is top-level
const int ci = 42;          // we cannot change ci; const is top-level
const int * p2 = &ci;       // we can change p2; const is low-level
const int * const p3 = p2;  // right-most const is top-level, left-most is not
const int & r = ci;         // const in reference types is always low-level
```

### 🌱 `constexpr`

`constexpr`函数默认`inline`。`constexpr`函数必须返回字面值。`constexpr`函数可以用于初始化常量。   
`constexpr`函数、`inline`函数以及模板的**定义和实现都应**写进头文件。

### 🌱 类型别名

```
typedef int * intptr;
using intptr2 = int *;

int a = 1;
const intptr p = &a;             // "const (int *)", i.e. `int * const`. NOT `const int *`!!!
const intptr2 p2 = &a, p3 = &a;  // 注意这里p3已经是指针了，不需要再加*
```

### 🌱 `auto` & `decltype`

- 如果希望`auto`推断出引用或者顶层常量，则声明`auto`时必须加上相应的描述符，例如`const auto & a = ...`。
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

- `static_cast<T>(expr)`：一般用于有精度损失的强制类型转换。转换结果与原始地址相等。
- `dynamic_cast<T>(expr)`：支持运行时的类型识别。
- `const_cast<T>(expr)`：
常常用于有函数重载的上下文中。
用于且只有它能用于去除运算对象的底层const（cast away the const）。
只能用于更改const属性，不能更改类型。
如果`expr`指向的对象**本身不是常量**，则通过`const_cast`获取写权限是合法行为。
但如果对象本身是常量，则结果未定义。
```
const char * pc;
char * p = const_cast<char *>(pc);   // 正确，但通过p写值是未定义的行为
char * q = static_cast<char *>(cp);  // 错误，static_cast不能用于去除const
static_cast<std::string>(pc);        // 正确，字符串字面值转换为std::string
const_cast<std::string>(pc);         // 错误，const_cast只能用于去除const
```
- `reinterpret_cast<T>(expr)`：强制编译器按照`T`类型重新解读一块内存。**十分危险**。 
```
int * a = new int(1);
char * pc = reinterpret_cast<char *>(a);  // 正确
std::string s(pc);                        // 可能会RE，（取决于从a开始多久出现0？）
```

#### 旧式的强制类型转换

```
T(expr);   // 函数式
(T) expr;  // C风格
```

根据具体位置不同，旧式的强制类型转换的效果与`static_cast`、`const_cast`或`reinterpret_cast`相同。
选择优先级：`static_cast` > `const_cast` > `reinterpret_cast`。
即：只有当前一种解释不合法时，才会考虑后一种。

```
int * ip;
char * cp = (char *) ip;  // 相当于 reinterpret_cast<char *>(ip);
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