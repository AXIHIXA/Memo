# _C++ Templates: The Complete Guide_ Notes

- _**C++ Templates**: The Complete Guide_ Second Edition
- David Vandevoorde
- Nicolai M. Josuttis
- Douglas Gregor






## 🌱 Part I The Basics


### 🎯 Chapter 1 Function Templates

### 📌 1.1 A First Look at Function Templates

#### Instantiation

Templates aren’t compiled into single entities that can handle any type. 
Instead, different entities are generated from the template for every type for which the template is used.


The process of replacing template parameters by concrete types is called _instantiation_. 
It results in an _instance_ of a template.


Note that the mere use of a function template can trigger such an instantiation process. 
There is no need for the programmer to request the instantiation separately.

#### Two-Phase Translation

Templates are “compiled” in two phases:
1. Without instantiation at _definition_ time, 
   the template code itself is checked for correctness ignoring the template parameters.
   This includes:
   – Syntax errors are discovered, such as missing semicolons.
   – Using unknown names (type names, function names, ...) 
     that don’t depend on template parameters are discovered.
   – Static assertions that don’t depend on template parameters are checked.
2. At _instantiation_ time, 
   the template code is checked (again) to ensure that all code is valid. 
   That is, now especially, all parts that depend on template parameters are double-checked.


The fact that names are checked twice is called _two-phase lookup_ 
and discussed in detail in Section 14.3.


Note that some compilers don’t perform the full checks of the first phase. 
So you might not see general problems until the template code is instantiated at least once.

#### Compiling and Linking

Two-phase translation leads to an important problem in the handling of templates in practice: 
When a function template is used in a way that triggers its instantiation, 
which is at _compile time_, 
a compiler will (at some point) need to see that template’s _definition_. 
This breaks the usual compile and link distinction for ordinary functions, 
when the declaration of a function is sufficient to compile its use. 
Methods of handling this problem are discussed in Chapter 9. 
For the moment, let’s take the simplest approach: 
Implement each template inside a header file.

### 📌 1.2 Template Argument Deduction

- During template type deduction, 
  arguments' reference-ness and top-level cv-constraints are ignored.
- When deducing types for universal reference parameters, 
  reference collapse may occur. 
- During template type deduction, 
  arguments that are array or function names decay to pointers, 
  unless they are used to initialize references.

#### Type Conversions During Type Deduction

Note that automatic type conversions are limited during type deduction: 
- When declaring call parameters by reference, 
  even trivial conversions do not apply to type deduction. 
  Two arguments declared with the same template parameter `T` must match exactly.
- When declaring call parameters by value, 
  only trivial conversions that decay are supported: 
  Qualifications with `const` or `volatile` are ignored, 
  references convert to the referenced type, 
  and raw arrays or functions convert to the corresponding pointer type. 
  For two arguments declared with the same template parameter `T`, 
  the _decayed_ types must match.


Three ways to handle type deduction failures dur to argument type mismatch:
1. Cast the arguments to so that they both match:
```c++
max(static_cast<double>(4), 7.2);
```
2. Specify (or qualify) explicitly the type of `T` 
   to prevent the compiler from attempting type deduction: 
```c++
max<double>(4, 7.2);
```
3. Specify that the parameters may have different types.

#### Type Deduction for Default Arguments

Note also that type deduction does **not** work for default call arguments. 
For example:
```c++
template <typename T>
void f(T = "");

f(1);  // OK: deduced T to be int, so that it calls f<int>(1)
f();   // ERROR: cannot deduce T
```
To support this case, 
you also have to declare a default argument for the template parameter:
```c++
template <typename T = std::string>
void f(T = "");

f();   // OK
```

### 📌 1.3 Multiple Template Parameters

Function templates have two distinct sets of parameters:
1. _Template parameters_, 
   which are declared in angle brackets before the function template name:
```c++
// T1, T2 are template parameters
template <typename T1, typename T2>
```
2. _Call parameters_, 
   which are declared in parentheses after the function template name:
```c++
// a and b are call parameters
T1 max(T1 a, T2 b)
{
    return b < a ? a : b;
}
```
It may appear desirable to be able to pass parameters of different types to the `max` template, 
but it raises a problem. 
If you use one of the parameter types as return type,
the argument for the other parameter might get converted to this type, 
regardless of the caller’s intention. 
Thus, the return type depends on the call argument order. 
The maximum of `66.66` and `42` will be the `double` `66.66`, 
while the maximum of `42` and `66.66` will be the `int` `66`. 


C++ provides different ways to deal with this problem (to de detailed in the next section):
- Introduce a third template parameter for the return type.
- Let the compiler deduce the return type.
- Declare the return type to be the “common type” of the two parameter types. 

#### Template Parameters for Return Types

_Template argument deduction_ allows us to call function templates 
with syntax identical to that of calling an ordinary function: 
We do **not** have to explicitly specify the types corresponding to the template parameters.


We also mentioned that we can still specify 
the types to use for the template parameters explicitly:
```c++
template <typename T>
T max (T a, T b);
...
::max<double>(4, 7.2);  // instantiate T as double
```
In cases when there is **no** connection between template and call parameters 
and when template parameters can **not** be determined, 
you must specify the template argument explicitly with the call. 
For example, you can introduce a third template argument type  `R`
to define the return type of a function template:
```c++
template <typename T1, typename T2, typename R>
R max(T1 a, T2 b)
{
    return b < a ? a : b;
}
```
However, template argument deduction does **not** take return types into account. 
Deduction can be seen as part of _overload resolution_, 
a process that is **not** based on selection of return types either. 
The sole exception is the return type of conversion operator members. 
In C++, the return type also cannot be deduced 
from the context in which the caller uses the call.

In this case, `R` can **not** be deduced,
and you have to specify all template arguments explicitly.
```c++
::max<int, double, double>(4, 7.2);
```

So far, we have looked at cases in which either all or none of 
the function template arguments were mentioned explicitly. 
Another approach is to specify only the first arguments explicitly 
and to allow the deduction process to derive the rest. 
In general, you must specify all the argument types up to 
the last argument type that can not be determined implicitly. 
Thus, if you change the order of the template parameters in our example, 
the caller needs to specify only the return type:
```c++
template <typename R, typename T1, typename T2>
R max(T1 a, T2 b)
{
    return b < a ? a : b;
}

// OK: return type is double, T1 and T2 are deduced
::max<double>(4, 7.2);
```
Note that these modified versions of `max` **don’t** lead to significant advantages. 
For the one-parameter version, you can already specify the parameter (and return) type 
if two arguments of a different type are passed. 
Thus, it is a good idea to keep it simple and use the one-parameter version of `max`. 


See Chapter 15 for details of the deduction process.

#### Deducing the Return Type

If a return type depends on template parameters, 
the simplest and best approach to deduce the return type 
is to let the compiler find out. 
Since C++14, it is possible to declare the return type to be `auto` or `decltype(auto)`. 
```c++
template <typename T1, typename T2>
auto max (T1 a, T2 b)
{
    return b < a ? a : b;
}
```
In fact, the use of `auto` for the return type without a corresponding _trailing return type_ 
indicates that the actual return type must be deduced from the return statements in the function body. 
Of course, deducing the return type from the function body has to be possible. 
Therefore, the code must be available and multiple return statements have to match. 


Before C++14, it is only possible to let the compiler determine the return type 
by more or less making the implementation of the function part of its declaration. 
In C++11 we can benefit from the fact that the trailing return type syntax allows us to use the call parameters. 
That is, we can declare that the return type is derived from what `operator?:` yields: 
```c++
template <typename T1, typename T2>
auto max (T1 a, T2 b) -> decltype(b < a ? a : b)
{
    return b < a ? a : b;
}
```
Here, the resulting type is determined by the rules for `operator?:`, 
which are fairly elaborate but generally produce an intuitively expected result 
(e.g., if `a` and `b` have different arithmetic types, 
a common arithmetic type is found for the result).


Note that
```c++
template <typename T1, typename T2>
auto max (T1 a, T2 b) -> decltype(b < a ? a : b)
```
is a _declaration_, so that the compiler uses the rules of `operator?:` called for parameters `a` and `b` 
to find out the return type of `max` at compile time. 
The implementation does **not** necessarily have to match. 
In fact, using `true` as the condition for `operator?:` in the declaration is enough:
```c++
template <typename T1, typename T2>
auto max (T1 a, T2 b) -> decltype(true ? a : b);
```
However, in any case this definition has a significant drawback: 
It might happen that the return type is a reference type, 
because under some conditions `T` might be a reference. 
For this reason you should return the type _decayed_ from `T`, 
which looks as follows:
```c++
#include <type_traits>

template <typename T1, typename T2>
auto max (T1 a, T2 b) -> std::decay_t<decltype(true ? a : b)>
{ 
    return b < a ? a : b;
}
```
Note that an initialization of type `auto` always decays. 
This also applies to return values when the return type is just `auto`. 
`auto` as a return type behaves just as in the following code, 
where `a` is declared by the decayed type of `i`, `int`:
```c++
int i = 42;
int const & ir = i;  // ir refers to i 
auto a = ir;         // a is declared as new object of type int
```

#### Return Type as Common Type

Since C++11, the C++ standard library provides a means to specify choosing “the more general type.” 
`std::common_type<>::type` yields the “common type” of two (or more) different types passed as template arguments. 
```c++
#include <type_traits>

// C++11
template <typename T1, typename T2>
typename std::common_type<T1, T2>::type max (T1 a, T2 b)
{
    return b < a ? a : b;
}

// C++14
template <typename T1, typename T2>
std::common_type_t<T1, T2> max (T1 a, T2 b)
{
    return b < a ? a : b;
}
```
The way `std::common_type` is implemented uses some tricky template programming, 
which is discussed in Section 26.5. 
Internally, it chooses the resulting type 
according to the language rules of operator `?:` 
or specializations for specific types. 
Thus, both `::max(4, 7.2)` and `::max(7.2, 4)` 
yield the same value `7.2` of type `double`. 
Note that `std::common_type` also decays.
See Section D.5 for details. 

### 📌 1.5 Overloading Function Templates

Like ordinary functions, function templates can be overloaded. 
That is, you can have different function definitions with the same function name 
so that when that name is used in a function call, 
a C++ compiler must decide which one of the various candidates to call. 
The rules for _overload resolution_ may become rather complicated, even without templates.
please look at Appendix C for details. 

1. Candidate function set
2. Viable function set
3. Best match
   - Prefer more-specialized versions

In this section we discuss overloading when templates are involved.
The following short program illustrates overloading a function template:
```c++
// maximum of two int values:
int max(int a, int b)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    return b < a ? a : b;
}

// maximum of two values of any type:
template <typename T>
T max(T a, T b)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    return b < a ? a : b;
}

int main(int argc, char * argv[])
{
    ::max(7, 42);          // Calls int max(int, int). 
                           // Prefer more-specialized version. 
    ::max(7.0, 42.0);      // Calls T max(T, T) [with T = double]. 
                           // Template provides best match. 
    ::max('a', 'b');       // Calls T max(T, T) [with T = char]. 
                           // Template provides best match. 
    ::max<>(7, 42);        // Calls T max(T, T) [with T = int]. 
                           // Calling template explicitly with template arguments to be deducted. 
    ::max<double>(7, 42);  // Calls T max(T, T) [with T = double]. 
                           // Calling template explicitly. 
    ::max('a', 42.7);      // Calls int max(int, int). 
                           // Automatic type conversion is not considered for deduced template parameters. 
                           // Template deduction fails. 
    
    return EXIT_SUCCESS;
}
```
An interesting example would be to overload the maximum template 
to be able to explicitly specify the return type only: 
```c++
template <typename T1, typename T2>
auto max(T1 a, T2 b)
{
    return b < a ? a : b;
} 

template <typename R, typename T1, typename T2>
R max(T1 a, T2 b)
{
    return b < a ? a : b;
}

auto a = ::max(4, 7.2);               // uses first template
auto b = ::max<long double>(7.2, 4);  // uses second template
auto c = ::max<int>(4, 7.2);          // ERROR: both function templates match
```
A useful example would be to overload the maximum template 
for pointers and ordinary C-strings:
```c++
// maximum of two values of any type:
template <typename T>
T max(T a, T b)
{
    return b < a ? a : b;
}

// maximum of two pointers:
template <typename T>
T * max(T * a, T * b)
{
    return *b < *a ? a : b;
}

// maximum of two C-strings:
char const * max (char const * a, char const * b)
{
    return std::strcmp(b, a) < 0 ? a : b;
}

int main ()
{
    int a = 7;
    int b = 42;
    auto m1 = ::max(a, b);     // max for two values of type int
    
    std::string s1 = "hey";
    std::string s2 = "you";
    auto m2 = ::max(s1,s2);    // max for two values of type std::string
    
    int * p1 = &b;
    int * p2 = &a;
    auto m3 = ::max(p1, p2);   // max for two pointers
    
    char const * x = "hello";
    char const * y = "world";
    auto m4 = ::max(x, y);     // max() for two C-strings
}
```
Note that in all overloads of `max` we pass the arguments by value. 
In general, it is a good idea **not** to change more than necessary 
when overloading function templates.
You should limit your changes to the number of parameters 
or to specifying template parameters explicitly. 
Otherwise, unexpected effects may happen. 
For example, if you implement your `max` template to pass the arguments by reference 
and overload it for two C-strings passed by value, 
you **can’t** use the three-argument version to compute the maximum of three C-strings:
```c++
// maximum of two values of any type (call-by-reference)
template <typename T> 
T const & max (T const & a, T const & b)
{
    return b < a ? a : b;
} 

// maximum of two C-strings (call-by-value)
char const * max (char const * a, char const * b)
{
    return std::strcmp(b, a) < 0 ? a : b;
}

// maximum of three values of any type (call-by-reference)
template <typename T>
T const & max (T const & a, T const & b, T const & c)
{
    return max(max(a, b), c); // error if max(a,b) uses call-by-value
} 

auto m1 = ::max(7, 42, 68);   // OK
char const* s1 = "frederic";
char const* s2 = "anica";
char const* s3 = "lucas";
auto m2 = ::max(s1, s2, s3);  // RUNTIME ERROR
```
The problem is that if you call `max` for three C-strings, the statement
```c++
return max (max(a,b), c);
```
becomes a run-time error because for C-strings, 
`max(a, b)` creates a new, temporary local value that is returned by reference, 
but that temporary value expires as soon as the return statement is complete, 
leaving `main` with a dangling reference. 
Unfortunately, the error is quite subtle and may not manifest itself in all cases.


Note, in contrast, that the first call to `max` in `main` doesn’t suffer from the same issue. 
There temporaries are created for the arguments (7, 42, and 68), 
but those temporaries are created in main where they persist until the statement is done. 


This is only one example of code that might behave differently than expected
as a result of detailed overload resolution rules. 
In addition, ensure that all overloaded versions of a function
are declared before the function is called. 
This is because the fact that not all overloaded functions are visible 
when a corresponding function call is made may matter. 
For example, defining a three-argument version of `max` 
without having seen the declaration of a special two-argument version of `max` for `int`s 
causes the two-argument template to be used by the three-argument version:
```c++
// maximum of two values of any type:
template <typename T>
T max(T a, T b)
{
    return b < a ? a : b;
}

// maximum of three values of any type:
template <typename T>
T max T a, T b, T c)
{
    // uses the template version even for ints 
    // because the following declaration comes too late:
    return max(max(a, b), c); 
}

int max(int a, int b)
{
    return b < a ? a : b;
}

::max(47, 11, 33); // OOPS: uses max<T>() instead of max(int, int)
```
We discuss details in Section 13.2. 


### 📌 1.6 But, Shouldn’t We...?

#### Pass by Value or by Reference?

You might wonder why we in general declare the functions 
to pass the arguments by value instead of using references. 
In general, passing by reference is recommended 
for types other than cheap simple types 
(such as fundamental types or `std::string_view`), 
because no unnecessary copies are created. 


However, for a couple of reasons, passing by value in general is often better:
- The syntax is simple.
- Compilers optimize better.
- Move semantics often makes copies cheap.
- And sometimes there is no copy or move at all.


In addition, for templates, specific aspects come into play:
- A template might be used for both simple and complex types, 
  so choosing the approach for complex types 
  might be counter-productive for simple types.
- As a caller you can often still decide to pass arguments by reference, 
  using `std::ref` and `std::cref` (see Section 7.3).
- Although passing string literals or raw arrays always can become a problem,
  passing them by reference often is considered to become the bigger problem. 
  All this will be discussed in detail in Chapter 7. 
  For the moment inside the book we will usually pass arguments by value 
  unless some functionality is only possible when using references.

#### Why Not `inline`?

In general, function templates **don’t** have to be declared with `inline`. 
Unlike ordinary non-`inline` functions, 
we can define non-`inline` function templates in a header file
and include this header file in multiple translation units.


The only exception to this rule are full specializations of templates for specific types, 
so that the resulting code is no longer generic (all template parameters are defined). 
See Section 9.2 for more details.


From a strict language definition perspective, 
`inline` _only_ means that a definition of a function can appear multiple times in a program. 
However, it is also meant as a _hint_ to the compiler that 
calls to that function should be “expanded inline”: 
Doing so can produce more efficient code for certain cases, 
but it can also make the code less efficient for many other cases. 
Nowadays, compilers usually are better at deciding this without the hint implied by the `inline` keyword. 
However, compilers still account for the presence of inline in that decision.

#### Why Not constexpr?

Since C++11, you can use `constexpr` to provide the ability 
to use code to compute some values at compile time. 
For a lot of templates this makes sense.


For example, 
to be able to use the maximum function at compile time, 
you have to declare it as follows:
```c++
template <typename T1, typename T2>
constexpr auto max(T1 a, T2 b)
{
    return b < a ? a : b;
}
```
With this, you can use the maximum function template in places with compile-time context: 
```c++
int a[::max(sizeof(char), 1000U)];
std::array<std::string, ::max(sizeof(char), 1000U)> arr;
```
Note that we pass `1000` as `unsigned int` to avoid warnings 
about comparing a signed with an unsigned value inside the template. 


Section 8.2 will discuss other examples of using `constexpr`.
However, to keep our focus on the fundamentals, 
we usually will skip `constexpr` when discussing other template features.


### 📌 1.7 Summary


- Function templates define a family of functions for different template arguments.
- When you pass arguments to function parameters depending on template parameters, 
  function templates deduce the template parameters 
  to be instantiated for the corresponding parameter types.
- You can explicitly qualify the leading template parameters.
- You can define default arguments for template parameters. 
  These may refer to previous template parameters 
  and be followed by parameters not having default arguments.
- You can overload function templates.
- When overloading function templates with other function templates, 
  you should ensure that only one of them matches for any call.
- When you overload function templates, 
  limit your changes to specifying template parameters explicitly.
- Ensure the compiler sees all overloaded versions of function templates before you call them.






### 🎯 Chapter 2 Class Templates


### 📌 2.1 Implementation of Class Template Stack


### 📌 2.2 Use of Class Template Stack


To use an object of a class template, 
until C++17 you must always specify the template arguments explicitly.
C++17 introduced _class argument template deduction_, 
which allows skipping template arguments if they can be derived from the constructor. 
This will be discussed in Section 2.9. 
```c++
template <typename T>
class Stack { ... };

Stack<int> intStack;

// manipulate int stack
intStack.push(7);
std::cout << intStack.top() << '\n';

// manipulate string stack
stringStack.push("hello");
std::cout << stringStack.top() << '\n';
stringStack.pop();
```
By declaring type `Stack<int>`, `int` is used as type `T` inside the class template.
Thus, `intStack` is created as an object that uses a vector of `int`s as elements and,
for all member functions that are called, code for this type is instantiated. 
Similarly, by declaring and using `Stack<std::string>,` 
an object that uses a vector of `std::string`s as elements is created, 
and for all member functions that are called, code for this type is instantiated.


Note that code is instantiated _only for template (member) functions that are called_. 
For class templates, member functions are instantiated only if they are used. 
This, of course, saves time and space and allows use of class templates only partially, 
which we will discuss in Section 2.3. 


In this example, the default constructor, `push`, and `top` 
are instantiated for both `int`s and `std::string`s.  
However, `pop` is instantiated only for `std::string`s. 
If a class template has `static` members, 
these are also instantiated once for each type for which the class template is used.


An instantiated class template’s type can be used just like any other type. 
You can qualify it with `const` or `volatile` or derive array and reference types from it.
You can also use it as part of a type definition with `typedef` or `using` (see Section 2.8) 
or use it as a type parameter when building another template type. 


Template arguments may be any type, such as pointers to `float`s or even stacks of `int`s:
```c++
Stack<float *> floatPointerStack;
Stack<Stack<int>> intStackStack;
```
The only requirement is that any operation that is called is possible according to this type.


Note that before C++11, you had to put whitespace between the two closing template brackets: 
```c++
Stack<Stack<int> > intStackStack;  // OK with all C++ versions
Stack<Stack<int>> intStackStack;   // ERROR before C++11
```
The reason for the old behavior was that it helped the first pass of a C++ compiler 
to tokenize the source code independent of the semantics of the code. 
However, because the missing space was a typical bug, 
which required corresponding error messages, 
the semantics of the code more and more had to get taken into account anyway. 
So, with C++11 the rule to put a space between two closing template brackets 
was removed with the “angle bracket hack” (see Section 13.3). 


### 📌 2.3 Partial Usage of Class Templates
















## 🌱

### 🎯

### 📌 








