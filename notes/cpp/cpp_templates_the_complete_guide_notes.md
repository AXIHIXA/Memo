# _C++ Templates: The Complete Guide_ Notes

- _**C++ Templates**: The Complete Guide_ Second Edition
- David Vandevoorde
- Nicolai M. Josuttis
- Douglas Gregor






## 🌱 Part I The Basics


### 🎯 Chapter 1 Function Templates

#### 📌 1.1 A First Look at Function Templates

##### Instantiation

Templates aren't compiled into single entities that can handle any type. 
Instead, different entities are generated from the template for every type for which the template is used.


The process of replacing template parameters by concrete types is called _instantiation_. 
It results in an _instance_ of a template.


Note that the mere use of a function template can trigger such an instantiation process. 
There is no need for the programmer to request the instantiation separately.

##### Two-Phase Translation

Templates are "compiled" in two phases:
1. Without instantiation at _definition_ time, 
   the template code itself is checked for correctness ignoring the template parameters.
   This includes:
   – Syntax errors are discovered, such as missing semicolons.
   – Using unknown names (type names, function names, ...) 
     that don't depend on template parameters are discovered.
   – Static assertions that don't depend on template parameters are checked.
2. At _instantiation_ time, 
   the template code is checked (again) to ensure that all code is valid. 
   That is, now especially, all parts that depend on template parameters are double-checked.


The fact that names are checked twice is called _two-phase lookup_ 
and discussed in detail in Section 14.3.


Note that some compilers don't perform the full checks of the first phase. 
So you might not see general problems until the template code is instantiated at least once.

##### Compiling and Linking

Two-phase translation leads to an important problem in the handling of templates in practice: 
When a function template is used in a way that triggers its instantiation, 
which is at _compile time_, 
a compiler will (at some point) need to see that template's _definition_. 
This breaks the usual compile and link distinction for ordinary functions, 
when the declaration of a function is sufficient to compile its use. 
Methods of handling this problem are discussed in Chapter 9. 
For the moment, let's take the simplest approach: 
Implement each template inside a header file.

#### 📌 1.2 Template Argument Deduction

##### Type Conversions During Type Deduction

- During template type deduction,
  arguments will _decay_ **unless** they are used to initialize references. 
  - Functions and arrays decay to corresponding pointer types;
  - Arguments' reference-ness and cv-constraints are removed.
- When deducing types for universal reference parameters,
  reference collapse may occur.
- Only the following implicit conversions are allowed when deducing a parameterized type: 
  - _Qualification conversion_ (adding `cv`-qualifiers).
    - Only bottom-level cv for reference and pointer parameter types.
    - Regular types only have top-level cv, which is decayed.
  - _Derived-to-base conversion_ (for regular types or pointer types),  
    **unless** deduction occurs for a conversion operator template.
  - If a parameterized type does **not** need to be deduced, all implicit conversions apply. 


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

##### Type Deduction for Default Arguments

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

#### 📌 1.3 Multiple Template Parameters

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
regardless of the caller's intention. 
Thus, the return type depends on the call argument order. 
The maximum of `66.66` and `42` will be the `double` `66.66`, 
while the maximum of `42` and `66.66` will be the `int` `66`. 


C++ provides different ways to deal with this problem (to de detailed in the next section):
- Introduce a third template parameter for the return type.
- Let the compiler deduce the return type.
- Declare the return type to be the "common type" of the two parameter types. 

##### Template Parameters for Return Types

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
Note that these modified versions of `max` **don't** lead to significant advantages. 
For the one-parameter version, you can already specify the parameter (and return) type 
if two arguments of a different type are passed. 
Thus, it is a good idea to keep it simple and use the one-parameter version of `max`. 


See Chapter 15 for details of the deduction process.

##### Deducing the Return Type

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
auto max(T1 a, T2 b) -> decltype(b < a ? a : b)
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
auto max(T1 a, T2 b) -> decltype(b < a ? a : b)
```
is a _declaration_, so that the compiler uses the rules of `operator?:` called for parameters `a` and `b` 
to find out the return type of `max` at compile time. 
The implementation does **not** necessarily have to match. 
In fact, using `true` as the condition for `operator?:` in the declaration is enough:
```c++
template <typename T1, typename T2>
auto max(T1 a, T2 b) -> decltype(true ? a : b);
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

##### Return Type as Common Type

Since C++11, the C++ standard library provides a means to specify choosing "the more general type." 
`std::common_type<>::type` yields the "common type" of two (or more) different types passed as template arguments. 
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

#### 📌 1.5 Overloading Function Templates

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
you **can't** use the three-argument version to compute the maximum of three C-strings:
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


Note, in contrast, that the first call to `max` in `main` doesn't suffer from the same issue. 
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


#### 📌 1.6 But, Shouldn't We...?

##### Pass by Value or by Reference?

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

##### Why Not `inline`?

In general, function templates **don't** have to be declared with `inline`. 
Unlike ordinary non-`inline` functions, 
we can define non-`inline` function templates in a header file
and include this header file in multiple translation units.


The only exception to this rule are full specializations of templates for specific types, 
so that the resulting code is no longer generic (all template parameters are defined). 
See Section 9.2 for more details.


From a strict language definition perspective, 
`inline` _only_ means that a definition of a function can appear multiple times in a program. 
However, it is also meant as a _hint_ to the compiler that 
calls to that function should be "expanded inline": 
Doing so can produce more efficient code for certain cases, 
but it can also make the code less efficient for many other cases. 
Nowadays, compilers usually are better at deciding this without the hint implied by the `inline` keyword. 
However, compilers still account for the presence of inline in that decision.

##### Why Not `constexpr`?

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


#### 📌 1.7 Summary


- Function templates define a family of functions for different template arguments.
- When you pass arguments to function parameters depending on template parameters, 
  function templates deduce the template parameters 
  to be instantiated for the corresponding parameter types.
  Template argument deduction only works for immediate calls. 
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


#### 📌 2.1 Implementation of Class Template Stack


#### 📌 2.2 Use of Class Template Stack


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


Note that **code is instantiated _only for template (member) functions that are called_**. 
For class templates, member functions are instantiated only if they are used. 
This, of course, saves time and space and allows use of class templates only partially, 
which we will discuss in Section 2.3. 


In this example, the default constructor, `push`, and `top` 
are instantiated for both `int`s and `std::string`s.  
However, `pop` is instantiated only for `std::string`s. 
If a class template has `static` members, 
these are also instantiated once for each type for which the class template is used.


An instantiated class template's type can be used just like any other type. 
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
was removed with the "angle bracket hack" (see Section 13.3). 


#### 📌 2.3 Partial Usage of Class Templates


A class template usually applies multiple operations 
on the template arguments it is instantiated for (including construction and destruction). 
This might lead to the impression that these template arguments 
have to provide all operations necessary for all member functions of a class template. 
But this is **not** the case: 
Template arguments only have to provide all necessary operations that are needed 
(instead of that could be needed).
If, for example, class `Stack<>` would provide a member function `printOn` to print the whole stack content, 
which calls `operator<<` for each element:
```c++
template <typename T>
class Stack 
{
    ...
    
    void printOn() (std::ostream & cout) const 
    {
        for (T const & elem : elems) 
        {
            cout << elem << ' ';
        }
    }
};
```
You can still use this class for elements that **don't** have `operator<<` defined:
```c++
// note: std::pair<> has no operator<< defined
Stack<std::pair<int, int>> ps; 
ps.push({4, 5});                       // OK
ps.push({6, 7});                       // OK
std::cout << ps.top().first << '\n';   // OK
std::cout << ps.top().second << '\n';  // OK
```
Only if you call `printOn` for such a stack, the code will produce an error, 
because it can't instantiate the call of `operator<<` for this specific element type:
```c++
// ERROR: operator<< not supported for element type
ps.printOn(std::cout);
```

##### Concepts

This raises the question: 
How do we know which operations are required for a template to be able to get instantiated? 
The term _concept_ is often used to denote a set of constraints that is repeatedly required in a template library. 
For example, the C++ standard library relies on such concepts 
as _random access iterator_ and _default constructible_. 


As of C++17, concepts can more or less only be expressed in the documentation (e.g., code comments). 
This can become a significant problem because failures to follow constraints 
can lead to terrible error messages (see Section 9.4).
For years, there have also been approaches and trials 
to support the definition and validation of concepts as a language feature. 
However, up to C++17 no such approach was standardized yet.
C++20 standard introduced concepts,
yet many compilers lack good support on C++20 (as of March 2022). 


Since C++11, you can at least check for some basic constraints 
by using the `static_assert` keyword and some predefined type traits. 
For example:
```c++
template <typename T>
class C
{
    static_assert(std::is_default_constructible<T>::value,
                  "Class C requires default-constructible elements");
    ...
};
```
Without this assertion the compilation will still fail, 
if the default constructor is required. 
However, the error message then might contain the entire template instantiation history 
from the initial cause of the instantiation 
down to the actual template definition in which the error was detected (see Section 9.4).


However, more complicated code is necessary to check,
for example, objects of type `T` provide a specific member function 
or that they can be compared using `operator<`. 
See Section 19.6 for a detailed example of such code.


See Appendix E for a detailed discussion of concepts for C++. 


#### 📌 2.4 Friends


Instead of printing the stack contents with `printOn`, 
it is better to implement `operator<<` for the stack.
However, as usual `operator<<` has to be implemented as nonmember function, 
which then could call `printOn` inline. 
Note that this means that `operator<<` for class `Stack` is **not** a function template, 
but an "ordinary" function instantiated with the class template if needed.
It is a _templated entity_, see Section 12.1. 

If we could accept to _declare and define_ friend `operator<<` 
_together_ inside the `Stack` body, thing are simple: 
```c++
template <typename T>
class Stack
{
    ...

    void printOn()(std::ostream & cout) const
    {
        ...
    }

    friend std::ostream & operator<<(std::ostream & cout, Stack const & s)
    {
        s.printOn(cout);
        return cout;
    }
};
```
However, when trying to _declare_ the friend function and _define_ it afterwards,
things become more complicated. 
In fact, we have two options:
1. We can implicitly declare a new function template, 
   which must use a different template parameter, such as `U`:
   ```c++
   template <typename T>
   class Stack
   {
       ...
   
       template <typename U>
       friend std::ostream & operator<<(std::ostream &, Stack<U> const &);
   };
   ```
   Neither using `T` again nor skipping the template parameter declaration would work
   (either the inner `T` hides the outer `T` or we declare a non-template function in namespace scope).
2. We can forward declare the output operator for a `Stack<T>` to be a template,
   which, however, means that we first have to forward declare `Stack<T>`. 
   Then, we can declare this function as friend:
   ```c++
   template <typename T>
   class Stack;
   
   template <typename T>
   std::ostream & operator<<(std::ostream &, Stack<T> const &);

   template <typename T>
   class Stack
   {
       ...
       
       friend std::ostream & operator<<<T>(std::ostream &, Stack<T> const &);
   };
   ```
   Note the `<T>` behind the "function name" `operator<<`. 
   Thus, we declare a specialization of the non-member function template as friend. 
   Without `<T>` we would declare a new non-template function. 
   See Section 12.5 for details.


In any case, you can still use this class for elements that don't have `operator<<` defined. 
Only calling `operator<<` for this stack results in an error:
```c++
// std::pair<> has no operator<< defined
Stack<std::pair<int, int>> ps;

// OK
ps.push({4, 5});
ps.push({6, 7});
std::cout << ps.top().first << '\n';
std::cout << ps.top().second << '\n';

// ERROR: operator<< not supported for element type
std::cout << ps << '\n'; 
```


#### 📌 2.5 Specializations of Class Templates


You can specialize a class template for certain template arguments. 
Similar to the overloading of function templates (see Section 1.5), 
specializing class templates allows you to optimize implementations for certain types 
or to fix a misbehavior of certain types for an instantiation of the class template. 
However, if you specialize a class template, you must also specialize all member functions. 
Although it is possible to specialize a single member function of a class template, 
once you have done so, 
you can **no longer** specialize the whole class template instance 
that the specialized member belongs to. 


To specialize a class template,
you have to declare the class with a leading `template<>` 
and a specification of the types for which the class template is specialized. 
The types are used as a template argument and must be specified directly following the name of the class.
For these specializations, any definition of a member function must be defined as an "ordinary" member function,
with each occurrence of `T` being replaced by the specialized type.
```c++
template <>
class Stack<std::string>
{
public:
    void push(std::string const &);

    void pop();

    [[nodiscard]] std::string const & top() const;

    [[nodiscard]] bool empty() const
    {
        return elems.empty();
    }

private:
    std::deque<std::string> elems;
};

void Stack<std::string>::push(std::string const & elem)
{
    elems.push_back(elem);
}

void Stack<std::string>::pop()
{
    assert(!elems.empty());
    elems.pop_back();
}

std::string const & Stack<std::string>::top() const
{
    assert(!elems.empty());
    return elems.back();
}
```
In this example, the specialization uses reference semantics to pass the `std::string` argument to `push`, 
which makes more sense for this specific type 
(we should even better pass a forwarding reference, though, 
which is discussed in Section 6.1). 


Another difference is to use a `std::deque` instead of a `std::vector` to manage the elements inside the stack. 
Although this has no particular benefit here,
it does demonstrate that the implementation of a specialization
might look very different from the implementation of the primary template.


#### 📌 2.6 Partial Specialization


Class templates can be partially specialized. 
You can provide special implementations for particular circumstances, 
but some template parameters must still be defined by the user. 
For example, we can define a special implementation of class `Stack` for pointers:
```c++
// partial specialization of class Stack<> for pointers:
template <typename T>
class Stack<T *>
{
public:
    void push(T *);

    T * pop();

    T * top() const;

    [[nodiscard]] bool empty() const
    {
        return elems.empty();
    }

private:
    std::vector<T *> elems;
};

template <typename T>
void Stack<T *>::push(T * elem)
{
    elems.push_back(elem);
}

template <typename T>
T * Stack<T *>::pop()
{
    assert(!elems.empty());
    T * p = elems.back();
    elems.pop_back();
    return p;
}

template <typename T>
T * Stack<T *>::top() const
{
    assert(!elems.empty());
    return elems.back();
}
```
with `template <typename T> Stack<T *>`, we define a class template, 
still parameterized for `T` but specialized for a pointer. 


Note again that the specialization might provide a (slightly) different interface.
Here, for example, `pop` returns the stored pointer, 
so that a user of the class template can call `delete` for the removed value, 
when it was created with `new`:
```c++
Stack<int *> ptrStack;
ptrStack.push(new int {42});
std::cout << *ptrStack.top() << '\n';
delete ptrStack.pop();
```

##### Partial Specialization with Multiple Parameters

Class templates might also specialize the relationship between multiple template parameters.
```c++
template <typename T1, typename T2>
class MyClass
{
    ...
};

// Partial specialization: 
// Both template parameters have same type
template <typename T>
class MyClass<T, T>
{
    ...
};

// Partial specialization: 
// Second type is int
template <typename T>
class MyClass<T, int>
{
    ...
};

// Partial specialization: 
// Both template parameters are pointer types
template <typename T1, typename T2>
class MyClass<T1 *, T2 *>
{
    ...
};

MyClass<int, float> mif;     // uses MyClass<T1, T2>
MyClass<float, float> mff;   // uses MyClass<T, T>
MyClass<float, int> mfi;     // uses MyClass<T, int>
MyClass<int *, float *> mp;  // uses MyClass<T1 *, T2 *>

// ERROR: Matches both MyClass<T, T> and MyClass<T, int>
MyClass<int, int> m;  

// ERROR: Matches both MyClass<T, T> and MyClass<T1 *, T2 *>
MyClass<int *, int *> m;
```
To resolve the second ambiguity, 
you could provide an additional partial specialization for pointers of the same type:
```c++
template <typename T>
class MyClass<T *, T *> 
{
    ...
};
```
For details of partial specialization, see Section 16.4. 


#### 📌 2.7 Default Class Template Arguments


As for function templates, 
you can define default values for class template parameters. 
For example, in class `Stack` you can define the container 
that is used to manage the elements as a second template parameter, 
using `std::vector` as the default value.
Note that we now have two template parameters,
so each definition of a member function must be defined with these two parameters. 
```c++
template <typename T, typename Cont = std::vector<T>>
class Stack
{
public:
    void push(T const & elem);

    void pop();

    T const & top() const;

    [[nodiscard]] bool empty() const
    {
        elems.empty();
    }

private:
    Cont elems;
};

template <typename T, typename Cont>
void Stack<T, Cont>::push(T const & elem)
{
    elems.push_back(elem);
}

template <typename T, typename Cont>
void Stack<T, Cont>::pop()
{
    assert(!elems.empty());
    elems.pop_back();
}

template <typename T, typename Cont>
T const & Stack<T, Cont>::top() const
{
    assert(!elems.empty());
    return elems.back();
}

// Stack<int, std::vector<int>>
Stack<int> intStack;

// Stack<double, std::deque<double>>
Stack<double, std::deque<double>> doubleStack;
```


#### 📌 2.8 Type Aliases

##### Typedefs and Alias Declarations

```c++
// Typedef
typedef Stack<int> intStack;

// Type alias
using IntStack = Stack<int>;
```

##### Alias Templates

Unlike a `typedef`, an alias declaration can be templated 
to provide a convenient name for a family of types. 
This is also available since C++11 and is called an _alias template_. 


Alias templates are sometimes (incorrectly) referred to as _typedef templates_
because they fulfill the same role that a `typedef` would 
if it could be made into a template.


The following alias template `DequeStack`, 
parameterized over the element type `T`,
expands to a `Stack` that stores its elements in a `std::deque`:
```c++
template <typename T>
using DequeStack = Stack<T, std::deque<T>>;
```
Thus, both class templates and alias templates can be used as a parameterized type.
But again, an alias template simply gives a new name to an existing type, 
which can still be used. 
Both `DequeStack<int>` and `Stack<int, std::deque<int>>` represent the same type.


Note again that, in general, templates can only be declared and defined in
namespace scope (including global namespace scope) or inside class declarations.

##### Alias Templates for Member Types

Alias templates are especially helpful to define shortcuts for types 
that are members of class templates. 
```c++
template <typename T>
struct C
{
    typedef ... iterator;
    ...
};

template <typename T>
struct MyType
{
    using iterator = ...;
    ...
};

template <typename T>
using MyTypeIterator = typename MyType<T>::iterator;
```

##### Type Traits Suffix `_t`

Since C++14, the standard library uses this technique to define shortcuts 
for all type traits in the standard library that yield a type. 
```c++
/// <type_traits>
/// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0

namespace std
{

template <typename T>
using add_const_t = typename add_const<T>::type;

}  // namespace std
```


#### 📌 2.9 Class Template Argument Deduction


Prior to C++17, 
you _always_ had to pass _all_ template parameter types to class templates, 
unless they have default values. 
Since C++17, the constraint was relaxed. 
Instead, you can skip to define the templates arguments explicitly, 
if the _constructor_ is able to deduce all template parameters (that don't have a default value).
(If there is no constructor, then class template argument deduction fails.)


For example, in all previous code examples, 
you can use a copy constructor **without** specifying the template arguments:
```c++
Stack<int> intStack1;
Stack<int> intStack2 = intStack1;
Stack intStack3 = intStack1;
```
By providing constructors that pass some initial arguments, 
you can support deduction of the element type of a stack. 
For example, we could provide a stack that can be initialized by a single element:
```c++
template <typename T>
class Stack
{
public:
    Stack() = default;
    Stack(T const & elem) : elems {elem} {}
    
    // ...

private:
    std::vector<T> elems;
};

Stack intStack = 0;  // Stack<int> deduced since C++17
```
By initializing the stack with the integer `0`, 
the template parameter `T` is deduced to be `int`, 
so that a `Stack<int>` is instantiated.


Note the following:
- Due to the definition of the `int` constructor, 
  you have to request the compiler-generated default constructors manually, 
  because the default constructor is available only if there is no other user-defined constructor. 
- The argument `elem` is passed to `elems` with braces around 
  to initialize the vector elems with an initializer list with `elem` as the only argument.
  There is no constructor for a vector that is able to 
  take a single parameter as initial element directly. 


Note that, unlike for function templates, 
class template arguments may **not** be deduced only partially 
(by explicitly specifying only some of the template arguments). 
See Section 15.12 for details.

##### Class Template Arguments Deduction with String Literals

In principle, you can even initialize the stack with a string literal:
```c++
Stack stringStack = "bottom";  // Stack<char const[7]> deduced since C++17
```
**BUT** this causes a lot of trouble: 
In general, when passing arguments of a template type `T` _by reference_, 
the parameter **doesn't** _decay_ (low-level `const`-ness is kept).  
This means that we really initialize a `Stack<char const[7]>` 
and use type `char const [7]` wherever `T` is used. 
For example, we may **not** push a string of different size, because it has a different type. 
For a detailed discussion see Section 7.4. 


However, when passing arguments of a template type `T` _by value_, the parameter _decays_. 
That is, the call parameter `T` of the constructor is deduced to be `char const *` 
so that the whole class is deduced to be a `Stack<char const*>`. 


For this reason, it might be worthwhile to declare the constructor so that the
argument is passed by value:
```c++
template <typename T>
class Stack
{
public:
    Stack(T elem) : elems {std::move(elem)} {}
    
    // ...

private:
    std::vector<T> elems;
};

Stack stringStack = "bottom";  // Stack<const char *> deduced
```

##### Deduction Guides


Instead of declaring the constructor to be called by value, 
there is a different solution: 
Because handling raw pointers in containers is a source of trouble, 
we should disable automatically deducing raw character pointers for container classes.


You can define specific _deduction guides_ 
to provide additional or fix existing class template argument deductions. 
For example, you can define that whenever a string literal or C string is passed, 
the stack is instantiated for `std::string`:
```c++
Stack(char const *) -> Stack<std::string>;
```
This guide has to appear in the same scope (namespace) as the class definition.
Usually, it follows the class definition. 
We call the type following the `->` I of the deduction guide. 


Now, the declaration with 
```c++
Stack stringStack {"bottom"};
```
deduces the stack to be a `Stack<std::string>`. 
However, the following still **doesn't** work:
```c++
// Stack<std::string> deduced, but still not valid
Stack stringStack = "bottom";
```
We deduce std::string so that we instantiate a Stack<std::string>:
```c++
class Stack
{
public:
    Stack(std::string const & elem) : elems {elem} {}
    // ...

private:
    std::vector<std::string> elems;
};
```
However, by language rules, you **can't** [copy initialize](https://en.cppreference.com/w/cpp/language/copy_initialization) 
(initialize using `=`) an object by passing a string literal to a constructor expecting a `std::string`.
Note that if this were possible, 
there will be an implicit conversion sequence 
```c++
const char * -> std::string -> Stack<std::string>
```
which involves two user-defined conversions, and that is prohibited. 


So you have to initialize the stack as follows:
```c++
Stack stringStack{"bottom"};  // Stack<std::string> deduced and valid
```
Note that, if in doubt, class template argument deduction copies. 
After declaring `stringStack` as `Stack<std::string>`, 
the following initializations declare `Stack<std::string>` (thus, calling the copy constructor) 
instead of `Stack<Stack<std::string>>`:
```c++
// Stack<std::string> deduced
Stack stack2 {stringStack};
Stack stack3 (stringStack);
Stack stack4 = {stringStack};
```
See Section 15.12 for more details about class template argument deduction.


#### 📌 2.10 Templatized Aggregates


Aggregate classes (classes/structs with **no** user-provided, explicit, or inherited constructors, 
**no** private or protected non-static data members, no virtual functions,
and **no** virtual, private, or protected base classes) 
can also be templates. 
For example: 
```c++
template <typename T>
struct ValueWithComment
{
    T value;
    std::string comment;
};
```
defines an aggregate parameterized for the type of the value it holds. 
You can declare objects as for any other class template and still use it as aggregate:
```c++
ValueWithComment<int> vc {1, "a"};
vc.value = 2;
vc.comment = "b";
```
Since C++17, you can even define _deduction guides_ for aggregate class templates:
```c++
ValueWithComment(char const *, char const *) -> ValueWithComment<std::string>;
ValueWithComment vc2 = {"hello", "initial value"};
```
Without the deduction guide, the initialization would **not** be possible, 
because `ValueWithComment` has no constructor to perform the deduction against. 


The standard library class `std::array` is also an aggregate, 
parameterized for both the element type and the size. 
The C++17 standard library also defines a deduction guide for it, 
which we discuss in Section 4.4.


#### 📌 2.11 Summary


- A class template is a class that is implemented with one or more type parameters left open.
- To use a class template, you pass the open types as template arguments. 
  The class template is then instantiated (and compiled) for these types.
- For class templates, only those member functions that are called are instantiated.
- You can (completely or partially) specialize class templates for certain types.
- Since C++17, class template arguments can automatically be deduced from constructors.
  You can also define _deduction guides_ to specialize the deduction result for certain types. 
  Template argument deduction only works for immediate calls.
- You can define aggregate class templates.
- Call parameters of a template type decay if declared to be called by value.
- Templates can only be declared and defined 
  in namespace scope (including global namespace scope) or inside class declarations.






### 🎯 Chapter 3 [Nontype Template Parameters]((https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter))


For function and class templates, template parameters **don't** have to be types. 
They can also be ordinary values.
When using such a template, you have to specify the value template arguments explicitly. 
The resulting code then gets instantiated.


#### 📌 3.1 Nontype Class Template Parameters


```c++
template <typename T, std::size_t maxSize>
class Stack
{
public:
    Stack();

    void push(T const & elem);

    void pop();

    T const & top() const;

    [[nodiscard]] bool empty() const
    {
        return numElems == 0;
    }

    [[nodiscard]] std::size_t size() const
    {
        return numElems;
    }

private:
    std::array<T, maxSize> elems;
    std::size_t numElems;
};

template <typename T, std::size_t maxSize>
Stack<T, maxSize>::Stack() : numElems(0)
{

}

template <typename T, std::size_t maxSize>
void Stack<T, maxSize>::push(T const & elem)
{
    assert(numElems < Maxsize);
    elems[numElems++] = elem;
}

template <typename T, std::size_t maxSize>
void Stack<T, maxSize>::pop()
{
    assert(!elems.empty());
    --numElems;
}

template <typename T, std::size_t maxSize>
T const & Stack<T, maxSize>::top() const
{
    assert(!elems.empty());
    return elems[numElems - 1];
}

Stack<int, 20> int20Stack;
Stack<int, 40> int40Stack;
Stack<std::string, 40> stringStack;

int20Stack.push(7);
std::cout << int20Stack.top() << '\n';
int20Stack.pop();

stringStack.push("hello");
std::cout << stringStack.top() << '\n';
stringStack.pop();
```
Note that each template instantiation is its own type. 
Thus, `int20Stack` and `int40Stack` are two different types, 
and **no** implicit or explicit type conversion between them is defined. 
Thus, one can **not** be used instead of the other, and you can **not** assign one to the other. 
Again, default arguments for the template parameters can be specified: 
```c++
template <typename T = int, std::size_t maxSize = 100>
class Stack 
{
    ...
};
```
However, from a perspective of good design, this may **not** be appropriate in this example. 
Default arguments should be intuitively correct. 
But neither type `int` nor a maximum size of `100` seems intuitive for a general stack type. 
Thus, it is better when the programmer has to specify both values explicitly 
so that these two attributes are always documented during a declaration.


#### 📌 3.2 Nontype Function Template Parameters


You can also define nontype parameters for function templates. 
For example, the following function template defines a group of functions 
for which a certain value can be added:
```c++
template <int val, typename T>
T addValue(T x)
{
    return x + val;
}
```
These kinds of functions can be useful if functions or operations are used as parameters. 
For example, if you use the C++ standard library, 
you can pass an instantiation of this function template 
to add a value to each element of a collection:
```c++
std::transform(source.begin(), source.end(), dest.begin(), addValue<5, int>);
```
The last argument instantiates the function template `addValue` to add `5` to a passed `int` value. 
The resulting function is called for each element in the source collection source, 
while it is translated into the destination collection `dest`.


Note that you have to specify the argument `int` for the template parameter `T` of `addValue`. 
Template argument deduction only works for immediate calls. 
`std::transform` needs a complete type to deduce the type of its fourth parameter. 
There is no support to substitute/deduce only some template parameters. 
Again, you can also specify that a template parameter is deduced from the previous parameter. 
For example, to derive the return type from the passed nontype:
```c++
template <auto val, typename T = decltype(val)>
T foo();
```
or to ensure that the passed value has the same type as the passed type:
```c++
template<typename T, T val>
T bar();
```


#### 📌 3.3 [Restrictions for Nontype Template Parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter)


Note that nontype template parameters carry some restrictions. 
In general, they can be only constant integral values (including enumerations), 
pointers to objects/functions/members, 
lvalue references to objects or functions, or `std::nullptr_t`. 


Prior to C++20, floating-point numbers and class-type objects 
are **not** allowed as nontype template parameters:
```c++
// ERROR: floating-point values are not allowed as template parameters
template <double c>
double process(double v)
{
    return v * c;
}

// ERROR: floating-point values are not allowed as template parameters
template <std::string name>
class MyClass
{
    // ...
};
```
When passing template arguments to pointers or references, 
the objects must **not** be string literals, temporaries, or data members and other sub-objects. 


Because these restrictions were relaxed with each and every C++ version before C++17, 
additional constraints apply:
- In C++11, the objects also had to have external linkage. 
- In C++14, the objects also had to have external or internal linkage. 
Thus, the following is not possible:
```c++
// ERROR: string literal "hello" not allowed
template <char const * name>
class Message
{
    ...
};

Message<"hello"> x; 
```
However, there are workarounds (again depending on the C++ version):
```c++
extern char const s03[] = "hi";  // external linkage
char const s11[] = "hi";         // internal linkage

int main()
{
    Message<s03> m03;                // OK (all versions)
    Message<s11> m11;                // OK since C++11
    static char const s17[] = "hi";  // no linkage
    Message<s17> m17;                // OK since C++17
}
```
In all three cases, a constant `char` array is initialized by `"hello"`, 
and this object is used as a template parameter declared with `char const *`. 
This is valid in all C++ versions if the object has external linkage (`s03`), 
in C++11 and C++14 also if it has internal linkage (`s11`), 
and since C++17 if it has no linkage at all.


See Section 12.3 for a detailed discussion 
and Section 17.2 for a discussion of possible future changes in this area. 

##### Avoiding Invalid Expressions

Arguments for nontype template parameters might be any compile-time expressions.
For example: 
```c++
template<int I, bool B>
class C { ... };
C<sizeof(int) + 4, sizeof(int) == 4> c;
```
However, note that if `operator>` is used in the expression, 
you have to put the whole expression into parentheses `()` so that the nested `>` ends the argument list:
```c++
C<42, sizeof(int) > 4> c;    // ERROR: the first > ends the template argument list
C<42, (sizeof(int) > 4)> c;  // OK
```


#### 📌 3.4 Template Parameter Type `auto`


Since C++17, you can define a nontype template parameter to 
generically accept any type that is allowed for a nontype parameter. 
Using this feature, we can provide an even more generic stack class with fixed size. 

By defining by using the placeholder type `auto`, 
you define `maxSize` to be a value of a type not specified yet. 
It might be any type that is allowed to be a nontype template parameter type.
Internally you can use both the value and its type. 
```c++
template <typename T, auto maxSize>
class Stack
{
public:
    using size_type = std::decay_t<decltype(maxSize)>;
    
    Stack();

    void push(T const & elem);

    void pop();

    T const & top() const;

    [[nodiscard]] bool empty() const
    {
        return numElems == 0;
    }

    [[nodiscard]] size_type size() const
    {
        return numElems;
    }

private:
    std::array<T, maxSize> elems;
    size_type numElems;
};

template <typename T, auto maxSize>
Stack<T, maxSize>::Stack() : numElems(0)
{

}

template <typename T, auto maxSize>
void Stack<T, maxSize>::push(T const & elem)
{
    assert(numElems < Maxsize);
    elems[numElems++] = elem;
}

template <typename T, auto maxSize>
void Stack<T, maxSize>::pop()
{
    assert(!elems.empty());
    --numElems;
}

template <typename T, auto maxSize>
T const & Stack<T, maxSize>::top() const
{
    assert(!elems.empty());
    return elems[numElems - 1];
}

Stack<int, 20U> int20Stack;          // Stack of up to 20 ints
Stack<std::string, 40> stringStack;  // stack of up to 40 std::strings

int20Stack.push(7);
std::cout << int20Stack.top() << '\n';
auto size1 = int20Stack.size();

stringStack.push("hello");
std::cout << stringStack.top() << '\n';
auto size2 = stringStack.size();

if constexpr (!std::is_same_v<decltype(size1), decltype(size2)>) 
{
    std::cout << "size types differ" << '\n';
}
```
Since C++14, you could `also` just use `auto` as return type 
to let the compiler find out the return type:
```c++
template <typename T, auto maxSize>
[[nodiscard]] inline auto Stack<T, maxSize>::size() const
{
    return numElems;
}
```
because you can also pass strings as constant `char` arrays, the following is possible:
```c++
template <auto someValue>
class Message 
{
public:
    void print() 
    {
        std::cout << someValue << '\n';
    }
};

Message<42> msg1;          // initialize with int 42
msg1.print();

char const s[] = "hello";
Message<s> msg2;           // initialize with char const [6] "hello"
msg2.print();
```
Note also that even `template <decltype(auto) N>` is possible, 
which allows instantiation of `N` as a reference:
```c++
template <decltype(auto) N>
class C
{
    ...
};

int i;
C<(i)> x;  // N is int &
```
See Section 15.10 for details.


#### 📌 3.5 Summary


- Templates can have template parameters that are values rather than types. 
- For arguments for nontype template parameters, 
  you can **not** use floating-point numbers, class-type objects, 
  or pointers/references to string literals, temporaries, and sub-objects
- Using `auto` enables templates to have nontype template parameters that are values of generic types.






### 🎯 Chapter 4 [Variadic Templates (Parameter Pack)](https://en.cppreference.com/w/cpp/language/parameter_pack)


Since C++11, templates can have parameters that accept a variable number of template arguments. 
This feature allows the use of templates in places 
where you have to pass an arbitrary number of arguments of arbitrary types. 
A typical application is to pass an arbitrary number of parameters of arbitrary type 
through a class or framework. 
Another application is to provide generic code to process any number of parameters of any type.


#### 📌 4.1 Variadic Templates


Template parameters can be defined to accept an unbounded number of template arguments. 
Templates with this ability are called _variadic templates_.

##### Variadic Templates: Function Parameter Pack And Template Parameter Pack

```c++
void print()
{
    
} 

template <typename T, typename ... Args>
void print(T firstArg, Args ... args)
{
    std::cout << firstArg << '\n';  // print first argument
    print(args...);                 // call print() for remaining arguments
}
```
If one or more arguments are passed, the function template is used, 
which by specifying the first argument separately 
allows printing of the first argument 
before recursively calling `print` for the remaining arguments. 
These remaining arguments named `args` are a _function parameter pack_: 
```c++
void print(T firstArg, Args ... args)
```
using different `Args` specified by a _template parameter pack_:
```c++
template <typename T, typename ... Args>
```
To end the recursion, the nontemplate overload of `print` is provided, 
which is issued when the parameter pack is empty. 


For example, a call such as
```c++
std::string s("world");
print(7.5, "hello", s);
```
would be expanded and instantiated to
```c++
void print()
{
    
}

void print<std::string>(std::string firstArg)
{
    std::cout << firstArg << '\n';
    print();
}

void print<char const *, std::string>(char const * firstArg, 
                                      std::string secondArg)
{
    std::cout << firstArg << '\n';
    print<std::string>(secondArg);
}

void print<double, char const *, std::string>(double firstArg, 
                                              char const * secondArg, 
                                              std::String thirdArg)
{
    std::cout << firstArg << '\n';
    print<char const *, std::string>(secondArg, thridArg);
}

std::string s("world");
print<double, char const*, std::string>(7.5, "hello", s);
```

##### Overloading Variadic and Nonvariadic Templates

Note that you can also implement the example above as follows:
```c++
template <typename T>
void print(T arg)
{
    std::cout << arg << '\n';
} 

template <typename T, typename ... Args>
void print(T firstArg, Args ... args)
{
    std::cout << firstArg << '\n';  // print first argument
    print(args...);                 // call print() for remaining arguments
}
```
That is, if two function templates only differ by a trailing parameter pack, 
the function template **without** the trailing parameter pack is preferred.
Section C.3 explains the more general overload resolution rule that applies here.

##### Operator `sizeof...`

C++11 also introduced a new form of the `sizeof` operator for variadic templates: `sizeof...`. 
It expands to the number of elements a parameter pack contains. 
Thus,
```c++i
template <typename T, typename ... Args>
void print (T firstArg, Args ... args)
{
    std::cout << sizeof...(Args) << '\n';  // print number of remaining types
    std::cout << sizeof...(args) << '\n';  // print number of remaining args
}
```
twice prints the number of remaining arguments after the first argument passed to `print`. 
As you can see, you can call `sizeof...` for both template parameter packs and function parameter packs.


This might lead us to think we can skip the function for the end of the recursion
by not calling it in case there are no more arguments:
```c++
// ERROR
template <typename T, typename ... Args>
void print(T firstArg, Args ... args)
{
    std::cout << firstArg << '\n';
    
    if (sizeof...(args))
    {
        print(args...);
    }
}
```
However, this approach **doesn't** work,
because both branches of `if` statements in function templates are instantiated. 
Whether the instantiated code is useful is a _run-time_ decision, 
while the instantiation of the call is a _compile-time_ decision. 
For this reason, if you call the `print` function template for one (last) argument, 
the statement with the call of `print(args...)` still is instantiated for no argument, 
and if there is no function `print` for no arguments provided, this is an error. 


However, note that since C++17, a compile-time `if` (`if constexpr`) is available. 
This will be discussed in Section 8.5. 
```c++
// OK since C++17
template <typename T, typename ... Args>
void print(T firstArg, Args ... args)
{
    std::cout << firstArg << '\n';
    
    if constexpr (sizeof...(args))
    {
        print(args...);
    }
}
```


#### 📌 4.2 [Fold Expressions](https://en.cppreference.com/w/cpp/language/fold)


Since C++17, there is a feature to compute the result of 
using a binary operator over all the arguments of a parameter pack (with an optional initial value). 
For example, the following function returns the sum of all passed arguments:
```c++
template <typename ... Args>
auto foldSum(Args ... args)
{
    return (... + args);
}

// (((1 + 2) + 3) + 4)
auto sum = foldSum(1, 2, 3, 4);
```
If the parameter pack is empty, the expression is usually ill-formed 
(with the exception that for `operator&&` the value is `true`, 
for `operator||` the value is `false`, 
and for the comma operator the value for an empty parameter pack is `void()`).


You can use almost all binary operators for fold expressions (see Section 12.4 for details). 
For example, you can use a fold expression to traverse a path in a binary tree using `operator.*`:
```c++
// define binary tree structure and traverse helpers:
struct Node
{
    explicit Node(int i = 0) : value(i), left(nullptr), right(nullptr) {}
    
    int value;
    Node * left;
    Node * right;
    
    // ...
};

Node * Node::* left = &Node::left;
Node * Node::* right = &Node::right;

template <typename NodePointer, typename ... NodeMemberPointers>
Node * traverse(NodePointer np, NodeMemberPointers ... nmps)
{
    return (np ->* ... ->* nmps);  // np->*nmp1->*nmp2->*...
}

Node root(0), a(1), b(2);
root.left = &a;
a.right = &b;
traverse(&root, left, right);
```
With such a fold expression using an initializer,
we might think about simplifying the variadic template to print all arguments, 
introduced above:
```c++
template <typename T>
class AddSpace
{
public:
    friend std::ostream & operator<<(std::ostream & cout, const AddSpace & as)
    {
        return cout << as.ref << ' ';
    }

    explicit AddSpace(const T & ref_) : ref(ref_) {}

private:
    const T & ref;
};

template <typename T>
std::ostream & operator<<(std::ostream & cout, const AddSpace<T> & as);

template <typename ... Args>
void print(const Args & ... args)
{
    (std::cout << ... << AddSpace(args)) << '\n';
}

print(1, 2.0, "hehe");  // 1 2.0 hehe
```
P.S. Stacking buffs like perfect forwarding and `enable_if`: 
```c++
template <typename T>
class AddSpace
{
public:
    // The enable_if might be redundant, 
    // as this function could be found only via ADL
    // (given that it is declared only inside template body)? 
    template <typename U,
              std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<U>>,
                                              AddSpace>,
                               bool> = true>
    friend std::ostream & operator<<(std::ostream & cout, U && as)
    {
        return cout << std::forward<U>(as).ref << ' ';
    }

    explicit AddSpace(const T & ref_) : ref(ref_) {}

private:
    const T & ref;
};

template <typename ... Args>
void print(Args && ... args)
{
    (std::cout << ... << AddSpace(std::forward<Args>(args))) << '\n';
}

int a = 10;
const int & b = a;
std::string s("string");
print(1, 3.0, "jj", a, b, std::move(s));  // 1 3 jj 10 10 string 
```
Note that the expression `AddSpace(args)` uses class template argument deduction 
to have the effect of `AddSpace<Args>(args)`, 
which for each argument creates an `AddSpace` object that refers to the passed argument 
and adds a space when it is used in output expressions.
See Section 12.4 for details about fold expressions.


#### 📌 4.3 Application of Variadic Templates


Variadic templates play an important role when implementing generic libraries, 
such as the C++ standard library. 
One typical application is the _forwarding_ of a variadic number of arguments of arbitrary type. 
For example, we use this feature when:
- Passing arguments to the constructor of a new heap object owned by a shared pointer:
```c++
auto sp = std::make_shared<std::complex<float>>(4.2, 7.7);
```
- Passing arguments to a thread, which is started by the library:
```c++
void foo(int, const std::string &) { ... }
std::thread t (foo, 42, "hello");
```
- Passing arguments to the constructor of a new element pushed into a vector:
```c++
std::vector<Customer> v;
v.emplace_back("Tim", "Jovi", 1962);
```
Usually, the arguments are "_perfectly forwarded_" with move semantics (see Section 6.1).


Note also that the same rules apply to variadic function template parameters as for ordinary parameters. 
For example, if passed by value, arguments are copied and decay (e.g., arrays become pointers), 
while if passed by reference, parameters refer to the original parameter and don't decay:
```c++
// args are copies with decayed types:
template <typename ...Args> 
void foo(Args ... args);

// args are nondecayed references to passed objects:
template <typename ... Args> 
void bar(Args const & ... args);
```


#### 📌 4.4 Variadic Class Templates and Variadic Expressions


Besides the examples above, parameter packs can appear in additional places,
including, for example, expressions, class templates, using declarations, and even deduction guides. 
Section 12.4 has a complete list.

##### Variadic Expressions

You can do more than just forwarding all the parameters. 
You can compute with them,
which means to compute with all the parameters in a parameter pack. 


For example, the following function doubles each parameter of the parameter pack `args` a
nd passes each doubled argument to `print`:
```c++
template <typename T>
void print(T && t)
{
    std::cout << std::forward<T>(t) << '\n';
}

template <typename T, typename ... Args>
void print(T && t, Args && ... args)
{
    std::cout << std::forward<T>(t) << ' ';
    print(std::forward<Args>(args)...);
}

template <typename ... Args>
void printDoubled(Args && ... args)
{
    // Function parameter pack unpacking expression, 
    // not C++17 fold expression
    print(std::forward<Args>(args) + std::forward<Args>(args)...);
}
```
If, for example, you call
```c++
printDoubled(7.5, std::string("hello"), std::complex<float>(4, 2));
```
the function has the following effect (except for any constructor side effects):
```c++
print(7.5 + 7.5,
      std::string("hello") + std::string("hello"),
      std::complex<float>(4, 2) + std::complex<float>(4, 2));
```
If you just want to add 1 to each argument, 
note that the dots from the ellipsis may **not** directly follow a numeric literal:
```c++
template <typename ... Args>
void printAddOne(Args && ... args)
{
    // These are all function parameter pack unpacking expressions
    print(std::forward<Args>(args) + 1...);    // ERROR: 1... is a literal with too many decimal points
    print(std::forward<Args>(args) + 1 ...);   // OK
    print((std::forward<Args>(args) + 1)...);  // OK
}
```
Compile-time expressions can include template parameter packs in the same way.
For example, the following function template returns 
whether the types of all the arguments are the same:
```c++
template <typename T, typename ... Args>
constexpr bool isHomogeneous(T, Args ...)
{
    // since C++17: unary right fold expression (pack op ...)
    return (std::is_same_v<T, Args> && ...);
}
```
This is an application of fold expressions: For
```c++
isHomogeneous(43, -1, "hello");
```
the expression for the return value expands to
```c++
std::is_same_v<int, int> && std::is_same_v<int, char const *>;
```
and yields `false`, while
```c++
isHomogeneous("hello", "", "world", "!");
```
yields `true` because all passed arguments are deduced to be `char const *` 
(note that the argument types decay because the call arguments are passed by value). 

##### Variadic Indices

As another example, the following function uses a variadic list of indices
to access the corresponding element of the passed first argument:
```c++
template <typename T>
void print(T && t)
{
    std::cout << t << '\n';
}

template <typename T, typename ... Args>
void print(T && t, Args && ... args)
{
    std::cout << t << ' ';
    print(std::forward<Args>(args)...);
}

template <typename Container, typename ... Idx>
void printElems(Container && c, Idx && ... idx)
{
    print(std::forward<Container>(c)[std::forward<Idx>(idx)]...);
}

std::vector<int> vec {0, 1, 2, 3, 4, 5, 6};

// Is equivalent to print(vec[2], vec[4], vec[6]);
// Prints 2 4 6
printElems(vec, 2, 4, 6);
```
You can also declare nontype template parameters to be parameter packs. 
For example:
```c++
template <std::size_t ... idx, typename Container>
void printIdx(Container && c)
{
    print(c[idx]...);
}

std::vector<int> vec {0, 1, 2, 3, 4, 5, 6};
printElems<2, 4, 6>(vec);  // 2 4 6
```

##### Variadic Class Templates


Variadic templates can also be class templates. 
An important example is a class where an arbitrary number of template parameters 
specify the types of corresponding members:
```c++
template <typename ... Elements>
class Tuple { ... };

Tuple<int, std::string, char> tup;
```
This will be discussed in Chapter 25. 


Another example is to be able to specify the possible types objects can have:
```c++
template <typename ... Types>
class Variant { ... };

Variant<int, std::string, char> var;
```
This will be discussed in Chapter 26. 


You can also define a class that as a type represents a list of indices:
```c++
// type for arbitrary number of indices:
template <std::size_t ...>
struct Indices { ... };
```
This can be used to define a function that calls `print` for the elements of a `std::array` or `std::tuple` 
using the compile-time access with `get` for the given indices:
```c++
template <typename T, std::size_t ... idx>
void printByIdx(T t, Indices<idx...>)
{
    print(std::get<idx>(t)...);
}

std::array<std::string, 5> arr = {"Hello", "my", "new", "!", "World"}; 
printByIdx(arr, Indices<0, 4, 3>());

std::tuple t {12, "monkeys", 2.0};
printByIdx(t, Indices<0, 1, 2>());
```
This is a first step towards meta-programming, 
which will be discussed in Section 8.1 and Chapter 23. 

##### Variadic Deduction Guides

Even deduction guides (see Section 2.9) can be variadic. 
For example, the C++ standard library defines the following deduction guide for `std::array`s:
```c++
namespace std 
{
template <typename T, typename ... U> 
array(T, U ...) -> array<enable_if_t<(is_same_v<T, U> && ...), T>, 
                         1 + sizeof...(U)>;
}  // namespace std
```
An initialization such as
```c++
std::array a {42, 45, 77};
```
deduces `T` in the guide to the type of the element, 
and the various `U ...` types to the types of the subsequent elements. 
The total number of elements is therefore `1 + sizeof...(U)`. 
```c++
std::array<int, 3> a {42, 45, 77};
```
The `std::enable_if_t` expression for the first array parameter 
is a fold expression that expands to:
```c++
is_same_v<T, U1> && is_same_v<T, U2> && is_same_v<T, U3> ...
```
If the result is not `true` (i.e., not all the element types are the same),
the deduction guide is discarded and the overall deduction fails. 
This way, the standard library ensures that 
all elements must have the same type for the deduction guide to succeed.

##### Variadic Base Classes and `using`

Finally, consider the following example:
```c++
class Customer
{
public:
    explicit Customer(std::string  n) : name(std::move(n)) {}
    [[nodiscard]] std::string getName() const { return name; }

private:
    std::string name;
};

struct CustomerEq
{
    bool operator()(Customer const & c1, Customer const & c2) const
    {
        return c1.getName() == c2.getName();
    }
};

struct CustomerHash
{
    std::size_t operator()(Customer const & c) const
    {
        return std::hash<std::string>()(c.getName());
    }
};

// define class that combines operator() for variadic base classes:
template <typename ... Bases>
struct Overloader : Bases...
{
    // OK since C++17
    using Bases::operator()...; 
    
    // ...
};
class Customer
{
public:
    explicit Customer(std::string  n) : name(std::move(n)) {}
    [[nodiscard]] std::string getName() const { return name; }

private:
    std::string name;
};

struct CustomerEq
{
    bool operator()(Customer const & c1, Customer const & c2) const
    {
        return c1.getName() == c2.getName();
    }
};

struct CustomerHash
{
    std::size_t operator()(Customer const & c) const
    {
        return std::hash<std::string>()(c.getName());
    }
};

// define class that combines operator() for variadic base classes:
template <typename ... Bases>
struct Overloader : Bases...
{
    // OK since C++17
    using Bases::operator()...; 
    
    // ...
};

// combine hasher and equality for customers in one type:
using CustomerOP = Overloader<CustomerHash, CustomerEq>;
std::unordered_set<Customer, CustomerHash, CustomerEq> s1;
std::unordered_set<Customer, CustomerOP, CustomerOP> s2;
```
See Section 26.4 for another application of this technique.


#### 📌 4.5 Summary


- By using parameter packs, 
  templates can be defined for an arbitrary number of template parameters of arbitrary type.
- To process the parameters, 
  you need recursion and/or a matching nonvariadic function.
- Operator `sizeof...` yields the number of arguments provided for a parameter pack.
- A typical application of variadic templates 
  is forwarding an arbitrary number of arguments of arbitrary type.
- By using fold expressions, you can apply operators to all arguments of a parameter pack. 






### 🎯 Chapter 5 Tricky Basics


This chapter covers some further basic aspects of templates
that are relevant to the practical use of templates: 
an additional use of the `typename` keyword, 
defining member functions and nested classes as templates, 
template template parameters,
zero initialization, 
and some details about using string literals as arguments for function templates. 
These aspects can be tricky at times, 
but every day-to-day programmer should have heard of them.


#### 📌 5.1 Keyword `typename`


The keyword typename was introduced during the standardization of C++ to
clarify that a _nest name_ (identifier inside a template) is a type. 
Consider the following example:
```c++
template <typename T>
class MyClass
{
public:
    // ...

    void foo()
    {
        typename T::SubType * ptr;
    }
};
```
Here, the second `typename` is used to clarify that `SubType` is a type defined within class `T`. 
Thus, `ptr` is a pointer to the type `T::SubType`.


Without `typename`, `SubType` would be assumed to be a nontype member 
(e.g., a static data member or an enumerator constant). 
As a result, the expression
```c++
T::SubType * ptr
```
would be a multiplication of the static `SubType` member of class `T` with `ptr`,
which is not an error, because for some instantiations of `MyClass` this could be valid code.


In general, `typename` has to be used whenever a name that depends on a template parameter is a type. 
This is discussed in detail in Section 13.3. 


One application of `typename` is the declaration to iterators of standard containers in generic code:
```c++
template <typename T>
void printContainer(const T & c)
{
    for (typename T::const_iteratorpos = c.cbegin(), end = c.cend(); pos != end; ++pos) 
    {
        std::cout << *pos << '\n';
    }
    
    std::cout << '\n';
}
```
See Section 13.3 for more details about the need for `typename` until C++17. 
Note that C++20 will probably remove the need for `typename` in many common cases (see Section 17.1 for details).


#### 📌 5.2 Zero Initialization


For trivial types such as `int`, `double`, or pointer types, 
there is **no** default constructor that initializes them with a useful default value. 
Instead, any noninitialized local variable has an undefined value:
```c++
void foo()
{
    int x;      // x has undefined value
    int * ptr;  // ptr points to a random location (instead of nowhere)
}
```
Now if you write templates and want to have variables of a template type initialized by a default value, 
you have the problem that a simple definition doesn't do this for built-in types:
```c++
template <typename T>
void foo()
{
    T x;  // x has undefined value if T is built-in type
}
```
For this reason, it is possible to call explicitly a default constructor 
for built-in types that initializes them with zero (or `false` for `bool` or `nullptr` for pointers). 
As a consequence, you can ensure proper initialization even for built-in types by writing the following:
```c++
template<typename T>
void foo()
{
    T x {};  // x is zero (or false) if T is a built-in type
}
```
This way of initialization is called _value initialization_, 
which means to either call a provided constructor or zero initialize an object. 
This even works if the constructor is `explicit`. 


Before C++11, the syntax to ensure proper initialization was
```c++
T x = T();  // x is zero (or false) if T is a built-in type
```
Prior to C++17, this mechanism (which is still supported) only worked 
if the constructor selected for the copy-initialization is **not** `explicit`.
The `explicit` specifier specifies that 
a constructor or conversion function or deduction guide
can **not** be used for _implicit conversions_ and _copy-initialization_. 


In C++17, _mandatory copy elision_ avoids that limitation and either syntax can work, 
but the braced initialized notation can use an initializer-list constructor 
if no default constructor is available. 


To ensure that a member of a class template, 
for which the type is parameterized, gets initialized, 
you can define a default constructor that uses a braced initializer to initialize the member:
```c++
template <typename T>
class MyClass 
{
public:
    // ensures that x is initialized even for built-in types
    explicit MyClass() : x {} {}

private:
    T x;
};
```
The pre-C++11 syntax also still works: 
```c++
// ensures that x is initialized even for built-in types
MyClass::MyClass() : x() {}
```
Since C++11, you can also provide a default in-class initialization for a nonstatic member,
so that the following is also possible:
```c++
template <typename T>
class MyClass 
{
private:
    T x {};  // zero-initialize x unless otherwise specified
    ...
};
```
However, note that default arguments cannot use that syntax. For example,
```c++
// ERROR
template <typename T>
void foo(T p {}) 
{ 
    ...
}
```
Instead, we have to write:
```c++
// OK (must use T() before C++11)
template <typename T>
void foo(T p = T {}) 
{ 
    ...
}
```


#### 📌 5.3 Using `this->` or `Base<T>::`


For class templates with base classes that depend on template parameters, 
using a name `x` by itself is **not** always equivalent to `this->x`, 
even though a member `x` is inherited. For example:
```c++
template <typename T>
class Base
{
public:
    void bar() {}
};

template <typename T>
class Derived : Base<T>
{
public:
    void foo()
    {
        // calls external bar() or error
        bar(); 
    }
};
```
In this example, for resolving the symbol `bar` inside `Derived<T>::foo`,
`bar` defined in `Base<T>` is **never** considered. 
Therefore, either you have an error, or another `bar` (such as a global one) is called. 


We discuss this issue in Section 13.4 in detail. 
For the moment, as a rule of thumb, we recommend that you always qualify any symbol
that is declared in a base that is somehow dependent on a template parameter 
with `this->` or `Base<T>::`. 


#### 📌 5.4 Templates for Raw Arrays and String Literals


When passing raw arrays or string literals to templates, some care has to be taken.
First, if the template parameters are declared as references (including universal references), 
the arguments **don't** decay. 
That is, a passed argument of `hello` has type `char const [6]`. 
This can become a problem if raw arrays or string arguments of different length 
are passed because the types differ. 
Only when passing the argument by value, the types decay, 
so that string literals are converted to type `char const *`. 
This is discussed in detail in Chapter 7.


Note that you can also provide templates that specifically deal with raw arrays or string literals. 
For example:
```c++
template <typename T, std::size_t N, std::size_t M>
bool less(T (& a)[N], T (& b)[M])
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    
    for (std::size_t i = 0; i < N && i < M; ++i)
    {
        if (a[i] < b[i])
        {
            return true;
        }
        
        if (b[i] < a[i])
        {
            return false;
        }
    }
    
    return N < M;
}

// bool less(T (&)[N], T (&)[M]) [with T = int; long unsigned int N = 3; long unsigned int M = 5]
// 1
int x[] {1, 2, 3};
int y[] {1, 2, 3, 4, 5};
std::cout << less(x, y) << '\n';

// bool less(T (&)[N], T (&)[M]) [with T = const char; long unsigned int N = 3; long unsigned int M = 4]
// 1
std::cout << less("ab", "abc") << '\n';
```
If you only want to provide a function template for string literals 
(and other char arrays), you can do this as follows:
```c++
template <std::size_t N, std::size_t M>
bool less(const char (& a)[N], const char (& b)[M])
{
    for (std::size_t i = 0; i < N && i < M; ++i)
    {
        if (a[i] < b[i])
        {
            return true;
        }
        
        if (b[i] < a[i])
        {
            return false;
        }
    }
    
    return N < M;
}
```
Note that you can and sometimes have to overload or partially specialize for arrays of unknown bounds. 
The following program illustrates all possible overloads for arrays:
```c++
// primary template
template <typename T>
struct MyClass;

// partial specialization for arrays of known bounds
template <typename T, std::size_t SZ>
struct MyClass<T[SZ]> 
{
    static void print()
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }
};

// partial specialization for references to arrays of known bounds
template <typename T, std::size_t SZ>
struct MyClass<T(&)[SZ]> 
{
    static void print()
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }
};

// partial specialization for arrays of unknown bounds
template <typename T>
struct MyClass<T []> 
{
    static void print()
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }
};

// partial specialization for references to arrays of unknown bounds
template <typename T>
struct MyClass<T (&)[]> 
{
    static void print()
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }
};

// partial specialization for pointers
template <typename T>
struct MyClass<T *> 
{
    static void print()
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }
};
```
Here, the class template `MyClass` is specialized for various types: 
arrays of known and unknown bound, 
references to arrays of known and unknown bounds,
and pointers. 
Each case is different and can occur when using arrays:
```c++
template <typename T1, typename T2, typename T3>
void foo(int a1[7],       // pointers by language rules
         int a2[],        // pointers by language rules
         int (& a3)[42],  // reference to array of known bound
         int (& x0)[],    // reference to array of unknown bound
         T1 x1,           // passing by value decays
         T2 & x2,         // passing by reference
         T3 && x3)        // passing by reference
{
    MyClass<decltype(a1)>::print();  // uses MyClass<T *>
    MyClass<decltype(a2)>::print();  // uses MyClass<T *>
    MyClass<decltype(a3)>::print();  // uses MyClass<T (&)[SZ]>
    MyClass<decltype(x0)>::print();  // uses MyClass<T (&)[]>
    MyClass<decltype(x1)>::print();  // uses MyClass<T *>
    MyClass<decltype(x2)>::print();  // uses MyClass<T (&)[]>
    MyClass<decltype(x3)>::print();  // uses MyClass<T (&)[]>
}

// static void MyClass<T [SZ]>::print() [with T = int; long unsigned int SZ = 42]
int a[42];
MyClass<decltype(a)>::print();

// forward declare array
// static void MyClass<T []>::print() [with T = int]
extern int x[];                 
MyClass<decltype(x)>::print();

// static void MyClass<T *>::print() [with T = int]
// static void MyClass<T *>::print() [with T = int]
// static void MyClass<T (&)[SZ]>::print() [with T = int; long unsigned int SZ = 42]
// static void MyClass<T (&)[]>::print() [with T = int]
// static void MyClass<T *>::print() [with T = int]
// static void MyClass<T (&)[]>::print() [with T = int]
// static void MyClass<T (&)[]>::print() [with T = int]
foo(a, a, a, x, x, x, x);
```
Note that a _function parameter_ declared as an array (with or without length) 
by language rules really has a pointer type. 
Note also that templates for arrays of unknown bounds can be used for an incomplete type such as
```c++
extern int i[];
```
And when this is passed by reference, it becomes a `int (&)[]`,
which can also be used as a template parameter.
Parameters of type `X (&)[]` for some arbitrary type `X` 
have become valid only since C++17, through the resolution of Core issue 393. 
However, many compilers accepted such parameters in earlier versions of the language. 

See Section 19.3 for another example using the different array types in generic code. 


#### 📌 5.5 Member Templates


Class members can also be templates. 
This is possible for both nested classes and member functions. 
The application and advantage of this ability can again be demonstrated with the `Stack` class template. 
Normally you can assign stacks to each other only when they have the same type, 
which implies that the elements have the same type. 
However, you can't assign a stack with elements of any other type,
even if there is an implicit type conversion for the element types defined:
```c++
Stack<int> intStack1;     // stack for ints
Stack<int> intStack2;     // stack for ints
Stack<float> floatStack;  // stack for floats
intStack1 = intStack2;    // OK: stacks have same type
floatStack = intStack1;   // ERROR: stacks have different types
```
The default assignment operator requires that both sides of the assignment operator
have the same type, which is not the case if stacks have different element types. 


By defining an assignment operator as a template, however, 
you can enable the assignment of stacks with elements 
for which an appropriate type conversion is defined.
```c++
template <typename T>
class Stack
{
public:
    void push(T const &);
    
    void pop();
    
    T const & top() const;
    
    bool empty() const
    {
        return elems.empty();
    }

    // assign stack of elements of type T2
    template <typename T2>
    Stack & operator=(Stack<T2> const &);

private:
    std::deque<T> elems; // elements
};

template <typename T>
template <typename T2>
Stack<T> & Stack<T>::operator=(Stack<T2> const & op2)
{
    Stack<T2> tmp(op2);
    elems.clear();
    
    while (!tmp.empty())
    {
        elems.push_front(tmp.top());
        tmp.pop();
    }
    
    return *this;
}
```
First let's look at the syntax to define a _member template_. 
Inside the template with template parameter `T`, 
an inner template with template parameter `T2` is defined:
```c++
template <typename T>
template <typename T2>
...
```
If you instantiate a class template for two different argument types,
you get two different class types. 
Inside the member function, 
the assigned stack `op2` has a different type. 
So you are restricted to using the public interface. 
It follows that the only way to access the elements is by calling `top`. 
However, each element has to become a top element, then. 
Thus, a copy of `op2` must first be made, 
so that the elements are taken from that copy by calling `pop`. 
Because `top` returns the last element pushed onto the stack, 
we might prefer to use a container that supports 
the insertion of elements at the other end of the collection. 
For this reason, we use a `std::deque`, 
which provides `push_front` to put an element on the other side of the collection.


To get access to all the members of `op2`, 
you can declare that all other stack instances are friends:
```c++
template <typename T>
class Stack
{
public:
    // to get access to private members of Stack<T2> for any type
    template <typename> 
    friend class Stack;
    
    void push(T const &);
    void pop();
    T const & top() const; 
    
    bool empty() const
    {
        return elems.empty();
    }

    // assign stack of elements of type T2
    template <typename T2>
    Stack & operator=(Stack<T2> const &);
    
private:
    std::deque<T> elems; // elements
};

template <typename T>
template <typename T2>
Stack<T> & Stack<T>::operator=(Stack<T2> const & op2)
{
    elems.clear();
    elems.insert(elems.begin(), op2.elems.cbegin(), op2.elems.cend());
    return *this;
}
```
Whatever your implementation is, having this member template, 
you can now assign a stack of `int`s to a stack of `float`s:
```c++
Stack<int> intStack;      // stack for ints
Stack<float> floatStack;  // stack for floats
floatStack = intStack;    // OK: stacks have different types,
                          // but int converts to float
```
Of course, this assignment does not change the type of the stack and its elements.
After the assignment, the elements of the `floatStack` are still `float`s and
therefore `top` still returns a `float`.


It may appear that this function would disable type checking 
such that you could assign a stack with elements of any type, but this is not the case. 
The necessary type checking occurs 
when the element of the (copy of the) source stack is moved to the destination stack:
```c++
elems.push_front(tmp.top());
```
If, for example, a stack of `std::string`s gets assigned to a stack of `float`s, 
the compilation of this line results in an error message stating that 
the `std::string` returned by `tmp.top()` cannot be passed as an argument to `elems.push_front`. 
The message varies depending on the compiler, but this is the gist of what is meant. 
```c++
Stack<std::string> stringStack;  // stack of std::strings
Stack<float> floatStack;         // stack of floats
floatStack = stringStack;        // ERROR: std::string doesn't convert to float
```
Again, you could change the implementation to parameterize the internal container type:
```c++
template <typename T, typename Cont = std::deque<T>>
class Stack
{
public:
    template <typename, typename>
    friend class Stack;
    
    void push(T const &);
    void pop();
    T const & top() const; 
    
    bool empty() const
    {
        return elems.empty();
    }
    
    template <typename T2, typename Cont2>
    Stack & operator=(Stack<T2, Cont2> const &);

private:
    Cont elems; // elements
};

template <typename T, typename Cont>
template <typename T2, typename Cont2>
Stack<T, Cont> & Stack<T, Cont>::operator=(Stack<T2, Cont2> const & op2)
{
    elems.clear();
    elems.insert(elems.begin(), op2.elems.cbegin(), op2.elems.cend());
    return *this;
}
```
Remember, for class templates, 
only those member functions that are called are instantiated. 
Thus, if you avoid assigning a stack with elements of a different type,
you could even use a `std::vector` as an internal container:
```c++
// stack for ints using a vector as an internal container
Stack<int, std::vector<int>> vStack;
vStack.push(42); 
vStack.push(7);
std::cout << vStack.top() << '\n';
```
Because the assignment operator template isn't necessary,
**no** error message of a missing member function `push_front` occurs
and the program is fine.

#### [Limitations on Member Function Templates](https://en.cppreference.com/w/cpp/language/member_template#Member_function_templates)

- Destructors and copy constructors can **not** be templates. 
  - If a template constructor is declared which could be instantiated 
    with the type signature of a copy constructor, 
    the implicitly-declared copy constructor is used instead.
- A member function template can **not** be virtual.  
  - A member function template in a derived class can **not** 
    override a virtual member function from the base class.
- A non-template member function and a template member function with the same name may be declared. 
  - In case of conflict (when some template specialization matches the non-template function signature exactly), 
    the use of that name and type refers to the _non-template member_ 
    unless an explicit template argument list is supplied. 
- An out-of-class definition of a member function template 
  must be equivalent to the declaration inside the class, 
  otherwise it is considered to be an _overload_.
- A _user-defined conversion function_ can be a template.
  - During overload resolution, 
    _specializations of conversion function templates_ 
    are **not** found by name lookup. 
    Instead, all visible conversion function templates are considered, 
    and every specialization produced by template argument deduction 
    (which has special rules for conversion function templates) 
    is used as if found by name lookup.
  - Using-declarations in derived classes can **not** refer to 
    _specializations of template conversion functions_ from base classes.
  - A _user-defined conversion function template_ can **not** have a deduced return type.

#### Specialization of Member Function Templates

Member function templates can also be partially or fully specialized. 
For example, for the following class:
```c++
class BoolString
{
public:
    explicit BoolString(std::string s) 
            : value(std::move(s)) 
    {
        
    }

    template <typename T = std::string>
    T get() const
    {
        return value;
    }

private:
    std::string value;
};
```
you can provide a full specialization for the member function template as follows:
```c++
// full specialization for BoolString::getValue<>() for bool
template <>
inline bool BoolString::get<bool>() const 
{
    return value == "true" || value == "1" || value == "on";
}
```
Note that you don't need and also **can't** declare the specializations; you only define them. 
Because it is a full specialization, and it is in a header file, 
you have to declare it with `inline` to avoid errors 
if the definition is included by different translation units. 


You can use class and the full specialization as follows:
```c++
std::cout << std::boolalpha;
BoolString s1("hello");
std::cout << s1.get() << '\n';        // hello
std::cout << s1.get<bool>() << '\n';  // false
BoolString s2("on");
std::cout << s2.get<bool>() << '\n';  // true
```

#### Special Member Function Templates

Special member functions (default constructor, destructor, copy/move constructor and assignment operator)
can also be member function templates.
Member templates **don't** count as _the_ special member functions that copy or move objects.
These template versions also **don't** count as user-defined versions. 
In this example, for assignments of stacks of the same type, 
the default assignment operator is still called. 


This effect can be good and bad:
- It can happen that a template constructor or assignment operator 
  is a better match than the predefined copy/move constructor or assignment operator, 
  although a template version is provided for initialization of other types only. 
  See Section 6.2 for details.
- It is not easy to "templify" a copy/move constructor, 
  for example, to be able to constrain its existence. 
  See Section 6.4 for details.

#### The `.template` Construct for Dependent Template Names

Sometimes, it is necessary to explicitly qualify template arguments 
when _calling_ a member template. 
This is _dependent template name_. _
In that case, you have to use the `template` keyword 
to ensure that the less-than token `<` 
denotes the beginning of the template argument list
instead of a call of `operator<`. 
Consider the following example using the standard `std::bitset` type:
```c++
template <unsigned long N>
void printBitset(std::bitset<N> const & bs)
{
    std::cout << bs.template to_string<char, std::char_traits<char>, std::allocator<char>>();
}
```
For the bitset `bs` we call the member function template `to_string`,
while explicitly specifying the string type details. 
Without that extra use of `.template`,
the compiler does not know that the less-than token `<` that follows 
is not really less-than but the beginning of a template argument list. 
Note that this is a problem only if the construct before the period depends on a template parameter. 
In our example, the parameter bs depends on the template parameter `N`. 


The `.template` notation (and similar notations such as `->template` and `::template`) 
should be used only inside templates and only if they follow something that depends on a template parameter. 
See Section 13.3 for details.

#### Generic Lambdas and Member Templates

Note that generic lambdas, introduced with C++14, 
are shortcuts for member templates. 
A simple lambda computing the "sum" of two arguments of arbitrary types:
```c++
[] (auto x, auto y)
{
    return x + y;
}
```
is a shortcut for a default-constructed object of the following class:
```c++
class SomeCompilerSpecificName 
{
public:
    // constructor only callable by compiler
    SomeCompilerSpecificName(); 
    
    template <typename T1, typename T2>
    auto operator() (T1 x, T2 y) const 
    {
        return x + y;
    }
};
```
See Section 15.10 for details.


#### 📌 5.6 [Variable Templates](https://en.cppreference.com/w/cpp/language/variable_template)


Since C++14, variables also can be parameterized by a specific type. 
Such a thing is called a _variable template_.


Yes, we have very similar terms for very different things: 
- A _variable template_ is a variable that is a template 
  (variable is a noun here). 
- A _variadic template_ is a template for a variadic number of template parameters 
  (variadic is an adjective here).


For example, you can use the following code to define the value of `pi` 
while still not defining the type of the value:
```c++
template <typename T>
constexpr T pi {3.141592653589793238462643383279502884L};
```
Note that, as for all templates, 
this declaration may **not** occur inside functions or block scope.


To use a variable template, you have to specify its type. 
For example, the following code uses different variables of the scope where `pi` is declared:
```c++
std::cout << pi<long double> << '\n';
std::cout << pi<double> << '\n';
std::cout << pi<float> << '\n';
```
You can also declare variable templates that are used in different translation units:
```c++
// "header.hpp"
// zero-initialized value
template <typename T> 
T val {};

// "1.cpp"
#include "header.hpp"

int main()
{
    val<long> = 42;
    print();
}

// "2.cpp"
#include "header.hpp"

void print()
{
    // OK: prints 42
    std::cout << val<long> << '\n';
}
```
Variable templates can also have default template arguments:
```c++
template <typename T = long double>
constexpr T pi {3.141592653589793238462643383279502884L};
```
You can use the default or any other type:
```c++
std::cout << pi<> << '\n';       // outputs a long double
std::cout << pi<float> << '\n';  // outputs a float
```
However, note that you _always_ have to specify the angle brackets. 
Just using `pi` is an error:
```c++
std::cout << pi << '\n';         // ERROR
```
Variable templates can also be parameterized by nontype parameters, 
which also may be used to parameterize the initializer. 
For example:
```c++
// array with N elements, zero-initialized
template <int N>
std::array<int, N> arr {}; 

// type of dval depends on passed value
template <auto N>
constexpr decltype(N) dval = N;

// N has value 'c' of type char
std::cout << dval<'c'> << '\n'; 

// sets first element of global arr
arr<10>[0] = 42; 

// uses values set in arr
for (std::size_t i = 0; i < arr<10>.size(); ++i) 
{ 
    std::cout << arr<10>[i] << '\n';
}
```
Again, note that even when the initialization of and iteration over `arr` happens in different translation units, 
the same variable `std::array<int, 10>` `arr` of global scope is still used.

#### Variable Templates for Data Members

A useful application of variable templates 
is to define variables that represent members of class templates.
For example, if a class template is defined as follows:
```c++
template <typename T>
class MyClass 
{
public:
    static constexpr int max = 1000;
};
```
which allows you to define different values for different specializations of `MyClass`, 
then you can define
```c++
template <typename T>
int myMax = MyClass<T>::max;
```
so that application programmers can just write
```c++
auto i = myMax<std::string>;
```
instead of
```c++
auto i = MyClass<std::string>::max;
```
This means, for a standard class such as
```c++
namespace std
{

template <typename T>
class numeric_limits
{
public:
    ...
    static constexpr bool is_signed = false;
    ...
};

}  // namespace std
```
you can define
```c++
template <typename T>
constexpr bool isSigned = std::numeric_limits<T>::is_signed;

// instead of:
isSigned<char>
std::numeric_limits<char>::is_signed
```

#### Type Traits Suffix `_v`

Since C++17, the standard library uses the technique of variable templates 
to define shortcuts for all type traits in the standard library that yield a (Boolean) value.
```c++
/// <type_traits>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04) 9.4.0
namespace std
{

template <typename T>
inline constexpr bool is_const_v = is_const<T>::value;

}  // namespace std

// instead of:
std::is_const_v<T>
std::is_const<T>::value
```


#### 📌 5.7 Template Template Parameters


It can be useful to allow a template parameter itself to be a class template. 
Again, our stack class template can be used as an example.


To use a different internal container for stacks, 
the application programmer has to specify the element type twice. 
Thus, to specify the type of the internal container,
you have to pass the type of the container _and_ the type of its elements again:
```c++
Stack<int, std::vector<int>> vStack;  // integer stack that uses a vector
```
With _template template parameters_, you can declare the `Stack` class template
by specifying the type of the container **without** respecifying the type of its elements:
```c++
Stack<int, std::vector> vStack;       // integer stack that uses a vector
```
To do this, you must specify the second template parameter as a _template template parameter_.


In principle, this looks as follows
```c++
template <typename T,
          template <typename Elem> class Cont = std::deque>
class Stack
{
public:
    void push(T const &);
    void pop();
    T const & top() const;
    
    bool empty() const
    {
        return elems.empty();
    }

private:
    Cont<T> elems;
};
```
The difference is that the second template parameter 
is declared as being a class template:
```c++
template <typename Elem> class Cont
```
The default value has changed from `std::deque<T>` to `std::deque`. 
This parameter has to be a class template, 
which is instantiated for the type `T`:
```c++
Cont<T> elems;
```
This use of the first template parameter
for the instantiation of the second template parameter 
is particular to this example.
In general, you can instantiate a template template parameter 
with any type inside a class template.


Before C++11, `Cont` could only be substituted by the name of a _class template_. 
Since C++11, we can also substitute `Cont` with the name of an _alias template_. 


As usual, instead of `typename`, you could use the keyword `class` for template parameters.
But it wasn't until C++17 that a corresponding change was made
to permit the use of the keyword `typename` instead of `class` 
to declare a template template parameter:
```c++
// OK
template <typename T,
template <class Elem> class Cont = std::deque>
class Stack { ... };

// ERROR before C++17
template <typename T,
          template <typename Elem> class Cont = std::deque>
class Stack { ... };
```
Those two variants mean exactly the same thing: 
Using `class` instead of `typename` does not prevent us 
from specifying an alias template as the argument corresponding to the `Cont` parameter.


Because the template parameter of the template template parameter is not used,
it is customary to omit its name (unless it provides useful documentation):
```c++
template <typename T, 
          template <typename> class Cont = std::deque>
class Stack { ... };
```
Member functions must be modified accordingly. 
```c++
template <typename T, template <typename> class Cont>
void Stack<T, Cont>::push (T const & elem)
{
    elems.push_back(elem);
}
```
Note that while template template parameters are placeholders for class or alias templates, 
there is no corresponding placeholder for function or variable templates.

#### Template Template Argument Matching

If you try to use the new version of `Stack`, 
you may get an error message saying that the default value `std::deque` 
is not compatible with the template template parameter `Cont`. 


A template template argument had to be a template whose parameters 
exactly match those of the substituted template template parameter. 
with some exceptions related to variadic template parameters (see Section 12.3). 


In C++17, default arguments are considered when matching the template parameters.
However, prior to C++17, default template arguments of template template arguments were **not** considered.  
That is, a match can **not** be achieved when not explicitly re-specifying those default arguments.
```c++
template <template <typename T, typename U> Cont>
class MyClass { ... };

template <typename T, typename U = int>
class C { ... };

MyClass<C<T1, T2>> mc;  // good
MyClass<C<T1>> mc2;     // ERROR before C++17 (default argument int was ignored during the matching process)
```
The pre-C++17 problem in this example is that 
the `std::deque` template has more than one parameter.  
`std::deque`'s second parameter `Allocator` has a default value.  
But prior to C++17, this was **not** considered when matching `std::deque` to the `Cont` parameter. 


We can rewrite the class declaration so that 
the `Cont` parameter expects containers with two template parameters:
```c++
template <typename T,
          template <typename Elem, 
                    typename Alloc = std::allocator<Elem>>
          class Cont = std::deque>
class Stack
{
private:
    Cont<T> elems;
    ...
};
```
Again, we could omit `Alloc` because it is not used.


The final version of our `Stack` template 
(including member templates for assignments of stacks of different element types) 
now looks as follows:
```c++
template <typename T,
          template <typename Elem,
                    typename = std::allocator<Elem>>
          class Cont = std::deque>
class Stack
{
public:
    template <typename, template <typename, typename> class>
    friend class Stack;
    
    void push(T const &);
    void pop();
    T const & top() const;
    
    bool empty() const
    {
        return elems.empty();
    }
    
    template <typename T2,
              template <typename Elem2,
                        typename = std::allocator<Elem2>> 
              class Cont2>
    Stack<T, Cont> & operator=(Stack<T2, Cont2> const &);
    
private:
    Cont<T> elems; // elements
};

template <typename T, template <typename, typename> class Cont>
void Stack<T, Cont>::push(T const & elem)
{
    elems.push_back(elem);
}

template <typename T, template <typename, typename> class Cont>
void Stack<T, Cont>::pop()
{
    assert(!elems.empty());
    elems.pop_back();
}

template <typename T, template <typename, typename> class Cont>
T const & Stack<T, Cont>::top() const
{
    assert(!elems.empty());
    return elems.back();
}

template <typename T, template <typename, typename> class Cont>
template <typename T2, template <typename, typename> class Cont2>
Stack<T, Cont> & Stack<T, Cont>::operator=(Stack<T2, Cont2> const & op2)
{
    elems.clear();
    elems.insert(elems.begin(), op2.elems.cbegin(), op2.elems.cend());
    return *this;
}
```
Still, **not** _all_ standard container templates can be used for `Cont` parameter. 
For example, `std::array` will **not** work because it includes a nontype template parameter for the array length 
that has no match in our template template parameter declaration.


For further discussion and examples of template template parameters, 
see Section 12.2, Section 12.3, and Section 19.2.


#### 📌 5.8 Summary


- To access a nested type name that depends on a template parameter,
  you have to qualify the name with a leading `typename`.
- To access members of bases classes that depend on template parameters, 
  you have to qualify the access by `this->` or their class name.
- Nested classes and member functions can also be templates. 
  One application is the ability to implement generic operations with internal type conversions. 
  Limitations apply to member function templates. 
- Template versions of constructors or assignment operators 
  **don't** replace predefined constructors or assignment operators.
- By using braced initialization or explicitly calling a default constructor, 
  you can ensure that variables and members of templates are initialized with a default value
  even if they are instantiated with a built-in type.
- You can provide specific templates for raw arrays, 
  which can also be applicable to string literals.
- When passing raw arrays or string literals, 
  arguments decay (perform an array-to-pointer conversion) during argument deduction 
  if and only if the parameter is **not** a reference.
- You can define variable templates. 
- You can also use class templates and alias templates as template parameters, 
  as template template parameters. 
- Template template arguments must usually match their parameters exactly. 






### 🎯 Chapter 6 Move Semantics and `std::enable_if`


#### 📌 6.1 Perfect Forwarding


Suppose you want to write generic code that forwards the basic property of passed arguments:
- Modifyable objects should be forwarded so that they still can be modified.
- Constant objects should be forwarded as read-only objects.
- Movable objects should be forwarded as movable objects.


To achieve this functionality without templates, we have to program all three cases.
For example, to forward a call of `f` to a corresponding function `g`:
```c++
class X { ... };

void g(X &) {}
void g(const X &) {}
void g(X &&) {}

void f(X & val)
{
    g(val);
}

void f(const X & val)
{
    g(val);
}

void f(X && val)
{
    // Function parameters are lvalues. 
    // Call std::move to cast to rvalue. 
    g(std::move(val));
}
```
Note that the code for movable objects (via an rvalue reference) 
differs from the other code: 
It needs a `std::move` because according to language rules, 
move semantics is **not** passed through.  
For the function parameter whose _type_ is rvalue reference, its _value category_ is lvalue. 


The fact that move semantics is **not** automatically passed through is intentional and important.
If it weren't, we would lose the value of a movable object the first time we use it in a function.


Although `val` in the third `f` is declared as rvalue reference, 
its value category when used as expression is a non-constant lvalue (see Appendix B) 
and behaves as `val` in the first `f`. 
Without `std::move`, `g(X &)` for non-constant lvalues instead of `g(X &&)` would be called.


C++11 introduces special rules for _perfect forwarding_ parameters.
If we want to combine all three cases in generic code.
```c++
template <typename T>
void f(T && val)
{
    g(std::forward<T>(val));
}
```
Note that `std::move` has no template parameter and "triggers" move semantics for the passed argument, 
while `std::forward` "forwards" potential move semantic depending on a passed template argument.
```c++
/// <type_traits>
/// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0

/// remove_reference
template <typename _Tp>
struct remove_reference
{
    typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &>
{
    typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &&>
{
    typedef _Tp type;
};

/// <bits/move.h>
/// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0

/**
 *  @brief  Convert a value to an rvalue.
 *  @param  __t  A thing of arbitrary type.
 *  @return The parameter cast to an rvalue-reference to allow moving it.
 */
template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&
move(_Tp && __t) noexcept
{
    return static_cast<typename std::remove_reference<_Tp>::type &&>(__t);
}

/**
 *  @brief  Forward an lvalue.
 *  @return The parameter cast to the specified type.
 *
 *  This function is used to implement "perfect forwarding".
 */
template <typename _Tp>
constexpr _Tp &&
forward(typename std::remove_reference<_Tp>::type & __t) noexcept
{
    return static_cast<_Tp &&>(__t);
}

/**
 *  @brief  Forward an rvalue.
 *  @return The parameter cast to the specified type.
 *
 *  This function is used to implement "perfect forwarding".
 */
template <typename _Tp>
constexpr _Tp &&
forward(typename std::remove_reference<_Tp>::type && __t) noexcept
{
    static_assert(!std::is_lvalue_reference<_Tp>::value,
                  "template argument substituting _Tp is an lvalue reference type");
    return static_cast<_Tp &&>(__t);
}
```
Don't assume that ` &&` for a template parameter `T` behaves as `X &&` for a specific type `X`. 
Different rules apply! 
However, syntactically they look identical:
- `X &&` for a specific type `X` declares a parameter to be an rvalue reference. 
  It can only be bound to a movable object 
  (a prvalue, such as a temporary object, and an xvalue, such as an object passed with `std::move`; 
  see Appendix B for details). 
  It is always mutable and you can always "steal" its value. 
  - A type like `X const &&` is valid but provides **no** common semantics in practice
    because "stealing" the internal representation of a movable object requires
    modifying that object. 
    It might be used, though, to force passing only temporaries or objects marked with `std::move` 
    without being able to modify them. 
- `T &&` for a template parameter `T` declares a _forwarding reference_ (also called _universal reference_). 
  - The term _universal reference_ was coined by Scott Meyers prior to C++17 
    as a common term that could result in either an "lvalue reference" or an "rvalue reference". 
    The C++17 standard introduced the term _forwarding reference_, 
    because the major reason to use such a reference is to forward objects. 
    However, note that it does **not** automatically forward. 
    The term does not describe what it is but what it is typically used for. 
  - It can be bound to a mutable, immutable (i.e., `const`), or movable object. 
    Inside the function definition, the parameter may be mutable, immutable, 
    or refer to a value you can "steal" the internals from. 


Note that `T` must really be the name of a template parameter. 
Depending on a template parameter is not sufficient. 
For a template parameter `T`, a declaration such as `typename T::iterator &&` is just an rvalue reference, 
**not** a forwarding reference.


Of course, perfect forwarding can also be used with variadic templates. 
See Section 15.6 on page 280 for details of perfect forwarding. 


#### 📌 6.2 Special Member Function Templates


Member function templates can also be used as special member functions, 
including as a constructor, which, however, might lead to surprising behavior.


Consider the following example:
```c++
class Person
{
public:
    explicit Person(std::string n) : name(std::move(n))
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }

    explicit Person(std::string && n) : name(std::move(n))
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }
    
    Person(const Person & p) : name(p.name)
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }

    Person(Person && p) noexcept : name(std::move(p.name))
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }

private:
    std::string name;
};

std::string s = "sname";
Person p1(s);              // Person::Person(const string &)      
Person p2("tmp");          // Person::Person(std::string &&)
Person p3(p1);             // Person::Person(const Person &)
Person p4(std::move(p1));  // Person::Person(Person &&)
```
Here, we have a class `Person` with a `std::string` member `name` 
for which we provide initializing constructors. 
To support move semantics, we overload the constructor taking a `std::string`. 


Now let's replace the two `std::string` constructors 
with one generic constructor perfect forwarding the passed argument to the member `name`:
```c++
class Person
{
public:
    template <typename String>
    explicit Person(String && n) : name(std::forward<String>(n))
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }
    
    Person(const Person & p) : name(p.name)
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }

    Person(Person && p) noexcept : name(std::move(p.name))
    {
        std::cout << __PRETTY_FUNCTION__ << '\n';
    }

private:
    std::string name;
};

std::string s = "sname";
Person p1(s);      // Person::Person(String &&) [with String = std::string &]
Person p2("tmp");  // Person::Person(String &&) [with String = const char (&)[4]]
```
Note how the construction of `p2` does **not** create a temporary `std::string` in this case: 
The parameter `String` is deduced to be of type `const char (&)[4]`. 
Applying `std::forward<String>` to the pointer parameter of the constructor 
has not much of an effect, 
and the `name` member is thus constructed from a null-terminated C character array. 
But when we attempt to call the copy constructor, we get an error: 
```c++
Person p3(p1);             // ERROR: Can not assign Person p1 to std::string name
Person p4(std::move(p1));  // Person::Person(Person &&)
```
Note that also copying a constant `Person` works fine:
```c++
const Person p2c("ctmp");
Person p3c(p2c);           // Person::Person(const Person&)
```
The problem is that, according to the overload resolution rules of C++ (see Section 16.2), 
for a non-`const` lvalue `Person` `p`, the member template
```c++
template <typename String>
explicit Person(String && n) : name(std::forward<String>(n))
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}
```
is a better match than the (usually predefined) copy constructor:
```c++
Person (Person const & p)
```
because the forwarding reference offers a _perfect match_, 
while the predefined copy constructor requires one 
_implicit conversion_ from non-`const` to `const`. 


You might think about solving this by also providing a non-`const` copy constructor:
```c++
Person (Person & p)
```
However, that is only a partial solution, because for objects of a _derived class_, 
the forwarding reference template is still a better match. 
What you really want is to disable the member template for the case 
that the passed argument is a `Person` or an expression that can be converted to a `Person`. 
This can be done by using `std::enable_if`, which is introduced in the next section. 
```c++
template <typename String,
          std::enable_if_t<std::is_convertible_v<String, 
                                                 std::string>, 
                           bool> = true>
explicit Person::Person(String && n) : name(std::forward<String>(n))
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}
```


#### 📌 6.3 Disable Templates with `std::enable_if`


Since C++11, the C++ standard library provides a helper template `std::enable_if` 
to ignore function templates under certain compile-time conditions. 
```c++
// <type_traits>
// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0

/// Define a member typedef @c type only if a boolean constant is true.
template <bool, typename _Tp = void>
struct enable_if
{
};

// Partial specialization for true.
template <typename _Tp>
struct enable_if<true, _Tp>
{
    typedef _Tp type;
};

/// Alias template for enable_if
template <bool _Cond, typename _Tp = void>
using enable_if_t = typename enable_if<_Cond, _Tp>::type;
```
`std::enable_if` is a type trait that evaluates
a given compile-time constant expression (`constexpr`)
passed as its first template argument, 
and behaves as follows:
- If the expression yields `true`, its type member `type` yields a type: 
  - The type is `void` if no second template argument is passed;
  - Otherwise, the type is the second template argument type. 
- If the expression yields `false`, the member type is not defined. 
  Due to a template feature called SFINAE (Substitution Failure Is Not An Error, see Section 8.4), 
  this has the effect that the function template with the `std::enable_if` expression is ignored.


For example, if a function template `foo` is defined as follows:
```c++
template <typename T>
std::enable_if_t<(4 < sizeof(T))>
foo() {}
```
this definition of `foo` is _ignored_ if `4 < sizeof(T)` yields `false`. 
If `4 < sizeof(T)` yields `true`, the function template instantiated to
```c++
void foo() {}
```
If a second argument is passed to `std::enable_if`:
```c++
template <typename T>
std::enable_if_t<(4 < sizeof(T)), T>
foo() 
{
    return T {};
}
```
Having the `std::enable_if` expression in the middle of a declaration is pretty clumsy. 
For this reason, the common way to use `std::enable_if` 
is to use `std::enable_if` as an _anonymous template (type of non-type) argument with a default value_.
If that is still too clumsy, and you want to make the requirement/constraint more explicit,
you can define your own name for it using an _alias template_: 
```c++
// use as anonymous type argument
template <typename T, 
          typename = std::enable_if_t<(4 < sizeof(T))>>
void foo() {}

// use as anonymous non-type argument
template <typename T, 
          std::enable_if_t<(4 < sizeof(T)), bool> = true>
void bar() {}

// use as alias template
template <typename T>
using EnableIfSizeGreaterThan4 = std::enable_if_t<(4 < sizeof(T))>;

template <typename T,
          typename = EnableIfSizeGreaterThan4<T>>
void fun() {}
```
which expands to the following if `4 < sizeof(T)`: 
```c++
// use as anonymous type argument
template <typename T, typename = void>
void foo() {}

// use as anonymous non-type argument
template <typename T, bool = true>
void bar() {}

// use as alias template
template <typename T, typename = void>
void fun() {}
```
See Section 20.3 for a discussion of how `std::enable_if` is implemented.


#### 📌 6.4 Using `std::enable_if`


```c++
template <typename String,
          typename = std::enable_if_t<std::is_convertible_v<String, std::string>>>
explicit Person::Person(String && n);
```
If type `String` is convertible to type `std::string`, the whole declaration expands to
```c++
template <typename String, typename = void>
explicit Person::Person(String && n);
```
If type `String` is not convertible to type `std::string`, the whole function template is ignored.


If you wonder why we don't instead check whether `String` is "not convertible to `Person`", beware:
We are defining a function that might allow us to convert a `std::string` to a `Person`.
So the constructor has to know whether it is enabled,
which depends on whether it is convertible, which depends on whether it is enabled, and so on.
**Never** use `std::enable_if` in places that impact the condition used by `std::enable_if`.
This is a logical error that compilers do not necessarily detect. 


Again, we can define our own name for the constraint by using an alias template:
```c++
template <typename T>
using EnableIfString = std::enable_if_t<std::is_convertible_v<T, std::string>>;

template <typename String, typename = EnableIfString<String>>
explicit Person::Person(String && n);
```
Note also that there is an alternative to using `std::is_convertible`
because it requires that the types are _implicitly convertible_. 
By using `std::is_constructible`, we also allow explicit conversions to be used for the initialization. 
However, the order of the arguments is the opposite is this case:
```c++
template <typename T>
using EnableIfString = std::enable_if_t<std::is_constructible_v<std::string, T>>;
```
See Section D.3 details about `std::is_constructible` and `std::is_convertible`. 
See Section D.6 for details and examples to apply `std::enable_if` on variadic templates. 

#### Disabling Special Member Functions

Note that normally we **can't** use `std::enable_if` to disable 
the predefined copy/move constructors and/or assignment operators. 


Recall Section 5.5 for one of the limitations on member function templates: 
- _Destructors_ and _copy constructors_ can **not** be templates. 
  - If a template constructor is declared which could be instantiated 
    with the type signature of a copy constructor, 
    the implicitly-declared copy constructor is used instead. 


Thus, with this declaration:
```c++
class C 
{
public:
    template <typename T>
    C (T const &) { ... }
    
    ... 
};
```
the predefined copy constructor is still used, when a copy of a `C` is requested:
```c++
C x;
C y {x};  // still uses the predefined copy constructor (not the member template)
```
(There is really no way to use the member template 
because there is no way to specify or deduce its template parameter `T`.)


Deleting the predefined copy constructor is **no** solution, 
because then the trial to copy a `C` results in an error. 


There is a tricky solution, though: 
We can declare a copy constructor for `const volatile` arguments and define it `= delete;`. 
Doing so prevents another copy constructor from being implicitly declared. 
With that in place, we can define a constructor template 
that will be preferred over the (deleted) copy constructor for non-volatile types:
```c++
class C
{
public:
    // user-define the predefined copy constructor as deleted
    // (with conversion to volatile to enable better matches)
    C(C const volatile &) = delete;
    
    // implement copy constructor template with better match:
    template <typename T>
    C(T const &) { ... }
    
    ...
};
```
Now the template constructors are used even for "normal" copying:
```c++
C x;
C y {x};  // uses the member template
```
In such a template constructor we can then apply additional constraints with `std::enable_if`. 
For example, to prevent being able to copy objects of a class template `C` 
if the template parameter is an integral type, 
we can implement the following:
```c++
template <typename T>
class C
{
public:
    // user-define the predefined copy constructor as deleted
    // (with conversion to volatile to enable better matches)
    C(C const volatile &) = delete;
    
    // if U is no integral type, 
    // provide copy constructor template with better match:
    template <typename U,
              typename = std::enable_if_t<!std::is_integral<U>::value>>
    C(C<U> const &) 
    {
        ...
    }
};
```


#### 📌 6.5 [Constraints And Concepts](https://en.cppreference.com/w/cpp/language/constraints) `(since C++20)`


Class templates, function templates, and non-template functions (typically members of class templates) 
may be associated with a _constraint_, which specifies the requirements on template arguments, 
which can be used to select the most appropriate function overloads and template specializations.


Named sets of such requirements are called _concepts_. 
Each concept is a predicate, evaluated at compile time, 
and becomes a part of the interface of a template where it is used as a constraint: 
```c++
#include <concepts>
#include <cstddef>
#include <string>
 
// Declaration of the concept "Hashable", 
// which is satisfied by any type 'T' such that for values 'a' of type 'T', 
// the expression std::hash<T>{}(a) compiles, 
// and its result is convertible to std::size_t
template <typename T>
concept Hashable = requires (T t)
{
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};
 
struct meow {};
 
// Constrained C++20 function template:
template <Hashable T>
void f(T) {}

// Alternative ways to apply the same constraint:
template <typename T> requires Hashable<T>
void g(T) {}

// Alternative ways to apply the same constraint:
template <typename T>
void h(T) requires Hashable<T> {}
 
int main()
{
    using std::operator""s;
 
    f("abc"s);   // OK: std::string literals satisfies Hashable
    f(meow {});  // ERROR: meow does not satisfy Hashable
}
```
Violations of constraints are detected at compile time, 
early in the template instantiation process, 
which leads to easy-to-follow error messages. 


The intent of concepts is to model semantic categories (Number, Range, RegularFunction) 
rather than syntactic restrictions (HasPlus, Array). 
According to _ISO C++ Core Guideline T.20_: 
> The ability to specify meaningful semantics 
> is a defining characteristic of a true concept, 
> as opposed to a syntactic constraint. 

#### Syntax Overview

Might not be correct, for quick-but-not-precise reference: 
```
# Usage of concept-related stuff
template-parameter-list : 'typename' parameter-name | 
                          concept-name parameter-name | 
                          ...
template-arguments      : requires-clause
function-declaration    : requires-clause

# Concept Defination
'template' '<' template-parameter-list '>'
'concept' concept-name '=' constraint-expression ';'

# Constraint expression
constraint-expression : conjuctions | 
                        disjunctions | 
                        atomic-constraints | 
                        requires-expression
conjuctions           : constraint-expression '&&' constraint-expression
disjunctions          : constraint-expression '||' constraint-expression
atomic-constraints    : normalizes to true iff. 
                        expression is valid AND 
                        is of type bool with value ture, no conversion permitted

# Requires clause
requires-clause     : constant-expression
constant-expression : primary-expression | 
                      primary-expression '&&' primary-expression | 
                      primary-expression '||' primary-expression | 
                      requires-expression 
primary-expression  : 'true' | 
                      'false' |
                      concept-name | 
                      std::is_integral<T>::value | 
                      std::is_object_v<Args> && ...;
                      '(' expr ')' | 
                      ...

# Requires expression
requires-expression   : 'requires' '{' requirement-seq '}' | 
                        'requires' '(' parameter-list(optional) ')' '{' requirement-seq '}'
requirement-seq       : simple-requirement(expression is valid) | 
                        type-requirements(named type is valid) | 
                        compound-requirements(properties of named expression) |  
                        nested-requirements(mixture of the three above)
compound-requirements : '{' expression '}' 'noexcept'(optional) '->' type-constraint(optional) ';'
nested-requirements   : 'requires' constraint-expression ';'
```

#### Concepts

A concept is a named set of requirements. 
The definition of a concept must appear at namespace scope.
The definition of a concept has the form
```
template <template-parameter-list>
concept concept-name = constraint-expression;
```
```c++
// concept
template <class T, class U>
concept Derived = std::is_base_of<U, T>::value;
```
Concepts can **not** recursively refer to themselves and can **not** be constrained:
```c++
template <typename T>
concept V = V<T *>;     // ERROR: recursive concept
 
template <class T>
concept C1 = true;

template <C1 T>
concept Error1 = true;  // ERROR: C1 T attempts to constrain a concept definition

template <class T> requires C1<T>
concept Error2 = true;  // ERROR: the requires-clause attempts to constrain a concept
```
Explicit instantiations, explicit specializations, or partial specializations of concepts are **not** allowed.  
That is, the meaning of the original definition of a constraint can **not** be changed. 


Concepts can be named in an id-expression. 
The value of the id-expression is `true` if the constraint expression is satisfied, and `false` otherwise.


Concepts can also be named in a _type-constraint_, as part of
- [Type Template Parameter Declaration](https://en.cppreference.com/w/cpp/language/template_parameters#Type_template_parameter)
- [Placeholder Type Specifier (`auto`)](https://en.cppreference.com/w/cpp/language/auto)
- [Compound Requirement](https://en.cppreference.com/w/cpp/language/constraints#Compound_Requirements)


In a type-constraint, a concept takes one less template argument than its parameter list demands, 
because the contextually deduced type is implicitly used as the first argument of the concept.
```c++
template <class T, class U>
concept Derived = std::is_base_of<U, T>::value;
 
template <Derived<Base> T>
void f(T);  // T is constrained by Derived<T, Base>
```

#### Constraints

A constraint is a sequence of logical operations and operands 
that specifies requirements on template arguments. 
They can appear within _requires-expressions_ and directly as bodies of concepts.


There are three types of constraints:
1. **Conjunctions**: Logical AND of multiple constraints via `&&`;
2. **Disjunctions**: Logical OR of multiple constraints via `||`;
3. **Atomic Constraints**: Parameter mapping. 


The constraint associated with a declaration are determined 
by [normalizing](https://en.cppreference.com/w/cpp/language/constraints#Constraint_normalization)
a logical AND expression of the operands. 


A constrained declaration may only be redeclared using the same syntactic form. 
No diagnostic is required:
```c++
template <Incrementable T>
void f(T) requires Decrementable<T>;
 
template <Incrementable T>
void f(T) requires Decrementable<T>;  // OK, redeclaration
 
template <typename T> requires Incrementable<T> && Decrementable<T>
void f(T);  // ill-formed, no diagnostic required
 
// the following two declarations have different constraints:
// the first declaration has Incrementable<T> && Decrementable<T>
// the second declaration has Decrementable<T> && Incrementable<T>
// Even though they are logically equivalent.
 
template <Incrementable T> 
void g(T) requires Decrementable<T>;
 
template <Decrementable T> 
void g(T) requires Incrementable<T>;  // ill-formed, no diagnostic required
```

##### Conjunctions

The conjunction of two constraints is formed by using the `&&` operator in the constraint expression:
```c++
template <class T>
concept Integral = std::is_integral<T>::value;

template <class T>
concept SignedIntegral = Integral<T> && std::is_signed<T>::value;

template <class T>
concept UnsignedIntegral = Integral<T> && !SignedIntegral<T>;
```
A conjunction of two constraints is satisfied only if both constraints are satisfied. 
Conjunctions are evaluated left to right and short-circuited. 

##### Disjunctions

The disjunction of two constraints is formed by using the `||` operator in the constraint expression.


A disjunction of two constraints is satisfied if either constraint is satisfied. 
Disjunctions are evaluated left to right and short-circuited. 
```c++
template <class T = void> requires EqualityComparable<T> || Same<T, void>
struct equal_to;
```

##### Atomic Constraints

An atomic constraint consists of an expression `E` and a mapping: 
from the template parameters that appear within `E`,  
to template arguments involving the template parameters of the constrained entity, 
called its _parameter mapping_.


Atomic constraints are formed during constraint normalization. 
`E` is **never** a logical AND or logical OR expression 
(those form conjunctions and disjunctions, respectively). 


Satisfaction of an atomic constraint is checked 
by substituting the parameter mapping and template arguments into the expression `E`. 
If the substitution results in an invalid type or expression, the constraint is not satisfied. 
Otherwise, `E`, after any lvalue-to-rvalue conversion, 
shall be a prvalue constant expression of type `bool` , 
and the constraint is satisfied if and only if it evaluates to `true`.


The type of `E` after substitution must be exactly `bool`. 
**No** conversion is permitted. 
```c++
template <typename T>
struct S
{
    constexpr operator bool() const { return true; }
};
 
template <typename T> requires (S<T> {})
void f(T);    // #1
 
void f(int);  // #2
 
void g()
{
    f(0);     // ERROR: S<int> {} does not have type bool when checking #1,
              // even though #2 is a better match
}
```
Two atomic constraints are considered _identical_ 
if they are formed from the same expression at the source level
and their parameter mappings are equivalent.
```c++
template <class T>
constexpr bool is_meowable = true;
 
template <class T>
constexpr bool is_cat = true;
 
template <class T>
concept Meowable = is_meowable<T>;
 
template <class T>
concept BadMeowableCat = is_meowable<T> && is_cat<T>;
 
template <class T>
concept GoodMeowableCat = Meowable<T> && is_cat<T>;
 
template <Meowable T>
void f1(T);  // #1
 
template <BadMeowableCat T>
void f1(T);  // #2
 
template <Meowable T>
void f2(T);  // #3
 
template <GoodMeowableCat T>
void f2(T);  // #4
 
void g()
{
    f1(0);   // ERROR, ambiguous:
             // The is_meowable<T> in Meowable and BadMeowableCat forms 
             // distinct atomic constraints that are not identical 
             // (and so do not subsume each other)
 
    f2(0);   // OK, calls #4, more constrained than #3
             // GoodMeowableCat got its is_meowable<T> from Meowable
}
```

##### Constraint Normalization

Constraint normalization is the process that 
transforms a constraint expression into a sequence of conjunctions and disjunctions of atomic constraints. 
The normal form of an expression is defined as follows:
- `(E)`: 
  The normal form of `E`;
- `E1 && E2`: 
  The conjunction of the normal forms of `E1` and `E2`.
- `E1 || E2`: 
  The disjunction of the normal forms of `E1` and `E2`.
- `C<A1, A2, ... , AN>`, where `C` names a concept:  
  The normal form of the constraint expression of `C`, 
  after substituting `A1, A2, ... , AN` for `C`'s respective template parameters 
  in the parameter mappings of each atomic constraint of `C`. 
  If any such substitution into the parameter mappings results in an invalid type or expression, 
  the program is ill-formed, no diagnostic required. 
```c++
concept A = T::value || true;
 
template <typename U>
concept B = A<U *>;    // OK: normalized to the disjunction of 
                       // - T::value (with mapping T -> U *) and
                       // - true (with an empty mapping).
                       // No invalid type in mapping even though
                       // T::value is ill-formed for all pointer types
 
template <typename V>
concept C = B<V &>;    // Normalizes to the disjunction of
                       // - T::value (with mapping T-> V & *) and
                       // - true (with an empty mapping).
                       // Invalid type V & * formed in mapping => ill-formed NDR
```
- Any other expression `E`:  
  The atomic constraint whose expression is `E` and whose parameter mapping is the identity mapping. 
  This includes all fold expressions, even those folding over the `&&` or `||` operators. 


User-defined overloads of `&&` or `||` have **no** effect on constraint normalization.


#### Requires Clauses

The keyword `requires` is used to introduce a _requires-clause_, 
which specifies constraints on template arguments or on a function declaration. 
```c++
// can appear as the last element of a function declarator
template <typename T>
void f(T &&) requires Eq<T>; 

// or right after a template parameter list
template <typename T> requires Addable<T> 
T add(T a, T b) { return a + b; }
```
In this case, the keyword `requires` must be followed by some constant expression 
(so it's possible to write `requires true`), 
but the intent is one of the following:  
- A named concept (as in the example above); 
- A conjunction/disjunction of named concepts; 
  - A _requires-expression_, which must have one of the following forms:
    - A [primary expression](https://en.cppreference.com/w/cpp/language/expressions#Primary_expressions);
      - E.g.: 
        - `Swappable<T>`; 
        - `std::is_integral<T>::value`;
        - `std::is_object_v<Args> && ...`; 
        - Any _parenthesized_ expression `(expr)`. 
    - A sequence of primary expressions joined with `&&`; 
    - A sequence of aforementioned expressions joined with `||`. 
```c++
template <class T>
constexpr bool is_meowable = true;
 
template <class T>
constexpr bool is_purrable() { return true; }
 
template <class T>
void f(T) requires is_meowable<T>;      // OK
 
template <class T>
void g(T) requires is_purrable<T>();    // ERROR, is_purrable<T>() is not a primary expression
 
template <class T>
void h(T) requires (is_purrable<T>());  // OK
```

#### Requires Expressions

The keyword `requires` is also used to begin a _requires-expression_, 
which is a prvalue expression of type `bool` that describes the constraints on some template arguments. 
Such an expression is `true` if the constraints are satisfied, and `false` otherwise:
```c++
// requires-expression
template <typename T>
concept Addable = requires (T t) { t + t; }; 
 
// requires-clause, not requires-expression
template <typename T> requires Addable<T> 
T add(T a, T b) { return a + b; }
 
// ad-hoc constraint, note keyword requires used twice
template <typename T> requires requires (T t) { t + t; } 
T add(T a, T b) { return a + b; }
```
The syntax of requires-expression is as follows:
```c++
requires { requirement-seq }
requires ( parameter-list(optional) ) { requirement-seq }
```
- _parameter-list_: 
  A comma-separated list of parameters like in a function declaration, 
  **except** that default arguments are **not** allowed 
  and it can **not** end with an ellipsis 
  (other than one signifying a pack expansion). 
  These parameters have **no** storage, linkage or lifetime,
  and are only used to assist in specifying requirements. 
  These parameters are in scope until the closing `}` of the _requirement-seq_.
- _requirement-seq_: 
  Sequence of requirements, each requirement ends with a semicolon.
  Each requirement in the _requirements-seq_ is one of the following:
  - **Simple Requirement**: Expression is valid;
  - **Type Requirements**: Named type is valid;
  - **Compound Requirements**: Properties of the named expression;
  - **Nested Requirements**: Mixture of the three above. 


Requirements may refer to the template parameters that are in scope, 
to the local parameters introduced in the _parameter-list_, 
and to any other declarations that are visible from the enclosing context.


The substitution of template arguments into a requires-expression used in a declaration of a templated entity 
may result in the formation of invalid types or expressions in its requirements,
or the violation of semantic constraints of those requirements. 
In such cases, the _requires-expression_ evaluates to `false` 
and does **not** cause the program to be ill-formed. 
The substitution and semantic constraint checking proceeds in lexical order 
and stops when a condition that determines the result of the _requires-expression_ is encountered. 
If substitution (if any) and semantic constraint checking succeed, the requires-expression evaluates to `true`.


If a substitution failure would occur in a _requires-expression_ for every possible template argument, 
the program is ill-formed, no diagnostic required: 
```c++
// invalid for every T: ill-formed, no diagnostic required
template <class T>
concept C = requires
{
    new int[-(int)sizeof(T)];
};
```
If a _requires-expression_ contains invalid types or expressions in its requirements, 
and it does **not** appear within the declaration of a templated entity, 
then the program is ill-formed.

##### Simple Requirements: Expression Is Valid

A simple requirement is an arbitrary expression statement 
that does **not** start with the keyword `requires`. 
It asserts that the _expression is valid_. 
The expression is an unevaluated operand. 
Only language correctness is checked.
```c++
template <typename T>
concept Addable = requires (T a, T b)
{
    a + b; // "the expression a+b is a valid expression that will compile"
};
 
template <class T, class U = T>
concept Swappable = requires (T && t, U && u)
{
    swap(std::forward<T>(t), std::forward<U>(u));
    swap(std::forward<U>(u), std::forward<T>(t));
};
```
A requirement that starts with the keyword `requires` is always interpreted as a nested requirement. 
Thus a simple requirement can **not** start with an unparenthesized _requires-expression_.

##### Type Requirements: Named Type is Valid

A type requirement is the keyword `typename` followed by a type name, optionally qualified. 
The requirement is that the _named type is valid_. 
This can be used to verify that a certain named nested type exists, 
or that a class template specialization names a type, 
or that an alias template specialization names a type. 
A type requirement naming a class template specialization does **not** require the type to be complete.
```c++
template <typename T>
using Ref = T &;
 
template <typename T>
concept C = requires
{
    typename T::inner;  // required nested member name
    typename S<T>;      // required class template specialization
    typename Ref<T>;    // required alias template substitution
};
 
template <class T, class U>
using CommonType = std::common_type_t<T, U>;
 
template <class T, class U>
concept Common = requires (T && t, U && u)
{
    // CommonType<T, U> is valid and names a type
    typename CommonType<T, U>;
    { CommonType<T, U>{std::forward<T>(t)} }; 
    { CommonType<T, U>{std::forward<U>(u)} }; 
};
```

##### Compound Requirements: Properties of The Named Expression

A compound requirement has the form
```
{ expression } noexcept(optional) -> type-constraint(optional) ;
```
and asserts _properties of the named expression_. 


Substitution and semantic constraint checking proceeds in the following order: 
1. Template arguments (if any) are substituted into expression;
2. If `noexcept` is used, expression must **not** be potentially throwing;
3. If _return-type-requirement_ is present, then:
   1. Template arguments are substituted into the _return-type-requirement_;
   2. `decltype((expression))` must satisfy the constraint imposed by the _type-constraint_. 
      Otherwise, the enclosing _requires-expression_ is `false`.
```c++
template <typename T>
concept C2 = requires(T x)
{
    // the expression *x must be valid
    // AND the type T::inner must be valid
    // AND the result of *x must be convertible to T::inner
    {*x} -> std::convertible_to<typename T::inner>;
 
    // the expression x + 1 must be valid
    // AND std::same_as<decltype((x + 1)), int> must be satisfied
    // i.e., (x + 1) must be a prvalue of type int
    {x + 1} -> std::same_as<int>;
 
    // the expression x * 1 must be valid
    // AND its result must be convertible to T
    {x * 1} -> std::convertible_to<T>;
};
```

##### Nested Requirements

A nested requirement has the form
```
requires constraint-expression ;
```
It can be used to specify additional constraints in terms of local parameters. 
The constraint-expression must be satisfied by the substituted template arguments, if any. 
Substitution of template arguments into a nested requirement 
causes substitution into the constraint-expression 
only to the extent needed to determine whether the constraint-expression is satisfied.
```c++
template <class T>
concept Semiregular = 
        DefaultConstructible<T> && 
        CopyConstructible<T> &&
        Destructible<T> && 
        CopyAssignable<T> &&
requires(T a, size_t n)
{  
    // nested: "Same<...> evaluates to true"
    requires Same<T *, decltype(& a)>; 
    
    // compound: "a.~T()" is a valid expression that doesn't throw
    { a.~T() } noexcept;
    
    // nested: "Same<...> evaluates to true"
    requires Same<T*, decltype(new T)>;
    
    // nested
    requires Same<T*, decltype(new T[n])>;  
    
    // compound
    { delete new T };
    
    // compound
    { delete new T[n] };  
};
```


#### 📌 6.6 Summary


- In templates, you can "perfectly" forward parameters by declaring them as forwarding references 
  (declared with a type formed with the name of a template parameter followed by `&&`) 
  and using `std::forward` in the forwarded call.
- When using perfect forwarding member function templates, 
  they might match better than the predefined special member function to copy or move objects. 
- With std::enable_if, you can disable a function template when a compile-time condition is `false` 
  (the template is then ignored once that condition has been determined). 
- By using `std::enable_if` you can avoid problems 
  when constructor templates or assignment operator templates
  that can be called for single arguments 
  are a better match than implicitly generated special member functions.
- You can templify (and apply `enable_if`) to special member functions 
  by `delete`ing the predefined special member functions for `const volatile`.
- Concepts will allow us to use a more intuitive syntax for requirements on function templates. 






### 🎯 Chapter 7 By Value or by Reference?


C++ provides both _call-by-value_ and _call-by-reference_, 
and it is **not** always easy to decide which one to choose:
Usually calling by reference is cheaper for nontrivial objects but more complicated. 
C++11 added move semantics to the mix, 
which means that we now have different ways to pass by reference:
1. `X const &` (constant lvalue reference):  
   The parameter refers to the passed object, **without** the ability to modify it; 
2. `X &` (nonconstant lvalue reference):  
   The parameter refers to the passed object, with the ability to modify it;
3. `X &&` (rvalue reference):  
   The parameter refers to the passed object, with move semantics, 
   meaning that you can modify or "steal" the value.
4. `X const &&` (constant rvalue reference):  
   Available, but with no established semantic meaning.   


Deciding how to declare parameters with known concrete types is complicated enough. 
In templates, types are **not** known, and therefore it becomes even harder 
to decide which passing mechanism is appropriate. 


Nevertheless, in Section 1.6 we did recommend passing parameters in function templates by value 
unless there are good reasons, such as the following:
- Copying (for lvalues, `since C++17`) is **not** possible. 
  Note that since C++17 you can pass temporary entities (rvalues) by value 
  even if **no** copy or move constructor is available (see Section B.2). 
  So, since C++17 the additional constraint is that copying for lvalues is not possible. 
- Parameters are used to return data. 
- Templates just perfect-forward the parameters to somewhere else 
  by keeping all the properties of the original arguments.
- There are significant performance improvements.


This chapter discusses the different approaches to declare parameters in templates, 
motivating the general recommendation to pass by value, 
and providing arguments for the reasons not to do so. 


It also discusses the tricky problems you run into 
when dealing with string literals and other raw arrays. 


When reading this chapter, it is helpful to be familiar with 
the terminology of _value categories_ (lvalue, rvalue, prvalue, xvalue, etc.), 
which is explained in Appendix B.


#### 📌 7.1 Passing by Value


When passing arguments by value, each argument must in principle be copied. 
Thus, each parameter becomes a copy of the passed argument. 
For classes, the object created as a copy 
is generally initialized by the copy constructor.


Calling a copy constructor can become expensive. 
However, there are various way to avoid expensive copying 
even when passing parameters by value: 
In fact, compilers might optimize away copy operations. 
Copying objects can become cheap even for complex objects by using move semantics.


For example, let's look at a simple function template 
implemented so that the argument is passed by value:
```c++
template <typename T>
void printV(T arg) 
{
    ...
}
```
When calling this function template for an integer, the resulting code is
```c++
void printV(int arg) 
{
    ...
}
```
Parameter `arg` becomes a copy of any passed argument, 
no matter whether it is an object, a literal, or a function return value.


If we define a `std::string` and call our function template for it:
```c++
void printV(std::string arg)
{
    ...
}
```
Again, when passing the `std::string`, `arg` becomes a copy. 
This time the copy is created by the copy constructor of the `std::string` class, 
which is a potentially expensive operation,
because in principle this copy operation creates a full or "deep" copy 
so that the copy internally allocates its own memory to hold the value. 


The implementation of the `std::string` class might itself 
have some optimizations to make copying cheaper. 
One is the _Small String Optimization (SSO)_, 
using some memory directly inside the object to hold the value 
without allocating memory as long as the value is not too long. 
Another is the copy-on-write optimization, which creates a copy using the same memory 
as the source as long as neither source nor the copy is modified. 
However, the copy-on-write optimization has significant drawbacks in multi-threaded code.
For this reason, it is forbidden for standard strings since C++11. 


However, the potential copy constructor is **not** always called. 
Consider the following:
```c++
std::string returnString();
std::string s = "hi";
printV(s);                   // copy constructor
printV(std::string("hi"));   // copying usually optimized away (if not, move constructor)
printV(returnString());      // copying usually optimized away (if not, move constructor)
printV(std::move(s));        // move constructor
```
In the first call we pass an lvalue, 
which means that the copy constructor is used. 
However, in the second and third calls, 
when directly calling the function template for prvalues 
(temporary objects created on the fly or returned by another function; see Appendix B), 
compilers usually optimize passing the argument 
so that **no** copying constructor is called at all. 
Note that since C++17, this optimization is required. 
Before C++17, a compiler that doesn't optimize the copying away, 
must at least have to try to use move semantics,
which usually makes copying cheap. 
In the last call, when passing an xvalue 
(an existing non-constant object with `std::move`), 
we force to call the move constructor
by signaling that we no longer need the value of `s`. 


Thus, calling an implementation of `printV` 
that declares the parameter to be passed by value 
usually is only expensive if we pass an lvalue 
(an object we created before and typically still use afterwards, 
as we didn't use `std::move` to pass it). 
Unfortunately, this is a pretty common case. 
One reason is that it is pretty common to create objects early 
to pass them later (after some modifications) to other functions.

##### Passing by Value Decays

There is another property of passing by value we have to mention: 
When passing arguments to a parameter by value, the type _decays_. 
This means that raw arrays get converted to pointers, 
and that top-level cv-constraints are removed, 
just like using the value as initializer for an object declared with `auto`.
```c++
template <typename T>
void printV(T arg) 
{
    ...
}

std::string const c = "hi";
printV(c);     // decays so that arg has type std::string
printV("hi");  // decays to pointer so that arg has type char const * 

int arr[4];
printV(arr);   // decays to pointer so that arg has type char const *
```
Thus, when passing the string literal `"hi"`, 
its type `char const [3]` decays to `char const *` so that this is the deduced type of `T`. 
Thus, the template is instantiated as follows:
```c++
void printV(char const * arg)
{
    ...
}
```
This behavior is derived from C and has its benefits and drawbacks. 
Often it simplifies the handling of passed string literals, 
but the drawback is that inside `printV` we **can't** distinguish between 
passing a pointer to a single element and passing a raw array. 
For this reason, we will discuss how to deal with string literals
and other raw arrays in Section 7.4.


#### 📌 7.2 Passing by Reference


Now let's discuss the different flavors of passing by reference. 
In all cases, **no** _copy_ gets created (because the parameter just refers to the passed argument). 
Also, passing the argument **never** _decays_. 
However, sometimes pass-by-reference is **not** possible, and if passing is possible, 
there are cases in which the resulting type of the parameter may cause problems.

##### 7.2.1 Passing by Constant Reference

To avoid any (unnecessary) copying, when passing non-temporary objects, 
we can use constant references. For example:
```c++
template <typename T>
void printR(T const & arg) 
{
    ...
}

std::string returnString();
std::string s = "hi";
printR(s);                  // no copy
printR(std::string("hi"));  // no copy
printR(returnString());     // no copy
printR(std::move(s));       // no copy
```
Even an `int` is passed by reference, 
which is a bit counter-productive 
but shouldn't matter that much: 
```c++
int i = 42;
printR(i);   // passes const int & (8 Bytes on 64-bit OS) 
             // instead of copying an int (4 Bytes)
```
Under the hood, passing an argument by reference 
is implemented by passing the address of the argument. 
Addresses are encoded compactly, and therefore 
transferring an address from the caller to the callee is efficient in itself. 
However, passing an address can create uncertainties for the compiler 
when it compiles the caller's code: 
What is the callee doing with that address? 
In theory, the callee can change all the values that are "reachable" through that address. 
That means, that the compiler has to assume that 
_all the values it may have cached (usually in machine registers) are invalid after the call_. 
Reloading all those values can be quite expensive. 


You may be thinking that we are passing by _constant_ reference: 
Cannot the compiler deduce from that that no change can happen? 
Unfortunately, that is **not** the case 
because the caller may modify the referenced object 
through its own, non-const reference.
Furthermore, the use of `const_cast` is another, more explicit, 
way to modify the referenced object.


This bad news is moderated by inlining: 
If the compiler can expand the call _inline_,
it can reason about the caller and the callee _together_ 
and in many cases "see" that the address is not used for anything but passing the underlying value. 
Function templates are often very short and therefore likely candidates for inline expansion. 
However, if a template encapsulates a more complex algorithm, inlining is less likely to happen.

##### Passing by Reference Does Not Decay

When passing arguments to parameters by reference, they do **not** decay. 
This means that raw arrays are **not** converted to pointers 
and that cv-constraints are untouched. 
However, because the _call_ parameter is declared as `T const &`, 
the _template_ parameter `T` itself is **not** deduced as `const`. 
For example:
```c++
template <typename T>
void printR(T const & arg) 
{
    ...
}

std::string const c = "hi";
printR(c);     // T deduced as std::string, arg is std::string const &
printR("hi");  // T deduced as char [3], arg is const char (&)[3]
int arr[4];
printR(arr);   // T deduced as int [4], arg is const int (&)[4]
```
Thus, local objects declared with type `T` in `printR` are **not** constant. 

##### 7.2.2 Passing by Non-constant Reference

When you want to return values through passed arguments 
(i.e., when you want to use _out_ or _inout_ parameters), 
you have to use non-constant references (unless you prefer to pass them via pointers). 
Again, this means that when passing the arguments, **no** copy gets created. 
The parameters of the called function template just get direct access to the passed argument.
Consider the following:
```c++
template <typename T>
void outR(T & arg) 
{
    ...
}
```
Note that calling `outR` for a temporary (prvalue) 
or an existing object passed with `std::move` (xvalue) 
usually is **not** allowed:
```c++
std::string returnString();
std::string s = "hi";
outR(s);                     // OK: T deduced as std::string, arg is std::string&
outR(std::string("hi"));     // ERROR: not allowed to pass a temporary (prvalue)
outR(returnString());        // ERROR: not allowed to pass a temporary (prvalue)
outR(std::move(s));          // ERROR: not allowed to pass an xvalue
```
You can pass raw arrays of non-constant types, which again **don't** decay:
```c++
int arr[4];
outR(arr);                   // OK: T deduced as int [4], arg is int (&)[4]
```
Thus, you can modify elements and, for example, deal with the size of the array. 
For example:
```c++
template <typename T>
void outR(T & arg)
{
    if constexpr (std::is_array_v<T>)
    {
        std::cout << "got array of " << std::extent_v<T> << "elems\n";
    }
    
    ...
}
```
However, templates are a bit tricky here. 
If you pass a `const` argument, the deduction might result in 
`arg` becoming a declaration of a constant reference, 
which means that passing an rvalue is suddenly allowed, 
where an lvalue is expected:
```c++
std::string const c = "hi";
outR(c);                    // OK: T deduced as std::string const
outR(returnConstString());  // OK: T deduced as std::string const
outR(std::move(c));         // OK: T deduced as std::string const
outR("hi");                 // OK: T deduced as char const [3]
```
Note: 
When passing `std::move(c)`, `std::move` first converts `c` to `std::string const &&`, 
which then has the effect that `T` is deduced as `std::string const`. 


Of course, any attempt to modify the passed argument inside the function template is an error in such cases. 
Passing a `const` object is possible in the call expression itself,
but when the function is fully instantiated (which may happen later in the compilation process), 
any attempt to modify the value will trigger an error 
(which, however, might happen deep inside the called template; see Section 9.4). 


If you want to disable passing constant objects to non-constant references, 
you can do the following:
- Use `static_assert` to trigger a compile-time error:
```c++
template <typename T>
void outR(T & arg) 
{
    static_assert(!std::is_const_v<T>, "out parameter of foo<T>(T &) is const");
    ...
}
```
- Disable the template for this case by `std::enable_if`:
```c++
template <typename T, 
          typename = std::enable_if_v<!std::is_const_v<T>>>
void outR(T & arg) 
{
    ...
}
```
- Disable the template for this case by concepts:
```c++
template <typename T> requires !std::is_const_v<T>
void outR(T & arg) 
{
    ...
}
```

##### 7.2.3 Passing by Forwarding Reference

One reason to use call-by-reference is to be able to perfect forward a parameter (see Section 6.1). 
But remember that when a forwarding reference is used, 
which is defined as an rvalue reference of a template parameter, special rules apply.
```c++
template <typename T>
void passR(T && arg) 
{
    ...
}
```
You can pass everything to a forwarding reference and, 
as usual when passing by reference, **no** copy gets created. 
However, the special rules for type deduction may result in some surprises:
```c++
std::string const c = "hi";
passR(c);     // OK: T deduced as std::string const &

// Note that C-style string literals are lvalues by language standard! 
passR("hi");  // OK: T deduced as char const(&)[3] (also the type of arg)

int arr[4];
passR(arr);   // OK: T deduced as int (&)[4] (also the type of arg)
```
In each of these cases, inside `passR` the parameter `arg` has a type that "knows"
whether we passed an rvalue (to use move semantics) or a constant/non-constant lvalue. 
This is the only way to pass an argument, 
such that it can be used to distinguish behavior for all of these three cases.
This gives the impression that declaring a parameter as a forwarding reference is almost perfect. 
But beware, there is no free lunch.
For example, this is the only case where the template parameter `T`
implicitly can become a reference type. 
As a consequence, it might become an error to use `T` to declare a local object without initialization:  
```c++
template <typename T>
void passR(T && arg) 
{
    // for passed lvalues, x is a reference, 
    // which requires an initializer
    T x;
    
    ...
}

foo(42);  // OK: T deduced as int
int i;
foo(i);   // ERROR: T deduced as int &, which makes the declaration of x in passR invalid
```
See Section 15.6 for further details about 
how you can deal with this situation. 

#### 7.3 Using [`std::ref` and `std::cref`](https://en.cppreference.com/w/cpp/utility/functional/ref)

Since C++11, you can let the caller decide, 
for a function template argument,
whether to pass it by value or by reference. 
When a template is declared to take arguments by value, 
the caller can use `std::cref` and `std::ref`,
declared in header file `<functional>`,
to pass the argument "as if by reference".
For example:
```c++
template <typename T>
void printT(T arg) 
{
    ...
}

std::string s = "hello";
printT(s);             // pass s by value
printT(std::cref(s));  // pass s "as if by reference"
```
However, note that `std::cref` does **not** change the handling of parameters in templates. 
Instead, it uses a trick: 
It wraps the passed argument `s` by an object that acts like a reference. 
In fact, it creates an object of type 
[`std::reference_wrapper`](https://en.cppreference.com/w/cpp/utility/functional/reference_wrapper) 
referring to the original argument 
and passes this object by value. 
The wrapper more or less supports only one operation: 
an implicit type conversion back to the original type, yielding the original object. 
So, whenever you have a valid operator for the passed object, 
you can use the reference wrapper instead. For example:
```c++
void printString(std::string const & s)
{
    std::cout << s << '\n';
}

template <typename T>
void printT(T arg)
{
    printString(arg);     // might convert arg back to std::string
}

std::string s = "hello";
printT(s);                // print s passed by value
printT(std::cref(s));     // print s passed "as if by reference"
```
The last call passes by value an object of type `std::reference_wrapper<string const>` to the parameter `arg`, 
which then passes and therefore converts it back to its underlying type `std::string`.
Note that the compiler has to know that an implicit conversion back to the original type is necessary. 
For this reason, `std::ref` and `std::cref` usually work fine _only if_ you pass objects _through_ generic code. 
For example, directly trying to output the passed object of the generic type `T` will fail 
because there is **no** output operator defined for `std::reference_wrapper`:
```c++
template <typename T>
void printV(T arg) 
{
    std::cout << arg << '\n';
}

std::string s = "hello";
printV(s);                // OK
printV(std::cref(s));     // ERROR: no operator<< for reference wrapper defined
```
Also, the following fails because you **can't** compare 
a reference wrapper with a `char const *` or `std::string`:
```c++
template <typename T1, typename T2>
bool isless(T1 arg1, T2 arg2)
{
    return arg1 < arg2;
}

std::string s = " hello";
if (isless(std::cref(s) < "world")) ...               // ERROR
if (isless(std::cref(s) < std::string("world"))) ...  // ERROR
```
It also **doesn't** help to give `arg1` and `arg2` a common type `T`:
```c++
template <typename T>
bool isless(T arg1, T arg2)
{
    return arg1 < arg2;
}
```
because then the compiler gets _conflicting types_ when trying to deduce `T` for `arg1` and `arg2`. 


Thus, the effect of class `std::reference_wrapper` is 
to be able to use a reference as a "first class object",  
which you can copy and therefore pass by value to function templates. 
You can also use it in classes, for example,
to hold references to objects in containers. 
But you always finally need a conversion back to the underlying type.


#### 📌 7.4 Dealing with String Literals and Raw Arrays


So far, we have seen the different effects for templates parameters 
when using string literals and raw arrays: 
- Call-by-value decays so that they become pointers to the element type. 
- Any form of call-by-reference does not decay 
  so that the arguments become references that still refer to arrays.


Both can be good and bad. 
When decaying arrays to pointers, you lose the ability to 
distinguish between handling pointers to elements from handling passed arrays. 
On the other hand, when dealing with parameters where string literals may be passed,
not decaying can become a problem,
because string literals of different size have different types. 
For example:
```c++
template <typename T>
void foo(T const & arg1, T const & arg2)
{
    ...
}

foo("hi", "guy");  // ERROR
```
Here, `foo("hi","guy")` fails to compile, 
because `"hi"` has type `char const [3]`, 
while `"guy"` has type `char const [4]`, 
but the template requires them to have the same type `T`. 
Only if the string literals were to have the same length would such code compile. 
For this reason, it is strongly recommended to use string literals of different lengths in test cases. 


By declaring the function template `foo` to pass the argument by value, the call is possible:
```c++
template <typename T>
void foo(T arg1, T arg2)
{
    ...
}

foo("hi", "guy");  // compiles, but...
```
But, that doesn't mean that all problems are gone. 
Even worse, compile-time problems may have become run-time problems. 
Consider the following code, where we compare the passed argument using `operator==`:
```c++
template <typename T>
void foo(T arg1, T arg2)
{
    if (arg1 == arg2)
    {
        // OOPS: compares addresses of passed arrays
        ...
    }
    
    ...
}

foo("hi", "guy");  // compiles, but...
```
As written, you have to know that you should 
interpret the passed character pointers as strings. 
But that's probably the case anyway, 
because the template also has to deal with arguments coming from 
string literals that have been decayed already 
(e.g., by coming from another function called by value 
or being assigned to an object declared with `auto`). 


Nevertheless, in many cases decaying is helpful,
especially for checking whether two objects 
(both passed as arguments, or one passed as argument and the other expecting the argument)
have or convert to the same type. 
One typical usage is perfect forwarding. 
But if you want to use perfect forwarding,
you have to declare the parameters as forwarding references. 
In those cases, you might explicitly decay the arguments 
using the type trait `std::decay`. 
See the story of `std::make_pair` in Section 7.6 for a concrete example. 


Note that other type traits sometimes also implicitly decay, 
such as `std::common_type`, 
which yields the common type of two passed argument types (see Section D.5). 

##### 7.4.1 Special Implementations for String Literals and Raw Arrays

You might have to distinguish your implementation according to 
whether a pointer or an array was passed. 
This, of course, requires that a passed array wasn't decayed yet. 


To distinguish these cases, you have to detect whether arrays are passed.
Basically, there are two options:
- You can declare template parameters so that they are only valid for arrays:
  ```c++
  template <typename T, std::size_t L1, std::size_t L2>
  void foo(T (& arg1)[L1], T (& arg2)[L2])
  {
      T * pa = arg1;  // decay arg1
      T * pb = arg2;  // decay arg2
      
      if (compareArrays(pa, L1, pb, L2)) 
      {
          ...
      }
  }
  ```
  Here, `arg1` and `arg2` have to be raw arrays of the same element type `T` 
  but with different sizes `L1` and `L2`. 
  However, note that you might need _multiple implementations_
  to support the four various forms for each raw array call parameter, 
  including whether the array is referenced or is bounded (see Section 5.4).
- You can use type traits to detect whether an array (or a pointer) was passed:
```c++
template <typename T,
          typename = std::enable_if_t<std::is_array_v<T>>>
void foo (T && arg1, T && arg2)
{
    ...
}
```
Due to these special handling, 
often the best way to deal with arrays in different ways 
is simply to use different function names. 
Even better, of course, is to ensure that the caller of a template 
uses `std::vector` or `std::array` or `std::string` literal. 
But as long as string literals are raw arrays, 
we always have to take them into account. 


#### 📌 7.5 Dealing with Return Values


For return values, you can also decide between returning by value or by reference. 
However, returning references is potentially a source of trouble, 
because you refer to something that is out of your control. 


First, **never** ~~return any type of reference to temporaries~~, 
including const lvalue reference and rvalue reference. 
The delay of temporary destruction due to references can not be passed on. 


There are a few cases where returning references is common programming practice:
- Returning elements of containers or strings (e.g., by `operator[]` or `front`);
- Granting write access to class members;
- Returning objects for chained calls 
  (`operator<<` and `operator>>` for streams and `operator=` for class objects in general). 


In addition, it is common to grant read access to members by returning `const` references.


Note that all these cases may cause trouble if used improperly. For example:
```c++
std::string * s = new std::string("whatever");
auto & c = (*s)[0];
delete s;
std::cout << c << '\n';  // run-time ERROR
```
Here, we obtained a reference to an element of a string, 
but by the time we use that reference, 
the underlying string no longer exists 
(i.e., we created a dangling reference),
and we have undefined behavior.
This example is somewhat contrived
(the experienced programmer is likely to notice the problem right away), 
but things easily become less obvious. For example:
```c++
auto s = std::make_shared<std::string>("whatever");
auto & c = (*s)[0];
s.reset();
std::cout << c << '\n';  //run-time ERROR
```
We should therefore ensure that function templates return their result by value. 
However, as discussed in this chapter, 
using a template parameter `T` is **no** guarantee that it is not a reference, 
because `T` might sometimes implicitly be deduced as a reference:
```c++
template <typename T>
T retR(T && p)       // p is a forwarding reference
{
    return T {...};  // OOPS: returns by reference when called for lvalues
}
```
Even when `T` is a template parameter deduced from a call-by-value call, 
it might become a reference type when explicitly specifying the template parameter to be a reference:
```c++
template <typename T>
T retV(T p)          // Note: T might become a reference
{
    return T {...};  // OOPS: returns a reference if T is a reference
}

int x;
retV<int &>(x);      // retT instantiated for T as int &
```
To be safe, you have two options:
- Use the type trait `std::remove_reference` (see Section D.4)
  to convert type `T` to a non-reference:
```c++
template <typename T>
std::remove_reference_t<T> retV(T p)
{
    return T {...};  // always returns by value
}
```
  Other traits, such as `std::decay` (see Section D.4), 
  may also be useful here because they also implicitly remove references.
- Let the compiler deduce the return type by just declaring the return type to be `auto`, 
  because `auto` always decays: 
```c++
template <typename T>
auto retV(T p)       // by-value return type deduced by compiler
{
    return T {...};  // always returns by value
}
```


#### 📌 7.6 Recommended Template Parameter Declarations


As we learned in the previous sections, 
we have very different ways to declare parameters that depend on template parameters:
- **Declare to pass the arguments by value**:  
  This approach is simple, it decays string literals and raw arrays, 
  but it **doesn't** provide the best performance for large objects. 
  Still the caller can decide to pass by reference using `std::cref` and `std::ref`, 
  but the caller must be careful that doing so is valid.
- **Declare to pass the arguments by-reference**:  
  This approach often provides better performance for somewhat large objects,
  especially when passing: 
  – Existing objects (lvalues) to lvalue references;
  – Temporary objects (prvalues) or objects marked as movable (xvalue) to rvalue references;
  – Or both to forwarding references.
  Because in all these cases the arguments **don't** decay, 
  you may need special care when passing string literals and other raw arrays. 
  For forwarding references, you also have to beware that with this approach 
  template parameters implicitly can deduce to reference types.

##### General Recommendations

With these options in mind, for function templates we recommend the following:
1. By default, declare parameters to be passed by value. 
   This is simple and usually works even with string literals. 
   The performance is fine for small arguments and for temporary or movable objects. 
   The caller can sometimes use `std::ref` and `std::cref` when passing existing large objects (lvalues)
   to avoid expensive copying. 
2. If there are good reasons, do otherwise:
   - If you need an out or inout parameter, 
     which returns a new object or allows to modify an argument to/for the caller, 
     pass the argument as a non-constant reference
     (unless you prefer to pass it via a pointer). 
     However, you might consider disabling accidentally accepting `const` objects
     as discussed in Section 7.2.2. 
   - If a template is provided to forward an argument, use perfect forwarding. 
     That is, declare parameters to be forwarding references and use `std::forward` where appropriate. 
     Consider using `std::decay` or `std::common_type` to "harmonize" 
     the different types of string literals and raw arrays.
   - If performance is key and it is expected that copying arguments is expensive, 
     use constant references. 
     This, of course, does not apply if you need a local copy anyway.
3. If you know better, don't follow these recommendations. 
   However, do **not** ~~make intuitive assumptions about performance~~. 
   Even experts fail if they try. Instead: Measure! 

##### Don't Be Over-Generic

Note that, in practice, function templates often are **not** for arbitrary types of arguments. 
Instead, some constraints apply. 
For example, you may know that only vectors of some type are passed. 
In this case, it is better **not** to declare such a function too generically, 
because, as discussed, surprising side effects may occur. 
Instead, use the following declaration:
```c++
template <typename T>
void printVector(std::vector<T> const & v)
{
    ...
}
```
With this declaration of parameter `v` in `printVector`, 
we can be sure that the passed `T` **can't** become a reference 
because vectors can't use references as element types. 
Also, it is pretty clear that passing a vector by value almost always can become expensive because 
the copy constructor of `std::vector` creates a copy of the elements. 
For this reason, it is probably **never** useful to declare such a vector parameter to be passed by value.
If we declare parameter `v` just as having type `T` deciding, 
between call-by-value and call-by-reference becomes less obvious. 

##### The `std::make_pair` Example

`std::make_pair` is a good example to demonstrate the pitfalls of
deciding a parameter passing mechanism. 
It is a convenience function template in the C++ standard library 
to create `std::pair` objects using type deduction. 
Its declaration changed through different versions of the C++ standard:
- In the first C++ standard, C++98, 
  `std::make_pair` was declared to use call-by-reference to avoid unnecessary copying:
  ```c++
  template <typename T1, typename T2>
  pair<T1, T2> make_pair(T1 const & a, T2 const & b)
  {
      return pair<T1, T2>(a, b);
  }
  ```
  This, however, almost immediately caused significant problems 
  when using pairs of string literals or raw arrays of different size.
- As a consequence, with C++03 the function definition was changed to use call-by-value:
  ```c++
  template <typename T1, typename T2>
  pair<T1, T2> make_pair(T1 a, T2 b)
  {
      return pair<T1, T2>(a, b);
  }
  ```
  As you can read in the rationale for the issue resolution, 
  > It appeared that this was a much smaller change to the standard than the other two suggestions, 
  > and any efficiency concerns were more than offset by the advantages of the solution.
- However, with C++11, `make_pair` had to support move semantics, 
  so that the arguments had to become forwarding references. 
  The complete implementation is even more complex: 
  To support `std::ref` and `std::cref`, the function also 
  unwraps instances of `std::reference_wrapper` into real references.
  ```c++
  /// <type_traits>
  /// g++ (Ubuntu 9.4.0-1ubuntu1~20.04) 9.4.0
  template <typename _Tp>
  class reference_wrapper;
  
  // Helper which adds a reference to a type when given a reference_wrapper
  template <typename _Tp>
  struct __strip_reference_wrapper
  {
      typedef _Tp __type;
  };
  
  template <typename _Tp>
  struct __strip_reference_wrapper<reference_wrapper<_Tp>>
  {
      typedef _Tp & __type;
  };
  
  template <typename _Tp>
  struct __decay_and_strip
  {
      typedef typename __strip_reference_wrapper<typename decay<_Tp>::type>::__type __type;
  };
  
  /// <stl_pair>
  /// g++ (Ubuntu 9.4.0-1ubuntu1~20.04) 9.4.0
  template <typename _T1, typename _T2>
  constexpr pair<typename __decay_and_strip<_T1>::__type,
                 typename __decay_and_strip<_T2>::__type>
  make_pair(_T1 && __x, _T2 && __y)
  {
      typedef typename __decay_and_strip<_T1>::__type __ds_type1;
      typedef typename __decay_and_strip<_T2>::__type __ds_type2;
      typedef pair<__ds_type1, __ds_type2> __pair_type;
      return __pair_type(std::forward<_T1>(__x), std::forward<_T2>(__y));
  }
  ```
The C++ standard library now perfectly forwards passed arguments 
in many places in similar way, 
often combined with using `std::decay`.


#### 📌 7.7 Summary


- When testing templates, use string literals of different length.
- Template parameters passed by value decay, 
  while passing them by reference does **not** decay. 
- The type trait `std::decay` allows you to decay parameters 
  in templates passed by reference.
- In some cases `std::cref` and `std::ref` allow you to pass arguments by reference 
  when function templates declare them to be passed by value.
- Passing template parameters by value is simple 
  but may not result in the best performance.
- Pass parameters to function templates by value 
  unless there are good reasons to do otherwise.
- Ensure that return values are usually passed by value 
  (which might mean that a template parameter **can't** be specified directly as a return type).
- Always measure performance when it is important. 
  Do **not** rely on intuition; it's probably wrong.






### 🎯 Chapter 8 Compile-Time Programming 


C++ has always included some simple ways to compute values at compile time. 
Templates considerably increased the possibilities in this area, 
and further evolution of the language has only added to this toolbox. 


In the simple case, you can decide whether to use certain or to choose between different template code. 
But the compiler even can compute the outcome of control flow at compile time, 
provided all necessary input is available. 


In fact, C++ has multiple features to support compile-time programming: 
- Since before C++98, templates have provided the ability to compute at compile time, 
  including the TMP-based implementation 
  of branches (via partial specializations) and loops (via recursion).
- With partial specialization we can choose at compile time 
  between different class template implementations 
  depending on specific constraints or requirements.
- With the SFINAE principle, we can allow 
  selection between different function template implementations 
  for different types or different constraints.
- In C++11 and C++14, compile-time computing became increasingly better supported 
  with the `constexpr` feature using "intuitive" execution path selection
  and, since C++14, most statement kinds (including for loops, switch statements, etc.). 
- C++17 introduced a `if constexpr` ("compile-time `if`") 
  to discard statements depending on compile-time conditions or constraints. 
  It works even outside of templates. 


This chapter introduces these features with a special focus on the role and context of templates.


#### 📌 8.1 Template Metaprogramming (TMP)


Templates are instantiated at compile time
(in contrast to dynamic languages, where genericity is handled at run time).
It turns out that some of the features of C++ templates 
can be combined with the instantiation process 
to produce a sort of primitive recursive "programming language" (Functional Programming)
within the C++ language itself. 


In fact, it was Erwin Unruh who first found it out 
by presenting a program computing prime numbers at compile time. 
See Section 23.7 for details. 


For this reason, templates can be used to "compute a program". 
Chapter 23 will cover the whole story and all features, 
but here is a short example of what is possible. 


The following code finds out at compile time 
whether a given number is a prime number:
```c++
// p: number to check, d: current divisor
template <unsigned p, unsigned d> 
struct DoIsPrime
{
    static constexpr bool value = (p % d != 0) && DoIsPrime<p, d - 1>::value;
};

// end recursion if divisor is 2 
template <unsigned p> 
struct DoIsPrime<p, 2>
{
    static constexpr bool value = (p % 2 != 0);
};

// primary template
template <unsigned p> 
struct IsPrime
{
    // start recursion with divisor from p / 2:
    static constexpr bool value = DoIsPrime<p, p / 2>::value;
};

// special cases (to avoid endless recursion with template instantiation):
template <>
struct IsPrime<0>
{
    static constexpr bool value = false;
};

template <>
struct IsPrime<1>
{
    static constexpr bool value = false;
};

template <>
struct IsPrime<2>
{
    static constexpr bool value = true;
};

template <>
struct IsPrime<3>
{
    static constexpr bool value = true;
};
```
The `IsPrime` template returns in member value 
whether the passed template parameter `p` is a prime number. 
To achieve this, it instantiates `DoIsPrime`,
which recursively expands to an expression checking for each divisor `d` 
between `p / 2` and `2` whether the divisor divides `p` without remainder. 


For example, the expression
```c++
IsPrime<9>::value
```
expands to
```c++
DoIsPrime<9, 4>::value
```
which expands to
```c++
9 % 4 != 0 && DoIsPrime<9, 3>::value
```
which expands to
```c++
9 % 4 != 0 && 9 % 3 != 0 && DoIsPrime<9, 2>::value
```
which expands to
```c++
9 % 4 != 0 && 9 % 3 != 0 && 9 % 2 != 0
```
which evaluates to `false`, because `9 % 3 == 0`.


As this chain of instantiations demonstrates:
- We use recursive expansions of `DoIsPrime` to iterate over all divisors 
  from `p / 2` down to `2` to find out whether any of these divisors 
  divide the given integer exactly (i.e., without remainder). 
- The partial specialization of `DoIsPrime` for `d` equal to `2` serves as 
  the criterion to end the recursion. 


Note that all this is done _at compile time_. That is,
```c++
IsPrime<9>::value
```
expands to `false` at compile time.


The template syntax is arguably clumsy, 
but code similar to this has been valid since C++98 (and earlier) 
and has proven useful for quite a few libraries. 


Before C++11, it was common to declare the value members 
as enumerator constants instead of static data members (the "`enum` hack")
to avoid the need to have an out-of-class definition 
of the static data member (see Section 23.6). 
For example:
```c++
// p: number to check, d: current divisor
template <unsigned p, unsigned d> 
struct DoIsPrime
{
    enum
    {
        value = (p % d != 0) && DoIsPrime<p, d - 1>::value
    };
};
```
See Chapter 23 for details. 


#### 📌 8.2 Computing with `constexpr`


C++11 introduced a new feature, `constexpr`, 
that greatly simplifies various forms of compile-time computation. 
In particular, given `constexpr` input, a `constexpr` function can be evaluated at compile time. 
While in C++11 `constexpr` functions were introduced with stringent limitations 
(e.g., each `constexpr` function definition was essentially limited to consist of a return statement), 
most of these restrictions were removed with C++14. 
Of course, successfully evaluating a `constexpr` function still requires that 
all computational steps be possible and valid at compile time: 
Currently, that excludes things like heap allocation or throwing exceptions.


Our example to test whether a number is a prime number could be implemented as follows in C++11: 
```c++
// p: number to check, d: current divisor
constexpr bool doIsPrime(unsigned p, unsigned d)
{
    // end recursion if divisor is 2 
    return d == 2 ? (p % 2 != 0) : (p % d != 0) && doIsPrime(p, d - 1);
};

constexpr bool isPrime(unsigned p)
{
    // handle special cases
    // start recursion with divisor from p / 2
    return (p < 4) ? 1 < p : doIsPrime(p, p / 2);
};
```
Due to the limitation of having only one statement, 
we can only use the conditional operator as a selection mechanism, 
and we still need recursion to iterate over the elements. 
But the syntax is ordinary C++ function code, 
making it more accessible than our first version relying on template instantiation.


With C++14, `constexpr` functions can make use of most control structures available in general C++ code. 
So, instead of writing unwieldy template code or somewhat arcane one-liners, we can now just use a plain for loop:
```c++
constexpr bool isPrime(unsigned p)
{
    for (unsigned d = 2; d <= p / 2; ++d)
    {
        if (p % d == 0)
        {
            return false;
        }
    }
    
    return 1 < p;
}
```
With both the C++11 and C++14 versions of our `constexpr` `isPrime` implementations, 
we can simply call
```c++
// evaluated at compile-time
constexpr bool b = isPrime(9);
```
Note that it can do so at compile time, but it need not necessarily do so. 
In a context that requires a compile-time value (e.g., an array length or a nontype template argument), 
the compiler will attempt to evaluate a call to a `constexpr` function at compile time 
and issue an error if that is not possible (since a constant must be produced in the end).
In contexts that do not need compile-time values,
when the compile-time evaluation fails,
no error is issued and the call is left as a run-time call instead. 
At the time of writing this book, even with `constexpr`, the compiler can decide
to compute the initial value of `b` at run time.


For example:
```c++
// evaluated at compile-time
constexpr bool b1 = isPrime(9);
```
will compute the value at compile time. The same is true with
```c++
// evaluated at compile time if in namespace scope
const bool b2 = isPrime(9); 
```
provided `b2` is defined at namespace scope (including the global namespace). 
At block scope, the compiler can decide whether to compute it at compile or run time. 
This, for example, is also the case here:
```c++
bool fiftySevenIsPrime() 
{
    // evaluated at compile or running time
    return isPrime(57); 
}
```
the compiler may or may not evaluate the call to isPrime at compile time.


On the other hand:
```c++
int x = 9;
std::cout << isPrime(x) << '\n';  // evaluated at run time
```
will generate code that computes at run time whether `x` is a prime number. 


#### 📌 8.3 Execution Path Selection with Partial Specialization


An interesting application of a compile-time test such as `isPrime` 
is to use partial specialization to select at compile time between different implementations. 
For example, we can choose between different implementations depending on 
whether a template argument is a prime number:
```c++
// primary helper template:
template <int SZ, bool = isPrime(SZ)>
struct Helper;

// implementation if SZ is not a prime number:
template <int SZ>
struct Helper<SZ, false>
{
    ...
};

// implementation if SZ is a prime number:
template <int SZ>
struct Helper<SZ, true>
{
    ...
};

template <typename T, std::size_t SZ>
long foo(std::array<T, SZ> const & coll)
{
    // implementation depends on whether array has prime number as size
    Helper<SZ> h; 
    ...
}
```
Here, depending on whether the size of the `std::array` argument is a prime number, 
we use two different implementations of class `Helper`. 
This kind of application of partial specialization is broadly applicable 
to select among different implementations of a function template 
depending on properties of the arguments it's being invoked for.


Above, we used two partial specializations to implement the two possible alternatives. 
Instead, we can also use the primary template for one of the alternatives (the default) case 
and partial specializations for any other special case:
```c++
// primary helper template (used if no specialization fits):
template <int SZ, bool = isPrime(SZ)>
struct Helper
{
    ...
};

// special implementation if SZ is a prime number:
template <int SZ>
struct Helper<SZ, true>
{
    ...
};
```
Because function templates do **not** support partial specialization, 
you have to use other mechanisms to change function implementation based on certain constraints.
Our options include the following:
- Use classes with static functions,
- Use `std::enable_if`, introduced in Section 6.3,
- Use the SFINAE feature, 
- Use the compile-time `if` feature, available since C++17. 


Chapter 20 discusses techniques for selecting a function implementation based on constraints. 


#### 📌 8.4 SFINAE (Substitution Failure Is Not An Error)


When a compiler sees a call to an overloaded function, 
it performs overload resolution. 
It must therefore consider each candidate separately, 
evaluating the arguments of the call and picking the candidate that matches best. 
See also Appendix C for some details about this process. 


In cases where the set of candidates for a call includes function templates, 
the compiler first has to determine what template arguments should be used for that candidate, 
then substitute those arguments in the function parameter list and in its return type, 
and then evaluate how well it matches (just like an ordinary function). 


However, the substitution process could run into problems:
It could produce constructs that make no sense. 
Rather than deciding that such meaningless substitutions lead to errors, 
the language rules instead say that candidates with such substitution problems are simply ignored.


We call this principle SFINAE (pronounced like sfee-nay), 
which stands for "Substitution Failure Is Not An Error". 


Note that the substitution process described here 
is distinct from the on-demand instantiation process 
(see Section 2.2): 
The substitution may be done even for potential instantiations that are not needed 
(so the compiler can evaluate whether indeed they are unneeded). 
It is a substitution of the constructs appearing directly in the declaration of the function (but not its body). 


Consider the following example:
```c++
// number of elements in a raw array:
template <typename T, unsigned N>
std::size_t len(T (&)[N])
{
    return N;
}

// number of elements for a type having size_type:
template <typename T>
typename T::size_type len(T const & t)
{
    return t.size();
}
```
Here, we define two function templates `len` taking one generic argument: 
1. The first function template declares the parameter as `T (&)[N]`, 
   which means that the parameter has to be an array of `N` elements of type `T`.
2. The second function template declares the parameter simply as `T`, 
   which places no constraints on the parameter but returns type `T::size_type`, 
   which requires that the passed argument type has a corresponding member `size_type`.

When passing a raw array or string literals, only the function template for raw arrays matches:
```c++
int a[10];
std::cout << len(a) << '\n';      // OK: only len for array matches
std::cout << len("tmp") << '\n';  // OK: only len for array matches
```
According to its signature, the second function template also matches 
when substituting (respectively) `int [10]` and `char const [4]` for `T`,
but those substitutions lead to potential errors in the return type `T::size_type`.
The second template is therefore ignored for these calls.


When passing a `std::vector`, only the second function template matches:
```c++
std::vector<int> v;
std::cout << len(v) << '\n';  // OK: only len for a type with size_type matches
```
When passing a raw pointer, **neither** of the templates match (**without** a failure). 
As a result, the compiler will complain that no matching `len` function is found:
```c++
int * p;
std::cout << len(p) << '\n';  // ERROR: no matching len function found
```
Note that this differs from passing an object of a type having a `size_type` member, 
but **no** `size` member function, as is, for example, the case for `std::allocator`:
```c++
std::allocator<int> x;
std::cout << len(x) << '\n';  // ERROR: len function found, but can't size
```
When passing an object of such a type, 
the compiler finds the second function template as matching function template. 
So instead of an error that no matching `len` function is found, 
this will result in a compile-time error that calling `size` for a `std::allocator<int>` is invalid. 
This time, the second function template is **not** ignored.


Ignoring a candidate when substituting its return type is meaningless
can cause the compiler to select another candidate whose parameters are a worse match.
For example:
```c++
// number of elements in a raw array:
template <typename T, unsigned N>
std::size_t len(T (&)[N])
{
    return N;
}

// number of elements for a type having size_type:
template <typename T>
typename T::size_type len(T const & t)
{
    return t.size();
}

// fallback for all other types:
std::size_t len(...)
{
    return 0;
}
```
Here, we also provide a general `len` function that always matches but has the worst match 
(match with ellipsis parameter list `(...)`) in overload resolution.
In practice, such a fallback function would usually provide a more useful default, 
throw an exception, or contain a static assertion to result in a useful error message. 
See Section C.2.


So, for raw arrays and vectors, we have two matches where the specific match is the better match. 
For pointers, only the fallback matches so that the compiler no longer complains about a missing `len` for this call. 
But for the allocator, the second and third function templates match, 
with the second function template as the better match.
So, still, this results in an error that no `size` member function can be called: 
```c++
int a[10];
std::cout << len(a) << '\n';      // OK: len for array is best match
std::cout << len("tmp") << '\n';  // OK: len for array is best match

std::vector<int> v;
std::cout << len(v) << '\n';      // OK: len for a type with size_type is best match

int * p;
std::cout << len(p) << '\n';      // OK: only fallback len matches

std::allocator<int> x;
std::cout << len(x) << '\n';      // ERROR: 2nd len function matches best,
                                  // but can't call size for x
```
See Section 15.7 for more details about SFINAE and Section 19.4 about some applications of SFINAE. 

##### SFINAE and Overload Resolution

Over time, the SFINAE principle has become so important and so prevalent among template designers 
that the abbreviation has become a verb. 
We say "we SFINAE out a function" if we mean to apply the SFINAE mechanism to ensure that 
function templates are ignored for certain constraints 
by instrumenting the template code to result in invalid code for these constraints. 
And whenever you read in the C++ standard that a function template 
> shall not participate in overload resolution unless... 

it means that SFINAE is used to "SFINAE out" that function template for certain cases.


For example, class `std::thread` declares a constructor:
```c++
/// <type_traits>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04) 9.4.0
namespace std
{

/// integral_constant
template <typename T, T v>
struct integral_constant
{
    static constexpr T value = v;
    
    typedef T value_type;
    typedef integral_constant<T, v> type;
    
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

template <typename T, T v>
constexpr T integral_constant<T, v>::value;

/// The type used as a compile-time boolean with true value.
typedef integral_constant<bool, true> true_type;

/// The type used as a compile-time boolean with false value.
typedef integral_constant<bool, false> false_type;

template <typename, typename>
struct is_same : public false_type {};

template <typename T>
struct is_same<T, T> : public true_type {};

template <typename P>
struct __not_ : public integral_constant<bool, !bool(P::value)> {};

}  // namespace std

/// <thread>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04) 9.4.0
namespace std 
{

class thread 
{
public:
    template <typename F, 
              typename ... Args,
              typename = enable_if_t<__not_same<F>>>
    explicit thread(F && f, Args && ... args) { /* ... */ }
    
    // ...
    
private:
    template <typename T>
    using __not_same = __not_<is_same<remove_cv_t<remove_reference_t<T>>, thread>>;
    
    // ...
};

}  // namespace std
```
with the following remark:
> Remarks: This constructor shall not participate in overload resolution 
> if decay_t<F> is the same type as std::thread. 


This means that the template constructor is ignored 
if it is called with a `std::thread` as first and only argument. 
The reason is that otherwise a member template like this sometimes might better match 
than any predefined copy or move constructor (see Section 6.2 and Section 16.2.4 for details). 
By SFINAE'ing out the constructor template when called for a thread,
we ensure that the predefined copy or move constructor is always used 
when a thread gets constructed from another thread. 
Since the copy constructor for class `thread` is deleted, 
this also ensures that copying is forbidden.


Applying this technique on a case-by-case basis can be unwieldy. 
Fortunately, the standard library provides tools to disable templates more easily. 
The best-known such feature is `std::enable_if`. 
It allows us to disable a template just by replacing a type with a construct containing the condition to disable it. 


See Section 20.3 for details about how `std::enable_if` is implemented, 
using partial specialization and SFINAE.

##### 8.4.1 Expression SFINAE with `decltype`

It's not always easy to find out and formulate the right expression 
to SFINAE out function templates for certain conditions.


Suppose, for example, that we want to ensure that the function template `len` 
is ignored for arguments of a type that has a `size_type` member but not a `size` member function. 
Without any form of requirements for a `size` member function in the function declaration, 
the function template is selected and its ultimate instantiation then results in an error:
```c++
template <typename T>
typename T::size_type len(T const & t)
{
    return t.size();
}

std::allocator<int> x;
std::cout << len(x) << '\n';  // ERROR: len selected, but x has no size
```
There is a common pattern or idiom to deal with such a situation
(prior to C++20 concepts): 
- Specify the return type with the _trailing return type_ syntax; 
- Define the return type using `decltype` and the comma operator; 
- Formulate all expressions that must be valid 
  at the beginning of the comma operator 
  (converted to `void` in case the comma operator is overloaded). 
- Define an object of the real return type at the end of the comma operator. 
For example: 
```c++
// template <typename T>
// concept Sizeable = requires (T t) 
// { 
//     t.size(); 
//     T::size_type(); 
// };
template <typename T>  // requires Sizeable<T>
auto len(T const & t) -> decltype(static_cast<void>(t.size()), T::size_type())
{
    return t.size();
}
```
The operand of the `decltype` construct is a comma-separated list of expressions, 
so that the last expression `T::size_type()` yields a value of the desired return type 
(which `decltype` uses to convert into the return type). 
Before the (last) comma, we have the expressions that must be valid, 
which in this case is just `t.size()`. 
The cast of the expression to `void` is to avoid 
the possibility of a user-defined comma operator overloaded for the type of the expressions. 
Note that the argument of `decltype` is an _unevaluated_ operand, 
which means that you, for example, can create "dummy objects" **without** calling constructors,
which is discussed in Section 11.2.3.


#### 📌 8.5 [Compile-Time `if`](https://en.cppreference.com/w/cpp/language/if#Constexpr_if)


Partial specialization, SFINAE, and `std::enable_if` 
allow us to enable or disable templates as a whole. 
C++17 additionally introduces a compile-time `if` statement 
that allows is to enable or disable specific statements based on compile-time conditions.


With the syntax `if constexpr`, the compiler uses a compile-time expression 
to decide whether to apply the _then_ part or the _else_ part (if any), 
and _discard_ the opponent. 


As a first example, consider the variadic function template `print` from Section 4.1. 
It prints its arguments (of arbitrary types) using recursion. 
Instead of providing a separate function to end the recursion, 
the `constexpr if` feature allows us to decide locally whether to continue the recursion. 
Although the code reads `if constexpr`, the feature is called `constexpr if`,
because it is the `constexpr` form of `if` (and for historical reasons). 
```c++
template <typename T>
void print(T && t)
{
    std::cout << std::forward<T>(t) << '\n';
}

template <typename T, typename ... Args>
void print(T && t, Args && ... args)
{
    std::cout << std::forward<T>(t) << ' ';
    print(std::forward<Args>(args)...);
}
```
```c++
template <typename T, typename ... Args>
void print(T && t, Args && ... args)
{
    std::cout << std::forward<T>(t) << '\n';
    
    if constexpr (0 < sizeof...(args)) 
    {
        // code only available if 0 < sizeof...(args) (since C++17)
        print(std::forward<Args>(args)...); 
    }
}
```
Here, if `print` is called for one argument only, 
`args` becomes an empty parameter pack so that `sizeof...(args)` becomes `0`. 
As a result, the recursive call of `print` becomes a _discarded statement_, 
for which the code is not instantiated.
Thus, a corresponding function is not required to exist and the recursion ends. 


The fact that the code is not instantiated means that only the first translation phase
(the definition time) is performed, 
which checks for correct syntax and names that don't depend on template parameters. 
For example:
```c++
template <typename T>
void foo(T t)
{
    if constexpr (std::is_integral_v<T>) 
    {
        if (t > 0) 
        {
            foo(t - 1); // OK
        }
    }
    else 
    {
        // error if not declared and not discarded (i.e. T is not integral)
        undeclared(t); 
        
        // error if not declared (even if discarded)
        undeclared(); 
        
        // always asserts (even if discarded)
        static_assert(false, "no integral");      
        
        // OK
        static_assert(!std::is_integral_v<T>, "no integral");  
    }
}
```
Note that `if constexpr` can be used in any function, not only in templates.
We only need a compile-time expression that yields a Boolean value. 
For example:
```c++
int main()
{
    if constexpr (std::numeric_limits<char>::is_signed 
    {
        foo(42); // OK
    }
    else 
    {
        // error if undeclared() not declared
        undeclared(42); 
        
        // always asserts (even if discarded)
        static_assert(false, "unsigned"); 
        
        // OK
        static_assert(!std::numeric_limits<char>::is_signed, "char is unsigned"); 
    }
}
```
With this feature, we can, for example, use our `isPrime` compile-time function,
introduced in Section 8.2, 
to perform additional code if a given size is not a prime number:
```c++
template <typename T, std::size_t SZ>
void foo(std::array<T, SZ> const & coll)
{
    if constexpr (!isPrime(SZ)) 
    {
        // special additional handling 
        // if the passed array has no prime number as size
    }
    
    ...
}
```
See Section 14.6 for further details.


#### 📌 8.6 Summary


- Templates provide the ability to compute at compile time 
  (using recursion to iterate and partial specialization or `operator?:` for selections).
- With `constexpr` functions, we can replace most compile-time computations 
  with "ordinary functions" that are callable in compile-time contexts. 
- With partial specialization, we can choose between different implementations of class templates
  based on certain compile-time constraints.
- Templates are used only if needed and 
  substitutions in function template declarations do **not** result in invalid code. 
  This principle is called SFINAE (Substitution Failure Is Not An Error). 
- SFINAE can be used to provide function templates only for certain types and/or constraints.
- Since C++17, a compile-time `if` allows us to enable or discard statements
  according to compile-time conditions (even outside templates). 






### 🎯 Chapter 9 Using Templates in Practice


#### 📌 9.1 The Inclusion Model


There are several ways to organize template source code. 
This section presents the most popular approach: the _inclusion model_.

##### 9.1.1 Linker Errors

Most C and C++ programmers organize their non-template code largely as follows:
- Classes and other types are entirely placed in _header files_. 
  Typically, this is a file with a `.hpp` (or `.H`, `.h`, `.hh`, `.hxx`) filename extension. 
- For global (non-inline) variables and (non-inline) functions, 
  only a declaration is put in a header file, 
  and the definition goes into a file compiled as its own translation unit. 
  Such a _CPP file_ typically is a file with a `.cpp` (or `.C`, `.c`, `.cc`, or `.cxx`) filename extension.


This works well: 
It makes the needed type definition easily available throughout the program 
and avoids duplicate definition errors on variables and functions from the linker.


With these conventions in mind, a common error about which beginning template programmers complain 
is illustrated by the following (erroneous) little program. 
As usual for "ordinary code":
```c++
/// "myfirst.hpp"
#ifndef MYFIRST_HPP
#define MYFIRST_HPP
// declaration of template
template <typename T>
void printTypeof(T const &);
#endif  // MYFIRST_HPP
```
```c++
/// "myfirst.cpp"
#include <iostream>
#include <typeinfo>
#include "myfirst.hpp"
// implementation/definition of template
template <typename T>
void printTypeof(T const& x)
{
    std::cout << typeid(x).name() << '\n';
}
```
```c++
/// "myfirstmain.cpp"
#include "myfirst.hpp"

// use of the template
int main(int argc, char * argv[])
{
    double ice = 3.0;
    printTypeof(ice);  // call function template for type double
}
```
A C++ compiler will most likely accept this program without any problems, 
but the linker will probably report an error, 
implying that there is **no** definition of the function `printTypeof`. 


The reason for this error is that the definition 
of the function template `printTypeof` has **not** been instantiated. 
In order for a template to be instantiated, 
the compiler must know which definition should be instantiated 
and for what template arguments it should be instantiated. 
Unfortunately, in the previous example, these two pieces of information
are in files that are compiled separately.
Therefore, when our compiler sees the call to `printTypeof` 
but has **no** definition in sight to instantiate this function for `double`, 
it just assumes that such a definition is provided elsewhere
and creates a reference (for the linker to resolve) to that definition. 
On the other hand, when the compiler processes the file `myfirst.cpp`, 
it has **no** indication at that point that it must instantiate 
the template definition it contains for specific arguments.

##### 9.1.2 Templates in Header Files

The common solution to the previous problem 
is to use the same approach that we would take with macros or with inline functions: 
We include the definitions of a template in the header file that declares that template.


That is, instead of providing a file `myfirst.cpp`, we rewrite `myfirst.hpp`
so that it contains all template declarations and template definitions:
```c++
/// "myfirst.hpp"
#ifndef MYFIRST_HPP
#define MYFIRST_HPP

#include <iostream>
#include <typeinfo>

// declaration of template
template <typename T>
void printTypeof(T const &);

// implementation/definition of template
template <typename T>
void printTypeof(T const & x)
{
    std::cout << typeid(x).name() << '\n'; 
}

#endif  // MYFIRST_HPP
```
This way of organizing templates is called the _inclusion model_. 
With this in place, you should find that 
our program now correctly compiles, links, and executes.


There are a few observations we can make at this point. 
The most notable is that this approach has considerably increased 
the cost of including the header file `myfirst.hpp`. 
In this example, the cost is not only the result of the size of the template definition itself, 
but also the result of the fact that we must also include the headers 
used by the definition of our template, 
in this case, `<iostream>` and `<typeinfo>`. 
You may find that this amounts to tens of thousands of lines of code 
because headers like `<iostream> `contain many template definitions of their own. 


This is a real problem in practice because it considerably increases 
the time needed by the compiler to compile significant programs.
We will therefore examine some possible ways to approach this problem, 
including _precompiled headers_ (see Section 9.3) 
and the use of _explicit template instantiation_ (see Section 14.5).


Despite this build-time issue, 
we do recommend following this inclusion model 
to organize your templates when possible 
until a better mechanism becomes available. 
As of 2022, such a mechanism is _modules_ `(since C++20)`, 
but still faces serious cross-platform issues 
(modules compiled by `gcc`, `clang` and `MSVC` are **not** interchangeable), etc. 
This is introduced in Section 17.11.
They are a language mechanism that allows the programmer to more logically organize code 
in such a way that a compiler can separately compile all declarations 
and then efficiently and selectively import the processed declarations whenever needed. 


Another (more subtle) observation about the inclusion approach is that 
non-inline function templates are distinct from inline functions and macros in an important way: 
They are **not** expanded at the call site. 
Instead, when they are instantiated, they create a new copy of a function. 
Because this is an automatic process, a compiler could end up creating two copies in two different files, 
and some linkers could issue errors when they find two distinct definitions for the same function. 
In theory, this should **not** be a concern of ours: 
It is a problem for the C++ compilation system to accommodate. 
In practice, things work well most of the time, and we don't need to deal with this issue at all. 
For large projects that create their own library of code, however, problems occasionally show up. 
A discussion of instantiation schemes in Chapter 14 
and a close study of the documentation that came with the C++ translation system (compiler) 
should help address these problems. 


Finally, we need to point out that what applies to the ordinary function template in our example 
also applies to member functions and static data members of class templates, 
as well as to member function templates.


#### 📌 9.2 Templates and `inline`


Declaring functions to be inline is a common tool to improve the running time of programs. 
The `inline` specifier was meant to be a hint for the implementation that
inline substitution of the function body at the point of call is preferred
over the usual function call mechanism.


However, an implementation may ignore the hint. 
Hence, the only guaranteed effect of `inline` is to 
allow a function definition to appear multiple times in a program 
(usually because it appears in a header file that is included in multiple places). 


Like inline functions, function templates can be defined in multiple translation units. 
This is usually achieved by placing the definition in a header file that is included by multiple CPP files.


This **doesn't** mean, however, that function templates use inline substitutions by default. 
It is entirely up to the compiler whether and when inline substitution of a function template body 
at the point of call is preferred over the usual function call mechanism. 
Perhaps surprisingly, compilers are often better than programmers at 
estimating whether inlining a call would lead to a net performance improvement. 
As a result, the precise policy of a compiler with respect to inline varies from compiler to compiler, 
and even depends on the options selected for a specific compilation.


Nevertheless, with appropriate performance monitoring tools, 
a programmer may have better information than a compiler 
and may therefore wish to override compiler decisions 
(e.g., when tuning software for particular platforms, 
such as mobiles phones, or particular inputs). 
Sometimes this is only possible with compiler-specific attributes 
such as `gcc`'s `__always_inline` derivative (available in `<cdefs.h>`). 


It's worth pointing out at this point that 
full specializations of function templates
act like ordinary functions in this regard: 
Their definition can appear only once unless they're defined `inline` (see Section 16.3). 
See also Appendix A for a broader, detailed overview of this topic.


#### 📌 9.3 Precompiled Headers


Even without templates, C++ header files can become very large and therefore take a long time to compile. 
Templates add to this tendency, and the outcry of waiting programmers has in many cases driven vendors 
to implement a scheme usually known as _precompiled headers (PCH)_. 
This scheme operates outside the scope of the standard and relies on vendor-specific options. 
Although we leave the details on how to create and use precompiled header files
to the documentation of the various C++ compilation systems that have this feature, 
it is useful to gain some understanding of how it works.


When a compiler translates a file, it does so 
starting from the beginning of the file and working through to the end. 
As it processes each token from the file (which may come from `#included` files), 
it adapts its internal state, 
including such things as adding entries to a table of symbols 
so that they may be looked up later. 
While doing so, the compiler may also generate code in object files.


The precompiled header scheme relies on the fact 
that code can be organized in such a manner that 
many files start with the same lines of code. 
Let's assume for the sake of argument 
that every file to be compiled starts with the same `N` lines of code.
We could compile these `N` lines 
and save the complete state of the compiler at that point in a _precompiled header_. 
Then, for every file in our program, we could reload the saved state 
and start compilation at line `N + 1`. 
At this point it is worthwhile to note that reloading the saved state 
is an operation that can be orders of magnitude faster than actually compiling the first `N` lines. 
However, saving the state in the first place is typically more expensive than just compiling the `N` lines. 
The increase in cost varies roughly from 20% to 200%.


The key to making effective use of precompiled headers is to ensure that
as much as possible  files start with a maximum number of common lines of code. 
In practice this means the files must start with the same `#include` directives, 
which (as mentioned earlier) consume a substantial portion of our build time. 
Hence, it can be very advantageous to pay attention to the order in which headers are included. 
For example, the following two files:
```c++
#include <vector>
#include <list>
```
```c++
#include <list>
#include <vector>
```
inhibit the use of precompiled headers because there is **no** common initial state in the sources.


Some programmers decide that it is better to #include some extra unnecessary headers 
than to pass on an opportunity to accelerate the translation of a file using a precompiled header. 
This decision can considerably ease the management of the inclusion policy. 


For example, `libstdc++` provides `<bits/stdc++.h>` that includes all the standard headers.
(Note this is an extreme example. 
Actual programs do **not** need a common header containing all STL headers!
Also, in theory, the standard headers do **not** actually need to correspond to physical files.
In practice, however, they do, and the files are very large.) 


This file can then be precompiled, and every program file that 
makes use of the standard library can then simply be started as follows:
```c++
#include <bits/stdc++.h>
```
Normally this would take a while to compile,
but given a system with sufficient memory, 
the pre-compiled header scheme allows it to be processed 
significantly faster than almost any single standard header would require without pre-compilation. 
The standard headers are particularly convenient in this way because they rarely change, 
and hence the precompiled header for `<bits/stdc++.h>` can be built once. 
Otherwise, precompiled headers are typically part of the dependency configuration of a project 
(e.g., they are updated as needed by the popular `make` tool 
or an integrated development environment's (IDE) project build tool).


One attractive approach to manage precompiled headers 
is to create layers of precompiled headers 
that go from the most widely used and stable headers 
to headers that aren't expected to change all the time and therefore are still worth precompiling. 
However, if headers are under heavy development, 
creating precompiled headers for them can take more time than what is saved by reusing them. 
A key concept to this approach is that a precompiled header for a more stable layer 
can be reused to improve the precompilation time of a less stable header. 


#### 📌 9.4 Decoding the Error Novel


Ordinary compilation errors are normally quite succinct and to the point. 
For example, when a compiler says "`error: ‘class X' has no member named ‘fun'`", 
it usually isn't too hard to figure out what is wrong in our code 
(e.g., we might have mistyped `run` as `fun`). 
Not so with templates. 
Try some examples.

##### Simple Type Mismatch

```c++
std::map<std::string, double> coll;

auto pos = std::find_if(coll.begin(), coll.end(), [](std::string const & s)
{
    return s != "";
});
```

##### Missing `const` on Some Compilers

```c++
class Customer
{
public:
    Customer(std::string const & n) : name(n) {}

    std::string getName() const
    {
        return name;
    }

private:
    std::string name;
};

struct MyCustomerHash
{
    // NOTE: missing const
    // in std::hash template argument
    // is only an error with g++ and clang:
    std::size_t operator()(Customer const & c)
    {
        return std::hash<std::string>()(c.getName());
    }
};

std::unordered_set<Customer, MyCustomerHash> coll;
```


#### 📌 9.5 Afternotes


The organization of source code in header files and CPP files is a practical consequence 
of various incarnations of the _One-Definition Rule (ODR)_. 
An extensive discussion of this rule is presented in Appendix A. 


The inclusion model is a pragmatic answer dictated largely 
by existing practice in C++ compiler implementations. 
However, the first C++ implementation was different: 
The inclusion of template definitions was implicit,
which created a certain illusion of separation 
(see Chapter 14 for details on this original model).


The first C++ standard (C++98) provided explicit support for the separation model 
of template compilation via _exported templates_. 
The separation model allowed template declarations marked as `export` to be declared in headers, 
while their corresponding definitions were placed in CPP files, 
much like declarations and definitions for non-template code. 
Unlike the inclusion model, this model was a theoretical model 
not based on any existing implementation, 
and the implementation itself proved far more complicated 
than the C++ standardization committee had anticipated. 
It took more than five years to see its first implementation published (May 2002), 
and no other implementations appeared in the years since. 
To better align the C++ standard with existing practice,
the C++ standardization committee removed exported templates from C++11. 


It is sometimes tempting to imagine ways of extending the concept of precompiled headers 
so that more than one header could be loaded for a single compilation. 
This would in principle allow for a finer grained approach to pre-compilation. 
The obstacle here is mainly the preprocessor: 
Macros in one header file can entirely change the meaning of subsequent header files. 
However, once a file has been precompiled, macro processing is completed, 
and it is hardly practical to attempt to patch a precompiled header 
for the preprocessor effects induced by other headers. 
Since C++20, _modules_ (see Section 17.11) are available. 
Macro definitions can **not** leak into module interfaces. 


#### 📌 9.6 Summary


- The inclusion model of templates is the most widely used model for organizing template code. 
  Alternatives are discussed in Chapter 14. 
- Only full specializations of function templates need `inline` 
  when defined in header files outside classes or structures. 
- To take advantage of precompiled headers, 
  be sure to keep the same order for `#include` directives. 
- Learn to pick out the most important part from a clumsy compiler error message. 






### 🎯 Chapter 10 Basic Template Terminology


#### 📌 10.1 "Class Template" or "Template Class"?


#### 📌 10.2 Substitution, Instantiation, and Specialization


When processing source code that uses templates, 
a C++ compiler must at various times substitute concrete template arguments 
for the template parameters in the template. 
Sometimes, this substitution is just tentative: 
The compiler may need to check if the substitution could be valid 
(see Section 8.4 and Section 15.7).


The process of actually creating a definition for 
a regular class, type alias, function, member function, or variable from a template 
by substituting concrete arguments for the template parameters 
is called _template instantiation_. 


Surprisingly, there is currently no standard or generally agreed upon term 
to denote the process of creating a declaration that is not a definition 
through template parameter substitution. 
We have seen the phrases _partial instantiation_ or _instantiation of a declaration_ used by some teams, 
but those are by no means universal. 
Perhaps a more intuitive term is _incomplete instantiation_ 
(which, in the case of a class template, produces an incomplete class). 


The entity resulting from an instantiation or an incomplete instantiation 
(i.e., a class, function, member function, or variable) 
is generically called a _specialization_.


However, in C++ the instantiation process is **not** the only way to produce a specialization. 
Alternative mechanisms allow the programmer to specify explicitly a declaration 
that is tied to a special substitution of template parameters. 
As we showed in Section 2.5, such a specialization is introduced with the prefix `template<>`:
```c++
// primary class template
template <typename T1, typename T2>
class MyClass 
{
    ...
};

// explicit specialization
template <> 
class MyClass<std::string, float> 
{
    ...
};
```
Strictly speaking, this is called an _explicit specialization_ 
(as opposed to an _instantiated_ or _generated specialization_). 


As described in Section 2.6, 
specializations that still have template parameters 
are called _partial specializations_:
```c++
// partial specification
template <typename T> 
class MyClass<T, T>
{
    ...
};

// partial specification
template <typename T>
class MyClass<bool, T>
{
    ...
};
```
When talking about (explicit or partial) specializations, 
the general template is also called the _primary template_. 


#### 📌 10.3 [Declarations versus Definitions](https://en.cppreference.com/w/cpp/language/definition)


So far, the words _declaration_ and _definition_ have been used only a few times in this book. 
However, these words carry with them a rather precise meaning in standard C++, 
and that is the meaning that we use. 


From [cppreference](https://en.cppreference.com/w/cpp/language/definition): 
> Declarations introduce (or re-introduce) names into the C++ program.   
> Each kind of entity is declared differently.   
> Definitions are declarations that fully define the entity introduced by the declaration.   
> Every declaration is a definition, except for several special cases. 


A _declaration_ is a C++ construct that introduces or reintroduces a name into a C++ scope. 
This introduction always includes a partial classification of that name,
but the details are **not** required to make a valid declaration.
Note that even though they have a "name", 
macro definitions and `goto` labels are **not** considered declarations in C++. 
```c++
class C;        // a declaration of C as a class
void f(int p);  // a declaration of f() as a function and p as a named parameter
extern int v;   // a declaration of v as a variable
```

Declarations become definitions when the details of their structure are made known 
or, in the case of variables, when storage space must be allocated. 
- For **class type definitions**, 
  this means a brace-enclosed body must be provided; 
- For **function definitions**, 
  this means a brace-enclosed body must be provided (in the common case), 
  or the function must be designated as `= default` or `= delete`; 
- For **variable definitions**, 
  initialization or the absence of an `extern` specifier 
  causes a declaration to become a definition. 


Here are examples that complement the preceding non-definition declarations:
```c++
// definition (and declaration) of class C
class C {}; 

// definition (and declaration) of function f
void f() {}

// an initializer makes this a definition for int v
extern int v = 1; 

// global variable declarations not preceded by extern
// are also definitions
int w; 
```
By extension, the declaration of a class template or function template
is called a definition if it has a body. Hence,
```c++
template <typename T>
void func(T);
```
is a declaration that is **not** a definition, whereas
```c++
template <typename T>
class S {};
```
is in fact a definition. 

##### 10.3.1 Complete versus Incomplete Types

Types can be _complete_ or _incomplete_, 
which is a notion closely related to the distinction 
between a _declaration_ and a _definition_. 
Some language constructs require _complete types_, 
whereas others are valid with _incomplete types_ too. 


Incomplete types are one of the following: 
- A class type (declaration) that has been declared but not yet defined. 
- An array type (declaration) with an unspecified bound. 
- An array type (declaration) with an incomplete element type. 
- `void`
- An enumeration type (declaration) as long as the underlying type or the enumeration values are not defined.
- Any type above to which `const` and/or `volatile` are applied. 
- All other types are _complete_. 


Valid operations on incomplete types: 
[HERE](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/cpp_primer_notes.md#%E7%B1%BB%E7%9A%84%E5%89%8D%E5%90%91%E5%A3%B0%E6%98%8E)


For example:
```c++
class C;             // C is an incomplete type
C const * cp;        // cp is a pointer to an incomplete type
extern C elems[10];  // elems has an incomplete type
extern int arr[];    // arr has an incomplete type

... 

class C {};          // C now is a complete type 
                     // (and therefore cp and elems no longer refer to an incomplete type)
int arr[10];         // arr now has a complete type
```
See Section 11.5 for hints about how to deal with incomplete types in templates. 


#### 📌 10.4 [The One-Definition Rule](https://en.cppreference.com/w/cpp/language/definition#One_Definition_Rule)


The C++ language definition places some constraints on the redeclaration of various entities. 
The totality of these constraints is known as the _One-Definition Rule (ODR)_. 
The details of this rule are a little complex and span a large variety of situations. 
Later chapters illustrate the various resulting facets in each applicable context,
and you can find a complete description of the ODR in Appendix A. 


For now, it suffices to remember the following ODR basics:
- Ordinary (i.e., not templates) non-inline functions and member functions, 
  as well as (non-inline) global variables and static data members 
  should be defined only once across the whole _program_.
  Global and static variables and data members can be defined as `inline` since C++17. 
  This removes the requirement that they be defined in exactly one translation unit.
- Class types (including structs and unions), 
  templates (including partial specializations but **not** full specializations), 
  and inline functions and variables should be defined at most once 
  _per translation unit_, 
  and all these definitions should be identical. 


A _translation unit_ is what results from preprocessing a source file; 
that is, it includes the contents named by `#include` directives and produced by macro expansions. 


In the remainder of this book, _linkable entity_ refers to any of the following: 
- A function or member function;
- A global variable; 
- A static data member, including any such things generated from a template, as visible to the linker.


#### 📌 10.5 Template Arguments versus Template Parameters


```c++
template <typename T, int N>
class ArrayInClass 
{
public:
    T array[N];
};

class DoubleArrayInClass 
{
public:
    double array[10];
};
```
The latter becomes essentially equivalent to the former 
if we replace the parameters `T` and `N` by `double` and `10` respectively. 
In C++, the name of this replacement is denoted as `ArrayInClass<double, 10>`. 


Regardless of whether these arguments are themselves dependent on template parameters,
the combination of the template name, followed by the arguments in angle brackets, 
is called a _template-id_.
This name can be used much like a corresponding non-template entity would be used: 
```c++
int main()
{
    ArrayInClass<double, 10> ad; 
    ad.array[0] = 1.0;
}
```
It is essential to distinguish between template parameters and template arguments. 
In short, you can say that "parameters are initialized by arguments".
In the academic world, 
arguments are sometimes called _actual parameters_,
whereas parameters are called _formal parameters_. 


Or more precisely: 
- _Template parameters_ are those names that are listed 
  after the keyword `template` in the template declaration or definition 
  (`T` and `N` in our example).
- _Template arguments_ are the items that are substituted for template parameters
  (`double` and `10` in our example). 
  Unlike template parameters, template arguments can be more than just "names". 


The substitution of template parameters by template arguments is explicit when indicated with a template-id, 
but there are various situations when the substitution is implicit 
(e.g., if template parameters are substituted by their default arguments). 


A fundamental principle is that 
any template argument must be a `constexpr` quantity or value 
that can be determined at compile time. 
As becomes clear later, this requirement translates into dramatic benefits 
for the run-time costs of template entities. 
Because template parameters are eventually substituted by compile-time values, 
they can themselves be used to form compile-time expressions. 
This was exploited in the `ArrayInClass` template to size the member array `array`. 
The size of an array must be a _constant-expression_, 
and the template parameter `N` qualifies as such. 


We can push this reasoning a little further: 
Because template parameters are compile-time entities, 
they can also be used to create valid template arguments. 
Here is an example:
```c++
template <typename T>
class Dozen 
{
public:
    ArrayInClass<T, 12> contents;
};
```
Note how in this example the name `T` is both a template parameter and a template argument. 
Thus, a mechanism is available to enable the construction of more complex templates from simpler ones. 
Of course, this is not fundamentally different from the mechanisms that allow us to assemble types and functions.


#### 📌 10.6 Summary


- _Template instantiation_ is the process of creating regular classes or functions 
  by replacing template _parameters_ with concrete _arguments_. 
  The resulting entity is a _specialization_.
- Types can be complete or incomplete.
- According to the One-Definition Rule (ODR), 
  non-inline functions, member functions, global variables, and static data members 
  should be defined only once across the whole program.






### 🎯 Chapter 11 Generic Libraries


#### 📌 11.1 Callables


_Callback_ refers to entities that are passed as function call arguments 
(as opposed to, e.g., template arguments). 
For example, a sort function may include a callback parameter as "sorting criterion", 
which is called to determine whether one element precedes another in the desired sorted order. 


In C++, there are several types that work well for _callbacks_ 
because they can both be passed as function call arguments 
and can be directly called with `operator()`:
- Pointer-to-function types;
- Class types with an overloaded `operator()` (sometimes called _functors_), including lambdas;
- Class types with a conversion function 
  yielding a pointer-to-function or reference-to-function. 


Collectively, these types are called _function object types_, 
and a value of such a type is a _function object_. 


The C++ standard library introduces the slightly broader notion of a _callable type_,
which is either a function object type or a pointer to member. 
An object of callable type is a _callable object_, 
which we refer to as a _callable_ for convenience. 


Generic code often benefits from being able to accept any kind of callable, 
and templates make it possible to do so. 

##### 11.1.1 Supporting Function Objects

```c++
template <typename InputIterator, typename Function>
Function foreach(InputIterator first, InputIterator last, Function f)
{
    for (; first != last; ++first) f(*first);
    return f;
}

void func(int & i) {}

class FunctionObject
{
public:
    void operator()(int & i) {}
};

std::vector<int> primes {2, 3, 5, 7, 11, 13, 17, 19};

// function as callable (decays to pointer)
foreach(primes.begin(), primes.end(), func); 

// function pointer as callable
foreach(primes.begin(), primes.end(), &func); 

// function object as callable
foreach(primes.begin(), primes.end(), FuncObj()); 

// lambda as callable
foreach(primes.begin(), primes.end(), [](int & i) {});
```
Let's look at each case in detail: 
- When we pass the name of a function as a function argument, 
  we **don't** really pass the function itself but a pointer or reference to it. 
  As with arrays (see Section 7.4), function arguments _decay_ to a pointer when passed by value, 
  and in the case of a parameter whose type is a template parameter, 
  a pointer-to-function type will be deduced.   

  Just like arrays, functions can be passed by reference without decay. 
  However, function types can **not** really be qualified with `const`. 
  If we were to declare the last parameter of `foreach` with type `Callable const &`, 
  the `const` would just _be ignored_. 
  (Generally speaking, references to functions are rarely used in mainstream C++ code.)
- Our second call explicitly takes a function pointer 
  by passing the address of a function name. 
  This is equivalent to the first call 
  (where the function name implicitly decayed to a pointer value)
  but is perhaps a little clearer. 
- When passing a functor, we pass a class type object as a callable. 
  Calling through a class type usually amounts to invoking its `operator()`. 
  So the call
  ```c++
  f(*first);
  ```
  is usually transformed into
  ```c++
  f.operator()(*first);
  ```
  Note that when defining `operator()`, 
  you should usually define it as a _constant member function_. 
  Otherwise, subtle error messages can occur when frameworks or libraries 
  expect this call not to change the state of the passed object.    
  It is also possible for a class type object to be implicitly convertible 
  to a pointer or reference to a _surrogate call function_ (discussed in Section C.3.5).
  In such a case, the call
  ```c++
  f(*first);
  ```
  would be transformed into
  ```c++
  (f.operator F())(*first);
  ```
  where `F` is the type of the pointer-to-function or reference-to-function 
  that the class type object can be converted to. 
  This is relatively unusual.
- Lambda expressions produce functors (called closures), 
  and therefore this case is not different from the functor case. 
  Lambdas are, however, a very convenient shorthand notation to introduce functors, 
  and so they appear commonly in C++ code since C++11.   
 
  The compiler-generated closure type for lambdas with **no** captures 
  provides a conversion operator to a function pointer. 
  However, that is **never** selected as a _surrogate call function_ 
  because it is always a worse match than the normal `operator()` of the closure. 

##### 11.1.2 Dealing with Member Functions and Additional Arguments

One possible entity to call was not used in the previous example: 
member functions.
That's because calling a non-static member function normally involves 
specifying an object to which the call is applied 
using syntax like `object.memfunc(...)` or `ptr->memfunc(...)` 
and that doesn't match the usual pattern `functionObject(...)`. 


Fortunately, since C++17, the C++ standard library provides a utility 
[`std::invoke`](https://en.cppreference.com/w/cpp/utility/functional/invoke) 
in header `<functional>`
that conveniently unifies this case with the ordinary function-call syntax cases, 
thereby enabling calls to any callable object with a single form. 
The following implementation of our `foreach` template uses `std::invoke`:
```c++
template <typename InputIterator, typename Function, typename ... Args>
Function foreach(InputIterator first, InputIterator last, Function f, Args const & ... args)
{
    for (; first != last; ++first)
    {
        // Do NOT perfect-forward f and args,
        // as the first call might steal their values!
        std::invoke(f, args..., *first);
    }

    return f;
}
```
Here, besides the callable parameter, we also accept an arbitrary number of additional parameters. 
The `foreach` template then calls `std::invoke` with the given callable 
followed by the additional given parameters along with the referenced element. 
`std::invoke(f, a1, a2, ...)` works roughly (details emitted!) follows: 
- If `f` is **pointer to member function**, 
  is equivalent to something like `a1.*f(a2, ...)`, 
  with special handling on reference wrappers, etc., to make the call valid;
- If `f` is **pointer to data member** and only `a1` is passed, 
  is equivalent to something like `a1.*f`, 
  with special handling on reference wrappers, etc., to make the call valid;
- **Otherwise**, is equivalent to `f(a1, a2, ...)`


Note that we **can't** use perfect forwarding here for the callable or additional parameters: 
The first call might "steal" their values,
leading to unexpected behavior calling `f` in subsequent iterations. 


With this implementation, we can still compile our original calls to `foreach` above. 
Now, in addition, we can also pass additional arguments to the callable 
and the callable can be a member function.


The following client code illustrates this:
```c++
// a class with a member function that shall be called
class MyClass
{
public:
    void memfunc(int i) const
    {
        std::cout << "MyClass::memfunc(" << i << ")\n";
    }
};

std::vector<int> primes {2, 3, 5, 7, 11, 13, 17, 19};

// pass lambda as callable and an additional argument:
foreach(primes.begin(), 
        primes.end(),
        [](std::string const & prefix, int i)
        {
            std::cout << prefix << i << '\n';
        },
        "- value: ");

// call obj.memfunc for/with each element in primes passed as argument
MyClass obj;
foreach(primes.begin(), primes.end(), &MyClass::memfunc, obj);
```

##### 11.1.3 Wrapping Function Calls


A common application of `std::invoke` is to wrap single function calls 
(e.g., to log the calls, measure their duration, or prepare some context such as starting a new thread for them). 
Now, we can support move semantics by perfect forwarding both the callable and all passed arguments: 
```c++
template <typename Callable, typename ... Args>
decltype(auto) call(Callable && op, Args && ... args)
{
    return std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...); 
}
```
The other interesting aspect is 
how to deal with the return value of a called function 
to "perfectly forward" it back to the caller. 
To support returning references (such as a `std::ostream &`) 
you have to use `decltype(auto)` instead of just `auto`. 


If you want to temporarily store the value returned by `std::invoke` 
in a variable to return it after doing something else,
(i.e., some post-processing that can be done only after `std::invoke` is called), 
you also have to declare the temporary variable with `decltype(auto)`:
```c++
decltype(auto) ret {std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...)};
... 
return ret;
```
Note that declaring `ret` with `auto &&` is **incorrect**. 
As a reference, `auto &&` extends the lifetime of the returned value 
until the end of its scope (see Section 11.3) 
but **not** beyond the return statement to the caller of the function. 


However, there is also a problem with using `decltype(auto)`: 
If the callable has return type `void`, 
the initialization of `ret` as `decltype(auto)` is **not** allowed, 
because `void` is an incomplete type. 
You have the following options:
• Declare an object in the line before that statement, 
  whose destructor performs the observable behavior that you want to realize: 
  ```c++
  struct cleanup 
  {
      ~cleanup() 
      {
          // code to perform after 
          // std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...)
      }
  } dummy;
  
  return std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...);
  ```
• Implement the `void` and non-`void` cases differently:
```c++
template <typename Callable, typename ... Args>
decltype(auto) call(Callable && op, Args && ... args)
{
    if constexpr(std::is_same_v<std::invoke_result_t<Callable, Args...>, void>)
    {
        // return type is void:
        std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...);
        // cleanup code here
        return;
    }
    else
    {
        // return type is not void:
        decltype(auto) ret {std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...)};
        // cleanup code here
        return ret;
    }
}
```


#### 📌 11.2 Other Utilities to Implement Generic Libraries

##### 11.2.1 [Type Traits](https://en.cppreference.com/w/cpp/header/type_traits)

##### 11.2.2 [`std::addressof`](https://en.cppreference.com/w/cpp/memory/addressof)

The `std::addressof` function template yields the actual address of an object or function. 
It works even if the object type has an overloaded `operator&`. 
Even though the latter is somewhat rare, it might happen (e.g., in smart pointers). 
Thus, it is recommended to use `addressof` if you need an address of an object of arbitrary type:
```c++
template <typename T>
void f (T && x)
{
    auto p = &x;                 // might fail with overloaded operator&
    auto q = std::addressof(x);  // works even with overloaded operator&

}
```

##### 11.2.3 [`std::declval`](https://en.cppreference.com/w/cpp/utility/declval)

```c++
template <typename T>
typename std::add_rvalue_reference<T>::type declval() noexcept;
```
The `std::declval` function template can be used as a placeholder 
for an object reference of a specific type.
It converts any type `T` to `T &&` 
(possibly cv-qualified, except for `void`, which remains unchanged), 
making it possible to use member functions in `decltype` expressions 
**without** the need to go through constructors.


`declval` is commonly used in templates 
where acceptable template parameters may have **no** constructor in common, 
but have the same member function whose return type is needed. 


The function **doesn't** have a definition and therefore can **not** be called (and **doesn't** create an object). 
Hence, it can only be used in 
[unevaluated expressions](https://en.cppreference.com/w/cpp/language/expressions#Unevaluated_expressions) 
(such as `typeid`, `sizeof`, `noexcept` and `decltype` constructs). 
So, instead of trying to create an object, you can assume you have an object of the corresponding type. 


For example, the following declaration deduces the default return type `R` 
from the passed template parameters `T1` and `T2`:
```c++
template <typename T1, 
          typename T2,
          typename R = std::decay_t<decltype(true ? std::declval<T1>() : std::declval<T2>())>>
R max(T1 a, T2 b)
{
    return b < a ? a : b;
}
```
To avoid that we have to call a (default) constructor for `T1` and `T2` 
to be able to call operator `?:` in the expression to initialize `R`, 
we use `std::declval` to "use" objects of the corresponding type **without** creating them. 
This is only possible in the _unevaluated context_ of `decltype`, though.


Don't forget to use the `std::decay` type trait to ensure 
the default return type can't be a reference, 
because `std::declval` itself yields rvalue references. 
Otherwise, calls such as `max(1, 2)` will get a return type of `int &&`.
See Section 19.3.4 for details.


#### 📌 11.3 Perfect Forwarding Temporaries


We can use _forwarding references_ and `std::forward` 
to "perfectly forward" generic parameters:
```c++
template <typename ... Args>
void f(Args && ... args)
{
    g(std::forward<Args>(args)...);
}
```
However, sometimes we have to perfectly forward data 
in generic code that does **not** come through a parameter. 
In that case, we can use `auto &&` to create a variable that can be forwarded. 
Assume, for example, that we have chained calls to functions `get` and `set` 
where the return value of `get` should be perfectly forwarded to `set`:
```c++
template <typename T>
void foo(T && x)
{
    set(get(std::forward<T>(x)));
}
```
Suppose further that we need to update our code to perform some operation 
on the intermediate value produced by `get`. 
We do this by holding the value in a variable declared with `auto &&`: 
```c++
template <typename T>
void foo(T && x)
{
    auto && val = get(std::forward<T>(x));
    set(std::forward<decltype(val)>(val));
}
```
This avoids extraneous copies of the intermediate value.


#### 📌 11.4 References as Template Parameters

Although it is not common, 
template type parameters can become reference types.
```c++
template <typename T>
constexpr bool templateParameterIsReference(T) 
{
    return std::is_reference_v<T>;
}

int i;
int & r = i;
std::cout << std::boolalpha;
std::cout << templateParameterIsReference(i) << '\n';         // false
std::cout << templateParameterIsReference(r) << '\n';         // false
std::cout << templateParameterIsReference<int &>(i) << '\n';  // true
std::cout << templateParameterIsReference<int &>(r) << '\n';  // true
```
Even if a reference variable is passed to `templateParameterIsReference`, 
the template parameter `T` is deduced to the type of the referenced type 
(because, for a reference variable `v`, the expression `v` has the referenced type; 
the type of an expression is **never** a reference). 
However, we can force the reference case by explicitly specifying the type of `T`. 


Doing this can fundamentally change the behavior of a template, 
and a template may not have been designed with this possibility in mind, 
thereby triggering errors or unexpected behavior:
```c++
template <typename T, T z = T {}>
class RefMem
{
public:
    RefMem() : zero {z} {}

private:
    T zero;
};

int null = 0;

int main()
{
    RefMem<int> rm1; 
    RefMem<int> rm2;
    rm1 = rm2;                // OK
    
    RefMem<int &> rm3;        // ERROR: invalid default value for z
    RefMem<int &, 0> rm4;     // ERROR: invalid default value for z
    
    extern int null;
    RefMem<int &, null> rm5
    RefMem<int &, null> rm6;
    rm5 = rm6;                // ERROR: operator= is deleted due to reference member
    
    return 0;
}
```
Here we have a class with a member of template parameter type `T`,
initialized with a non-type template parameter `z` 
that has a zero-initialized default value. 
Instantiating the class with type `int` works as expected. 
However, when trying to instantiate it with a reference, things become tricky:
- The default initialization no longer works. 
- You can no longer pass just `0` as initializer for an `int`. 
- And, perhaps most surprising, the assignment operator is no longer available
  because classes with nonstatic reference members have deleted default assignment operators.


Also, using reference types for non-type template parameters 
is tricky and can be dangerous. 
Consider this example:
```c++
// Note: size is reference
template <typename T, int & sz> 
class Arr
{
public:
    Arr() : elems(sz) {}

    void print() const
    {
        for (int i = 0; i < sz; ++i)
        {
            std::cout << elems[i] << ' ';
        }
    }
    
private:
    std::vector<T> elems;
};

int size = 10;

int main()
{
    Arr<int, size> x;     // initializes internal vector with 10 elements
    x.print();            // OK

    size += 100;          // OOPS: modifies sz in Arr
    x.print();            // run-time ERROR: invalid memory access: loops over 110 elements

    Arr<int &, size> y;   // compile-time ERROR deep in the code of class std::vector
}
```
Here, the attempt to instantiate `Arr` for elements of a reference type 
results in an error deep in the code of class `std::vector`,
because it **can't** be instantiated with references as elements. 


Perhaps worse is the run-time error resulting from making `sz` a reference: 
It allows the recorded size value to change without the container being aware of it 
(i.e., the size value can become invalid). 
Thus, operations using the `size` (like the `print` member) 
are bound to run into undefined behavior (causing the program to crash, or worse). 


Note that changing the template parameter `sz` to be of type `int const &` 
does **not** address this issue, because `size` itself is still modifiable. 


Arguably, this example is far-fetched. 
However, in more complex situations, issues like these do occur. 
Also, in C++17 non-type parameters can be deduced: 
```c++
template <typename T, decltype(auto) sz>
class Arr;
```
Using `decltype(auto)` can easily produce reference types 
and is therefore generally avoided in this context (use `auto` by default). 
See Section 15.10.3 for details. 


The C++ standard library for this reason sometimes has surprising specifications and constraints. 
For example:
- In order to still have an assignment operator even if the template parameters are instantiated for references, 
  classes `std::pair` and `std::tuple` _implement_ the assignment operator instead of using the default behavior.
- Because of the complexity of possible side effects, 
  instantiation of the C++17 standard library class templates `std::optional` and `std::variant`
  for reference types is ill-formed (at least in C++17).   
  To disable references, a simple static assertion is enough:
  ```c++
  template <typename T>
  class optional
  {
      static_assert(!std::is_reference<T>::value, "Invalid instantiation of optional<T> for references");
      ...
  };
  ```
  Reference types in general are quite unlike other types 
  and are subject to several unique language rules. 
  This impacts, for example, the declaration of call parameters (see Section 7) 
  and also the way we define type traits (see Section 19.6.1).


#### 📌 11.5 Defer Evaluations


When implementing templates, sometimes the question comes up 
whether the code can deal with incomplete types. 
Consider the following class template:
```c++
template <typename T>
class Container
{
public:
    ...
private:
    T * elems;
    std::size_t size;
};
```
So far, this class can be used with incomplete types. 
This is useful, for example, with classes that refer to elements of their own type:
```c++
struct Node
{
    std::string value;
    Container<Node> next;  // only possible if Cont accepts incomplete types
};
```
However, for example, just by using some traits, 
you might lose the ability to deal with incomplete types. 
For example:
```c++
template <typename T>
class Containter
{
public:
    ...

    std::conditional_t<std::is_move_constructible_v<T>, 
                       T &&, 
                       T &>
    foo();

private:
    T * elems;
    std::size_t size;
};
```
Here, we use the trait `std::conditional` to decide 
whether the return type of the member function `foo` is `T &&` or `T &`. 
The decision depends on whether the template parameter type `T` supports move semantics.


The problem is that the trait `std::is_move_constructible` requires that
its argument is a complete type
(and is `not` void or an array of unknown bound; see Section D.3.2). 
Thus, with this declaration of `foo`, the declaration of `struct Node` fails.


**Not** all compilers yield an error if `std::is_move_constructible` is not an incomplete type. 
This is allowed, because for this kind of error, no diagnostics is required. 
Thus, this is at least a portability problem. 


We can deal with this problem by replacing `foo` by a member template 
so that the evaluation of `std::is_move_constructible` is deferred 
to the point of instantiation of `foo`:
```c++
template <typename T>
class Containter
{
public:
    ...
    
    template <typename D = T>
    std::conditional_t<std::is_move_constructible_v<D>, 
                       T &&, 
                       T &>
    foo();

private:
    T * elems;
    std::size_t size;
};
```
Now, the traits depends on the template parameter `D` 
(defaulted to `T`, the value we want anyway) 
and the compiler has to wait until `foo` is called for a concrete type like `Node` 
before evaluating the traits 
(by then `Node` is a complete type; it was only incomplete while being defined). 


#### 📌 11.6 Things to Consider When Writing Generic Libraries


Let's list some things to remember when implementing generic libraries 
(note that some of them might be introduced later in this book): 
• Use forwarding references to forward values in templates. 
  If the values do **not** depend on template parameters, use `auto &&` (see Section 11.3). 
• When parameters are declared as forwarding references, 
  be prepared that a template parameter has a reference type when passing lvalues 
  (see Section 15.6.2). 
• Use `std::addressof` when you need the address of an object depending on a template parameter 
  to avoid surprises when it binds to a type with overloaded `operator&` (Section 11.2.2). 
• For member function templates, 
  ensure that they don't match better than 
  the predefined copy/move constructor or assignment operator (Section 6.4). 
• Consider using `std::decay` when template parameters might be string literals 
  and are not passed by value (Section 7.4 and Section D.4). 
• If you have out or inout parameters depending on template parameters, 
  remember to SFINAE-out `const` template call arguments (see Section 7.2.2). 
• Be prepared to deal with the side effects of template parameters being references
  (see Section 11.4 details and Section 19.6.1 for an example). 
  In particular, you might want to ensure that the return type can't become a reference 
  (see Section 7.5). 
• Be prepared to deal with incomplete types to support, 
  for example, recursive data structures (see Section 11.5).
• Overload for all array types and not just `T[sz]` (see Section 5.4). 


#### 📌 11.7 Summary


- Templates allow you to pass functions, function pointers, function objects, functors, and lambdas as _callables_.
- When defining classes with an overloaded `operator()`, 
  declare it as `const` unless the call changes its state. 
- With `std::invoke`, you can implement code that can handle all callables, including member functions.
- Use `decltype(auto)` (**not** `auto &&`) to forward a return value perfectly. 
- Use `auto &&` to perfectly forward objects in generic code 
  if their type does not depend on template parameters. 
- Type traits are type functions that check for properties and capabilities of types. 
- Use `std::addressof` when you need the address of an object in a template.
- Use `std::declval` to create values of specific types in unevaluated expressions.
- Be prepared to deal with the side effects of template parameters being references. 
- You can use templates to defer the evaluation of expressions until the templates get instantiated 
  (e.g., to support using incomplete types in class templates). 






## 🌱 Part II Templates in Depth



### 🎯 Chapter 12 Fundamentals in Depth


In this chapter we review some of the fundamentals 
introduced in the first part of this book in depth: 
the declaration of templates, 
the restrictions on template parameters,
the constraints on template arguments, 
and so forth.


#### 📌 12.1 Parameterized Declarations


C++ currently supports four fundamental kinds of templates: 
- Class templates;
- Function templates; 
- Variable templates; 
- Alias templates.
Each of these template kinds can appear in namespace scope,
but also in class scope.
In class scope, they become 
- Nested class templates; 
- Member function templates; 
- Static data member templates; 
- Member alias templates. 


Such templates are declared much like 
ordinary classes, functions, variables, and type aliases 
(or their class member counterparts) 
except for being introduced 
by a _parameterization clause_ of the form
```
template <template-parameter-list>
...
```
Note that C++17 introduced another construct that is introduced with such a parameterization clause: 
_deduction guides_ (see Section 2.9 and Section 15.12.1 ). 
Those **aren't** called templates (e.g., they are **not** instantiated), 
but the syntax was chosen to be reminiscent of function templates. 


We'll come back to the actual template parameter declarations in a later section. 
First, some examples illustrate the four kinds of templates. 
They can occur in namespace scope (including global namespace scope) as follows:
```c++
// a namespace scope class template
template <typename T>
class Data
{
public:
    static constexpr bool copyable = true;
};

// a namespace scope function template
template <typename T>
void log(T x) {}

// a namespace scope variable template (since C++14)
template <typename T> 
T pi = 3.141592653589793238462643383279502884L;

// a namespace scope variable template (since C++14)
template <typename T> 
bool dataCopyable = Data<T>::copyable;

// a namespace scope alias template
template <typename T> 
using DataList = Data<T *>;
```
Note that in this example, the static data member `Data<T>::copyable` is **not** a variable template, 
even though it is indirectly parameterized through the parameterization of the class template Data. 
However, a variable template can appear in class scope (as the next example will illustrate), 
and in that case it is a static data member template.


The following example shows the four kinds of templates as class members 
that are defined within their parent class:
```c++
class Collection
{
public:
    // an in-class member class template definition
    template <typename T>
    class Node {};

    // an in-class (and therefore implicitly inline)
    // member function template definition
    template <typename T>
    T * alloc() {}

    // a static member variable template (since C++14)
    template <typename T>
    static T pi = 3.141592653589793238462643383279502884L;

    // a member alias template
    template <typename T>
    using NodePtr = Node<T> *;
};
```
Note that in C++17, variables (including static data members) 
and variable templates can be `inline`, 
which means that their definition can be repeated across translation units. 
This is redundant for variable templates, 
which can always be defined in multiple translation units. 
Unlike member functions, however, a static data member being defined in its enclosing class 
does **not** make it inline: 
The keyword `inline` must be specified in all cases. 


Finally, the following code demonstrates how member templates 
that are **not** alias templates can be defined out-of-class:
```c++
// a namespace scope class template
template <typename T> 
class List 
{
public:
    // because a template constructor is defined
    List() = default;
    
    // another member class template,
    // without its definition
    template <typename U> 
    class Handle;

    // a member function template (constructor)
    template <typename U> 
    List(List<U> const &);

    // a member variable template (since C++14)
    template <typename U> 
    static U pi;
};

// out-of-class member class template definition
template <typename T> 
template <typename U>
class List<T>::Handle {};

// out-of-class member function template definition
template <typename T> 
template <typename U>
List<T>::List (List<U> const & b) {}

// out-of-class static data member template definition
template <typename T> 
template <typename U>
U List<T>::pi = 3.141592653589793238462643383279502884L;
```
Member templates defined outside their enclosing class
may need multiple `template<...>` parameterization clauses: 
one for every enclosing class template and one for the member template itself. 
The clauses are listed starting from the outermost class template.


Note also that a constructor template 
(a special kind of member function template)
**disables** the implicit declaration of the default constructor 
(because it is only implicitly declared if **no** other constructor is declared). 
Adding a defaulted declaration
```c++
List() = default;
```
ensures an instance of `List<T>` is default-constructible 
with the semantics of an implicitly declared constructor. 

##### Union Templates

_Union templates_ are possible too (and they are considered a kind of class template):
```c++
template <typename T>
union AllocChunk 
{
    T object;
    unsigned char bytes[sizeof(T)];
};
```

##### Default Call Arguments

Function templates can have default call arguments just like ordinary function declarations:
```c++
template <typename T>
void report_top(Stack<T> const & s, int number = 10);

// T {} is zero for built-in types
template <typename T>
void fill(Array<T> & a, T const & v = T {}); 
```
The latter declaration shows that a default call argument could depend on a template parameter. 
It also can be defined as (the only way possible before C++11, see Section 5.2)
```c++
template <typename T>
void fill(Array<T> & a, T const & v = T()); 
```
When the `fill` function is called, the default argument is **not** instantiated 
if a second function call argument is supplied. 
This ensures that no error is issued if the default call argument 
can **not** be instantiated for a particular `T`:  
```c++
// no default constructor
class Value 
{
public:
    explicit Value(int); 
};

void init(Array<Value> & a)
{
    Value zero(0);
    fill(array, zero);  // OK: default constructor not used
    fill(array);        // ERROR: undefined default constructor for Value is used
}
```

##### Non-template Members of Class Templates

In addition to the four fundamental kinds of templates declared inside a class,
you can also have ordinary class members parameterized by being part of a class template. 
They are occasionally (erroneously) also referred to as _member templates_.
Although they can be parameterized, such definitions **aren't** quite first-class templates. 
Their parameters are entirely determined by the template of which they are members. 
For example:
```c++
template <int I>
class CupBoard
{
    class Shelf;                // ordinary class in class template
    void open();                // ordinary function in class template
    enum Wood : unsigned char;  // ordinary enumeration type in class template
    static double totalWeight;  // ordinary static data member in class template
};
```
The corresponding definitions only specify a parameterization clause for the parent class templates, 
but **not** for the member itself, because it is **not** a template 
(i.e., no parameterization clause is associated with the name appearing after the last `::`):
```c++
// definition of ordinary class in class template
template <int I> 
class CupBoard<I>::Shelf {};

// definition of ordinary function in class template
template <int I>
void CupBoard<I>::open() {}

// definition of ordinary enumeration type class in class template
template <int I> 
enum CupBoard<I>::Wood 
{
    MAPLE, 
    CHERRY, 
    OAK
};

// definition of ordinary static member in class template
template <int I> 
double CupBoard<I>::totalWeight = 0.0;
```
Since C++17, the static `totalWeight` member can be initialized 
inside the class template using `inline`:
```c++
template <int I>
class CupBoard
{
    inline static double totalWeight = 0.0;
    ...
};
```
Although such parameterized definitions are commonly called _templates_, 
the term doesn't quite apply to them. 
A term that has been occasionally suggested for these entities is _temploid_. 
Since C++17, the C++ standard does define the notion of a _templated entity_, 
which includes templates and temploids as well as, 
recursively, any entity defined or created in templated entities 
(that includes, e.g., a friend function defined inside a class template 
or the closure type of a lambda expression appearing in a template). 
Neither _temploid_ nor _templated entity_ has gained much traction so far, 
but they may be useful terms to communicate more precisely about C++ templates in the future. 

##### 12.1.1 Virtual Member Functions

Member function templates can **not** be declared virtual. 
This constraint is imposed because the usual implementation of the virtual function call mechanism
uses a _fixed-size virtual table_ with one entry per virtual function. 
However, the number of instantiations of a member function template 
is **not** fixed until the entire program has been translated. 
Hence, supporting virtual member function templates would require support 
for a whole new kind of mechanism in C++ compilers and linkers. 


In contrast, the ordinary members of class templates can be virtual 
because their number is fixed when a class is instantiated:
```c++
template <typename T>
class Dynamic 
{
public:
    // OK: one destructor per instance of Dynamic<T>
    virtual ~Dynamic(); 
    
    // ERROR: unknown number of instances of copy
    // given an instance of Dynamic<T>
    template <typename U>
    virtual void copy(U const &);
};
```

##### 12.1.2 Linkage of Templates

Every template must have a name, and that name must be unique within its scope,
except that function templates can be overloaded (see Chapter 16). 
Note especially that, unlike class types, 
class templates can **not** share a name with a different kind of entity:
```c++
// OK: 
// class names and non-class names 
// are in a different "space"
int C;

class C;

// ERROR: conflict with variable X
int X;

template <typename T>
class X;

// ERROR: conflict with struct S
struct S;

template <typename T>
class S; 
```
Template names have linkage, but they can **not** have C linkage. 
Nonstandard linkages may have an implementation-dependent meaning 
(however, we don't know of an implementation that supports nonstandard name linkages for templates):
```c++
// this is the default: 
// the linkage specification could be left out
extern "C++" template <typename T>
void normal();

// ERROR: 
// templates can not have C linkage
extern "C" template <typename T>
void invalid();

// nonstandard, 
// but maybe some compiler will someday support
// linkage compatible with Java generics
extern "Java" template <typename T>
void javaLink();
```
Templates usually have external linkage. 
The only exceptions are namespace scope function templates with the `static` specifier, 
templates that are direct or indirect members of an unnamed namespace (which have internal linkage), 
and member templates of unnamed classes (which have no linkage). 
For example:
```c++
// refers to the same entity as a declaration 
// of the same name (and scope) in another file
template <typename T>
void external(); 

// unrelated to a template with the same name in another file
template <typename T>
static void internal();

// redeclaration of the previous declaration
template <typename T>
static void internal();

namespace
{

// also unrelated to a template with the same name in another file,
// even one that similarly appears
template <typename>
void otherInternal(); 

}  // namespace anonymous

namespace
{

// redeclaration of the previous template declaration
template <typename>
void otherInternal();

}  // namespace anonymous

struct
{
    // no linkage: cannot be redeclared
    template <typename T>
    void f(T) {} 
} x;
```
Note that since the latter member template has no linkage, 
it must be defined within the unnamed class 
because there is **no** way to provide a definition outside the class. 


As of C++17, templates can **not** be declared in function scope or local class scope, 
but _generic lambdas_ (see Section 15.10.6), 
which have associated closure types that contain member function templates, 
can appear in local scopes, which effectively implies a kind of local member function template.


The linkage of an instance of a template is that of the template. 
For example, a function `internal<void>` 
instantiated from the template `internal` declared above
will have internal linkage. 
This has an interesting consequence in the case of variable templates: 
```c++
template <typename T> 
T zero = T {};
```
All instantiations of `zero` have external linkage, 
even something like `zero<int const>`. 
That's perhaps counter-intuitive given that
```c++
int const zero = int {};
```
has internal linkage because it is declared with a `const` type. 
Similarly, all instantiations of the template
```c++
template <typename T> 
int const max_volume = 11;
```
have external linkage, 
despite all those instantiations also having type `int const`.

##### 12.1.3 Primary Templates

Normal declarations of templates declare _primary templates_. 
Such template declarations are declared **without** 
adding template arguments in angle brackets after the template name:
```c++
// OK: primary template
template <typename T>
class Box;

// ERROR: does not specialize
template <typename T>
class Box<T>;

// OK: primary template
template <typename T>
void translate(T);

// ERROR: not allowed for functions
template <typename T>
void translate<T>(T);

// OK: primary template
template <typename T> 
constexpr T zero = T {};

// ERROR: does not specialize
template <typename T> 
constexpr T zero<T> = T {}; 
```
Non-primary templates occur when declaring _partial specializations_ of class or variable templates. 
Those are discussed in Chapter 16. 
Function templates must always be primary templates (thus can **not** be partially specialized). 


#### 📌 12.2 Template Parameters


There are three basic kinds of template parameters:
1. Type parameters (these are by far the most common);
2. Non-type parameters; 
3. Template template parameters.


Any of these basic kinds of template parameters
can be used as the basis of a _template parameter pack_ (see Section 12.2.4). 


Template parameters are declared in the introductory parameterization clause of a template declaration.
An exception since C++14 are the implicit template type parameters for a generic lambda. 
See Section 15.10.6.

Such declarations do **not** necessarily need to be named:
```c++
// X is parameterized by a type and an integer
template <typename, int>
class X; 
```
A parameter name is, of course, required if the parameter is referred to later in the template. 
Note also that a template parameter name can be referred to in a subsequent parameter declaration (but not before):
```c++
template <typename T, 
          T Root, 
          template <T> class Buf> 
class Structure;
```

##### 12.2.1 Type Parameters


Type parameters are introduced with either the keyword `typename` or the keyword `class`: 
The two are entirely equivalent.
The keyword `class` does **not** imply that the substituting argument should be a class type. 
It could be any accessible type.


The keyword must be followed by a simple identifier, 
and that identifier must be followed by one of the following: 
- A comma `,` to denote the start of the next parameter declaration; 
- A closing angle bracket `>` to denote the end of the parameterization clause;
- An equal sign `=` to denote the beginning of a default template argument. 


Within a template declaration, a type parameter acts much like a type alias. 
For example, it is **not** possible to use an elaborated name of the form `class T` when `T` is a template parameter, 
even if `T` were to be substituted by a class type:
```c++
template <typename Allocator>
class List 
{
    class Allocator * allocptr;  // ERROR: use "Allocator * allocptr"
    friend class Allocator;      // ERROR: use "friend Allocator"
    ...
};
```

##### 12.2.2 Non-type Parameters

Non-type template parameters stand for constant values 
that can be determined at compile or link time (i.e. satisfies `constexpr`). 


Template template parameters do **not** denote types either. 
However, they are distinct from non-type parameters. 
This oddity is historical: 
Template template parameters were added to the language 
_after_ type parameters and non-type parameters.


The type of such a parameter 
(in other words, the type of the value for which it stands) 
must be one of the following:
- An integer type or an enumeration type;
- A pointer type;
  - As of C++17, 
    only "pointer to object" and "pointer to function" types are permitted, 
    which **excludes** types like `void *`. 
    However, all compilers appear to accept `void *` also. 
- A pointer-to-member type;
- An lvalue reference type 
  (both references to objects and references to functions are acceptable);
- `std::nullptr_t`;
- A type containing `auto` or `decltype(auto)` `(since C++17)` (see Section 15.10.1).


All other types are excluded as of C++17, 
although floating-point types may be added in the future (see Section 17.2).  


The declaration of a non-type template parameter can 
in some cases also start with the keyword `typename`:
```c++
template <typename T,                         // a type parameter
          typename T::Allocator * Allocator>  // a non-type parameter
class List;
```
or with the keyword `class`:
```c++
class X {};

// a non-type parameter of pointer type
// keyword class is optional here
template <class X *>
class Y;

// a non-type parameter of pointer type
template <X *>
class Z;
```
The two cases are easily distinguished 
because the first is followed by a simple identifier 
and then one of a small set of tokens 
(`=` for a default argument, 
`,` to indicate that another template parameter follows, 
or a closing `>` to end the template parameter list). 
Keyword `typename` in the first non-type parameter 
because it is a nested name dependent on template parameters 
(see Section 5.1 and Section 13.3.2). 


Function and array types can be explicitly specified as non-type template parameters, 
but they are implicitly adjusted to the pointer type to which they decay:
```c++
template <int buf[5]> class Lexer;     // buf is really an int*
template <int * buf> class Lexer;      // OK: this is a redeclaration
template <int fun()> struct FuncWrap;  // fun really has pointer to function type
template <int (*)()> struct FuncWrap;  // OK: this is aredeclaration
```
Non-type template parameters are declared much like variables, 
but they can **not** have non-type specifiers like `static`, `mutable`, and so forth. 
They can have `const` and `volatile` qualifiers, 
but if such a qualifier appears at the outermost level of the parameter type, 
it is simply ignored:
```c++
template <int const length> class Buffer;  // const is useless here
template <int length> class Buffer;        // same as previous declaration
```
Finally, non-reference non-type parameters are always prvalues when used in expressions. 
Their address can **not** be taken, and they can **not** be assigned to. 
A non-type parameter of lvalue reference type, on the other hand, can be used to denote an lvalue:
```c++
template <int & Counter>
struct LocalIncrement 
{
    // OK: reference to an integer
    LocalIncrement() { Counter = Counter + 1; } 
    ~LocalIncrement() { Counter = Counter - 1; }
};
```
Rvalue references are **not** permitted.

##### 12.2.3 Template Template Parameters

Template template parameters are placeholders for class or alias templates. 
They are declared much like class templates, 
but the keywords `struct` and `union` can **not** be used:
```c++
template <template <typename X> class C>     // OK
void f(C<int> * p);

template <template <typename X> struct C>    // ERROR: struct not valid here
void f(C<int> * p);

template <template <typename X> union C>     // ERROR: union not valid here
void f(C<int> * p);
```
C++17 allows the use of `typename` instead of `class`: 
That change was motivated by the fact that template template parameters can be substituted 
not only by class templates but also by alias templates 
(which instantiate to arbitrary types). 
So, in C++17, our example above can be written instead as
```c++
template <template <typename X> typename C>  // OK since C++17
void f(C<int> * p);
```
In the scope of their declaration, template template parameters 
are used just like other class or alias templates. 


The parameters (`typename T`, `typename A`) 
of template template parameters (`class Container`) 
can have default template arguments (`A = MyAllocator`). 
These default arguments apply 
when the corresponding parameters are **not** specified 
in uses of the template template parameter:
```c++
template <template <typename T, 
                    typename A = MyAllocator> 
          class Container>
class Adaptation 
{
    // implicitly equivalent to Container<int, MyAllocator>
    Container<int> storage;
    ...
};
```
`T` and `A` are the names of the template parameter 
of the template template parameter `Container`.
These names be used _only_ in the declaration 
of other parameters of that template template parameter.
The following contrived template illustrates this concept:
```c++
template <template <typename T, T *> 
          class Buf>
class Lexer 
{
    // ERROR: 
    // names of the template parameters 
    // of a template template parameter 
    // can not be used here 
    static T * storage; 
    ...
};
```
Usually however, the names of the template parameters of a template template parameter 
are **not** needed in the declaration of other template parameters 
and are therefore often left unnamed altogether. 
For example, our earlier `Adaptation` template could be declared as follows:
```c++
template <template <typename, 
                    typename = MyAllocator> 
          class Container>
class Adaptation 
{
    // implicitly equivalent to Container<int, MyAllocator>
    Container<int> storage;
    ...
};
```

##### 12.2.4 Template Parameter Packs

Since C++11, any kind of template parameter can be turned into a _template parameter pack_ 
by introducing an ellipsis `...` prior to the template parameter name or, 
if the template parameter is unnamed, where the template parameter name would occur:
```c++
// declares a template parameter pack named Types
template <typename ... Types> 
class Tuple;
```
A template parameter pack behaves like its underlying template parameter,
but with a crucial difference:
While a normal template parameter matches exactly one template argument, 
a template parameter pack can match _any number of_ template arguments. 
This means that the `Tuple` class template declared above accepts 
any number of (possibly distinct) types as template arguments:
```c++
using IntTuple = Tuple<int>;             // OK: one template argument 
using IntCharTuple = Tuple<int, char>;   // OK: two template arguments
using IntTriple = Tuple<int, int, int>;  // OK: three template arguments
using EmptyTuple = Tuple<>;              // OK: zero template arguments
```
Similarly, template parameter packs of non-type and template template parameters
can accept any number of non-type or template template arguments, respectively:
```c++
// OK: declares a non-type template parameter pack
template <typename T, unsigned ... Dimensions>
class MultiArray;

// OK: 3x3 matrix
using TransformMatrix = MultiArray<double, 3, 3>;

// OK: declares a template template parameter pack
template <typename T, template <typename, typename> ... Containers>
void testContainers(); 
```
The `MultiArray` example requires all non-type template arguments to be of the same type `unsigned`. 
C++17 introduced the possibility of deduced non-type template arguments, 
which allows us to work around that restriction to some extent (see Section 15.10.1): 
```c++
template <auto ... vs>
struct Values {};

Values<1, 2, 3> beginning;
Values<1, 'x', nullptr> triplet;

// homogeneous pack of non-type parameters
template <auto v1, decltype(v1) ... vs>
struct HomogeneousValues {};
```
Primary class templates, primary variable templates, and alias templates
may have _at most one_ template parameter pack and,
if present, the template parameter pack must be _the last_ template parameter. 
Function templates have a weaker restriction: 
Multiple template parameter packs are permitted, 
as long as each template parameter subsequent to a template parameter pack 
either has a default value (see the next section) or can be deduced (see Chapter 15):
```c++
// ERROR: template parameter pack is not the last template parameter
template <typename ... Types, typename Last>
class LastType;

// OK: template parameter pack is followed by a deducible template parameter
template <typename ... TestTypes, typename T>
void runTests(T value); 

template <unsigned ...> 
struct Tensor;

// OK: the tensor dimensions can be deduced
template <unsigned ... dims1, unsigned ... dims2>
auto compose(Tensor<dims1...>, Tensor<dims2...>);
```
The last example is the declaration of a function with a deduced return type `(since C++14)`. 
See also Section 15.10.1.


Declarations of _partial specializations_ of class/variable templates (see Chapter 16) 
_can_ have multiple parameter packs, unlike their primary template counterparts. 
That is because partial specialization are selected through a deduction process 
that is nearly identical to that used for function templates. 
```c++
template <typename ...> 
struct Typelist;

template <typename X, typename Y> 
struct Zip;

// OK: partial specialization uses deduction 
// to determine the Xs and Ys substitutions
template <typename ... Xs, typename ... Ys>
struct Zip<Typelist<Xs...>, Typelist<Ys...>>;
```
A type parameter pack can **not** be expanded in its own parameter clause. 
For example:
```c++
// ERROR: Ts cannot be expanded in its own parameter list
template <typename ... Ts, Ts... vals> 
struct StaticValues {};
```
However, nested templates can create similar valid situations:
```c++
template <typename ... Ts> struct ArgList 
{
    template<Ts ... vals> 
    struct Vals {};
};

ArgList<int, char, char>::Vals<3, 'x', 'y'> tada;
```
A template that contains a template parameter pack is called a _variadic template_ 
because it accepts a variable number of template arguments. 
Chapter 4 and Section 12.4 describe the use of variadic templates. 

##### 12.2.5 Default Template Arguments

Any kind of template parameter that is **not** a template parameter pack 
can be equipped with a default argument, 
although it must match the corresponding parameter in kind 
(e.g., a type parameter can **not** have a non-type default argument). 
A default argument can **not** depend on its own parameter, 
because the name of the parameter is **not** in scope until after the default argument. 
However, it may depend on previous parameters:
```c++
template <typename T, 
          typename Allocator = std::allocator<T>>
class List;
```
A template parameter for a class template, variable template, or alias template 
can have a default template argument 
only if default arguments were also supplied for the subsequent parameters. 
(A similar constraint exists for default function call arguments.) 
The subsequent default values are usually provided in the same template declaration, 
but they could also have been declared in a previous declaration of that template. 
The following example makes this clear:
```c++
// OK
template <typename T1, 
          typename T2, 
          typename T3,
          typename T4 = char, 
          typename T5 = char>
class Quintuple; 

// OK: T4 and T5 already have defaults
template <typename T1, 
          typename T2, 
          typename T3 = char,
          typename T4, 
          typename T5>
class Quintuple;

// ERROR: 
// T1 can not have a default argument
// because its successor T2 does not have one
template <typename T1 = char, 
          typename T2, 
          typename T3,
          typename T4, 
          typename T5>
class Quintuple; 
```
Default template arguments for template parameters of function templates 
do **not** require subsequent template parameters to have a default template argument, 
as those can still be determined by template argument deduction (see Chapter 15): 
```c++
// OK: if not explicitly specified, R will be void
template <typename R = void, typename T>
R * addressof(T & value); 
```
Default template arguments can **not** be repeated:
```c++
template <typename T = void>
class Value;

// ERROR: repeated default argument
template <typename T = void>
class Value; 
```
A number of contexts do **not** permit default template arguments:
- Partial specializations
```c++
template <typename T>
class C;

// ERROR
template <typename T = int>
class C<T *>;
```
- Parameter packs
```c++
template <typename ... Ts = int> 
struct X;
```
- The out-of-class definition of a member of a class template:
```c++
template<typename T> 
struct X
{
    T f();
};

// ERROR
template <typename T = int> 
T X<T>::f() {}
```
- A friend class template declaration:
```c++
struct S 
{
    template <typename = void> 
    friend struct F;
};
```
- A friend function template declaration **unless** it is a definition 
  and **no** declaration of it appears anywhere else in the translation unit:
```c++
struct S
{
    // ERROR: not adefinition
    template <typename = void> 
    friend void f(); 
    
    // OK so far
    template <typename = void> 
    friend void g() {}
};

// ERROR: 
// g was given a default template argument when defined.  
// No other declaration may exist here. 
template <typename> 
void g();
```


#### 📌 12.3 Template Arguments


When instantiating a template, template parameters are substituted by template arguments. 
The arguments can be determined using several different mechanisms:
- **Explicit template arguments**:  
  A template name can be followed by explicit template arguments enclosed in angle brackets. 
  The resulting name is called a _template-id_.
- **Injected class name**:  
  Within the scope of a class template `X` with template parameters `P1`, `P2`, ..., 
  the name `X` can be equivalent to the template-id `X<P1, P2, ...>`. 
  See Section 13.2.3 for details. 
- **Default template arguments**:  
  Explicit template arguments can be omitted from template instances 
  if default template arguments are available. 
  However, for a class or alias template,
  even if all template parameters have a default value, 
  the (possibly empty) angle brackets must be provided. 
- **Argument deduction**:  
  Function template arguments that are **not** explicitly specified 
  may be deduced from the types of the function call arguments in a call. 
  This is described in detail in Chapter 15.
  Deduction is also done in a few other situations.
  If all the template arguments can be deduced, 
  **no** angle brackets need to be specified after the name of the function template. 
  C++17 also introduces the ability to deduce class template arguments
  from the initializer of a variable declaration or functional-notation type conversion. 
  See Section 15.12 for a discussion. 

##### 12.3.1 Function Template Arguments

Template arguments for a function template can be specified explicitly, 
deduced from the way the template is used, 
or provided as a default template argument. 
For example: 
```c++
template <typename T>
T max(T a, T b)
{
    return b < a ? a : b;
}

max<double>(1.0, -3.0);  // explicitly specify template argument
max(1.0, -3.0);          // template argument is implicitly deduced to be double
max<int>(1.0, 3.0);      // the explicit <int> inhibits the deduction;
                         // hence the result has type int
```
Some template arguments can **never** be deduced 
because their corresponding template parameter does **not** appear in a function parameter type 
or for some other reason (see Section 15.2). 
The corresponding parameters are typically placed at the beginning of the list of template parameters, 
so they can be specified explicitly while allowing the other arguments to be deduced:
```c++
// SrcT can be deduced, but DstT can not
template <typename DstT, typename SrcT>
DstT implicit_cast(SrcT const & x) 
{
    return x;
}

int main()
{
    double value = implicit_cast<double>(-1);
}
```
If we had reversed the order of the template parameters in this example 
(in other words, if we had written `template<typename SrcT, typename DstT>`),
a call of `implicit_cast` would have to specify _both_ template arguments explicitly. 


Moreover, such parameters can't usefully 
be placed after a template parameter pack 
or appear in a partial specialization, 
because there would be **no** way to explicitly specify or deduce them.
```c++
// useless declaration,
// because N can not be specified or deduced
template <typename ... Ts, int N>
void f(double (&)[N + 1], Ts ... ps);
```
Because function templates can be overloaded, 
explicitly providing all the arguments for a function template 
may **not** be sufficient to identify a single function: 
In some cases, it identifies a _set_ of functions. 
The following example illustrates a consequence of this observation:
```c++
template <typename Func, typename T>
void apply(Func f, T x)
{
    f(x);
}

template <typename T> 
void single(T);

template <typename T> 
void multi(T);

template <typename T> 
void multi(T *);

int main()
{
    apply(&single<int>, 3);  // OK
    apply(&multi<int>, 7);   // ERROR: ambiguous call: no single multi<int>
}
```
In this example, the first call to `apply` works 
because the type of the expression `&single<int>` is unambiguous. 
As a result, the template argument value for the `Func` parameter is easily deduced. 
In the second call, however, `&multi<int>` could be one of two different types 
and therefore `Func` can **not** be deduced in this case.


Furthermore, it is possible that substituting template arguments in a function template 
results in an attempt to construct an invalid C++ type or expression. 
Consider the following overloaded function template (`RT1` and `RT2` are unspecified types):
```c++
template <typename T> 
RT1 test(typename T::X const *);

template <typename T> 
RT2 test(/* ... */);
```
The expression `test<int>` makes **no** sense 
for the first of the two function templates 
because type `int` has **no** member type `X`. 
However, the second template has no such problem. 
Therefore, the expression `&test<int>` identifies the address of a single function. 
The fact that the substitution of `int` into the first template fails does not make the expression invalid. 
This SFINAE principle is an important ingredient to make the overloading of function templates practical 
and is discussed in Section 8.4 and Section 15.7.

##### 12.3.2 Type Arguments

Template type arguments are the "values" specified for template type parameters.
Any type (including `void`, function types, reference types, etc.) can be used as a template argument, 
as long as their substitution for the template parameters must lead to valid constructs:
```c++
// requires that the unary * be applicable to T
template <typename T>
void clear(T p)
{
    *p = 0;
}

int main()
{
    int a;
    clear(a);  // ERROR: int doesn't support the unary *
}
```

##### 12.3.3 Non-type Arguments

Non-type template arguments are the values substituted for non-type parameters. 
Such a value must be one of the following things: 
- Another non-type template parameter that has the right type. 
- A compile-time constant value of integer (or enumeration) type. 
  This is acceptable only if the corresponding parameter has a type 
  that matches that of the value or a type 
  to which the value can be implicitly converted **without** narrowing. 
  For example, a `char` value can be provided for an `int` parameter, 
  but `500` is **not** valid for an 8-bit `char` parameter. 
- The name of an external variable or function 
  preceded by the built-in unary address-of operator `operator&`. 
  For functions and array variables, `operator&` can be left out. 
  Such template arguments match non-type parameters of a pointer type. 
  C++17 relaxed this requirement to permit any constant-expression 
  that produces a pointer to a function or variable. 
- The previous kind of argument but **without** a leading `operator&` 
  is a valid argument for a non-type parameter of reference type. 
  C++17 relaxed the constraint to permit any constant-expression `glvalue` for a function or variable. 
- A pointer-to-member constant.  
  In other words, an expression of the form `&ClassType::nonStaticMember`. 
  This matches non-type parameters of pointer-to-member type _only_. 
  In C++17, any constant-expression evaluating to a matching pointer-to-member constant is permitted. 
- A null pointer constant is a valid argument for a non-type parameter of pointer or pointer-to-member type. 


For non-type parameters of integral type, 
implicit conversions to the parameter type are considered. 
With the introduction of `constexpr` conversion functions in C++11, 
this means that the argument before conversion can have a class type. 


Prior to C++17, when matching an argument to a parameter that is a pointer or reference, 
user-defined conversions (constructors for one argument and conversion operators) 
and derived-to-base conversions are **not** considered,
even though in other circumstances they would be valid implicit conversions. 
Implicit conversions that make an argument more `const` and/or more `volatile` are fine. 
Here are some valid examples of non-type template arguments:
```c++
template <typename T, T nonTypeParam>
class C;

// integer type
C<int, 33> * c1;

// address of an external variable
int a;
C<int*, &a> * c2;

// Name of a function. 
// Overload resolution selects f(int) in this case. 
// The & is implied. 
void f();
void f(int);

C<void (*)(int), f> * c3;

// function template instantiations are functions
template <typename T> 
void templ_func();

C<void (), &templ_func<double>> * c4; 

struct X 
{
    static bool b;
    int n;
    constexpr operator int() const { return 42; }
};

// static class members are acceptable variable/function names 
C<bool &, X::b> * c5;

// an example of a pointer-to-member constant 
C<int X::*, &X::n> * c6;

// OK. 
// X is first converted to int via a constexpr conversion function, 
// and then to long via a standard integer conversion. 
C<long, X {}> * c7; 
```
A general constraint of template arguments is that 
a compiler or a linker must be able to express their value when the program is being built. 
Values that aren't known until a program is run (e.g., the address of local variables) 
**aren't** compatible with the notion that templates are instantiated when the program is built. 


There are some constant values that are, **not** currently valid:
- Floating-point numbers;
- String literals;
- Null pointer constants `(until C++11)`


One of the problems with string literals is that 
two identical literals can be stored at two distinct addresses. 
An alternative (but cumbersome) way to express templates instantiated over constant strings 
involves introducing an additional variable to hold the string:
```c++
template <char const * str>
class Message {};

extern char const hello[] = "Hello World!";
char const hello11[] = "Hello World!";

void foo()
{
    static char const hello17[] = "Hello World!";
    Message<hello> msg03;    // OK in all versions
    Message<hello11> msg11;  // OK since C++11
    Message<hello17> msg17;  // OK since C++17
}
```
The requirement is that a non-type template parameter declared as reference or pointer
can be a constant expression with: 
- External linkage; 
- Internal linkage `(since C++11)`; 
- Any linkage `(since C++17)`. 

Here are a few other invalid examples:
```c++
template <typename T, T nontypeParam>
class C;

struct Base 
{
    int i;
};

struct Derived : public Base {}; 

Base base;
Derived derived;

C<Base*, &derived> * err1;  // ERROR: derived-to-base conversions are not considered
C<int &, base.i> * err2;    // ERROR: fields of variables aren't considered to be variables

int a[10];
C<int *, &a[0]> * err3;     // ERROR: aren't acceptable either
```

##### 12.3.4 Template Template Arguments

A template template argument must generally be a class template or alias template
with parameters that exactly match the parameters of the template template parameter it substitutes. 
Prior to C++17, default template arguments of a template template argument were _ignored_ 
(but if the template template parameter has default arguments, 
they are considered during the instantiation of the template). 
C++17 relaxed the matching rule to just require that 
the template template parameter be _at least as specialized (see Section 16.2.2) as_ 
the corresponding template template argument.


This makes the following example invalid prior to C++17:
```c++
// Container expects one parameter
template<typename T1, 
         typename T2,
         template <typename> class Container>
class Rel {};

// ERROR before C++17: 
// std::list<T, Alloc = std::allocator<T>> has more than template parameter
// Default template arguments are not considered during substitution
Rel<int, double, std::list> rel;
```
The problem in this example is that the `std::list` template has more than one parameter. 
The second parameter (which describes an allocator) has a default value, 
but prior to C++17, that is **not** considered when matching `std::list` to the `Container` parameter. 


Variadic template template parameters are an **exception** 
to the pre-C++17 "exact match" rule described above 
and offer a solution to this limitation: 
They enable more general matching against template template arguments. 
A template template parameter pack can match zero or more template parameters 
of the same kind (type, non-type, template) in the template template argument:
```c++
// Container expects any number of type parameters
template <typename T1, 
          typename T2,
          template <typename ...> class Container>
class Rel {};

// OK: 
// std::list has two template arguments,
// but can be used with one argument. 
Rel<int, double, std::list> rel;
```
Template parameter packs can only match template arguments
of the same kind (type, non-type, template). 
For example, the following class template can be instantiated 
with any class template or alias template having only template type parameters, 
because the template type parameter pack passed there as `TT` 
can match zero or more template type parameters:
```c++
template <template <typename... > class TT>
class AlmostAnyTmpl {};

AlmostAnyTmpl<std::vector> withVector;  // two type parameters
AlmostAnyTmpl<std::map> withMap;        // four type parameters

// ERROR: 
// A template type parameter pack
// doesn't match a non-type template parameter
AlmostAnyTmpl<std::array> withArray;
```
Prior to C++17, only the keyword `class` could be used to declare a template template parameter. 
But this does **not** indicate that 
only class templates declared with the keyword `class` 
were allowed as substituting arguments. 
`struct`, `union`, and alias templates are all valid arguments 
for a template template parameter. 
This is similar to the observation that any type can be used as an argument
for a template type parameter declared with the keyword `class`.

##### 12.3.5 Equivalence

Two sets of template arguments are equivalent 
when values of the arguments are identical one-for-one. 
For type arguments, type aliases **don't** matter: 
It is the type ultimately underlying the type alias declaration that is compared. 
For integer non-type arguments, the value of the argument is compared. 
How that value is expressed **doesn't** matter:
```c++
// Note NO template definition is needed 
// to establish template equivalence!
template <typename T, int I>
class Mix;

using Int = int;

// p2 has the same type as p1
Mix<int, 3 * 3> * p1;
Mix<Int, 4 + 5> * p2;
```
As is clear from this example, 
**no** template definition is needed 
to establish the equivalence of the template argument lists. 


In template-dependent contexts, 
the "value" of a template argument can **not** always be established definitely, 
and the rules for equivalence become a little more complicated.
```c++
template <int N> struct I {};
template <int M, int N> void f(I<M + N>);  // #1
template <int N, int M> void f(I<N + M>);  // #2
template <int M, int N> void f(I<N + M>);  // #3 ERROR
```
`#1` and `#2` are _equivalent_ and declare the same function template `f`,  
because these two templates are the same by swapping the naming of `N` and `M`.
The expressions `M + N` and `N + M` in those two declarations are called _equivalent_.

Declaration `#3` is subtly different: The order of the operands is inverted. 
That makes the expression `N + M` in `#3` **not** equivalent to either of the other two expressions. 
However, because the expression will produce the same result for any values of the template parameters involved, 
those expressions are called _functionally equivalent_. 
It is an **error** for templates to be declared in ways that differ only because 
the declarations include functionally equivalent expressions that are not actually equivalent. 


However, such an error need **not** be diagnosed by your compiler. 
That's because some compilers may 
internally represent `N + 1 + 1` in exactly the same way as `N + 2`, 
whereas other compilers may not. 
Rather than impose a specific implementation approach, 
the standard allows either one and requires programmers to be careful in this area.


A function generated from a function template is **never** equivalent to an ordinary function
even though they may have the same type and the same name. 
This has two important consequences for class members:
1. A function generated from a member function template 
   **never** overrides a virtual function.
2. A constructor generated from a constructor template is 
   **never** a copy or move constructor. 
   However, a constructor template can be a default constructor. 
   Similarly, an assignment generated from an assignment template 
   is **never** a copy-assignment or move-assignment operator. 


This can be good and bad: 
- It can happen that a template constructor or assignment operator
  is a better match than the predefined copy/move constructor or assignment operator,
  although a template version is provided for initialization of other types only.
  See Section 6.2 for details.
- It is not easy to "templify" a copy/move constructor,
  for example, to be able to constrain its existence.
  See Section 6.4 for details.


#### 📌 12.4 Variadic Templates


Variadic templates are templates that contain at least one template parameter pack. 
The term variadic is borrowed from C's variadic functions, 
which accept a variable number of function arguments. 
Variadic templates also borrowed from C the use of the ellipsis to denote zero or more arguments 
and are intended as a typesafe replacement for C's variadic functions for some applications. 


Variadic templates are useful when a template's behavior 
can be generalized to any number of arguments. 
The `Tuple` class template introduced in Section 12.2.4 is one such type, 
because a tuple can have any number of elements, 
all of which are treated the same way. 
We can also imagine a simple `print` function 
that takes any number of arguments and displays each of them in sequence. 


When template arguments are determined for a variadic template, 
each template parameter pack in the variadic template will match 
a sequence of zero or more template arguments (an _argument pack_). 
```c++
template <typename ... Types>
class Tuple {};

int main() 
{
    Tuple<> t0;            // Types contains an empty list
    Tuple<int> t1;         // Types contains int
    Tuple<int, float> t2;  // Types contains int and float
}
```
Because a template parameter pack represents a list of template arguments 
rather than a single template argument, 
it must be used in a context where the same language construct 
applies to all of the arguments in the argument pack. 
One such construct is the `sizeof...` operation, 
which counts the number of arguments in the argument pack:
```c++
template <typename ... Types>
class Tuple
{
public:
    static constexpr std::size_t length = sizeof...(Types);
};

int a1[Tuple<int>::length];               // array of one integer
int a3[Tuple<short, int, long>::length];  // array of three integers
```

##### 12.4.1 Pack Expansions

The `sizeof...` expression is an example of a _pack expansion_. 
A pack expansion is a construct that expands an argument pack into separate arguments. 
`sizeof...` is one form of pack expansion. 
Other forms pack expansions are identified by an ellipsis `...` to the right of an element in the list. 
```c++
template <typename ...>
class Tuple {};

template <typename ... Types>
class MyTuple : public Tuple<Types ...> {};

MyTuple<int, float> t2;  // inherits from Tuple<int, float>
```
Note that you **can't** access the individual elements of a parameter pack directly by name, 
because names for individual elements are **not** defined in a variadic template.
If you need the types, the only thing you can do is to pass them (recursively) to another class or function. 


Each pack expansion has a pattern, 
which is the type or expression that will be repeated for each argument in the argument pack
and typically comes before the ellipsis that denotes the pack expansion. 
Our prior examples have had only trivial patterns (the name of the parameter pack) 
but patterns can be arbitrarily complex. 
```c++
template <typename ... Types>
class PtrTuple : public Tuple<Types * ...> {};

PtrTuple<int, float> t3;  // Inherits from Tuple<int *, float *>
```

##### 12.4.2 Where Can Pack Expansions Occur?

Our examples thus far have focused on the use of pack expansions 
to produce a sequence of template arguments. 
In fact, pack expansions can be used essentially anywhere 
in the language where the grammar provides a comma-separated list,
including:
- In the list of base classes.
- In the list of base class initializers in a constructor.
- In a list of call arguments (the pattern is the argument expression).
- In a list of initializers (e.g., in a braced initializer list).
- In the template parameter list of a class, function, or alias template.
- In the list of exceptions that can be thrown by a function 
  (deprecated in C++11 and C++14, and disallowed in C++17). 
- Within an attribute, if the attribute itself supports pack expansions 
  (although no such attribute is defined by the C++ standard).
- When specifying the alignment of a declaration.
- When specifying the capture list of a lambda.
- In the parameter list of a function type.
- In using declarations `(since C++17)`. 


We've already mentioned `sizeof...` as a pack-expansion mechanism 
that does **not** actually produce a list. 
C++17 also adds _fold expressions_, 
which are another mechanism that does **not** produce a comma-separated list.
```c++
template <typename ... Mixins>
class Point : public Mixins...  // base class pack expansion
{
public:
    Point() : Mixins()... {}  // base class initializer pack expansion

    template <typename Visitor>
    void visitMixins(Visitor visitor)
    {
        visitor(static_cast<Mixins &>(*this)...);  // call argument pack expansion
    }

private:
    double x, y, z;
};

struct Color { char red, green, blue; };
struct Label { std::string name; };

Point<Color, Label> p;  // inherits from both Color and Label
```
A pack expansion can also be used within a template parameter list 
to create a non-type or template parameter pack:
```c++
template <typename ... Ts>
struct Values 
{
    template <Ts ... vs>
    struct Holder {};
};

int i;
Values<char, int, int *>::Holder<'a', 17, &i> valueHolder;
```
Note that once the type arguments for `Values` have been specified, 
the non-type argument list for `Values::Holder` has a fixed length; 
the parameter pack `vs` is thus **not** a variable-length parameter pack. 

`Values` is a non-type template parameter pack for which 
each of the actual template arguments can have a different type, 
as specified by the types provided for the template type parameter pack `Ts`. 
Note that the ellipsis in the declaration of `Values` plays a dual role, 
both declaring the template parameter as a template parameter pack 
and declaring the type of that template parameter pack as a pack expansion. 

##### 12.4.3 Function Parameter Packs

A _function parameter pack_ is a function parameter that matches zero or more function call arguments. 
Like a template parameter pack, a function parameter pack is introduced 
using an ellipsis `...` prior to (or in the place of) the function parameter name. 
A function parameter pack must be expanded by a pack expansion whenever it is used. 
Template parameter packs and function parameter packs together are referred to as _parameter packs_. 


Unlike template parameter packs, function parameter packs are always pack expansions, 
so their declared types must include at least one parameter pack. 
```c++
template <typename ... Mixins>
class Point : public Mixins...
{
public:
    Point(Mixins... mixins) : Mixins(mixins)... {}
    
private:
    double x, y, z;
};

struct Color { char red, green, blue; };
struct Label { std::string name; };

Point<Color, Label> p({0x7f, 0, 0x7f}, {"center"});
```
A function parameter pack for a function template may depend on 
template parameter packs declared in that template, 
which allows the function template to accept an arbitrary number of call arguments 
without losing type information:
```c++
template <typename ... Types>
void print(Types ... values);

int main()
{
    std::string welcome("Welcome to ");
    print(welcome, "C++", 2011, '\n');  //calls print<std::string, char const *, int, char>
}
```
There is a syntactic ambiguity between an unnamed function parameter pack 
appearing at the end of a parameter list and a C-style `vararg` parameter: 
```c++
template <typename T> 
void c_style(int, T ...);

template <typename ... T> 
void pack(int, T ...);
```
In the first case, the `T ...` is treated as `T, ...`: 
an unnamed parameter of type `T` followed by a C-style `vararg` parameter. 
In the second case, the `T ...` construct is treated as a function parameter pack 
because `T` is a valid expansion pattern. 
The disambiguation can be forced by adding a comma before the ellipsis 
(which ensures the ellipsis is treated as a C-style `vararg` parameter) 
or by following the `...` by an identifier, 
which makes it a named function parameter pack. 
Note that in generic lambdas, a trailing `...` will be treated as denoting a parameter pack 
if the type that immediately precedes it (with **no** intervening comma) contains `auto`.

##### 12.4.4 Multiple and Nested Pack Expansions

The pattern of a pack expansion can be arbitrarily complex 
and may include multiple, nested, distinct parameter packs.
When instantiating a pack expansion containing multiple parameter packs, 
all of the parameter packs must have the _same length_.
```c++
template <typename F, typename ... Types>
void forwardCopy(F f, Types const & ... values) 
{
    f(Types(values)...);
}
```
```c++
template <typename ... OuterTypes>
class Nested 
{
    template <typename ... InnerTypes>
    void f(InnerTypes const & ... innerValues) 
    {
        g(OuterTypes(InnerTypes(innerValues)...)...);
    }
};
```

##### 12.4.5 Zero-Length Pack Expansions

The syntactic interpretation of pack expansions can be a useful tool 
for understanding how an instantiation of a variadic template 
will behave with different numbers of arguments. 
However, the syntactic interpretation often **fails** in the presence of _zero-length argument packs_. 
To illustrate this, consider the Point class template syntactically substituted with zero arguments: 
```c++
template <typename ... Mixins>
struct Point : public Mixins...
{
    Point(Mixins... mixins) : Mixins(mixins)... {}
};
```
```c++
template <>
struct Point : 
{
    Point() : {}
};
```
The code as written above is ill-formed, 
since the template parameter list is now empty 
and the empty base class and base class initializer lists 
each have a stray colon character.


Pack expansions are actually semantic constructs, 
and the substitution of an argument pack of any size does **not** affect how the pack expansion 
(or its enclosing variadic template) is parsed. 
Rather, when a pack expansion expands to an empty list, 
the program behaves (semantically) as if the list were **not** present. 
The instantiation `Point<>` ends up having **no** base classes, 
and its default constructor has **no** base class initializers but is otherwise well-formed. 
This semantic rules holds even when the syntactic interpretation of zero-length pack expansion 
would be well-defined (but different) code:
```c++
template <typename T, typename ... Types>
void g(Types ... values)
{
    T v(values...);
}
```
The variadic function template `g` creates a value `v` 
that is direct-initialized from the sequence of values it is given. 
If that sequence of values is empty, the declaration of `v` looks 
syntactically like a function declaration `T v()`. 
However, since substitution into a pack expansion is semantic 
and can **not** affect the kind of entity produced by parsing, 
`v` is initialized with zero arguments (value-initialization). 


There is a similar restriction on members of class templates and nested classes within class templates: 
If a member is declared with a type that does not appear to be a function type, 
but after instantiation the type of that member is a function type, 
the program is ill-formed because the semantic interpretation of the member 
has changed from a data member to a member function. 

##### 12.4.6 Fold Expressions

A recurring pattern in programming is the _fold_ of an operation on a sequence of values. 
For example, a *right fold o*f a function fn over a sequence 
`x[1], x[2], ..., x[n-1], x[n]` is given by `fn(x[1], fn(x[2], fn(..., fn(x[n - 1], x[n])...)))`. 
While exploring a new language feature, 
the C++ committee ran into the need to deal with such constructs 
for the special case of a logical binary operator (i.e., `&&` or `||`) applied to a pack expansion. 
Without an extra feature, we might write the following code to achieve that for the `&&` operator:
```c++
bool and() 
{ 
    return true; 
}

template <typename T>
bool and(T cond) 
{  
    return cond; 
}

template <typename T, typename ... Ts>
bool and(T cond, Ts ... conds) 
{
    return cond && and(cond...);
}
```
With C++17, a new feature called _fold expressions_ was added (see Section 4.2). 
It applies to all binary operators **except** `.,` `->`, and `[]`. 
Given an unexpanded expression pattern _pack_ and a non-pattern expression _value_,
C++17 allows us to write for a _right fold_ of the operator `op` (_binary right fold_): 
```c++
(pack op ... op value)
```
or for a _left fold_ of the operator `op` (_binary left fold_):
```c++
(value op ... op pack)
```
Note that the parentheses are required here. 


The fold operation applies to the sequence 
that results from expanding the pack
and adding value as either the last element of the sequence (for a right fold) 
or the first element of the sequence (for a left fold). 


With this feature available: 
```c++
template <typename ... T> bool g() 
{
    return and(trait<T>()...);
}
```
turns into
```c++
template <typename ... T> 
bool g() 
{
    return (trait<T>() && ... && true);
}
```
Fold expressions are pack expansions. 
Note that if the pack is empty, 
the type of the fold expression can still be determined from the non-pack operand (`value` in the forms above). 


However, the designers of this feature also wanted an option to leave out the `value` operand. 
Two other forms are therefore available in C++17: 
The _unary right fold_
```c++
(pack op ...)
```
and the _unary left fold_
```c++
(... op pack)
```
Again, the parentheses are required. 
Clearly this creates a problem for empty expansions: 
How do we determine their type and value? 
The answer is that an empty expansion of a unary fold is generally an error, 
with three exceptions:
- An empty expansion of a unary fold of `&&` produces the value `true`.
- An empty expansion of a unary fold of `||` produces the value `false`.
- An empty expansion of a unary fold of the comma operator `,` produces a `void` expression. 


Note that this will create surprises if you overload one of these special operators
in a somewhat unusual way:
```c++
struct BooleanSymbol {};

BooleanSymbol operator||(BooleanSymbol, BooleanSymbol);

template <typename ... BTs> 
void symbolic(BTs ... ps) 
{
    BooleanSymbol result = (ps || ...);
}
```
Suppose we call `symbolic` with types that are derived from `BooleanSymbol`.
For all expansions, the result will produce a `BooleanSymbol` value
**except** for the empty expansion, which will produce a bool `value`. 


Because overloading these three special operators is unusual, 
this problem is fortunately rare (but subtle). 
The original proposal for fold expressions included empty expansion values 
for more common operators like `+` and `*`, which would have caused more serious problems. 


We therefore generally caution against the use of unary fold expressions, 
and recommend using binary fold expressions instead 
(with an explicitly specified empty expansion value). 


#### 📌 12.5 Friends


The basic idea of friend declarations is a simple one: 
Identify classes or functions that have a privileged connection 
with the class in which the friend declaration appears. 
Matters are somewhat complicated, by two facts:
1. A friend declaration may be the only declaration of an entity. 
2. A friend function declaration can be a definition. 

##### 12.5.1 Friend Classes of Class Templates

Friend class declarations can **not** be definitions and therefore are rarely problematic. 
In the context of templates, the only new facet of friend class declarations 
is the ability to name a particular instance of a class template as a friend:
```c++
template <typename T>
class Node;

template <typename T>
class Tree 
{
    friend class Node<T>;
};
```
Note that the class template must be visible at the point 
where one of its instances is made a friend of a class or class template. 
With an ordinary class, there is **no** such requirement:
```c++
template <typename T>
class Tree 
{
    friend class Factory;  // OK even if first declaration of Factory
    friend class Node<T>;  // ERROR if Node is not visible
};
```
Section 13.2.2 has more to say about this. 


One application is the declaration of other class template instantiations to be friends:
```c++
template <typename T>
class Stack 
{
public:
    // to get access to private members of Stack<U> for any type U:
    template <typename> friend class Stack;
    
    // assign stack of elements of type U
    template <typename U>
    Stack<T> & operator= (Stack<U> const &);

    ...
};
```
C++11 also added syntax to make a template parameter a friend:
```c++
template <typename T>
class Wrap 
{
    friend T;
    ...
};
```
This is valid for any type `T` but is ignored if `T` is **not** actually a class type. 

##### 12.5.2 Friend Functions of Class Templates

An instance of a function template can be made a friend 
by making sure the name of the friend function is followed by angle brackets. 
The angle brackets can contain the template arguments, 
but if the arguments can be deduced, 
the angle brackets can be left empty:
```c++
template <typename T1, typename T2>
void combine(T1, T2);

class Mixer 
{
    // OK: T1 = int &, T2 = int &
    friend void combine<>(int &, int &);
    
    // OK: T1 = int, T2 = int
    friend void combine<int, int>(int, int);
    
    // OK: T1 = char T2 = int
    friend void combine<char>(char, int);
    
    // OK: definition of a specialization
    friend void combine<>(long, long) {}
    
    // ERROR: does not match combine template
    friend void combine<char>(char &, int);
};
```
Note that we can **not** define a template instance (at most, we can define a specialization), 
and hence a friend declaration that names an instance can **not** be a definition.
If the name is **not** followed by angle brackets, there are two possibilities:
1. If the name isn't qualified (in other words, it doesn't contain `::`), 
   it **never** refers to a template instance. 
   If no matching non-template function is visible at the point of the friend declaration, 
   the friend declaration is the first declaration of that function. 
   The declaration could also be a definition.
2. If the name is qualified (it contains `::`), 
   the name _must_ refer to a previously declared function or function template. 
   A matching function is preferred over a matching function template. 
   However, such a friend declaration can **not** be a definition. 
   An example may help clarify the various possibilities:
```c++
void multiply(void *);

template <typename T>
void multiply(T);

class Comrades 
{
    // defines a new function ::multiply(int)
    friend void multiply(int) {}

    // refers to the ordinary function above,
    // not to the multiply<void *> instance
    friend void ::multiply(void *);
    
    // refers to an instance of the template
    friend void ::multiply(int);

    // qualified names can also have angle brackets,
    // but a template must be visible
    friend void ::multiply<double *>(double *);

    // ERROR: a qualified friend can not be a definition
    friend void ::error() {}
};
```
In our previous examples, we declared the friend functions in an ordinary class. 
The same rules apply when we declare them in class templates, 
but the template parameters may participate in identifying the function that is to be a friend:
```c++
template <typename T>
class Node 
{
    Node<T> * allocate();
    ...
};

template <typename T>
class List 
{
    friend Node<T> * Node<T>::allocate();
    ...
};
```
A friend function may also be defined within a class template, 
in which case it is only instantiated when it is actually used. 
This typically requires the friend function 
to use the class template itself in the type of the friend function, 
which makes it easier to express functions on the class template 
that can be called as if they were visible in namespace scope
(name found via ADL):
```c++
template <typename T>
class Creator 
{
    // Every T instantiates a different function ::feed
    friend void feed(Creator<T>) {}
};

int main()
{
    Creator<void> one;
    feed(one);  // Instantiates ::feed(Creator<void>)
    
    Creator<double> two;
    feed(two);  // Instantiates ::feed(Creator<double>)
}
```
In this example, every instantiation of `Creator` generates a different function. 
Note that even though these functions are generated as part of the instantiation of a template, 
the functions themselves are ordinary functions, `not` instances of a template.
However, they are considered _templated entities_ (see Section 12.1) 
and their definition is instantiated only when used. 
Also note that because the body of these functions is defined inside a class definition, 
they are implicitly inline.
Hence, it is **not** an error for the same function to be generated in two different translation units. 
Section 13.2.2 Section 21.2.1 have more to say about this topic.

##### 12.5.3 Friend Templates

Usually when declaring a friend that is an instance of a function or a class template,
we can express exactly which entity is to be the friend. 
Sometimes it is useful to express that all instances of a template are friends of a class. 
This requires a _friend template_:
```c++
class Manager 
{
    template <typename T>
    friend class Task;
    
    template <typename T>
    friend void Schedule<T>::dispatch(Task<T> *);
    
    template <typename T>
    friend int ticket() 
    {
        return ++Manager::counter;
    }
    
    static int counter;
};
```
Just as with ordinary friend declarations, a friend template can be a definition 
only if it names an unqualified function name that is **not** followed by angle brackets
(except defining specializations of existing templates). 


A friend template can declare only primary templates and members of primary templates. 
Any partial specializations and explicit specializations associated with a primary template 
are automatically considered friends too. 






### 🎯 Chapter 13 Names in Templates


When a C++ compiler encounters a name, 
it must "look it up" to identify the entity being referred. 
From an implementer's point of view, C++ is a hard language in this respect. 
Consider the C++ statement `x * y;`. 
If `x` and `y` are the names of variables, this statement is a multiplication, 
but if `x` is the name of a type, then the statement declares `y` as a pointer to an entity of type `x`. 


This small example demonstrates that C++ (like C) is a _context-sensitive_ language: 
A construct can **not** always be understood without knowing its wider context. 
How does this relate to templates? 
Well, templates are constructs that must deal with multiple wider contexts: 
1. The context in which the template appears;
2. The context in which the template is instantiated;
3. The contexts associated with the template arguments for which the template is instantiated. 


#### 📌 13.1 Name Taxonomy


C++ classifies names in a large variety of ways. 
Fortunately, you can gain good insight into most C++ template issues 
by familiarizing yourself with two major naming concepts:
1. A name is a _qualified name_ if the scope to which it belongs
   is explicitly denoted using a scope-resolution operator (`::`) or a member access operator (`.` or `->`). 
   For example, `this->count` is a qualified name, 
   but `count` is not (even though the plain `count` might actually refer to a class member). 
2. A name is a _dependent name_ if it depends in some way on a template parameter. 
   For example, `std::vector<T>::iterator` is usually a dependent name if `T` is a template parameter, 
   but it is a non-dependent name if `T` is a known type alias (such as the `T` from `using T = int`). 

- **Identifier**:  
  A name that consists solely of an uninterrupted sequences of letters, underscores `_`, and digits. 
  It can **not** start with a digit, and some identifiers are reserved for the implementation: 
  You should **not** introduce them in your programs 
  (as a rule of thumb, avoid leading underscores and double underscores). 
  The concept of "letter" should be taken broadly and includes special _universal character names_ 
  (UCNs) that encode glyphs from non-alphabetical languages.
- **Operator-function-id**:  
  The keyword `operator` followed by the symbol for an operator. 
  for example, `operator new` and `operator []`. 
- **Conversion-function-id**:  
  Used to denote a user-defined implicit conversion operator. 
  For example, `operator int &`, 
  which could also be obfuscated as `operator int bitand`. 
- **Literal-operator-id**:   
  Used to denote a user-defined literal operator. 
  For example, `operator ""_km`, which will be used when writing a literal such as `100_km`.
- **Template-id**:  
  The name of a template followed by template arguments enclosed in angle brackets. 
  For example, `List<T, int, 0>`. 
  A template-id may also be an `operator-function-id` or a `literal-operator-id`
  followed by template arguments enclosed in angle brackets. 
  For example, `operator+<X<int>>`.
- **Unqualified-id**:  
  The generalization of an identifier.
  It can be any of the above 
  (`identifier`, `operator-function-id`, `conversion-function-id`, `literal-operator-id`, or `template-id`) 
  or a "destructor name" 
  (e.g., notations like `~Data` or `~List<T, T, N>`). 
- **Qualified-id**:  
  An `unqualified-id` that is qualified with the name of a class, enum, or namespace, 
  or just with the global scope resolution operator. 
  Note that such a name itself can be qualified.
  Examples are `::X`, `S::x`, `Array<T>::y`, and `::N::A<T>::z`. 
- **Qualified Name**:  
  This term is not defined in the standard, 
  but we use it to refer to names that undergo _qualified lookup_. 
  Specifically, this is a `qualified-id` or an `unqualified-id` 
  that is used after an explicit member access operator (`.` or `->`). 
  Examples are `S::x`, `this->f`, and `p->A::m`. 
  However, just `class_mem` in a context that is implicitly equivalent to `this->class_mem` is **not** a qualified name: 
  The member access must be explicit. 
- **Unqualified Name**:  
  An `unqualified-id` that is **not** a qualified name.
  This is not a standard term but corresponds to names that undergo 
  what the standard calls unqualified lookup.
- **Name**:  
  Either a qualified or an unqualified name.
- **Dependent Name**:  
  A name that depends in some way on a template parameter.
  Typically, a qualified or unqualified name that explicitly contains a template parameter is dependent. 
  Furthermore, a qualified name that is qualified by a member access operator (`.` or `->`) 
  is typically dependent if the type of the expression on the left of the access operator is type-dependent, 
  a concept that is discussed in Section 13.3.6. 
  In particular, `b` in `this->b` is generally a dependent name when it appears in a template. 
  Finally, a name that is subject to _Argument-Dependent Lookup (ADL)_ (described in Section 13.2), 
  such as ident in a call of the form `ident(x, y)` or `+` in the expression `x + y`, 
  is a dependent name if and only if any of the argument expressions is type-dependent. 
- **Non-dependent Name**:  
  A name that is **not** a dependent name by the above description. 


#### 📌 13.2 Looking Up Names


Qualified names are looked up in the scope implied by the qualifying construct. 
If that scope is a class, then base classes may also be searched. 
However, enclosing scopes of the qualified scope are **not** considered when looking up qualified names.
```c++
int x;

class B 
{
public:
    int i;
};

class D : public B {};

void f(D * pd)
{
    pd->i = 3;  // finds B::i
    D::x = 2;   // ERROR: No x in D and its children scopes
}
```
In contrast, unqualified names are typically looked up in successively more enclosing scopes.  
In member function definitions, 
the scope of the class and its base classes is searched before any other enclosing scopes. 
This is called _ordinary lookup_. 
```c++
extern int count;               // #1

int lookup_example(int count)   // #2
{
    if (count < 0) 
    {
        int count = 1;          // #3
        lookup_example(count);  // refers to #3
    }
    
    return count + ::count;     // refers to #2 + #1
    
}
```


#### 📌 13.2.1 Argument-Dependent Lookup


Unqualified names sometimes undergo _Argument-Dependent Lookup (ADL)_ in addition to ordinary lookup.
In C++98/C++03, this was also called _Koenig Lookup_ (or _Extended Koenig Lookup_)
after Andrew Koenig, who first proposed a variation of this mechanism.


Suppose we need to apply function template `max` to a type defined in another namespace:
```c++
template <typename T>
inline T max(T a, T b)
{
    return b < a ? a : b;
}

namespace BigMath 
{

class BigNumber {};

bool operator<(BigNumber const &, BigNumber const &);

}  // namespace BigMath

using BigMath::BigNumber;

void g(BigNumber const & a, BigNumber const & b)
{
    BigNumber x = ::max(a, b);
}
```
The problem here is that the `max` template is unaware of the `BigMath` namespace. 
Ordinary lookup would **not** find the `operator<` applicable to values of type `BigNumber`.
Without ADL, this greatly reduces the applicability of templates in the presence of C++ namespaces. 


ADL applies primarily to unqualified names of non-member functions 
in a function call or operator invocation. 
ADL does **not** happen if ordinary lookup finds
- The name of a member function;
- The name of a variable;
- The name of a type;
- The name of a block-scope function declaration.
ADL is also **inhibited** if the name of the function to be called 
is enclosed in parentheses.


Otherwise, if the name is followed by a list of argument expressions enclosed in parentheses, 
ADL proceeds by looking up the name in _associated namespaces_ and _associated classes_ of the call arguments.
ADL then looks up the name in all the associated namespaces as if
the name had been qualified with each of these namespaces in turn,
**except** that using directives are **ignored**.


The precise definition of the set of _associated namespaces_ and _associated classes_ for a given type 
is determined by the following rules:
- **Built-in types**: 
  - Empty set.
- **Pointer types** and **array types**:
  - Associated namespaces and classes: That of the underlying type.
- **Enumeration** types: 
  - Associated namespace: Namespace in which the enumeration is declared.
- **Class members**:
  - Associated class: The enclosing class
- **Class types** (including union types): 
  - Associated classes: 
    The type itself, the enclosing class, and any direct and indirect base classes. 
  - Associated namespaces: 
    The namespaces in which the associated classes are declared. 
  - If the class is a class template instance, 
    then the types of the template type arguments and the classes and namespaces 
    in which the template template arguments are declared are also included. 
- **Function types**:
  - Associated namespaces and classes: 
    The namespaces and classes associated with all the parameter types 
    and those associated with the return type.
- **Pointer-to-member-of-class-`X` types**, 
  - Associated namespaces and classes: 
    Those associated with `X` in addition to those associated with the type of the member. 
    (If it is a pointer-to-member-function type, then the parameter and return types can contribute too.) 

```c++
namespace X 
{

template <typename T> 
void f(T);
    
}  // namespace X

namespace N 
{
    using namespace X;
    
    enum E { e1 };
    
    void f(E) 
    {
        std::cout << "N::f(N::E)\n";
    }
}  // namespace N

void f(int)
{
    std::cout << "::f(int)\n";
}

int main()
{
    ::f(N::e1);  // Qualified function name, no ADL
    f(N::e1);    // Ordinary lookup finds ::f(int) and ADL finds N::f(E), 
                 // the latter is preferred. 
                 // Note that using directives are ignored during ADL, 
                 // so X::f(T) is not found. 
}
```
Note that in this example, 
the using directive in namespace `N` is **ignored**when ADL is performed. 
Hence `X::f()` is **never** even a candidate for the call.

##### 13.2.2 Argument-Dependent Lookup of Friend Declarations

A friend function declaration can be the first declaration of the nominated function.
(And it could also be the only declaration if the friend function is defined in friend declaration.) 
If this is the case, then the function is assumed to be declared in the nearest namespace scope 
(which may be the global scope) enclosing the class containing the friend declaration. 
However, such a friend declaration is **not** directly visible in that scope:
```c++
template <typename T>
class C 
{
    friend void f();
    friend void f(C<T> const &);
};

void g(C<int> * p)
{ 
    f();    // ERROR: f() can not be found
    f(*p);  // OK: f(C<int> const &) visible here via ADL
}
```
If friend declarations were visible in the enclosing namespace, 
then instantiating a class template may make visible the declaration of ordinary functions. 
This would lead to surprising behavior: 
The call `f()` would result in a compilation error 
unless an instantiation of the class `C` occurred earlier in the program!


On the other hand, it can be useful to declare (and define) a function in a friend declaration only 
(see Section 21.2.1 for a technique that relies on this behavior). 
Such a function can be found when the class of which they are a friend 
is among the associated classes considered by ADL. 


Reconsider our last example. 
The call `f()` has **no** associated classes or namespaces because there are no arguments: 
It is an invalid call in our example. 
However, the call `f(*p)` does have the associated class `C<int>`
and the global namespace is also associated 
(because this is the namespace in which the class template `C` is declared). 
Therefore, the second friend function declaration could be found
provided the class `C<int>` was actually fully instantiated prior to the call. 


To ensure this, it is assumed that a call involving a lookup for friends in associated classes 
actually causes the class to be instantiated (if not done already). 
Although this was clearly intended by those who wrote the C++ standard, 
it is **not** clearly spelled out in the standard.


The ability of Argument-Dependent Lookup to find friend declarations and definition 
is sometimes referred to as _friend name injection_. 
However, this term is somewhat misleading, 
because it is the name of a pre-standard C++ feature 
that did in fact "inject" the names of friend declarations into the enclosing scope, 
making them visible to normal name lookup. 
In our example above, this would mean that both calls would be well-formed. 

##### 13.2.3 Injected Class Names

The name of a class is injected into the class scope of that class itself 
and is therefore accessible as an unqualified name in that scope. 
However, it is **not** accessible as a qualified name 
because this is the notation used to denote the constructors. 
```c++
int C;

class C 
{
public:
    static int f() 
    {
        return sizeof(C);
    }

private:
    int i[2];
};

int f()
{
    return sizeof(C);
}

int main()
{
    std::cout << "C::f() = " << C::f() << '\n'
              << " ::f() = " << ::f() << '\n';
}
```
The member function `C::f()` returns the size of type `C`, 
whereas the function `::f()` returns the size of the variable `C`
(in other words, the size of an `int` object). 


Class templates also have injected class names. 
However, they're stranger than ordinary injected class names: 
They can be followed by template arguments (in which case they are injected class _template_ names), 
But, when they are **not** followed by template arguments, 
depending on the context, they could represent:
- An alias for the class type being defined
- The template type with its parameters as its arguments 
  (specialization arguments for a partial specialization). 
```c++
template <template <typename> class TT>
class X {};

template <typename T>
class C
{
    C * a;        // OK: same as "C<T> * a;"
    C<void> & b;  // OK 
    X<C> c;       // OK: C without a template argument list denotes the template C
    X<::C> d;     // OK: ::C is not the injected class name and therefore always denotes the template
};
```
The injected class name for a variadic template has an additional wrinkle: 
If the injected class name were directly formed 
by using the variadic template's template parameters as the template arguments, 
the injected class name would contain template parameter packs that have **not** been expanded. 
Therefore, when forming the injected class name for a variadic template, 
the template argument that corresponds to a template parameter pack 
is a pack expansion whose pattern is that template parameter pack:
```c++
template <int I, typename ... Ts> 
class V 
{
    V * a;         // OK: same as "V<I, Ts...> * a;"
    V<0, void> b;  // OK
};
```

##### 13.2.4 Current Instantiations

The injected class name of a class or class template is effectively 
an alias for the type being defined. 
Inside a class template or a nested class within a class template, 
each template instantiation produces a different type. 
This property means that the injected class name refers to 
the same instantiation of the class template 
rather than some other specialization of that class template.  
The same holds for nested classes of class templates. 


Within a class template, 
the injected class name or any type that is equivalent to the injected class name
is said to refer to a _current instantiation_. 
Types that depend on a template parameter (i.e., _dependent types_) 
but do **not** refer to a current instantiation 
are said to refer to an _unknown specialization_, 
which may be instantiated from the same class template 
or some entirely different class template. 
```c++
template <typename T> 
class Node
{
    using Type = T;
    
    Node * next;            // Node refers to the current instantiation
    Node<Type> * previous;  // Node<Type> refers to the current instantiation too
    Node<T *> * parent;     // Node<T *> refers to an unknown instantiation
};
```
In the presence of nested classes and class templates. 
The injected class names of enclosing classes and class templates (or types equivalent to them) 
refer to a current instantiation, 
while the names of other nested classes or class templates do **not**: 
```c++
template <typename T>
class C
{
    using Type = T;
    
    struct I
    {
        C * c;         // C refers to a current instantiation
        C<Type> * c2;  // C<Type> refers to a current instantiation
        I * i;         // I refers to a current instantiation
    };
    
    struct J
    {
        C * c;         // C refers to a current instantiation
        C<Type> * c2;  // C<Type> refers to a current instantiation
        I * i;         // I refers to an unknown instantiation because I does not enclose J
        J * j;         // J refers to a current instantiation
    };
};
```
When a type refers to a current instantiation, 
the contents of that instantiated class are guaranteed to be instantiated 
from the class template or nested class thereof that is currently being defined. 


This hints another way to determine whether a type `X` within the definition of a class template 
refers to a current instantiation or an unknown specialization: 
If another programmer can write an explicit specialization (described in detail in Chapter 16) 
such that `X` refers to that specialization, then `X` refers to an unknown specialization. 


For example, consider the instantiation of the type `C<int>::J` in the context of the above example. 
We could not explicitly specialize `C` or `J` for those defined in `J`, 
so, the references to `J` and `C<int>` within `J` refer to a current instantiation. 
On the other hand, one could write an explicit specialization for `C<int>::I` as follows:
```c++
template <> 
struct C<int>::I 
{
    // definition of the specialization
};
```
Here, the specialization of `C<int>::I` provides a completely different definition
than the one that was visible from the definition of `C<T>::J`, 
so the `I` inside the definition of `C<T>::J` refers to an unknown specialization. 


#### 📌 13.3 Parsing Templates


Two fundamental activities of compilers for most programming languages are
- _Tokenization_ (also called _Scanning_ or _Lexing_);
- Parsing. 


The tokenization process reads the source code as a sequence of characters 
and generates a sequence of tokens from it. 
A parser will then find known patterns in the token sequence 
by recursively reducing tokens or previously found patterns into grammar.

##### 13.3.1 Context Sensitivity in Non-templates

Tokenizing is easier than parsing. 
Parsing is a subject for which a solid theory has been developed, 
and many useful languages are not hard to parse using this theory. 
However, the theory works best for _Context-Free Grammar/Languages (CFG)_, 
and we have already noted that C++ is _context sensitive_. 


To handle this, a C++ compiler will couple a symbol table to the tokenizer and parser: 
When a declaration is parsed, it is entered in the symbol table. 
When the tokenizer finds an identifier, 
it looks it up and annotates the resulting token if it finds a type. 


For example, if the C++ compiler sees
```c++
x *
```
the tokenizer looks up `x`. 
If it finds a type, 
the parser receives from the tokenizer
```
identifier, type, x, symbol, *
```
concludes that a declaration has started. 
However, if `x` is **not** found to be a type,
then the parser receives from the tokenizer
```c++
identifier, non-type, x, symbol, *
```
and the construct can be parsed validly only as a multiplication. 
The details of these principles are dependent on the particular implementations. 


Another example of context sensitivity:
```c++
X<1>(0)
```
If `X` is the name of a class template, 
then the previous expression casts the integer `0` to the type `X<1>` generated from that template. 
If X is **not** a template, then the previous expression is equivalent to
```c++
(X < 1) > 0
```
In other words, `X` is compared with `1`, 
and the result of that comparison is then compared with `0`. 
Although code like this is rarely used, it is valid C++ (and valid C, for that matter). 
A C++ parser will therefore look up names appearing before a `<` and treat the `<` as an angle bracket 
only if the name is known to be that of a template. 
Otherwise, the `<` is treated as an ordinary less-than operator. 


This form of context sensitivity is an unfortunate consequence of 
having chosen angle brackets to delimit template argument lists. 
Here is another such consequence:
```c++
template <bool B>
class Invert 
{
public:
    static bool const result = !B;
};

void g()
{
    bool test = Invert<(1 > 0)>::result;  // parentheses required!
}
```
If the parentheses in `Invert<(1 > 0)>` were omitted,
the greater-than symbol would be mistaken for the closing of the template argument list. 
This would make the code invalid because the compiler would read it to be equivalent to
```c++
((Invert<1>))0>::result
```
Note the double parentheses to avoid parsing `(Invert<1>)0` as a cast operation, 
yet another source of syntactic ambiguity. 


The tokenizer isn't spared problems with the angle-bracket notation either.
For example, in
```
List<List<int>> a;
~~~~~~~~~~~~~^~~~~  no space between right angle brackets
```
the two `>` characters combine into a right-shift token `>>` 
and hence are **never** treated as two separate tokens by the tokenizer. 
This is a consequence of the maximum munch tokenization principle: 
A C++ implementation must collect as many consecutive characters as possible into a token.
Specific exceptions were introduced to address 
tokenization issues described in this section.


As mentioned in Section, since C++11, the C++ standard specifically calls out this case, 
where a nested template-id is closed by a right-shift token `>>`, 
and, within the parser, treats the right shift as being equivalent to 
two separate right angle brackets `>` and `>` to close two template-ids at once. 


The 1998 and 2003 versions of the C++ standard did **not** support this "angle bracket hack". 
However, the need to introduce a space between the two consecutive right angle brackets
was such a common stumbling block for beginning template users
that the committee decided to codify this hack in the 2011 standard.


This change silently changes the meaning of some admittedly contrived programs: 
```c++
template <int I> 
struct X 
{
    static int const c = 2;
};

template <> 
struct X<0> 
{
    typedef int c;
};

template <typename T> 
struct Y
{
    static int const c = 3;
};

static int const c = 4;

int main()
{
    std::cout << (Y<X<1> >::c >::c>::c) << ' ';
              << (Y<X< 1>>::c >::c>::c) << '\n';
}
```
This is a valid C++98 program that outputs `0 3`
It is also a valid C++11 program, 
but there the angle bracket hack makes the two parenthesized expressions equivalent,
and the output is `0 0`. 
Some compilers that provide a C++98 or C++03 mode keep the C++11 behavior in those modes 
and thus print `0 0` even when formally compiling C++98/C++03 code.


A similar problem existed because of the existence of the digraph `<:` 
as an alternative for the source character `[` (which is not available on some traditional keyboards): 
```c++
template <typename T> 
struct G {};

struct S;
G<::S> gs;  // valid since C++11, but an error before that
```
Before C++11, that last line of code was equivalent to `G[:S> gs;`,
which is clearly invalid. 
Another "lexical hack" was added to address that problem: 
When a compiler sees the characters `<::` not immediately followed by `:` or `>`, 
the leading pair of characters `<:` is **not** treated as a digraph token equivalent to `[`.
This is therefore an exception to the aforementioned maximum munch principle. 


This _digraph hack_ can make previously valid (but somewhat contrived) programs invalid: 
```c++
#define F(X) X ## :

int a[] = {1, 2, 3}; 
int i = 1;

int n = a F(<::)i];  // valid in C++98/C++03, but not in C++11
```

##### 13.3.2 Dependent Names of Types And `typename` Prefix

The problem with names in templates is that they can **not** always be sufficiently classified. 
In particular, one template can **not** look into another template 
because the contents of that other template can be made invalid by an explicit specialization. 
The following contrived example illustrates this:
```c++
template <typename T>
class Trap
{
public:
    enum { x };          // #1 x is not a type here
};

template <typename T>
class Victim
{
public:
    void poof()
    {
        Trap<T>::x * y;  // #2 declaration or multiplication?
    }

    int y;
};

template <>
class Trap<void>
{
public:
    using x = int;       // #3 x is a type here
};

void boom(Victim<void> & bomb)
{
    bomb.poof();
}
```
As the compiler is parsing line `#2`, 
it must decide whether it is seeing a declaration or a multiplication. 
This decision in turn depends on whether `Trap<T>::x` is a type name. 


`Trap<T>` is a _dependent name_ because it depends on the template parameter `T`. 
Moreover, `Trap<T>` refers to an _unknown specialization_, 
which means that the compiler can **not** safely look inside the primary template 
to determine whether the name `Trap<T>::x` is a type or not. 
Actually, another specification `Trap<void>` just contradicts with the primary template
on what `Trap<T>::x` is. 


However, as illustrated by the example, 
name lookup into an unknown specialization is still a problem. 
The language definition resolves this problem by specifying that 
in general a dependent qualified name does **not** denote a type 
unless that name is prefixed with the keyword `typename`. 
If it turns out that the name is not the name of a type after template argument substitution, 
the program is invalid and your C++ compiler should complain at instantiation time. 
Note that this use of `typename` differs from the use to denote template type parameters. 
Unlike type parameters, you can **not** equivalently replace `typename` with `class`.


The `typename` prefix to a name is _required_ when the name satisfies all of the following conditions: 
1. It is qualified and **not** itself followed by `::` to form a more qualified name.
2. It is **not** part of an `elaborated-type-specifier`, 
   i.e., a type name that starts with one of the keywords `class`, `struct`, `union`, or `enum`. 
3. It is **not** used in a _list of base class specifications_ 
   or in a _list of member initializers_ introducing a constructor definition. 
   Syntactically, only type names are permitted within these contexts, 
   so a qualified name is always assumed to name a type.
4. It is dependent on a template parameter.
5. It is a member of an unknown specialization, 
   meaning that the type named by the qualifier refers to an unknown specialization. 


Furthermore, the `typename` prefix is _**not** allowed_ 
unless at least the first two previous conditions hold. 

##### 13.3.3 Dependent Names of Templates And `template` Prefix

A problem very similar to the one encountered in the previous section occurs 
when a name of a template is dependent. 
In general, a C++ compiler is required to treat a `<` following the name of a template 
as the beginning of a template argument list. 
Otherwise, it is a less-than operator. 
As is the case with type names, a compiler has to assume that a dependent name does **not** refer to a template 
unless the programmer provides extra information using the keyword `template`:
```c++
template <typename T>
class Shell
{
public:
    template <int N>
    class In
    {
    public:
        template <int M>
        class Deep
        {
        public:
            virtual void f() {}
        };
    };
};

template <typename T, int N>
class Weird
{
public:
    void case1(typename Shell<T>::template In<N>::template Deep<N> * p)
    {
        p->template Deep<N>::f();  // inhibit virtual call
    }

    void case2(typename Shell<T>::template In<N>::template Deep<N> & p)
    {
        p.template Deep<N>::f();   // inhibit virtual call
    }
};
```
This example shows how all the operators that can qualify a name (`::`, `->`, and `.`) 
may need to be followed by the keyword `template`. 

Specifically, this is the case whenever 
the type of the name or expression preceding the qualifying operator 
is dependent on a template parameter and refers to an unknown specialization, 
and the name that follows the operator is a `template-id` 
(in other words, a template name followed by template arguments in angle brackets). 
```c++
dependent-name-to-unknown-specification ::/->/. template template-id
```
For example, in the expression
```c++
p.template Deep<N>::f()
```
the type of `p` depends on the template parameter `T`. 
Consequently, a C++ compiler can `not` look up `Deep` to see if it is a template, 
and we must explicitly indicate that `Deep` is the name of a template by inserting the prefix `template`. 


Without the `template` prefix, 
`Deep` will be parsed as non-template member of `p`, 
and `p.Deep<N>::f()` will be parsed as `((p.Deep) < N) > f()`. 
Note also that this may need to happen multiple times within a qualified name because qualifiers
themselves may be qualified with a dependent qualifier.


If the keyword `template` is omitted in cases such as these,
the opening and closing angle brackets are parsed as less-than and greater-than operators. 
As with the `typename` keyword, one can safely add the `template` prefix 
to indicate that the following name is a template-id, 
even if the `template` prefix is **not** strictly needed.

##### 13.3.4 Dependent Names in Using Declarations

Using declarations can bring in names from two places: namespaces and classes. 
The namespace case is not relevant in this context 
because there are **no** such things as _namespace templates_. 
Using declarations that bring in names from classes, on the other hand, 
can bring in names only from a base class to a derived class. 
Such using declarations behave like "symbolic links" or "shortcuts"
in the derived class to the base declaration, 
thereby allowing the members of the derived class to access the nominated name 
as if it were actually a member declared in that derived class. 
```c++
class BX 
{
public:
    void f(int);
    void f(char const *);
    void g();
};

class DX : private BX 
{
public:
    using BX::f;
};
```
By now you can probably perceive the problem when a using declaration brings in a name from a dependent class. 
Although we know about the name, we **don't** know whether it's the name of a type, a template, or something else:
```c++
template <typename T>
class BXT 
{
public:
    using Mystery = T;
    
    template <typename U>
    struct Magic {};
};

template <typename T>
class DXTT : private BXT<T> 
{
public:
    using typename BXT<T>::Mystery;
    
    Mystery * p;  // would be a syntax error without the earlier typename
};
```
Again, if we want a dependent name to be brought in by a using declaration to denote a type, 
we must explicitly say so by inserting the keyword `typename`.
Strangely, the C++ standard does **not** provide for a similar mechanism to mark such dependent names as templates.
```c++
template <typename T>
class DXTM : private BXT<T>
{
public:
    using BXT<T>::template Magic;  // ERROR: not standard
    Magic<T> * plink;              // SYNTAX ERROR: Magic is not a known template
};
```
The standardization committee has not been inclined to address this issue. 
However, C++11 alias templates do provide a partial workaround:
```c++
template <typename T>
class DXTM : private BXT<T>
{
public:
    template <typename U> 
    using Magic = typename BXT<T>::template Magic<U>;  // alias template
    
    Magic<int> * plink;                                // OK
}
```
This is a little unwieldy, but it achieves the desired effect for the case of class templates. 
The case of function templates (arguably less common) remains **unaddressed**, unfortunately. 

##### 13.3.5 ADL and Explicit Template Arguments

```c++
namespace N 
{

class X {};

template <int I> 
void select(X *) {}

}  // namespace N

void g(N::X * xp)
{
    select<3>(xp);
}
```
In this example, we may expect that the template `select` is found through ADL in the call `select<3>(xp)`. 
However, this is **not** the case because 
a compiler can **not** decide that `xp` is a function call argument 
until it has decided that `<3>` is a template argument list. 
Furthermore, a compiler can **not** decide that `<3>` is a template argument list 
until it has found `select` to be a template. 
Because this chicken-and-egg problem can not be resolved, 
the expression is parsed as `(select < 3) > (xp)`, which makes no sense. 


This example may give the impression that ADL is disabled for `template-id`s, but it is **not**. 
The code can be fixed by introducing a function template named select that is visible at the call:
```c++
void g(N::X * xp)
{
    template <typename T> 
    void select();
    
    select<3>(xp);
}
```
Even though it **doesn't** make any sense for the call `select<3>(xp)`, 
the presence of this function template ensures that `select<3>` will be parsed as a `template-id`. 
ADL will then find the function template `N::select`, and the call will succeed.

##### 13.3.6 Dependent Expressions

Like names, expressions themselves can be dependent on template parameters. 
An expression that depends on a template parameter can behave differently from one instantiation to the next. 
For example, selecting a different overloaded function or producing a different type or constant value.
In contrast, expressions that do not depend on a template parameter provide the same behavior in all instantiations.


An expression can be dependent on a template parameter in several ways: 
- **Type-Dependent Expressions**: Type of the expression itself can vary from one instantiation to the next;
- **Value-Dependent Expressions**: Produce different constant values from one instantiation to the next. 


_Type-dependent expressions_ are those whose type can vary from one instantiation to the next. 
For example, an expression that refers to a function parameter whose type is that of a template parameter:
```c++
template <typename T> 
void typeDependent1(T x)
{
    // the expression type-dependent, 
    // because the type of x can vary
    x;
}
```
Expressions that have type-dependent subexpressions are generally type-dependent themselves. 
For example, calling a function `f` with the argument `x`:
```c++
template <typename T> 
void typeDependent1(T x)
{
    // the expression type-dependent, 
    // because the type of x can vary
    f(x);
}
```
Here, note that type of `f(x)` can vary from one instantiation to the next 
both because `f` might resolve to a template whose result type depends on the argument type 
and because _two-phase lookup_ (discussed in Section 14.3.1) might find 
completely different functions named `f` in different instantiations. 


Expressions that produce different constant values from one instantiation to the next
are called _value-dependent expressions_. 
The simplest of which are those that refer to a non-type template parameter of non-dependent type:
```c++
template <int N> 
void valueDependent1()
{
    // the expression is value-dependent but not type-dependent,
    // because N has a fixed type but a varying constant value
    N;
}
```
Like type-dependent expressions, an expression is generally value-dependent 
if it is composed of other value-dependent expressions, 
so `N + N` or `f(N)` are also value-dependent expressions.


Some operations, such as `sizeof`, have a known result type, 
so they can turn a type-dependent operand into a value-dependent expression that is `not` type-dependent:
```c++
template <typename T> 
void valueDependent2(T x)
{
    sizeof(x);  // the expression is value-dependent but not type-dependent
}
```
The `sizeof` operation always produces a value of type `std::size_t` regardless of its input, 
so a `sizeof` expression is **never** type-dependent, even if its subexpression is type-dependent. 
However, the resulting constant value will vary from one instantiation to the next, 
so `sizeof(x)` is a value-dependent expression.


What if we apply `sizeof` on a value-dependent expression?
```c++
template <typename T> 
void maybeDependent(T const & x)
{
    sizeof(sizeof(x));
}
```
Here, the inner `sizeof` expression is value-dependent, as noted above. 
However, the outer sizeof expression always computes the size of a `std::size_t`, 
so both its type and constant value are consistent across all instantiations of the template, 
despite the innermost expression `(x)` being type-dependent. 
Any expression that involves a template parameter is an _instantiation-dependent_ expression, 
even if both its type and constant value are invariant across valid instantiations. 
However, an instantiation-dependent expression may turn out to be invalid when instantiated. 
For example, instantiating `maybeDependent` with an incomplete class type will trigger an error, 
because sizeof can **not** be applied to such types.


Type-dependence, value-dependence, and instantiation-dependence 
can be thought of as a series of increasingly more inclusive classifications of expressions. 
Any type-dependent expression is also considered to be value-dependent, 
because an expression whose type that varies from one instantiation to the next 
will naturally have its constant value vary from one instantiation to the next. 
Similarly, an expression whose type or value varies from one instantiation to the next 
depends on a template parameter in some way, 
so both type-dependent expressions and value-dependent expressions are instantiation-dependent.


As one proceeds from the innermost context (type-dependent expressions) to the outermost context, 
more of the behavior of the template is determined when the template is parsed 
and therefore can **not** vary from one instantiation to the next. 
For example, consider the call `f(x)`: 
If `x` is type-dependent, 
then `f` is a dependent name that is subject to two-phase lookup (Section 14.3.1), 
whereas if `x` is value-dependent but not type-dependent, 
`f` is a non-dependent name for which name lookup can be completely determined 
at the time that the template is parsed. 

##### 13.3.7 Compiler Errors

A C++ compiler is permitted (but **not** required!) to diagnose errors at the time the template is parsed 
when all of the instantiations of the template would produce that error. 
Let's expand on the `f(x)` example from the previous section to explore this further:
```c++
void f() {}

template <int x> 
void nondependentCall()
{
    // x is value-dependent, so f is non-dependent. 
    // This call will never succeed
    f(x);
}
```
Here, the call `f(x)` will produce an error in every instantiation 
because `f` is a non-dependent name and the only visible `f` accepts zero arguments, not one. 
A C++ compiler can produce an error when parsing the template 
or may wait until the first template instantiation: 
Commonly used compilers differ even on this simple example. 
One can construct similar examples with expressions that are instantiation-dependent but not value-dependent:
```c++
template <int N>
void instantiationDependentBound()
{
    constexpr int x = sizeof(N);
    constexpr int y = sizeof(N) + 1;
    int array[x - y];  // negative size in all instantiations
}
```


#### 📌 13.4 Inheritance and Class Templates


Class templates can inherit or be inherited from.
For many purposes, there is nothing significantly different between the template and non-template scenarios. 
However, there is one important subtlety 
when deriving a class template from a base class referred to by a dependent name. 
Let's first look at the somewhat simpler case of non-dependent base classes.

##### 13.4.1 Non-dependent Base Classes

In a class template, a non-dependent base class is one with a complete type
that can be determined without knowing the template arguments. 
In other words, the name of this base is denoted using a non-dependent name:
```c++
template <typename>
class Base
{
public:
    using T = int;
    
    int basefield;
};

// not a template case really
class D1 : public Base<Base<void>>
{
public:
    void f()
    {
        basefield = 3;
    }
};

// usual access to non-dependent base
template <typename T>
class D2 : public Base<double>
{
public:
    void f()
    {
        // usual access to inherited member
        basefield = 7;
    }
    
    // T is Base<double>::T (aka int), 
    // not the template parameter!
    T strange;
};
```
Non-dependent bases in templates behave very much like bases in ordinary non-template classes, 
but there is a slightly unfortunate surprise: 
When an unqualified name is looked up in the templated derivation, 
the non-dependent bases are considered _before_ the list of template parameters. 
This means that in the previous example, 
the member `strange` of the class template `D2` always has the type `T` 
corresponding to `Base<double>::T` (aka `int`). 
For example, the following function is **not** valid C++ (assuming the previous declarations):
```c++
void g(D2<int *> & d2, int * p)
{
    // ERROR: type mismatch!
    d2.strange = p;
}
```

##### 13.4.2 Dependent Base Classes

In the previous example, the base class is fully determined. 
It does **not** depend on a template parameter. 
This implies that a C++ compiler can look up non-dependent names in those base classes
_as soon as_ the template definition is seen. 
An alternative (**not** allowed by the C++ standard) would consist in delaying the lookup of such names 
until the template is instantiated. 
The disadvantage of this alternative approach is that it also delays 
any error messages resulting from missing symbols until instantiation. 
Hence, the C++ standard specifies that a non-dependent name appearing in a template 
is looked up _as soon as_ it is encountered:
```c++
template <typename>
class Base
{
public:
    using T = int;

    int basefield;
};

// dependent base
template <typename T>
class DD : public Base<T>
{
public:
    void f()
    {
        basefield = 0;  // #1 ERROR: use of undeclared identifier basefield
    }
};

// explicit specialization
template <>
class Base<bool>
{
public:
    enum
    {
        basefield = 42  // #2
    };
};

void g(DD<bool> & d)
{
    d.f();              // #3
}
```
At point `#1` we find our reference to a non-dependent name `basefield`: 
It must be looked up _right away_. 
Suppose we look it up in the template `Base` and bind it to the `int` member that we find therein. 
However, shortly after this we override the generic definition of `Base` with an explicit specialization. 
As it happens, this specialization changes the meaning of the `basefield` member 
to which we already committed! 
So, when we instantiate the definition of `DD::f` at point `#3`, 
we find that we too eagerly bound the non-dependent name at point `#1`. 
There is **no** modifiable `basefield` in `DD<bool>` that was specialized at point `#2`, 
and an error message should have been issued. 


To circumvent this problem, standard C++ says that 
non-dependent names are **not** looked up in dependent base classes. 
(But they are still looked up as soon as they are encountered.) 
This is part of the two-phase lookup rules that distinguish 
between a first phase when template definitions are first seen 
and a second phase when templates are instantiated (see Section 14.3.1). 


So, a standard C++ compiler will emit a diagnostic at point `#1`. 
To correct the code, it suffices to make the name `basefield` dependent 
because dependent names can be looked up only at the time of instantiation, 
and at that time the concrete base instance that must be explored will be known. 
For example, at point `#3`, the compiler will know that 
the base class of `DD<bool>` is `Base<bool>`
and that this has been explicitly specialized by the programmer. 
In this case, our preferred way to make the name dependent is as follows:
```c++
// Variation 1:
template <typename T>
class DD1 : public Base<T> 
{
public:
    // lookup delayed
    void f() { this->basefield = 0; }
};
```
An alternative consists in introducing a dependency using a qualified name:
```c++
// Variation 2:
template <typename T>
class DD2 : public Base<T> 
{
public:
    void f() { Base<T>::basefield = 0; }
};
```
Care must be taken with this solution,
because if the unqualified non-dependent name is used to form a virtual function call, 
then the qualification **inhibits** the virtual call mechanism and the meaning of the program changes. 
Nonetheless, there are situations when the first variation can **not** be used and this alternative is appropriate:
```c++
template <typename T>
class B
{
public:
    enum E
    {
        e1 = 6, 
        e2 = 28, 
        e3 = 496
    };

    virtual void zero(E e = e1) {}

    virtual void one(E & e) {}
};

template <typename T>
class D : public B<T>
{
public:
    void f()
    {
        typename D<T>::E e;  // this->E would not be valid syntax
        this->zero();        // D<T>::zero() would inhibit virtuality
        one(e);              // one is dependent because its argument is dependent
    }
};
```
Note how we used `D<T>::E` instead of `B<T>::E` in this example. 
In this case, either one works. 
In _multiple-inheritance_ cases, however, we may **not** know which base class provides the desired member 
(in which case using the derived class for qualification works) 
or multiple base classes may declare the same name 
(in which case we may have to use a specific base class name for disambiguation). 


Note that the name one in the call `one(e)` is dependent on the template parameter 
simply because the type of one of the call's explicit arguments is dependent. 
Implicitly-used default arguments with a type that depends on a template parameter do **not** count 
because the compiler can **not** verify this until it already has decided the lookup (a chicken-and-egg problem). 
To avoid subtlety, we prefer to use the `this->` prefix in all situations that allow it, 
even for non-template code. 
If you find that the repeated qualifications are cluttering up your code, 
you can bring a name from a dependent base class in the derived class once and for all:
```c++
// Variation 3:
template <typename T>
class DD3 : public Base<T> 
{
public:
    using Base<T>::basefield;    // #1 dependent name now in scope
    void f() { basefield = 0; }  // #2 fine
};
```
The lookup at point `#2` succeeds and finds the using declaration of point `#1`.
However, the using declaration is **not** verified until instantiation time and our goal is achieved. 
There are some subtle limitations to this scheme. 
For example, if multiple bases are derived from, 
the programmer must select exactly which one contains the desired member. 


When searching for a qualified name within the current instantiation, 
the C++ standard specifies that name lookup 
first search in the current instantiation and in all non-dependent bases, 
similar to the way it performs unqualified lookup for that name. 
If any name is found, then the qualified name refers to a member of a current instantiation 
and will **not** be a dependent name.
However, the lookup is nonetheless repeated when the template is instantiated,
and if a different result is produced in that context, the program is ill-formed. 
If no such name is found, and the class has any dependent bases, 
then the qualified name refers to a member of an unknown specialization. 
For example:
```c++
class NonDep
{
public:
    using Type = int;
};

template <typename T>
class Dep
{
public:
    using OtherType = T;
};

template <typename T>
class DepBase : public NonDep, public Dep<T>
{
public:
    void f()
    {
        // Finds NonDep::Type. 
        // typename keyword is optional
        typename DepBase<T>::Type t;

        // Finds nothing. 
        // DepBase<T>::OtherType is a member of an unknown specialization
        typename DepBase<T>::OtherType * ot;
    }
};
```






### 🎯 Chapter 14 Instantiation


Template instantiation is the process that 
generates types, functions, and variables from generic template definitions.
The term _instantiation_ is sometimes also used to refer to the creation of objects from types. 
In this book, however, it always refers to template instantiation.


#### 📌 14.1 On-Demand Instantiation


When a C++ compiler encounters the use of a template specialization, 
it will create that specialization by substituting the required arguments for the template parameters. 
The term _specialization_ is used in the general sense of an entity that is a specific instance of a template. 
It does **not** refer to the _explicit specialization_ mechanism described in Chapter 16. 


This is done automatically and requires **no** direction from the client code 
(or from the template definition, for that matter). 
This _on-demand instantiation_ feature is sometimes also called 
_implicit instantiation_ or _automatic instantiation_.


On-demand instantiation implies that the compiler often needs access to the full definition 
(in other words, not just the declaration) 
of the template and some of its members at the point of use. 
Consider the following tiny source code file:
```c++
// #1 declaration only
template <typename T>
class C;

// #2 fine: definition of C<int>
C<int> * p = nullptr;


template <typename T>
class C
{
public:
    void f();       // #3 member declaration
};                  // #4 class template definition completed

void g(C<int> & c)  // #5 use class template declaration only
{
    // #6 use class template definition,
    // will need definition of C::f in this translation unit
    c.f();    
}

// required definition due to #6
template <typename T>
void C<T>::f() {}
```
At point `#1` in the source code, only the declaration of the template is available, 
**not** the definition (such a declaration is sometimes called a _forward declaration_). 
As is the case with ordinary classes, 
we do **not** need the definition of a class template to be visible to declare pointers or references to this type, 
as was done at point `#2`. 
For example, the type of the parameter of function `g` 
does **not** require the full definition of the template `C`. 
However, as soon as a component needs to know the size of a template specialization 
or if it accesses a member of such a specialization, 
the entire class template definition is required to be visible. 
This explains why at point `#6` in the source code, the class template definition must be seen. 
Otherwise, the compiler can **not** verify that the member exists and is accessible
(not private or protected). 
Furthermore, the member function definition is needed too, 
since the call at point `#6` requires `C<int>::f` to exist.


Here is another expression that needs the instantiation: 
```c++
C<void> * p = new C<void>;
```
In this case, instantiation is needed so that the compiler can determine the size of `C<void>`, 
which the new-expression needs to determine how much storage to allocate. 
You might observe that for this particular template, 
the type of the argument substituted for `T` 
will **not** influence the size of the template 
because in any case, `C<X>` is an empty class 
(without explicit specializations, without data member in the primary template). 
However, a compiler is **not** required to avoid
instantiation by analyzing the template definition 
(and all compilers do perform the instantiation in practice). 
Furthermore, instantiation is also needed in this example 
to determine whether `C<void>` has an accessible default constructor 
and to ensure `C<void>` does not declare member `operator new` or `operator delete`. 


The need to access a member of a class template is **not** 
always very explicitly visible in the source code. 
For example, C++ overload resolution requires visibility 
into class types for parameters of candidate functions:
```c++
template <typename T>
class C
{
public:
    // a constructor that can be called with a single parameter
    // may be used for implicit conversions
    C(int) {}
};

void candidate(C<double>) {}  // #1

void candidate(int) {}        // #2

int main()
{
    // both previous function declarations can be called
    candidate(42);
}
```
The call `candidate(42)` will resolve to the overloaded declaration at point `#2`.
However, the declaration at point `#1` could also be instantiated 
to check whether it is a viable candidate for the call
(it is in this case because the one-argument constructor 
can implicitly convert `42` to an rvalue of type `C<double>`). 
Note that the compiler is allowed (but **not** required) to perform this instantiation 
if it can resolve the call **without** it 
(as could be the case in this example because an implicit conversion would not be selected over an exact match). 
Note also that the instantiation of `C<double>` could trigger an error, which may be surprising. 


#### 📌 14.2 Lazy Instantiation


Requirements on templates are not fundamentally different from that when using non-template classes.  
Many uses require a class type to be complete (see Section 10.3.1). 
For the template case, the compiler will instantiate the template. 
But a compiler should be "lazy" when instantiating templates, 
only instantiating parts that are needed right away. 

##### 14.2.1 Partial and Full Instantiation

The compiler sometimes **doesn't** need to substitute the complete definition of a template. 
```c++
template <typename T> 
T f (T p) 
{ 
    return 2 * p; 
}

decltype(f(2)) x = 2;
```
In this example, the type indicated by `decltype(f(2))` does **not** require 
the complete instantiation of the function template `f`. 
A compiler is therefore only permitted to substitute the _declaration_ of `f`, 
but **not** its "body". 
This is sometimes called _partial instantiation_. 


Similarly, if an instance of a class template is referred to 
without the need for that instance to be a complete type, 
the compiler should **not** perform a complete instantiation of that class template instance: 
```c++
template <typename T> 
class Q 
{
    using Type = typename T::Type;
};

Q<int> * p = 0;  // OK: the body of Q<int> is not substituted
```
Here, the full instantiation of `Q<int>` would trigger an error, 
because `T::Type` doesn't make sense when `T` is `int`. 
But because `Q<int>` need **not** be complete in this example, 
**no** full instantiation is performed and the code is okay (albeit suspicious). 


Variable templates also have a "full" vs. "partial" instantiation distinction: 
```c++
template <typename T> 
T v = T::default_value();

decltype(v<int>) s;  // OK: initializer of v<int> not instantiated
```
A full instantiation of `v<int>` would elicit an error, 
but that is **not** needed if we only need the type of the variable template instance. 


Alias templates do **not** have this distinction: 
There are **no** two ways of substituting them.


In C++, when speaking about "template instantiation" 
without being specific about full or partial instantiation, the former is intended. 
That is, instantiation is full instantiation by default. 

##### 14.2.2 Instantiated Components

When a class template is implicitly (fully) instantiated, 
each declaration of its members is instantiated as well, 
but the corresponding definitions are **not** (i.e., the member are _partially_ instantiated). 
There are a few exceptions to this. 
1. **Anonymous unions**: 
   The members of that union's definition are also instantiated. 
   Anonymous unions are always special in this way: 
   Their members can be considered to be members of the enclosing class. 
   An anonymous union is primarily a construct that says that some class members share the same storage. 
2. **Virtual member functions**: 
   Their definitions _may or may not_ be instantiated as a result of instantiating a class template. 
   Many implementations will instantiate the definition 
   because the internal structure that enables the virtual call mechanism
  requires the virtual functions actually to exist as linkable entities. 


Default function call arguments are considered separately when instantiating templates. 
Specifically, they are **not** instantiated unless there is a call to that function (or member function) 
that actually makes use of the default argument. 
If, on the other hand, the function is called only with explicit arguments that override the default,
then the default arguments are **not** instantiated. 


Similarly, exception specifications and default member initializers are not instantiated unless they are needed.


Let's put together some examples that illustrate some of these principles:
```c++
template <typename T>
class Safe {};

template <int N>
class Danger
{
    int arr[N];                    // OK here, although would fail for N <= 0
};

template <typename T, int N>
class Tricky
{
public:
    void noBodyHere(Safe<T> = 3);  // OK until usage of default value results in an error

    void inclass()
    {
        Danger<N> noBoomYet;       // OK until inclass is used with N <= 0
    }

    struct Nested 
    {
        Danger<N> pfew;            // OK until Nested is used with N <= 0
    };  
    
    union                          // due anonymous union:
    {
        Danger<N> anonymous;       // OK until Tricky is instantiated with N <= 0
        int align;
    };

    void unsafe(T (* p)[N]);       // OK until Tricky is instantiated with N <= 0

    void error()
    {
        Danger<-1> boom;           // Always ERROR (which not all compilers detect)
    }
};
```
A standard C++ compiler will examine these template definitions 
to check the syntax and general semantic constraints. 
While doing so, it will "assume the best" when checking constraints involving template parameters. 
For example, the parameter `N` in the member `Danger::arr` could be zero or negative (which would be invalid), 
but it is assumed that this isn't the case.
GCC allow zero-length arrays as extensions and may therefore accept this code even when `N` ends up being `0`.


The definitions of `inclass`, struct `Nested`, and the anonymous union are thus not a problem. 
For the same reason, the declaration of the member `unsafe(T (* p)[N])` is not a problem,
as long as `N` is an unsubstituted template parameter. 


The default argument specification (`Safe<T> = 3`) on the declaration of the member `noBodyHere`
is suspicious because the template `Safe` isn't initializable with an integer,
but the assumption is that either the default argument won't actually be needed for the generic definition of `Safe<T>` 
or that `Safe<T>` will be specialized (see Chapter 16) to enable initialization with an integer value. 
However, the definition of the member function `error` is an error even when the template is not instantiated, 
because the use of `Danger<-1>` requires a complete definition of the class `Danger<-1>`, 
and generating that class runs into an attempt to define an array with negative size. 
Interestingly, while the standard clearly states that this code is invalid, 
it also allows compilers **not** to diagnose the error when the template instance is not actually used. 
That is, since `Tricky<T, N>::error` is **not** used for any concrete `T` and `N`, 
a compiler is **not** required to issue an error for this case. 
For example, GCC and Visual C++ do **not** diagnose this error at the time of this writing.


Now let's analyze what happens when we add the following definition:
```c++
Tricky<int, -1> inst;
```
This causes the compiler to (fully) instantiate `Tricky<int, -1>` 
by substituting `int` for `T` and `-1` for `N` in the definition of template `Tricky`. 
Not all the member definitions will be needed, 
but the default constructor and the destructor (both implicitly declared in this case) are definitely called, 
and hence their definitions must be available somehow 
(which is the case in our example, since they are implicitly generated). 
As explained above, the members of `Tricky<int, -1>` are partially instantiated 
(i.e., their declarations are substituted): 
That process can potentially result in errors. 
For example, the declaration of `unsafe(T (* p)[N])` creates an array type with a negative of number elements, 
and that is an error.
Similarly, the member anonymous now triggers an error, because type `Danger<-1>` can not be completed. 
In contrast, the definitions of the members `inclass` and `struct Nested` are not yet instantiated, 
and thus no errors occur from their need for the complete type `Danger<-1>` 
(which contains an invalid array definition). 


When instantiating a template, the definitions of virtual members should also be provided. 
Otherwise, linker errors are likely to occur:
```c++
template <typename T>
class VirtualClass
{
public:
    virtual ~VirtualClass() {}

    virtual T vmem();  // Likely ERROR if instantiated without definition
};

int main()
{
    VirtualClass<int> inst;
}
```
Finally, a note about `operator->`. Consider:
```c++
template <typename T>
class C 
{
public:
    T operator->();
};
```
Normally, `operator->` must return a pointer type or another class type to which `operator->` applies. 
This suggests that the completion of `C<int>` triggers an error, 
because it declares a return type of `int` for `operator->`. 
However, certain natural class template definitions trigger these kinds of definitions, 
typical examples are smart pointer templates (e.g., `std::unique_ptr<T>`). 
Thus, the language rule is more flexible:
Only when a user-defined `operator->` _is actually selected by overload resolution_, 
it is required to return a type to which another `operator->` applies. 
This is true even outside templates (although the relaxed behavior is less useful in those contexts).
Hence, the declaration here triggers **no** error, even though `int` is substituted for the return type.


#### 📌 14.3 The C++ Instantiation Model

Template instantiation is the process of obtaining a regular type, function, or variable
from a corresponding template entity by appropriately substituting the template parameters. 

##### 14.3.1 Two-Phase Lookup

In Chapter 13 we saw that dependent names can **not** be resolved when parsing templates. 
Instead, they are looked up again at the point of instantiation.
Non-dependent names, however, are looked up early so that many errors can be diagnosed when the template is first seen. 
This leads to the concept of _two-phase lookup_ (_two-stage lookup_, _two-phase name lookup_): 
The first phase is the parsing of a template, and the second phase is its instantiation:
1. During the first phase, while _parsing_ a template: 
   - Non-dependent names are looked up using both the _ordinary lookup rules_ and _Argument-Dependent Lookup (ADL)_; 
   - Dependent unqualified names (all unqualified names are dependent) are looked up using the ordinary lookup rules, 
     but the result of the lookup is **not** considered complete 
     until an additional lookup is performed in the second phase (when the template is instantiated). 
2. During the second phase, while _instantiating_ a template at a point called the _Point of Instantiation (POI)_: 
   - Dependent qualified names are looked up 
     with the template parameters replaced with the template arguments for that specific instantiation; 
   - An additional ADL is performed for the unqualified dependent names 
     that were looked up using ordinary lookup in the first phase. 


For unqualified names (all unqualified names are dependent), 
the initial ordinary lookup (while not complete) is used to decide whether the name is a template:
```c++
namespace N 
{

template <typename> void g() {}
enum E { e };

}  // namespace N

template <typename> void f() {}

template <typename T> 
void h(T P) 
{
    f<int>(p);  // #1
    g<int>(p);  // #2 ERROR
}

int main() 
{
    h(N::e);
}
```
In line `#1`, when seeing the name `f` followed by a `<`, 
the compiler has to decide whether that `<` is an angle bracket or a less-than sign. 
That depends on whether `f` is known to be the name of a template or not. 
In this case, ordinary lookup finds the declaration of `f`, 
which is indeed a template, and so parsing succeeds with angle brackets.


Line `#2` produces an error because **no** template `g` is found using ordinary lookup.
The `<` is thus treated as a less-than sign, which is a syntax error in this example. 
If we could get past this issue, we'd eventually find the template `N::g` using ADL 
when instantiating `h` for `T = N::E` (since `N` is a namespace associated with `E`), 
but we can **not** get that far until we successfully parse the generic definition of `h`. 

##### 14.3.2 Points of Instantiation

A _Point of Instantiation (POI)_ is created when a code construct refers to a template specialization
in such a way that the definition of the corresponding template needs to be instantiated to create that specialization. 
The POI is a point in the source where the substituted template could be inserted:
```c++
class MyInt
{
public:
    MyInt(int i);
};

MyInt operator-(MyInt const &);

bool operator>(MyInt const &, MyInt const &);

using Int = MyInt;

template <typename T>
void f(T i)
{
    if (i > 0)
    {
        g(-i);
    }
}

// #1
void g(Int)
{
    // #2
    f<Int>(42);  // point of call
    // #3
}
// #4
```
When a C++ compiler sees the call `f<Int>(42)`, 
it knows the template `f` will need to be instantiated 
for `T` substituted with `MyInt`: 
A POI is created. 


Points `#2` and `#3` are very close to the point of call, 
but they can **not** be POIs because C++ does **not** allow us to insert the definition of `::f<Int>(Int)` there. 
The essential difference between point `#1` and point `#4` is that at point `#4` the function `g(Int)` is visible, 
and hence the template-dependent call `g(-i)` can be resolved. 
However, if point `#1` were the POI, then that call could **not** be resolved because `g(Int)` is **not** yet visible. 
Fortunately, C++ defines the POI "for a reference to a function template specialization" 
to be _immediately after the nearest namespace scope declaration or definition that contains that reference_. 
In our example, this is point `#4`.


You may wonder why this example involved the type `MyInt` rather than simple `int`. 
The answer lies in the fact that the second lookup performed at the POI
(after instantiating `f`, at `#4`) is only an ADL. 
Because `int` has **no** associated namespace, 
the POI lookup would therefore **not** take place and would **not** find function `g`. 
Hence, if we were to replace the type alias declaration for `Int` with
```c++
using Int = int;
```
the previous example would no longer compile. 
The following example suffers from a similar problem:
```c++
template <typename T>
void f1(T x)
{
    g1(x);  // #1
}

void g1(int) {}

int main()
{
    f1(7);  // ERROR: g1 not found!
}
// #2 POI for f1<int>(int)
```
The call `f1(7)` creates a POI for `f1<int>(int)` just outside of `main` at point `#2`. 
In this instantiation, the key issue is the lookup of function `g1`. 
When the definition of the template `f1` is first encountered, 
it is noted that the unqualified name `g1` is dependent 
because it is the name of a function in a function call with dependent arguments 
(the type of the argument `x` depends on the template parameter `T`). 
Therefore, `g1` is looked up at point `#1` using ordinary lookup rules. 
However, **no** `g1` is visible at this point. 
At point `#2`, the POI, the function is looked up again via ADL in associated namespaces and classes, 
but the only argument type is `int`, and it has **no** associated namespaces and classes. 
Therefore, `g1` is **never** found even though ordinary lookup at the POI would have found `g1`. 


The point of instantiation for variable templates is handled similarly to that of function templates. 
Surprisingly, this is not clearly specified in the standard at the time of this writing. 
However, it is not expected to be a controversial issue. 


For class template specializations, the situation is different:
```c++
template <typename T>
class S 
{
public:
    T m;
};

// #1
unsigned long h()
{
    // #2
    return static_cast<std::size_t>(sizeof(S<int>));
    // #3
}
// #4
```
Again, the function scope points `#2` and `#3` can **not** be POIs 
because a definition of a namespace scope class `S<int>` can **not** appear there. 
Generally, templates can generally **not** appear in function scope,
yet the call operator of generic lambdas are a subtle exception to that observation. 


If we were to follow the rule for function template instances, the POI would be at point `#4`, 
but then the expression `sizeof(S<int>)` is invalid 
because the size of `S<int>` can **not** be determined until point `#4` is reached. 
Therefore, the POI for a reference to a generated class instance is defined to be 
_the point immediately before the nearest namespace scope declaration or definition 
that contains the reference to that instance_. 
In our example, this is point `#1`. 


When a template is actually instantiated, the need for additional instantiations may appear: 
```c++
template <typename T>
class S
{
public:
    using I = int;
};

// #1
template <typename T>
void f()
{
    S<char>::I var1 = 41;
    typename S<T>::I var2 = 42;
}

int main()
{
    f<double>();
}
// #2 : #2a , #2b
```
Our preceding discussion already established that the POI for `f<double>` is at point `#2`. 
The function template `f` also refers to the class specialization `S<char>` with a POI that is therefore at point `#1`. 
It references `S<T>` too, but because this is still dependent, we can **not** really instantiate it at this point. 
However, if we instantiate `f<double>` at point `#2`, 
we notice that we also need to instantiate the definition of `S<double>`. 
Such secondary or transitive POIs are defined slightly differently. 
For function templates, the secondary POI is exactly the same as the primary POI. 
For class entities, the secondary POI immediately precedes (in the nearest enclosing namespace scope) the primary POI. 
In our example, this means that the POI of `f<double>` can be placed at point `#2b`, 
and just before it is the secondary POI for `S<double>` at `#2a`. 
Note how this differs from the POI for `S<char>`. 


A translation unit often contains multiple POIs for the same instance. 
For class template instances, only the first POI in each translation unit is retained, 
and the subsequent ones are ignored (they are **not** really considered POIs). 
For instances of function and variable templates, all POIs are retained. 
In either case, the _One-Definition Rule (ODR)_ requires 
that the instantiations occurring at any of the retained POIs be equivalent, 
but a C++ compiler does not need to verify and diagnose violations of this rule. 
This allows a C++ compiler to pick just one non-class POI to perform the actual instantiation 
without worrying that another POI might result in a different instantiation. 


In practice, most compilers delay the actual instantiation of most function templates 
to the end of the translation unit.
This effectively moves the POIs of the corresponding template specializations to the end of the translation unit,
which is permitted by the C++ standard as an alternative POI.


Some instantiations can **not** be delayed,
including cases where instantiation is needed to determine a deduced return type 
(see Section 15.10.1 and Section 15.10.4) 
and cases where the function is `constexpr` and must be evaluated to produce a constant result.


Some compilers instantiate inline functions when they're first used 
to potentially inline the call right away.
In modern compilers the inlining of calls is typically handled 
by a mostly language-independent component of the compiler dedicated to optimizations 
(a "back end" or "middle end"). 
However, C++ "front ends" (the C++-specific part of the C++ compiler) 
that were designed in the earlier days of C++ may also have the ability to expand calls inline
because older back ends were too conservative when considering calls for inline expansion. 

##### 14.3.3 The Inclusion Model

Whenever a POI is encountered, the definition of the corresponding template must somehow be accessible. 
For class specializations, this means that 
the class template definition must have been seen earlier in the translation unit.
This is also needed for the POIs of function templates and variable templates 
(and member functions and static data members of class templates), 
and typically template definitions are simply added to header files that are `#included` into the translation unit, 
even when they're non-type templates. 
This source model for template definitions is called the _inclusion model_, 
and it is the only automatic source model for templates supported by the current C++ standard. 


Although the inclusion model encourages programmers to place all their template definitions in header files 
so that they are available to satisfy any POIs that may arise, 
it is also possible to explicitly manage instantiations using 
_explicit instantiation declarations_ and _explicit instantiation definitions_ (see Section 14.5). 
Doing so is logistically not trivial and most of the time 
programmers will prefer to rely on the automatic instantiation mechanism instead. 
One challenge for an implementation with the automatic scheme is to 
deal with the possibility of having POIs for the same specialization of a function or variable templates 
(or the same member function or static data member of a class template instance) across different translation units. 
We discuss approaches to this problem next. 


#### 📌 14.4 Implementation Schemes


In this section we review some ways in which C++ implementations support the inclusion model. 
All these implementations rely on two classic components: 
a compiler and a linker. 
The compiler translates source code to object files, 
which contain machine code with symbolic annotations (cross-referencing other object files and libraries). 
The linker creates executable programs or libraries 
by combining the object files and resolving the symbolic cross-references they contain. 


In what follows, we assume such a model, 
even though it is entirely possible (but not popular) to implement C++ in other ways, 
such as _C++ interpreters_ like [Cling](https://root.cern/cling/). 


When a class template specialization is used in multiple translation units, 
a compiler will repeat the instantiation process in every translation unit. 
This poses very few problems because class definitions do not directly create low-level code. 
They are used only internally by a C++ implementation 
to verify and interpret various other expressions and declarations. 
In this regard, the multiple instantiations of a class definition are **not** materially different
from the multiple (header-file) inclusions of a class definition in various translation units. 


However, if you instantiate a (non-inline) function template, the situation may be different. 
If you were to provide multiple definitions of an ordinary non-inline function, you would violate the ODR. 
Assume, for example, that you compile and link a program consisting of two definitions of one ordinary function:
```c++
/// "a.cpp"
int main() {}

/// "b.cpp"
int main() {}
```
C++ compilers will compile each module separately without any problems 
because indeed they are valid C++ translation units. 
However, your linker will most likely protest if you try to link the two together: 
Duplicate definitions are **not** allowed.


In contrast, consider the template case:
```c++
/// "t.hpp"
/// common header (inclusion model)
template <typename T>
class S
{
public:
    void f();
};

template <typename T>
void S::f() {}

void helper(S<int> *);

/// "a.cpp"
#include "t.hpp"

void helper(S<int> * s)
{
    s->f();  // #1 first point of instantiation of S::f
    
}

/// "b.cpp"
#include "t.hpp"

int main()
{
    S<int> s;
    helper(&s);
    s.f();   // #2 second point of instantiation of S::f
}
```
If the linker treats instantiated member functions of class templates 
just like it does for ordinary functions or member functions, 
the compiler needs to ensure that it generates code at only one of the two POIs: 
at points `#1` or `#2`, but **not** both. 
To achieve this, a compiler has to carry information from one translation unit to the other, 
and this is something C++ compilers were **never** required to do prior to the introduction of templates. 
In what follows, we discuss the three broad classes of solutions that have been used by C++ implementers. 


Note that the same problem occurs with all linkable entities produced by template instantiation: 
instantiated function templates and member function templates, 
as well as instantiated static data members and instantiated variable templates. 

##### 14.4.1 Greedy Instantiation

The first C++ compilers that popularized greedy instantiation were produced by a company called Borland. 
It has grown to be by far the most commonly used technique among the various C++ systems. 


Greedy instantiation assumes that the linker is aware that linkable template instantiations 
may in fact appear in duplicate across the various object files and libraries. 
The compiler will typically mark these entities in a special way. 
When the linker finds multiple instances, it keeps one and discards all the others. 
There is not much more to it than that.


Greedy instantiation has some serious drawbacks:
- The compiler may be wasting time on generating and optimizing `N` instantiations,
  of which only one will be kept.
- Linkers typically do **not** check that two instantiations are identical 
  because some insignificant differences in generated code can validly occur 
  for multiple instances of one template specialization. 
  - These small differences should **not** cause the linker to fail. 
    These differences could result from tiny differences 
    in the state of the compiler at the instantiation times.
  - However, this often also results in the linker **not** noticing more substantial differences, 
    such as when one instantiation was compiled with strict floating-point math rules.  
    whereas the other was compiled with relaxed, higher-performance floating-point math rules. 
  - Current systems have grown to detect certain other differences.
    For example, they might report if one instantiation has associated debugging information and another does not.
- The sum of all the object files could potentially be much larger than with alternatives 
  because the same code may be duplicated many times. 


Greedy instantiation has the following merits:
- The traditional source-object dependency is preserved. 
  In particular, one translation unit generates but one object file, 
  and each object file contains compiled code for all the linkable definitions 
  in the corresponding source file, which includes the instantiated definitions. 
- All function template instances are candidates for inlining 
  without resorting to expensive “link-time” optimization mechanisms 
  (and, in practice, function template instances are often small functions that benefit from inlining). 
  The other instantiation mechanisms treat _inline_ function template instances specially 
  to ensure they can be expanded inline. 
  However, greedy instantiation allows even non-inline function template instances to be expanded inline. 


The linker mechanism that allows duplicate definitions of linkable entities 
is also typically used to handle 
- Duplicate _spilled inlined functions_: 
  - When a compiler is unable to "inline" every call 
    to a function that you marked with the keyword `inline`,
    a separate copy of the function is emitted in the object file. 
    This may happen in multiple object files. 
- _Virtual function dispatch tables_: 
  - Virtual function calls are usually implemented as indirect calls 
    through a table of pointers to functions.


If this mechanism is not available, 
the alternative is usually to emit these items with internal linkage, 
at the expense of generating larger code. 
The requirement that an inline function have a single address makes it difficult 
to implement that alternative in a standard-conforming way. 

##### 14.4.2 Queried Instantiation

In the mid-1990s, a company called _Sun Microsystems_ (later acquired by Oracle) 
released a reimplementation of its C++ compiler (version 4.0) 
with a new and interesting solution of the instantiation problem, 
which we call _queried instantiation_. 
Queried instantiation is conceptually remarkably simple and elegant, 
and yet it is chronologically the most recent class of instantiation schemes that we review here. 
This scheme maintains a database shared by the compilations of all translation units. 
This database keeps track of which specializations have been instantiated and on what source code they depend. 
The generated specializations themselves are typically stored with this information in the database. 
Whenever a point of instantiation for a linkable entity is encountered, 
one of three things can happen: 
1. No specialization is available: 
   In this case, instantiation occurs, and the resulting specialization is entered in the database.
2. A specialization is available 
   but is out of date because source changes have occurred since it was generated. 
   Here, too, instantiation occurs, but the resulting specialization replaces the old one. 
3. An up-to-date specialization is available in the database. 
   Nothing needs to be done. 


Although conceptually simple, this design presents a few implementation challenges:
- It is **not trivial** to maintain correctly the dependencies of the database contents 
  with respect to the state of the source code. 
  Although it is not incorrect to mistake the third case for the second, 
  doing so increases the amount of work done by the compiler (and hence overall build time). 
- It is quite common to compile multiple source files concurrently. 
  Hence, an industrial-strength implementation needs to provide 
  the appropriate amount of concurrency control in the database.


Unfortunately, the use of a database may also present some problems to the programmer. 
The origin of most of these problems lies in that fact that 
the traditional compilation model inherited from most C compilers no longer applies: 
A single translation unit no longer produces a single standalone object file. 
Assume that you wish to link your final program. 
This link operation needs not only the contents of each of the object files associated with your various translation units, 
but also the object files stored in the database. 
Similarly, if you create a binary library, you need to ensure that the tool that creates that library 
(typically a linker or an archiver) is aware of the database contents. 
More generally, any tool that operates on object files may need to be made aware of the contents of the database. 
Many of these problems can be alleviated by not storing the instantiations in the database, 
but instead by emitting the object code in the object file that caused the instantiation in the first place. 


Libraries present yet another challenge. 
A number of generated specializations may be packaged in a library. 
When the library is added to another project, 
that project's database may need to be made aware of the instantiations that are already available. 
If not, and if the project creates some of its own points of instantiation 
for the specializations present in the library, 
duplicate instantiation may occur.
A possible strategy to deal with such situations 
is to use the same linker technology that enables greedy instantiation:
Make the linker aware of generated specializations and have it weed out duplicates 
(which should nonetheless occur much less frequently than with greedy instantiation). 
Various other subtle arrangements of sources, object files, and libraries 
can lead to frustrating problems such as missing instantiations
because the object code containing the required instantiation 
was not linked in the final executable program.


Ultimately, queried instantiation did **not** survive in the marketplace, 
and even Sun's compiler now uses greedy instantiation. 

##### 14.4.3 Iterated Instantiation

The first compiler to support C++ templates was Cfront 3.0, 
a direct descendant of the compiler that Bjarne Stroustrup wrote to develop the language.
Do **not** let this phrase mislead you into thinking that Cfront was an academic prototype: 
It was used in industrial contexts and formed the basis of many commercial C++ compiler offerings. 
Release 3.0 appeared in 1991 but was plagued with bugs. 
Version 3.0.1 followed soon thereafter and made templates usable. 

An inflexible constraint on Cfront was that it had to be very portable from platform to platform, 
and this meant that it 
1. Used the C language as a common target representation across all target platforms;
2. Used the local target linker. 


In particular, this implied that the linker was **not** aware of templates. 
In fact, Cfront emitted template instantiations as ordinary C functions, 
and therefore it had to avoid duplicate instantiations. 
Although the Cfront source model was different from the standard inclusion model, 
its instantiation strategy can be adapted to fit the inclusion model. 
As such, it also merits recognition as the first incarnation of _iterated instantiation_. 


The Cfront iteration can be described as follows: 
1. Compile the sources **without** instantiating any required linkable specializations; 
2. Link the object files using a _prelinker_; 
3. The prelinker invokes the linker and parses its error messages to determine 
   whether any are the result of missing instantiations. 
   If so, the prelinker invokes the compiler on sources that contain the needed template definitions, 
   with options to generate the missing instantiations. 
4. Repeat step 3 if any definitions are generated. 


The need to iterate step 3 is prompted by the observation that 
the instantiation of one linkable entity may lead to the need 
for another such entity that was not yet instantiated. 
Eventually the iteration will “converge”, and the linker will succeed in building a complete program.


The drawbacks of the original Cfront scheme are quite severe: 
- The perceived time to link is augmented not only by the prelinker overhead 
  but also by the cost of every required recompilation and relinking. 
  Some users of Cfront-based systems reported link times of “a few days” 
  compared with “about an hour” with the alternative schemes reported earlier. 
- Diagnostics (errors, warnings) are delayed until link time. 
  This is especially painful when linking becomes expensive 
  and the developer must wait hours just to find out about a typo in a template definition. 
- Special care must be taken to remember where the source containing a particular definition is located (step 1). 
  Cfront in particular used a central repository, 
  which had to deal with some of the challenges of the central database in the queried instantiation approach. 
  In particular, the original Cfront implementation was **not** engineered to support concurrent compilations.


The iteration principle was subsequently refined both by the Edison Design Group's (EDG) implementation and by HP's aC++, 
eliminating some of the drawbacks of the original Cfront implementation. 
In practice, these implementations work quite well, and,
although a build “from scratch” is typically more time consuming than the alternative schemes, 
subsequent build times are quite competitive. 
Still, relatively few C++ compilers use iterated instantiation anymore. 
HP's aC++ also added greedy instantiation made that the default mechanism. 


#### 📌 14.5 Explicit Instantiation


It is possible to create explicitly a point of instantiation for a template specialization. 
The construct that achieves this is called an _explicit instantiation directive_. 
Syntactically, it consists of the keyword `template` followed by a declaration of the specialization to be instantiated. 
```c++
template <typename T>
void f(T) {}

// four valid explicit instantiations:
template void f<int>(int);
template void f<>(float);
template void f(long);
template void f(char);
```
Note that every instantiation directive is valid. 
Template arguments can be deduced (see Chapter 15). 


Members of class templates can also be explicitly instantiated in this way:
```c++
template <typename T>
class S 
{
public:
    void f() {}
};

template void S<int>::f();
template class S<void>;
```
Furthermore, all the members of a class template specialization can be explicitly instantiated 
by explicitly instantiating the class template specialization. 
Because these explicit instantiation directives ensure that 
a definition of the named template specialization (or member thereof) is created, 
the explicit instantiation directives above are more accurately referred to as _explicit instantiation definitions_. 
A template specialization that is explicitly instantiated should **not** be explicitly specialized, 
and vice versa, because that would imply that the two definitions could be different (thus violating the ODR). 

##### 14.5.1 Manual Instantiation

Many C++ programmers have observed that 
automatic template instantiation has a nontrivial negative impact on build times. 
This is particularly true with compilers that implement greedy instantiation (Section 14.4.1),
because the same template specializations may be instantiated and optimized in many different translation units.


A technique to improve build times consists in 
_manually instantiating_ those template specializations that the program requires in a single location 
and _inhibiting_ the instantiation in all other translation units. 
One portable way to ensure this _inhibition_ is to **not** provide the template definition 
except in the translation unit where it is explicitly instantiated.
In the 1998 and 2003 C++ standards, 
this was the only portable way to inhibit instantiation in other translation units.
```c++
/// translation unit 1:
// no definition: prevents instantiation in this translation unit
template <typename T>
void f(); 

void g()
{
    f<int>();
}

/// translation unit 2:
template <typename T>
void f()
{
    // implementation
}

// manual instantiation
template void f<int>();

void g();

int main()
{
    g();
}
```
In the first translation unit, the compiler can **not** see the definition of the function template `f`, 
so it can **not** produce an instantiation of `f<int>`. 
The second translation unit provides the definition of `f<int>` via an explicit instantiation definition.  
Without it, the program would **fail** to link. 


Manual instantiation has a clear disadvantage: 
We must carefully keep track of which entities to instantiate. 
For large projects this quickly becomes an excessive burden, 
hence we do **not** recommend it. 
We have worked on several projects that initially underestimated this burden, 
and we came to regret our decision as the code matured. 


However, manual instantiation also has a few advantages 
because the instantiation can be tuned to the needs of the program. 
Clearly, the overhead of large headers is avoided, 
as is the overhead of repeatedly instantiating the same templates
with the same arguments in multiple translation units. 
Moreover, the source code of template definition can be kept hidden, 
but then **no** additional instantiations can be created by a client program.


Some of the burden of manual instantiation can be alleviated
by placing the template definition into a third source file,
conventionally with the extension `.tpp`.
For our function `f`, this breaks down into:
```c++
/// "f.hpp"
// no definition: prevents instantiation
template <typename T>
void f(); 

/// "f.tpp"
#include "f.hpp"

//definition
template <typename T>
void f()
{
    // implementation
}

// "f.cpp"
#include "f.tpp"

// manual instantiation
template void f<int>();
```
This structure provides some flexibility. 
One can include only `f.hpp` to get the declaration of `f`, with **no** automatic instantiation. 
Explicit instantiations can be manually added to `f.cpp` as needed. 
Or, if manual instantiations become too onerous, one can also include `f.tpp` to enable automatic instantiation. 

##### 14.5.2 Explicit Instantiation Declarations

A more targeted approach to the elimination of redundant automatic instantiations 
is the use of an _explicit instantiation declaration_, 
which is an explicit instantiation directive prefixed by the keyword `extern`. 
An explicit instantiation declaration _generally_ suppresses automatic instantiation 
of the named template specialization,
because it declares that the named template specialization will be defined somewhere in the program 
(by an explicit instantiation definition). 
We say _generally_, because there are many **exceptions** to this:
- Inline functions can still be instantiated for the purpose of expanding them inline
  (but no separate object code is generated).
- Variables with deduced `auto` or `decltype(auto)` types 
  and functions with deduced return types 
  can still be instantiated to determine their types.
- Variables whose values are usable as constant-expressions 
  can still be instantiated so their values can be evaluated. 
- Variables of reference types can still be instantiated 
  so the entity they reference can be resolved.
- Class templates and alias templates can still be instantiated to check the resulting types.


Using explicit instantiation declarations, 
we can provide the template definition for `f` in the header (`t.hpp`), 
then suppress automatic instantiation for commonly used specializations, as follows:
```c++
/// "t.hpp"
template <typename T> 
void f() {}

// declared but not defined
extern template void f<int>();
extern template void f<float>(); 

/// "t.cpp"
// definition
template void f<int>();
template void f<float>();
```
Each explicit instantiation declaration must be paired with a corresponding explicit instantiation definition, 
which must follow the explicit instantiation declaration.
Omitting the definition will result in a linker error. 


Explicit instantiation declarations can be used to improve compile or link times
when certain specializations are used in many different translation units. 
Unlike with manual instantiation, which requires manually updating 
the list of explicit instantiation definitions each time a new specialization is required, 
explicit instantiation declarations can be introduced as an optimization at any point. 
However, the compile-time benefits may not be as significant as with manual instantiation, 
both because some redundant automatic instantiation is likely to occur 
and because the template definitions are still parsed as part of the header.


An interesting part of this optimization problem is to determine exactly 
which specializations are good candidates for explicit instantiation declarations. 
Low-level utilities such as the common Unix tool `nm` can be useful 
in identifying which automatic instantiations actually made it into the object files that comprise a program.


#### 📌 14.6 Compile-Time `if` Statements


As introduced in Section 8.5, 
C++17 added a new statement kind that turns out to be remarkably useful when writing templates: 
compile-time `if`. 
It also introduces a new wrinkle in the instantiation process.


The following example illustrates its basic operation:
```c++
template <typename T>
bool f(T p)
{
    if constexpr (sizeof(T) <= sizeof(long long))
    {
        return 0 < p;
    }
    else
    {
        return 0 < p.compare(0);
    }
}

bool g(int n)
{
    return f(n);  // OK
}
```
Prior to C++17 and its `constexpr if` statements, 
avoiding such errors required explicit template specialization or overloading (see Chapter 16) 
to achieve similar effects.
The example above, in C++14, might be implemented as follows:
```c++
template <bool>
struct Dispatch
{
    template <typename T>
    static bool f(T p)
    {
        return 0 < p.compare(0);
    }
};
    
template <>
struct Dispatch<true>
{
    template <typename T>
    static bool f(T p)
    {
        return 0 < p;
    }
};

template <typename T>
bool f(T p)
{
    return Dispatch<sizeof(T) <= sizeof(long long)>::f(p);
}

bool g(int n)
{
    return f(n);  // OK
}
```
Clearly, the `constexpr if` alternative expresses our intention far more clearly and concisely. 
However, it requires implementations to refine the unit of instantiation: 
Whereas previously function definitions were always instantiated as a whole, 
now it must be possible to inhibit the instantiation of parts of them. 


Another very handy use of `constexpr if` is expressing the recursion needed to handle function parameter packs. 
To generalize the example, introduced in Section 8.5:
```c++
template <typename Head, typename ... Remainder>
void f(Head && h, Remainder && ... r)
{
    doSomething(std::forward<Head>(h));
    
    if constexpr (sizeof...(r) != 0)
    {
        // handle the remainder recursively (perfectly forwardingthe arguments):
        f(std::forward<Remainder>(r)...);
    }
}
```
Without `constexpr if` statements, this requires an additional overload of the `f` template 
to ensure that recursion terminates.
```c++
template <typename Head, typename ... Remainder>
void f(Head && h, Remainder && ... r)
{
    doSomething(std::forward<Head>(h));
    f(std::forward<Remainder>(r)...);
}

// explicit specialization of f for end of recursion
template <typename Head>
void f(Head && h)
{
    doSomething(std::forward<Head>(h));
}
```
Even in non-template contexts, `constexpr if` statements have a somewhat unique effect:
```c++
void h();  // no definition!

void g() 
{
    if constexpr (sizeof(int) == 1) 
    {
        h();
    }
}
```
On most platforms, the condition in `g` is `false` and the call to `h` is therefore discarded. 
As a consequence, `h` need **not** necessarily be defined at all (unless it is used elsewhere, of course). 
Had the keyword `constexpr` been omitted in this example, 
a lack of a definition for `h` would often elicit an error at link time.
Optimization may nonetheless mask the error. 
With `constexpr if` the problem is guaranteed not to exist. 


#### 📌 14.7 In the Standard Library


The C++ standard library includes a number of templates that are only commonly used with a few basic types. 
For example, the `std::basic_string` class template is most commonly used with `char` 
(because `std::string` is a type alias of `std::basic_string<char>`) or `wchar_t`, 
although it is possible to instantiate it with other character-like types. 
Therefore, it is common for standard library implementations to introduce explicit instantiation declarations 
for these common cases. 
```c++
namespace std
{

template <typename charT, 
          typename traits = char_traits<charT>,
          typename Allocator = allocator<charT>>
class basic_string
{
    // ...
};

extern template class basic_string<char>;

extern template class basic_string<wchar_t>;

}  // namespace std
```
The source files implementing the standard library will then contain the
corresponding explicit instantiation definitions, 
so that these common implementations can be shared among all users of the standard library. 
Similar explicit instantiations often exist for the various stream classes,
such as `std::basic_iostream`, `std::basic_istream`, and so on.






### 🎯 Chapter 15 Template Argument Deduction


#### 📌 15.1 The Deduction Process


The basic deduction process 
compares the types of an argument of a function call with the corresponding parameterized type of a function template 
and attempts to conclude the correct substitution for one or more of the deduced parameters. 
Each argument-parameter pair is analyzed independently, 
and if the conclusions differ in the end, the deduction process fails:
```c++
template <typename T>
T max (T a, T b)
{
    return b < a ? a : b;
}

auto g = max(1, 1.0);  // ERROR
```
Here the first call argument is of type `int`, 
so the parameter `T` of our original `max` template is tentatively deduced to be `int`. 
The second call argument is a `double`, however, and so `T` should be `double` for this argument: 
This conflicts with the previous conclusion. 
Note that we say that “the deduction process fails”, 
not that “the program is invalid”. 
After all, it is possible that the deduction process would succeed for another template named `max` 
(function templates can be overloaded much like ordinary functions. See Section 1.5 and Chapter 16). 


If all the deduced template parameters are consistently determined, 
the deduction process can still fail if substituting the arguments 
in the rest of the function declaration results in an invalid construct: 
```c++
template <typename T>
typename T::ElementT at(T a, int i)
{
    return a[i];
}

void f(int * p)
{
    int x = at(p, 7);
}
```
Here `T` is concluded to be `int *` 
(there is only one parameter type where `T` appears,
so there are obviously no analysis conflicts). 
However, substituting `int *` for `T` in the return type `T::ElementT` is clearly invalid C++, 
and the deduction process fails.
In this case, deduction failure follows the SFINAE principle (see Section 8.4): 
If there were another function for which deduction succeeds, the code could be valid. 


We still need to explore how argument-parameter matching proceeds. 
We describe it in terms of 
matching a type `A` (derived from the call argument type) 
to a parameterized type `P` (derived from the call parameter declaration). 
If the call parameter is declared with a reference declarator, 
`P` is taken to be the type referenced, and `A` is the type of the argument. 
Otherwise, `P` is the declared parameter type, 
and `A` is obtained from the type of the argument 
by _decaying_ array and function types to pointer types 
(_decay_ is the term used to refer to the implicit conversion of function and array types to pointer types),
ignoring top-level `const` and `volatile` qualifiers:
```c++
// parameterized type P is T
template <typename T>
void f(T) {}

// parameterized type P is T
template <typename T>
void g(T &) {}

double arr[20];
int const seven = 7;

f(arr);    // T = double *
g(arr);    // T = double [20]
f(seven);  // T = int
g(seven);  // T = const int
f(7);      // T = int
g(7);      // T = int, ERROR: can't bind 7 to int &
```
The fact that **no** decay occurs for arguments bound to reference parameters
can be surprising when the arguments are string literals. 
Reconsider our `max` template declared with references:
```c++
template <typename T>
T const & max(T const & a, T const & b);
```
It would be reasonable to expect that for the expression `max("Apple", "Pie")`, 
`T` is deduced to be `char const *`. 
However, the type of `"Apple"` is `char const[6]`, and the type of `"Pie"` is `char const[4]`.
**No** array-to-pointer decay occurs (because the deduction involves reference parameters), 
and therefore `T` would have to be both `char[6]` and `char[4]` for deduction to succeed. 
That is, of course, impossible. 
See Section 7.4 for a discussion about how to deal with this situation. 


#### 📌 15.2 Deduced Contexts


Parameterized types that are considerably more complex than just `T` 
can be matched to a given argument type:
```c++
template <typename T>
void f1(T *);

template <typename E, int N>
void f2(E (&)[N]);

template <typename T1, typename T2, typename T3>
void f3(T1 (T2::*)(T3 *));

class S
{
public:
    void f(double *);
};

void g(int *** ppp)
{
    bool b[42];
    f1(ppp);    // deduces T to be int **
    f2(b);      // deduces E to be bool and N to be 42
    f3(&S::f);  // deduces T1 = void, T2 = S, and T3 = double
}
```
Complex type declarations are built from more elementary constructs 
(pointer, reference, array, and function declarators; pointer-to-member declarators; template-ids; and so forth), 
and the matching process proceeds from the top-level construct and recurses through the composing elements. 
It is fair to say that most type declaration constructs can be matched in this way, 
and these are called _deduced contexts_. 
However, a few constructs are 
[Non-Deduced Contexts](https://en.cppreference.com/w/cpp/language/template_argument_deduction#Non-deduced_contexts). 
For example:
- **Qualified type names**.  
  For example, a type name like `Q<T>::X` will **never** be used to deduce a template parameter `T`. 
- **Non-type expressions that are not just a non-type parameter**.  
  For example, a type name like `S<I + 1>` will **never** be used to deduce `I`. 
  **Neither** will `T` be deduced by matching against a parameter of type `int (&)[sizeof(S<T>)]`. 
- ...


These limitations should come as no surprise because the deduction would, 
in general, not be unique (or even finite), 
although this limitation of qualified type names is sometimes easily overlooked. 
A non-deduced context does not automatically imply that the program is in error 
or even that the parameter being analyzed can not participate in type deduction. 
To illustrate this, consider the following, more intricate example:
```c++
template <int N>
class X
{
public:
    using I = int;

    void f(int) {}
};

template <int N>
void fppm(void (X<N>::* p)(typename X<N>::I));

int main()
{
    // fine: N deduced to be 33
    fppm(&X<33>::f);
}
```
In the function template `fppm`, the subconstruct `X<N>::I` is a non-deduced context. 
However, the member-class component `X<N>` of the pointer-to-member type is a deducible context, 
and when the parameter `N`, which is deduced from it, is plugged in the non-deduced context, 
a type compatible with that of the actual argument `&X<33>::f` is obtained. 
The deduction therefore succeeds on that argument-parameter pair. 


Conversely, it is possible to deduce contradictions for a parameter type entirely built from deduced contexts. 
For example, assuming suitably declared class templates `X` and `Y`:
```c++
template <typename, typename> 
class X {};

template <typename> 
class Y {};

template <typename T>
void f(X<Y<T>, Y<T>>) {}

void g()
{
    f(X<Y<int>, Y<int>>());   // OK
    f(X<Y<int>, Y<char>>());  // ERROR: deduction fails
}
```
The problem with the second call to the function template `f` is that
the two arguments deduce different arguments for the parameter `T`, which is not valid. 
In both cases, the function call argument is a temporary object 
obtained by calling the default constructor of the class template `X`.

 
#### 📌 15.3 Special Deduction Situations


There are several situations in which the pair `(A, P)` used for deduction 
is **not** obtained from the arguments to a function call and the parameters of a function template. 
The first situation occurs when the address of a function template is taken. 
In this case, `P` is the type of the function template declaration, 
and `A` is the function type underlying the pointer that is initialized or assigned to: 
```c++
template <typename T>
void f(T, T) {}

void (*pf)(char, char) = &f;
```
In this example, `P` is `void (T, T)` and `A` is `void (char, char)`. 
Deduction succeeds with `T` substituted with `char`, 
and `pf` is initialized to the address of the specialization `f<char>`. 


Similarly, function types are used for `P` and `A` for a few other special situations:
- Determining a partial ordering between overloaded function templates;
- Matching an explicit specialization to a function template;
- Matching an explicit instantiation to a template;
- Matching a friend function template specialization to a template;
- Matching a placement `operator delete` or `operator delete[]` 
  to a corresponding placement `operator new` template or `operator new[]` template. 


Some of these topics, along with the use of template argument deduction for class template partial specializations, 
are further developed in Chapter 16.


Another special situation occurs with conversion function templates:
```c++
class S 
{
public:
    template <typename T> 
    operator T&();
};
```
In this case, the pair `(P, A)` is obtained as if it involved: 
- an argument of the conversion-target type; 
- a parameter type that is the return type of the conversion function. 
The following code illustrates one variation:
```c++
void f(int (&)[20]) {}

void g(S s)
{
    f(s);
}
```
Here we are attempting to convert `S` to `int (&)[20]`. 
Type `A` is therefore `int [20]` and type `P` is `T`. 
The deduction succeeds with `T` substituted with `int [20]`. 


Some special treatment is also needed for the deduction of the `auto` placeholder type. 
That is discussed in Section 15.10.4.


#### 📌 15.4 Initializer Lists


When the argument of a function call is an initializer list, 
that argument **doesn't** have a specific type, 
so in general no deduction will be performed from that given pair `(A, P)` because there is no `A`:
```c++
template <typename T> 
void f(T t) {}

int main() 
{
    f({1, 2, 3});  // ERROR: can not deduce T from an initializer list
}
```
However, if the parameter type `P`, after removing references and top-level cv, 
is equivalent to `std::initializer_list<P′>` for some type `P′` that has a deducible pattern,
deduction proceeds by comparing `P′` to the type of each element in the initializer list, 
succeeding only if all of the elements have the same type:
```c++
template <typename T> 
void f(std::initializer_list<T> il) {}

int main() 
{
    f({1, 2, 3});  // OK: T deduced to int
    f({1, '2'});   // ERROR: T is deduced to both in and char
}
```
Similarly, if the parameter type `P` is a reference to an array type 
with element type `P′` for some type `P′` that has a deducible pattern, 
deduction proceeds by comparing `P′` to the type of each element in the initializer list, 
succeeding only if all of the elements have the same type. 
Furthermore, if the bound has a deducible pattern (i.e., just names a non-type template parameter), 
then that bound is deduced to the number of elements in the list. 
```c++
template <typename T, std::size_t N>
void f(T (&&)[N])
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

int main(int argc, char * argv[])
{
    f({1, 2, 3});  // T = int; N = 3
}
```


#### 📌 15.5 Parameter Packs


```c++
template <typename First, typename ... Rest>
void f(First first, Rest ... rest) {}

void g(int i, double j, int * k)
{
    f(i, j, k);  // deduces First to int, Rest to {double, int *}
}
```
Deduction for the first function parameter is simple, since it does not involve any parameter packs.
Deduction determines the value of the parameter pack `Rest` to be the sequence `{double, int *}`.
Substituting the results of that deduction and the deduction for the first function parameter 
yields the function type `void (int, double, int*)`, which matches the argument types at the call site. 


Values for multiple template parameters and parameter packs 
can be determined from each of the argument types: 
```c++
template <typename T, typename ... Rest>
void h1(std::pair<T, Rest> const & ...) {}

template <typename ... Ts, typename ... Rest>
void h2(std::pair<Ts, Rest> const & ...) {}

void foo(std::pair<int, float> pif,
         std::pair<int, double> pid,
         std::pair<double, double> pdd)
{
    h1(pif, pid);  // OK: deduces T to int, Rest to {float,double}
    h2(pif, pid);  // OK: deduces Ts to {int, int}, Rest to {float, double}
    
    h1(pif, pdd);  // ERROR: T deduced to int from the 1st arg, but to double from the 2nd
    h2(pif, pdd);  // OK: deduces Ts to {int, double}, Rest to {float, double}
}
```
Deduction for parameter packs is **not** limited to function parameter packs 
where the argument-parameter pairs come from call arguments. 
In fact, this deduction is used wherever a pack expansion 
is at the end of a function parameter list or a template argument list.
If a pack expansion occurs anywhere else in a function parameter list or template argument list, 
that pack expansion is considered a non-deduced context. 
```c++
template <typename ... Types>
bool f1(std::tuple<Types ...>, std::tuple<Types ...>) {}

template <typename ... Types1, typename ... Types2>
bool f2(std::tuple<Types1 ...>, std::tuple<Types2 ...>) {}

void bar(std::tuple<short, int, long> sv,
         std::tuple<unsigned short, unsigned, unsigned long> uv)
{
    f1(sv, sv);  // OK: Types is deduced to {short, int, long}.
    f2(sv, sv);  // OK: Types1 is deduced to {short, int, long}, Types2 is deduced to {short, int, long}.
    f1(sv, uv);  // ERROR: Types is deduced to {short, int, long} from the 1st arg, but to {unsigned short, unsigned, unsigned long} from the 2nd.
    f2(sv, uv);  // OK: Types1 is deduced to {short, int, long}, Types2 is deduced to {unsigned short, unsigned, unsigned long}.
}
```

##### 15.5.1 Literal Operator Templates

C++11 allows user-defined literals (could be `constexpr`) as follows:
- Cooked: Pre-processed by processor. 
```c++
T operator "" _suffix(unsigned long long);
T operator "" _suffix(long double);
T operator "" _suffix(char);
T operator "" _suffix(wchar_t);
T operator "" _suffix(char16_t);
T operator "" _suffix(char32_t);
T operator "" _suffix(char const *, std::size_t);
T operator "" _suffix(wchar_t const *, std::size_t);
T operator "" _suffix(char16_t const *, std::size_t);
T operator "" _suffix(char32_t const *, std::size_t);
```
- Raw: Only suffix numeric types! 
```c++
T operator "" _suffix(const char *);

// literal operator template
template <char ...>
T operator "" _suffix();
```
When the compiler identifies a user-defined literal and has to call the appropriate user-defined literal operator, 
it will pick the overload from the overload set according to the following rules: 
- **For integral literals**, it calls in the following order: 
  1. The operator that takes an `unsigned long long`, 
  2. The _raw_ literal operator that takes a `const char *`, 
  3. the literal operator template. 
- **For floating-point literals**, it calls in the following order: 
  1. The operator that takes a `long double`, 
  2. The _raw_ literal operator that takes a `const char *`, 
  3. The literal operator template.
- **For character literals**, it calls the appropriate operator, 
  depending on the character type (`char`, `wchar_t`, `char16_t`, and `char32_t`). 
- **For string literals**, it calls the appropriate operator, 
  that takes a pointer to the string of characters and the size, 
  depending on the string type. 


Literal operator templates have their argument determined in a unique way. 
```c++
template <char...>
int operator "" _B7();  // #1

int a = 121_B7;         // #2
```
Here, the initializer for `#2` contains a user-defined literal, 
which is turned into a call to the literal operator template `#2` 
with the template argument list `<'1', '2', '1'>`. 
Thus, an implementation of the literal operator such as
```c++
template <char ... cs>
int operator "" _B7()
{
    std::array<char, sizeof...(cs)> chars {cs...};

    for (char c: chars)
    {
        std::cout << '\'' << c << '\'';
    }

    std::cout << '\n';
    return ...;
}
```
will output `'1' '2' '1' '.' '5'` for `121.5_B7`. 
Note that raw literal operators are _only supported for valid numeric (integral and floating-point) literals_: 
```c++
auto b = 01.3_B7;    // OK: deduces {'0', '1', '.', '3'}
auto c = 0xFF00_B7;  // OK: deduces {'0', 'x', 'F', 'F', '0', '0'}
auto d = 0815_B7;       // ERROR: invalid digit '8' in octal constant
auto e = hello_B7;      // ERROR: use of undeclared identifier hello_B7
auto f = "hello"_B7;    // ERROR: no matching operator "" _B7
```
See Section 25.6 for an application of the feature to compute integral literals at compile time. 


#### 📌 15.6 Rvalue References 

##### 15.6.1 Reference Collapsing

Programmers are **not** allowed to _directly_ declare a “reference to a reference”:
```c++
int const & r = 42;
int const & & ref2ref = i;  // ERROR: reference to reference is invalid
```
However, when composing types through the substitution of
template parameters, type aliases, or `decltype` constructs, 
such situations are permitted:
```c++
using RI = int &;
int i = 42;
RI r = i;
R const & rr = r;  // OK: rr has type int &. 
                   // const qualifier on a reference type RI (aka int &) has no effect. 
```
The rules that determine the type resulting from such a composition 
are known as the _reference collapsing_ rules: 
1. Any `const` or `volatile` qualifiers applied on top of the inner reference are simply **discarded**. 
   I.e., only the bottom-level cv is retained. 
2. The two references are reduced to a single reference. 
   - Only rvalue reference of rvalue references are rvalue references. 
   - All other compositions yields lvalue references. 
```c++
using RCI = int const &;
RCI volatile && r = 42;   // OK: r has type int const &, top-level volatile is discarded
using RRI = int &&;
RRI const && rr = 42;     // OK: rr has type int &&, top-level const is discarded
```

##### 15.6.2 Forwarding References

Template argument deduction behaves in a special way when a function parameter is a _forwarding reference_ 
(an rvalue reference to a template parameter of that function template). 
In this case, template argument deduction considers **not** just the type of the function call argument 
but also whether that argument is an lvalue or an rvalue. 
- **When the argument is an lvalue**, 
  the type determined by template argument deduction is an lvalue reference to the argument type, 
  and the reference collapsing rules (see above) ensure that the substituted parameter will be an lvalue reference. 
- **Otherwise**, 
  the type deduced for the template parameter is simply the argument type (**not** a reference type), 
  and the substituted parameter is an rvalue reference to that type. 
```c++
template <typename T> 
void f(T && p) {}

void g()
{
    int i;
    int const j = 0;
    f(i);  // T = int &
    f(j);  // T = const int &
    f(2);  // T = int
}
```
The deduction of `T` as a reference type can have some interesting effects on the instantiation of the template. 
For example, a local variable declared with type `T` will, after instantiation for an lvalue, 
have reference type and will therefore require an initializer: 
```c++
template <typename T> 
void f(T &&)
{
    T x;  // ERROR for lvalue arguments
}
```
Solution:
```c++
template <typename T> 
void f(T &&)
{
    std::remove_reference_t<T> x;  // ERROR again for array arguments...
}
```

##### 15.6.3 Perfect Forwarding

Forwarding reference parameters accept almost any argument (bit fields are an exception) 
and captures its salient properties (both its _type_ and its _value category_). 
```c++
class C {};

void g(C &) {}
void g(C const &) {}
void g(C &&) {}

template <typename T>
void forwardToG(T && x)
{
    g(static_cast<T &&>(x));
}

void foo()
{
    C v;
    C const c;
    
    forwardToG(v);             // g(C &)
    forwardToG(c);             // g(const C &)
    forwardToG(C());           // g(C &&)
    forwardToG(std::move(v));  // g(C &&)
}
```
The use of `static_cast` within the function `forwardToG` requires some additional explanation. 
In each instantiation of `forwardToG`, 
the type of expression `x` could either be lvalue reference type or rvalue reference type. 
Regardless, the value category of expression `x` will always be an lvalue.


Treating a parameter of rvalue reference type as an lvalue is intended as a safety feature, 
because anything with a name (like a parameter) can easily be referenced multiple times in a function. 
If each of those references could be implicitly treated as an rvalue, 
its value could be _destroyed_ unbeknownst to the programmer. 
Therefore, one must explicitly state when a named entity should be treated as an rvalue. 
For this purpose, the C++ standard library function `std::move` treats any value as an rvalue 
(or, more precisely, an xvalue. See Appendix B for details). 


The `static_cast` casts `x` to its original type and value category. 
The type `T &&` will either collapse to an lvalue reference 
(if the original argument was an lvalue causing `T` to be an lvalue reference) 
or will be an rvalue reference (if the original argument was an rvalue), 
so the result of the `static_cast` has the same type and value category as the original argument, 
thereby achieving perfect forwarding. 


The C++ standard library provides a function template `std::forward` in header `<utility>`
that should be used in place of `static_cast` for perfect forwarding. 
Using that utility template better documents the programmer's intent 
than the arguably opaque `static_cast` constructs shown above 
and prevents errors such as omitting one `&`. 
```c++
template <typename T>
void forwardToG(T && x)
{
    g(std::forward<T>(x));
}
```
Perfect forwarding is **not** “perfect” and has several failure cases.  
Refer to [Effective Modern C++ Notes Item 30](./effective_cpp_notes_04_effective_modern_cpp.md#-item-30-familiarize-yourself-with-perfect-forwarding-failure-cases).
- Perfect forwarding fails when template type deduction fails or when it deduces the wrong type.
  The kinds of arguments that lead to perfect forwarding failure are:
    - braced initializers (decreed failure, create a temp var via auto);
    - null pointers expressed as `0` or `NULL` (deduces "wrong" type `int`, use `nullptr`);
    - declaration-only integral const static data members (no address, not referencable, provide definition);
    - template and overloaded function names (fails, explicitly cast to desired function pointer type);
    - bitfields (address not aligned, not non-const referencable, create a temp copy).


For example, it does **not** distinguish whether an lvalue is a bit-field lvalue,
**nor** does it capture whether the expression has a specific constant value.
The latter causes problems particularly when we’re dealing with the null pointer constant `NULL`,
which is a value of integral type that evaluates to the constant value zero.
Since the constant value of an expression is **not** captured by perfect forwarding,
overload resolution in the following example will behave differently
for the direct call to `g` than for the forwarded call to `g`:
```c++
void g(int);
void g(int *);

template <typename T> 
void forwardToG(T && x)
{
    g(std::forward<T>(x));
}

void foo()
{
    g(NULL);           // g(int *) 
    forwardToG(NULL);  // g(int) !
}
```
This is yet another reason to use `nullptr` (introduced in C++11) instead of null pointer constants:
```c++
g(nullptr);           // g(int *) 
forwardToG(nullptr);  // g(int *)
```

##### Perfect Forwarding for Variadic Templates

```c++
template <typename ... Ts> 
void forwardToG(Ts && ... xs)
{
    g(std::forward<Ts>(xs)...);
}
```
All of our examples of perfect forwarding have focused on forwarding the function arguments
while maintaining their _precise_ type (**not** decayed!) and value categories. 
The same problem occurs when forwarding the return value of a call to another function, 
with precisely the same type and value category.
```c++
// C++11
template <typename ... Ts>
auto forwardToG(Ts && ... xs) -> decltype(g(std::forward<Ts>(xs)...))
{
    return g(std::forward<Ts>(xs)...);  // forward all xs to g()
}

// C++14
template <typename ... Ts>
decltype(auto) forwardToG(Ts && ... xs)
{
    return g(std::forward<Ts>(xs)...);  // forward all xs to g()
}
```

##### 15.6.4 Deduction Surprises

The results of the special deduction rule for rvalue references are very useful for perfect forwarding. 
However, they can come as a surprise, because function templates typically generalize the types in the function signature
**without** affecting what kinds of arguments (lvalue or rvalue) it allows:
```c++
void int_lvalues(int &);

template <typename T>
void lvalues(T &);

void int_rvalues(int &&);

template <typename T>
void anything(T &&);  // SURPRISE: accepts lvalues and rvalues of any type
```
Programmers who are simply abstracting a concrete function like `int_rvalues` to its template equivalent 
would likely be surprised by the fact that the function template `anything` accepts lvalues. 
Fortunately, this deduction behavior only applies 
when the function parameter is written specifically with the form `T &&`, 
is part of a function template, 
and the named template parameter is declared by that function template. 
Therefore, this deduction rule does **not** apply in any of the following situations:
```c++
template <typename T>
class X
{
public:
    X(X &&);     // X is not a template parameter

    X(T &&);     // this constructor is not a function template

    template <typename Other>
    X(X<U> &&);  // X<U> is not a template parameter

    template <typename U>
    X(U, T &&);  // T is a template parameter from an outer template
}
```
One can use a combination of SFINAE (see Section 8.4 and Section 15.7) 
and type traits such as `std::enable_if` (see Section 6.3 and Section 20.3) 
to restrict the template to rvalues:
```c++
template <typename T>
std::enable_if_t<!std::is_lvalue_reference_v<T>>
rvalues(T &&) {}
```


#### 📌 15.7 SFINAE (Substitution Failure Is Not An Error)


The SFINAE (Substitution Failure Is Not An Error) principle is an important aspect of template argument deduction 
that prevents unrelated function templates from causing errors during overload resolution. 
SFINAE also applies to the substitution of partial class template specializations (see Section 16.4). 
```c++
template <typename T, unsigned N>
T * begin(T (& array)[N])
{
    return array;
}

template <typename Container>
typename Container::iterator begin(Container & c)
{
    return c.begin();
}

int main()
{
    std::vector<int> v;
    int a[10];
    ::begin(v);  // OK: only container begin matches, because the first deduction fails
    ::begin(a);  // OK: only array begin matches, because the second substitution fails
}
```

##### 15.7.1 Immediate Context

SFINAE protects against attempts to form invalid types or expressions, 
including errors due to ambiguities or access control violations,
that occur within the _immediate context_ of the function template substitution. 


The _immediate context_ includes many things, 
including various kinds of lookup, alias template substitutions, overload resolution, etc. 
Arguably the term is a bit of a misnomer, 
because some of the activities it includes are not closely tied to the function template being substituted. 


Only the failures in the types and expressions in the _immediate context_ of:
- The function type; 
- Its template parameter types;
- Its [explicit specifier](https://en.cppreference.com/w/cpp/language/explicit)
are SFINAE errors.


If the evaluation of a substituted type/expression causes a side effect such as 
- Instantiation of some template specialization;
- Generation of an implicitly-defined member function;
- ...
errors _in_ those side effects are treated as hard errors.
A lambda expression is **not** considered part of the _immediate context_. 


Specifically, the following errors are **not** covered by SFINAE (and thus result in compilation errors):
- Anything that happens _during the instantiation_ of
  - The definition of a class template (i.e., its "body" and list of base classes);
  - The definition of a function template ("body" and, in the case of a constructor, its constructor-initializers);
  - The initializer of a variable template;
  - A default argument;
  - A default member initializer;
  - An exception specification;
- Any implicit definition of special member functions triggered by the substitution process. 


So if substituting the template parameters of a function template declaration 
requires the _instantiation_ of the body of a class template because a member of that class is being referred to, 
an error during that instantiation is **not** in the _immediate context_ of the function template substitution, 
and is therefore a real error (even if another function template matches without error):
```c++

template <typename T>
class Array
{
public:
    using iterator = T *;
};

template <typename T>
void f(Array<T>::iterator first, Array<T>::iterator last);

template <typename T>
void f(T *, T *);

int main()
{
    f<int &>(0, 0);  // ERROR: substituting int & for T in the first function template
}
// Instantiation of Array<int &> fails
// Error happens in the instantiation of Array, so not SFINAE
```
The main difference between a SFINAE error and a hard error and  is _where the failure occurs_.
- Consider `typename std::enable_if<Cond>::type`. 
  The _instantiation_ of `std::enable_if` _always succeed_. 
  An error will happen when accessing its `type` member for an already-instantiated `std::enable_if<false>` class. 
- In this example, the failure occurs _in_ the _instantiation_ of `Array<int &>`,
  which actually occurs in the context of the class template `Array`,
  although `Array` was triggered from `f`'s context. 
  Therefore, the SFINAE principle does **not** apply, and the compiler will produce an error.   


Here is a C++14 example relying on deduced return types (see Section 15.10.1) 
that involves an error during the instantiation of a function template definition:
```c++
template <typename T> 
auto f(T p) 
{
    return p->m;
}

int f(...);

template <typename T> 
auto g(T p) -> decltype(f(p));

int main()
{
    g(42);
}
```
The call `g(42)` deduces `T` to be i`n`t. 
Making that substitution in the declaration of `g` requires us to determine the type of `f(p)`
(where `p` is now known to be of type `int`) and therefore to determine the return type of `f`. 
There are two candidates for `f`. 
The non-template candidate is a match, but not a very good one because it matches with an ellipsis parameter. 
Unfortunately, the template candidate has a deduced return type, 
and so we must instantiate its definition to determine that return type. 
That instantiation **fails** because `p->m` is **not** valid when `p` is an `int`, 
and since the failure is outside the _immediate context_ of the substitution 
(because it’s in a subsequent instantiation of a function definition), 
the failure produces an error. 
Because of this, we recommend avoiding deduced return types 
if they can easily be specified explicitly.


SFINAE was originally intended to eliminate surprising errors 
due to unintended matches with function template overloading, 
as with the container `begin` example. 
However, the ability to detect an invalid expression or type enables remarkable compile-time techniques, 
allowing one to determine whether a particular syntax is valid. 
These techniques are discussed in Section 19.4. 


See especially Section 19.4.4 for an example of making a type trait _SFINAE-friendly_ 
to avoid problems due to the _immediate context_ issue. 


#### 📌 15.8 Limitations of Deduction


##### 15.8.1 Allowable Argument Conversions

Normally, template deduction attempts to find a substitution of the function template parameters 
that make the parameterized type `P` identical to type `A`. 
However, when this is **not** possible, the following differences are tolerable: 
- _Qualification conversion_ (adding `cv`-qualifiers). 
  - Only bottom-level cv for reference and pointer parameter types. 
  - Regular types only have top-level cv, which is decayed. 
- _Derived-to-base conversion_ (for regular types or pointer types),  
  **unless** deduction occurs for a conversion operator template. 
```c++
template <typename T>
class B {};

template <typename T>
class D : public B<T> {};

template <typename T> 
void f(B<T> *) {}

void g(D<long> dl)
{
    f(&dl);  // deduction succeeds with T substituted with long
}
```
If `P` does **not** contain _a template parameter in a deduced context_, 
then all implicit conversion are permissible:
```c++
template <typename T>
void f(T, typename T::X) {}

struct V
{
    struct X { X(double) {} };

    V() = default;
};

// OK:
// T is deduced to V through the first parameter,
// which causes the second parameter to have type V::X (not deduced because already clear),
// which can be constructed from a double value
f(V {}, 7.0);
```

##### 15.8.2 Class Template Arguments

Prior to C++17, template argument deduction applied 
exclusively to function templates and member function templates. 
The arguments for a class template were **not** deduced: 
```c++
template <typename T>
class S
{
public:
    S(T b) : a {b} {}

private:
    T a;
};

// ERROR before C++17: 
// the class template parameter T was not deduced
// from the constructor call argument 12
S x(12);
```

##### 15.8.3 Default Call Arguments

Default function call arguments can be specified in function templates
just as they are in ordinary functions.
The default function call argument can depend on a template parameter. 
Such a dependent default argument is instantiated only if **no** explicit argument is provided:
```c++
template <typename T>
void init(T * loc, T const & val = T())
{
    *loc = val;
}

class S
{
public:
    S(int, int);
};

S s(0, 0);

int main()
{
    // T() is invalid for T = S, 
    // but the default call argument T() needs no instantiation
    // because an explicit argument is given
    init(&s, S(7, 42));
}
```
Even when a default call argument is **not** dependent, 
it can **not** be used to deduce template arguments:
```c++
template <typename T>
void f (T x = 42) {}

f<int>();  // OK: T = int
f();       // ERROR: can not deduce T from default call argument
```

##### 15.8.4 Exception Specifications

Exception specifications are also only instantiated when they are needed. 
This means that they do **not** participate in template argument deduction.
```c++
// #1
template <typename T>
void f(T, int) noexcept(nonexistent(T()));

// #2 (C-style vararg function)
template <typename T>
void f(T, ...);

void test(int i)
{
    // ERROR: chooses #1 , but the expression nonexistent(T()) is ill-formed
    f(i, i);
}
```
The noexcept specification in function `#1` tries to call a nonexistent function. 
Normally, such an error directly within the declaration of the function template 
would trigger a template argument deduction failure (SFINAE), 
allowing the call `f(i, i)` to succeed by selecting function `#2`. 
However, because exception specifications do **not** participate in template argument deduction, 
overload resolution selects `#1` and the program becomes ill-formed 
when the noexcept specification is later instantiated. 


The same rules apply to exception specifications that list the potential exception types:
```c++
// #1
template <typename T>
void g(T, int) throw(typename T::Nonexistent);

// #2
template <typename T>
void g(T, ...);

void test(int i)
{
    // ERROR: chooses #1, but the type T::Nonexistent is ill-formed
    g(i, i);
}
```
However, dynamic exception specifications have been deprecated since C++11 and were removed in C++17. 


#### 📌 15.9 Explicit Function Template Arguments


It is possible to explicitly specify it following the function template name.
This is useful when a function template argument can **not** be deduced: 
```c++
template <typename T>
void compute(T p) {}

template <typename T>
T default_value()
{
    return T {};
}

int main(int argc, char * argv[])
{
    compute<double>(2);    // 2 implicitly converted to double; 
                           // not possible without explicit specification
    default_value<int>();  // non-deductible
    
    return EXIT_SUCCESS;
}
```
Once a template argument is explicitly specified, 
its corresponding parameter is no longer subject to deduction. 
That, allows conversions to take place on the function call parameter 
that would **not** be possible in a deduced call. 
In the example above, the argument `2` in the call `compute<double>(2)` 
will be implicitly converted to `double`.


It is possible to explicitly specify some template arguments while having others be deduced. 
However, the explicitly specified ones are always matched left-to-right with the template parameters. 
Therefore, parameters that can **not** be deduced (or that are likely to be specified explicitly) 
should be specified first.
```c++
template <typename Out, typename In>
Out convert(In p) { /* ... */ }

int main()
{
    // the type of parameter p is deduced,
    // but the return type is explicitly specified
    auto x = convert<double>(42);
}
```
It is occasionally useful to specify an empty template argument list 
to ensure the selected function is a template instance 
while still using deduction to determine the template arguments:
```c++
int f(int) {}  

template <typename T>
T f(T) {}

int main()
{
    auto x = f(42);    // f(int)
    auto y = f<>(42);  // f(T) [with T = int]
}
```
In the context of friend function declarations,
the presence of an explicit template argument list has an interesting effect:
```c++
void f();

template <typename>
void f();

namespace N
{

class C
{
    friend int f();    // OK

    friend int f<>();  // ERROR: return type conflict
};

}  // namespace N
```
When a plain identifier is used to name a friend function, 
that function is only looked up within the nearest enclosing scope, 
and if it is not found there, a new entity is _declared_ in that scope 
(but it remains "invisible" except when looked up via ADL). 
That is what happens with our first friend declaration above: 
No `f` is declared within namespace `N`, and so a new `N::f` is "invisibly" declared.


However, when the identifier naming the friend is followed by a template argument list, 
a template must be visible through normal lookup at that point, 
and normal lookup will go up any level of (enclosing) scopes. 
So, our second declaration above will find the global function template `f`, 
but the compiler will then issue an error because the return types do **not** match 
(since no ADL is performed here, the declaration of `f` one line above is not found). 


Explicitly specified template arguments are substituted using SFINAE principles: 
If the substitution leads to an error in the immediate context of that substitution, 
the function template is discarded, but other templates may still succeed:
```c++
template <typename T> 
typename T::EType f() {}  // #1

template <typename T> 
T f() {}                  // #2

int main() 
{
    auto x = f<int *>();
}
```
```c++
template <typename T> 
void f(T) {}

template <typename T> 
void f(T, T) {}

int main() 
{
    auto x = f<int *>;  // ERROR: Ambiguous. Two possible f<int *> here
}
```
is invalid because `f<int *>` does **not** identify a single function in that case.


Variadic function templates can also be used with explicit template arguments.
```c++
template <typename ... Ts>
void f(Ts ... ps);

void foo()
{
    f<double, double, int>(1, 2, 3);
}
```
A pack can be partially explicitly specified and partially deduced.
```c++
template <typename ... Ts>
void f(Ts ... ps);

void foo()
{
    f<double, int>(1, 2, 3);
}
```


#### 📌 15.10 Deduction from Initializers and Expressions


##### 15.10.1 `auto` And `decltype`

Refer to Effective Modern C++ for details.  

##### Deducible Non-type Parameters

Prior to C++17, non-type template arguments had to be declared with a specific type. 
That type could be a template parameter type: 
```c++
// Prior to C++17
template <typename T, T V> 
struct S {};

S<int, 42> s;
```
C++17 added the ability to declare non-type template parameters 
whose actual types are deduced from the corresponding template argument:
```c++
template <auto V> 
struct S {};

S<42> s;
```
Here the type of `V` for `S<42>` is deduced to be `int` because `42` has type `int`. 
Had we written `S<42U>` instead, the type of `V` would have been deduced to be `unsigned int`.


Note that the general constraints on the type of non-type template parameters remain in effect:
```c++
S<3.14> d;  // ERROR: floating-point non-type argument not allowed
```
A template definition with that kind of deducible non-type parameter
often also needs to express the actual type of the corresponding argument. 
That is easily done using the `decltype` construct:
```c++
template <auto V> 
struct Value 
{
    using ArgType = decltype(V);
};
```
`auto` non-type template parameters are also useful 
to parameterize templates on members of classes:
```c++
template <typename>
struct PMClassT {};

template <typename C, typename M>
struct PMClassT<M C::*>
{
    using Type = C;
};

template <typename PM> 
using PMClass = typename PMClassT<PM>::Type;

template <auto PMD>
class CounterHandle
{
public:
    CounterHandle(PMClass<decltype(PMD)> & c) : c(c) {}

    void incr() { ++(c.*PMD); }
    
private:
    PMClass<decltype(PMD)> & c;
};

struct S
{
    int i;
};

int main()
{
    S s {41};
    CounterHandle<&S::i> h(s);
    h.incr();  // increases s.i
    
    return EXIT_SUCCESS;
}
```
Prior to C++17, we also have to specify a pointer-member-type
```c++
OldCounterHandle<int S::*, &S::i>
```
This feature can also be used for non-type parameter packs:
```c++
template <auto ... VS>
struct Values {};

Values<1, 2, 3> beginning;
Values<1, 'x', nullptr> triplet;
```
The triplet example shows that each non-type parameter element of the pack 
can be deduced to a distinct type.


If we want to force a homogeneous pack of non-type template parameters, 
that is possible too: 
```c++
template <auto V1, decltype(V1) ... VRest> 
struct HomogeneousValues {};
```
However, the template argument list can **not** be empty in that particular case. 

##### 15.10.2 Expressing the Type of an Expression with `decltype`

The `decltype` keyword allows a programmer to express the _precise_ type of an expression or declaration. 
However, programmers should be careful about a subtle difference in what `decltype` produces, 
depending on whether the passed argument is a declared entity or an expression:
- If `e` is the name of an entity (such as a variable, function, enumerator, or data member) or a class member access, 
  `decltype(e)` yields the declared type of that entity or the denoted class member.
- Otherwise, if `e` is any other expression, 
  `decltype(e)` produces a type that reflects the type and value category of that expression as follows:
  - If `e` is an `lvalue` of type `T`, `decltype(e)` produces `T &`.
  - If `e` is an `xvalue` of type `T`, `decltype(e)` produces `T &&`.
  - If `e` is a `prvalue` of type `T`, `decltype(e)` produces `T`.
```c++
void g(std::string && s)
{
    // check the type of s: 
    std::is_lvalue_reference_v<decltype(s)>;        // false
    std::is_rvalue_reference_v<decltype(s)>;        // true
    std::is_same_v<decltype(s), std::string &>;     // false
    std::is_same_v<decltype(s), std::string &&>;    // true

    // check the value category of s used as expression: 
    std::is_lvalue_reference_v<decltype((s))>;      // true
    std::is_rvalue_reference_v<decltype((s))>;      // false
    std::is_same_v<decltype((s)), std::string &>;   // true
    std::is_same_v<decltype((s)), std::string &&>;  // false
}
```

##### 15.10.3 `decltype(auto)`

Since C++17, `decltype(auto)` can also be used for deducible non-type parameters. 
```c++
template <decltype(auto) Val> 
class S {};

constexpr int c = 42;
extern int v = 42;

S<c> sc;    // #1 produces S<42>
S<(v)> sv;  // #2 produces S<static_cast<int &>(v)>
```
`decltype` as call parameter is non-deduced context:
```c++
template <auto V> 
int f(decltype(V) p) {}

int r1 = f<42>(42);  // OK
int r2 = f(42);      // ERROR: decltype(V) is a non-deduced context
```
In this case, `decltype(V)` is a non-deduced context: 
There is **no** unique value of `V` that matches the argument `42` 
(e.g., `decltype(7)` produces the same type as `decltype(42)`). 
Therefore, the non-type template parameter must be specified explicitly to be able to call this function. 

##### 15.10.4 Special Situations for `auto` Deduction

`auto` with initializer list:
```c++
template <typename T>
void deduceT (T) {}

deduceT({2, 3, 4});  // ERROR
deduceT({1});        // ERROR
```
```c++
template <typename T>
void deduceInitList(std::initializer_list<T>);

deduceInitList({2, 3, 5, 7});  // OK: T deduced as int
```
Copy-initializing an `auto` variable with an initializer list 
is therefore defined in terms of that more specific parameter:
```c++
auto primes = {2, 3, 5, 7};  // OK: primes is std::initializer_list<int>
deduceT(primes);             // OK: T deduced as std::initializer_list<int>
```
Before C++17, the corresponding direct-initialization of `auto` variables was also handled that way, 
but this was changed in C++17 to better match the behavior expected by most programmers:
```c++
auto oops {0, 8, 15};  // ERROR in C++17
auto val {2};          // OK: val has type int in C++17
```
Prior to C++17, both initializations were valid, 
initializing both `oops` and `val` of type `initializer_list<int>`. 


It is invalid to return a braced initializer list for a function with a _deducible placeholder type_:
```c++
auto subtleError() 
{
    return {1, 2, 3};
}
```
That is because an initializer list in function scope 
is an object that _points to_ an underlying stack array object
(that expires when the function returns). 
Allowing the construct would thus encourage what is in effect a _dangling reference_.


Another special situation occurs when multiple variable declarations share the same `auto`:
```c++
auto first = container.begin(), last = container.end();
```
In such cases, deduction is performed independently for each declaration. 
In other words, there is an invented template type parameter `T1` for first 
and another invented template type parameter `T2` for last. 
Only if both deductions succeed, and the deductions for `T1` and `T2` are the same type, 
are the declarations well-formed. This can produce some interesting cases: 
```c++
char c;
auto * cp = &c, d = c;  // OK
auto e = c, f = c + 1;  // ERROR
```
A somewhat parallel special situation can also occur with placeholders for deduced return types.
```c++
auto f(bool b) 
{
    if (b) 
    {
        return 42.0;  // deduces return type double
    }
    else 
    {
        return 0;     // deduces return type double
    }
}
```
If the returned expression calls the function recursively, 
deduction can **not** occur and the program is invalid 
unless a prior deduction already determined the return type: 
```c++
auto f(int n)
{
    if (n > 1)
    {
        return n * f(n - 1);  // ERROR: type of f(n-1) unknown
    }
    else
    {
        return 1;
    }
}
```
but the following otherwise equivalent code is fine:
```c++
auto f(int n)
{
    if (n <= 1)
    {
        return 1;             // return type is deduced to be int
    }
    else
    {
        return n * f(n - 1);  // OK: type of f(n-1) is int and so istype of n*f(n-1)
    }
}
```
Deduced return types have another special case with no counterpart
in deduced variable types or deduced non-type parameter types:
```c++
auto f1() {}           // OK: return type is void
auto f2() { return; }  // OK: return type is void
```
However, if the return type pattern can **not** match `void`, such cases are invalid:
```c++
auto * f3() {}         //ERROR: auto * can not deduce as void
```
Any use of a function template with a deduced return type 
requires the _immediate instantiation of that template_ 
to determine the return type with certainty. 
This instantiation is **not** in the _immediate context of the function call_ 
and is thus **not** covered by SFINAE: 
```c++
template <typename T, typename U>
auto addA(T t, U u) -> decltype(t + u)
{
    return t + u;
}

void addA(...) {}

template <typename T, typename U>
auto addB(T t, U u) -> decltype(auto)
{
    return t + u;
}

void addB(...) {}

struct X {};

// OK: AddResultA is void
using AddResultA = decltype(addA(X(), X()));

// ERROR: 
// Instantiation of addB<X, X> is ill-formed because there is no operator+(const X &, const X &). 
// This error is NOT covered by SFINAE so results in compiling error (even though there is another match)! 
using AddResultB = decltype(addB(X(), X()));
```
Here, the use of `decltype(auto)` rather than `decltype(t + u)` for `addB` 
causes an error during overload resolution: 
The function body of the `addB` template must be fully instantiated to determine its return type. 
That instantiation **isn’t** in the immediate context of the call to `addB`, 
and therefore **doesn’t** fall under the SFINAE filter but results in an outright error. 


Deduced return types should be used with care, 
that they **shouldn’t** be called in the signatures 
of other function templates that would count on SFINAE properties. 

##### 15.10.5 Structured Bindings

Syntactically, a structured binding must always have an `auto` type 
optionally extended by `cv` qualifiers, `&` or `&&`, 
but **not** a `*` pointer declarator or some other declarator construct. 
It is followed by a bracketed list containing at least one identifier. 
That in turn has to be followed by an initializer.


Three different kinds of entities can initialize a structured binding:
1. **Simple class type, where all the non-static data members are public**. 
   - All the nonstatic data members have to be public 
     (either all directly in the class itself 
     or all in the same unambiguous public base class. 
     No anonymous unions may be involved). 
   - The number of bracketed identifiers must equal the number of members.  
   - Using one of these identifiers within the scope of the structured bindings 
     amounts to using the corresponding member of the object, 
     with all the associated properties. 
     - E.g., if the corresponding member is a bit field, it is invalid to take its address. 
2. **Arrays**. 
   - The bracketed initializers are just shorthand for the corresponding elements of the unnamed array variable. 
   - The number of array elements must equal the number of bracketed initializers. 
   ```c++
   double pt[3];
   auto & [x, y, z] = pt;
   x = 3.0; y = 4.0; z = 0.0;
   ```
   ```c++
   auto f() -> int (&)[2]; 
   auto [x, y] = f();       // #1
   auto & [r, s] = f();     // #2
   ```
   Line `#1` is special: 
   Ordinarily, the entity `e` would be deduced like this:
   ```c++
   auto e = f();
   ```
   However, that would deduce the _decayed_ pointer to the array, 
   which is **not** what happens when performing the structured binding of an array. 
   Instead, `e` is deduced to be a variable of type corresponding to the type of the initializer. 
   Then that array is _copied_ from the initializer, element by element: 
   That is a somewhat unusual concept for built-in arrays. 
   The other two places where built-in arrays are copied are lambda captures and generated copy constructors.
   Finally, `x` and `y` become aliases for the expressions `e[0]` and `e[1]`, respectively.   
   Line `#2`, does **not** involve array copying and follows the usual rules for `auto`. 
   So the hypothetical `e` is declared as follows:
   ```c++
   auto & e = f();
   ```
   which yields a reference to an array, 
   and `x` and `y` again become aliases for the expressions `e[0]` and `e[1]`.
3. **`std::tuple`-like classes**. 
   - Decomposed through a template-based protocol using `get`. 
   - Let `E` be `std::remove_reference_t<decltype((e))>` with `e` declared as above. 
     If the expression `std::tuple_size<E>::value` is a valid integral constant expression, 
     it must equal the number of bracketed identifiers. 
   - Denote the bracketed identifiers by `n0`, `n1`, `n2`, and so forth. 
     - If `e` has any member named `get`, then the behavior is as if these identifiers are declared as:
       - If `e` was deduced to have a reference type
       ```c++
       std::tuple_element<i, E>::type & ni = e.get<i>();
       ```
       - Otherwise
       ```c++
       std::tuple_element<i, E>::type && ni = e.get<i>();
       ```
     - If `e` has **no** member `get`, then the corresponding declarations become one of the following:
     ```c++
     std::tuple_element<i, E>::type & ni = get<i>(e);
     std::tuple_element<i, E>::type && ni = get<i>(e);
     ```
     where `get` is only looked up in associated classes and namespaces. 
     (In all cases, `get` is assumed to be a template and therefore the `<` follows is an angle bracket.)
   - The `std::tuple`, `std::pair`, and `std::array` templates all implement this protocol
   ```c++
   std::tuple<bool, int> bi{true, 42};
   auto [b, i] = bi;
   int r = i;
   // initializes r to 42
   ```
However, it is not difficult to add specializations of `std::tuple_size`, `std::tuple_element`, 
and a function template or member function template `get` 
that will make this mechanism work for an arbitrary class or enumeration type:
```c++
enum M {};

template <>
class std::tuple_size<M>
{
public:
    // map M to a pair of values
    static unsigned const value = 2; 
};

template <>
class std::tuple_element<0, M>
{
public:
    // the first value will have type int
    using type = int;
    
};

template <>
class std::tuple_element<1, M>
{
public:
    // the second value will have type double
    using type = double;
};

template <int>
auto get(M);

template <>
auto get<0>(M)
{
    return 42;
}

template <>
auto get<1>(M)
{
    return 7.0;
}

// as if: int && i = 42; double && d = 7.0;
auto [i, d] = M(); 
```
In addition, note that the third case above (using the tuple-like protocol) 
performs an actual initialization of the bracketed initializers 
and the bindings are actual reference variables. 
They are **not** just aliases for another expression 
(unlike the first two cases using simple class types and arrays). 


That’s of interest because that reference initialization could go wrong. 
For example, it might throw an exception, and that exception is now unavoidable. 

##### 15.10.6 Generic Lambdas

Within templates, lambdas can become fairly verbose
due to the need to spell out the parameter and result types. 
```c++
// C++11
template <typename Iter>
Iter findNegative(Iter first, Iter last)
{
    return std::find_if(first, last, [](typename std::iterator_traits<Iter>::value_type value)
    {
        return value < 0;
    });
}

// C++14
template <typename Iter>
Iter findNegative(Iter first, Iter last)
{
    return std::find_if(first, last, [](auto value)
    {
        return value < 0;
    });
}
```
An `auto` in a parameter of a lambda is handled similarly to 
an `auto` in the type of a variable with an initializer: 
It is replaced by an invented template type parameter `T`. 


However, unlike in the variable case, the deduction **isn’t** performed immediately
because the argument **isn’t** known at the time the lambda is created. 
Instead, the lambda itself becomes generic (if it wasn’t already), 
and the invented template type parameter is added to its template parameter list. 
Thus, the lambda above can be invoked with any argument type, 
so long as that argument type supports the `< 0` operation whose result is convertible to `bool`. 
For example, this lambda could be called with either an `int` or a `float` value.


Given the lambda
```c++
[] (int i) 
{
    return i < 0;
}
```
the C++ compiler translates this expression into an instance of a newly invented class specific to this lambda. 
This instance is called a _closure_ or _closure object_, and the class is called a _closure type_. 
The closure type has a function call operator, and hence the closure is a function object.
This translation model of lambdas is actually used in the specification of the C++ language, 
making it both a convenient and an accurate description of the semantics. 
Captured variables become data members, 
the conversion of a non-capturing lambda to a function pointer is modeled 
as a conversion function in the class, and so on. 
And because lambdas are function objects, 
whenever rules for function objects are defined, they also apply to lambdas. 


For this lambda, the closure type would look something like the following 
(leaving out the conversion function to a pointer-to-function value):
```c++
class SomeCompilerSpecificNameX
{
public:
    // only callable by the compiler
    SomeCompilerSpecificNameX(); 
    
    bool operator()(int i) const
    {
        return i < 0;
    }
};
```





### 🎯

#### 📌







15 
16 
17 Note the distinction between a placeholder type, which is auto or
decltype(auto) and can resolve to any kind of type, and a placeholder class
type, which is a template name and can only resolve to a class type that is an
instance of the indicated template.
18 As with ordinary function template deduction, SFINAE could apply if, for
example, substituting the deduced arguments in the guided type failed. That is not
the case in this simple example.
19 Chapter 16 introduces the ability to “specialize” class templates in various ways.
Such specializations do not participate in class template argument deduction.


### 🎯

#### 📌










### 🎯

#### 📌


## 🌱 Part III Templates and Design

### 🎯

#### 📌 










