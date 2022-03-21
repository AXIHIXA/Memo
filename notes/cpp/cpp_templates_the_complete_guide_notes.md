# _C++ Templates: The Complete Guide_ Notes

- _**C++ Templates**: The Complete Guide_ Second Edition by
  - David Vandevoorde
  - Nicolai M. Josuttis
  - Douglas Gregor
- Contents
    - **[Part I The Basics](./cpp_templates_the_complete_guide_notes.md)**
    - [Part II Templates in Depth](./cpp_templates_the_complete_guide_notes_part_2.md)
    - [Part III Templates and Design](./cpp_templates_the_complete_guide_notes_part_3.md)






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

Template member functions can be used wherever special member functions allow copying or moving objects. 
Similar to assignment operators as defined above, they can also be constructors. 
However, note that template constructors or template assignment operators 
**don’t** replace predefined (compiler-generated) constructors or assignment operators. 
Member templates **don’t** count as _the_ special member functions that copy or move objects. 
In this example, for assignments of stacks of the same type, the default assignment operator is still called. 


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


Member function templates can also be used as special member functions, including as a constructor. 
While, a constructor template with forwarding reference could hide the copy constructor.
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
for a non-`const` lvalue `Person p`, the member template
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

