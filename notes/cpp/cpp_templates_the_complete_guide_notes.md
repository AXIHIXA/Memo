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

#### Type Conversions During Type Deduction

- During template type deduction,
  arguments' reference-ness and top-level cv-constraints are ignored.
    - Note only pointers and references have top-level cv-constraints.
      A raw type like `const int` is considered bottom-level cv-constraint.
- When deducing types for universal reference parameters,
  reference collapse may occur.
- During template type deduction,
  arguments that are array or function names decay to pointers,
  unless they’re used to initialize references.
- Automatic type conversions are limited during type deduction:
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


Note that **code is instantiated _only for template (member) functions that are called_**. 
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
You can still use this class for elements that **don’t** have `operator<<` defined:
```c++
// note: std::pair<> has no operator<< defined
Stack<std::pair<int, int>> ps; 
ps.push({4, 5});                       // OK
ps.push({6, 7});                       // OK
std::cout << ps.top().first << '\n';   // OK
std::cout << ps.top().second << '\n';  // OK
```
Only if you call `printOn` for such a stack, the code will produce an error, 
because it can’t instantiate the call of `operator<<` for this specific element type:
```c++
// ERROR: operator<< not supported for element type
ps.printOn(std::cout);
```

#### Concepts

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


### 📌 2.4 Friends


Instead of printing the stack contents with `printOn`, 
it is better to implement `operator<<` for the stack.
However, as usual `operator<<` has to be implemented as nonmember function, 
which then could call `printOn` inline. 
Note that this means that `operator<<` for class `Stack` is **not** a function template, 
but an “ordinary” function instantiated with the class template if needed.
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
   Note the `<T>` behind the “function name” `operator<<`. 
   Thus, we declare a specialization of the non-member function template as friend. 
   Without `<T>` we would declare a new non-template function. 
   See Section 12.5 for details.


In any case, you can still use this class for elements that don’t have `operator<<` defined. 
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


### 📌 2.5 Specializations of Class Templates


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
For these specializations, any definition of a member function must be defined as an “ordinary” member function,
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


### 📌 2.6 Partial Specialization


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

#### Partial Specialization with Multiple Parameters

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


### 📌 2.7 Default Class Template Arguments


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


### 📌 2.8 Type Aliases

#### Typedefs and Alias Declarations

```c++
// Typedef
typedef Stack<int> intStack;

// Type alias
using IntStack = Stack<int>;
```

#### Alias Templates

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

#### Alias Templates for Member Types

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

#### Type Traits Suffix_t

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


### 📌 2.9 Class Template Argument Deduction


Prior to C++17, 
you _always_ had to pass _all_ template parameter types to class templates, 
unless they have default values. 
Since C++17, the constraint was relaxed. 
Instead, you can skip to define the templates arguments explicitly, 
if the _constructor_ is able to deduce all template parameters (that don’t have a default value).
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

#### Class Template Arguments Deduction with String Literals

In principle, you can even initialize the stack with a string literal:
```c++
Stack stringStack = "bottom";  // Stack<char const[7]> deduced since C++17
```
**BUT** this causes a lot of trouble: 
In general, when passing arguments of a template type `T` _by reference_, 
the parameter **doesn’t** _decay_ (low-level `const`-ness is kept).  
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

#### Deduction Guides


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
However, by language rules, you **can’t** [copy initialize](https://en.cppreference.com/w/cpp/language/copy_initialization) 
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


### 📌 2.10 Templatized Aggregates


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


### 📌 2.11 Summary


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


For function and class templates, template parameters **don’t** have to be types. 
They can also be ordinary values.
When using such a template, you have to specify the value template arguments explicitly. 
The resulting code then gets instantiated.


### 📌 3.1 Nontype Class Template Parameters


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


### 📌 3.2 Nontype Function Template Parameters


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


### 📌 3.3 [Restrictions for Nontype Template Parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter)


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

#### Avoiding Invalid Expressions

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


### 📌 3.4 Template Parameter Type `auto`


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


### 📌 3.5 Summary


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


### 📌 4.1 Variadic Templates


Template parameters can be defined to accept an unbounded number of template arguments. 
Templates with this ability are called _variadic templates_.

#### Variadic Templates: Function Parameter Pack And Template Parameter Pack

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

#### Overloading Variadic and Nonvariadic Templates

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

#### Operator `sizeof...`

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
However, this approach **doesn’t** work,
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


### 📌 4.2 [Fold Expressions](https://en.cppreference.com/w/cpp/language/fold)


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


### 📌 4.3 Application of Variadic Templates


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
Usually, the arguments are “_perfectly forwarded_” with move semantics (see Section 6.1).


Note also that the same rules apply to variadic function template parameters as for ordinary parameters. 
For example, if passed by value, arguments are copied and decay (e.g., arrays become pointers), 
while if passed by reference, parameters refer to the original parameter and don’t decay:
```c++
// args are copies with decayed types:
template <typename ...Args> 
void foo(Args ... args);

// args are nondecayed references to passed objects:
template <typename ... Args> 
void bar(Args const & ... args);
```


### 📌 4.4 Variadic Class Templates and Variadic Expressions


Besides the examples above, parameter packs can appear in additional places,
including, for example, expressions, class templates, using declarations, and even deduction guides. 
Section 12.4 has a complete list.

#### Variadic Expressions

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

#### Variadic Indices

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

#### Variadic Class Templates


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

#### Variadic Deduction Guides

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

#### Variadic Base Classes and using

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


### 📌 4.5 Summary


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


### 📌 5.1 Keyword `typename`










## 🌱

### 🎯

### 📌 








