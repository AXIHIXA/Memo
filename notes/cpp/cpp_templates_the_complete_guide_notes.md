# _C++ Templates: The Complete Guide_ Notes

- _**C++ Templates**: The Complete Guide_ Second Edition
- David Vandevoorde
- Nicolai M. Josuttis
- Douglas Gregor






## 🌱 Part I The Basics


### 🎯 Chapter 1 Function Templates

#### 📌 1.1 A First Look at Function Templates

##### Instantiation

Templates aren’t compiled into single entities that can handle any type. 
Instead, different entities are generated from the template for every type for which the template is used.


The process of replacing template parameters by concrete types is called _instantiation_. 
It results in an _instance_ of a template.


Note that the mere use of a function template can trigger such an instantiation process. 
There is no need for the programmer to request the instantiation separately.

##### Two-Phase Translation

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

##### Compiling and Linking

Two-phase translation leads to an important problem in the handling of templates in practice: 
When a function template is used in a way that triggers its instantiation, 
which is at _compile time_, 
a compiler will (at some point) need to see that template’s _definition_. 
This breaks the usual compile and link distinction for ordinary functions, 
when the declaration of a function is sufficient to compile its use. 
Methods of handling this problem are discussed in Chapter 9. 
For the moment, let’s take the simplest approach: 
Implement each template inside a header file.

#### 📌 1.2 Template Argument Deduction

##### Type Conversions During Type Deduction

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
regardless of the caller’s intention. 
Thus, the return type depends on the call argument order. 
The maximum of `66.66` and `42` will be the `double` `66.66`, 
while the maximum of `42` and `66.66` will be the `int` `66`. 


C++ provides different ways to deal with this problem (to de detailed in the next section):
- Introduce a third template parameter for the return type.
- Let the compiler deduce the return type.
- Declare the return type to be the “common type” of the two parameter types. 

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
Note that these modified versions of `max` **don’t** lead to significant advantages. 
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

##### Return Type as Common Type

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


#### 📌 1.6 But, Shouldn’t We...?

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

##### Class Template Arguments Deduction with String Literals

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


For function and class templates, template parameters **don’t** have to be types. 
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

##### Variadic Base Classes and using

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
you have the problem that a simple definition doesn’t do this for built-in types:
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
the arguments **don’t** decay. 
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
However, you can’t assign a stack with elements of any other type,
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
First let’s look at the syntax to define a _member template_. 
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
floatStack = stringStack;        // ERROR: std::string doesn’t convert to float
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
Because the assignment operator template isn’t necessary,
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
Note that you don’t need and also **can’t** declare the specializations; you only define them. 
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
Member templates **don’t** count as _the_ special member functions that copy or move objects.
These template versions also **don’t** count as user-defined versions. 
In this example, for assignments of stacks of the same type, 
the default assignment operator is still called. 


This effect can be good and bad:
- It can happen that a template constructor or assignment operator 
  is a better match than the predefined copy/move constructor or assignment operator, 
  although a template version is provided for initialization of other types only. 
  See Section 6.2 for details.
- It is not easy to “templify” a copy/move constructor, 
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
A simple lambda computing the “sum” of two arguments of arbitrary types:
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

// N has value ’c’ of type char
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
But it wasn’t until C++17 that a corresponding change was made
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
  **don’t** replace predefined constructors or assignment operators.
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
If it weren’t, we would lose the value of a movable object the first time we use it in a function.


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
Note that `std::move` has no template parameter and “triggers” move semantics for the passed argument, 
while `std::forward` “forwards” potential move semantic depending on a passed template argument.
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
Don’t assume that ` &&` for a template parameter `T` behaves as `X &&` for a specific type `X`. 
Different rules apply! 
However, syntactically they look identical:
- `X &&` for a specific type `X` declares a parameter to be an rvalue reference. 
  It can only be bound to a movable object 
  (a prvalue, such as a temporary object, and an xvalue, such as an object passed with `std::move`; 
  see Appendix B for details). 
  It is always mutable and you can always “steal” its value. 
  - A type like `X const &&` is valid but provides **no** common semantics in practice
    because “stealing” the internal representation of a movable object requires
    modifying that object. 
    It might be used, though, to force passing only temporaries or objects marked with `std::move` 
    without being able to modify them. 
- `T &&` for a template parameter `T` declares a _forwarding reference_ (also called _universal reference_). 
  - The term _universal reference_ was coined by Scott Meyers prior to C++17 
    as a common term that could result in either an “lvalue reference” or an “rvalue reference”. 
    The C++17 standard introduced the term _forwarding reference_, 
    because the major reason to use such a reference is to forward objects. 
    However, note that it does **not** automatically forward. 
    The term does not describe what it is but what it is typically used for. 
  - It can be bound to a mutable, immutable (i.e., `const`), or movable object. 
    Inside the function definition, the parameter may be mutable, immutable, 
    or refer to a value you can “steal” the internals from. 


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


Now let’s replace the two `std::string` constructors 
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
          typename = std::enable_if_t<
                  std::is_convertible_v<String, std::string>>>
explicit Person::Person(String && n);
```
If type `String` is convertible to type `std::string`, the whole declaration expands to
```c++
template <typename String, typename = void>
explicit Person::Person(String && n);
```
If type `String` is not convertible to type `std::string`, the whole function template is ignored.


If you wonder why we don’t instead check whether `String` is “not convertible to `Person`”, beware:
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

Note that normally we **can’t** use `std::enable_if` to disable 
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
Now the template constructors are used even for “normal” copying:
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
concept Hashable = requires(T a)
{
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
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
No conversion is permitted:
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
concept Addable = requires (T x) { x + x; }; 
 
// requires-clause, not requires-expression
template <typename T> requires Addable<T> 
T add(T a, T b) { return a + b; }
 
// ad-hoc constraint, note keyword requires used twice
template <typename T> requires requires (T x) { x + x; } 
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
  - **Nested Requirements**: 


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


- In templates, you can “perfectly” forward parameters by declaring them as forwarding references 
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
   meaning that you can modify or “steal” the value.
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









## 🌱

### 🎯

#### 📌 








