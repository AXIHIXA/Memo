# _C++ Templates: The Complete Guide_ Notes

- _**C++ Templates**: The Complete Guide_ Second Edition
- David Vandevoorde
- Nicolai M. Josuttis
- Douglas Gregor






## 🌱 Part I. The Basics


### 🎯 Chapter 1. Function Templates

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
and discussed in detail in Chapter 14.


Note that some compilers don’t perform the full checks of the first phase. 
So you might not see general problems until the template code is instantiated at least once.

#### Compiling and Linking**

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
  Qualifications with const or volatile are ignored, 
  references convert to the referenced type, 
  and raw arrays or functions convert to the corresponding pointer type. 
  For two arguments declared with the same template parameter `T`, 
  the decayed types must match.


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
template <typename T>  // T is template parameter
```
2. _Call parameters_, 
   which are declared in parentheses after the function template name:
```c++
T max(T a, T b);       // a and b are call parameters
```







## 🌱

### 🎯

### 📌 








