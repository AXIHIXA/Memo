# _Effective C++_ Notes

- Notes of reading <u>Scott Meyers</u>'s _Effective C++_ series:
    1. *[`Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_01_effective_cpp.md)*
    2. *[`More Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_02_more_effective_cpp.md)*
    3. ***[`Effective STL`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_03_effective_stl.md)***
    4. *[`Effective Modern C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_04_effective_modern_cpp.md)*






---

## 🌱 _Effective STL_

### 🎯 Chapter 1. Containers

### 📌 Item 1: Choose your containers with care

### 📌 Item 2: Beware the illusion of container-independent code

- Usage of `using` type aliases makes types easier to be extended.  


Given the inevitability of having to change container types from time to time,
you can facilitate such changes in the usual manner: 
by encapsulating, encapsulating, encapsulating. 
One of the easiest ways to do this is through the liberal use of 
`using` type aliases for container types. 
Hence, instead of writing this,
```c++
class Widget { ... };
std::vector<Widget> vw;
Widget bestWidget;
auto it = std::find(vw.begin(), vw.end(), bestWidget); 
```
write this:
```c++
class Widget { ... };
using WidgetContainer = std::vector<Widget>;
WidgetContainer cw;
Widget bestWidget;
auto it = std::find(cw.begin(), cw.end(), bestWidget);
```
This makes it a lot easier to change container types, 
something that’s especially convenient if the change in question 
is simply to add a custom allocator. 
(Such a change doesn’t affect the rules for iterator/pointer/reference invalidation.)
```c++
class Widget { ... };

template <typename T>
class SpecialAllocator { ... };

using WidgetContainer = std::vector<Widget, SpecialAllocator<Widget>>;
WidgetContainer cw;
Widget bestWidget;
auto it = std::find(cw.begin(), cw.end(), bestWidget);
```






### 📌 Item 3: Make copying cheap and correct for objects in containers


An easy way to make copying efficient, correct, and immune to the slicing problem 
is to create containers of _pointers_ instead of containers of objects.
That is, instead of creating a container of `Widget`, create a container of `std::shared_ptr<Widget>`. 
Copying pointers is fast, it always does exactly what you expect 
(it copies the bits making up the pointer), 
and nothing gets sliced when a pointer is copied.
You can read about them in Items 7 and 33. 






### 📌 Item 4: Call `empty()` instead of checking `size()` against zero

- `empty()` is _constant-time_ operation for all standard containers, 
- `std::list<T, Allocator>::size` may take: 
  - constant or _linear time_ `(until C++11)`;
  - constant time `(since C++11)`. 






### 📌 Item 5: Prefer range member functions to their single-element counterparts

- Range member functions are favored over their single-element counterparts because 
  range member functions could be faster under several circumstances 
  (e.g., `std::list<T, Allocator>::assign` involves `pred`/`succ` pointer assignments). 
- Range construction, range insertion, range erasure. 


```c++
std::vector<int> v1, v2;
v1.assign(v2.vbegin() + v2.size() / 2, v2.cend());

// NOT GOOD
// loops, etc. 
```






### 📌 Item 6: Be alert for C++’s most vexing parse

- Notice the difference between parentheses:
  - **Around a parameter name**:
    Are ignored;
  - **Standing by themselves**:
    Indicate the existence of a _parameter list_.
    They announce the presence of a parameter that is itself a pointer to a function.
- To fight C++'s most vexing parse:
  - Pass named arguments instead of anonymous objects; 
  - It’s **not** legal to surround a formal parameter declaration with parentheses, 
    this may be used to prone out the undesired parsing; 
  - Adopt uniform initialization. 


Suppose you have a file of `int`s and you’d like to copy those ints into a `std::list`.
This seems like a reasonable way to do it:
```c++
// Warning! This doesn't do what you think it does
std::ifstream dataFile("ints.dat");
std::list<int> data(std::istream_iterator<int>(dataFile), std::istream_iterator<int>());
```
The idea here is to pass a pair of `std::istream_iterator`s to `std::list`’s range constructor, 
thus copying the ints in the file into the list. 


This code will compile, but at runtime, it **won’t** do anything. 
It **won’t** read any data out of a file. It **won’t** even create a list. 
That’s because the second statement doesn’t declare a list and it doesn’t call a constructor. 


We’ll start with the basics.
This line declares a function `f` taking a `double` and returning an `int`:
```c++
int f(double d);
```
This next line does the same thing. 
The parentheses around the parameter name `d` are superfluous and are ignored:
```c++
int f(double (d));
```
The line below declares the same function. It simply omits the parameter name:
```c++
int f(double);
```
Those three declaration forms should be familiar to you, 
though the ability to put parentheses around a parameter name may have been new. 


Let’s now look at three more function declarations. 
The first one declares a function `g` taking a parameter
that’s a pointer to a function taking nothing and returning a `double`:
```c++
int g(double (* pf)());
```
Here’s another way to say the same thing. 
The only difference is that `pf` is declared using _non-pointer syntax_ 
(a syntax that’s valid in both C and C++):
```c++
// same as above; pf is implicitly a pointer
int g(double pf()); 
```
As usual, parameter names may be omitted, so here’s a third declaration for `g`,
one where the name `pf` has been eliminated:
```c++
// same as above; parameter name is omitted
int g(double ()); 
```
Notice the difference between parentheses: 
- **Around a parameter name**: 
  Are ignored; 
- **Standing by themselves**:
  Indicate the existence of a _parameter list_. 
  They announce the presence of a parameter that is itself a pointer to a function.


Having warmed ourselves up with these declarations for `f` and `g`, 
we are ready to examine the code that began this Item. Here it is again:
```c++
// Warning! This doesn't do what you think it does
std::ifstream dataFile("ints.dat");
std::list<int> data(std::istream_iterator<int>(dataFile), std::istream_iterator<int>());
```
This f**king statement declares a _function_, `data`, whose return type is `std::list<int>`.
The function data takes two parameters:
- **The first parameter** is named `dataFile`. 
  Its type is `std::istream_iterator<int>`.
  The parentheses around `dataFile` are superfluous and are ignored. 
- **The second parameter** has no name. 
  Its type is pointer to function taking nothing and returning an `std::istream_iterator<int>`. 


Amazing, huh? But it’s consistent with a universal rule in C++, 
which says that pretty much anything that can be parsed as a function declaration will be. 
If you’ve been programming in C++ for a while, 
you’ve almost certainly encountered another manifestation of this rule. 
How many times have you seen this mistake?
```c++
class Widget { ... };  // assume Widget has a default constructor
Widget w();            // uh oh...
```
This **doesn’t** declare a `Widget` named `w`, 
it declares a function named `w` that takes nothing and returns a `Widget`. 
Learning to recognize this sh*t is a veritable rite of passage for C++ programmers. 


All of which is interesting (in its own twisted way),
but it doesn’t help us say what we want to say, 
which is that a `std::list<int>` object should be initialized with the contents of a file. 
Now that we know what parse we have to defeat, that’s easy to express. 
It’s **not** legal to surround a formal parameter declaration with parentheses, 
but it is legal to surround an argument to a function call with parentheses,
so by adding a pair of parentheses, we force compilers to see things our way:
```c++
// note new parens around first argument to std::list's constructor
std::list<int> data((std::istream_iterator<int>(dataFile)), std::istream_iterator<int>()); 
```
This is the proper way to declare data, and given the utility of
istream_iterators and range constructors (again, see Item 5), it’s worth
knowing how to do it.


A better solution is to step back from the 
trendy use of anonymous `std::istream_iterator` objects in `data`’s declaration
and simply give those iterators names. 
The following code should work everywhere:
```c++
std::ifstream dataFile("ints.dat");
std::istream_iterator<int> dataBegin(dataFile);
std::istream_iterator<int> dataEnd;
std::list<int> data(dataBegin, dataEnd);
```
This use of named iterator objects runs contrary to common STL programming style, 
but you may decide that’s a price worth paying for code 
that’s unambiguous to both compilers and the humans who have to work with them.






### 📌 Item 7: When using containers of newed pointers, remember to delete the pointers before the container is destroyed






### 📌 Item 8: Never create containers of `std::auto_ptr`s

- `std::auto_ptr` itself is already deprecated since C++11, so this item is also outdated. 






### 📌 Item 9: Choose carefully among erasing options






### 📌 Item 10: Be aware of allocator conventions and restrictions






### 📌 Item 11: Understand the legitimate uses of custom allocators






### 📌 Item 12: Have realistic expectations about the thread safety of STL containers






### 🎯 Chapter 2. `std::vector` and `std::string`

### 📌 Item 13: Prefer `std::vector` and `std::string` to dynamically allocated arrays






### 📌 Item 14: Use `reserve()` to avoid unnecessary reallocations






### 📌 Item 15: Be aware of variations in `std::string` implementations






### 📌 Item 16: Know how to pass `std::vector` and `std::string` data to legacy APIs






### 📌 Item 17: Use “the `swap` trick” to trim excess capacity






### 📌 Item 18: Avoid using `std::vector<bool>`






### 🎯 Chapter 3. Associative Containers

### 📌 Item 19: Understand the difference between equality and equivalence






### 📌 Item 20: Specify comparison types for associative containers of pointers






### 📌 Item 21: Always have comparison functions return `false` for equal values






### 📌 Item 22: Avoid in-place key modification in `std::set` and `std::multiset`






### 📌 Item 23: Consider replacing associative containers with sorted `std::vector`s






### 📌 Item 24: Choose carefully between `std::map::operator[]` and `std::map::insert` when efficiency is important






### 📌 Item 25: Familiarize yourself with the nonstandard hashed containers






### 🎯 Chapter 4. Iterators

### 📌 Item 26: Prefer `iterator` to `const_iterator`, `reverse_iterator`, and `const_reverse_iterator`

- Deprecated. Refer to _Effective Modern C++_ Item 13 for details. 






### 📌 Item 27: Use `distance()` and `advance()` to convert a container’s `const_iterator`s to `iterator`s






### 📌 Item 28: Understand how to use a `reverse_iterator`’s base `iterator`






### 📌 Item 29: Consider `std::istreambuf_iterator`s for character-by-character input






### 🎯 Chapter 5. Algorithms

### 📌 Item 30: Make sure destination ranges are big enough






### 📌 Item 31: Know your sorting options






### 📌 Item 32: Follow `remove`-like algorithms by erase if you really want to remove something






### 📌 Item 33: Be wary of `remove`-like algorithms on containers of pointers






### 📌 Item 34: Note which algorithms expect sorted ranges






### 📌 Item 35: Implement simple case-insensitive string comparisons via `mismatch` or `lexicographical_compare`






### 📌 Item 36: Understand the proper implementation of `std::copy_if`






### 📌 Item 37: Use `std::accumulate` or `std::for_each` to summarize ranges






### 🎯 Chapter 6. Functors, Functor Classes, Functions, etc. 

### 📌 Item 38: Design functor classes for pass-by-value






### 📌 Item 39: Make predicates pure functions






### 📌 Item 40: Make functor classes adaptable






### 📌 Item 41: Understand the reasons for `std::ptr_fun`, `std::mem_fun`, and `std::mem_fun_ref`






### 📌 Item 42: Make sure `std::less<T>` means `operator<`






### 🎯 Chapter 7. Programming with the STL 

### 📌 Item 43: Prefer algorithm calls to hand-written loops






### 📌 Item 44: Prefer member functions to algorithms with the same names






### 📌 Item 45: Distinguish among `std::count`, `std::find`, `std::binary_search`, `std::lower_bound`, `std::upper_bound`, and `std::equal_range`






### 📌 Item 46: Consider function objects instead of functions as algorithm parameters






### 📌 Item 47: Avoid producing write-only code






### 📌 Item 48: Always `#include` the proper headers






### 📌 Item 49: Learn to decipher STL-related compiler diagnostics






### 📌 Item 50: Familiarize yourself with STL-related web sites




