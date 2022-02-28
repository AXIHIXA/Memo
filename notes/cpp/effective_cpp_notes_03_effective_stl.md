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






### 📌 Item 7: When using containers of `new`ed pointers, remember to `delete` the pointers before the container is destroyed

- Let containers contain smart pointers to eliminate resource leaks and keep exception safety. 






### 📌 Item 8: Never create containers of `std::auto_ptr`s

- `std::auto_ptr` itself is already deprecated since C++11. 





### 📌 Item 9: Choose carefully among erasing options

- **To eliminate all objects in a container that have a particular value**:
  - If the container is a `std::vector`, `std::string`, or `std::deque`, use the `erase-remove` idiom.
  - If the container is a `std::list`, use `std::list::remove`.
  - If the container is a standard associative container, use its `erase` member function.
- **To eliminate all objects in a container that satisfy a particular predicate**:
  - If the container is a `std::vector`, `std::string`, or `std::deque`, use the `erase-remove_if` idiom.
  - If the container is a `std::list`, use `std::list::remove_if`.
  - If the container is a standard associative container, use `std::remove_copy_if` and `std::swap`, 
    or write a loop to walk the container elements, 
    being sure to postincrement your iterator when you pass it to `erase`. 
- **To do something inside the loop (in addition to erasing objects)**:
  - If the container is a standard sequence container, 
    write a loop to walk the container elements, 
    being sure to update your iterator with `erase`’s return value each time you call it.
  - If the container is a standard associative container, 
    write a loop to walk the container elements, 
    being sure to postincrement your iterator when you pass it to `erase`.






### 📌 Item 10: Be aware of allocator conventions and restrictions

- Things you need to remember if you ever want to write a custom allocator: 
  - Make your allocator a template, with the template parameter `T` 
    representing the type of objects for which you are allocating memory. 
  - Provide the `typedef`s of `pointer` and `reference`, 
    but always have pointer be `T *` and reference be `T &`.
  - **Never** give your allocators per-object state. 
    The standard requires allocators with same `T` be identical with good reasons. 
    That means that allocators should have **no** non`static` data member. 
  - Remember that an allocator’s `allocate` member functions 
    are passed the number of objects for which memory is required, 
    not the number of Bytes needed. 
    Also remember that these functions return `T *` pointers (via the `pointer` `typedef`), 
    even though no `T` objects have yet been constructed. 
  - Allocators for `std::list` and all STL ordered associative containers 
    are **never** asked to allocate memory. 
  - Be sure to provide the nested `rebind` template on which standard containers depend. 


The list of restrictions on allocators begins with 
their vestigial `typedef`s for `pointer`s and `reference`s. 
Allocators were originally conceived of as abstractions for memory models, 
and as such it made sense for allocators to provide `typedef`s 
for `pointer`s and `reference`s in the memory model they defined. 
In the C++ standard, the default allocator for objects of type `T` (`std::allocator<T>`) 
offers the `typedef`s `std::allocator<T>::pointer` and `std::allocator<T>::reference`, 
and it is expected that user-defined allocators will provide these `typedef`s, too. 


Old C++ hands immediately recognize that this is suspect, 
because there’s **no** way to fake a reference in C++. 
Doing so would require the ability to "overload" `operator.`, 
which is actually **not** permitted. 
In addition, creating objects that act like references 
is an example of the use of proxy objects, 
and proxy objects lead to a number of problems. 
(One such problem motivates Item 18. Also refer to More Effective C++ Item 30.)


In the case of allocators in the STL, 
it’s not any technical shortcomings of proxy objects 
that render the `pointer` and `reference` `typedef`s impotent, 
it’s the fact that the Standard explicitly allows library implementers 
to assume that every allocator’s `pointer` `typedef` is a synonym for `T *` 
and every allocator’s `reference` `typedef` is the same as `T &`. 
That’s right, library implementers may ignore the `typedef`s and use raw pointers and references directly! 
So even if you could somehow find a way to write an allocator 
that successfully provided new pointer and reference types, it **wouldn’t** do any good, 
because the STL implementations you were using would be free to ignore your `typedef`s. 


While you’re admiring that quirk of standardization, I’ll introduce another.
Allocators are objects, and that means they may have member functions,
nested types and `typedef`s (such as `pointer` and `reference`), etc., 
but the Standard says that an implementation of the STL is permitted to assume that
all allocator objects of the same type are equivalent and always compare equal. 
Offhand, that doesn’t sound so awful, and there’s certainly good motivation for it. 
Consider this code:
```c++
class Widget { /* ... */ };

template <typename T>
class SpecialAllocator { /* ... */ };

using SpecialWidgetAllocator = SpecialAllocator<Widget>; 

std::list<Widget, SpecialWidgetAllocator> L1;
std::list<Widget, SpecialWidgetAllocator> L2;
L1.splice(L1.begin(), L2);
```
Recall that when `std::list` elements are `splice`d from one `std::list` to another, 
nothing is copied. 
Instead, a few pointers are adjusted, 
and the `std::list` nodes that used to be in one list find themselves in another. 
This makes splicing operations both fast and exception-safe. 
In the example above, the nodes that were in `L2` prior to the `splice` are in `L1` after the `splice`.


When `L1` is destroyed, of course, it must destroy all its nodes (and deallocate their memory), 
and because it now contains nodes that were originally part of `L2`, 
`L1`’s allocator must deallocate the nodes that were originally allocated by `L2`’s allocator. 
Now it should be clear why the Standard permits implementers of the STL to assume that 
allocators of the same type are equivalent. 
It’s so memory allocated by one allocator object (such as `L2`’s) 
may be safely deallocated by another allocator object (such as `L1`’s). 
Without being able to make such an assumption, 
`splice` operations would be more difficult to implement. 
Certainly they wouldn’t be as efficient as they can be now.


That’s all well and good, but the more you think about it, 
the more you’ll realize just how draconian a restriction it is 
that STL implementations may assume that allocators of the same type are equivalent. 
It means that portable allocator objects 
(allocators that will function correctly under different STL implementations) may **not** have state. 


Let’s be explicit about this: 
it means that _portable_ allocators may **not** have any non`static` data members, 
at least not any that affect their behavior. 
That means, for example, you can’t have one `SpecialAllocator<int>` that allocates from one heap 
and a different `SpecialAllocator<int>` that allocates from a different heap. 
Such allocators **wouldn’t** be equivalent, 
and STL implementations exist where attempts to use both allocators 
could lead to corrupt runtime data structures. 



Notice that this is a runtime issue. 
Allocators with state will compile just fine.
They just may not run the way you expect them to. 
The responsibility for ensuring that all allocators of a given type are equivalent is yours. 
**Don’t** expect compilers to issue a warning if you violate this constraint.
The C++ standard (It might be C++03 when Meyers wrote this book?) put the following statement immediately after 
the text that permits STL implementers to assume that allocators of the same type are equivalent:


> Implementors are encouraged to supply libraries that ... support non-equal instances. 
> In such implementations, ... the semantics of containers and algorithms 
> when allocator instances compare non-equal are implementation-defined.


This is a lovely sentiment, 
but as a user of the STL who is considering the development of a custom allocator with state, 
it offers you next to nothing.
You can take advantage of this statement only if:
1. You know that the STL implementations you are using support inequivalent allocators;
2. You are willing to delve into their documentation to determine 
   whether the implementation-defined behavior of “non-equal” allocators is acceptable to you;
3. You are not concerned about porting your code to STL implementations 
   that may take advantage of the latitude expressly extended to them by the Standard. 


I remarked earlier that allocators are like `operator new` in that they allocate raw memory, 
but their interface is different. 
This becomes apparent if you look at the declaration 
of the most common forms of `operator new` and `std::allocator<T>::allocate`:
```c++
void * operator new (std::size_t count);

template <typename T>
[[nodiscard]] constexpr T * std::allocator<T>::allocate(std::size_t n);
```
Both take a parameter specifying how much memory to allocate, 
but in the case of `operator new`, this parameter specifies a certain number of Bytes, 
while in the case of `std::allocator<T>::allocate`, 
it specifies how many `T` objects are to fit in the memory. 
On a platform where `sizeof(long) == 8`, 
you pass `8` to `operator new` if you wanted enough memory to hold an `long`, 
but you pass `1` to `std::allocator<long>::allocate`. 


`operator new` and `std::allocator<T>::allocate` differ in return types, too. 
`operator new` returns a `void *`, 
which is the traditional C++ way of representing a pointer to uninitialized memory. 
`std::allocator<T>::allocate` returns a `T *`, 
which is not only untraditional, but also premeditated fraud. 
The pointer returned from `std::allocator<T>::allocate` doesn’t point to a `T` object, 
because no `T` has yet been constructed! 
Implicit in the STL is the expectation that `std::allocator<T>::allocate`’s caller 
will eventually construct one or more `T` objects in the memory it returns 
(possibly via `std::allocator_traits<std::allocator<T>>::construct`, 
`std::uninitialized_fill`, or some application of `std::raw_storage_iterator`s),
though in the case of `std::vector::reserve` or `std::string::reserve`, that may never happen. 


That brings us to the final curiosity of STL allocators, 
that most of the standard containers never make a single call 
to the allocators with which they are instantiated:
```c++
// Same as std::list<int, std::allocator<int>>. 
// std::allocator<int> is never asked to allocate memory!
std::list<int> L;

class Widget { /* ... */ };

template <typename T>
class SpecialAllocator { /* ... */ };

using SpecialWidgetAllocator = SpecialAllocator<Widget>;

// SpecialWidgetAllocator will never allocate memory! 
std::set<Widget, SpecialWidgetAllocator> s;
```
This oddity is true for list and all STL ordered associative containers. 
That’s because these are _node-based containers_,
i.e., containers based on data structures 
in which a new node is dynamically allocated each time a value is to be stored. 
In the case of `std::list`, the nodes are list nodes. 
In the case of the STL ordered associative containers, 
the nodes are usually tree nodes, 
because the standard associative containers are typically
implemented as red-black trees. 


Think for a moment about how a `std::list<T>` is likely to be implemented.
The list itself will be made up of nodes,
each of which holds a `T` object as well as pointers to the next and previous nodes in the list:
```c++
namespace std
{

template <typename T, typename Allocator = std::allocator<T>>
class list
{
private:
    struct Node
    {
        T data;
        Node * pred;
        Node * succ;
    };
    
    Allocator alloc;
    
    ...
};

}  // namespace std
```
When a new node is added to the list, 
we need to get memory for it from an allocator, 
but we **don’t** need memory for a `T`, 
we need memory for a `std::list::Node` that contains a `T`. 
That makes our allocator object all but useless, 
because it doesn’t allocate memory for `std::list::Node`s, it allocates memory for `T`s. 
Now you understand why `std::list` never asks its allocator to do any allocation: 
the allocator **can’t** provide what `std::list` needs. 


What `std::list` needs is a way to get from the allocator type it has 
to the corresponding allocator for `std::list::Node`s. 
By convention, allocators provide a `typedef` `std::allocator<T>::template rebind<U>::other`that does the job.
```c++
namespace std
{

template <typename T>
class allocator
{
public:
    template <typename U>
    struct rebind
    {
        typedef allocator<U> other;
    };
    
    ...
};

}  // namespace std
```
In the code implementing `std::list<T>`, 
there is a need to determine the type of the allocator for `std::list::Node`s 
that corresponds to the allocator we have for `T`s. 
The type of the allocator we have for `T`s is the template parameter `Allocator`. 
That being the case, the type of the corresponding allocator for `std::list::Node`s is this:
```c++
Allocator::rebind<Node>::other
```
Every allocator template `A` (e.g., `std::allocator`) is expected to have 
a nested struct template called `rebind`. 
`rebind` takes a single type parameter `U`, and defines nothing but a `typedef` `other`. 
`other` is simply a name for `A<U>`. 
As a result, `std::list<T>` can get from its allocator for `T` objects (called `Allocator`) 
to the corresponding allocator for `std::list<T>::Node` objects 
by referring to `Allocator::rebind<std::list<T>::Node>::other`. 


As a user of the STL who may want to write a custom allocator, 
you don’t really need to know how it works. 
What you do need to know is that if you choose to write allocators 
and use them with the standard containers, 
your allocators must provide the `rebind` template, 
because standard containers assume it will be there. 
(For debugging purposes, it’s also helpful to know why node-based containers of `T` objects 
never ask for memory from the allocators for `T` objects.)






### 📌 Item 11: Understand the legitimate uses of custom allocators

- Allocators are used to customize STL container's memory management. 


So you’ve benchmarked, profiled, and experimented your way to the conclusion
that the default STL memory manager (i.e., `std::allocator<T>`) is too slow, wastes memory, 
or suffers excessive fragmentation for your STL needs,
and you’re certain you can do a better job yourself. 
Or you discover that `std::allocator<T>` takes precautions to be thread-safe, 
but you’re interested only in single-threaded execution, 
and you don’t want to pay for the synchronization overhead you don’t need. 
Or you know that objects in certain containers are typically used together, 
so you’d like to place them near one another in a special heap to maximize locality of reference. 
Or you’d like to set up a unique heap that corresponds to shared memory, 
then put one or more containers in that memory, so they can be shared by other processes. 
Each of these scenarios corresponds to a situation 
where custom allocators are well suited to the problem.


For example, suppose you have special routines modeled after `std::malloc` and `std::free`
for managing a heap of shared memory,
and you’d like to make it possible to put the contents of STL containers in that shared memory: 
```c++
[[nodiscard]] void * mallocShared(std::size_t bytesNeeded);
void freeShared(void * ptr) noexcept;

template <typename T>
class SharedMemoryAllocator 
{
public:
    ...
    
    pointer allocate(size_type numObjects, const void * localityHint = 0)
    {
        return static_cast<pointer>(mallocShared(numObjects * sizeof(T)));
    }
    
    void deallocate(pointer ptrToMemory, size_type numObjects)
    {
        freeShared(ptrToMemory);
    }
    
    ...
};
```
You could use `SharedMemoryAllocator` like this:
```c++
using SharedDoubleVec = std::vector<double, SharedMemoryAllocator<double>>;

{
    ...
    // create a vector whose elements are in shared memory
    SharedDoubleVec v;  
    ...
}
```
The wording in the comment next to `v`’s definition is important. 
`v` is using a `SharedMemoryAllocator`, 
so the memory `v` allocates to hold its elements will come from shared memory.
However, `v` itself (including all its data members) will almost certainly **not** be placed in shared memory. 
`v` is just a normal stack-based object, 
so it will be located in whatever memory the runtime system uses for all normal stack-based objects. 
That’s almost never shared memory. 
To put both `v`’s contents and `v` itself into shared memory, you’d have to do something like this:
```c++
// allocate enough shared memory to hold a SharedDoubleVec object
void * pVectorMemory = mallocShared(sizeof(SharedDoubleVec));

// use "placement new" to construct a SharedDoubleVec object in the memory
SharedDoubleVec * pv = new (pVectorMemory) SharedDoubleVec;

// use the object (via pv)
// ...

// destruct the object in the shared memory
pv->~SharedDoubleVec();

// deallocate the initial chunk of shared  memory
freeShared(pVectorMemory);
```
Fundamentally, you acquire some shared memory, 
then construct a `std::vector` in it that uses shared memory for its own internal allocations. 
When you’re done with the vector, you invoke its destructor, 
then release the memory the vector occupied. 
The code isn’t terribly complicated, 
but it’s a lot more demanding than just declaring a local variable as we did above. 
Unless you really need a container (as opposed to its elements) to be in shared memory, 
I encourage you to avoid this manual four-step allocate/construct/destroy/deallocate process.


In this example, you’ve doubtless noticed that the code ignores the possibility
that `mallocShared` might return a null pointer. 
Obviously, production code would have to take such a possibility into account. 
Also, construction of the vector in the shared memory is accomplished by _placement `new`_.


As a second example of the utility of allocators, 
suppose you have two heaps, identified by the classes `Heap1` and `Heap2`. 
Each heap class has `static` member functions for performing allocation and deallocation:
```c++
class Heap1 
{
public:
    ...
    static void * alloc(std::size_t numBytes, const void *memoryBlockToBeNear);
    static void dealloc(void * ptr);
    ...
};

// has the same interface
class Heap2 { ... }; 
```
Further, suppose you’d like to co-locate the contents of some STL containers in different heaps. 
First you write an allocator designed to use classes like `Heap1` and `Heap2` for the actual memory management:
```c++
template <typename T, typename Heap>
class SpecificHeapAllocator
{
public:
    ...

    pointer allocate(size_type numObjects, const void * localityHint = 0)
    {
        return static_cast<pointer>(Heap::alloc(numObjects * sizeof(T), localityHint));
    }

    void deallocate(pointer ptrToMemory, size_type numObjects)
    {
        Heap::dealloc(ptrToMemory);
    }

    ...
};
```
Then you use `SpecificHeapAllocator` to cluster containers’ elements together:
```c++
// put both v's and s's elements in Heap1
std::vector<int, SpecificHeapAllocator<int, Heap1>> v; 
std::set<int, SpecificHeapAllocator<int, Heap1>> s;

// put both L's and m's elements in Heap2
std::list<Widget, SpecificHeapAllocator<Widget, Heap2>> L; 
std::map<int, std::string, 
         std::less<int>, 
         SpecificHeapAllocator<std::pair<const int, std::string>, Heap2>> m;
```
In this example, it’s quite important that `Heap1` and `Heap2` be types and not objects. 
The STL offers a syntax for initializing different STL containers with different allocator objects of the same type, 
but I’m not going to show you what it is. 
That’s because if `Heap1` and `Heap2` were objects instead of types,
they’d be inequivalent allocators, 
and that would violate the equivalence constraint on allocators that is detailed in Item 10.


As these examples demonstrate, 
allocators are useful in a number of contexts.
As long as you obey the constraint that 
all allocators of the same type must be equivalent, 
you’ll have no trouble employing custom allocators 
to control general memory management strategies, 
clustering relationships, and use of shared memory and other special heaps.






### 📌 Item 12: Have realistic expectations about the thread safety of STL containers

- STL has **no** thread-safety guarantees on its algorithms and containers. 
- Use `std::lock_guard` to manage `std::mutex`es when accessing STL containers. 


**None** of the following locking policies on STL containers are thread-safe:
- Lock a container for the duration of each call to its member functions. 
- Lock a container for the lifetime of each iterator it returns (via, e.g., calls to `begin` or `end`).
- Lock a container for the duration of each algorithm invoked on that container.
  (This actually makes no sense, because, as Item 32 explains, 
  algorithms have no way to identify the container on which they are operating. 
  Nevertheless, we’ll examine this option here, because it’s instructive to see 
  why it wouldn’t work even if it were possible.)


Now consider the following code. 
It searches a `std::vector<int>` for the first occurrence of the value `5` and changes that value to `0`.
```c++
std::vector<int> v;
auto it = std::find(v.begin(), v.end(), 5);  // Line 1
if (it != v.end())                           // Line 2
{ 
    *it = 0;                                 // Line 3
}
```
In a multithreaded environment, 
it’s possible that a different thread will modify the data in `v` 
immediately after completion of Line 1. 
If that were to happen, the test of `it` against `v.end` on Line 2 would be meaningless, 
because `v`’s values would be different from what they were at the end of Line 1. 

In fact, such a test could yield undefined results, 
because another thread could have intervened between Lines 1 and 2 and invalidated `it`, 
perhaps by performing an insertion that caused the vector to reallocate its underlying memory. 
(That would invalidate all the vector’s iterators. 
For details on this reallocation behavior, turn to Item 14.) 
Similarly, the assignment to `*it` on Line 3 is unsafe,
because another thread might execute between Lines 2 and 3 in such a way as to invalidate `it`, 
perhaps by erasing the element it points to (or at least used to point to).


**None** of the approaches to locking listed above would prevent these problems. 
The calls to `begin` and `end` in Line 1 both return too quickly to offer any help, 
the iterators they generate last only until the end of that line, 
and `std::find` also returns at the end of that line.


For the code above to be thread safe, 
`v` must remain locked from Line 1 through Line 3, 
and it’s difficult to imagine how an STL implementation could deduce that automatically. 
Bearing in mind the typically high cost of synchronization primitives 
(e.g., semaphores, mutexes, etc.), 
it’s even more difficult to imagine how an implementation could do it 
without imposing a significant performance penalty on programs that knew a priori, 
that were designed in such a way that 
no more than one thread had access to `v` during the course of Lines 1-3.


Such considerations explain why you **can’t** expect 
any STL implementation to make your threading woes disappear. 
Instead, you’ll have to manually take charge of synchronization control in these kinds of scenarios. 
In this example, you might do it like this:
```c++
std::vector<int> vec {0, 1, 2, 3, 4, 5};
std::mutex vecMutex;

{
    std::lock_guard g(vecMutex);
    auto it = std::find(vec.begin(), vec.end(), 5);
    if (it != vec.end())  *it = 0;
}
```






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




