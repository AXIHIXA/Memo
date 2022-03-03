# _Effective C++_ Notes

- Notes of reading <u>Scott Meyers</u>'s _Effective C++_ series:
    1. ***[`Effective C++`](./effective_cpp_notes_01_effective_cpp.md)***
    2. *[`More Effective C++`](./effective_cpp_notes_02_more_effective_cpp.md)*
    3. *[`Effective STL`](./effective_cpp_notes_03_effective_stl.md)*
    4. *[`Effective Modern C++`](./effective_cpp_notes_04_effective_modern_cpp.md)*






---

## 🌱 _Effective C++_

### 🎯 Chapter 1. Accustoming Yourself to C++

### 📌 Item 1: View C++ as a federation of languages

- Rules for effective C++ programming vary, depending on the part of C++ you are using.






### 📌 Item 2: Prefer `const`s, `enum`s, and `inline`s to `#define`s
    
- For simple constants, prefer `const`, `constexpr` objects or `enum`s to `#define`s.
- For function-like macros, prefer `inline` functions to `#define`s.


This Item might better be called “prefer the compiler to the preprocessor,”
because `#define` may be treated as if it’s not part of the language. 
That’s one of its problems. When you do something like this,
```c++
#define ASPECT_RATIO 1.653
```
the symbolic name `ASPECT_RATIO`may be removed by the preprocessor before the source code gets to a compiler. 
As a result, the name `ASPECT_RATIO` may not get entered into the <u>_symbol table_</u>. 
This can be confusing if you get an error during compilation involving the use of the constant, 
because the error message may refer to 1.653, not `ASPECT_RATIO`. 


The solution is to replace the macro with a constant:
```c++
const double ASPECT_RATIO = 1.653;
```
As a language constant, `ASPECT_RATIO` is definitely seen by compilers and is certainly entered into their symbol tables. 
In addition, in the case of a floating point constant (such as in this example), 
use of the constant may yield smaller code than using a `#define`. 
That’s because the preprocessor’s blind substitution of the macro name `ASPECT_RATIO` with 1.653 
could result in multiple copies of 1.653 in your object code,
while the use of the constant `ASPECT_RATIO` should never result in more than one copy.


When replacing `#define`s with constants, two special cases are worth mentioning. 
The first is defining <u>_constant pointers_</u>. 
Because constant definitions are typically put in header files (where many different source files will include them), 
it’s important that the pointer be declared `const`, usually in addition to what the pointer points to. 
To define a constant `char *`-based string in a header file, for example, you have to write const twice:
```c++
const char * const authorName = "Scott Meyers";
```
For a complete discussion of the meanings and uses of `const`, especially in conjunction with pointers, see Item 3. 
However, it’s worth reminding you here that string objects are generally preferable to their `char *`-based progenitors, 
so `authorName` is often better defined this way:
```c++
const std::string authorName("Scott Meyers");
```
The second special case concerns <u>_class-specific constants_</u>. 
To limit the scope of a constant to a class, you must make it a member, 
and to ensure there’s at most one copy of the constant, you must make it a `static` member:
```c++
class GamePlayer
{
private:
    static const int NUM_TURNS = 5;   // constant declaration & in-class initialization
    int scores[NUM_TURNS];            // use of constant
    // ...
};
```
What you see above is a <u>_declaration_</u> for `NUM_TURNS`, **not** a definition.
Usually, C++ requires that you provide a definition for anything you use, 
but class-specific constants that are `static` and of integral type (e.g., integers, `char`s, `bool`s) are an exception.
As long as you don’t take their address, you can declare them and use them without providing a definition. 
If you do take the address of a class constant, 
or if your compiler incorrectly insists on a definition even if you don’t take the address, 
you provide a separate definition like this:
```c++
// GamePlayer.h
class GamePlayer 
{
private: 
    static const int NUM_TURNS = 5;   // constant declaration & in-class initialization
    int scores[NUM_TURNS];            // use of constant
};

// GamePlayer.cpp
const int GamePlayer::NUM_TURNS;      // definition of NumTurns; see below for why no value is given
```
You put the definition in an implementation file, not a header file. 
Because the initial value of class constants is provided where the constant is declared 
(e.g., `NUM_TURNS` is initialized to 5 when it is declared), no initial value is permitted at the point of definition.
Note, by the way, that there’s no way to create a class-specific constant using a `#define`, 
because `#define`s don’t respect scope. 
Once a macro is defined, it’s in force for the rest of the compilation (unless it’s `#undef`ed somewhere along the line). 
Which means that not only can’t `#define`s be used for class-specific constants, 
they also can’t be used to provide any kind of encapsulation, 
i.e., there is no such thing as a “private” `#define`.
Of course, `const` data members can be encapsulated; `NUM_TURNS` is.


Older compilers may not accept the syntax above, 
because it used to be illegal to provide an initial value for a static class member at its point of declaration. 
Furthermore, in-class initialization is allowed only for integral types and only for constants. 
In cases where the above syntax can’t be used, you put the initial value at the point of definition:
```c++
// "CostEstimate.h"
class CostEstimate
{
private:
    static const double FUDGE_FACTOR;            // declaration of static class constant
    // ...                  
};

// "CostEstimate.cpp"
const double CostEstimate::FUDGE_FACTOR = 1.35;  // definition of static class constant
```
The above block of code is all you need almost all the time. 
The only exception is when you *need the value of a class constant during compilation* of the class, 
such as in the declaration of the array `GamePlayer::scores` above 
(where compilers insist on knowing the size of the array during compilation). 
Then the accepted way to compensate for compilers that (incorrectly) forbid 
the in-class specification of initial values for static integral class constants 
is to use what is affectionately (and non-pejoratively) known as the `enum` hack. 


(P.S. This book, _Effective C++_, is written for C++98. 
Since C++11, `constexpr` is definitely a better choice than the `enum` hack. 
Refer to _Effective Modern C++_ for details. )


This technique takes advantage of the fact that the values of an enumerated type can be used where ints are expected, 
so `GamePlayer` could just as well be defined like this:
```c++
class GamePlayer 
{
private:
    enum {NUM_TURNS = 5};   // "the enum hack" makes NUM_TURNS a symbolic name for 5
    int scores[NUM_TURNS];  // fine
};
```
The `enum` hack is worth knowing about for several reasons. 
- **Access Constraints**. 
  The `enum` hack behaves in some ways more like a `#define` than a `const` does, 
  and sometimes that's what you want. 
  For example, it's legal to take the address of a `const`, 
  but it's **not legal** to take the address of an `enum`, 
  and it's typically **not legal** to take the address of a `#define`, either. 
  If you don't want to let people get a pointer or reference to one of your integral constants, 
  an enum is a good way to enforce that constraint. 
- **Memory Allocation**. 
  Though good compilers won't set aside storage for const objects of integral types 
  (unless you create a pointer or reference to the object), 
  sloppy compilers may, and you may not be willing to set aside memory for such objects. 
  Like `#define`s, `enum`s never result in that kind of unnecessary memory allocation.
- **Pragmatic**. 
  Lots of code employs it, so you need to recognize it when you see it. 
  In fact, the `enum` hack is a fundamental technique of template metaprogramming. 


Another common misuse of the `#define` directive is that without proper parenthesis: 
```c++
// WRONG! should be 
// #define MAX(a, b) ((a) > (b) ? (a) : (b)) 
// to make it work!
#define MAX(a, b) (a) > (b) ? (a) : (b)  

int a = 10;
int b = 15;
std::cout << 10 * MAX(a, b) << '\n';  // 10
```
The even more terrible case is macros that look like functions but that don’t incur the overhead of a function call: 
```c++
// call f with the maximum of a and b
// even if everything is properly parenthesised, there can still be problems! 
#define CALL_WITH_MAX(a, b) f((a) > (b) ? (a) : (b))

int a = 5;
int b = 0;
CALL_WITH_MAX(++a, b);       // a is incremented TWICE!
CALL_WITH_MAX(++a, b + 10);  // a is incremented once
```
You can get all the efficiency of a macro plus all the predictable behavior and type safety 
of a regular function by using a `template` for an `inline` function: 
```c++
template <typename T1, typename T2>
inline void callWithMax(T1 && a, T2 && b)
{
    f(a > b ? std::forward<T1>(a) : std::forward<T2>(b));
}
```
This template generates a whole family of functions, 
each of which takes two objects of the same type and calls `f` with the greater of the two objects. 
There's no need to parenthesize parameters inside the function body, 
no need to worry about evaluating parameters multiple times, etc. 
Furthermore, because `callWithMax` is a real function, it obeys scope and access rules. 
For example, it makes perfect sense to talk about an inline function that is private to a class. 
In general, there's just no way to do that with a macro.


Given the availability of `const`s, `enum`s (`constexpr`s), and `inline`s, 
your need for the preprocessor (especially `#define`) is reduced, but it’s not eliminated.
`#include` remains essential, 
and `#ifdef` / `#ifndef` continue to play important roles in controlling compilation. 
It’s not yet time to retire the preprocessor, but you should definitely give it long and frequent vacations.






### 📌 Item 3: Use `const` whenever possible

- Declaring something `const` helps compilers detect usage errors. 
  `const` can be applied to objects at any scope, 
  to function parameters and return types, 
  and to member functions as a whole.
- Compilers enforce bitwise constness, 
  but you should program using logical constness.
- When `const` and non-`const` member functions have essentially identical implementations, 
  code duplication can be avoided by having the non-`const` version call the `const` version. 

#### `const` iterators

*STL iterators are modeled on pointers*, so an iterator acts much like a `T *` pointer. 
Declaring an iterator `const` is like declaring a pointer `const` (i.e., declaring a `T * const` pointer): 
the iterator isn't allowed to point to something different, but the thing it points to may be modified. 
If you want an iterator that points to something that can't be modified 
(i.e., the STL analogue of a `const T *` pointer), you want a `const_iterator`:
```c++
std::vector<int> vec {0, 1, 2, 3, 4, 5};

const std::vector<int>::iterator iter = vec.begin();    // iter acts like a T * const
*iter = 10;                                             // OK, changes what iter points to
++iter;                                                 // error! iter is const

std::vector<int>::const_iterator cIter = vec.cbegin();  // cIter acts like a const T*
*cIter = 10;                                            // error! *cIter is const
++cIter;                                                // fine, changescIter
```

#### `const`s in function return values

Having a function return a constant value often makes it possible to 
reduce the incidence of client errors without giving up safety or efficiency: 
```c++
class Rational { /* ... */ };

const Rational operator*(const Rational & lhs, const Rational & rhs);
```
Many programmers squint when they first see this. 
Why should the result of `operator*` be a `const` object? 
Because if it weren't, clients would be able to commit atrocities like this:
```c++
Rational a, b, c;
(a * b) = c;       // invoke operator= on the result of (a * b)!
```
I don't know why any programmer would want to make an assignment to the product of two numbers, 
but I do know that many programmers have tried to do it without wanting to. 
All it takes is a simple typo (and a type that can be implicitly converted to `bool`):
```c++
if (a * b = c)     // oops, meant to do a comparison!
{
    // ...
}
```
Such code would be flat-out illegal if `a` and `b` were of a built-in type. 
One of the hallmarks of good user-defined types is that they avoid gratuitous incompatibilities with the built-ins, 
and allowing assignments to the product of two numbers seems pretty gratuitous to me. 
Declaring `operator*`'s return value `const` prevents it, and that's why it's The Right Thing To Do. 

#### `const` member functions

Many people overlook the fact that *member functions differing only in their `const`ness can be overloaded*, 
but this is an important feature of C++. 
Incidentally, `const` objects most often arise in real programs 
as a result of being passed by pointer-to-const or reference-to-const.
What does it mean for a member function to be `const`? 
There are two prevailing notions: 

##### Bitwise constness (also known as _physical constness_) 

The bitwise `const` camp believes that a member function is `const`
iff. it doesn't modify any of the object's data members (excluding those that are `static`), 
i.e., iff. it *doesn't modify any of the bits inside the object*. 
The nice thing about bitwise constness is that it's easy to detect violations: 
compilers just look for assignments to data members. 
In fact, bitwise constness is C++'s definition of constness, 
and a `const` member function isn't allowed to modify 
any of the non-`static` data members of the object on which it is invoked. 


Unfortunately, many member functions that don't act very `const` pass the bitwise test. 
In particular, *a member function that modifies what a pointer points* to frequently doesn't act `const`. 
But if only the pointer is in the object, the function is bitwise `const`, and compilers won't complain. 
That can lead to counterintuitive behavior. 


For example, suppose we have a `TextBlock`-like class that stores its data as a `char *` instead of a `string`, 
because it needs to communicate through a C API that doesn't understand `string` objects. 
```c++
class CTextBlock 
{
public:
    // inappropriate (but bitwise const) declaration of operator[]
    char & operator[](std::size_t position) const 
    { 
        return pText[position]; 
    } 
    
private:
    char * pText;
};
```
This class (inappropriately) declares `operator[]` as a `const` member function,
even though that function returns a reference to the object's internal data. 
Set that aside and note that `operator[]`'s implementation doesn't modify `pText` in any way. 
As a result, compilers will happily generate code for `operator[]`; 
it is, after all, bitwise `const`, and that's all compilers check for. 
But look what it allows to happen:
```c++
const CTextBlock cctb("Hello");  // declare constant object
char *pc = &cctb[0];             // call the const operator[] to get a  pointer to cctb's data
*pc = 'J';                       // cctb now has the value "Jello"
```

##### Logical constness

A `const` member function *might modify some of the bits in the object* on which it's invoked, 
but *only in ways that clients cannot detect*. 
For example, your `CTextBlock` class might want to cache the length of the textblock whenever it's requested:
```c++
class CTextBlock 
{
public:
    std::size_t length() const;
    
private:
    char * pText;
    std::size_t textLength;               // last calculated length of textblock
    bool lengthIsValid;                   // whether length is currently valid
};

std::size_t CTextBlock::length() const
{
    if (!lengthIsValid) 
    {
        textLength = std::strlen(pText);  // error! can't assign to textLength and lengthIsValid in a const member function
        lengthIsValid = true;
    } 
    
    return textLength;
}
```
The solution is simple: take advantage of C++'s `const`-related wiggle room known as `mutable`. 
`mutable` frees non-`static` data members from the constraints of bitwise constness: 
```c++
class CTextBlock 
{
public:
    std::size_t length() const;
    
private:
    char * pText;
    mutable std::size_t textLength;       // last calculated length of textblock
    mutable bool lengthIsValid;           // whether length is currently valid
};

std::size_t CTextBlock::length() const
{
    if (!lengthIsValid) 
    {
        textLength = std::strlen(pText);  // can modify mutables in a const member function
        lengthIsValid = true;
    } 
    
    return textLength;
}
```

#### Avoiding Duplication in `const` and Non-`const` Member Functions

```c++
class TextBlock 
{
public:
    const char & operator[](std::size_t position) const  // same as before
    {
        // do a lot of stuff here...
        return text[position];
    }
    
    char & operator[](std::size_t position)              // now just calls const op[]
    {
        // 1. add const to (*this)'s type;
        // 2. call const version of op[];
        // 3. cast away const on op[]'s return type. 
        return const_cast<char &>(static_cast<const TextBlock &>(*this)[position]);
    }
};
```
To avoid infinite recursion, 
we have to specify that we want to call the `const` `operator[]`, 
but there's no direct way to do that. 
Instead, we cast `*this` from its native type of `TextBlock &` to `const TextBlock &`.
Yes, we use a cast to add const! So we have two casts: 
one to add const to `*this` (so that our call to `operator[]` will call the `const` version), 
the second to remove the `const` from the `const` `operator[]`'s return value. 
The cast that adds `const` is just forcing a safe conversion 
(from a non-`const` object to a `const` one), 
so we use a `static_cast` for that. 
The one that removes `const` can be accomplished only via a `const_cast`, 
so we don't really have a choice there. 
<br><br>
Even more worth knowing is that trying to do things the other way around 
(avoiding duplication by ~~having the `const` version call the non-`const` version~~)
is **not** something you want to do. 
Remember, a `const` member function promises never to change the logical state of its object, 
but a non-`const` member function makes no such promise. 
If you were to call a non-`const` function from a `const` one, 
you'd run the risk that the object you'd promised not to modify would be changed. 
That's why having a `const` member function call a non-`const` one is wrong: 
the object could be changed. 
In fact, to get the code to compile, you'd have to use a `const_cast` to get rid of the `const` on `*this`,
a clear sign of trouble. 
The reverse calling sequence (the one we used above) is safe: 
the non-`const` member function can do whatever it wants with an object, 
so calling a `const` member function imposes no risk. <br>
That's why a `static_cast` works on `*this` in that case: there's no `const`-related danger. 






### 📌 Item 4: Make sure that objects are initialized before they're used

- Manually initialize objects of built-in type, because C++ only sometimes initializes them itself.
- Base classes are initialized before derived classes,
  and within a class, data members are initialized in the order in which they are declared. 
- In a constructor, prefer use of the member initialization list to assignment inside the body of the constructor. 
  List data members in the initialization list in the same order they're declared in the class.
- Avoid initialization order problems across translation units 
  by replacing non-local `static` objects with local `static` objects (via Meyers-singleton-like getter functions).


there are rules that describe when object initialization is guaranteed to take place and when it isn’t.
Unfortunately, the rules are too complicated to be worth memorizing. 
The best way to deal with this seemingly indeterminate state of affairs 
is to always initialize your objects before you use them.
For non-member objects of built-in types, you’ll need to do this manually: 
```c++
int x = 0;                               // manual initialization of an int
const char * text = "A C-style string";  // manual initialization of a pointer
double d;                                // "initialization" by reading from an input stream
std::cin >> d; 
```
For almost everything else, the responsibility for initialization falls on constructors. 
The rule is simple: make sure that *all constructors initialize everything in the object*. 


It’s important not to confuse <u>_assignment_</u> with <u>_initialization_</u>. 
Consider a constructor for a class representing entries in an address book:
```c++
class PhoneNumber 
{ 
    // ... 
};

class ABEntry  // Address Book Entry
{
public:
    ABEntry(const std::string & name, 
            const std::string & address, 
            const std::list<PhoneNumber> & phones)
    {
        // these are all assignments, not initializations
        theName = name; 
        theAddress = address; 
        thePhones = phones;
        numTimesConsulted = 0;
    }

private:
    std::string theName;
    std::string theAddress;
    std::list<PhoneNumber> thePhones;
    int num TimesConsulted;
};
```
This will yield `ABEntry` objects with the values you expect, but it's still not the best approach. 
The rules of C++ stipulate that data members of an object are initialized _before_ the body of a constructor is entered. 
Inside the `ABEntry` constructor, 
`theName`, `theAddress`, and `thePhones` **aren't** ~~being initialized~~, they're being _assigned_. 
Initialization took place earlier: 
when their default constructors were automatically called prior to entering the body of the `ABEntry` constructor. 
This isn't true for `numTimesConsulted`, because it's a built-in type. 
For it, there's **no** guarantee it was initialized at all prior to its assignment. 


A better way to write the `ABEntry` constructor is to use the <u>_member initialization list_</u> instead of assignments:
```c++
ABEntry::ABEntry(const std::string & name, 
                 const std::string & address, 
                 const std::list<PhoneNumber> & phones)
        : theName(name), 
          theAddress(address), 
          thePhones(phones), 
          numTimesConsulted(0)
{
    // the constructor body is now empty
}
```
This constructor yields the same end result as the one above, but it will often be more efficient.


The assignment-based version first called <u>_default constructor_</u>s to initialize `theName`, `theAddress`, and `thePhones`,
then promptly assigned new values on top of the default-constructed ones. 
All the work performed in those default constructions was therefore **wasted**. 


The member initialization list approach avoids that problem, 
because the arguments in the initialization list are used as constructor arguments for the various data members. 
In this case, `theName` is copy-constructed from `name`, `theaddress` is copy-constructed from `address`, 
and `thePhones` is copy-constructed from `phones`.
For most types, a single call to a copy constructor is more efficient (sometimes much more efficient)
than a call to the default constructor followed by a call to the copy assignment operator. 


For objects of built-in type like `numTimesConsulted`, 
there is no difference in cost between initialization and assignment, 
but for consistency, it's often best to initialize everything via member initialization.


Similarly, you can use the member initialization list even when you want to default-construct a data member; 
just specify nothing as an initialization argument. 
For example, if `ABEntry` had a constructor taking no parameters, it could be implemented like this:
```c++
ABEntry::ABEntry()
        : theName(),            // call theName's default constructor;
          theAddress(),         // do the same for theAddress;
          thePhones(),          // and for thePhones;
          numTimesConsulted(0)  // but explicitly initialize numTimesConsulted to zero
{

} 
```
Because compilers will automatically call default constructors for data members of user-defined types 
when those data members have no initializers on the member initialization list, 
some programmers consider the above approach overkill. 
That’s understandable, but having a policy of always listing every data member on the initialization list 
avoids having to remember which data members may go uninitialized if they are omitted. 
Because `numTimesConsulted` is of a built-in type, for example, 
leaving it off a member initialization list could open the door to undefined behavior.


Sometimes the initialization list must be used, even for built-in types.
For example, data members that are `const` or are references **can’t** be assigned and thus must be initialized. 
To avoid having to memorize when data members must be initialized in the member initialization list and when it’s optional, 
the easiest choice is to always use the initialization list. 
It’s sometimes required, and it’s often more efficient than assignments.


Many classes have multiple constructors, and each constructor has its own member initialization list. 
If there are many data members and/or base classes, 
the existence of multiple initialization lists introduces 
undesirable repetition (in the lists) and boredom (in the programmers).
In such cases, it’s not unreasonable to omit entries in the lists 
for data members where assignment works as well as true initialization,
moving the assignments to a single (typically private) function that all the constructors call.
This approach can be especially helpful if the true initial values for the data members
are to be read from a file or looked up in a database. 
In general, however, true member initialization (via an initialization list) 
is preferable to pseudo-initialization via assignment.


One aspect of C++ that isn’t fickle is the order in which an object’s data is initialized. 
This order is always the same: 
**base classes are initialized before derived classes, 
and within a class, data members are initialized in the order in which they are declared.**
In `ABEntry`, for example, `theName` will always be initialized first, 
`theAddress` second, `thePhones` third, and `numTimesConsulted` last. 
This is true even if they are listed in a different order on the member initialization list. 
To avoid reader confusion, as well as the possibility of some truly obscure behavioral bugs,
always list members in the initialization list in the same order as they’re declared in the class.


Once you’ve taken care of explicitly initializing non-member objects of built-in types 
and you’ve ensured that your constructors initialize their base classes and data members using the member initialization list, 
there’s only one more thing to worry about. 
That thing is the order of initialization of non-local `static` objects defined in different translation units.


A <u>_static object_</u> is one that has 
[static storage duration](./cpp_primer_notes.md#-%E5%AD%98%E5%82%A8%E6%9C%9F%E5%92%8C%E9%93%BE%E6%8E%A5storage-duration-and-linkage) 
(i.e. exists from the time it’s constructed until the end of the program). 
Stack-based objects (auto storage duration) and heap-based objects (dynamic storage duration) are thus excluded.
Included are global objects, objects defined at namespace scope, objects declared static inside classes, 
objects declared static inside functions, and objects declared static at file scope. 
Static objects inside functions are known as <u>_local static objects_</u> (because they’re local to a function), 
and the other kinds of static objects are known as <u>_non-local static objects_</u>. 
Static objects are destroyed when the program exits, i.e., their destructors are called when main finishes executing.


A <u>_translation unit_</u> is the source code giving rise to a single object file, say, `foo.o`. 
It’s basically a single source file, plus all of its `#include` files.


The problem we’re concerned with, then, involves at least two separately compiled source files, 
each of which contains at least one non-local static object 
(i.e., an object that’s global, at namespace scope, or static in a class or at file scope). 
And the actual problem is this: 
if initialization of a non-local static object `a` in one translation unit 
uses another non-local static object `b` in a different translation unit, 
`b` could be uninitialized when `a` refers to it, 
because the relative order of initialization of non-local static objects defined in different translation units is undefined.


An example will help. 
Suppose you have a `FileSystem` class that makes files on the Internet look like they’re local.
```c++
// "FileSystem.h"
class FileSystem
{
public:
    // ...
    std::size_t numDisks() const;  // one of many member functions
    // ...
};


// "FileSystem.cpp"
// declare object for clients to use (“tfs” = “the file system” );
extern FileSystem tfs; 
```
A `FileSystem` object is decidedly non-trivial, so use of the `tfs` object before its construction would be disastrous.
Now suppose some client creates a class for directories in a file system.
Naturally, their class uses the `tfs` object:
```c++
class Directory
{ 
public:
    Directory(params);
    // ...
};


Directory::Directory(params)
{
    // ...
    std::size_t disks = tfs.numDisks();  // use the tfs object
    // ...
}
```
Further suppose this client decides to create a single `Directory` object for temporary files:
```c++
Directory tempDir(params); // directory for temporary files
```
Now the importance of initialization order becomes apparent: 
unless `tfs` is initialized before `tempDir`, 
`tempDir`’s constructor will attempt to use `tfs` before it’s been initialized. 
But `tfs` and `tempDir` are non-local static objects defined in different translation units.
The relative order of initialization of non-local static objects defined in different translation units is undefined, 
because determining the “proper” order in which to initialize non-local static objects very hard. 
In its most general form with multiple translation units and non-local static objects
generated through implicit template instantiations (which may themselves arise via implicit template instantiations), 
it’s not only impossible to determine the right order of initialization,
it’s typically not even worth looking for special cases where it is possible to determine the right order.


Fortunately, a small design change eliminates the problem entirely.
All that has to be done is to move each non-local static object into its own function, where it’s declared static. 
These functions return references to the objects they contain. 
Clients then call the functions instead of referring to the objects.
(i.e. writing get functions as of the Singleton pattern, except for the limit on number of instances.)
In other words, non-local static objects are replaced with local static objects.


This approach is founded on C++’s guarantee that local static objects are initialized 
when the object’s definition is first encountered during a call to that function. 
So if you replace direct accesses to non-local static objects 
with calls to functions that return references to local static objects, 
you’re guaranteed that the references you get back will refer to initialized objects. 
As a bonus, if you never call a function emulating a non-local static object, 
you never incur the cost of constructing and destructing the object, 
something that can’t be said for true non-local static objects.


Here’s the technique applied to both `tfs` and `tempDir`:
```c++
class FileSystem { /* ... */ };

// this replaces the tfs object; 
// it could be static in the FileSystem class
FileSystem & tfs() 
{
    static FileSystem fs;  // define & initialize a local static object
    return fs;             // return a reference to it
}

class Directory { /* ... */ };

// as before, except references to tfs are now to tfs()
Directory::Directory(params) 
{ 
    // ...
    std::size_t disks = tfs().numDisks();
    // ...
}

// this replaces the tempDir object; 
// it could be static in the Directory class
// Meyers' Singleton flavor
Directory & tempDir() 
{ 
    static Directory td(params);  // define & initialize local static object
    return td;                    // return reference to it
}
```
Clients of this modified system program exactly as they used to,
except they now refer to `tfs()` and `tempDir()` instead of `tfs` and `tempDir`.
That is, they use functions returning references to objects instead of using the objects themselves.


The reference-returning functions dictated by this scheme are always simple: 
**define and initialize a local static object, and return a reference to it**. 
This simplicity makes them excellent candidates for inlining, especially if they’re called frequently. 
On the other hand, the fact that these functions contain static objects makes them problematic in multithreaded systems. 
Then again, any kind of non-`const` static object (local or non-local) 
is trouble waiting to happen in the presence of multiple threads. 
One way to deal with such trouble is to manually invoke all the reference-returning functions 
during the single- threaded startup portion of the program. 
This eliminates initialization-related race conditions.


Of course, the idea of using reference-returning functions to prevent initialization order problems 
is dependent on there being a reasonable initialization order for your objects in the first place. 
If you have a system where object `A` must be initialized before object `B`, 
i.e., there is no deadlock on initialization ordering.


To avoid using objects before they’re initialized, then, you need to do only three things. 

1. Manually initialize non-member objects of built-in types;
2. Use member initialization lists to initialize all parts of an object;
3. Design around the initialization order uncertainty that 
   afflicts non-local static objects defined in separate translation units (via Meyers-singleton-like getter functions).






### 🎯 Chapter 2. Constructors, Destructors, and Assignment Operators

### 📌 Item 5: Know what functions C++ silently writes and calls

**OUTDATED**. Refer to _Effective Modern C++_ Item 17. 






### 📌 Item 6: Explicitly disallow the use of compiler-generated functions you do not want

**OUTDATED**. Refer to _Effective Modern C++_ Item 11.



### 📌 Item 7: Declare destructors virtual in polymorphic base classes

- Polymorphic base classes should declare virtual destructors. 
  If a class has any virtual functions, it should have a virtual destructor.
- Classes not designed to be base classes or not designed to be used polymorphically 
  should **not** declare virtual destructors (virtual function tables and pointers are not for free).

Consider the following `TimeKeeper` base class along with derived classes for different approaches to timekeeping:
```c++
class TimeKeeper 
{
public:
    TimeKeeper();
    ~TimeKeeper();
    // ...
};

class AtomicClock : public TimeKeeper { /* ... */ };
class WaterClock : public TimeKeeper { /* ... */ };
class WristWatch : public TimeKeeper { /* ... */ };
```
Many clients will want access to the time without worrying about the details of how it’s calculated, 
so a factory function that returns a base class pointer to a newly-created derived class object
can be used to return a pointer to a timekeeping object:
```c++
// returns a pointer to a dynamically allocated object of a class derived from TimeKeeper
std::unique_ptr<TimeKeeper> getTimeKeeper();
```
The problem is that `getTimeKeeper` returns a pointer to a derived class object (e.g., `AtomicClock`), 
that object is being deleted via a base class pointer (i.e., a `std::unique_ptr<TimeKeeper>`), 
and the base class (`TimeKeeper`) has a <u>_non-virtual destructor_</u>. 
This is a recipe for disaster, because C++ specifies that when a derived class object 
is deleted through a pointer to a base class with a non-virtual destructor, results are undefined.
What typically happens at runtime is that the derived part of the object is never destroyed, 
thus leading to a curious “partially destroyed” object.


Eliminating the problem is simple: 
give the base class a virtual destructor.
Then deleting a derived class object will do exactly what you want. 
It will destroy the entire object, including all its derived class parts:
```c++
class TimeKeeper 
{
public:
    TimeKeeper();
    virtual ~TimeKeeper();
    // ...
};


{
    std::unique_ptr<TimeKeeper> getTimeKeeper();
    // ...
}
```
Base classes like `TimeKeeper` generally contain virtual functions other than the destructor, 
because the purpose of virtual functions is to allow customization of derived class implementations (see Item 34).
For example, `TimeKeeper` might have a virtual function, `getCurrentTime`, 
which would be implemented differently in the various derived classes. 
Any class with virtual functions should almost certainly have a virtual destructor.


If a class does not contain virtual functions, that often indicates it is **not** meant to be used as a base class. 
When a class is not intended to be a base class, making the destructor virtual is usually a bad idea.
Consider a class for representing 2D points:
```c++
class Point2i
{
public:
    Point2i(int xCoord, int yCoord);
    ~Point2i();

private:
    int x, y;
};
```
If an `int` occupies 32 bits, a `Point` object can typically fit into a 64-bit register. 
Furthermore, such a `Point` object can be passed as a 64-bit quantity to functions written in other languages, such as C or FORTRAN.
If `Point`’s destructor is made virtual, however, the situation changes.


The implementation of virtual functions requires that objects carry information 
that can be used at runtime to determine which virtual functions should be invoked on the object. 
This information typically takes the form of a pointer called a `vptr` (“virtual table pointer”). 
The `vptr` points to an array of function pointers called a `vtbl` (“virtual table”); 
each class with virtual functions has an associated `vtbl`. 
When a virtual function is invoked on an object, the actual function called is determined 
by following the object’s `vptr` to a `vtbl` and then looking up the appropriate function pointer in the `vtbl`.


The details of how virtual functions are implemented are unimportant.
What is important is that if the `Point2i` class contains a virtual function, 
objects of that type will increase in size. 
On a 32-bit architecture, they’ll go from 64 bits (for the two ints) to 96 bits (for the ints plus the `vptr`); 
on a 64-bit architecture, they may go from 64 to 128 bits, because pointers on such architectures are 64 bits in size. 
Addition of a `vptr` to `Point2i` will thus increase its size by 50–100%! 
No longer can Point objects fit in a 64-bit register. 
Furthermore, `Point2i` objects in C++ can no longer look like the same structure declared in another language such as C, 
because their foreign language counterparts will lack the `vptr`. 
As a result, it is no longer possible to pass `Point2i`s to and from functions written in other languages 
unless you explicitly compensate for the `vptr`, which is itself an implementation detail and hence unportable.


The bottom line is that gratuitously declaring all destructors virtual is just as wrong as never declaring them virtual. 
In fact, many people summarize the situation this way: 
declare a virtual destructor in a class if and only if that class contains at least one virtual function.


It is possible to get bitten by the non-virtual destructor problem even in the complete absence of virtual functions. 
For example, the standard `std::string` type contains no virtual functions, 
but misguided programmers sometimes use it as a base class anyway:
```c++
class SpecialString : public std::string 
{ 
    // bad idea! std::string has a non-virtual destructor
    // ... 
};
```
At first glance, this may look innocuous, but if anywhere in an application 
you somehow convert a pointer-to-SpecialString into a pointer-to-string and you then use `delete` on the string pointer, 
you are instantly transported to the realm of undefined behavior. 


The same analysis applies to any class lacking a virtual destructor, including all the STL container types 
(e.g., `std::vector`, `std::list`, `std::set`, `std::unordered_map`, etc.). 
If you’re ever tempted to inherit from a standard container or any other class with a non-virtual destructor, 
resist the temptation!


Occasionally it can be convenient to give a class a <u>_pure virtual destructor_</u>. 
Recall that pure virtual functions result in <u>_abstract classes_</u>, 
i.e., classes that can’t be instantiated (i.e., you can’t create objects of that type). 
Sometimes, however, you have a class that you’d like to be abstract, but you don’t have any pure virtual functions. 
What to do?
Well, because an abstract class is intended to be used as a base class,
and because a base class should have a virtual destructor, 
and because a pure virtual function yields an abstract class, 
the solution is simple: declare a pure virtual destructor in the class you want to be abstract.
This class has a pure virtual function, so it’s abstract, and it has a virtual destructor,
so you won’t have to worry about the destructor problem.
There is one twist, however: you must provide a definition for the pure virtual destructor:
```c++
// “Abstract w/o Virtuals”
class AWOV
{
public:
    virtual ~AWOV() = 0;  // declare pure virtual destructor
};

AWOV::~AWOV() = default;  // definition of pure virtual dtor
```
The way destructors work is that the most derived class’s destructor is called first,
then the destructor of each base class is called. 
Compilers will generate a call to `~AWOV` from its derived classes’ destructors,
so you have to be sure to provide a body for the function. 
If you don’t, the linker will complain.


The rule for giving base classes virtual destructors applies only to polymorphic base classes: 
to base classes designed to allow the manipulation of derived class types through base class interfaces.
`TimeKeeper` is a polymorphic base class, because we expect to be able to manipulate `AtomicClock` and `WaterClock` objects, 
even if we have only `TimeKeeper` pointers to them.


Not all base classes are designed to be used polymorphically. 
Neither the standard `std::string` type, for example, nor the STL container types are designed to be base classes at all, 
much less polymorphic ones. 
Some classes are designed to be used as base classes, yet are not designed to be used polymorphically. 
Such classes are not designed to allow the manipulation of derived class objects via base class interfaces.
As a result, they don’t need virtual destructors.






### 📌 Item 8: Prevent exceptions from leaving destructors

- Destructors should **never** ~~emit exceptions~~.
  If functions called in a destructor may throw,
  the destructor should catch any exceptions,
  then swallow them or terminate the program.
- If class clients need to be able to react to exceptions thrown during an operation,
  the class should provide a regular (i.e., non-destructor) function that performs the operation.


C++ doesn’t prohibit destructors from emitting exceptions, but it certainly discourages the practice with good reason.
```c++
class Widget
{
public:
    // ...
    
    ~Widget()
    {
        // ...
        // assume this might emit an exception
    }
};

void doSomething()
{
    std::vector<Widget> v;
    // ...
    // v is automatically destroyed here
} 
```
When the vector `v` is destroyed, it is responsible for destroying all the `Widget`s it contains.
Suppose `v` has ten `Widget`s in it, and during destruction of the first one, an exception is thrown.
The other nine `Widget`s still have to be destroyed (otherwise any resources they hold would be leaked),
so `v` should invoke their destructors.
But suppose that during those calls, a second `Widget` destructor throws an exception.
Now there are two simultaneously active exceptions, and that’s one too many for C++.
Depending on the precise conditions under which such pairs of simultaneously active exceptions arise,
program execution either terminates or yields undefined behavior.
In this example, it yields undefined behavior.
It would yield equally undefined behavior using any other standard library containers, or even an array.
Premature program termination or undefined behavior can result from destructors emitting exceptions even without using containers and arrays.

But what should you do if your destructor needs to perform an operation that may fail by throwing an exception?
For example, suppose you’re working with a class for database connections:
```c++
class DBConnection
{
public:
    // ...
    
    static DBConnection create();  // function to return DBConnection objects; 
                                   // params omitted for simplicity
    
    void close();                  // close connection; 
                                   // throw an exception if closing fails
}; 
```
To ensure that clients don’t forget to call close on `DBConnection` objects,
a reasonable idea would be to create a resource-managing class for `DBConnection` that calls close in its destructor.
```c++
// class to manage DBConnection objects
class DBConn
{
public:
    // ...
    
    ~DBConn()
    {
        // make sure database connections are always closed
        db.close();
    }

private:
    DBConnection db;
};
```
That allows clients to program like this:
```c++
// open a block
{
    DBConn dbc(DBConnection::create());  // create DBConnection object 
                                         // and turn it over to a DBConn object to manage
    // ...                               // use the DBConnection object via the DBConn interface
}
// at end of block, the DBConn object is destroyed, 
// thus automatically calling close on the DBConnection object
```
This is fine as long as the call to close succeeds, but if the call yields an exception,
`DBConn`’s destructor will propagate that exception, i.e., allow it to leave the destructor.
That’s a problem, because destructors that throw mean trouble.


There are two primary ways to avoid the trouble. `DBConn`’s destructor could:

- **Terminate the program** if close throws, typically by calling `std::abort` or `std::terminate`:
  ```c++
  DBConn::~DBConn()
  {
      try
      {
          db.close();
      }
      catch (...)
      {
          // make log entry that the call to close failed;
          std::terminate();
      }
  }
  ```
  This is a reasonable option if the program cannot continue to run after an error is encountered during destruction.
  It has the advantage that if allowing the exception to propagate from the destructor would lead to undefined behavior,
  this prevents that from happening.
  That is, calling `std::abort` or `std::terminate` may forestall undefined behavior.
- **Swallow the exception** arising from the call to close:
  ```c++
  DBConn::~DBConn()
  {
      try
      {
          db.close();
      }
      catch (...)
      {
          // make log entry that the call to close failed;
          // and do nothing. 
      }
  }
  ```
  In general, swallowing exceptions is a bad idea,
  because it suppresses the important information that something failed!
  Sometimes, however, swallowing exceptions is preferable
  to running the risk of premature program termination or undefined behavior.
  For this to be a viable option, the program must be able to reliably continue execution
  even after an error has been encountered and ignored.

Neither of these approaches is especially appealing.
The problem with both is that the program has no way to react to the condition
that led to close throwing an exception in the first place.


A better strategy is to design `DBConn`’s interface so that
its clients have an opportunity to react to problems that may arise.
For example, `DBConn` could offer a close function itself,
thus giving clients a chance to handle exceptions arising from that operation.
It could also keep track of whether its `DBConnection` had been closed,
closing it itself in the destructor if not.
That would prevent a connection from leaking.
If the call to close were to fail in the `DBConn` destructor,
however, we’d be back to terminating or swallowing:
```c++
class DBConn
{
public:
    // ...
    
    // new function for client use
    void close()
    {
    db.close();
    closed = true;
    }

    ~DBConn()
    {
        if (!closed)
        {
            try
            {
                // close the connection if the client didn’t
                db.close();
            }
            catch (...)
            {
                // if closing fails,
                // make log entry that call to close failed; 
                // note that and terminate or swallow
            }
        }
    }

private:
    DBConnection db;
    bool closed;
};
```
If an operation may fail by throwing an exception and there may be a need to handle that exception,
the exception has to come from some non-destructor function.
That’s because destructors that emit exceptions are dangerous,
always running the risk of premature program termination or undefined behavior.
In this example, telling clients to call close themselves doesn’t impose a burden on them;
it gives them an opportunity to deal with errors they would otherwise have no chance to react to.
If they don’t find that opportunity useful (perhaps because they believe that no error will really occur),
they can ignore it, relying on `DBConn`’s destructor to call close for them.
If an error occurs at that point, i.e., if close does throw,
they’re in no position to complain if `DBConn` swallows the exception or terminates the program.
After all, they had first crack at dealing with the problem, and they chose not to use it.






### 📌 Item 9: Never call virtual functions during construction or destruction

- Don't call virtual functions during construction or destruction,
  because such calls will never go to a more derived class
  than that of the currently executing constructor or destructor.


You **shouldn’t** ~~call virtual functions during construction or destruction~~,
because the calls won’t do what you think, and if they did, you’d still be unhappy.
If you’re a recovering Java or C# programmer, pay close attention to this Item.


Suppose you’ve got a class hierarchy for modeling stock transactions,
e.g., buy orders, sell orders, etc.
It’s important that such transactions be auditable,
so each time a transaction object is created,
an appropriate entry needs to be created in an audit log:
```c++
class Transaction
{
public:
    Transaction()
    {
        // ...
        // as final action, log this transaction
        logTransaction();
    }
    
    // make type-dependent log entry
    virtual void logTransaction() const = 0;
    
    // ...
};


class BuyTransaction : public Transaction
{
public:
    // how to log transactions of this type
    virtual void logTransaction() const; 
    
    // ...
};


class SellTransaction : public Transaction
{
public:
    // how to log transactions of this type
    virtual void logTransaction() const;  
    
    // ...
};
```
Consider what happens when this code is executed:
```c++
BuyTransaction b;
```
Clearly a `BuyTransaction` constructor will be called, but first, a `Transaction` constructor must be called; 
base class parts of derived class objects are constructed before derived class parts are.
The last line of the `Transaction` constructor calls the virtual function `logTransaction`, 
but this is where the surprise comes in. 
The version of `logTransaction` that’s called is the one in `Transaction`, **not** the one in `BuyTransaction`, 
even though the type of object being created is `BuyTransaction`. 
**During base class construction, virtual functions never go down into derived classes.** 
Instead, the object behaves as if it were of the base type.
Informally speaking, during base class construction, virtual functions aren’t.


There’s a good reason for this seemingly counterintuitive behavior.
Because base class constructors execute before derived class constructors,
derived class data members have not been initialized when base class constructors run. 
If virtual functions called during base class construction went down to derived classes, 
the derived class functions would almost certainly refer to local data members, 
but those data members would not yet have been initialized. 


Thus, C++ rules are that during base class construction of a derived class object, 
the type of the object is that of the base class. 
Not only do virtual functions resolve to the base class, 
but the parts of the language using runtime type information (e.g., `dynamic_cast` and `typeid`) 
treat the object as a base class type. 
In our example, while the `Transaction` constructor 
is running to initialize the base class part of a `BuyTransaction` object, 
the object is of type `Transaction`. 
That’s how every part of C++ will treat it, and the treatment makes sense: 
the `BuyTransaction`-specific parts of the object haven’t been initialized yet, 
so it’s safest to treat them as if they didn’t exist.
An object doesn’t become a derived class object until execution of a derived class constructor begins.


The same reasoning applies during destruction. 
Once a derived class destructor has run, the object’s derived class data members assume undefined values, 
so C++ treats them as if they no longer exist. 
Upon entry to the base class destructor, the object becomes a base class object, 
and all parts of C++ (virtual functions, dynamic_casts, etc.) treat it that way.


In the example code above, the `Transaction` constructor made a direct call to a virtual function, 
a clear and easy-to-see violation of this Item’s guidance. 
The violation is so easy to see, some compilers issue a warning about it. 
(Others don’t. See Item 53 for a discussion of warnings.) 
Even without such a warning, the problem would almost certainly become apparent before runtime, 
because the `logTransaction` function is pure virtual in `Transaction`. 
Unless it had been defined (unlikely, but possible — see Item 34), the program wouldn’t link: 
the linker would be unable to find the necessary implementation of `Transaction::logTransaction`.


It’s not always so easy to detect calls to virtual functions during construction or destruction. 
If Transaction had multiple constructors, each of which had to perform some of the same work, 
it would be good software engineering to avoid code replication by putting the common initialization code, 
including the call to `logTransaction`, into a private nonvirtual initialization function, say, `init`:
```c++
class Transaction
{
public:
    Transaction()
    {
        // call to non-virtual ...
        init();
    } 
    
    virtual void logTransaction() const = 0;

    // ...
    
private:
    void init()
    {
        // ... that calls a virtual!
        logTransaction(); 
    }
};
```
This code is conceptually the same as the earlier version, 
but it’s more insidious, because it will typically compile and link without complaint.
In this case, because `logTransaction` is pure virtual in `Transaction`, 
most runtime systems will abort the program when the pure virtual is called (typically issuing a message to that effect). 
However, if `logTransaction` were a “normal” virtual function (i.e., not pure virtual) with an implementation in `Transaction`, 
that version would be called, and the program would merrily trot along, 
leaving you to figure out why the wrong version of `logTransaction` was called when a derived class object was created. 
The only way to avoid this problem is to make sure that none of your constructors or destructors call virtual functions
on the object being created or destroyed and that all the functions they call obey the same constraint.


But how do you ensure that the proper version of `logTransaction` is called 
each time an object in the `Transaction` hierarchy is created?
Clearly, calling a virtual function on the object from the `Transaction` constructor(s) is the wrong way to do it.


There are different ways to approach this problem. 
One is to turn `logTransaction` into a non-virtual function in `Transaction`, 
then require that derived class constructors pass the necessary log information to the `Transaction` constructor. 
That function can then safely call the non-virtual `logTransaction`:
```c++
class Transaction
{
public:
    explicit Transaction(const std::string & logInfo)
    {
        // ...
        // now a non-virtual call
        logTransaction(logInfo);
    }

    // now a non-virtual function
    void logTransaction(const std::string & logInfo) const; 
    
    virtual ~Transsaction() = 0;
    
    // ...
};


class BuyTransaction : public Transaction
{
public:
    // pass log info to base class constructor
    BuyTransaction(parameters) : Transaction(createLogString(parameters)) 
    {
        // ...
    }
    
    // ...
private:
    static std::string createLogString(parameters);
};
```
In other words, since you can’t use virtual functions to call down from base classes during construction, 
you can compensate by having derived classes pass necessary construction information up to base class constructors instead.


In this example, note the use of the (private) static function `createLogString` in `BuyTransaction`. 
Using a helper function to create a value to pass to a base class constructor is often more convenient (and more readable) 
than going through contortions in the member initialization list to give the base class what it needs. 
By making the function static, there’s no danger of accidentally referring to 
the nascent `BuyTransaction` object’s as-yet-uninitialized data members. 
That’s important, because the fact that those data members will be in an undefined state is why calling virtual functions 
during base class construction and destruction doesn’t go down into derived classes in the first place.






### 📌 Item 10: Have assignment operators return a reference to `*this`

- Have assignment operators return a reference to `*this`.


Assignments can be chained together:
```c++
int x, y, z;
x = y = z = 15;  // chain of assignments
```
Assignments are right-associative, so the above assignment chain is parsed like this:
```c++
x = (y = (z = 15));
```
Here, 15 is assigned to `z`, 
then the result of that assignment (the updated `z`) is assigned to `y`, 
then the result of that assignment (the updated `y`) is assigned to `x`.


The way this is implemented is that assignment returns a reference to its left-hand argument, 
and that’s the convention you should follow when you implement assignment operators for your classes:
```c++
class Widget
{
public:
    // ...

    // return type is a reference to the current class
    Widget & operator=(const Widget & rhs) 
    {
        // ...
        return *this;  // return the left-hand object
    }

    // ...
};
```
This convention applies to all assignment operators, not just the standard form shown above. Hence:
```c++
class Widget
{
public:
    // ...

    // the convention applies to +=, -=, *=, etc.
    Widget & operator+=(const Widget & rhs) 
    { 
        // ...
        return *this;
    }

    // it applies even if the operator’s parameter type is unconventional
    Widget & operator=(int rhs) 
    {
        // ... 
        return *this;
    }

    // ...
};
```
This is only a convention; code that doesn’t follow it will compile. 
However, the convention is followed by all the built-in types as well as by all the types in the standard library
(e.g., `std::string`, `std::vector`, `std::complex`, `std::shared_ptr`, etc.). 
Unless you have a good reason for doing things differently, don’t.






### 📌 Item 11: Handle assignment to self in `operator=`

- Make sure `operator=` is well-behaved when an object is assigned to itself. Techniques include:
    - comparing addresses of source and target objects; 
    - careful statement ordering;
    - copy-and-`swap`.
- Make sure that any function operating on more than one object behaves correctly
  if two or more of the objects are the same.


An <u>_assignment to self_</u> occurs when an object is assigned to itself:
```c++
class Widget { ... };
Widget w;
// ...
w = w;  // assignment to self
```
This looks silly, but it’s legal, so rest assured that clients will do it.
Besides, assignments aren’t always so recognizable: 
```c++
a[i] = a[j];  // potential assignment to self if i == j
*px = *py;    // potential assignment to self if px == py
```
These less obvious assignments to self are the result of <u>_aliasing_</u>:
having more than one way to refer to an object. 
In general, code that operates on references or pointers to multiple objects of the same type
needs to consider that the objects might be the same. 
Considering polymorphism, 
the two objects need not even be declared to be of the same type if they’re from the same hierarchy: 
```c++
class Base { ... };

class Derived : public Base { ... };

// rb and *pd might actually be the same object
void doSomething(const Base & rb, Derived * pd); 
```
If you follow the advice of Items 13 and 14, you’ll always use objects to manage resources, 
and you’ll make sure that the resource-managing objects behave well when copied. 
When that’s the case, your assignment operators will probably be self-assignment-safe 
without your having to think about it. 
If you try to manage resources yourself, however 
(which you’d certainly have to do if you were writing a resource-managing class), 
you can fall into the trap of accidentally releasing a resource before you’re done using it. 
For example, suppose you create a class that holds a raw pointer to a dynamically allocated bitmap:
```c++
class Bitmap { ... };

class Widget 
{
    // ...
    
private:
    Bitmap * pb;  // ptr to a heap-allocated object
};
```
Here’s an implementation of `Widget::operator=` that looks reasonable on the surface 
but is unsafe in the presence of assignment to self. 
(It’s also not exception-safe, but we’ll deal with that in a moment.)
```c++
Widget & Widget::operator=(const Widget & rhs)
{
    delete pb;                 // stop using current bitmap
    pb = new Bitmap(*rhs.pb);  // start using a copy of rhs’s bitmap
    return *this;
}
```
The self-assignment problem here is that inside `Widget::operator=`, 
`*this` (the target of the assignment) and `rhs` could be the same object. 
When they are, the `delete` not only destroys the bitmap for the current object, 
it destroys the bitmap for `rhs`, too. 
At the end of the function, the `Widget` finds itself holding a dangling pointer. 


The traditional way to prevent this error is to check for assignment to self 
via an identity test at the top of `operator=`:
```c++
Widget & Widget::operator=(const Widget & rhs)
{
    Bitmap * pOrig = pb;       // remember original pb
    pb = new Bitmap(*rhs.pb);  // point pb to a copy of rhs’s bitmap
    delete pOrig;              // delete the original pb
    return *this;
}
```
Now, if `new Bitmap` throws an exception, `pb` (and the `Widget` it’s inside of) remains unchanged. 
Even without the identity test, this code handles assignment to self, because we make a copy of the original bitmap,
point to the copy we made, then delete the original bitmap. 
It may not be the most efficient way to handle self-assignment, but it does work.


If you’re concerned about efficiency, you could put the identity test back at the top of the function. 
Before doing that, however, ask yourself how often you expect self-assignments to occur, 
because the test isn’t free. 
It makes the code (both source and object) a bit bigger, and it introduces a branch into the flow of control, 
both of which can decrease runtime speed. 
The effectiveness of instruction prefetching, caching, and pipelining can be reduced, for example.


An alternative to manually ordering statements in `operator=` to make sure the implementation is 
both exception-safe and self-assignment-safe is to use the technique known as <u>_copy and swap_</u>. 
This technique is closely associated with exception safety, so it’s described in Item 29.
However, it’s a common enough way to write `operator=` that it’s worth seeing what such an implementation often looks like:
```c++
class Widget
{
    // ...

    // exchange *this's and rhs's data
    void swap(Widget & rhs);
    
    // ...
};

Widget & Widget::operator=(const Widget & rhs)
{
    using std::swap;
    Widget temp(rhs);  // make a copy of rhs’s data
    swap(temp);        // swap *this’s data with the copy’s
    return *this;
}
```
A variation on this theme takes advantage of the facts that 

1. a class’s copy assignment operator may be declared to take its argument by value;
2. passing something by value makes a copy of it (see Item 20):

```c++
// rhs is a copy of the object passed in; 
// note pass-by-value
Widget & Widget::operator=(Widget rhs) 
{ 
    using std::swap;
    swap(rhs);        // swap *this’s data with the copy’s
    return *this;
}
```
Personally, I worry that this approach sacrifices clarity at the altar of cleverness, 
but by moving the copying operation from the body of the function to construction of the parameter, 
it’s a fact that compilers can sometimes generate more efficient code.






### 📌 Item 12: Copy all parts of an object

- Copying functions should be sure to copy all of an object's data members 
  and all of its base class parts (by invoking the base class's copying functions).
- Don't try to implement one of the copying functions in terms of the other.
  Instead, put common functionality in a third function that both call.



In well-designed object-oriented systems that encapsulate the internal parts of objects, 
only two functions copy objects: the aptly named <u>_copy constructor_</u> and <u>_copy assignment operator_</u>. 
We’ll call these the <u>_copying functions_</u>. 
Item 5 observes that compilers will generate the copying functions if needed, 
and it explains that the compiler-generated versions do precisely what you’d expect: 
they copy all the data of the object being copied.


When you declare your own copying functions, 
you are indicating to compilers that there is something about the default implementations you don’t like. 
Compilers seem to take offense at this, and they retaliate in a curious fashion: 
they don’t tell you when your implementations are almost certainly wrong.


Consider a class representing customers, 
where the copying functions have been manually written so that calls to them are logged:
```c++
class Customer
{
public:
    // ...

    Customer(const Customer & rhs);

    Customer & operator=(const Customer & rhs);

    // ...
private:
    std::string name;
};

Customer::Customer(const Customer & rhs) : name(rhs.name)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

Customer & Customer::operator=(const Customer & rhs)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    name = rhs.name;
    return *this;
}
```
Everything here looks fine, and in fact everything is fine, until another data member is added to `Customer`:
```c++
class Date
{
    // ...
};

class Customer
{
public:
    // as before
    // ...
    
private:
    std::string name;
    Date lastTransaction;
};
```
At this point, the existing copying functions are performing a <u>_partial copy_</u>: 
they’re copying the customer’s `name`, but not its `lastTransaction`.
Yet most compilers say nothing about this, not even at maximal warning level. 
You reject the copying functions they’d write, so they don’t tell you if your code is incomplete. 
The conclusion is obvious: 
if you add a data member to a class, you need to make sure that you update the copying functions, too. 
(You’ll also need to update all the constructors as well as any non-standard forms of `operator=` in the class.
If you forget, compilers are unlikely to remind you.)


One of the most insidious ways this issue can arise is through inheritance. Consider:
```c++
class PriorityCustomer : public Customer
{
public:
    // ...

    PriorityCustomer(const PriorityCustomer & rhs);

    PriorityCustomer & operator=(const PriorityCustomer & rhs);

    // ...
private:
    int priority;
};

PriorityCustomer::PriorityCustomer(const PriorityCustomer & rhs) : priority(rhs.priority)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

PriorityCustomer & PriorityCustomer::operator=(const PriorityCustomer & rhs)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    priority = rhs.priority;
    return *this;
}
```
`PriorityCustomer`’s copying functions look like they’re copying everything in `PriorityCustomer`, but look again. 
Yes, they copy the data member that `PriorityCustomer` declares, 
but every `PriorityCustomer` also contains a copy of the data members it inherits from `Customer`, 
and those data members are not being copied at all! 
`PriorityCustomer`’s copy constructor specifies no arguments to be passed to its base class constructor
(i.e., it makes no mention of `Customer` on its member initialization list), 
so the `Customer` part of the `PriorityCustomer` object will be initialized 
by the default `Customer` constructor taking no arguments. (Assuming it has one. If not, the code won’t compile.)
That constructor will perform a <u>_default initialization_</u> for `name` and `lastTransaction`.


The situation is only slightly different for `PriorityCustomer`’s copy assignment operator. 
It makes no attempt to modify its base class data members in any way, so they’ll remain unchanged.


Any time you take it upon yourself to write copying functions for a derived class, 
you must take care to also copy the base class parts.
Those parts are typically private, of course (see Item 22), so you can’t access them directly. 
Instead, derived class copying functions must invoke their corresponding base class functions:
```c++
PriorityCustomer::PriorityCustomer(const PriorityCustomer & rhs)
        : Customer(rhs),  // invoke base class copy constructor
          priority(rhs.priority)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

PriorityCustomer & PriorityCustomer::operator=(const PriorityCustomer & rhs)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
    Customer::operator=(rhs);  // assign base class parts
    priority = rhs.priority;
    return *this;
}
```
The meaning of “copy all parts” in this Item’s title should now be clear.
When you’re writing a copying function, be sure to 

1. copy all local data members;
2. invoke the appropriate copying function in all base classes. 

In practice, the two copying functions will often have similar bodies, 
and this may tempt you to try to avoid code duplication by having one function call the other. 
Your desire to avoid code duplication is laudable,
but having one copying function call the other is the wrong way to achieve it.


It makes no sense to ~~have the copy assignment operator call the copy constructor~~, 
because you’d be trying to construct an object that already exists. 
This is so nonsensical, there’s not even a syntax for it.


~~Having the copy constructor call the copy assignment operator~~ is equally nonsensical. 
A constructor initializes new objects, 
but an assignment operator applies only to objects that have already been initialized. 
Performing an assignment on an object under construction would mean 
doing something to a not-yet-initialized object that makes sense only for an initialized object.


Instead, if you find that your copy constructor and copy assignment operator have similar code bodies, 
eliminate the duplication by creating a third member function that both call. 
Such a function is typically private and is often named init. 
This strategy is a safe, proven way to eliminate code duplication in copy constructors and copy assignment operators.






### 🎯 Chapter 3. Resource Management

A resource is something that, once you’re done using it, you need to return to the system. 
If you don’t, bad things happen. 
In C++ programs, the most commonly used resource is dynamically allocated memory 
(if you allocate memory and never deallocate it, you’ve got a memory leak), 
but memory is only one of many resources you must manage. 
Other common resources include file descriptors, mutex locks, 
fonts and brushes in graphical user interfaces (GUIs), database connections, and network sockets. 
Regardless of the resource, it’s important that it be released when you’re finished with it.


Trying to ensure this by hand is difficult under any conditions, 
but when you consider exceptions, functions with multiple return paths,
and maintenance programmers modifying software without fully comprehending the impact of their changes, 
it becomes clear that ad hoc ways of dealing with resource management **aren’t** sufficient.


This chapter begins with a straightforward object-based approach 
to resource management built on C++’s support for constructors, destructors, and copying operations. 
Experience has shown that disciplined adherence to this approach can all but eliminate resource management problems.
The chapter then moves on to Items dedicated specifically to memory management. 
These latter Items complement the more general Items that come earlier, 
because objects that manage memory have to know how to do it properly.






### 📌 Item 13: Use objects to manage resources

- To prevent resource leaks, use RAII objects
  that acquire resources in their constructors and release them in their destructors.
- ~~Two commonly useful RAII classes are `tr1::shared_ptr` and `std::auto_ptr`.
  `tr1::shared_ptr` is usually the better choice, because its behavior when copied is intuitive.
  Copying an `std::auto_ptr` sets it to null.~~ 
- Refer to _Effective Modern C++_ Chapter 4 for details.






### 📌 Item 14: Think carefully about copying behavior in resource-managing classes

- Copying an RAII object entails copying the resource it manages,
  so the copying behavior of the resource determines the copying behavior of the RAII object.
- Common RAII class copying behaviors are disallowing copying and performing reference counting,
  but other behaviors are possible.


Not all resources are heap-based, and for such resources, 
smart pointers like `std::unique_ptr` and `std::shared_ptr` are generally inappropriate as resource handlers. 
That being the case, you’re likely to find yourself needing to create your own resource-managing classes.


For example, for RAII objects for `std::mutex`s.
[`std::lock_guard`](https://en.cppreference.com/w/cpp/thread/lock_guard)
and [`std::scoped_lock`](https://en.cppreference.com/w/cpp/thread/scoped_lock)
are already a proper choices for mutexes. 
To make sure that you never forget to unlock a `std::mutex` you’ve locked, you’d like to create a class to manage locks.
The basic structure of such a class is dictated by the RAII principle that resources are acquired 
during construction and released during destruction:
```c++
/// <std_mutex.h>
/// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0

namespace std
{

/// Do not acquire ownership of the mutex.
struct defer_lock_t { explicit defer_lock_t() = default; };

/// Try to acquire ownership of the mutex without blocking.
struct try_to_lock_t { explicit try_to_lock_t() = default; };

/// Assume the calling thread has already obtained mutex ownership
/// and manage it.
struct adopt_lock_t { explicit adopt_lock_t() = default; };

/// Tag used to prevent a scoped lock from acquiring ownership of a mutex.
_GLIBCXX17_INLINE constexpr defer_lock_t defer_lock { };

/// Tag used to prevent a scoped lock from blocking if a mutex is locked.
_GLIBCXX17_INLINE constexpr try_to_lock_t try_to_lock { };

/// Tag used to make a scoped lock take ownership of a locked mutex.
_GLIBCXX17_INLINE constexpr adopt_lock_t adopt_lock { };

/** @brief A simple scoped lock type.
 *
 * A lock_guard controls mutex ownership within a scope, 
 * releasing ownership in the destructor.
 */
template <typename _Mutex>
class lock_guard
{
public:
    typedef _Mutex mutex_type;
    
    explicit lock_guard(mutex_type & __m) : _M_device(__m)
    {
        _M_device.lock();
    }
    
    lock_guard(mutex_type & __m, adopt_lock_t) noexcept : _M_device(__m)
    {
        // calling thread owns mutex
    } 
    
    ~lock_guard()
    {
        _M_device.unlock();
    }
    
    lock_guard(const lock_guard &) = delete;
    
    lock_guard & operator=(const lock_guard &) = delete;

private:
    mutex_type & _M_device;
};

}  // namespace std
```
Clients use `LockGuard` in the conventional RAII fashion:
```c++
std::mutex mutex;  // define the mutex you need to use
// ...

// create block to define critical section
{ 
    std::lock_guard grd(mutex);  // lock the mutex
    // ...                       // perform critical section operations
} 
// automatically unlock mutex at end of block
```
This is fine, but what should happen if a lock guard object is copied?
```c++
Lock ml1(mutex);  // lock mutex
Lock ml2(ml1);    // copy ml1 to ml2; what should happen here?
```
This is a specific example of a more general question, one that every RAII class author must confront: 
what should happen when an RAII object is copied? 
Most of the time, you’ll want to choose one of the following possibilities:

- **Prohibit copying**. 
  In many cases, it makes no sense to allow RAII objects to be copied. 
  This is likely to be true for a class like `std::lock_guard`, 
  because it rarely makes sense to have “copies” of synchronization primitives. 
  When copying makes no sense for an RAII class, you should prohibit it
  (via `delete`ing its copy constructors and copy assignment operators). 
- **Reference-count the underlying resource**. 
  Sometimes it’s desirable to hold on to a resource until the last object using it has been destroyed. 
  When that’s the case, copying an RAII object should increment the count of the number of objects referring to the resource.
  This is the meaning of “copy” used by `std::shared_ptr`. 
  Often, RAII classes can implement reference-counting copying behavior by containing a `std::shared_ptr` data member. 
  For example, if our lock guard wanted to employ reference counting,
  it could change the type of mutex from `std::mutex &` to `std::shared_ptr<std::mutex>` and provide a custom deleter.
  ```c++
  std::mutex mutex;
  std::shared_ptr<std::mutex> mutexPtr(&mutex, [](std::mutex * p) { p->unlock(); });
  ```
  In this case, notice how the Lock class no longer declares a destructor.
  That’s because there’s no need to. 
  Item 5 explains that a class’s destructor (regardless of whether it is compiler-generated or user-defined) 
  automatically invokes the destructors of the class’s non-static data members. 
  In this example, that’s `mutexPtr`.
  But `mutexPtr`’s destructor will automatically call the `std::shared_ptr`’s deleter (`std::mutex::unlock`)
  when the mutex’s reference count goes to zero.
- **Copy the underlying resource**. 
  Sometimes you can have as many copies of a resource as you like,
  and the only reason you need a resource-managing class is to make sure that 
  each copy is released when you’re done with it. 
  In that case, copying the resource-managing object should also copy the resource it wraps. 
  That is, copying a resource-managing object performs a “deep copy.” 
  Some implementations of the standard `std::string` type consist of pointers to heap memory, 
  where the characters making up the `std::string` are stored. 
  Objects of such `std::string`s contain a pointer to the heap memory. 
  When a string object is copied, a copy is made of both the pointer and the memory it points to. 
  Such strings exhibit deep copying.
- **Transfer ownership of the underlying resource**. 
  On rare occasion, you may wish to make sure that only one RAII object refers to a raw resource 
  and that when the RAII object is copied, 
  ownership of the resource is transferred from the copied object to the copying object. 
  This is the meaning of “copy” used by `std::unique_ptr`s.


The copying functions (copy constructor and copy assignment operator) may be generated by compilers, 
so unless the compiler-generated versions will do what you want,
you’ll need to write them yourself. 






### 📌 Item 15: Provide access to raw resources in resource-managing classes

- APIs often require access to raw resources,
  so each RAII class should offer a way to get at the resource it manages.
- Access may be via explicit conversion or implicit conversion.
  In general, explicit conversion is safer,
  but implicit conversion is more convenient for clients.






### 📌 Item 16: Use the same form in corresponding uses of `new` and `delete`

- If you use `[]` in a `new` expression, you must use `[]` in the corresponding `delete` expression.
  If you don't use `[]` in a `new` expression, you mustn't use `[]` in the corresponding `delete` expression.
- Do not use `typedef` on array types (doesn't know what form of `delete` to use)


Considering the following code: 
```c++
std::string * stringArray = new std::string[100];
// ...
delete stringArray;
```
The `new` is matched with a `delete`, but the program’s behavior is undefined.
At the very least, 99 of the 100 string objects pointed to by `stringArray` 
are **unlikely** to ~~be properly destroyed~~, 
because their destructors will probably **never** be called.


When you employ a <u>_`new` expression_</u> (i.e., dynamic creation of an object via a use of `new`), two things happen: 
memory is allocated (via a function named `operator new`; see Items 49 and 51); 
then one or more constructors are called for that memory.


When you employ a <u>_`delete` expression_</u> (i.e., use `delete`), two other things happen: 
one or more destructors are called for the memory;
then the memory is deallocated (via a function named `operator delete`; see Item 51). 


The big question for `delete` is this: how many objects reside in the memory being deleted? 
The answer to that determines how many destructors must be called.


When you use `delete` on a pointer, 
the only way for `delete` to know whether the array size information is there is for you to tell it. 
If you use brackets in your use of `delete`, `delete` assumes an array is pointed to. 
Otherwise, it assumes that a single object is pointed to:
```c++
std::string * stringPtr1 = new std::string;
std::string * stringPtr2 = new std::string[100];
// ...
delete stringPtr1;     // delete an object
delete [] stringPtr2;  // delete an array of objects
```
Using the “`delete []`” form on `stringPtr1` is undefined behavior that is unlikely to be pretty.
Not using the “`delete []`” form on `stringPtr2` is also undefined behavior, 
but you can see how it would lead to too few destructors being called. 
Furthermore, it’s undefined (and sometimes harmful) for built-in types like `int`s, too, 
even though such types lack destructors.


If you use `[]` in a `new` expression, you must use `[]` in the corresponding `delete` expression.
If you don't use `[]` in a `new` expression, you mustn't use `[]` in the corresponding `delete` expression.


This is a particularly important rule to bear in mind 
when you are writing a class containing a pointer to dynamically allocated memory
and also offering multiple constructors, 
because then you must be careful to use the <u>_same form_</u> of `new` in all the constructors 
to initialize the pointer member. 
If you don’t, how will you know what form of `delete` to use in your destructor?


This rule is also noteworthy for the `typedef`-inclined, 
because it means that a `typedef`’s author must document 
which form of `delete` should be employed when `new` is used to conjure up objects of the `typedef` type.
For example, consider this `typedef`:
```c++
typedef std::string AddressLines[4];   // a person’s address has 4 lines, 
                                       // each of which is a string
```
Because `AddressLines` is an array, this use of `new`,
```c++
std::string * pal = new AddressLines;  // note that “new AddressLines” returns a string *, 
                                       // just like “new string[4]” would
```
must be matched with the array form of `delete`:
```c++
delete pal;                            // undefined!
delete [] pal;                         // fine
```
To avoid such confusion, abstain from `typedef`s for array types. 
That’s easy, because STL library `std::string` and `std::array`, as well as alias (`using` expression). 
and those templates reduce the need for dynamically allocated arrays to nearly zero.
Here, for example, `AddressLines` could be defined to be a `std::array` of `std::string`s, 
i.e., the type `using AddressLines = std::array<std::string, 4>`. 






### 📌 Item 17: Store `new`ed objects in smart pointers in standalone statements

- Store `new`ed objects in smart pointers in standalone statements.
  Failure to do this can lead to subtle resource leaks when exceptions are thrown.
- Refer to _Effective Modern C++_ Item 21 for details.






### 🎯 Chapter 4. Designs and Declarations

### 📌 Item 18: Make interfaces easy to use correctly and hard to use incorrectly

- Good interfaces are easy to use correctly and hard to use incorrectly.
  You should strive for these characteristics in all your interfaces.
- Ways to facilitate correct use include
  consistency in interfaces
  and behavioral compatibility with built-in types.
- Ways to prevent errors include creating new types,
  restricting operations on types,
  constraining object values,
  and eliminating client resource management responsibilities.
- Smart pointers supports custom deleters.
  This prevents the cross-DLL problem,
  can be used to automatically unlock mutexes (See Item 14), etc.


Developing interfaces that are easy to use correctly and hard to use incorrectly 
requires that you consider the kinds of mistakes that clients might make. 
For example, suppose you’re designing the constructor for a class representing dates in time:
```c++
class Date 
{
public:
    Date(int month, int day, int year);
    // ...
};
```
At first glance, this interface may seem reasonable, but there are at least two errors that clients might easily make.
First, they might pass parameters in the wrong order:
```c++
Date d(30, 3, 1995);  // incorrect order of arguments
Date d(3, 40, 1995);  // invalid range of arguments
```
Many client errors can be prevented by the introduction of new types.
Indeed, the type system is your primary ally in preventing undesirable code from compiling. 
In this case, we can introduce simple wrapper types to distinguish days, months, and years, 
then use these types in the `Date` constructor:
```c++
class Date
{
public:
    Date(const std::chrono::month & month, const Day & day, const Year & year);
    // ...
};

Date d(30, 3, 1995);                    // error! wrong types
Date d(Day(30), Month(3), Year(1995));  // error! wrong types
Date d(Month(3), Day(30), Year(1995));  // okay, types are correct
```
Making `Day`, `Month`, and `Year` full-fledged `class`es with encapsulated data 
would be better than the simple use of `struct`s above (see Item 22), 
but even `struct`s suffice to demonstrate that the judicious introduction of new types 
can work wonders for the prevention of interface usage errors.


Once the right types are in place, it can sometimes be reasonable to restrict the values of those types. 
For example, there are only 12 valid month values, so the `Month` type should reflect that. 
One way to do this would be to use an `enum` to represent the month, 
but `enum`s are not as type-safe as we might like. 
For example, `enum`s can be used like `int`s (see Item 2). 
(Of course, scoped `enum` is a prefect solution to this problem. )
A safer solution is to predefine the set of all valid `Month`s:
```c++
class Month
{
public:
    // functions returning all valid Month values; 
    // see below forwhy these are functions, not objects
    static Month Jan()
    {
        return Month(1);
    } 
    
    static Month Feb()
    {
        return Month(2);
    }
    
    // ...
    
    static Month Dec()
    {
        return Month(12);
    }
    
    // ...                  // other member functions
    
private:
    explicit Month(int m);  // prevent creation of new Month values
    
    // ...                  // month-specific data
};

Date d(Month::Mar(), Day(30), Year(1995));
```

If the idea of using functions instead of objects to represent specific months strikes you as odd, 
it may be because you have forgotten that reliable initialization of non-local static objects can be problematic.


Another way to prevent likely client errors is to restrict what can be done with a type. 
A common way to impose restrictions is to add `const`.
For example, Item 3 explains how `const`-qualifying the return type from `operator *` 
can prevent clients from making this error for userdefined types:
```c++
if (a * b = c)  // oops, meant to do a comparison!
{
    // ...  
}
```
In fact, this is just a manifestation of another general guideline 
for making types easy to use correctly and hard to use incorrectly: 
**unless there’s a good reason not to, have your types behave consistently with the built-in types**. 
Clients already know how types like `int` behave, 
so you should strive to have your types behave the same way whenever reasonable. 
For example, assignment to `a * b` isn’t legal if `a` and `b` are `int`s, 
so unless there’s a good reason to diverge from this behavior,
it should be illegal for your types, too. 
When in doubt, do as the `int`s do.


The real reason for avoiding gratuitous incompatibilities with the built-in types 
is to offer interfaces that behave consistently.
Few characteristics lead to interfaces that are easy to use correctly as much as consistency, 
and few characteristics lead to aggravating interfaces as much as inconsistency. 
The interfaces to STL containers are largely (though not perfectly) consistent, 
and this helps make them fairly easy to use. 
For example, every STL container has a member function named `size` that tells how many objects are in the container.
Contrast this with Java, 
where you use the `length` <u>_property_</u> for arrays, 
the `length` <u>_method_</u> for `String`s, 
and the `size` <u>_method_</u> for `List`s; 
and with .NET, 
where `Array`s have a property named `Length`, 
while `ArrayList`s have a property named `Count`. 
Some developers think that integrated development environments (IDEs) render such inconsistencies unimportant,
but they are mistaken. 
Inconsistency imposes mental friction into a developer’s work that no IDE can fully remove.
Any interface that requires that clients remember to do something is prone to incorrect use, 
because clients can forget to do it. 
For example, Item 13 introduces a factory function that 
returns pointers to dynamically allocated objects in an `Investment` hierarchy:
```c++
Investment * createInvestment();
```
To avoid resource leaks, the pointers returned from `createInvestment` must eventually be deleted, 
but that creates an opportunity for at least two types of client errors: 
failure to delete a pointer, and deletion of the same pointer more than once.


Item 13 shows how clients can store `createInvestment`’s return value in a smart pointer 
like `std::unique_ptr` or `std::shared_ptr`, 
thus turning over to the smart pointer the responsibility for using `delete`. 
But what if clients forget to use the smart pointer? 
In many cases, a better interface decision would be to preempt the problem by having the factory function
return a smart pointer in the first place:
```c++
std::shared_ptr<Investment> createInvestment();
```
This essentially forces clients to store the return value in a `std::shared_ptr`, 
all but eliminating the possibility of forgetting to 
`delete` the underlying `Investment` object when it’s no longer being used.


In fact, returning a `std::shared_ptr` makes it possible 
for an interface designer to prevent a host of other client errors regarding resource release, 
because smart pointers allow a custom deleter to be bounded during creation. 


Suppose clients who get an `Investment *` pointer from `createInvestment`
are expected to pass that pointer to a function called `getRidOfInvestment`
instead of using `delete` on it. 
Such an interface would open the door to a new kind of client error, 
one where clients use the wrong resource-destruction mechanism (i.e., `delete` instead of `getRidOfInvestment`).
The implementer of `createInvestment` can forestall such problems by returning a `std::shared_ptr` 
with `getRidOfInvestment` bound to it as its deleter: 
```c++
std::shared_ptr<Investment> createInvestment()
{
    return std::shared_ptr<new Investment(), [](Investment * p) { getRidOfInvestment(p); }>;
}
```
An especially nice feature of `std::shared_ptr` is that it automatically uses its per-pointer deleter 
to eliminate another potential client error, the “cross-DLL problem”. 
This problem crops up when an object is created using `new` in one dynamically linked library (DLL) 
but is deleted in a different DLL. 
On many platforms, such cross-DLL `new` / `delete` pairs lead to runtime errors. 
`std::shared_ptr` avoids the problem, because its default deleter uses `delete` 
from the same DLL where the `std::shared_ptr` is created. 
This means, for example, that if `Stock` is a class derived from `Investment` 
and `createInvestment` is implemented like this,
```c++
std::shared_ptr<Investment> createInvestment()
{
    return std::shared_ptr<Investment>(new Stock);
}
```
the returned `std::shared_ptr` can be passed among DLLs without concern for the cross-DLL problem. 
The `std::shared_ptr`s pointing to the `Stock` keep track of which DLL’s `delete` should be used 
when the reference count for the `Stock` becomes zero.






### 📌 Item 19: Treat class design as type design

- Class design is type design. Every class requires that you confront the following questions:
    - **How should objects of your new type be created and destroyed?**
      How this is done influences the design of your class’s constructors and destructor 
      (and remember the rule of three / five when defining copy control members), 
      as well as its memory allocation and deallocation functions 
      (`operator new`, `operator new[]`, `operator delete`, and `operator delete[]`) if you write them.
    - **How should object initialization differ from object assignment?**
      The answer to this question determines the behavior of and the differences 
      between your constructors and your assignment operators. 
      It’s important not to confuse initialization with assignment, 
      because they correspond to different function calls.
    - **What does it mean for objects of your new type to be passed by value?** 
      Remember, the copy constructor defines how pass-by-value is implemented for a type.
    - **What are the restrictions on legal values for your new type?**
      Usually, only some combinations of values for a class’s data members are valid. 
      Those combinations determine the invariants your class will have to maintain. 
      The invariants determine the error checking you’ll have to do inside your member functions, 
      especially your constructors, assignment operators, and “setter” functions. 
      It may also affect the exceptions your functions throw and, on the off chance you use them, 
      your functions’ exception specifications.
    - **Does your new type fit into an inheritance graph?** 
      If you inherit from existing classes, 
      you are constrained by the design of those classes, 
      particularly by whether their functions are `virtual` or non-`virtual`. 
      If you wish to allow other classes to inherit from your class, 
      that affects whether the functions you declare are `virtual`, especially your destructor.
    - **What kind of type conversions are allowed for your new type?**
      Your type exists in a sea of other types, so should there be conversions between your type and other types? 
      If you wish to allow objects of type `T1` to be implicitly converted into objects of type `T2`, 
      you will want to write either a type conversion function in `class T1` (e.g., `operator T2`) 
      or a non-`explicit` constructor in `class T2` that can be called with a single argument. 
      If you wish to allow explicit conversions only, 
      you’ll want to write functions to perform the conversions, 
      but you’ll need to avoid making them type conversion operators or non-`explicit` constructors 
      that can be called with one argument. 
    - **What operators and functions make sense for the new type?**
      The answer to this question determines which functions you’ll declare for your class.
      Some functions will be member functions, but some will not.
    - **What standard functions should be `delete`d?**
    - **Who should have access to the members of your new type?** 
      This question helps you determine which members are `public`, 
      which are `protected`, and which are `private`. 
      It also helps you determine which classes and/or functions should be `friend`s, 
      as well as whether it makes sense to nest one class inside another.
    - **What is the “undeclared interface” of your new type?** 
      What kind of guarantees does it offer with respect to performance,
      exception safety, and resource usage (e.g., locks and dynamic memory)? 
      The guarantees you offer in these areas will impose constraints on your class implementation.
    - **How general is your new type?** 
      Perhaps you’re not really defining a new type. 
      Perhaps you’re defining a whole <u>_family_</u> of types. 
      If so, you don’t want to define a new class, 
      you want to define a new <u>_class template_</u>.
    - **Is a new type really what you need?** 
      If you’re defining a new derived class only so you can add functionality to an existing class, 
      perhaps you’d better achieve your goals by simply defining one or more non-member functions or templates.






### 📌 Item 20: Prefer pass-by-reference-to-`const` to pass-by-value

- Prefer pass-by-reference-to-`const` over pass-by-value. 
  It's typically more efficient and it avoids the slicing problem. 
- The rule **doesn't** apply to built-in types, STL iterator, and function object types. 
  For them, pass-by-value is usually appropriate. 


By default, C++ passes objects to and from functions by value (as from C). 
Unless otherwise specified, function parameters are initialized with <u>_copies_</u> of the actual arguments, 
and function callers get back a <u>_copy_</u> of the value returned by the function. 
These copies are produced by the objects’ copy constructors. 
This can make pass-by-value an expensive operation. 
For example, consider the following class hierarchy:
```c++
class Person
{
public:
    Person();           // parameters omitted
    virtual ~Person();  // see Item 7 for why this is virtual
    
    // ...
    
private:
    std::string name;
    std::string address;
};

class Student : public Person
{
public:
    Student();          // parameters omitted
    virtual ~Student();

    // ...
    
private:
    std::string schoolName;
    std::string schoolAddress;
};
```
Now consider the following code, in which we call a function, `validateStudent`,
that takes a `Student` argument (by value) and returns whether it has been validated:
```c++
bool validateStudent(Student s);          // function taking a Student by value

Student plato;                            // Plato studied under Socrates
bool platoIsOK = validateStudent(plato);  // call the function
```
When this function is called, 
the `Student` copy constructor is called to initialize the parameter `s` from `plato`. 
Equally clearly, `s` is destroyed when `validateStudent` returns. 
So the parameter-passing cost of this function is 
one call to the `Student` copy constructor and one call to the `Student` destructor.


But that’s not the whole story. 
A `Student` object has two `std::string` objects within it, 
so every time you construct a `Student` object, you must also construct two `std::string` objects. 
A `Student` object also inherits from a `Person` object, 
so every time you construct a `Student` object, you must also construct a `Person` object. 
A `Person` object has two additional `std::string` objects inside it, 
so each `Person` construction also entails two more `std::string` constructions. 
The end result is that passing a `Student` object by value leads to 
one call to the `Student` copy constructor, 
one call to the `Person` copy constructor, 
and four calls to the `std::string` copy constructor. 
When the copy of the `Student` object is destroyed, 
each constructor call is matched by a destructor call, 
so the overall cost of passing a `Student` by value is six constructors and six destructors!


Now, this is correct and desirable behavior. 
After all, you want all your objects to be reliably initialized and destroyed. 
Still, it would be nice if there were a way to bypass all those constructions and destructions.
There is: pass by reference-to-`const`:
```c++
bool validateStudent(const Student & s);
```
This is much more efficient: no constructors or destructors are called,
because no new objects are being created. 
The `const` in the revised parameter declaration is important. 
The original version of `validateStudent` took a `Student` parameter by value, 
so callers knew that they were shielded from any changes the function might make to the `Student` they passed in; 
`validateStudent` would be able to modify only a copy of it. 
Now that the `Student` is being passed by reference, 
it’s necessary to also declare it `const`, 
because otherwise callers would have to worry about `validateStudent` making changes to the `Student` they passed in.


Passing parameters by reference also avoids the <u>_slicing problem_</u>. 
When a derived class object is passed (by value) as a base class object, 
the base class copy constructor is called, 
and the specialized features that make the object behave like a derived class object are “sliced” off.
You’re left with a simple base class object, since a base class constructor created it. 
This is almost **never** what you want.
For example, suppose you’re working on a set of classes for implementing a graphical window system:
```c++
class Window
{
public:
    // ...

    std::string name() const;      // return name of window
    virtual void display() const;  // draw window and contents
};

class WindowWithScrollBars : public Window
{
public:
    // ...

    virtual void display() const;
};
```
All `Window` objects have a name, which you can get at through the `name` function, 
and all windows can be displayed, which you can bring about by invoking the `display` function. 
The fact that display is `virtual` tells you that 
the way in which simple base class `Window` objects are displayed 
is apt to differ from the way in which the fancier `WindowWithScrollBars` objects are displayed (see Items 34 and 36). 


Now suppose you’d like to write a function to print out a window’s name and then display the window. 
Here’s the **wrong** way to write such a function:
```c++
// incorrect! parameter may be sliced!
void printNameAndDisplay(Window w)
{
    std::cout << w.name() << '\n';
    w.display();
}
```
Consider what happens when you call this function with a `WindowWithScrollBars` object:
```c++
WindowWithScrollBars wwsb;
printNameAndDisplay(wwsb);
```
The parameter `w` will be constructed as a `Window` object, 
and all the specialized information that made `wwsb` act like a `WindowWithScrollBars` object will be sliced off.
Inside `printNameAndDisplay`, 
`w` will always act like an object of class `Window` (because it is an object of class `Window`), 
regardless of the type of object passed to the function. 
In particular, the call to display inside `printNameAndDisplay` will always call `Window::display`, 
never `WindowWithScrollBars::display`.
The way around the slicing problem is to pass `w` by reference-to-`const`:
```c++
void printNameAndDisplay(const Window & w)
{
    std::cout << w.name() << '\n';
    w.display();
}
```
Now `w` will act like whatever kind of window is actually passed in.


If you peek under the hood of a C++ compiler, 
you’ll find that references are typically implemented as pointers, 
so passing something by reference usually means really passing a pointer. 
As a result, if you have an object of a built-in type (e.g., an int), 
it’s often more efficient to pass it by value than by reference. 
For built-in types, then, when you have a choice between pass-by-value and pass-by-reference-to-`const`,
it’s not unreasonable to choose pass-by-value. 
This same advice applies to iterators and function objects in the STL, 
because, by convention, they are designed to be passed by value. 
Implementers of iterators and function objects are responsible for seeing to it that 
they are efficient to copy and are not subject to the slicing problem. 


Built-in types are small, so some people conclude that all small types are good candidates for pass-by-value, 
even if they’re user-defined.
This is shaky reasoning. Just because an object is small doesn’t mean that calling its copy constructor is inexpensive. 
Many objects (most STL containers among them) contain little more than a pointer, 
but copying such objects entails copying everything they point to. 
That can be very expensive.


Even when small objects have inexpensive copy constructors, there can be performance issues. 
Some compilers treat built-in and userdefined types differently, even if they have the same underlying representation.
For example, some compilers refuse to put objects consisting of only a `double` into a register, 
even though they happily place naked `double`s there on a regular basis. 
When that kind of thing happens, you can be better off passing such objects by reference, 
because compilers will certainly put pointers (the implementation of references) into registers.


Another reason why small user-defined types are not necessarily good pass-by-value candidates is that, 
being user-defined, their size is subject to change. 
A type that’s small now may be bigger in a future release, because its internal implementation may change. 
Things can even change when you switch to a different C++ implementation. 
As I write this, for example, some implementations of the `std::string` type are <u>_seven times_</u> as big as others.
In general, the only types for which you can reasonably assume that pass-by-value is inexpensive 
are built-in types and STL iterator and function object types. 
For everything else, follow the advice of this Item and prefer pass-by-reference-to-const over pass-by-value.






### 📌 Item 21: Don't try to return a reference when you must return an object

- **Never** return a pointer or reference to a local stack object, 
  a reference to a heap-allocated object, 
  or a pointer or reference to a local static object
  if there is a chance that more than one such object will be needed. 


Consider a class for representing rational numbers, including a function for multiplying two rationals together:
```c++
class Rational
{
public:
    friend const Rational operator*(const Rational & lhs, const Rational & rhs);
    
    // see Item 24 for why this constructor isn’t declared explicit
    Rational(int numerator = 0, int denominator = 1); 
    
    // ...
    
private:
    int n, d; // numerator and denominator
};
```
This version of `operator*` is returning its result object by value, 
and you’d be shirking your professional duties if you failed to worry about 
the cost of that object’s construction and destruction. 
You don’t want to pay for such an object if you don’t have to. 
So the question is this:do you have to pay?


Well, you don’t have to if you can return a reference instead. 
But remember that a reference is just a name for some existing object. 
Whenever you see the declaration for a reference,
you should immediately ask yourself what it is another name for,
because it must be another name for something.
In the case of `operator*`, if the function is to return a reference,
it must return a reference to some `Rational` object that already exists 
and that contains the product of the two objects that are to be multiplied together.


There is certainly no reason to expect that such an object exists prior to the call to `operator*`. 
That is, if you have
```c++
Rational a(1, 2);    // a = 1/2
Rational b(3, 5);    // b = 3/5
Rational c = a * b;  // c should be 3/10
```
it seems unreasonable to expect that there already happens to exist a rational number with the value 3/10. 
No, if `operator*` is to return a reference to such a number, it must create that number object itself.


A function can create a new object in only two ways: on the stack or on the heap. 
Creation on the stack is accomplished by defining a local variable. 
Using that strategy, you might try to write `operator*` this way:
```c++
// warning! bad code!
const Rational & operator*(const Rational & lhs, const Rational & rhs)
{
    Rational result(lhs.n * rhs.n, lhs.d * rhs.d);
    return result;
}
```
You can reject this approach out of hand, because your goal was to avoid a constructor call, 
and result will have to be constructed just like any other object. 
A more serious problem is that this function returns a reference to result, 
but result is a local object, and local objects are destroyed when the function exits. 
This version of `operator*`, then returns a <u>_dangling reference_</u>. 
Any caller so much as glancing at this function’s return value would instantly enter the realm of undefined behavior. 
The fact is, any function returning a reference (or a pointer) to a local object is broken.


Let us consider, then, the possibility of constructing an object on the heap and returning a reference to it. 
Heap-based objects come into being through the use of `new`, so you might write a heap-based `operator*` like this:
```c++
// warning! worse code!
const Rational & operator*(const Rational & lhs, const Rational & rhs)
{
    Rational * result = new Result(lhs.n * rhs.n, lhs.d * rhs.d);
    return *result;
}
```
Well, you still have to pay for a constructor call, 
because the memory allocated by new is initialized by calling an appropriate constructor,
but now you have a different problem: 
who will apply `delete` to the object conjured up by your use of `new`?


Even if callers are conscientious and well intentioned, 
there’s not much they can do to prevent leaks in reasonable usage scenarios like this:
```c++
Rational w, x, y, z;
w = x * y * z;        // same as operator*(operator*(x, y), z)
```
Here, there are two calls to `operator*` in the same statement, 
hence two uses of new that need to be undone with uses of `delete`. 
Yet there is no reasonable way for clients of `operator*` to make those calls, 
because there’s no reasonable way for them to get at the pointers 
hidden behind the references being returned from the calls to `operator*`. 
This is a guaranteed resource leak.


But perhaps you notice that both the on-the-stack and on-the-heap approaches 
suffer from having to call a constructor for each result returned from `operator*`. 
Perhaps you recall that our initial goal was to avoid such constructor invocations. 
Perhaps you think you know a way to avoid all but one constructor call. 
Perhaps the following implementation occurs to you, 
an implementation based on `operator*`returning a reference to a <u>_static_</u> `Rational` object, 
one defined inside the function:
```c++
// warning! yet more worse code!
const Rational & operator*(const Rational & lhs, const Rational & rhs)
{
    static Rational result;  // static object to which a reference will be returned
    result = ...;            // multiply lhs by rhs and put the product inside result
    return result;
}
```
Like all designs employing the use of static objects,
this one immediately raises our thread-safety hackles, 
but that’s its more obvious weakness. 
To see its deeper flaw, consider this perfectly reasonable client code:
```c++
bool operator==(const Rational& lhs, const Rational& rhs);

Rational a, b, c, d;

// ...

if ((a * b) == (c * d))
{
    // do whatever’s appropriate when the products are equal;
}
else
{
    // do whatever’s appropriate when they’re not;
}
```
The expression `((a * b) == (c * d))` will <u>_always_</u> evaluate to `true`, 
regardless of the values of `a`, `b`, `c`, and `d`!
This revelation is easiest to understand when the code is rewritten in its equivalent functional form:
```c++
if (operator==(operator*(a, b), operator*(c, d)))
```
Notice that when `operator==` is called, there will already be two active calls to `operator*`, 
each of which will return a reference to the static `Rational` object inside `operator*`. 
Thus, `operator==` will be asked to compare the value of the static `Rational` object inside `operator*` 
with the value of the static Rational object inside `operator*`. 
It would be surprising indeed if they did not compare equal.


This should be enough to convince you that returning a reference from a function like `operator*` is a waste of time,
but some of you are now thinking, “Well, if one static isn’t enough, maybe a static array will do the trick...”

First, you must choose `n`, the size of the array. 
If `n` is too small, you may run out of places to store function return values, 
in which case you’ll have gained nothing over the single-static design we just discredited. 
But if `n` is too big, you’ll decrease the performance of your program,
because <u>_every_</u> object in the array will be constructed the first time the function is called. 
That will cost you `n` constructors and `n` destructors (The destructors will be called once at program shutdown), 
even if the function in question is called only once. 
If “optimization” is the process of improving software performance, 
this kind of thing should be called “pessimization.” 
Finally, think about how you’d put the values you need into the array’s objects 
and what it would cost you to do it.
The most direct way to move a value between objects is via assignment,
but what is the cost of an assignment? 
For many types, it’s about the same as a call to a destructor (to destroy the old value) 
plus a call to a constructor (to copy over the new value).
But your goal is to avoid the costs of construction and destruction!


The right way to write a function that must return a new object is to have that function return a new object. 
For `Rational::operator*`, that means either the following code or something essentially equivalent:
```c++
inline const Rational operator*(const Rational & lhs, const Rational & rhs)
{
    return Rational(lhs.n * rhs.n, lhs.d * rhs.d);
}
```
Sure, you may incur the cost of constructing and destructing `operator*`’s return value 
(if **not** considering return value optimization), 
but in the long run, that’s a small price to pay for correct behavior. 
Besides, the bill that so terrifies you may never arrive. 
Like all programming languages, C++ allows compiler implementers to apply optimizations
to improve the performance of the generated code without changing its observable behavior, 
and it turns out that in some cases, 
construction and destruction of `operator*`’s return value can be safely eliminated. 
When compilers take advantage of that fact (and compilers often do), 
your program continues to behave the way it’s supposed to, just faster than you expected.
It all boils down to this: 
when deciding between returning a reference and returning an object, 
your job is to make the choice that offers correct behavior. 
Let your compiler vendors wrestle with figuring out how to make that choice as inexpensive as possible.






### 📌 Item 22: Declare data members `private`

- Declare data members `private`.
  It gives clients syntactically uniform access to data,
  affords fine-grained access control,
  allows invariants to be enforced,
  and offers class authors implementation flexibility.
- `protected` is **no** more encapsulated than `public`.


Data members should **not** be `public` (`protected`) because: 

1. Interface consistency: 
   All data member access to be done via function calls, rather than some by functions and others by direct access. 
2. Fine-grained access control: 
   Everyone has read-write access to `public` members, 
   but access functions can implement read-only, write-only, read-write... access. 
3. Encapsulation: 
   - Data members _should_ be hidden. Rarely does every data member need a getter and setter. 
   - Implementation flexibility: 
     Can switch between multiple implementations of an expensive getter at a cheap cost (only re-complication).  
     Eliminating a `public` data member breaks an unknowably large amount of code. 




### 📌 Item 23: Prefer non-member non-friend functions to member functions

- Prefer non-member non-friend functions to member functions.
  Doing so increases encapsulation, packaging flexibility, and functional extensibility.


Two ways to implement a clear function in a web browser: 
```c++
/// WebBrowser.h
namespace WebBrowserStuff
{

class WebBrowser
{
public:
    // ...
    void clearCache();
    void clearHistory();
    void removeCookies();
    
    // CASE 1
    // calls clearCache, clearHistory, and removeCookies
    void clearEverything(); 
    
    // ...
};

}  // namespace WebBrowserStuff


/// WebBrowserUtil.h
namespace WebBrowserStuff
{

// CASE 2
void clearBrowser(WebBrowser & wb)
{
    wb.clearCache();
    wb.clearHistory();
    wb.removeCookies();
}

}  // namespace WebBrowserStuff
```
The member function `clearEverything` actually yields **less** encapsulation than the non-member `clearBrowser`, 
It also allows for greater packaging flexibility for `WebBrowser`-related functionality, 
and in turn yields fewer compilation dependencies and an increase in `WebBrowser` extensibility. 

The non-member non-friend functions afford us 
the flexibility to change things in a way that affects only a limited number of clients.
Data members should be `private`, otherwise an unlimited number of functions can access them.
`public` / `protected` data members have no encapsulation at all. 
For `private` data members, the number of functions that can access them is 
the number of member functions of the class plus the number of `friend` functions.
Given a choice between a member function 
(which can access not only the `private` data, but also `private` functions, `enum`s, aliases, etc.)
and a non-member non-`friend` function 
(which can access **none** of these things) providing the same functionality, 
the choice yielding greater encapsulation is the non-member non-`friend` function,
because it **doesn’t** increase the number of functions that can access the `private` parts of the class.

NOTE:

1. This non-member function can actually be a member of _another class_.
   This may prove a mild salve to programmers accustomed to languages where all functions must be in classes
   (e.g., Eiffel, Java, C#, etc.).
   For example, we could make `clearBrowser` a `static` member function of some utility class.
   As long as it’s not part of (or a `friend` of) `WebBrowser`,
   it doesn’t affect the encapsulation of `WebBrowser`’s `private` members.
2. Make `clearBrowser` a non-member function <u>_in the same namespace_</u> as `WebBrowser`:
   `namespace`s can be spread across multiple source files.
   That’s important, because functions like `clearBrowser` are convenience functions.
   A class like `WebBrowser` might have a large number of convenience functions.
   As a general rule, most clients will be interested in only some of these sets of convenience functions.
   There’s no reason for a bookmark-related client to be compilation dependent on cookie-related convenience functions.
```c++
/// “WebBrowser.h”
/// header for class WebBrowser itself
/// as well as “core” WebBrowser-related functionality
namespace WebBrowserStuff
{

class WebBrowser
{
    // ...
};

// “core” related functionality, e.g.
// non-member functions almost all clients need
// ... 

}  // namespace WebBrowserStuff

/// “WebBrowserBookmarks.h”
namespace WebBrowserStuff
{

// bookmark-related convenience functions
// ... 

}  // namespace WebBrowserStuff

/// “WebBrowserCookies.h”
namespace WebBrowserStuff
{

// cookie-related convenience functions
// ... 

}  // namespace WebBrowserStuff
```

P.S. This is exactly how the standard C++ library is organized.
Rather than having a single monolithic `<C++StandardLibrary>` header (`<bits/stdc++.h>` is a `gcc` extension)
containing everything in `namespace std`, 
there are dozens of headers (e.g., `<vector>`, `<algorithm>`, `<memory>`, etc.),
each declaring some of the functionality in `std`.
This allows clients to be compilation dependent only on the parts of the system they actually use.

Putting all convenience functions in multiple header files but one namespace 
also means that clients can easily extend the set of convenience functions. 
All they have to do is add more non-member non-`friend` functions to the namespace.
This is another feature classes **can’t** offer, because class definitions are closed to extension by clients.
(Clients can derive new classes, but derived classes have no access to `private` members in the base class. ) 
Besides, as Item 7 explains, not all classes are designed to be base classes.






### 📌 Item 24: Declare non-member functions when type conversions should apply to all parameters

- If you need type conversions on all parameters to a function
  (including the one that would otherwise be pointed to by the `this` pointer),
  the function must be a non-member.
- Not all operators should be implemented as friends. 
  (Consider the case that they can be implemented only using public interfaces. )


Having classes support implicit type conversion is generally a bad idea. 
One exception is numerical type (like `double`-`int` conversion). 
Consider `operator*` for `Rational` type. 
Making `operator*` a member function **fails** sometimes: 
```c++
class Rational
{
public:
    // CASE 1: friend non-member
    friend const Rational operator*(const Rational & lhs, const Rational & rhs);
    
public:
    // deliberate non-explicit constructor allowing implicit int-to-Rational conversions
    Rational(int numerator = 0, int denominator = 1);

    int numerator() const; 

    int denominator() const; 
    
    // CASE 2: member
    const Rational operator*(const Rational & rhs) const;
    
private:
    // ...
};

// CASE 3: non-member non-friend
const Rational operator*(const Rational & lhs, const Rational & rhs);

Rational oneHalf(1, 2);
Rational one = oneHalf * 2;  // fine
one = 2 * oneHalf;           // CASE 1 fine, CASE 2 error!
                             // operator*(2, oneHalf) v.s. 2.operator*(oneHalf)
```
Making things worse: 
If `Rational::Rational(int numerator = 0, int denominator = 1)` is made `explicit`, 
**neither** of these 2 statements will compile. 

LESSON: Parameters are eligible for implicit type conversion _only if they are listed in the parameter list_. 

NOTE: In this case, `operator*` should **not** be made friend of `Rational`, 
as it could have been implemented only using `Rational`'s public interfaces. 






### 📌 Item 25: Consider support for a `noexcept` `swap`

- Provide a swap member function when `std::swap` would be inefficient for your type.
  Make sure your swap is `noexcept`.
- If you offer a member `swap`, also offer a non-member `swap` that calls the member.
  For classes (not templates), specialize `std::swap`, too.
  For class templates, overload swap (as function templates can **not** be partially specialized. ). 
- When calling `swap`, employ a using declaration `using std::swap;`,
  then call `swap` **without** namespace qualification.
- It's fine to totally specialize `std` templates for user-defined types,
  but **never** try to add something completely new to `namespace std`.


First, if the default implementation of swap 
offers acceptable efficiency for your class or class template, 
you don't need to do anything. 

Second, if the default implementation of swap isn't efficient enough 
(which almost always means that your class or template is using some variation of the _pimpl idiom_), do the following:

1. Offer a `public` `noexcept` swap member function that efficiently swaps the value of two objects of your type.
2. Offer a non-member swap in the same namespace as your class or template. Have it call your swap member function.
3. If you're writing a class (not a class template), specialize `std::swap` for your class. 
   Have it also call your swap member function.
   
Finally, if you're calling swap, be sure to include a `using` declaration to make `std::swap` visible in your function, 
then call swap without any namespace qualification.

```c++
/// Adapted and simplified from 
/// <bits/move.h>
/// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
template <typename T>
inline
typename std::enable_if_t<std::__and_v<std::__not_<std::__is_tuple_like<T>>,
        std::is_move_constructible<T>,
        std::is_move_assignable<T>>>
swap(T & a, T & b)
noexcept(std::__and_v<std::is_nothrow_move_constructible<T>,
        std::is_nothrow_move_assignable<T>>)
{
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}
```

```c++
namespace WidgetStuff
{

template <typename T>
class Widget
{
public: 
    // ...
    
    void swap(Widget & other)
    {
        using std::swap;
        swap(this->pImpl, other.pImpl);
    }
    
    // ...
};

// ...

template <typename T>
void swap(Widget<T> & a, Widget<T> & b)
{
    a.swap(b);
}

}  // namespace WidgetStuff
```






### 🎯 Chapter 5. Implementations

### 📌 Item 26: Postpone variable definitions as long as possible

- Postpone variable definitions as long as possible.
  It increases program clarity and improves program efficiency.






### 📌 Item 27: Minimize casting

- Avoid casts whenever practical, especially `dynamic_cast`s in performance-sensitive code.
  If a design requires casting, try to develop a cast-free alternative.
- When casting is necessary, try to hide it inside a function.
  Clients can then call the function instead of putting casts in their own code.
- Prefer C++-style casts to old-style casts.
  They are easier to see, and they are more specific about what they do.

#### Single object with multiple address

```c++
struct Base1
{
    virtual ~Base1() = default;
};

struct Base2
{
    virtual ~Base2() = default;
};

struct Derived : public Base1, public Base2
{
    ~Derived() override = default;
};

Derived obj;
Derived * d = &obj;  // 0x7ffd59fcc230
Base1 * b1 = &obj;   // 0x7ffd59fcc230
Base2 * b2 = &obj;   // 0x7ffd59fcc238 !!!
```
When creating a base class pointer to a derived class object, 
the two pointer values will **not** be the same. 
When that's the case,
an offset is applied at runtime to the `Derived *` pointer to get the correct `Base *` pointer value.

This example demonstrates that a single object (e.g., an object of type `Derived`) might have more than one address 
(e.g., its address when pointed to by a `Base2 *` pointer and its address when pointed to by a `Derived *` pointer). 
When multiple inheritance is in use, it happens virtually all the time, but it can happen under single inheritance, too. 
Among other things, that means you should generally avoid ~~making assumptions about how things are laid out~~ in C++, 
and you should certainly **not** perform casts based on such assumptions. 
For example, casting object addresses to `char *` and then using pointer arithmetic on them 
almost always yields _undefined behavior_. 

#### `static_cast<Non-reference-type>` yields prvalue copy

An interesting thing about casts is that it's easy to write something that 
looks right (and might be right in other languages) but is wrong. 
Many application frameworks require that 
virtual member function implementations in derived classes call their base class counterparts first.
Suppose we have a `Window` base class and a `SpecialWindow` derived class, 
both of which define the virtual function `onResize`. 
Further suppose that `SpecialWindow::onResize` is expected to invoke `Window::onResize` first.
Here's a way to implement this that looks like it does the right thing, but **doesn't**:
```c++
class Window
{
public:
    virtual void onResize()
    {
        // ...
    }
};

class SpecialWindow : public Window
{
public:
    void onResize() override
    {
        // WRONG! 
        // onResize is applied on a temporary Window instance returned by static_cast!
        static_cast<Window>(*this).onResize(); 
        // ...
    }
};
```
`Window::onResize` is **not** invoked on the current object! 
Instead, the cast creates a new, temporary copy of the base class part of `*this`, then invokes `onResize` on the copy! 
The above code it calls `Window::onResize` on a copy of the base class part of the current object 
before performing `SpecialWindow`-specific actions on the current object.
The solution is to eliminate the cast. 
You don't want to trick compilers into treating `*this` as a base class object; 
you want to call the base class version of `onResize` on the current object. 
```c++
class SpecialWindow : public Window
{
public:
    void onResize() override
    {
        // Correct. Call Window::onResize() on *this
        Window::onResize();
        // ...
    }
};
```

#### `dynamic_cast`

**Performance**. 
Many implementations of `dynamic_cast` are slow. 
E.g., one common implementation is based on `strcmp` of class names. 
If you're performing a `dynamic_cast` on an object in a single-inheritance hierarchy 10 levels deep, 
each `dynamic_cast` under such an implementation could cost you _up to 10 calls to `strcmp`_ to compare class names! 
There are reasons that some implementations work this way (they have to do with support for dynamic linking). 

The need for `dynamic_cast` generally arises when performing derived class operations on base class pointers/references. 
There are two ways to bypass this issue:

1. Use `Container<std::shared_ptr<Derived *>>` directly. 
2. Use virtual functions. 






### 📌 Item 28: Avoid returning handles to object internals

- Avoid returning handles (references, pointers, or iterators) to object internals.
  Not returning handles increases encapsulation,
  helps const member functions act `const`,
  and minimizes the creation of dangling handles.






### 📌 Item 29: Strive for exception-safe code

- Exception-safe functions leak no resources
  and allow no data structures to become corrupted,
  even when exceptions are thrown.
  Such functions offer the 
- basic (no resource leak, no corrupted data), 
  strong (no resource leak, data either completely updated or completely untouched), 
  or `noexcept` guarantees. 
- The strong guarantee can often be implemented via copy-and-swap,
  but the strong guarantee is not practical for all functions.
- A function can usually offer a guarantee **no** stronger
  than the weakest guarantee of the functions it calls.






### 📌 Item 30: Understand the ins and outs of inlining

- Limit most inlining to small, frequently called functions.
  This facilitates debugging and binary upgradability, minimizes potential code bloat,
  and maximizes the chances of greater program speed.
- **Don't** declare function templates `inline` just because they appear in header files.


`inline` is a request to compilers, **not** a command. 
The request can be given implicitly or explicitly. 
The implicit way is to _define a function inside a class definition_.
The explicit way to declare an inline function is to precede its definition with the `inline` keyword. 

Inline functions must typically be in header files, because most build environments do inlining during compilation. 
In order to replace a function call with the body of the called function,
compilers must know what the function looks like. 
(Some build environments can inline during linking, and a few can actually inline at runtime. 
Such environments are the exception, however, not the rule. 
Inlining in most C++ programs is a compile-time activity. )

Templates are typically in header files, 
because compilers need to know what a template looks like in order to instantiate it when it's used. 
(Again, this is not universal. 
Some build environments perform template instantiation during linking. 
However, compile-time instantiation is more common. )

Template instantiation is independent of inlining. 

`inline` is a request that compilers may ignore. 
Most compilers refuse to inline functions they deem too complicated (e.g., those that contain loops or are recursive).  
All but the most trivial calls to virtual functions defy inlining (can not determine which function to call during compile time). 

Sometimes compilers generate a function body for an inline function 
even when they are perfectly willing to inline the function. 
For example, if your program takes the address of an inline function (explictly by programmer, or implicitly by compiler), 
compilers must typically generate an outlined function body for it.
```c++
inline void f() 
{
    // ...
}

void (*pf)() = f;

f();   // inline
pf();  // NOT inline
```






### 📌 Item 31: Minimize compilation dependencies between files

- The general idea behind minimizing compilation dependencies
  is to depend on declarations instead of definitions (except standard libraries).
  Two approaches based on this idea are Handle classes (pimpl idiom) and Interface classes (abstract base class).
- Library header files should exist in full and declaration-only forms.
  This applies regardless of whether templates are involved 
  (Some environment allows template to be implemented separately).


A C++ class definition specifies not only a class interface but also a fair number of implementation details.
The class `Person` **can’t** be compiled without access to definitions for data members, 
which are typically _provided through `#include` directives_. 
```c++
#include <string>     // dependency

#include "Address.h"  // dependency
#include "Date.h"     // dependency


class Person
{
public:
    Person(const std::string & name, const Date & birthday, const Address & address);

    std::string name() const;
    std::string birthday() const;
    std::string address() const;

    // ...
    
private:
    std::string mName;  // implementation detail
    Date mBirthday;     // implementation detail
    Address mAddress;   // implementation detail
};
```

<u><i>Pimpl Idiom</i></u>: 
A _handle class_ hide object implementation behind pointers. 
CONS: Adds one additional level of indirection per access. 
Takes more space and incurs dynamic memory management. 
```c++
// standard library should NEVER be forward-declared
#include <memory>
#include <string>     


class Date;
class Address;


class Person
{
public:
    Person(const std::string & name, const Date & birthday, const Address & address);
    ~Person();  // declaration only, = default in implementation file, to bypass unique_ptr-related type error
    
    std::string name() const;
    std::string birthday() const;
    std::string address() const;

    // ...
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
```

General guideline: 
1. **Avoid using objects when references and pointers will do**.
2. **Depend on forward declarations (except standard libraries) instead of definitions whenever you can**.
   - **Never** add (forward-declare, declare, define, etc.) anything to `namespace std`. 
3. **Provide separate header files for both definitions and forward declarations** 
   (`Widget.h` + `WidgetFwd.h`). 

<u><i>Interface Class</i></u>: 
Abstract base class + derived implementation class. 
CONS: Pay for indirect jump in virtual function calls. Takes more space (virtual function table pointer). 
```c++
// "Person.h"
class Person
{
public:
    virtual ~Person() = 0;

    virtual std::string name() const = 0;
    virtual std::string birthDate() const = 0;
    virtual std::string address() const = 0;

    static std::unique_ptr<Person> create(const std::string & name, const Date & birthday, const Address & address);

    // ...
};

// "PersonImpl.h"
class PersonImpl : public Person
{
public:
    PersonImpl(const std::string & name, const Date & birthday, const Address & address)
            : mName(name), mBirthday(birthday), mAddress(address)
    {
    
    }

    virtual ~PersonImpl() = default;
    
    std::string name() const;
    std::string birthDate() const;
    std::string address() const;

private:
    std::string mName;
    Date mBirthday;
    Address mAddress;
};

// "PersonImpl.cpp"
std::unique_ptr<Person> Person::create(const std::string & name, const Date & birthday, const Address & address)
{
    return std::make_unique<PersonImpl>(name, birthday, address);
}

// client code somewhere
std::unique_ptr<Person> pp(Person::create(name, birthday, address));
std::cout << pp->name() << ' ' << pp->birthday << ' ' << pp->address() << '\n';
```






### 🎯 Chapter 6. Inheritance and Object-Oriented Design

### 📌 Item 32: Make sure public inheritance models “is-a”

- Public inheritance means “is-a”.
  Everything that applies to base classes must also apply to derived classes,
  because every derived class object is a base class object.






### 📌 Item 33: Avoid hiding inherited names

- Names in derived classes hide names in base classes.
  Under public inheritance, this is **never** desirable.
- To make hidden names visible again,
  employ using declarations or forwarding functions.






### 📌 Item 34: Differentiate between inheritance of interface and inheritance of implementation

- Inheritance of interface is different from inheritance of implementation.
  Under public inheritance, derived classes always inherit base class interfaces.
- **Pure virtual functions** specify inheritance of _interface only_.
- **Simple (impure) virtual functions** specify inheritance of interface plus inheritance of a _default implementation_.
- **Non-virtual functions** specify inheritance of interface plus inheritance of a _mandatory implementation_.






### 📌 Item 35: Consider alternatives to virtual functions

- Alternatives to virtual functions include the NVI idiom and various forms of the Strategy design pattern.
  The NVI idiom is itself an example of the Template Method design pattern.
- A disadvantage of Template Method (moving functionality from a member function to a function outside the class)
  is that the non-member function lacks access to the class's non-public members.
- `std::function` objects act like generalized function pointers.
  Such objects support all callable entities compatible with a given target signature.

```c++
class GameCharacter
{
public:
    virtual int healthValue() const;  // impurity implies there is a default implementation
    
    // ... 
};
```

#### The Template Method Pattern via the Non-Virtual Interface (NVI) Idiom

_Non-Virtual Interface (NVI) Idiom_: 
Have clients call **private** virtual functions indirectly through public non-virtual member functions.
This is a particular manifestation of the more general design pattern called _Template Method_. 
```c++
class GameCharacter 
{
public:
    int healthValue() const
    {
        // ... 
        int retVal = doHealthValue();
        // ...
        return retVal;
    }
    
    // ...
    
private:
    virtual int doHealthValue() const
    {
        // default algorithm for calculating character’s health
    }
};
```

#### The Strategy Pattern via Function Pointers

Strategy Pattern: 
```c++
class GameCharacter;

int calculateHealth(const GameCharacter & gc);

class GameCharacter
{
public:
    using HealthCalculator = int (*)(const GameCharacter &);

    explicit GameCharacter(HealthCalculator hcf = calculateHealth) : healthFunc(hcf)
    {
        
    }

    int healthValue() const
    {
        return healthCalculator(*this);
    }
    
    // ...
    
private:
    HealthCalculator healthCalculator;
};
```

#### The Strategy Pattern via `std::function`

Skipped. (`std::function` and `std::bind` are dissed in Effective Modern C++. )






### 📌 Item 36: Never re-define an inherited non-virtual function

- **Never** re-define an inherited non-virtual function.






### 📌 Item 37: Never redefine a function's inherited default parameter value

- **Never** redefine an inherited default parameter value,
  because default parameter values are statically bound,
  while virtual functions (the only functions you should be overriding) are dynamically bound.


You may end up invoking a virtual function defined in a derived class 
but using a default parameter value from a base class. 






### 📌 Item 38: Model “has-a” or “is-implemented-in-terms-of” through composition

- Composition has meanings completely different from that of public inheritance.
- In the application domain, composition means has-a. 
  In the implementation domain, it means is-implemented-in-terms-of.








### 📌 Item 39: Use private inheritance judiciously (only when having considered all the alternatives)

- Private inheritance means is-implemented-in-terms-of.
  It's usually inferior to composition,
  but it makes sense when a derived class needs access to protected base class members
  or needs to redefine inherited virtual functions.
- Unlike composition, private inheritance can enable the empty base optimization.
  This can be important for library developers who strive to minimize object sizes.


In terms of "is-implemented-in-terms-of", private inheritance is not necessary, 
as composition do the same stuff. 


This accurately reflects the fact that the reason for the base class 
is only to facilitate the derived classes’ implementations, 
not to express a conceptual is-a relationship. 


One common use of private inheritance could be something like `DisableCopy`:
```c++
// Almost identical to boost::noncopyable
// available in <boost/core/noncopyable.hpp>
// https://www.boost.org/doc/libs/1_63_0/libs/core/doc/html/core/noncopyable.html
class DisableCopy
{
public:
    DisableCopy(const DisableCopy &) = delete;
    DisableCopy operator=(const DisableCopy &) = delete;
    
protected:
    DisableCopy() = default;
    ~DisableCopy() = default;
};


class Derived : private DisableCopy
{
public:
    Derived() = default;
    
    // ...
};`
o
```






### 📌 Item 40: Use multiple inheritance judiciously (only when having considered all the alternatives)

- Multiple inheritance is more complex than single inheritance.
  It can lead to new ambiguity issues and to the need for virtual inheritance.
- Virtual inheritance imposes costs in size, speed, and complexity of initialization and assignment.
  It's most practical when virtual base classes have no data (e.g. abstract virtual base class).
- Multiple inheritance does have legitimate uses.
  One scenario involves combining public inheritance from an Interface class
  with private inheritance from a class that helps with implementation.


One C++ multiple inheritance hierarchy: 
`basic_ios` (virtual base), `basic_istream`, `basic_ostream`, `basic_iostream`. 






### 🎯 Chapter 7. Templates and Generic Programming

The C++ template mechanism is itself _Turing-complete_: 
It can be used to compute any computable value. 
That led to _Template Metaprogramming (TMP)_: 
The creation of programs that execute inside C++ compilers and that stop running when compilation is complete. 






### 📌 Item 41: Understand implicit interfaces and compile-time polymorphism

- Both classes and templates support interfaces and polymorphism.
- For classes, interfaces are explicit and centered on function signatures.
  Polymorphism occurs _at runtime_ through virtual functions.
- For template parameters, interfaces are implicit and based on valid expressions.
  Polymorphism occurs _during compilation_ through template instantiation and function overloading resolution.


**Object-oriented programming (OOP)** revolves around _explicit interfaces_ and _runtime polymorphism_.
```c++
class Widget
{
public:
    Widget();
    
    virtual ~Widget();
    
    virtual std::size_t size() const;
    
    virtual void normalize();
    
    void swap(Widget &);
};


void doProcessing(Widget & w)
{
    if (10 < w.size() && w != someNastyWidget)
    {
        Widget temp(w);
        temp.normalize();
        temp.swap(w);
    }
}
```

- Because `w` is declared to be of type `Widget`, `w` must support the `Widget` interface. 
  We can look up this interface in the source code to see exactly what it looks like, 
  so I call this an _explicit interface_: 
 one explicitly visible in the source code.
- Because some of `Widget`’s member functions are `virtual`, 
 `w`’s calls  to those functions will exhibit _runtime polymorphism_: 
  the specific function to call will be determined at runtime based on `w`’s dynamic type.


**Template meta programming (TMP)** is fundamentally different. 
Explicit interfaces and runtime polymorphism continue to exist yet being less important. 
Instead, _implicit interfaces_ and _compile-time polymorphism_ move to the fore.
```c++
template <typename T>
void doProcessing(T & w)
{
    if (10 < w.size() && w != someNastyWidget)
    {
        T temp(w);
        temp.normalize();
        temp.swap(w);
    }
}
```
- The interface that `w` must support is determined by the operations performed on `w` in the template. 
  In this example, it appears that `T` must support the `size`, `normalize`, `swap` member functions, 
  and be _copy-constructable_ and _comparable_. 
  The set of expressions that must be valid in order for the template to compile 
  is the _implicit interface_ that `T` must support. 
- The calls to functions involving `w` such as `operator<` and `operator!=`
  may involve instantiating templates to make these calls succeed. 
  Such instantiation occurs during compilation. 
  Because instantiating function templates with different template parameters 
  leads to different functions being called, 
  this is known as _compile-time polymorphism_. 


Runtime/compile-time polymorphism is just 
dynamic binding of virtual functions (at runtime)
and overload resolution (at compile-time). 


An explicit interface typically consists of _function signatures_.
An implicit interface consists of _valid expressions_. 


The implicit interface for `T` appears to have these constraints:
- It "must" offer a member function named `size` that returns an integral value.
- It "must" support `operator!=` function that compares two objects of type `T`. 
  (Here, we assume that `someNastyWidget` is of type `T`.)


Thanks to the possibility of _operator overloading_, **neither** of these constraints need be satisfied. 
Yes, `T` must support a size member function, though the function might be inherited from a base class. 
But this member function need **not** return a numeric type, not even a type for which `operator<` is defined. 
All it needs to do is return an object of some type `X` such that 
there is an `operator<` that can be called with an object of type `X` and an `int`. 
The `operator<` need **not** take a parameter of type `X`, 
because it could take a parameter of type `Y` which can be converted from `X` implicitly. 


Similarly, there is **no** requirement that `T` support `operator!=`, 
because it would be just as acceptable for `operator!= `to take one `X` and one `Y`. 
As long as `T` can be converted to `X` and `someNastyWidget`’s type can be converted to `Y`, 
the call to `operator!=` would be valid.


(As an aside, this analysis **doesn’t** take into account the possibility  that `operator&&` could be overloaded, 
thus changing the meaning of the above expression from a conjunction to something potentially quite different.)






### 📌 Item 42: Understand the two meanings of `typename`

- When declaring _template parameters_, `class` and `typename` are identical.
- Use `typename` to identify nested dependent type names,
  **except** in base class lists or as a base class identifier in a member initialization list.


Names in a template that are dependent on a template parameter are called _dependent names_. 
When a dependent name is nested inside a class, I call it a _nested dependent name_. 
E.g., `C::const_iterator` is a nested dependent name. 
In fact, it’s a `nested dependent type name`, i.e., a nested dependent name that refers to a type.


On the other hand, `int` is a name that does **not** depend on any template parameter. 
Such names are known as `non-dependent names`. 


Nested dependent names can lead to parsing difficulties.
```c++
template <typename C>
void print2nd(const C & container)
{
    C::const_iterator * x;
    // ...
}
```
This looks like we’re declaring `x` as a local variable that’s a pointer to a `C::const_iterator`. 
But it looks that way only because we “know” that `C::const_iterator` is a type. 
But what if `C::const_iterator` weren’t a type?
What if `C` had a `static` data member that happened to be named `const_iterator`, 
and what if `x` happened to be the name of a global variable?
In that case, the code above wouldn’t declare a local variable, 
it would be a multiplication of `C::const_iterator` by `x`! 


Until `C` is known, there’s no way to know whether `C::const_iterator` is a type or isn’t, 
and when the template `print2nd` is parsed, `C` isn’t known.
C++ has a rule to resolve this ambiguity: 
if the parser encounters a nested dependent name in a template, 
it assumes that the name is **not** a type unless you use keyword `typename`, with two exceptions. 
By default, nested dependent names are not types.


The general rule is simple: 
Anytime you refer to a nested dependent type name in a template, 
you must immediately precede it by the keyword `typename`.
(With two exceptions to be detailed afterwards.) 


The exception to the “`typename` must precede nested dependent type names” rule is that 
`typename` must not precede nested dependent type names: 
- in a list of base classes, or: 
- as a base class identifier in a member initialization list.
```c++
template <typename T>
class Derived : public Base<T>::Nested  // base class list: typename not allowed
{ 
public: 
    explicit Derived(int x)
            : Base<T>::Nested(x)  // base class identifier in member initializer list: typename not allowed
    {
        // use of nested dependent typename not in a base class list 
        // or as a base class identifier in a member initializer list: typename required
        typename Base<T>::Nested temp; 
    }
};
```






### 📌 Item 43: Know how to access names in templatized base classes

- In derived class templates, refer to names in base class templates
  via a `this->` prefix (hints compiler to search base template scope too),
  via using declarations,
  or via an explicit base class qualification.


```c++
template <typename T>
class Base
{
public:
    T get() 
    { 
        return T(1); 
    }
};

template <typename T>
class Derived : public Base<T>
{
public:
    T foo1()
    {
        return this->get();
    }
    
    using Base<T>::get;

    T foo2()
    {
        return get();
    }

    T foo3()
    {
        return Base<T>::get();
    }
};

Derived<int> d;
std::cout << d.foo1() << d.foo2() << d.foo3() << d.get() << '\n';  // 1111
```


### 📌 Item 44: Factor parameter-independent code out of templates

- Templates generate multiple classes and multiple functions,
  so any template code not dependent on a template parameter causes bloat.
- Bloat due to non-type template parameters can often be eliminated
  by replacing template parameters with function parameters or class data members.
- Bloat due to type parameters can be reduced
  by sharing implementations for instantiation types with identical binary representations.


Using templates can lead to _code bloat_: Binaries with replicated (or almost replicated) code, data, or both.

This template takes a _type parameter_ `T`, and a _non-type parameter_ `std::size_t n`.
```c++
template <typename T, std::size_t n>
class SquareMatrix
{
public:
    // ...
    void invert();
};
```

Two copies of `invert` will be instantiated here. 
Other than the constants 5 and 10, the two functions will be the same. 
This is a classic way for template-induced code bloat to arise.
```c++
SquareMatrix<double, 5> sm1;
// ...
sm1.invert(); // call SquareMatrix<double, 5>::invert

SquareMatrix<double, 10> sm2;
// ...
sm2.invert(); // call SquareMatrix<double, 10>::invert
```

Solution: Share a single copy of the base class template’s version of invert.
`private` inheritance accurately reflects the fact that the reason for `SquareMatrixBase`
is only to facilitate the `SquareMatrix`’ implementations,
not to express a conceptual is-a relationship.
```c++
template <typename T>
class SquareMatrixBase
{
protected:
    SquareMatrixBase(std::size_t n, T * data) : mSize(n), mData(data)
    {
        
    }
    
    void setData(T * ptr)
    {
        mData = ptr;
    }
    
    // ...
    
private:
    std::size_t mSize;
    T * mData;
};

template <typename T, std::size_t n>
class SquareMatrix : private SquareMatrixBase<T>
{
public:
    SquareMatrix() : SquareMatrixBase<T>(n, nullptr)
    { 
        SquareMatrixBase<T>::setData(data.data());
    }
    
    void invert()
    {
        SquareMatrixBase<T>::invert(n);
    }

    // ...
    
private:
    std::array<T, n * n> data;
};
```






### 📌 Item 45: Use member function templates to accept “all compatible types”

- Use member function templates to generate functions that accept all compatible types.
- If you declare member templates for generalized copy construction or generalized assignment,
  you'll still need to declare the normal copy constructor and copy assignment operator, 
  as function templates do **not** affect auto-generation of special member functions.


There is **no** inherent relationship among different instantiations of the same template, 
so compilers view `SmartPtr<Middle>` and `SmartPtr<Top>` as completely different classes, 
no more closely related than, say, `std::vector<float>` and `Widget`.

_Generalized copy constructors via member function templates_.
```c++
template <typename T>
class SmartPtr
{
public:
    // Generalized copy constructor. 
    // Deliberately non-explicit, enabling implicit conversions between SmartPtrs. 
    // Complies iff. there is an implicit conversion from U * to T *.
    template <typename U>
    SmartPtr(const SmartPtr & other) : ptr(other.get())
    {
        
    }
    
    // Regular copy constructor
    // NOTE: function templates do NOT affect auto-generation of regular default copy constructors. 
    // You still need to provide a regular copy constructor
    SmartPtr(const SmartPtr<T> & other) : ptr(other.get())
    {
        
    }
    
    T * get() const 
    { 
        return ptr; 
    }
    
    // ...
};

private:
    T * ptr;
```






### 📌 Item 46: Define non-member functions inside templates (as friends) when type conversions are desired on all arguments

- Implicit type conversion functions are **never** considered during _template argument deduction_.
- Class templates **don’t** depend on _template argument deduction_ (which applies only to function templates). 
    - Might be incorrect in later C++ standards (some class template usage requires no template argument too). 
- When writing a class template offering functions
  that support implicit type conversions on all arguments to this class template,
  define those functions as friends inside the class template.


Implicit type conversion functions are **never** considered during _template argument deduction_.
Implicit type conversion is used during _function calls_. 
The compiler do _overload resolution_ to determine which function to call, 
and the first step of overload resolution, i.e., finding candidate functions, involves template argument deduction. 
The following code will **not** compile because failure in template argument deduction (`Rational<int>` v.s. `int`):
```c++
template <typename T>
class Rational
{
public:
    Rational(const T & numerator = 0, const T & denominator = 1);
    const T numerator() const;
    const T denominator() const;
    
    // ...
};

// DOES NOT COMPILE!
template <typename T>
Rational<T> operator*(const Rational<T> & lhs, const Rational<T> & rhs)
{
    return {lhs.numerator() * rhs.numerator(), lhs.denominator() * rhs.denominator()};
}

Rational<int> oneHalf {1, 2};
Rational<int> res = oneHalf * 2;  // won't compile here
```

Class `Rational<T>` can declare `operator*` for `Rational<T>` as a friend function.
This code compiles because when the object `oneHalf` is declared to be of type `Rational<int>`, 
the class `Rational<int>` is already instantiated, and as part of that process, 
the friend function `operator*` that takes `Rational<int>` parameters is automatically declared. 
As a declared _function_ (**not** a _function template_), 
compilers can use implicit conversion functions (such as `Rational`’s non-explicit constructor) when calling it.


Inside a class template, the name of the template can be used as shorthand for the template and its parameters,
so inside `Rational<T>`, we can just write `Rational` instead of `Rational<T>`.
Yet, this code **won't** link:
```c++
template <typename T>
class Rational
{
public:
    // COMPILES BUT WON'T LINK! 
    friend Rational operator*(const Rational &, const Rational &);
    
public:
    Rational(const T & numerator = 0, const T & denominator = 1);
    const T numerator() const;
    const T denominator() const;
    
    // ...
};

// COMPILES BUT WON'T LINK! 
template <typename T>
Rational<T> operator*(const Rational<T> & lhs, const Rational<T> & rhs)
{
    return {lhs.numerator() * rhs.numerator(), lhs.denominator() * rhs.denominator()};
}

Rational<int> oneHalf {1, 2};
Rational<int> res = oneHalf * 2;  // won't link here
```
The _function_ (**not** _function template_)
`operator*(const Rational<int> &, const Rational<int> &)` 
is only _declared_ inside `Rational`, but **not** _defined_ at all. 
A function template definition (not yet instantiated with `T = int`) is **not** a definition of this function.
In this case, we never provide a definition, and that’s why linkers can’t find one.


The simplest thing that could possibly work is to merge the body of `operator*` into its declaration:
```c++
template <typename T>
class Rational
{
public:
    friend Rational operator*(const Rational & lhs, const Rational & rhs)
    {
        return {lhs.numerator() * rhs.numerator(), lhs.denominator() * rhs.denominator()};
    }
    
public:
    Rational(const T & numerator = 0, const T & denominator = 1);
    const T numerator() const;
    const T denominator() const;
    
    // ...
};
```
An interesting observation about this technique is that 
_the use of `friend`ship has **nothing** to do with a need to access non-public parts of the class._ 
In order to make type conversions possible on all arguments, we need a non-member function; 
and in order to have the proper function automatically instantiated, 
we need to declare the function inside the class. 
The only way to declare a non-member function inside a class is to make it a `friend`.
(P.S. 
This friend declaration inside class body is not visible to regular name lookup (both restricted and unrestricted), 
but can be found via Argument-dependent Lookup, or ADL.)


Functions defined inside a class are implicitly declared `inline`, 
and that includes `friend` functions like `operator*`. 
You can minimize the impact of such inline declarations by having `operator*`
do nothing but call a helper function defined outside the class. 
In the example in this Item, there’s not much point in doing that,
because `operator*` is already implemented as a one-line function, 
but for more complex function bodies, it may be desirable. 
It’s worth taking a look at the “have the friend call a helper” approach.
```c++
template <typename T> 
class Rational;

namespace
{

template <typename T>
Rational<T> doMultiply(const Rational<T> & lhs, const Rational<T> & rhs)
{
    return {lhs.numerator() * rhs.numerator(), lhs.denominator() * rhs.denominator()};
}

}  // namespace

template <typename T>
class Rational
{
public:
    friend Rational operator*(const Rational & lhs, const Rational & rhs)
    {
        return doMultiply(lhs, rhs);
    }
    

public:
    Rational(const T & numerator = 0, const T & denominator = 1);
    const T numerator() const;
    const T denominator() const;

    // ...
};
```
As a template, `doMultiply` **won’t** support mixed-mode multiplication, but it doesn’t need to. 
It will only be called by `operator*`, and `operator*` does support mixed-mode operations! 
In essence, the function `operator*` supports whatever type conversions are necessary 
to ensure that two `Rational` objects are being multiplied, 
then it passes these two objects to an appropriate instantiation of the `doMultiply` template 
to do the actual multiplication. 






### 📌 Item 47: Use traits classes for information about types

- Traits classes make information about types available during compilation.
  They're implemented using templates and template specializations.
- In conjunction with overloading (master API dispatching internal calls to workers with different traits flags), 
  traits classes make it possible to perform compile-time `if-else` tests on types.


### [`std::iterator_traits`](https://en.cppreference.com/w/cpp/iterator/iterator_traits)

C++ iterator types:
- Input iterator
  - Supports `++`, each position can be read _only once_. 
  - E.g., `std::istream_iterator`. 
- Output iterator
  - Supports `++`, each position can be written _only once_.
  - E.g., `std::ostream_iterator`.
- Forward iterator
  - Supports `++`, each position can be read/written for multiple times
  - E.g., iterators for `std::forward_list`, `std::unordered_(multi)set/map`
- Bidirectional iterator
  - Supports `++`, `--`, each position can be read/written for multiple times
  - E.g., iterators for `std::set`, `std::map`, `std::multiset`, `std::multimap`
- Random access iterator
  - Supports _iterator arithmetic_, each position can be read/written for multiple times
  - E.g., iterators for `std::vector`, `std::deque`, `std::string`


For each of the five iterator categories, C++ has a _tag struct_ in the standard library that serves to identify it:
```c++
namespace std
{

struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag : public input_iterator_tag {};
struct bidirectional_iterator_tag : public forward_iterator_tag {};
struct random_access_iterator_tag : public bidirectional_iterator_tag {};

}  // namespace std
```

_traits_ allow you to get information about a type during compilation.
Traits **aren’t** a keyword or a predefined construct in C++; 
they’re a technique and a convention followed by C++ programmers.
One of the demands made on the technique is that it has to work as well 
for built-in types as it does for user-defined types.


The fact that traits must work with built-in types means that 
things like nesting information inside types **won’t** work, 
because there’s no way to nest information inside built-in pointers. 
The traits information for a type must be external to the type. 
The standard technique is to put it into a template and one or more specializations of that template. 
For iterators, the template in the standard library is named `iterator_traits`:
```c++
/// <type_traits>
/// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
namespace std
{

template <typename ...>
using __void_t = void;

}  // namespace std

/// <bits/stl_iterator_base_types.h>
/// g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
namespace std
{

template <typename _Iterator, typename = __void_t<>>
struct __iterator_traits {};

template <typename _Iterator>
struct __iterator_traits<_Iterator,
        __void_t<typename _Iterator::iterator_category,
            typename _Iterator::value_type,
            typename _Iterator::difference_type,
            typename _Iterator::pointer,
            typename _Iterator::reference>>
{
    typedef typename _Iterator::iterator_category iterator_category;
    typedef typename _Iterator::value_type value_type;
    typedef typename _Iterator::difference_type difference_type;
    typedef typename _Iterator::pointer pointer;
    typedef typename _Iterator::reference reference;
};

template <typename _Iterator>
struct iterator_traits : public __iterator_traits<_Iterator> {};

// Partial specialization for pointer types.
template <typename _Tp>
struct iterator_traits<_Tp *>
{
    typedef random_access_iterator_tag iterator_category;
    typedef _Tp                        value_type;
    typedef ptrdiff_t                  difference_type;
    typedef _Tp *                      pointer;
    typedef _Tp &                      reference;
};

// Partial specialization for const pointer types.
template <typename _Tp>
struct iterator_traits<const _Tp *>
{
    typedef random_access_iterator_tag iterator_category;
    typedef _Tp                        value_type;
    typedef ptrdiff_t                  difference_type;
    typedef const _Tp *                pointer;
    typedef const _Tp &                reference;
};

}  // namespace std
```
By convention, traits are always implemented as structs.
Another convention is that the structs used to implement traits are known as traits _classes_.


The way `iterator_traits` works is that for each type `IterT`, 
a typedef named `iterator_category` is declared in the struct `iterator_traits<IterT>`. 
This typedef identifies the iterator category of `IterT`.


For user-defined types, it imposes the requirement that 
any user-defined iterator type must typedef a proper tag struct as `iterator_category`,
and `iterator_traits` just parrots back the typedef.
```c++
class SomeRandomAccessIterator
{
public:
    typedef std::random_access_iterator_tag iterator_category;
    // ...
};
```
For built-in pointers, there’s no such thing as a pointer with a nested typedef.
`std::iterator_traits` offers a _partial template specialization_ for pointer types. 


How to design and implement a traits class:
- Identify some information about types you’d like to make available
  (e.g., for iterators, their iterator category).
- Choose a name to identify that information (e.g., `std::iterator_category`).
- Provide a template and set of specializations (e.g., `std::iterator_traits`) 
  that contain the information for the types you want to support.


`If-else` test during compilation via function overloading with TMP:  
```c++
template <typename IterT, typename DistT>
void doAdvance(IterT & iter, DistT d, std::random_access_iterator_tag)
{
    iter += d;
}

template <typename IterT, typename DistT>
void doAdvance(IterT & iter, DistT d, std::bidirectional_iterator_tag)
{
    if (0 <= d)
    {
        while (d--)
        {
            ++iter;
        }
    }
    else
    {
        while (d++)
        {
            --iter;
        }
    }
}

template <typename IterT, typename DistT>
void doAdvance(IterT & iter, DistT d, std::input_iterator_tag)
{
    if (d < 0)
    {
        throw std::out_of_range("Negative distance");
    }
    while (d--)
    {
        ++iter;
    }
}


template <typename IterT, typename DistT>
void advance(IterT & iter, DistT d)
{
    doAdvance(iter, d, typename std::iterator_traits<IterT>::iterator_category());
}
```
How to use a traits class:
- Create a set of overloaded “worker” functions or function templates that differ in a traits parameter. 
  Implement each function in accord with the traits information passed.
- Create a “master” function or function template that calls the workers, 
  passing information provided by a traits class.


Commonly-used traits in C++ STL:
- `std::iterator_traits<IterT>`
- `std::char_traits<CharT>`
- `std::numeric_limits<NumericT>`
- `std::is_fundamental<T>`
- `std::is_array<T>`
- `std::is_base_of<T1, T2>`
- ...



### 📌 Item 48: Be aware of template metaprogramming (TMP)

- Template metaprogramming (TMP) can shift work from runtime to compile-time, 
  thus enabling earlier error detection and higher runtime performance. 
- TMP can be used to generate custom code based on combinations of policy choices, 
  and it can also be used to avoid generating code inappropriate for particular types. 


Template metaprogramming (TMP) is the process of 
writing template-based C++ programs that execute during compilation. 
A template metaprogram is a program written in C++ that executes inside the C++ compiler. 
When a template metaprogram finishes running, 
its output (pieces of C++ source code instantiated from templates) is then compiled as usual. 


C++ was not designed for template metaprogramming, 
but since TMP was discovered (**not** invented) in the early 1990s, it has proven to be so useful,
extensions are likely to be added to both the language and its standard library to make TMP easier.
The features underlying TMP were introduced when templates were added to C++. 
All that was needed was for somebody to notice how they could be used in clever and unexpected ways.


Two great strengths of TMP:
- It makes some things easy that would otherwise be hard or impossible;
- It shifts work from runtime to compile-time (because template metaprograms execute during C++ compilation);
    - Earlier error detection: 
        - Some kinds of errors that are usually detected at runtime can be found during compilation. 
    - Better runtime performance in just about every way: 
        - Smaller executables;
        - Shorter runtimes;
        - Less memory requirements.
    - Compilation takes _much_ longer.


TMP has been shown to be _Turing-complete_, which means that it is powerful enough to compute anything. 
Using TMP, you can declare variables, perform loops, write and call functions, etc. 
But such constructs look very different from their “normal” C++ counterparts. 
For example, Item 47 shows how `if-else` conditionals in TMP are expressed
via templates and template specializations. 
Another example: Loops are done in TMP via _recursive template instantiations_.


But that’s assembly-level TMP. 
Libraries for TMP 
(e.g., [Boost.MPL](https://www.boost.org/doc/libs/1_78_0/libs/mpl/doc/index.html)) 
offer a higher-level syntax, though still not something you’d mistake for “normal” C++.
```c++
template <unsigned long long N>
struct Binary
{
    // prepend higher bits to the lowest bit
    static constexpr unsigned value = (binary<N / 10>::value << 1) | (N % 10);
};

// specialization terminates recursion
template <>
struct Binary<0>
{
    static constexpr unsigned value = 0;
};

constexpr unsigned ONE = Binary<1>::value;
constexpr unsigned THREE = Binary<11>::value;
constexpr unsigned FIVE = Binary<101>::value;
constexpr unsigned SEVEN = Binary<111>::value;
constexpr unsigned NINE = Binary<1001>::value;
```
To grasp why TMP is worth knowing about, it’s important to have a better understanding of what it can accomplish. 
Here are three examples:
- **Ensuring dimensional unit correctness**. 
  In scientific and engineering applications, it’s essential that dimensional units 
  (e.g., mass, distance, time, etc.) be combined correctly. 
  Assigning a variable representing mass to a variable representing velocity, for example, is an error, 
  but dividing a distance variable by a time variable and assigning the result to a velocity variable is fine. 
  Using TMP, it’s possible to ensure (during compilation) that all dimensional unit combinations in a program are correct, 
  no matter how complex the calculations. 
  (This is an example of how TMP can be used for early error detection.) 
  One interesting aspect of this use of TMP is that fractional dimensional exponents can be supported. 
  This requires that such fractions be reduced _during compilation_ so that compilers can confirm, 
  for example, that the unit `time^{1/2}` is the same as `time^{4/8}`.
- **Optimizing matrix operations**. 
  Item 21 explains that some functions, including `operator*`, must return new objects, 
  and Item 44 introduces the `SquareMatrix` class, so consider the following code. 
  Calculating result in the “normal” way calls for the creation of four temporary matrices,
  one for the result of each call to `operator*`.
  Furthermore, the independent multiplications generate a sequence of four loops over the matrix elements.
  Using an advanced template technology related to TMP called _expression templates_,
  it’s possible to eliminate the temporaries and merge the loops,
  all without changing the syntax of the client code above.
  The resulting software uses less memory and runs dramatically faster.
```c++
using BigMatrix = SquareMatrix<double, 10000>;
BigMatrix m1, m2, m3, m4, m5;               // create matrices and give them values
BigMatrix result = m1 * m2 * m3 * m4 * m5;  // compute their product
```
- **Generating custom design pattern implementations**. 
  Design patterns like Strategy (see Item 35), Observer, Visitor, etc. 
  can be implemented in many ways. 
  Using a TMP-based technology called _policy-based design_, 
  it’s possible to create templates representing independent design choices (“policies”) 
  that can be combined in arbitrary ways to yield pattern implementations with custom behavior.
  For example, this technique has been used to allow a few templates implementing smart pointer behavioral policies 
  to generate (during compilation) any of hundreds of different smart pointer types. 
  Generalized beyond the domain of programming artifacts like design patterns and smart pointers, 
  this technology is a basis for what’s known as generative programming. 






### 🎯 Chapter 8. Customizing `new` and `delete`

### 📌 Item 49: Understand the behavior of the `new`-handler

- `set_new_handler` allows you to specify a function to be called
  when memory allocation requests cannot be satisfied.
- A new-handler function must do one of the following to end the endless loop inside `operator new`:
    - Make more memory available;
    - Install a different new-handler;
    - Uninstall the new-handler;
    - Throw a `std::bad_alloc` exception or something derived from it;
    - Not return.
- [Curiously Recurring Template Pattern (CRTP)](https://en.cppreference.com/w/cpp/language/crtp)
    - `class Derived : Base<Derived> { ... }`
- Nothrow `new` is of limited utility,
  because it applies only to memory allocation;
  associated constructor calls may still throw exceptions.


Before `operator new` throws an exception in response to an unsatisfiable request for memory, 
it calls a client-specifiable error-handling function called a _new-handler_. 
(This is **not** quite true. 
What _operator new_ really does is a bit more complicated. 
Details are provided in Item 51.) 
To specify the out-of-memory-handling function, clients call `std::set_new_handler`, 
a standard library function declared in `<new>`:
```c++
/// <new>
namespace std
{

typedef void (* new_handler)();

new_handler set_new_handler(new_handler p) throw();

}  // namespace std


/// client code
// function to call if operator new can’t allocate enough memory
void outOfMem()
{
    std::cerr << "Unable to satisfy request for memory\n";
    std::abort();
}

int main(int argc, char * argv[])
{
    std::set_new_handler(outOfMem);
    int * pBigDataArray = new int[100000000ULL];
    // ...
}
```
When operator new is unable to fulfill a memory request, 
it calls the new-handler function repeatedly until it can find enough memory. 
The code giving rise to these repeated calls is shown in Item 51, 
but this high-level description is enough to conclude that 
a well-designed new-handler function must do one of the following:
- **Make more memory available**. 
  This may allow the next memory allocation attempt inside `operator new` to succeed. 
  One way to implement this strategy is to allocate a large block of memory at program start-up, 
  then release it for use in the program the first time the new-handler is invoked.
- **Install a different new-handler**. 
  If the current new-handler can’t make any more memory available, 
  perhaps it knows of a different new-handler that can. 
  If so, the current new-handler can install the other new-handler in its place (by calling `std::set_new_handler`). 
  The next time `operator new` calls the new-handler function, 
  it will get the one most recently installed. 
  (A variation on this theme is for a new-handler to modify its own behavior, 
  so the next time it’s invoked, it does something different. 
  One way to achieve this is to have the new-handler modify 
  static, namespace-specific, or global data 
  that affects the new-handler’s behavior.)
- **Uninstall the new-handler**, 
  i.e., pass `nullptr` to `std::set_new_handler`. 
  With no new-handler installed, `operator new` will throw an exception when memory allocation is unsuccessful.
- **Throw an exception** 
  of type `std::bad_alloc` or something derived from it. 
  Such exceptions will not be caught by `operator new`, 
  so they will propagate to the site originating the request for memory.
- **Not return**, 
  typically by calling `std::abort` or exit.

Sometimes you’d like to handle memory allocation failures in different ways, 
depending on the class of the object being allocated: 
```c++
struct X
{
    static void outOfMemory();
    // ...
};

struct Y
{
    static void outOfMemory();
    // ...
};

X * p1 = new X; // if allocation is unsuccessful, call X::outOfMemory
Y * p2 = new Y; // if allocation is unsuccessful, call Y::outOfMemory
```
C++ **doesn’t** need any class-specific new-handler. 
You just have each class provide its own versions of `set_new_handler` and `operator new`. 
The class’s `set_new_handler` allows clients to specify the new-handler for the class 
(exactly like the standard `std::set_new_handler` allows clients to specify the global new-handler). 
The class’s `operator new` ensures that the class-specific new-handler is used 
in place of the global new-handler when memory for class objects is allocated.


Suppose you want to handle memory allocation failures for the `Widget` class. 
You’ll have to keep track of the function to call when `operator new` can’t allocate enough memory for a `Widget` object, 
so you’ll declare a static member of type `std::new_handler` to point to the new-handler function for the class. 
`Widget` will just inherit `EnableCustomNewHandler<Widget>`:
```c++
// "EnableCustomNewHandler.h"
template <typename T>
class EnableCustomNewHandler
{
public:
    static std::new_handler set_new_handler(std::new_handler p) noexcept
    {
        std::new_handler oldHandler = mNewHandler;
        mNewHandler = p;
        return oldHandler;
    }

    static void * operator new(std::size_t count)
    {
        std::new_handler oldHandler = std::set_new_handler(mNewHandler));
        void * ptr = ::operator new(count);  // can NOT use std::malloc here!
        std::set_new_handler(oldHandler);
        return ptr;
    }

private:
    static std::new_handler mNewHandler {nullptr};
};

// "Widget.h"
class Widget : public EnableCustomNewHandler<Widget>
{
    // ...
};
```
Finally, `Widget::operator new` will do the following: 
1. Call the standard `set_new_handler` with `Widget`’s error-handling function. 
   This installs `Widget`’s new-handler as the global new-handler.
2. Call the global operator new to perform the actual memory allocation.
   If allocation fails, the global operator new invokes `Widget`’s own version of new-handler, 
   because that function was just installed as the global new-handler. 
   If the global `operator new` is ultimately unable to allocate the memory, 
   it throws a `std::bad_alloc` exception. 
   In that case, `Widget::operator new` must restore the original global new-handler,
   then propagate the exception. 
   To ensure that the original new-handler is always reinstated, 
   `Widget` treats the global new-handler as a resource 
   and follows the advice of Item 13 
   to use resource-managing objects to prevent resource leaks.
4. If the global `operator new` was able to allocate enough memory for a `Widget` object, 
   `Widget::operator new` returns a pointer to the allocated memory. 
   The destructor for the object managing the global new-handler 
   automatically restores the global new-handler
   to what it was prior to the call to `Widget::operator new.`


Clients of `Widget` use its new-handling capabilities like this:
```c++
void outOfMem();

Widget::set_new_handler(outOfMem); 
Widget * pw1 = new Widget;
std::string * ps = new std::string;

Widget::set_new_handler(nullptr);
Widget * pw2 = new Widget;
```


But why `Widget` inherits from `EnableCustomNewHandler<Widget>`? 
This template `EnableCustomNewHandler` **never** uses its type parameter `T`. 
It doesn’t need to. 
All we need is a different copy of `EnableCustomNewHandler::mNewHandler` 
for each class that inherits from `EnableCustomNewHandler`. 
The template parameter `T` just _distinguishes one inheriting class from another_. 
The template mechanism itself automatically generates a copy of `mNewHandler` 
for each `T` with which `EnableCustomNewHandler` is instantiated.


As for `Widget` inheriting from a templatized base class that takes `Widget` as a type parameter, 
it has a name called the 
[Curiously Recurring Template Pattern (CRTP)](https://en.cppreference.com/w/cpp/language/crtp). 
```c++
// The Curiously Recurring Template Pattern (CRTP)
// Base form: Deriving a Base class template with Derived class as argument
template <typename T>
class Base
{
    // methods within Base can use template to access members of Derived
};

class Derived : public Base<Derived>
{
    // ...
};

// One common use: Singleton base class template
template <typename T>
class Singleton
{
public:
    Singleton(const Singleton &) = delete;
    Singleton & operator=(const Singleton &) = delete;
    
    Singleton(Singleton &&) = delete;
    Singleton & operator=(Singleton &&) = delete;
    
    ~Singleton() = default;
    
    T & getInstance()
    {
        static T instance;
        return instance;
    }

private:
    Singleton() = default;
};


class A : public Singleton<A>
{
    // ...
};
```
Until 1993, C++ required that `operator new` return `NULL` when it was unable to allocate the requested memory. 
`operator new` is now specified to throw a `std::bad_alloc` exception. 
_Placement `new`_ on `std::nothrow` objects (defined in the header `<new>`) 
offers the traditional failure-yields-`NULL` behavior (nothrow forms): 
```c++
class Widget { ... };
Widget * pw1 = new Widget;                 // throws std::bad_alloc if allocation fails
if (!pw1) ...                              // this test must fail
Widget * pw2 = new (std::nothrow) Widget;  // returns nullptr if allocation for the Widget fails
if (!pw2) ...                              // this test may succeed
```
Nothrow `new` offers a less compelling guarantee about exceptions than is initially apparent. 
In the expression `new (std::nothrow) Widget`, two things happen. 
First, the nothrow version of `operator new` is called to allocate enough memory for a `Widget` object. 
If that allocation fails, `operator new` returns `nullptr`, just as advertised. 
If it succeeds, however, the `Widget` constructor is called, and at that point, all bets are off. 
The `Widget` constructor can do whatever it likes. 
It might itself `new` up some memory, and if it does, it’s **not** constrained to use nothrow `new`.
Although the `operator new` call in `new (std::nothrow) Widget` won’t throw, then, the `Widget` constructor might. 
If it does, the exception will be propagated as usual. 
Using nothrow new guarantees only that `operator new` won’t throw, 
**not** that an expression like `new (std::nothrow) Widget` will never yield an exception. 
In all likelihood, you will never have a need for nothrow `new`.


Regardless of whether you use “normal” (i.e., exception-throwing) `new` or its somewhat stunted nothrow cousin, 
it’s important that you understand the behavior of the new-handler, because it’s used with both forms.






### 📌 Item 50: Understand when it makes sense to replace `new` and `delete`

- There are many valid reasons for writing custom versions of `new` and `delete`,
  including improving performance, debugging heap usage errors, and collecting heap usage information.
- A good memory manager takes account of a lot of details like alignment, thread safety, new-handlers, etc. 
  Do **not** attempt a custom memory manager unless you really have to. 


Three most-commonly-seen reasons for customizing `new` and `delete`
1. **To detect usage errors**. 
   - Failure to `delete` memory conjured up by `new` leads to memory leaks. 
     Using more than one `delete` on `new`-ed memory yields undefined behavior. 
     If `operator new` keeps a list of allocated addresses and `operator delete` removes addresses from the list, 
     it’s easy to detect such usage errors. 
   - Similarly, a variety of programming mistakes can lead to 
     _data overruns_ (writing beyond the end of an allocated block) and 
     _data underruns_ (writing prior to the beginning of an allocated block). 
     Custom `operator new` s can over-allocate blocks so there’s room to put known Byte patterns (“signatures”) 
     before and after the memory made available to clients. 
     `operator delete`s can check to see if the signatures are still intact. 
     If they’re not, an overrun or underrun occurred sometime during the life of the allocated block, 
     and `operator delete` can log that fact, along with the value of the offending pointer. 
2. **To improve efficiency**. 
   The versions of `operator new` and `operator delete` that ship with compilers are designed for general-purpose use. 
   They have to be acceptable for long-running programs (e.g., web servers), 
   but they also have to be acceptable for programs that execute for less than a second. 
   They have to handle series of requests for large blocks of memory, small blocks, and mixtures of the two. 
   They have to accommodate allocation patterns ranging 
   from the dynamic allocation of a few blocks that exist for the duration of the program 
   to the constant allocation and deallocation of a large number of short-lived objects. 
   They have to worry about heap fragmentation, 
   a process that, if unchecked, eventually leads to the inability to satisfy requests for large blocks of memory, 
   even when ample free memory is distributed across many small blocks.   
   Given the demands made on memory managers, it’s no surprise that 
   the `operator new`s and `operator delete`s that ship with compilers take a middle-of-the-road strategy. 
   They work reasonably well for everybody, but optimally for nobody. 
   If you have a good understanding of your program’s dynamic memory usage patterns, 
   you can often find that custom versions of `operator new` and `operator delete` outperform the default ones. 
   By “outperform,” I mean they run faster and require less memory. 
   For some (though by no means all) applications, replacing the stock `new` and `delete` with custom versions 
   is an easy way to pick up significant performance improvements.
3. **To collect usage statistics**. 
   Before heading down the path of writing custom `new`s and `delete`s, 
   it’s prudent to gather information about how your software uses its dynamic memory. 
   What is the distribution of allocated block sizes? 
   What is the distribution of their lifetimes? 
   Do they tend to be allocated and deallocated in FIFO (“first in, first out”) order, 
   LIFO (“last in, first out”) order, or something closer to random order? 
   Do the usage patterns change over time, 
   e.g., does your software have different allocation/deallocation patterns in different stages of execution? 
   What is the maximum amount of dynamically allocated memory in use at any one time (i.e., its “high watermark”)? 
   Custom versions of `operator new` and `operator delete` make it easy to collect this kind of information. 


In concept, writing a custom `operator new` is pretty easy. 
For example, here’s a quick first pass at a global `operator new` that 
facilitates the detection of underruns and overruns. 
There are a lot of little things **wrong** with it, 
but we’ll worry about those in a moment.
```c++
namespace
{

constexpr unsigned long long signature {0xDEADBEEFULL};

using Byte = unsigned char;

}  // namespace anonymous


// this code has several flaws — see below
void * operator new(std::size_t count)
{
    // increase size of request so 2 signatures will also fit inside
    std::size_t realSize = count + 2 * sizeof(unsigned long long);

    // call malloc to get the actual
    void * ptr = std::malloc(realSize);

    // maybe with a maximum number of tries other than endless loop?
    while (!ptr)
    {
        if (std::new_handler newHandler = std::get_new_handler())
        {
            (*newHandler)();
            ptr = std::malloc(realSize);
        }
        else
        {
            throw std::bad_alloc();
        }
    }
    
    // write signature into first and last parts of the memory
    *(static_cast<unsigned long long*>(ptr)) = signature;
    *(reinterpret_cast<unsigned long long*>(static_cast<Byte *>(ptr) + realSize - sizeof(unsigned long long))) = signature;
    
    // return a pointer to the memory just past the first signature
    return static_cast<Byte *>(ptr) + sizeof(unsigned long long);
}
```
**Alignment**.
Many computer architectures require that data of particular types be placed in memory at particular kinds of addresses.
For example, an architecture might require 
that pointers occur at addresses that are a multiple of four (i.e., be four-Byte aligned), 
or that `double`s must occur at addresses that are a multiple of eight (i.e., be eight-Byte aligned). 
Failure to follow such constraints could lead to hardware exceptions at runtime. 
Other architectures are more forgiving, though they may offer better performance if alignment preferences are satisfied. 
For example, `double`s may be aligned on any Byte boundary on the Intel x86 architecture, 
but access to them is a lot faster if they are eight-Byte aligned. 


Alignment is relevant here because C++ requires that 
all `operator new`s return pointers that are suitably aligned for any data type. 
`std::malloc` labors under the same requirement, 
so having `operator new` return a pointer it gets from `std::malloc` is safe. 
However, in `operator new` above, we’re not returning a pointer we got from `std::malloc`, 
we’re returning a pointer we got from `std::malloc` offset by the size of an `unsigned`. 
There is no guarantee that this is safe! 
If the client called `operator new` to get enough memory for a `double` 
(or, if we were writing `operator new[]` , an array of `double`s), 
and we were running on a machine 
where `int`s were four Bytes in size but `double`s were required to be eight-Byte aligned, 
we’d probably return a pointer with improper alignment. 
That might cause the program to crash. 
Or it might just cause it to run more slowly. 
Either way, it’s probably not what we had in mind.


Details like alignment are the kinds of things that 
distinguish professional-quality memory managers from ones thrown together 
by programmers distracted by the need to get on to other tasks. 
Writing a custom memory manager that almost works is pretty easy.
Writing one that works well is a lot harder.
As a general rule, **do not attempt a custom memory manager unless you have to**.


In many cases, you don’t have to. 
Some compilers have switches that enable debugging and logging functionality in their memory management functions.
Many commercial products can also replace the memory management functions that ship with compilers.
All you need do is to buy them and relink. 


Another option is open source memory managers like 
[Boost.Pool](https://www.boost.org/doc/libs/1_78_0/libs/pool/doc/html/index.html). 
Boost.Pool offers _memory pools_ for allocation of a large number of small objects. 
Many C++ books show the code for a high-performance small-object allocator, 
but they usually omit such pesky details as portability and alignment considerations, thread safety, etc. 
Real libraries tend to have code that’s a lot more robust. 
Even if you decide to write your own `new`s and `delete`s, 
looking at open source versions is likely to give you insights into the easy-to-overlook details 
that separate almost working from really working. 


The topic of this Item is to know when it can make sense to replace the default versions of `new` and `delete`, 
either globally or on a per-class basis. 
We’re now in a position to summarize when in more detail than we did before.
- **To detect usage errors**
- **To collect statistics about the use of dynamically allocated memory** 
- **To increase the speed of allocation and deallocation**. 
  General-purpose allocators are often (though not always) a lot slower than custom versions, 
  especially if the custom versions are designed for objects of a particular type. 
  Class-specific allocators are an example application of fixed-size allocators such as those offered by Boost.Pool. 
  If your application is single-threaded, but your compilers’ default memory management routines are thread-safe, 
  you may be able to win measurable speed improvements by writing thread-unsafe allocators. 
  Of course, before jumping to the conclusion that `operator new` and `operator delete` are worth speeding up, 
  be sure to profile your program to confirm that these functions are truly a bottleneck.
- **To reduce the space overhead of default memory management**. 
  General-purpose memory managers are often (though not always) not just slower than custom versions, 
  they often use more memory, too. 
  That’s because they often incur some overhead for each allocated block. 
  Allocators tuned for small objects (such as those in Boost.Pool) essentially eliminate such overhead.
- **To compensate for suboptimal alignment in the default allocator**. 
  As I mentioned earlier, it’s fastest to access `double`s on the x86 architecture when they are eight-Byte aligned. 
  Unfortunately, `operator new`s that ship with some compilers 
  **don’t** guarantee eight-Byte alignment for dynamic allocations of `double`s. 
  In such cases, replacing the default `operator new` with one that guarantees eight-Byte alignment 
  could yield big increases in program performance. 
- **To cluster related objects near one another**. 
  If you know that particular data structures are generally used together, 
  and you’d like to minimize the frequency of page faults when working on the data, 
  it can make sense to create a separate heap for the data structures, 
  so they are clustered together on as few pages as possible. 
  Placement versions of `new` and `delete` (see Item 52) can make it possible to achieve such clustering.
- **To obtain unconventional behavior**. 
  Sometimes you want `operator new`s and `operator delete`s to do something that the compiler-provided versions don’t offer. 
  For example, you might want to allocate and deallocate blocks in shared memory, 
  but have only a C API through which to manage that memory. 
  Writing custom versions of `new` and `delete` (probably placement versions) 
  would allow you to drape the C API in C++ clothing. 
  As another example, you might write a custom `operator delete` 
  that overwrites deallocated memory with zeros in order to increase the security of application data.






### 📌 Item 51: Adhere to convention when writing `new` and `delete`

- `operator new` should contain an infinite loop trying to allocate memory,
  should call the new-handler if it can't satisfy a memory request,
  and should handle zero-sized requests. 
  Class-specific versions should handle requests for children (blocks that are larger than expected)
  by forwarding `size != sizeof(Base)` requests to `::operator new`. 
- `operator delete` is `noexcept` and should do nothing if passed `nullptr`. 
  Class-specific versions should handle requests for children (blocks that are larger than expected)
  by forwarding `size != sizeof(Base)` requests to `::operator delete`.


Implementing a conformant `operator new` requires having the right return value, 
calling the new-handling function when insufficient memory is available (see Item 49), 
and being prepared to cope with requests for no memory. 
You’ll also want to avoid inadvertently hiding the “normal” form of `new`, 
though that’s more a class interface issue than an implementation requirement; 
it’s addressed in Item 52.


The return value part of `operator new` is easy. 
If you can supply the requested memory, you return a pointer to it. 
If you can’t, you follow the rule described in Item 49 and throw a `std::bad_alloc` exception.


It’s not quite that simple, however, because `operator new` actually tries to allocate memory more than once, 
calling the new-handling function after each failure. 
The assumption here is that the new-handling function might be able to do something to free up some memory. 
Only when the pointer to the new-handling function is `nullptr` does `operator new` throw an exception.


Curiously, C++ requires that `operator new` return a legitimate pointer even when zero bytes are requested. 
(Requiring this odd-sounding behavior simplifies things elsewhere in the language.) 
That being the case, pseudocode for a non-member `operator new` looks like this:
```c++
void * operator new(std::size_t count)
{
    if (count == 0)
    {
        // handle 0-byte requests by treating them as 1-byte requests
        count = 1;
    }

    while (true)
    {
        if (void * ptr = std::malloc(count))
        {
            return ptr;
        }
        else
        {
            if (std::new_handler newHandler = std::get_new_handler())
            {
                (*newHandler)();
            }
            else
            {
                throw std::bad_alloc();
            }
        }
    }
}
```
Item 49 remarks that `operator new` contains an infinite loop, 
and the code above shows that loop explicitly as `while (true)`. 
The only way out of the loop is for memory to be successfully allocated 
or for the new-handling function to do one of the things described in Item 49: 
- Make more memory available, 
- Install a different new-handler, 
- Uninstall the new-handler, 
- Throw an exception `std::bad_alloc` or something derived from it,
- Fail to return. 
It should now be clear why the new-handler must do one of those things. 
If it doesn’t, the loop inside `operator new` will never terminate.


Many people don’t realize that `operator new` member functions are inherited by derived classes. 
That can lead to some interesting complications. 
In the pseudocode for `operator new` above, notice that the function tries to allocate size bytes (unless size is zero). 
That makes perfect sense, because that’s the argument that was passed to the function. 
However, as Item 50 explains, one of the most common reasons for writing a custom memory manager 
is to optimize allocation for objects of a specific class, 
not for a class or any of its derived classes. 
That is, given an `operator new` for a class `X`, 
the behavior of that function is typically tuned for objects of size `sizeof(X)`, 
nothing larger and nothing smaller. 
Because of inheritance, however, it is possible that 
the `operator new` in a base class will be called to allocate memory for an object of a derived class:
```c++
class Base
{
public:
    static void * operator new(std::size_t count);
    // ...
};

class Derived : public Base
{
    // Derived doesn't declare operator new
};

// Calls Base::operator new!
Derived * p = new Derived;
```
If `Base::operator new` wasn’t designed to cope with this (and chances are that it wasn’t),
the best way for it to handle the situation is to 
dispatch calls requesting the “wrong” amount of memory to `::operator new`:
```c++
void * Base::operator new(std::size_t count)
{
    // If size is "wrong", have ::operator new handle the request
    if (count != sizeof(Base))
    {
        return ::operator new(count);
    }
    
    // Otherwise, handle the request for Base here
    // ...
}
```
**Test for zero sizes** 
is incorporated into `if (size != sizeof(Base))`. 
C++ works in some mysterious ways, and one of those
ways is to decree that all freestanding objects have non-zero size (see Item 39). 
By definition, `sizeof(Base)` can **never** be zero, so if `size` is zero,
the request will be forwarded to `::operator new`, 
and it will be `::operator new`’s responsibility to handle zero-size memory allocations.


If you’d like to control memory allocation for arrays on a per-class basis, 
you need to implement the _array new_ operator `operator new[]`. 
If you decide to write `operator new[]`, 
remember that all you’re doing is allocating a chunk of raw memory.
You can’t do anything to the as-yet-nonexistent objects in the array. 
In fact, you can’t even figure out how many objects will be in the array. 
First, you don’t know how big each object is. 
After all, a base class’s `operator new[]` might, through inheritance,
be called to allocate memory for an array of derived class objects, 
and derived class objects are usually bigger than base class objects.


Hence, you can’t assume inside `Base::operator new[]` that the size of each object going into the array is `sizeof(Base)`,
and that means you can’t assume that the number of objects in the array is `(bytesRequested) / sizeof(Base)`. 
Second, the `std::size_t` parameter passed to `operator new[]` may be for more memory than will be filled with objects,
because, as Item 16 explains, dynamically allocated arrays may include extra space to store the number of array elements.


For `operator delete`, things are simpler. 
About all you need to remember is that C++ guarantees it’s always safe to `delete nullptr`, 
so you need to honor that guarantee. 
Here’s pseudocode for a non-member `operator delete`:
```c++
void operator delete(void * ptr) noexcept
{
    if (ptr)
    {
        // deallocate the memory pointed to by ptr
    }
}
```
The member version of `operator delete` is simple, too, 
except you’ve got to be sure to check the size of what’s being deleted. 
Assuming your class-specific `operator new` forwards requests of the “wrong” size to `::operator new`, 
you’ve got to forward “wrongly sized” deletion requests to `::operator delete`:
```c++
class Base
{
public:
    static void * operator new(std::size_t count);
    static void operator delete(void * ptr, std::size_t count) noexcept;
    // ...
};

void Base::operator delete(void * ptr, std::size_t count) noexcept
{
    if (ptr)
    {
        if (count != sizeof(Base))
        {
            ::operator delete(ptr);
        }
        else
        {
            // deallocate ptr here
        }
    }
}
```
The `std::size_t` value C++ passes to `operator delete` may be incorrect 
if the object being deleted was derived from a base class lacking a virtual destructor. 
This is reason enough for making sure your base classes have virtual destructors, 
but Item 7 describes a second, arguably better reason. 
For now, simply note that if you omit virtual destructors in base classes, 
`operator delete` functions may not work correctly. 






### 📌 Item 52: Write both normal and placement `delete` if you write placement `new`

- When you write a placement version of `operator new`,
  be sure to write both the normal version and the corresponding placement version of `operator delete`. 
    - No normal version of `operator delete`: program ill-formed;
    - No corresponding placement `delete`: program may experience subtle, intermittent memory leaks.
- Placement `delete` is called _only if_ an exception arises from 
  a constructor call that’s coupled to a call to a placement `new`. 
  Applying `delete` to a pointer (such as `p` above) 
  **never** yields a call to a placement version of `delete`.
- When you declare placement versions of `new` and `delete`,
  be sure **not** to ~~unintentionally hide the normal versions of those functions~~.


When you write a new expression such as `Widget * p = new Widget;`,
two functions are called: 
one to `operator new` to allocate memory, 
a second to `Widget`’s default constructor. 


Suppose that the first call succeeds, but the second call results in an exception being thrown. 
In that case, the memory allocation performed in step 1 must be undone. 
Otherwise we’ll have a memory leak. 
Client code can’t deallocate the memory, 
because if the `Widget` constructor throws an exception, `p` is never assigned. 
There’d be no way for clients to get at the pointer to the memory that should be deallocated. 
The responsibility for undoing step 1 must therefore fall on the C++ runtime system.


The runtime system is happy to call the `operator delete` 
that corresponds to the version of `operator new` it called in step 1, 
but it can do that only if it knows which `operator delete` is the proper one to call. 
This isn’t an issue if you’re dealing with the versions of `new` and `delete` that have the normal signatures, 
because the normal `operator new` 
```c++
void * operator new(std::size_t)
```
corresponds to the normal `operator delete`
```c++
void operator delete(void *) noexcept;
void operator delete(void *, std::size_t) noexcept;
```
When you’re using only the normal forms of `new` and `delete`, 
the runtime system has no trouble finding the `delete` that knows how to undo what `new` did.
However, the which-`delete`-goes-with-this-`new` issue does arise 
when you start declaring non-normal forms of `operator new`: 
forms that take additional parameters.


For example, suppose you write a class-specific `operator new` 
that requires specification of a `std::ostream` to which allocation information should be logged, 
and you also write a normal class-specific `operator delete`:
```c++
class Widget
{
public:
    // WRONG!
    static void * operator new(std::size_t count, std::ostream & lout);
    static void operator delete(void * ptr, std::size_t size) noexcept;
    // ...
};
```
When an `operator new` function takes extra parameters (other than the mandatory `std::size_t` argument), 
that function is known as a _placement `new`_.
A particularly useful placement `new` is the one that takes a pointer specifying where an object should be constructed:
```c++
void * operator new(std::size_t count, void * ptr) noexcept;
```
Placement `new` is part of C++’s standard library, 
and you have access to it whenever you `#include <new>`. 
Placement `new` is used inside `std::vector` (together with `std::allocator`)
to create objects in the vector’s unused capacity.
Placement `new` is overloaded. 
Most of the time when people talk about placement `new`, 
they’re talking about this specific taking a single extra argument of type `void *`. 
Less commonly, they’re talking about any version that takes extra arguments. 
Context generally clears up any ambiguity, but it’s important to understand that the general term 
_placement `new`_ means any version of `operator new` taking extra arguments, 
because the phrase _placement `delete`_  (which we’ll encounter in a moment) derives directly from it.


But let’s get back to the declaration of the `Widget` class, 
the one whose design I said was problematic. 
The difficulty is that this class will give rise to subtle memory leaks. 
Consider this client code, which logs allocation information to `std::cerr` when dynamically creating a `Widget`:
```c++
// Call operator new, passing std::cerr as the std::ostream object. 
// This leaks memory if the Widget constructor throws an exception. 
Widget * p = new (std::cerr) Widget;
```
Once again, if memory allocation succeeds and the `Widget` constructor throws an exception, 
the runtime system is responsible for undoing the allocation that `operator new` performed. 
However, the runtime system can’t really understand how the called version of `operator new` works, 
so it can’t undo the allocation itself. 
Instead, the runtime system looks for a version of `operator delete` 
that takes the same number and types of extra arguments as `operator new`, 
and, if it finds it, that’s the one it calls. 
In this case, `operator new` takes an extra argument of type `std::ostream &`, 
so the corresponding operator delete would have this signature:
```c++
void operator delete(void *, std::ostream &) noexcept;
```
By analogy with placement versions of `new`, 
versions of `operator delete` that take extra parameters are known as _placement `delete`_ s. 
In this case, `Widget` declares no placement version of `operator delete`, 
so the runtime system doesn’t know how to undo what the call to placement `new` does. 
As a result, it does nothing. 
In this example, _no operator `delete` is called_ if the `Widget` constructor throws an exception! 

The rule is simple: 
if an `operator new` with extra parameters isn’t matched by an `operator delete` with the same extra parameters, 
no `operator delete` will be called if a memory allocation by the `new` needs to be undone. 
To eliminate the memory leak in the code above, 
`Widget` needs to declare a placement `delete` that corresponds to the logging placement `new`:
```c++
class Widget
{
public:
    static void * operator new(std::size_t count, std::ostream & lout);
    static void operator delete(void * ptr) noexcept;
    static void operator delete(void * ptr, std::ostream & lout) noexcept;
    // ...
};
```
With this change, if an exception is thrown from the `Widget` constructor
```c++
Widget * p = new (std::cerr) Widget;  // as before, but no leak this time
```
the corresponding placement `delete` is automatically invoked, 
and that allows `Widget` to ensure that no memory is leaked.


However, consider what happens if no exception is thrown (which will usually be the case) 
and we get to a `delete` in client code:
```c++
delete p;  // invokes the normal operator delete
```
As the comment indicates, this calls the normal `operator delete`, not the placement version. 
Placement `delete` is called _only if_ an exception arises from 
a constructor call that’s coupled to a call to a placement `new`. 
Applying `delete` to a pointer (such as `p` above) never yields a call to a placement version of `delete`. 


This means that to forestall all memory leaks associated with placement versions of `new`, 
you must provide both the normal `operator delete` and a placement version 
that takes the same extra arguments as operator `new` does. 


Incidentally, because member function names hide functions with the same names in outer scopes, 
you need to be careful to avoid having class-specific `new`s 
hide other `new`s (including the normal versions) that your clients expect. 
For example, if you have a base class that declares only a placement version of `operator new`, 
clients will find that the normal form of `new` is unavailable to them:
```c++
class Base
{
public:
    static void * operator new(std::size_t count, std::ostream & lout);
    // ...
};

Base * p1 = new Base;              // error, normal version of operator new is hidden
Base * p2 = new (std::cerr) Base;  // fine, call Base::operator new(std::size_t, std::ostream &)
```
Similarly, `operator new`s in derived classes hide both global and inherited versions of `operator new`:
```c++
class Derived : public Base
{
public:
    static void * operator new(std::size_t count);
    // ...
};

Derived * p1 = new (std::clog) Derived;  // error, Base’s placement new is hidden
Derived * p2 = new Derived;              // fine, calls Derived’s operator new
```
Item 33 discusses this kind of name hiding in considerable detail, 
but for purposes of writing memory allocation functions, 
what you need to remember is that by default, 
C++ offers the following forms of `operator new` at global scope:
```c++
void * operator new(std::size_t count);
void * operator new(std::size_t count, const std::nothrow_t & tag);
void * operator new(std::size_t count, void * ptr);
```
If you declare any `operator new`s in a class, you’ll hide all these standard forms. 
Unless you mean to prevent class clients from using these forms, 
be sure to make them available in addition to any custom `operator new` forms you create. 
For each `operator new` you make available, of course, be sure to offer the corresponding `operator delete` too. 
If you want these functions to behave in the usual way, 
just have your class-specific versions call the global versions.
An easy way to do this is to create a base class containing all the normal forms of `new` and `delete`:
```c++
class StandardNewDeleteForms
{
public:
    // normal new/delete
    static void * operator new(std::size_t count)
    {
        return ::operator new(count);
    }

    static void operator delete(void * ptr) noexcept
    {
        ::operator delete(ptr);
    }

    // placement new/delete
    static void * operator new(std::size_t count, void * ptr) noexcept
    {
        return ::operator new(count, ptr);
    }

    static void operator delete(void * ptr, void * place) noexcept
    {
        return ::operator delete(ptr, place);
    }

    // nothrow new/delete
    static void * operator new(std::size_t count, const std::nothrow_t & tag) noexcept
    {
        return ::operator new(count, tag);
    }

    static void operator delete(void * ptr, const std::nothrow_t &) noexcept
    {
        ::operator delete(ptr);
    }
};
```
Clients who want to augment the standard forms with custom forms 
can then just use inheritance and `using` declarations to get the standard forms:
```c++
class Widget : public StandardNewDeleteForms
{
public:
    using StandardNewDeleteForms::operator new;
    using StandardNewDeleteForms::operator delete;

    static void * operator new(std::size_t count, std::ostream & lout);
    static void operator delete(void * ptr, std::ostream & lout) noexcept;

    // ...
};
```






### 🎯 Chapter 9. Miscellany

### 📌 Item 53: Pay attention to compiler warnings

- Take compiler warnings seriously,
  and strive to compile warning-free at the maximum warning level supported by your compilers.
- **Don't** become dependent on compiler warnings, because different compilers warn about different things.
  Porting to a new compiler may eliminate warning messages you've come to rely on.






### 📌 Item 54: Familiarize yourself with the standard library, including TR1

- The primary standard C++ library functionality consists of the STL, iostreams, and locales.
  The C89 standard library is also included.
- ~~TR1 adds support for smart pointers (e.g., `tr1::shared_ptr`),
  generalized function pointers (`tr1::function`),
  hash-based containers, regular expressions, and 10 other components.~~
- ~~TR1 itself is only a specification.
  To take advantage of TR1, you need an implementation.
  One source for implementations of TR1 components is Boost.~~






### 📌 Item 55: Familiarize yourself with Boost

- Boost is a community and website for the development of free, open source, peer-reviewed C++ libraries.
  Boost plays an influential role in C++ standardization.
- Boost offers implementations of many TR1 components,
  but it also offers many other libraries, too.

