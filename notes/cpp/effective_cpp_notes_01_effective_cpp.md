# _Effective C++_ Notes

- Notes of reading <u>Scott Meyers</u>'s _Effective C++_ series:
    1. ***[`Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_01_effective_cpp.md)***
    2. *[`More Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_02_more_effective_cpp.md)*
    3. *[`Effective STL`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_03_effective_stl.md)*
    4. *[`Effective Modern C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_04_effective_modern_cpp.md)*






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
[static storage duration](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/cpp_primer_notes.md#-%E5%AD%98%E5%82%A8%E6%9C%9F%E5%92%8C%E9%93%BE%E6%8E%A5storage-duration-and-linkage) 
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

- Compilers may implicitly generate a class's default constructor, copy constructor, copy assignment operator, and destructor.

**OUTDATED**. Refer to _Effective Modern C++_ Item 17. 






### 📌 Item 6: Explicitly disallow the use of compiler-generated functions you do not want

- To disallow functionality automatically provided by compilers,
  declare the corresponding member functions private and give no implementations. 
  Using a base class like `Uncopyable` is one way to do this. 

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
class SpecialString: public std::string 
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

class Derived: public Base { ... };

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
    Bitmap * pb; // ptr to a heap-allocated object
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
  and all of its base class parts (by invoking base class's copying functions).
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

### 📌 Item 13: Use objects to manage resources

- To prevent resource leaks, use RAII objects
  that acquire resources in their constructors and release them in their destructors.
- ~~Two commonly useful RAII classes are `tr1::shared_ptr` and `std::auto_ptr`.
  `tr1::shared_ptr` is usually the better choice, because its behavior when copied is intuitive.
  Copying an `std::auto_ptr` sets it to null.~~ Refer to _Effective Modern C++_ Chapter 4 for details.






### 📌 Item 14: Think carefully about copying behavior in resource-managing classes

- Copying an RAII object entails copying the resource it manages,
  so the copying behavior of the resource determines the copying behavior of the RAII object.
- Common RAII class copying behaviors are disallowing copying and performing reference counting,
  but other behaviors are possible.






### 📌 Item 15: Provide access to raw resources in resource-managing classes

- APIs often require access to raw resources,
  so each RAII class should offer a way to get at the resource it manages.
- Access may be via explicit conversion or implicit conversion.
  In general, explicit conversion is safer,
  but implicit conversion is more convenient for clients.






### 📌 Item 16: Use the same form in corresponding uses of `new` and `delete`

- If you use `[]` in a `new` expression, you must use `[]` in the corresponding `delete` expression.
  If you don't use `[]` in a `new` expression, you mustn't use `[]` in the corresponding `delete` expression.






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
- `tr1::shared_ptr` supports custom deleters.
  This prevents the cross-DLL problem,
  can be used to automatically unlock mutexes (see Item 14), etc.






### 📌 Item 19: Treat class design as type design

- Class design is type design.
  Before defining a new type, be sure to consider all the issues discussed in this Item.






### 📌 Item 20: Prefer pass-by-reference-to-`const` to pass-by-value

- Prefer pass-by-reference-to-const over pass-by-value.
  It's typically more efficient and it avoids the slicing problem.
- The rule doesn't apply to built-in types and STL iterator and function object types.
  For them, pass-by-value is usually appropriate.






### 📌 Item 21: Don't try to return a reference when you must return an object

- **Never** return a pointer or reference to a local stack object,
  a reference to a heap-allocated object,
  or a pointer or reference to a local static object
  if there is a chance that more than one such object will be needed.
  (Item 4 provides an example of a design where returning a reference to a local static is reasonable,
  at least in single-threaded environments.)






### 📌 Item 22: Declare data members `private`

- Declare data members `private`.
  It gives clients syntactically uniform access to data,
  affords fine-grained access control,
  allows invariants to be enforced,
  and offers class authors implementation flexibility.
- `protected` is **no** more encapsulated than `public`.






### 📌 Item 23: Prefer non-member non-friend functions to member functions

- Prefer non-member non-friend functions to member functions.
  Doing so increases encapsulation, packaging flexibility, and functional extensibility.






### 📌 Item 24: Declare non-member functions when type conversions should apply to all parameters

- If you need type conversions on all parameters to a function
  (including the one that would otherwise be pointed to by the `this` pointer),
  the function must be a non-member.






### 📌 Item 25: Consider support for a non-throwing `swap`

- Provide a swap member function when `std::swap` would be inefficient for your type.
  Make sure your swap **doesn't** throw exceptions.
- If you offer a member `swap`, also offer a non-member `swap` that calls the member.
  For classes (not templates), specialize `std::swap`, too.
- When calling `swap`, employ a using declaration `using std::swap;`,
  then call `swap` **without** namespace qualification.
- It's fine to totally specialize `std` templates for user-defined types,
  but **never** try to add something completely new to `std`.






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






### 📌 Item 28: Avoid returning handles to object internals

- Avoid returning handles (references, pointers, or iterators) to object internals.
  Not returning handles increases encapsulation,
  helps const member functions act `const`,
  and minimizes the creation of dangling handles.






### 📌 Item 29: Strive for exception-safe code

- Exception-safe functions leak no resources
  and allow no data structures to become corrupted,
  even when exceptions are thrown.
  Such functions offer the basic, strong, or `nothrow` guarantees.
- The strong guarantee can often be implemented via copy-andswap,
  but the strong guarantee is not practical for all functions.
- A function can usually offer a guarantee **no** stronger
  than the weakest guarantee of the functions it calls.






### 📌 Item 30: Understand the ins and outs of inlining

- Limit most inlining to small, frequently called functions.
  This facilitates debugging and binary upgradability,
  minimizes potential code bloat,
  and maximizes the chances of greater program speed.
- **Don't** declare function templates `inline`
  just because they appear in header files.






### 📌 Item 31: Minimize compilation dependencies between files

- The general idea behind minimizing compilation dependencies
  is to depend on declarations instead of definitions.
  Two approaches based on this idea are Handle classes and Interface classes.
- Library header files should exist in full and declaration-only forms.
  This applies regardless of whether templates are involved.






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
- Pure virtual functions specify inheritance of interface only.
- Simple (impure) virtual functions specify inheritance of interface plus inheritance of a default implementation.
- Non-virtual functions specify inheritance of interface plus inheritance of a mandatory implementation.






### 📌 Item 35: Consider alternatives to virtual functions

- Alternatives to virtual functions include the NVI idiom and various forms of the Strategy design pattern.
  The NVI idiom is itself an example of the Template Method design pattern.
- A disadvantage of moving functionality from a member function to a function outside the class
  is that the non-member function lacks access to the class's non-public members.
- `tr1::function` objects act like generalized function pointers.
  Such objects support all callable entities compatible with a given target signature.






### 📌 Item 36: Never redefine an inherited non-virtual function

- Never redefine an inherited non-virtual function.






### 📌 Item 37: Never redefine a function's inherited default parameter value

- **Never** redefine an inherited default parameter value,
  because default parameter values are statically bound,
  while virtual functions (the only functions you should be overriding) are dynamically bound.






### 📌 Item 38: Model “has-a” or “is-implemented-in-terms-of” through composition

- Composition has meanings completely different from that of public inheritance.
- In the application domain, composition means has-a. In the implementation domain,
  it means is-implemented-in-terms-of.






### 📌 Item 39: Use private inheritance judiciously

- Private inheritance means is-implemented-in-terms of.
  It's usually inferior to composition,
  but it makes sense when a derived class needs access to protected base class members
  or needs to redefine inherited virtual functions.
- Unlike composition, private inheritance can enable the empty base optimization.
  This can be important for library developers who strive to minimize object sizes.






### 📌 Item 40: Use multiple inheritance judiciously

- Multiple inheritance is more complex than single inheritance.
  It can lead to new ambiguity issues and to the need for virtual inheritance.
- Virtual inheritance imposes costs in size, speed, and complexity of initialization and assignment.
  It's most practical when virtual base classes have no data.
- Multiple inheritance does have legitimate uses.
  One scenario involves combining public inheritance from an Interface class
  with private inheritance from a class that helps with implementation.






### 🎯 Chapter 7. Templates and Generic Programming

### 📌 Item 41: Understand implicit interfaces and compiletime polymorphism

- Both classes and templates support interfaces and polymorphism.
- For classes, interfaces are explicit and centered on function signatures.
  Polymorphism occurs at runtime through virtual functions.
- For template parameters, interfaces are implicit and based on valid expressions.
  Polymorphism occurs during compilation through template instantiation and function overloading resolution.






### 📌 Item 42: Understand the two meanings of `typename`

- When declaring template parameters, `class` and `typename` are interchangeable.
- Use `typename` to identify nested dependent type names,
  **except** in base class lists or as a base class identifier in a member initialization list.






### 📌 Item 43: Know how to access names in templatized base classes

- In derived class templates, refer to names in base class templates
  via a `this->` prefix,
  via using declarations,
  or via an explicit base class qualification.





### 📌 Item 44: Factor parameter-independent code out of templates

- Templates generate multiple classes and multiple functions,
  so any template code not dependent on a template parameter causes bloat.
- Bloat due to non-type template parameters can often be eliminated
  by replacing template parameters with function parameters or class data members.
- Bloat due to type parameters can be reduced
  by sharing implementations for instantiation types with identical binary representations.






### 📌 Item 45: Use member function templates to accept “all compatible types”

- Use member function templates to generate functions that accept all compatible types.
- If you declare member templates for generalized copy construction or generalized assignment,
  you'll still need to declare the normal copy constructor and copy assignment operator, too.






### 📌 Item 46: Define non-member functions inside templates when type conversions are desired

- When writing a class template that offers functions related to
  the template that support implicit type conversions on all parameters,
  define those functions as friends inside the class template.






### 📌 Item 47: Use traits classes for information about types

- Traits classes make information about types available during compilation.
  They're implemented using templates and template specializations.
- In conjunction with overloading, traits classes make it possible to perform compile-time `if`-`else` tests on types.






### 📌 Item 48: Be aware of template metaprogramming

- Template metaprogramming can shift work from runtime to compile-time,
  thus enabling earlier error detection and higher runtime performance.
- TMP can be used to generate custom code based on combinations of policy choices,
  and it can also be used to avoid generating code inappropriate for particular types.






### 🎯 Chapter 8. Customizing `new` and `delete`

### 📌 Item 49: Understand the behavior of the `new`-handler

- `set_new_handler` allows you to specify a function to be called
  when memory allocation requests cannot be satisfied.
- Nothrow `new` is of limited utility,
  because it applies only to memory allocation;
  associated constructor calls may still throw exceptions.






### 📌 Item 50: Understand when it makes sense to replace `new` and `delete`

- There are many valid reasons for writing custom versions of `new` and `delete`,
  including improving performance, debugging heap usage errors, and collecting heap usage information.






### 📌 Item 51: Adhere to convention when writing `new` and `delete`

- `operator new` should contain an infinite loop trying to allocate memory,
  should call the `new`-handler if it can't satisfy a memory request,
  and should handle requests for zero bytes.
  Class-specific versions should handle requests for larger blocks than expected.
- `operator delete` should do nothing if passed a pointer that is null.
  Class-specific versions should handle blocks that are larger than expected.






### 📌 Item 52: Write placement `delete` if you write placement `new`

- When you write a placement version of operator `new`,
  be sure to write the corresponding placement version of operator `delete`.
  If you don't, your program may experience subtle, intermittent memory leaks.
- When you declare placement versions of `new` and `delete`,
  be sure **not** to ~~unintentionally hide the normal versions of those functions~~.






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

- Boost is a community and web site for the development of free, open source, peer-reviewed C++ libraries.
  Boost plays an influential role in C++ standardization.
- Boost offers implementations of many TR1 components,
  but it also offers many other libraries, too.

