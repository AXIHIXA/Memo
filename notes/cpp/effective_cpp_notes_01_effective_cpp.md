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
    
- For simple constants, prefer `const` objects or `enum`s to `#define`s.
- For function-like macros, prefer `inline` functions to `#define`s.

#### The `enum` hack

For class-specific constants, use `enum`s instead of `static const` data members 
```c++
// GamePlayer.h
class GamePlayer 
{
private: 
    static const int NumTurns = 5;   // constant declaration & in-class initialization
    int scores[NumTurns];            // use of constant
};

// GamePlayer.cpp
const int GamePlayer::NumTurns;      // definition of NumTurns; see below for why no value is given
```
Usually, C++ requires that you provide a definition for anything you use, 
but class-specific constants that are `static` and of integral type 
(e.g., `int`s, `char`s, `bool`s) are an *exception*. 
As long as you don't *take their address*, 
you can declare them and use them without ~~providing a definition~~. 
If you do take the address of a class constant, 
or if your compiler incorrectly insists on a definition even if you don't take the address, 
you provide a separate definition in implementation file. 
<br><br>
Older compilers may not accept the syntax above, 
because it used to be ~~illegal to provide an initial value for a static class member at its point of declaration~~. 
Furthermore, in-class initialization is allowed only for *integral types* and only for *constants*. 
In cases where the above syntax can't be used, you put the initial value at the point of definition: 
```c++
// CostEstimate.h
class CostEstimate 
{
private:
    static const double FudgeFactor;            // declaration of static class constant
};

// CostEstimate.cpp
const double CostEstimate::FudgeFactor = 1.35;  // definition of static classconstant
```
The above block of code is all you need almost all the time. 
The only exception is when you *need the value of a class constant during compilation* of the class, 
such as in the declaration of the array `GamePlayer::scores` above 
(where compilers insist on knowing the size of the array during compilation). 
Then the accepted way to compensate for compilers that (incorrectly) forbid 
the in-class specification of initial values for static integral class constants 
is to use what is affectionately (and non-pejoratively) known as the `enum` hack. 
This technique takes advantage of the fact that the values of an enumerated type can be used where ints are expected, 
so `GamePlayer` could just as well be defined like this:
```c++
class GamePlayer 
{
private:
    enum {NumTurns = 5};   // "the enum hack" makes NumTurns a symbolic name for 5
    int scores[NumTurns];  // fine
};
```
The `enum` hack is worth knowing about for several reasons. 
- *Access Constraints*. <br>
  The `enum` hack behaves in some ways more like a `#define` than a `const` does, 
  and sometimes that's what you want. 
  For example, it's legal to take the address of a `const`, 
  but it's **not legal** to take the address of an `enum`, 
  and it's typically **not legal** to take the address of a `#define`, either. 
  If you don't want to let people get a pointer or reference to one of your integral constants, 
  an enum is a good way to enforce that constraint. 
- *Memory Allocation*. <br>
  Though good compilers won't set aside storage for const objects of integral types 
  (unless you create a pointer or reference to the object), 
  sloppy compilers may, and you may not be willing to set aside memory for such objects. 
  Like `#define`s, `enum`s never result in that kind of unnecessary memory allocation.
- *Pragmatic*. <br>
  Lots of code employs it, so you need to recognize it when you see it. 
  In fact, the `enum` hack is a fundamental technique of template metaprogramming. 

#### Common (mis)use of `#define` directives

Using it to implement macros that look like functions but that don't incur the overhead of a function call
```c++
// call f with the maximum of a and b
// even if everything is properly parenthesised, there can still be problems! 
#define CALL_WITH_MAX(a, b) f((a) > (b) ? (a) : (b))

int a = 5, b = 0;
CALL_WITH_MAX(++a, b);       // a is incremented twice
CALL_WITH_MAX(++a, b + 10);  // a is incremented once
```
You can get all the efficiency of a macro plus all the predictable behavior and type safety 
of a regular function by using a `template` for an `inline` function: 
```c++
// because we don't know what T is, we pass by reference-to-const
template <typename T> 
inline void callWithMax(const T & a, const T & b) 
{ 
    f(a > b ? a : b);
}
```
This template generates a whole family of functions, 
each of which takes two objects of the same type and calls `f` with the greater of the two objects. 
There's no need to parenthesize parameters inside the function body, 
no need to worry about evaluating parameters multiple times, etc. 
Furthermore, because callWithMax is a real function, it obeys scope and access rules. 
For example, it makes perfect sense to talk about an inline function that is private to a class. 
In general, there's just no way to do that with a macro.






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

Many people overlook the fact that *member functions differing only in their constness can be overloaded*, but this is an important feature of C++. 
Incidentally, const objects most often arise in real programs as a result of being passed by pointer-to-const or reference-to-const.
What does it mean for a member function to be const? 
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

- Manually initialize objects of built-in type, because C++ only
  sometimes initializes them itself.
- In a constructor, prefer use of the member initialization list to
  assignment inside the body of the constructor. List data
  members in the initialization list in the same order they're
  declared in the class.
- Avoid initialization order problems across translation units by
  replacing non-local static objects with local static objects.

#### Always initialize your objects before you use them

For non-member objects of built-in types, you'll need to do this manually: 
```c++
int x = 0;                               // manual initialization of an int
const char * text = "A C-style string";  // manual initialization of a pointer
double d;                                // "initialization" by reading from an input stream
std::cin >> d; 
```
For almost everything else, the responsibility for initialization falls on constructors. 
The rule there is simple: make sure that *all constructors initialize everything in the object*. 

#### **Not** to confuse assignment with initialization

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

#### Member initialization list

```c++
ABEntry::ABEntry(const std::string & name, 
                 const std::string & address, 
                 const std::list<PhoneNumber> & phones)
        : theName(name), 
          theAddress(address), 
          thePhones(phones), 
          numTimesConsulted(0)
{
    // the ctor body is now empty
}
```
This constructor yields the same end result as the one above, but it will often be more efficient. 
There's no need to do default initialization for class-type objects 
like `theName`, `theAddress`, and `thePhones` before entering the constructor body. 
For objects of built-in type like `numTimesConsulted`, 
there is no difference in cost between initialization and assignment, 
but for consistency, it's often best to initialize everything via member initialization.


Similarly, you can use the member initialization list even when you want to default-construct a data member; 
just specify nothing as an initialization argument. 
For example, if `ABEntry` had a constructor taking no parameters, it could be implemented like this:
```c++
ABEntry::ABEntry()
        : theName(),            // call theName's default ctor;
          theAddress(),         // do the same for theAddress;
          thePhones(),          // and for thePhones;
          numTimesConsulted(0)  // but explicitly initialize numTimesConsulted to zero
{

} 
```






### 🎯 Chapter 2. Constructors, Destructors, and Assignment Operators

### 📌 Item 5: Know what functions C++ silently writes and calls

- Compilers may implicitly generate a class's 
  default constructor, copy constructor, copy assignment operator, and destructor.



### 📌 Item 6: Explicitly disallow the use of compiler-generated functions you do not want

- To disallow functionality automatically provided by compilers,
  declare the corresponding member functions private and give no implementations. 
  Using a base class like `Uncopyable` is one way to do this. 




### 📌 Item 7: Declare destructors virtual in polymorphic base classes

- Polymorphic base classes should declare virtual destructors. 
  If a class has any virtual functions, it should have a virtual destructor.
- Classes not designed to be base classes or not designed to be used polymorphically 
  should **not** declare virtual destructors.






### 📌 Item 8: Prevent exceptions from leaving destructors

- Destructors should never emit exceptions. 
  If functions called in a destructor may throw, 
  the destructor should catch any exceptions, 
  then swallow them or terminate the program.
- If class clients need to be able to react to exceptions thrown during an operation, 
  the class should provide a regular (i.e., non-destructor) function that performs the operation.






### 📌 Item 9: Never call virtual functions during construction or destruction

- Don't call virtual functions during construction or destruction,
  because such calls will never go to a more derived class 
  than that of the currently executing constructor or destructor.






### 📌 Item 10: Have assignment operators return a reference to `*this`

- Have assignment operators return a reference to `*this`.






### 📌 Item 11: Handle assignment to self in `operator=`

- Make sure `operator=` is well-behaved when an object is assigned to itself. 
  Techniques include comparing addresses of source and target objects, 
  careful statement ordering, 
  and copy-and-`swap`.
- Make sure that any function operating on more than one object behaves correctly 
  if two or more of the objects are the same.






### 📌 Item 12: Copy all parts of an object

- Copying functions should be sure to copy all of an object's data members and all of its base class parts.
- Don't try to implement one of the copying functions in terms of the other. 
  Instead, put common functionality in a third function that both call.






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

