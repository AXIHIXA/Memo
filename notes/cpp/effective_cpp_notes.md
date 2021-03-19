# *`Effective C++`* Notes

- Notes of reading: 
    - *Effective C++ Digital Collection: 140 Ways to Improve Your Programming*
        - *Effective C++: 55 Specific Ways to Improve Your Programs and Designs*
        - *More Effective C++: 35 New Ways to Improve Your Programs and Designs*
        - *Effective STL: 50 Specific Ways to Improve Your Use of the Standard Template Library*
    - *Effective Modern C++: 42 Specific Ways to Improve Your Use of C++11 And C++14*






---

## 🌱 Effective C++: 55 Specific Ways to Improve Your Programs and Designs

## 1. Accustoming Yourself to C++

### 📌 Item 2: Prefer `const`s, `enum`s, and `inline`s to `#define`s
    
- For simple constants, prefer `const` objects or `enum`s to `#define`s.
- For function-like macros, prefer `inline` functions to `#define`s.

#### The `enum` hack

For class-specific constants, use `enum`s instead of `static const` data members 
```
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
```
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
```
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
```
// call f with the maximum of a and b
// even if everything is properly parenthesised, there can still be problems! 
#define CALL_WITH_MAX(a, b) f((a) > (b) ? (a) : (b))

int a = 5, b = 0;
CALL_WITH_MAX(++a, b);       // a is incremented twice
CALL_WITH_MAX(++a, b + 10);  // a is incremented once
```
You can get all the efficiency of a macro plus all the predictable behavior and type safety 
of a regular function by using a `template` for an `inline` function: 
```
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
```
std::vector<int> vec {0, 1, 2, 3, 4, 5};

const std::vector<int>::iterator iter = vec.begin();    // iter acts like a T * const
*iter = 10;                                             // OK, changes what iter points to
++iter;                                                 // error! iter is const

std::vector<int>::const_iterator cIter = vec.cbegin();  // cIter acts like a const T*
*cIter = 10;                                            // error! *cIter is const
++cIter;                                                // fine, changescIter
```

#### **`const`s in function return values

Having a function return a constant value often makes it possible to 
reduce the incidence of client errors without giving up safety or efficiency: 
```
class Rational { ... };
const Rational operator*(const Rational & lhs, const Rational & rhs);
```
Many programmers squint when they first see this. 
Why should the result of `operator*` be a `const` object? 
Because if it weren't, clients would be able to commit atrocities like this:
```
Rational a, b, c;
(a * b) = c;       // invoke operator= on the result of (a * b)!
```
I don't know why any programmer would want to make an assignment to the product of two numbers, 
but I do know that many programmers have tried to do it without wanting to. 
All it takes is a simple typo (and a type that can be implicitly converted to `bool`):
```
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
- *bitwise constness* (also known as *physical constness*) 
    The bitwise `const` camp believes that a member function is `const`
    iff. it doesn't modify any of the object's data members (excluding those that are `static`), 
    i.e., iff. it *doesn't modify any of the bits inside the object*. 
    The nice thing about bitwise constness is that it's easy to detect violations: 
    compilers just look for assignments to data members. 
    In fact, bitwise constness is C++'s definition of constness, 
    and a `const` member function isn't allowed to modify 
    any of the non-`static` data members of the object on which it is invoked. 
    <br><br>
    Unfortunately, many member functions that don't act very `const` pass the bitwise test. 
    In particular, *a member function that modifies what a pointer points* to frequently doesn't act `const`. 
    But if only the pointer is in the object, the function is bitwise `const`, and compilers won't complain. 
    That can lead to counterintuitive behavior. 
    For example, suppose we have a `TextBlock`-like class that stores its data as a `char *` instead of a `string`, 
    because it needs to communicate through a C API that doesn't understand `string` objects. 
    ```
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
    ```
    const CTextBlock cctb("Hello");  // declare constant object
    char *pc = &cctb[0];             // call the const operator[] to get a  pointer to cctb's data
    *pc = 'J';                       // cctb now has the value "Jello"
    ```
- *logical constness* <br>
    A `const` member function *might modify some of the bits in the object* on which it's invoked, 
    but *only in ways that clients cannot detect*. 
    For example, your `CTextBlock` class might want to cache the length of the textblock whenever it's requested:
    ```
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
    ```
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

```
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

- **Always initialize your objects before you use them** <br>
    For non-member objects of built-in types, you'll need to do this manually: 
    ```
    int x = 0;                               // manual initialization of an int
    const char * text = "A C-style string";  // manual initialization of a pointer
    double d;                                // "initialization" by reading from an input stream
    std::cin >> d; 
    ```
    For almost everything else, the responsibility for initialization falls on constructors. 
    The rule there is simple: make sure that *all constructors initialize everything in the object*. 
    - ***Not** to confuse assignment with initialization**. <br>
        Consider a constructor for a class representing entries in an address book:
        ```
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
        The rules of C++ stipulate that data members of an object are initialized *before* the body of a constructor is entered. 
        Inside the `ABEntry` constructor, `theName`, `theAddress`, and `thePhones` **aren't** being initialized, they're being *assigned*. 
        Initialization took place earlier: when their default constructors were automatically called prior to entering the body of the `ABEntry` constructor. 
        This isn't true for `numTimesConsulted`, because it's a built-in type. 
        For it, there's **no** guarantee it was initialized at all prior to its assignment.
    - *Member initialization list* <br>
        ```
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
        <br><br>
        Similarly, you can use the member initialization list even when you want to default-construct a data member; 
        just specify nothing as an initialization argument. 
        For example, if `ABEntry` had a constructor taking no parameters, it could be implemented like this:
        ```
        ABEntry::ABEntry()
                : theName(),            // call theName's default ctor;
                  theAddress(),         // do the same for theAddress;
                  thePhones(),          // and for thePhones;
                  numTimesConsulted(0)  // but explicitly initialize numTimesConsulted to zero
        {
        
        } 
        ```
        
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 






























---

## 🌱 More Effective C++: 35 New Ways to Improve Your Programs and Designs

### 📌 Item 1


### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 





















---

## 🌱 Effective STL: 50 Specific Ways to Improve Your Use of the Standard Template Library

### 📌 Item 1


### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
### 📌 
















--- 

## 🌱 Effective Modern C++: 42 Specific Ways to Improve Your Use of C++11 And C++14

## [CHAPTER 1] Deducing Types

### 📌 Item 1: Understand template type deduction

- During template type deduction, arguments' reference-ness and top-level cv-constraints are ignored.
- When deducing types for universal reference parameters, reference collapse may occur. 
- During template type deduction, arguments that are array or function names decay to pointers, unless they’re used to initialize references. 


If you’re willing to overlook a pinch of pseudocode, we can think of a function template as looking like this:
```
template <typename T>
void f(ParamType param);

f(expr);  // call f with some expression
```
During compilation, compilers use expr to deduce two types: one for `T` and one for `ParamType`. 
These types are frequently **different**, because `ParamType` often contains adornments, e.g., `const` or reference qualifiers. 
E.g., for the following case, `T` is deduced to be `int`, but `ParamType` is deduced to be `const int &`. 
```
template <typename T>
void f(const T & param);  // ParamType is const T &

int x = 0;
f(x);                     // call f with an int
```
The type deduced for `T` is **not** always the same as the type of the argument passed to the function, i.e., that `T` is the type of `expr`. 
Because the type deduced for `T` is dependent not just on the type of `expr`, but also on the form of `ParamType`. 
There are three cases: 
- `ParamType` is a *pointer* or *non-universal reference* type 
    - Universal references are described in Item 24. 
      At this point, all you need to know is that they exist 
      and that they’re not the same as lvalue references or rvalue references. 
    - Workflow: 
        - If `expr`’s type is a reference, ignore the reference part.
        - Then pattern-match `expr`’s type against `ParamType` to determine `T`. 
    - For example: 
    ```
    template <typename T>
    void f(T & param);     // param is a reference

    int x = 27;            // x is an int
    const int cx = x;      // cx is a const int
    const int & rx = x;    // rx is a reference to x as a const int
    ```
    The deduced types for `param` and `T` in various calls are as follows:
    ```
    f(x);                  // T is       int, param's type is int &
    f(cx);                 // T is const int, param's type is const int &
    f(rx);                 // T is const int, param's type is const int &
    ```
    Passing a const object to a template taking a `T &` parameter is safe:
    the constness of the object becomes part of the type deduced for `T`. 
    
    
    These examples all show *lvalue reference* parameters, but type deduction works exactly the same way for *rvalue reference* parameters. 
    Of course, only rvalue arguments may be passed to rvalue reference parameters, but that restriction has nothing to do with type deduction. 
    
    
    If we change the type of `f`’s parameter from `T &` to `const T &`, 
    things change a little, but not in any really surprising ways. 
    The constness of `cx` and `rx` continues to be respected,
    but because we’re now assuming that `param` is a reference-to-const, 
    there’s no longer a need for `const` to be deduced as part of `T`:
    ```
    template <typename T>
    void f(const T & param);  // param is now a ref-to-const
    
    int x = 27;               // as before
    const int cx = x;         // as before
    const int& rx = x;        // as before
    
    f(x);                     // T is int, param's type is const int &
    f(cx);                    // T is int, param's type is const int &
    f(rx);                    // T is int, param's type is const int &
    ```
    If `param` were a pointer (or a pointer to const) instead of a reference, things would work essentially the same way: 
    ```
    template <typename T>
    void f(T * param);        // param is now a pointer
    
    int x = 27;               // as before
    const int * px = &x;      // px is a ptr to x as a const int
    
    f(&x);                    // T is       int, param's type is int *
    f(px);                    // T is const int, param's type is const int *
    ```
    By now, you may find yourself yawning and nodding off, 
    because C++’s type deduction rules work so naturally for reference and pointer parameters, 
    seeing them in written form is really dull. 
    Everything’s just obvious! 
    *Which is exactly what you want in a type deduction system*. 
- `ParamType` is a *universal reference* <br>
    Such parameters are declared like rvalue references 
    (i.e., in a function template taking a type parameter `T`, a universal reference’s declared type is `T &&`), 
    but they behave differently when lvalue arguments are passed in. 
    The complete story is told in Item 24, but here’s the headline version: 
    - If `expr` is an *lvalue*, both `T` and `ParamType` are deduced (*collapse*) to be *lvalue references*; 
    - If `expr` is an *rvalue*, the normal (i.e., Case 1) rules apply.
    - For example:
    ```
    template <typename T>
    void f(T && param);    // param is now a universal reference
    
    int x = 27;            // as before
    const int cx = x;      // as before
    const int & rx = x;    // as before
    
    f(x);                  //  x is lvalue, so T is       int &, param's type is       int &
    f(cx);                 // cx is lvalue, so T is const int &, param's type is const int &
    f(rx);                 // rx is lvalue, so T is const int &, param's type is const int &
    f(27);                 // 27 is rvalue, so T is       int  , param's type is       int &&
    ```
- `ParamType` is neither a ~~pointer~~ nor a ~~reference~~ <br>
    We’re dealing with pass-by-value:
    ```
    template<typename T>
    void f(T param);       // param is now passed by value
    ```
    That means that `param` will be a *copy* of whatever is passed in: a completely new object. 
    The fact that `param` will be a new object motivates the rules that govern how `T` is deduced from `expr`:
    - Remove reference-ness and top-level cv-constraints (top-level const-ness and/or volatile-ness)
        - `volatile` objects are uncommon. They’re generally used only for implementing device drivers. For details, see Item 40.
        - This is because reference-ness and top-level cv-constraints are **ignored** during parameter type deduction.
    - For example: 
    ```
    int x = 27;            // as before
    const int cx = x;      // as before
    const int & rx = x;    // as before
    
    f(x);                  // T's and param's types are both int
    f(cx);                 // T's and param's types are again both int
    f(rx);                 // T's and param's types are still both int
    ```
    Note that even though `cx` and `rx` represent `const` values, param isn’t `const`. 
    That makes sense. 
    `param` is an object that’s completely independent of `cx` and `rx`: a copy of `cx` or `rx`. 
    The fact that `cx` and `rx` can’t be modified says nothing about whether `param` can be. 
    That’s why `expr`’s const-ness (and volatile-ness, if any) is ignored when deducing a type for `param`: 
    just because `expr` can’t be modified doesn’t mean that a copy of it can’t be. 
    
    
    It’s important to recognize that only *top-level cv-constraints* are ignored. 
    *Low-level cv-constraints* are preserved properly. 
    That is,  `const` (and `volatile`) is ignored only for by-value parameters. 
    As we’ve seen, for parameters that are references-to-const or pointers-to-const, 
    the constness of `expr` is preserved during type deduction.  
    Consider the case where `expr` is a `const` pointer to a `const` object, and `expr` is passed to a by-value `param`: 
    ```
    template <typename T>
    void f(T param);                               // param is still passed by value
    
    const char * const ptr = "Fun with pointers";  // ptr is const pointer to const object
    f(ptr);                                        // pass arg of type const char * const
    ```
    In this case, `T` is deducted to `const char *`.

#### Array Arguments

In many contexts, an array decays into a pointer to its first element. 
This decay is what permits code like this to compile:
```
const char name[] = "J. P. Briggs";  // name's type is const char[13]
const char * ptrToName = name;       // array decays to pointer
```
Here, the `const char *` pointer `ptrToName` is being initialized with `name`, which is a `const char[13]`. 
These types (`const char *` and `const char[13]`) are **not** the same, but because of the array-to-pointer decay rule, the code compiles. 


But what if an array is passed to a template taking a by-value parameter? What happens then?
```
template <typename T>
void f(T param);                     // template with by-value parameter

f(name);                             // what types are deduced for T and param?
```
We begin with the observation that there is no such thing as a function parameter that’s an array. 
In parameter lists, an array declaration is treated as a pointer declaration: 
```
void myFunc(int param[]);
void myFunc(int * param);            // same function as above
```
Because array parameter declarations are treated as if they were pointer parameters,
the type of an array that’s passed to a template function by value is deduced to be a pointer type. 
That means that in the call to the template `f`, its type parameter `T` is deduced to be `const char *`. 


But now comes a curve ball. 
Although functions can’t declare parameters that are truly arrays, 
they can declare parameters that are *references to arrays*! 
So if we modify the template `f` to take its argument by reference, 
```
template <typename T>
void f(T & param);                   // template with by-reference parameter

f(name);                             // what types are deduced for T and param?
```
the type deduced for `T` is the *actual type of the array*! 
That type includes the size of the array, so in this example, 
`T` is deduced to be `const char [13]`, 
and the type of `f`’s parameter (a reference to this array) is `const char (&)[13]`. 


Interestingly, the ability to declare references to arrays enables creation of a template
that deduces the number of elements that an array contains:
```
// return size of an array as a compile-time constant. 
// (The array parameter has no name, 
// because we care only about the number of elements it contains.)
template <typename T, std::size_t N> 
constexpr std::size_t arraySize(T (&)[N]) noexcept
{
    return N;
}
```
As Item 15 explains, declaring this function `constexpr` makes its result available during compilation. 
That makes it possible to declare, say, an array 
with the same number of elements as a second array whose size is computed from a *braced initializer*: 
```
int keyVals[] = {1, 3, 7, 9, 11, 22, 35};        // keyVals has 7 elements
int mappedVals[arraySize(keyVals)];              // so does mappedVals
```
Of course, as a modern C++ developer, you’d naturally prefer a `std::array` to a built-in array:
```
std::array<int, arraySize(keyVals)> mappedVals;  // mappedVals' size is 7
```
As for `arraySize` being declared `noexcept`, that’s to help compilers generate better code. For details, see Item 14. 

#### Function Arguments

Function types can decay into function pointers, 
and everything we’ve discussed regarding type deduction for arrays 
applies to type deduction for functions and their decay into function pointers. 
As a result:
```
void someFunc(int, double);  // someFunc is a function; type is void(int, double)

template <typename T>
void f1(T param);            // in f1, param passed by value

template<typename T>
void f2(T & param);          // in f2, param passed by ref

f1(someFunc);                // param deduced as ptr-to-func; type is void (*)(int, double)
f2(someFunc);                // param deduced as ref-to-func; type is void (&)(int, double)
```






### 📌 Item 2: Understand `auto` type deduction

- `auto` type deduction is usually the same as template type deduction, 
  but `auto` type deduction assumes that a *braced initializer* represents a `std::initializer_list`, 
  and template type deduction **doesn’t**.
- `auto` in a *function return type* or a *lambda parameter* implies *template type deduction*, **not** ~~auto type deduction~~.


With only one curious exception, `auto` type deduction is template type deduction. 
There’s a direct mapping between template type deduction and `auto` type deduction. 
There is literally an algorithmic transformation from one to the other. 


In Item 1, template type deduction is explained using this general function template: 
```
template <typename T>
void f(ParamType param);

f(expr);  // call f with some expression
```
In the call to `f`, compilers use `expr` to deduce types for `T` and `ParamType`. 


When a variable is declared using `auto`, 
`auto` plays the role of `T` in the template, 
and the type specifier for the variable acts as `ParamType`. 
This is easier to show than to describe, so consider this example:
```
auto x = 27;
const auto cx = x;
const auto & rx = x;
```
To deduce types for `x`, `cx`, and `rx` in these examples,
compilers act as if there were a template for each declaration 
as well as a call to that template with the corresponding initializing expression:
```
template <typename T>
void func_for_x(T param);

func_for_x(27);

template <typename T> 
void func_for_cx(const T param); 

func_for_cx(x); 


template <typename T> 
void func_for_rx(const T & param);

func_for_rx(x);
```
As I said, deducing types for `auto` is, 
with only one exception (which we’ll discuss soon), 
the same as deducing types for templates. 


In a variable declaration using `auto`, 
the type specifier takes the place of `ParamType`, 
so there are three cases for that, too:
- Case 1: The type specifier is a pointer or reference, but not a universal reference. 
- Case 2: The type specifier is a universal reference. 
- Case 3: The type specifier is neither a pointer nor a reference. 


Array and function names also decay into pointers for non-reference type specifiers in `auto` type deduction:
```
const char name[] = "R. N. Briggs";  // name's type is const char[13]

auto arr1 = name;                    // arr1's type is const char *
auto & arr2 = name;                  // arr2's type is const char (&)[13]

void someFunc(int, double);          // someFunc is a function; type is void(int, double)

auto func1 = someFunc;               // func1's type is void (*)(int, double)
auto & func2 = someFunc;             // func2's type is void (&)(int, double)
```

#### `auto` with Braced Initializer

One way that `auto` type deduction differs from template type deduction: 
```
// legacy initialization since C++98
auto x1 = 27;    // type is int, value is 27
auto x2(27);     // type is int, value is 27

// uniform initialization since C++11
auto x3 = {27};  // type is std::initializer_list<int>, value is {27}
auto x4 {27};    // type is std::initializer_list<int>, value is {27}
```
This is due to a special type deduction rule for `auto`. 
When the initializer for an auto-declared variable is enclosed in braces, 
the deduced type is a `std::initializer_list`. 
If such a type can’t be deduced 
(e.g., because the values in the braced initializer are of different types), 
the code will be rejected:
```
auto x5 = {1, 2, 3.0};  // error! can't deduce T for std::initializer_list<T>
```
As the comment indicates, type deduction will fail in this case, 
but it’s important to recognize that there are actually *two kinds of type deduction* taking place. 
One kind stems from the use of auto: `x5`’s type has to be deduced. 
Because `x5`’s initializer is in braces, `x5` must be deduced to be a `std::initializer_list`. 
But `std::initializer_list` is a *template*. 
Instantiations are `std::initializer_list<T>` for some type `T`, and that means that `T`’s type must also be deduced. 
Such deduction falls under the purview of the second kind of type deduction occurring here: 
*template type deduction*. 
In this example, that deduction fails, 
because the values in the braced initializer don’t have a single type, 
just like the following example for template functions: 
```
template <typename T>
T add(T x1, T x2)
{
    return x1 + x2;
}

add(1, 2.0);           // error! dedeced conflict types for parameter 'T' ('int' vs. 'double')
```
The treatment of braced initializers is the only way 
in which `auto` type deduction and template type deduction differ. 
When an auto-declared variable is initialized with a braced initializer, 
the deduced type is an instantiation of `std::initializer_list`.
But if the corresponding template is passed the same initializer, type deduction fails, and the code is rejected:
```
auto x = {11, 23, 9};  // x's type is std::initializer_list<int>

template<typename T>   // template with parameter declaration equivalent to x's declaration
void f(T param); 

f({11, 23, 9});        // error! can't deduce type for T
```
However, if you specify in the template that param is a `std::initializer_list<T>` for some unknown `T`, 
template type deduction will deduce what `T` is:
```
template <typename T>
void f(std::initializer_list<T> initList);

f({11, 23, 9});        // T deduced as int, and initList's type is std::initializer_list<int>
```
So the only real difference between `auto` and template type deduction is that 
`auto` assumes that a braced initializer represents a `std::initializer_list`, 
but template type deduction doesn’t.


For C++11, this is the full story, but for C++14, the tale continues. 
C++14 permits `auto` to indicate that a *function’s return type* should be deduced (see Item 3), 
and C++14 lambdas may use `auto` in parameter declarations. 
However, these uses of `auto` employ template type deduction, **not** `auto` type deduction. 
So a function with an `auto` return type that returns a braced initializer **won’t** compile:
```
auto createInitList()
{
    return {1, 2, 3};  // error: can't deduce type
}
```
The same is true when `auto` is used in a parameter type specification in a C++14 lambda:
```
std::vector<int> v;

auto resetV = [&v](const auto & newValue) 
{
    v = newValue;
};

resetV({1, 2, 3});     // error! can't deduce type for {1, 2, 3}
```






### 📌 Item 3: Understand `decltype`

- `decltype` almost always yields the type of a variable or expression without any modifications.
- For lvalue expressions of type `T` other than names, `decltype` always reports a type of `T &`.
- C++14 supports `decltype(auto)`, which, like `auto`, deduces a type from its initializer, but it performs the type deduction using the `decltype` rules.


In contrast to what happens during type deduction for templates and `auto` (see Items 1 and 2),
`decltype` typically parrots back the exact type of the name or expression you give it:
```
const int i = 0;           // decltype(i) is const int
bool f(const Widget & w);  // decltype(w) is const Widget &
                           // decltype(f) is bool(const Widget &)

struct Point 
{
int x; 
int y;                     // decltype(Point::x) is int
};                         // decltype(Point::y) is int

Widget w;                  // decltype(w) is Widget

if (f(w)) {}               // decltype(f(w)) is bool

template<typename T>       // simplified version of std::vector
class vector
{
public:
    T & operator[](std::size_t index);
};

vector<int> v;             // decltype(v) is vector<int>

if (v[0] == 0) {}          // decltype(v[0]) is int &
```
In C++11, perhaps the primary use for `decltype` is declaring function templates
where the function’s return type depends on its parameter types. 


`operator[]` on a container of objects of type `T` typically returns a `T &`. 
This is the case for `std::deque`, for example, and it’s almost always the case for `std::vector`. 
For `std::vector<bool>`, however, `operator[]` does **not** ~~return a `bool &`~~. 
Instead, it returns a brand new object. The whys and hows of this situation are explored in Item 6, 
but what’s important here is that the type returned by a container’s `operator[]` depends on the container. 


`decltype` makes it easy to express that. 
Here’s a first cut at the template we’d like to write, 
showing the use of `decltype` to compute the return type. 
```
template <typename Container, typename Index>
auto authAndAccess(Container & c, Index i) -> decltype(c[i])  // C++11; needs refinement if C++14
{
    authenticateUser();
    return c[i];
}
```
The use of `auto` in functions with *trailing return type* has nothing to do with type deduction.
A *trailing return type* has the advantage that 
the function’s parameters can be used in the specification of the return type. 


C++11 permits return types for *single-statement lambdas* to be deduced, 
and C++14 extends this to both *all lambdas* and *all functions*, 
including those with multiple statements. 
In the case of `authAndAccess`, 
that means that in C++14 we can omit the trailing return type, leaving just the leading `auto`. 
With that form of declaration, `auto` does mean that type deduction will take place. 
In particular, it means that compilers will deduce the function’s return type from the function’s implementation:
```
template <typename Container, typename Index>
auto authAndAccess(Container & c, Index i)  // C++14; NOT quite correct
{
    authenticateUser();
    return c[i];                            // return type deduced from c[i]
}
```
Item 2 explains that for functions with an `auto` return type specification, 
compilers employ template type deduction. 
In this case, that’s problematic. 
As we’ve discussed, `operator[]` for most containers-of-`T` returns a `T &`, 
but Item 1 explains that during template type deduction, 
the reference-ness of an initializing expression is ignored.
Consider what that means for this client code:
```
std::deque<int> d;
authAndAccess(d, 5) = 10;  // authenticate user, return d[5], then assign 10 to it; this won't compile!
```
Here, `d[5]` returns an `int &`, 
but `auto` return type deduction for `authAndAccess` will strip off the reference, 
thus yielding a return type of `int`. 
That `int`, being the return value of a function, is an rvalue, 
and the code above thus attempts to assign `10` to an rvalue `int`. 
That’s forbidden in C++, so the code won’t compile.

#### `decltype(auto)` Specifier: `auto` Type Deduction Using `decltype` Deduction Rule

To get `authAndAccess` to work as we’d like, 
we need to use `decltype` type deduction for its return type, 
i.e., to specify that `authAndAccess` should return *exactly the same type* that the expression `c[i]` returns. 
The guardians of C++, anticipating the need to use `decltype` type deduction rules in some cases where types are inferred, 
make this possible in C++14 through the *`decltype(auto)` specifier*. 
What may initially seem contradictory (`decltype` and `auto`?) actually makes perfect sense: 
`auto` specifies that the type is to be deduced, and `decltype` says that `decltype` rules should be used during the deduction. 
We can thus write `authAndAccess` like this:
```
template <typename Container, typename Index>
decltype(auto) authAndAccess(Container && c, Index i)                                   // C++14
{
    authenticateUser();
    return std::forward<Container>(c)[i];
}

template <typename Container, typename Index>
auto authAndAccess(Container && c, Index i) -> decltype(std::forward<Container>(c)[i])  // C++11
{
    authenticateUser();
    return std::forward<Container>(c)[i];
}
```
Now authAndAccess will truly return whatever `c[i]` returns. 
In particular, for the common case where `c[i]` returns a `T &`, 
`authAndAccess` will also return a `T &`, 
and in the uncommon case where `c[i]` returns an object, 
`authAndAccess` will return an object, too.
The use of `decltype(auto)` is not limited to function return types. 
It can also be convenient for declaring variables 
when you want to apply the `decltype` type deduction rules to the initializing expression:
```
Widget w;
const Widget & cw = w;
auto myWidget1 = cw;            //     auto type deduction: myWidget1's type is       Widget
decltype(auto) myWidget2 = cw;  // decltype type deduction: myWidget2's type is const Widget &
```

#### `decltype((x))`: Enforce lvalue Reference-ness on Reported Type

Applying `decltype` to a name yields the declared type for that name. 
(By the way, names are lvalue expressions, but that doesn’t affect `decltype`’s behavior. )
For lvalue expressions more complicated than names, however, 
`decltype` ensures that the type reported is always an lvalue reference. 
That is, if an lvalue expression other than a name has type `T`, 
`decltype` reports that type as `T &`. 
This seldom has any impact, 
because the type of most lvalue expressions inherently includes an lvalue reference qualifier. 
Functions returning lvalues, for example, always return lvalue references. 


There is an implication of this behavior that is worth being aware of, however. In
```
int x = 0;
```
`x` is the name of a variable, so `decltype(x)` is `int`. 
But wrapping the name `x` in parentheses `(x)` yields an expression more complicated than a name. 
Being a name, `x` is an lvalue, and C++ defines the expression `(x)` to be an lvalue, too.
`decltype((x))` is therefore `int &`. 
Putting parentheses around a name can change the type that decltype reports for it!


In C++11, this is little more than a curiosity; 
but in conjunction with C++14’s support for `decltype(auto)`, 
it means that a seemingly trivial change in the way you write a return statement can affect the deduced type for a function:
```
decltype(auto) f1()
{
    int x = 0;
    return x;        // decltype(x)   is int  , so f1 returns int
}

decltype(auto) f2()
{
    int x = 0;
    return (x);      // decltype((x)) is int &, so f2 returns int &
}
```
Note that not only does `f2` have a different return type from `f1`, 
it’s also returning a reference to a local variable, thus creating dangling references! 


The primary lesson is to *pay very close attention when using `decltype(auto)`*.
Seemingly insignificant details in the expression whose type is being deduced 
can affect the type that `decltype(auto)` reports. 
To ensure that the type being deduced is the type you expect, use the techniques described in Item 4. 






### 📌 Item 4: Know how to view deduced types.

- Deduced types can often be seen using IDE editors, compiler error messages, and the Boost TypeIndex library. 
- The results of some tools may be neither helpful nor accurate, so an understanding of C++’s type deduction rules remains essential. 


#### IDE Editors
 
```
const int theAnswer = 42;
auto x = theAnswer;            // int
auto y = &theAnswer;           // const int *
decltype(auto) z = theAnswer;  // const int
                               // PS. CLion 2020.2.3 is telling you this is 'int'. 
                               //     This bug was reported 4 years ago but still not fixed as of 03/18/2021. 
```

#### Compiler Diagnostics

An effective way to get a compiler to show a type it has deduced is 
to use that type in a way that leads to compilation problems. 
The error message reporting the problem is virtually sure to mention the type that’s causing it. 
Suppose, for example, we’d like to see the types that were deduced for `x` and `y` in the previous example. 
We first declare a class template that we don’t define. Something
like this does nicely:
```
template <typename T>   // declaration only for TD;
class TD;               // TD == "Type Displayer"
```
Attempts to instantiate this template will elicit an error message, 
because there’s no template definition to instantiate. 
To see the types for `x` and `y`, just try to instantiate `TD` with their types:
```
TD<decltype(x)> xType;  // elicit errors containing
TD<decltype(y)> yType;  // x's and y's types
```
Error message (`g++ Ubuntu 9.3.0-17ubuntu1~20.04 9.3.0`):
```
error: aggregate 'TD<int> xType' has incomplete type and cannot be defined
error: aggregate 'TD<const int *> yType' has incomplete type and cannot be defined
```

#### Runtime Output

```
boost::core::demangle(typeid(x).name());
```
Sadly, the results of `std::type_info::name` are **not** reliable. 
In this case, for example, the type that all three compilers report for param are incorrect. 
Furthermore, they’re essentially required to be incorrect, 
because the specification for `std::type_info::name` mandates that the type be treated 
*as if it had been passed to a template function as a by-value parameter*. 
As Item 1 explains, that means that if the type is a reference, its reference-ness is ignored, 
and if the type after reference removal is const (or volatile), its constness (or volatileness) is also ignored. 
That’s why param’s type (which is `const Widget * const &`) is reported as `const Widget *`.
First the type’s reference-ness is removed, and then the constness of the resulting pointer is eliminated.


Equally sadly, the type information displayed by IDE editors is also not reliable, or at least not reliably useful. 


If you’re more inclined to rely on libraries than luck, 
you’ll be pleased to know that where `std::type_info::name` and IDEs may fail, 
the *Boost TypeIndex library* (often written as `Boost.TypeIndex`) is designed to succeed. 
The library isn’t part of Standard C++, but neither are IDEs or templates like TD. 
Furthermore, the fact that Boost libraries (available at [boost.org](https://www.boost.org/)) 
are cross-platform, open source, 
and available under a license designed to be palatable to even the most paranoid corporate legal team 
means that code using Boost libraries is nearly as portable as code relying on the Standard Library.
```
#include <boost/type_index.hpp>

template <typename T>
void f(const T & param)
{
    std::cout << "T = "
              << type_id_with_cvr<T>().pretty_name()
              << '\n';

    std::cout << "param = "
              << boost::typeindex::type_id_with_cvr<decltype(param)>().pretty_name()
              << '\n';
}
```






## [CHAPTER 2] `auto`

### 📌 Item 5: Prefer `auto` to explicit type declarations

- Using `auto`
    - *is the best choice to directly hold closures*
    - *is the only choice for functions that need return type deduction*
    - *avoids uninitialized variables*
    - *avoids type shortcuts*
    - *avoids unintentional type mismatches*
    - *avoids verbose variable declarations and facilitates API reformatting*
- `auto`-typed variables are subject to the pitfalls described in Items 2 and 6. 

Ah, the simple joy of
```
int x;
```
Wait. Damn. I forgot to initialize `x`, so its value is indeterminate. Maybe. 
It might actually be initialized to zero. Depends on the context. Sigh. 


Never mind. Let’s move on to the simple joy of declaring a local variable to be initialized by dereferencing an iterator:
```
template <typename It>  // algorithm to dwim ("do what I mean")
void dwim(It b, It e)   // for all elements in range from b to e
{ 
    while (b != e) 
    {
        typename std::iterator_traits<It>::value_type currValue = *b;
    }
}
```
Okay, simple joy (take three): 
the delight of declaring a local variable whose type is that of a closure (lambda expression). 
Oh, right. The type of a closure is known only to the compiler, hence can’t be written out. 


As of C++11, all these issues go away, courtesy of `auto`. 
`auto` variables have their type deduced from their initializer, so they *must be initialized*.
That means you can wave goodbye to a host of uninitialized variable problems as you speed by on the modern C++ superhighway: 
```
int x1;       // potentially uninitialized
auto x2;      // error! initializer required
auto x3 = 0;  // fine, x's value is well-defined
```
Said highway lacks the potholes associated with declaring a local variable whose value is that of a dereferenced iterator: 
```
template <typename It>  // algorithm to dwim ("do what I mean")
void dwim(It b, It e)   // for all elements in range from b to e
{ 
    while (b != e) 
    {
        auto currValue = *b;
    }
}
```
And because `auto` uses *type deduction* (see Item 2), it can represent types known only to compilers:
```
// C++11 comparison func for Widgets pointed to by std::unique_ptrs
auto derefUPLess = [](const std::unique_ptr<Widget> & p1, const std::unique_ptr<Widget> & p2) 
{ 
    return *p1 < *p2; 
}; 
```
Very cool. In C++14, the temperature drops further, because parameters to lambda expressions may involve auto:
```
// C++14 comparison func for Widgets pointed to by anything pointer-like
auto derefUPLess = [](const auto & p1, const auto & p2) 
{ 
    return *p1 < *p2; 
}; 
```
Coolness notwithstanding, perhaps you’re thinking we don’t really need `auto` to declare a variable that holds a closure, 
because we can use a `std::function` object. 
It’s true, we can, but possibly that’s not what you were thinking. 
And maybe now you’re thinking “What’s a `std::function` object?” So let’s clear that up. 


`std::function` is a template in the C++11 Standard Library that generalizes the idea of a function pointer. 
Whereas function pointers can point only to functions, 
however, `std::function` objects can refer to *any callable object*, 
i.e., to anything that can be invoked like a function. 
Just as you must specify the type of function to point to when you create a function pointer 
(i.e., the signature of the functions you want to point to), 
you must specify the type of function to refer to when you create a `std::function` object. 
You do that through `std::function`’s template parameter.
For example, to declare a `std::function` object named func 
that could refer to any callable object acting as if it had this signature: 
```
std::function<bool (const std::unique_ptr<Widget> &,
                    const std::unique_ptr<Widget> &)> func;
```
Because lambda expressions yield callable objects, closures can be stored in `std::function` objects. 
That means we could declare the C++11 version of `derefUPLess` without using `auto` as follows:
```
std::function<bool (const std::unique_ptr<Widget> &,
                    const std::unique_ptr<Widget> &)>
derefUPLess = [](const std::unique_ptr<Widget> & p1, const std::unique_ptr<Widget> & p2) 
{ 
    return *p1 < *p2; 
}; 
```
It’s important to recognize that even setting aside the syntactic verbosity and need to repeat the parameter types, 
using `std::function` is **not** the same as using `auto`. 

- *Memory* <br>
    An `auto`-declared variable holding a closure has the *same type as the closure*, 
    and as such it uses only as much memory as the closure requires. 
    The type of a `std::function`-declared variable holding a closure is an *instantiation of the `std::function` template*,
    and that has a fixed size for any given signature. 
    This size may **not** be adequate for the closure it’s asked to store, and when that’s the case, 
    the `std::function` constructor will allocate heap memory to store the closure. 
    The result is that the `std::function` object typically *uses more memory* than the `auto`-declared object. 
- *Efficiency* <br>
    And, thanks to implementation details that *restrict inlining* and *yield indirect function calls*, 
    invoking a closure via a `std::function` object is almost certain to be *slower* than calling it via an `auto`-declared object. 
    In other words, the `std::function` approach is generally *bigger and slower* than the `auto` approach, 
    and it may yield out-of-memory exceptions, too. 
- *Verbosity* <br>
    Plus, as you can see in the examples above, writing `auto` is a whole lot less work than writing the type of the `std::function` instantiation. 

In the competition between `auto` and `std::function` for holding a closure, it’s pretty much game, set, and match for `auto`. 
(A similar argument can be made for `auto` over `std::function` for holding the result of calls to `std::bind`, 
but in Item 34, I do my best to convince you to use lambdas instead of `std::bind`, anyway. )


The advantages of `auto` extend beyond 

- *the avoidance of uninitialized variables*
- *verbose variable declarations*
- *the ability to directly hold closures*
- *type shortcuts*
- *unintentional type mismatches*

One is the ability to avoid what I call problems related to . 


Here’s something you’ve probably seen, possibly even written: 
```
std::vector<int> v;
unsigned sz = v.size();
```
The official return type of `v.size()` is `std::vector<int>::size_type`, but few developers are aware of that. 
`std::vector<int>::size_type` is specified to be an unsigned integral type, 
so a lot of programmers figure that `unsigned` is good enough and write code such as the above. 
This can have some interesting consequences. 
E.g., `g++ Ubuntu 9.3.0-17ubuntu1~20.04 9.3.0` implements `std::vector<int>::size_type` as `size_t` (aka. `unsigned long`). 
On 32-bit ubuntu, for example, both `unsigned` and `size_t` are the same size, 32 bits; 
but on 64-bit ubuntu, `unsigned` is 32 bits, while `size_t` is 64 bits. 
This means that code that works under 32-bit ubuntu may behave incorrectly under 64-bit ubuntu, 
and when porting your application from 32 to 64 bits, who wants to spend time on issues like that? 


Using `auto` ensures that you don’t have to: 
```
auto sz = v.size();  // sz's type is std::vector<int>::size_type, aka. size_t, aka. unsigned long
```
Still unsure about the wisdom of using `auto`? Then consider this code: 
```
std::unordered_map<std::string, int> m;

for (const std::pair<std::string, int> & p : m)
{
    // do something with p
}
```
Recognizing what’s amiss requires remembering that the key of a `std::unordered_map` is `const`, 
so the type of `std::pair` in the hash table (which is what a `std::unordered_map` is) 
**isn’t** `std::pair<std::string, int>`, it’s `std::pair<const std::string, int>`. 
But that’s **not** the type declared for the variable `p` in the loop above. 
As a result, compilers will strive to find a way to 
*convert `std::pair<const std::string, int>` objects (i.e., what’s in the hash table) to 
`std::pair<std::string, int>` objects (the declared type for `p`)*. 
They’ll succeed by *creating a temporary object* of the type that `p` wants to bind to 
*by copying each object in `m`*, then binding the reference `p` to that temporary object. 
At the end of each loop iteration, the temporary object will be destroyed. 
If you wrote this loop, you’d likely be surprised by this behavior, 
because you’d almost certainly intend to simply bind the reference `p` to each element in `m`. 


Such *unintentional type mismatches* can be `auto`ed away:
```
std::unordered_map<std::string, int> m;

for (const auto & p : m)
{
    // as before
}
```
This is not only more efficient, it’s also easier to type. 
Furthermore, this code has the very attractive characteristic that if you take `p`’s address, 
you’re sure to get a pointer to an element within `m`. 
In the code not using `auto`, you’d get a pointer to a temporary object: 
an object that would be destroyed at the end of the loop iteration.


The last two examples: 
writing `unsigned` when you should have written `std::vector<int>::size_type` and 
writing `std::pair<std::string, int>` when you should have written `std::pair<const std::string, int>`
demonstrate how explicitly specifying types can lead to implicit conversions that you neither want nor expect. 
If you use `auto` as the type of the target variable, you need not worry about mismatches 
between the type of variable you’re declaring and the type of the expression used to initialize it. 






### 📌 Item 6: Use the explicitly typed initializer idiom when `auto` deduces undesired types

- “Invisible” proxy types can cause `auto` to deduce the “wrong” type for an initializing expression. 
- The *explicitly typed initializer idiom* forces auto to deduce the type you want it to have. 

#### `auto` and proxy classes

```
std::vector<bool> vec {false, true}; 

bool b1 = vec[0];                     // of type bool
auto b2 = vec[1];                     // of type std::vector<bool>::reference
```
Though `std::vector<bool>` conceptually holds `bools`, 
`operator[]` for `std::vector<bool>` **doesn’t** return a *reference to an element* of the container 
(which is what `std::vector::operator[]` returns for every type **except** `bool`). 
Instead, it returns an object of type [`std::vector<bool>::reference`](https://en.cppreference.com/w/cpp/container/vector_bool/reference) 
(a class nested inside `std::vector<bool>`). 


`std::vector<bool>::reference` exists 
because `std::vector<bool>` is specified to represent its `bool`s in packed form, one bit per `bool`. 
That creates a problem for `std::vector<bool>`’s `operator[]`, 
because `operator[]` for `std::vector<T>` is supposed to return a `T &`, 
but C++ forbids references to bits. 
Not being able to return a `bool &`, 
`operator[]` for `std::vector<bool>` returns an object that acts like a `bool &`.
For this act to succeed, `std::vector<bool>::reference` objects must be usable in essentially all contexts where `bool &`s can be. 
Among the features in `std::vector<bool>::reference` that make this work is an implicit conversion to `bool`. 
(Not to `bool &`, to `bool`. )


The following code results in *undefined behavior*: 
```
std::vector<bool> features(const Widget & w);
Widget w;
bool highPriority = features(w)[5];            // depending on implementation, this is maybe a dangling pointer!
```
`features` returns a `std::vector<bool>` object, and, again, `operator[]` is invoked on it. 
`operator[]` continues to return a `std::vector<bool>::reference` object, and `auto` deduces that as the type of `highPriority`. 
`highPriority` **doesn’t** have the value of bit 5 of the `std::vector<bool>` returned by features at all. 


The value it does have depends on how `std::vector<bool>::reference` is implemented.
One implementation is for such objects to contain 
a *pointer to the machine word holding the referenced bit*, 
plus the offset into that word for that bit. 
Consider what that means for the initialization of `highPriority`, 
assuming that such a `std::vector<bool>::reference` implementation is in place. 


The call to `features` returns a *temporary* `std::vector<bool>` object. 
This object has no name, but for purposes of this discussion, I’ll call it `temp`. 
`operator[]` is invoked on `temp`, 
and the `std::vector<bool>::reference` it returns 
contains a pointer to a word in the data structure holding the bits that are managed by `temp`,
plus the offset into that word corresponding to bit 5. 
`highPriority` is a copy of this `std::vector<bool>::reference` object, 
so `highPriority`, too, contains a pointer to a word in `temp`, plus the offset corresponding to bit 5. 
At the end of the statement, *`temp` is destroyed*, because it’s a temporary object. 
Therefore, `highPriority` contains a dangling pointer, and that’s the cause of the undefined behavior. 


`std::vector<bool>::reference` is an example of a *proxy class*: 
a class that exists for the purpose of emulating and augmenting the behavior of some other type. 
Proxy classes are employed for a variety of purposes. 
`std::vector<bool>::reference` exists to offer the illusion that 
`operator[]` for `std::vector<bool>` returns a reference to a bit, for example, 
and the Standard Library’s *smart pointer types* (see Chapter 4) 
are proxy classes that graft resource management onto raw pointers. 
The utility of proxy classes is well-established. 
In fact, the design pattern *Proxy* is one of the most longstanding members of the software design patterns Pantheon. 
Some proxy classes are designed to be apparent to clients. 
That’s the case for `std::shared_ptr` and `std::unique_ptr`, for example. 
Other proxy classes are designed to act more or less invisibly. 
`std::vector<bool>::reference` is an example of such “invisible” proxies, 
as is its `std::bitset` compatriot, `std::bitset::reference`. 


Also in that camp are some classes in C++ libraries employing a technique known as *expression templates*. 
Such libraries were originally developed to improve the efficiency of numeric code. 
Given a class `Matrix` and `Matrix` objects `m1`, `m2`, `m3`, and `m4`, for example, the expression
```
Matrix sum = m1 + m2 + m3 + m4;
```
can be computed much more efficiently if `Matrix::operator+` returns a proxy for the result instead of the result itself. 
That is, `Matrix::operator+` for would return an object of a proxy class such as `Sum<Matrix, Matrix>` instead of a `Matrix` object. 
As was the case with `std::vector<bool>::reference` and `bool`, 
there’d be an implicit conversion from the proxy class to `Matrix`, 
which would permit the initialization of sum from the proxy object produced by the expression on the right side of the `=`. 
(The type of that object would traditionally encode the entire initialization expression, 
i.e., be something like `Sum<Sum<Sum<Matrix, Matrix>, Matrix>, Matrix>`. 
That’s definitely a type from which clients should be shielded. )


As a general rule, “invisible” proxy classes **don’t** play well with `auto`. 
Objects of such classes are often **not** designed to live longer than a single statement, 
so *creating variables of those types tends to violate fundamental library design assumptions*. 
That’s the case with `std::vector<bool>::reference`, and we’ve seen that violating that assumption can lead to undefined behavior.

#### The explicitly typed initializer idiom

The *explicitly typed initializer idiom* involves declaring a variable with `auto`, 
but casting the initialization expression to the type you want auto to deduce. 
Here’s how it can be used to force `highPriority` to be a `bool`, for example: 
```
auto highPriority = static_cast<bool>(features(w)[5]);
```
Applications of the idiom **aren’t** ~~limited to initializers yielding proxy class types~~. 
It can also be useful to emphasize that you are deliberately creating a variable of a type
that is different from that generated by the initializing expression. 
For example, suppose you have a function to calculate some tolerance value: 
```
double calcEpsilon();                          // return tolerance value
float ep1 = calcEpsilon();                     // impliclitly convert double -> float
auto ep2 = static_cast<float>(calcEpsilon());
```






## [CHAPTER 3] Moving to Modern C++

### 📌 Item 7: Distinguish between `()` and `{}` when creating objects

- Braced initialization is the most widely usable initialization syntax, 
  it prevents narrowing conversions, 
  and it’s immune to C++’s most vexing parse. 
- During constructor overload resolution, 
  braced initializers are matched to `std::initializer_list` parameters if at all possible, 
  even if other constructors offer seemingly better matches. 
- An example of where the choice between parentheses and braces can make a significant difference 
  is creating a `std::vector<numeric type>` with two arguments. 
- Choosing between parentheses and braces for object creation inside templates can be challenging. 


#### Initialization Syntaxes

Initialization values may be specified with parentheses, an equals sign, or braces: 
```
int x(0);     // initializer is in parentheses
int y = 0;    // initializer follows "="
int z {0};    // initializer is in braces
```
In many cases, it’s also possible to use an equals sign and braces together: 
```
int z = {0};  // initializer uses "=" and braces
              // C++ usually treats it the same as the braces-only version
```
C++ usually treats it the *same as the braces-only version*. 

#### Uniform Initialization

To address the confusion of multiple initialization syntaxes, 
as well as the fact that they don’t cover all initialization scenarios, 
C++11 introduces *uniform initialization* based on braces:
a single initialization syntax that can, at least in concept, 
be used anywhere and express everything. 
*Uniform initialization* is an idea. 
*Braced initialization* is a syntactic construct. 


Braced initialization lets you express the formerly inexpressible. Using braces, specifying
the initial contents of a container is easy:
```
std::vector<int> v {1, 3, 5};  // v's initial content is 1, 3, 5
```
Braces can also be used to specify default initialization values for *non-static data members*. 
This capability (new to C++11) is shared with the `=` initialization syntax, but **not** ~~with parentheses~~: 
```
class Widget 
{
private:
    int x {0};  // fine, x's default value is 0
    int y = 0;  // also fine
    int z(0);   // error!
};
```
On the other hand, *uncopyable objects* (e.g., `std::atomics`, see Item 40) 
may be initialized using braces or parentheses, but **not** ~~using `=`~~: 
```
std::atomic<int> ai1 {0};  // fine
std::atomic<int> ai2(0);   // fine
std::atomic<int> ai3 = 0;  // error!
```
It’s thus easy to understand why braced initialization is called *uniform*. 
Of C++’s three ways to designate an initializing expression, only braces can be used everywhere. 
A novel feature of braced initialization is that it *prohibits ~~implicit narrowing conversions among built-in types~~*. 
If the value of an expression in a braced initializer 
isn’t guaranteed to be expressible by the type of the object being initialized, 
the code won’t compile: 
```
double x, y, z;
int sum1 {x + y + z};  // error! sum of doubles may not be expressible as int
```
Initialization using parentheses and `=` **doesn’t** ~~check for narrowing conversions~~, 
because that could break too much legacy code: 
```
int sum2(x + y + z);   // okay (value of expression truncated to an int)
int sum3 = x + y + z;  // okay (value of expression truncated to an int)
```
Another noteworthy characteristic of braced initialization is its immunity to *C++’s most vexing parse*. 
A side effect of C++’s rule that anything that can be parsed as a declaration must be interpreted as one, 
the most vexing parse most frequently afflicts developers 
*when they want to default-construct an object, but inadvertently end up declaring a function instead*. 
The root of the problem is that if you want to call a constructor with an argument, you can do it like this,
```
Widget w1(10);  // call Widget ctor with argument 10
```
but if you try to call a Widget constructor with zero arguments using the analogous syntax, 

you declare a function instead of an object:
```
Widget w2();    // most vexing parse! declares a function named w2 that returns a Widget! 
```
Functions **can’t** be declared using braces for the parameter list, 
so default-constructing an object using braces doesn’t have this problem: 
```
Widget w3 {};   // calls Widget ctor with no args
```
The drawback to braced initialization is the sometimes-surprising behavior that accompanies it. 
Such behavior grows out of the unusually tangled relationship 
among braced initializers, `std::initializer_lists`, and constructor overload resolution. 
Their interactions can lead to code that seems like it should do one thing, 
but actually does another. 
For example, Item 2 explains that when an `auto`-declared variable has a braced initializer, 
the type deduced is `std::initializer_list`, 
even though other ways of declaring a variable with the same initializer would yield a more intuitive type. 
As a result, the more you like `auto`, the less enthusiastic you’re likely to be about braced initialization. 
In constructor calls, parentheses and braces have the same meaning as long as `std::initializer_list` parameters are not involved: 
```
class Widget 
{
public:
    Widget(int i, bool b);
    Widget(int i, double d);
};

Widget w1(10, true);          // calls first ctor
Widget w2 {10, true};         // also calls first ctor
Widget w3(10, 5.0);           // calls second ctor
Widget w4 {10, 5.0};          // also calls second ctor
```
If, however, one or more constructors declare a parameter of type `std::initializer_list`, 
calls using the braced initialization syntax strongly prefer the overloads taking `std::initializer_list`s. 
Strongly. If there’s any way for compilers to construe a call using a braced initializer 
to be to a constructor taking a `std::initializer_list`, compilers will employ that interpretation. 
If the Widget class above is augmented with a constructor taking a `std::initializer_list<long double>`, for example,
```
class Widget 
{
public:
    Widget(int i, bool b);
    Widget(int i, double d);
    Widget(std::initializer_list<long double> il);
};
```
Widgets `w2` and `w4` will be constructed using the new constructor, 
even though the type of the `std::initializer_list` elements (`long double`) is, 
compared to the non-`std::initializer_list` constructors, a worse match for both arguments! 
```
Widget w1(10, true);   // uses parens and, as before, calls first ctor
Widget w2 {10, true};  // uses braces, but now calls std::initializer_list ctor (10 and true convert to long double)
Widget w3(10, 5.0);    // uses parens and, as before, calls second ctor
Widget w4{10, 5.0};    // uses braces, but now calls std::initializer_list ctor (10 and 5.0 convert to long double)
```
Even what would normally be *copy and move construction can be hijacked* by `std::initializer_list` constructors: 
```
class Widget 
{
public:
    Widget(int i, bool b);
    Widget(int i, double d);
    Widget(std::initializer_list<long double> il);
    operator float() const;
};

Widget w5(w4);                                      // uses parens, calls copy ctor
Widget w6 {w4};                                     // uses braces, calls std::initializer_list ctor 
                                                    // (w4 converts to float, and float converts to long double)
Widget w7(std::move(w4));                           // uses parens, calls move ctor
Widget w8 {std::move(w4)};                          // uses braces, calls std::initializer_list ctor
                                                    // (for same reason as w6)
```
Compilers’ determination to match braced initializers with constructors taking `std::initializer_list`s is so strong, 
it *prevails even if the best-match `std::initializer_list` constructor can’t be called*. For example: 
```
class Widget 
{
public:
    Widget(int i, bool b);
    Widget(int i, double d);
    Widget(std::initializer_list<bool> il); 
}; 

Widget w {10, 5.0};                          // error! requires narrowing conversions
```
Here, compilers will ignore the first two constructors 
(the second of which offers an exact match on both argument types) 
and try to call the constructor taking a `std::initializer_list<bool>`. 
Calling that constructor would require converting an `int` (`10`) and a `double` (`5.0`) to `bool`s. 
Both conversions would be narrowing (`bool` can’t exactly represent either value), 
and narrowing conversions are prohibited inside braced initializers, so the call is invalid, and the code is rejected. 


Only if there’s no way to convert the types of the arguments in a braced initializer to
the type in a `std::initializer_list` do compilers fall back on normal overload resolution. 
For example, if we replace the `std::initializer_list<bool>` constructor
with one taking a `std::initializer_list<std::string>`, 
the non-`std::initializer_list` constructors become candidates again, 
because there is no way to convert `int`s and `bool`s to `std::string`s:
```
class Widget 
{
public:
    Widget(int i, bool b);
    Widget(int i, double d);
    Widget(std::initializer_list<std::string> il);
};

Widget w1(10, true);                                // uses parens, still calls first ctor
Widget w2 {10, true};                               // uses braces, now calls first ctor
Widget w3(10, 5.0);                                 // uses parens, still calls second ctor
Widget w4 {10, 5.0};                                // uses braces, now calls second ctor
```
This brings us near the end of our examination of braced initializers and constructor overloading, 
but there’s an interesting edge case that needs to be addressed. 
Suppose you use an empty set of braces to construct an object that supports default construction
and also supports `std::initializer_list` construction. 
What do your empty braces mean? If they mean “no arguments,” you get default construction, 
but if they mean “empty `std::initializer_list`,” you get construction from a `std::initializer_list` with no elements.


The rule is that you get default construction. 
Empty braces mean no arguments, **not** ~~an empty `std::initializer_list`~~:
```
class Widget 
{
public:
    Widget(); 
    Widget(std::initializer_list<int> il);
};

Widget w1;                                  // calls default ctor
Widget w2 {};                               // also calls default ctor
Widget w3();                                // most vexing parse! declares a function!
```
If you *want to* call a `std::initializer_list` constructor with an empty `std::initializer_list`, 
you do it by making the empty braces a constructor argument: 
by putting the empty braces inside the parentheses or braces demarcating what you’re passing: 
```
Widget w4({});                              // calls std::initializer_list ctor with empty list
Widget w5 {{}};                             // ditto
```
One of the classes directly affected by the above rule is `std::vector`.
`std::vector` has a non-`std::initializer_list` constructor 
that allows you to specify the initial size of the container and a value each of the initial elements should have, 
but it also has a constructor taking a `std::initializer_list` 
that permits you to specify the initial values in the container. 
If you create a `std::vector` of a numeric type (e.g., a `std::vector<int>`) and you pass two arguments to the constructor,
whether you enclose those arguments in parentheses or braces makes a tremendous difference: 
```
std::vector<int> v1(10, 20);   // use non-std::initializer_list ctor: create 10-element std::vector, all elements have value of 20
std::vector<int> v2 {10, 20};  // use std::initializer_list ctor: create 2-element std::vector, element values are 10 and 20
```
But let’s step back from `std::vector` and also from the details of parentheses, braces, and constructor overloading resolution rules. 
There are two primary takeaways from this discussion. 

- First, as a class author, you need to be aware that if your set of overloaded constructors 
  includes one or more functions taking a `std::initializer_list`, 
  client code using braced initialization may see only the `std::initializer_list` overloads. 
  As a result, it’s best to design your constructors so that the overload called **isn’t** affected by whether clients use parentheses or braces. 
  In other words, learn from what is now viewed as an error in the design of the `std::vector` interface, and design your classes to avoid it. 
  An implication is that 
  if you have a class with no `std::initializer_list` constructor, and you add one, 
  client code using braced initialization may find that 
  calls that used to resolve to non-`std::initializer_list` constructors 
  now resolve to the new function. 
  Of course, this kind of thing can happen any time you add a new function to a set of overloads: 
  calls that used to resolve to one of the old overloads might start calling the new one. 
  The difference with `std::initializer_list` constructor overloads is that 
  a `std::initializer_list` overload **doesn’t** just compete with other overloads, 
  it *overshadows them* to the point where the other overloads may hardly be considered. 
  So add such overloads only with great deliberation. 
- The second lesson is that as a class client, you must choose carefully between parenthesesnand braces when creating objects. 
  Most developers end up choosing one kind of delimiter as a default, using the other only when they have to. 
  Braces-by-default folks are attracted by their unrivaled breadth of applicability, 
  their prohibition of narrowing conversions, and their immunity to C++’s most vexing parse. 
  Such folks understand that in some cases 
  (e.g., creation of a `std::vector` with a given size and initial element value), 
  parentheses are required. 
  On the other hand, the go-parentheses-go crowd embraces parentheses as their default argument delimiter.
  They’re attracted to its consistency with the C++98 syntactic tradition, 
  its avoidance of the `auto`-deduced-a-`std::initializer_list` problem, 
  and the knowledge that their object creation calls won’t be inadvertently waylaid by `std::initializer_list` constructors. 
  They concede that sometimes only braces will do 
  (e.g., when creating a container with particular values). 
  There’s no consensus that either approach is better than the other, so my advice is to *pick one and apply it consistently*. 

#### Braced Initialization in Templates

If you’re a template author, 
the tension between parentheses and braces for object creation can be especially frustrating, 
because, in general, it’s not possible to know which should be used. 
For example, suppose you’d like to create an object of an arbitrary type from an arbitrary number of arguments. 
A variadic template makes this conceptually straightforward:
```
template <typename T, typename... Ts>
void doSomeWork(Ts && ... params)
{
    // create local T object from params...
}
```
There are two ways to turn the line of pseudocode into real code (see Item 25 for information about `std::forward`): 
```
T localObject(std::forward<Ts>(params)...);   // using parens
T localObject {std::forward<Ts>(params)...};  // using braces
```
So consider this calling code:
```
doSomeWork<std::vector<int>>(10, 20);
```
If `doSomeWork` uses parentheses when creating `localObject`, the result is a `std::vector<int>` with 10 elements. 
If `doSomeWork` uses braces, the result is a `std::vector<int>` with 2 elements. 
Which is correct? The author of `doSomeWork` can’t know. Only the caller can. 
This is precisely the problem faced by the Standard Library functions `std::make_unique` and `std::make_shared` (see Item 21). 
These functions resolve the problem by internally using parentheses and by documenting this decision as part of their interfaces. 






### 📌 Item 8: Prefer `nullptr` to `0` and `NULL`

- Prefer `nullptr` to `0` and `NULL`. 
    - Template type deduction will deduct `0` and `NULL` as their true type rather than the fallback meaning of null pointer. 
    - Passing `0` or `NULL` to functions overloaded on integral and pointer types will never call the pointer version. 
- Avoid overloading on integral and pointer types. 
    

So here’s the deal: the literal `0` is an `int`, **not** a pointer. 
If C++ finds itself looking at `0` in a context where only a pointer can be used, 
it’ll grudgingly interpret `0` as a null pointer, 
but that’s a fallback position. 
C++’s primary policy is that `0` is an `int`, not a pointer. 
<br><br>
Practically speaking, the same is true of `NULL`: 
```
// <stddef.h>
// g++ Ubuntu 9.3.0-17ubuntu1~20.04 9.3.0

#ifdef __GNUG__
#define NULL __null
#else                      /* G++ */
#ifndef __cplusplus
#define NULL ((void *) 0)
#else                      /* C++ */
#define NULL 0
#endif                     /* C++ */
#endif                     /* G++ */
```
There is some uncertainty in the details in `NULL`’s case, 
because implementations are allowed to give `NULL` an integral type other than `int` (e.g., `long`). 
That’s not common, but it doesn’t really matter, 
because the issue here isn’t the exact type of `NULL`, 
it’s that neither `0` nor `NULL` has a pointer type. 
<br><br>
In C++98, the primary implication of this was that 
overloading on pointer and integral types could lead to surprises. 
Passing `0` or `NULL` to such overloads **never** ~~called a pointer overload~~: 
```
void f(int);     // three overloads of f
void f(bool);
void f(void *);

f(0);            // calls f(int), not f(void*)
f(NULL);         // might not compile, but typically calls f(int). Never calls f(void*)
```
The uncertainty regarding the behavior of `f(NULL)` 
is a reflection of the leeway granted to implementations regarding the type of `NULL`. 
If `NULL` is defined to be, say, `0L`, the call is *ambiguous*, 
because conversion from `long` to `int`, `long` to `bool`, and `0L` to `void *` are considered equally good. 
The interesting thing about that call is the contradiction between the apparent meaning of the source code 
(“I’m calling `f` with `NULL`, the null pointer”) and its actual meaning (“I’m calling `f` with some kind of integer, **not** the null pointer”). 
This counterintuitive behavior is what led to the guideline for C++98 programmers to avoid overloading on pointer and integral types. 
That guideline remains valid in C++11, because, the advice of this item notwithstanding, 
it’s likely that some developers will continue to use `0` and `NULL`,
even though `nullptr` is a better choice. 
<br><br>
`nullptr`’s advantage is that it **doesn’t** have an integral type. 
To be honest, it doesn’t have a pointer type, either, but you can think of it as a pointer of all types. 
`nullptr`’s actual type is `std::nullptr_t`, and, 
in a wonderfully circular definition,
`std::nullptr_t` is defined to be the type of `nullptr`. 
The type `std::nullptr_t` *implicitly converts to all raw pointer types*, 
and that’s what makes `nullptr` act as if it were a pointer of all types. 
Calling the overloaded function `f` with `nullptr` calls the `void *` overload, 
because `nullptr` can’t be viewed as anything integral: 
```
f(nullptr);      // calls f(void*) overload
```
Using `nullptr` instead of `0` or `NULL` thus avoids overload resolution surprises, but that’s not its only advantage. 
It can also improve code clarity, especially when `auto` variables are involved. 
For example, suppose you encounter this in a code base: 
```
auto result = findRecord(/* arguments */);

if (result == 0) 
{
    // ...
}
```
If you don’t happen to know (or can’t easily find out) what `findRecord` returns, 
it may not be clear whether result is a pointer type or an integral type. 
After all, `0` (what result is tested against) could go either way. 
If you see the following, on the other hand,
```
auto result = findRecord(/* arguments */);

if (result == nullptr) 
{
    // ...
}
```
there’s no ambiguity: result must be a pointer type. 
<br><br>
`nullptr` shines especially brightly when templates enter the picture. 
Suppose you have some functions that should be called only when the appropriate mutex has been locked. 
Each function takes a different kind of pointer: 
```
int f1(std::shared_ptr<Widget> spw);           // call these only when
double f2(std::unique_ptr<Widget> upw);        // the appropriate
bool f3(Widget * pw);                          // mutex is locked
```
Calling code that wants to pass null pointers could look like this: 
```
std::mutex f1m, f2m, f3m;                      // mutexes for f1, f2, and f3
using MuxGuard = std::lock_guard<std::mutex>;

{
    MuxGuard g(f1m);                           // lock mutex for f1
    auto result = f1(0);                       // pass 0 as null ptr to f1
}                                              // unlock mutex

{
    MuxGuard g(f2m);                           // lock mutex for f2
    auto result = f2(NULL);                    // pass NULL as null ptr to f2
}                                              // unlock mutex

{
    MuxGuard g(f3m);                           // lock mutex for f3
    auto result = f3(nullptr);                 // pass nullptr as null ptr to f3
}
```
The failure to use `nullptr` in the first two calls in this code is sad, but the code works, and that counts for something. 
However, the repeated pattern in the calling code: lock mutex, call function, unlock mutex, is more than sad: it’s disturbing.
This kind of source code duplication is one of the things that templates are designed to avoid, so let’s templatize the pattern: 
```
template <typename FuncType, typename MuxType, typename PtrType>
auto lockAndCall(FuncType func, MuxType & mutex, PtrType ptr) -> decltype(func(ptr))  // C++11 
{
    MuxGuard g(mutex);
    return func(ptr);
}

template <typename FuncType, typename MuxType, typename PtrType>
decltype(auto) lockAndCall(FuncType func, MuxType & mutex, PtrType ptr)               // C++14
{
    MuxGuard g(mutex);
    return func(ptr);
}
```
Given the `lockAndCall` template (either version), callers can write code like this: 
```
auto result1 = lockAndCall(f1, f1m, 0);        // error!
auto result2 = lockAndCall(f2, f2m, NULL);     // error!
auto result3 = lockAndCall(f3, f3m, nullptr);  // fine
```
In two of the three cases, the code won’t compile. 
The problem in the first call is that when `0` is passed to `lockAndCall`, 
template type deduction kicks in to figure out its type. 
The type of `0` is, was, and always will be `int`, 
so that’s the type of the parameter ptr inside the instantiation of this call to `lockAndCall`. 
Unfortunately, this means that in the call to `func` inside `lockAndCall`, an `int` is being passed, 
and that’s **not** compatible with the `std::shared_ptr<Widget>` parameter that `f1` expects. 
The `0` passed in the call to `lockAndCall` was intended to represent a null pointer, 
but what actually got passed was a run-of-the-mill `int`. 
Trying to pass this `int` to `f1` as a `std::shared_ptr<Widget>` is a type error. 
The call to `lockAndCall` with `0` fails because inside the template, 
an `int` is being passed to a function that requires a `std::shared_ptr<Widget>`. 
<br><br>
The analysis for the call involving `NULL` is essentially the same. 
When `NULL` is passed to `lockAndCall`, an integral type is deduced for the parameter `ptr`, 
and a type error occurs when ptr, an `int` or `int`-like type, is passed to `f2`, 
which expects to get a `std::unique_ptr<Widget>`. 
<br><br>
In contrast, the call involving `nullptr` has no trouble. 
When `nullptr` is passed to `lockAndCall`, the type for `ptr` is deduced to be `std::nullptr_t`. 
When `ptr` is passed to `f3`, there’s an *implicit conversion* from `std::nullptr_t` to `Widget *`,
because std::nullptr_t implicitly converts to all pointer types.
<br><br>
The fact that template type deduction deduces the “wrong” types for `0` and `NULL` 
(i.e., their true types, rather than their fallback meaning as a representation for a null pointer) 
is the most compelling reason to use `nullptr` instead of `0` or `NULL` when you want to refer to a null pointer. 
With `nullptr`, templates pose no special challenge. 
Combined with the fact that `nullptr` doesn’t suffer from the overload resolution surprises that `0` and `NULL` are susceptible to, 
the case is ironclad. 
When you want to refer to a null pointer, use `nullptr`, **not** `0` or `NULL`.

### 📌 Item 9: Prefer alias declarations to typedefs





### 📌 Item 10: Prefer scoped `enum`s to unscoped `enum`s





### 📌 Item 11: Prefer deleted functions to private undefined ones




### 📌 Item 12: Declare overriding functions `override`







### 📌 Item 13: Prefer `const_iterator`s to `iterator`s






### 📌 Item 14: Declare functions `noexcept` if they won’t emit exceptions








### 📌 Item 15: Use `constexpr` whenever possible








### 📌 Item 16: Make `const` member functions thread safe




### 📌 Item 17: Understand special member function generation





## [CHAPTER 4] Smart Pointers

### 📌 Item 18: Use `std::unique_ptr` for exclusive-ownership resource management




### 📌 Item 19: Use `std::shared_ptr` for shared-ownership resource management



### 📌 Item 20: Use `std::weak_ptr` for `std::shared_ptr`-like pointers that can dangle



### 📌 Item 21: Prefer `std::make_unique` and `std::make_shared` to direct use of `new`





### 📌 Item 22: When using the Pimpl Idiom, define special member functions in the implementation file







## [CHAPTER 5] Rvalue References, Move Semantics, and Perfect Forwarding

### 📌 Item 23: Understand `std::move` and `std::forward`



### 📌 Item 24: Distinguish universal references from rvalue references



### 📌 Item 25: Use `std::move` on rvalue references, `std::forward` on universal references




### 📌 Item 26: Avoid overloading on universal references





### 📌 Item 27: Familiarize yourself with alternatives to overloading on universal references




### 📌 Item 28: Understand reference collapsing





### 📌 Item 29: Assume that move operations are not present, not cheap, and not used



### 📌 Item 30: Familiarize yourself with perfect forwarding failure cases



## [CHAPTER 6] Lambda Expressions

### 📌 Item 31: Avoid default capture modes




### 📌 Item 32: Use init capture to move objects into closures





### 📌 Item 33: Use `decltype` on `auto &&` parameters to `std::forward` them




### 📌 Item 34: Prefer lambdas to `std::bind`





## [CHAPTER 7] The Concurrency API

### 📌 Item 35: Prefer task-based programming to threadbased



### 📌 Item 36: Specify `std::launch::async` if asynchronicity is essential




### 📌 Item 37: Make `std::thread`s unjoinable on all paths






### 📌 Item 38: Be aware of varying thread handle destructor behavior





### 📌 Item 39: Consider `void` futures for one-shot event communication




### 📌 Item 40: Use `std::atomic` for concurrency, `volatile` for special memory



## [CHAPTER 8] Tweaks

### 📌 Item 41: Consider pass by value for copyable parameters that are cheap to move and always copied





### 📌 Item 42: Consider emplacement instead of insertion




