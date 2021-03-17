# *`Effective C++`* Notes

- Notes of reading ++***Effective C++ Digital Collection: 140 Ways to Improve Your Programming***++






---

## 🌱 Effective C++: 55 Specific Ways to Improve Your Programs and Designs

### 📌 Item 2: Prefer `const`s, `enum`s, and `inline`s to `#define`s

- **Things to Remember**
    - For simple constants, prefer `const` objects or `enum`s to `#define`s.
    - For function-like macros, prefer `inline` functions to `#define`s.
- **The `enum` hack**: For class-specific constants, use `enum`s instead of `static const` data members 
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
    (e.g., `int`s, `char`s, `bool`s) are an *exception*. <br>
    As long as you don't *take their address*, 
    you can declare them and use them without ~~providing a definition~~. <br>
    If you do take the address of a class constant, 
    or if your compiler incorrectly insists on a definition even if you don't take the address, 
    you provide a separate definition in implementation file. <br>
    Older compilers may not accept the syntax above, 
    because it used to be ~~illegal to provide an initial value for a static class member at its point of declaration~~. <br>
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
    The above block of code is all you need almost all the time. <br>
    The only exception is when you *need the value of a class constant during compilation* of the class, 
    such as in the declaration of the array `GamePlayer::scores` above 
    (where compilers insist on knowing the size of the array during compilation). <br>
    Then the accepted way to compensate for compilers that (incorrectly) forbid 
    the in-class specification of initial values for static integral class constants 
    is to use what is affectionately (and non-pejoratively) known as the `enum` hack. <br>
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
      and sometimes that's what you want. <br>
      For example, it's legal to take the address of a `const`, 
      but it's **not legal** to take the address of an `enum`, 
      and it's typically **not legal** to take the address of a `#define`, either. <br>
      If you don't want to let people get a pointer or reference to one of your integral constants, 
      an enum is a good way to enforce that constraint. 
    - *Memory Allocation*. <br>
      Though good compilers won't set aside storage for `const` objects of integral types 
      (unless you create a pointer or reference to the object), 
      sloppy compilers may, and you may not be willing to set aside memory for such objects. <br>
      Like `#define`s, `enum`s never result in that kind of unnecessary memory allocation.
    - *Pragmatic*. <br>
      Lots of code employs it, so you need to recognize it when you see it. <br>
      In fact, the `enum` hack is a fundamental technique of template metaprogramming. 
- **Common (mis)use of `#define` directives**: 
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
    each of which takes two objects of the same type and calls `f` with the greater of the two objects. <br>
    There's no need to parenthesize parameters inside the function body, 
    no need to worry about evaluating parameters multiple times, etc. <br>
    Furthermore, because callWithMax is a real function, it obeys scope and access rules. <br>
    For example, it makes perfect sense to talk about an inline function that is private to a class. <br>
    In general, there's just no way to do that with a macro.

### 📌 Item 3: Use `const` whenever possible

- **Things to Remember**
    - Declaring something `const` helps compilers detect usage errors. 
      `const` can be applied to objects at any scope, 
      to function parameters and return types, 
      and to member functions as a whole.
    - Compilers enforce bitwise constness, but you should program using logical constness.
    - When `const` and non-`const` member functions have essentially identical implementations, 
      code duplication can be avoided by having the non-`const` version call the const version.
- **`const` iterators**: <br>
    *STL iterators are modeled on pointers*, so an iterator acts much like a `T *` pointer. <br>
    Declaring an iterator `const` is like declaring a pointer const (i.e., declaring a `T * const` pointer): 
    the iterator isn't allowed to point to something different, but the thing it points to may be modified. <br>
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
- **`const`s in function return values**. <br>
    Having a function return a constant value often makes it possible to 
    reduce the incidence of client errors without giving up safety or efficiency: 
    ```
    class Rational { ... };
    const Rational operator*(const Rational & lhs, const Rational & rhs);
    ```
    Many programmers squint when they first see this. <br>
    Why should the result of `operator*` be a `const` object? <br>
    Because if it weren't, clients would be able to commit atrocities like this:
    ```
    Rational a, b, c;
    (a * b) = c;       // invoke operator= on the result of (a * b)!
    ```
    I don't know why any programmer would want to make an assignment to the product of two numbers, 
    but I do know that many programmers have tried to do it without wanting to. <br>
    All it takes is a simple typo (and a type that can be implicitly converted to `bool`):
    ```
    if (a * b = c)     // oops, meant to do a comparison!
    {
        // ...
    }
    ```
    Such code would be flat-out illegal if `a` and `b` were of a built-in type. <br>
    One of the hallmarks of good user-defined types is that they avoid gratuitous incompatibilities with the built-ins, 
    and allowing assignments to the product of two numbers seems pretty gratuitous to me. <br>
    Declaring `operator*`'s return value `const` prevents it, and that's why it's The Right Thing To Do. 
- **`const` member functions**
    











---

## 🌱 Effective Modern C++: 42 Ways to Improve Your of C++11 and C++14

### 📌 Item 2























---

## 🌱 Effective STL: 50 Specific Ways to Improve Your Use of the Standard Template Library

### 📌 Item 2