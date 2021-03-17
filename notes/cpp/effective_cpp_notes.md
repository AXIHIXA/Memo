# *`Effective C++`* Notes

- Notes of reading ***Effective C++ Digital Collection: 140 Ways to Improve Your Programming***






---

## 🌱 Effective C++: 55 Specific Ways to Improve Your Programs and Designs

### 📌 Item 2: Prefer `const`s, `enum`s, and `inline`s to `#define`s

- **Things to Remember**
    1. For simple constants, prefer `const` objects or `enum`s to `#define`s.
    2. For function-like macros, prefer `inline` functions to `#define`s.

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
    //because we don't know what T is, we pass by reference-to-const
    template <typename T> 
    inline void callWithMax(const T & a, const T & b) 
    { 
        f(a > b ? a : b);
    }
    ```


















---

## 🌱 Effective Modern C++: 42 Ways to Improve Your of C++11 and C++14

### 📌 Item 2























---

## 🌱 Effective STL: 50 Specific Ways to Improve Your Use of the Standard Template Library

### 📌 Item 2