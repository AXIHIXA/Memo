# *`Effective C++`* Notes

- Notes of reading ***Effective C++ Digital Collection: 140 Ways to Improve Your Programming***






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
      code duplication can be avoided by having the non-`const` version call the `const` version, 
      by invoking `const_cast` to cast away constness.
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
- **`const` member functions** <br>
    Many people overlook the fact that *member functions differing only in their constness can be overloaded*, but this is an important feature of C++. <br>
    Incidentally, const objects most often arise in real programs as a result of being passed by pointer-to-const or reference-to-const.
    What does it mean for a member function to be const? <br>
    There are two prevailing notions: 
    - *bitwise constness* (also known as *physical constness*) <br>
        The bitwise `const` camp believes that a member function is `const`
        iff. it doesn't modify any of the object's data members (excluding those that are `static`), 
        i.e., iff. it *doesn't modify any of the bits inside the object*. <br>
        The nice thing about bitwise constness is that it's easy to detect violations: 
        compilers just look for assignments to data members. <br>
        In fact, bitwise constness is C++'s definition of constness, 
        and a `const` member function isn't allowed to modify 
        any of the non-`static` data members of the object on which it is invoked. <br>
        Unfortunately, many member functions that don't act very `const` pass the bitwise test. <br>
        In particular, *a member function that modifies what a pointer points* to frequently doesn't act `const`. <br>
        But if only the pointer is in the object, the function is bitwise `const`, and compilers won't complain. <br>
        That can lead to counterintuitive behavior. <br>
        For example, suppose we have a `TextBlock`-like class that stores its data as a `char *` instead of a `string`, 
        because it needs to communicate through a C API that doesn't understand `string` objects. <br>
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
        even though that function returns a reference to the object's internal data. <br>
        Set that aside and note that `operator[]`'s implementation doesn't modify `pText` in any way. <br>
        As a result, compilers will happily generate code for `operator[]`; 
        it is, after all, bitwise `const`, and that's all compilers check for. <br>
        But look what it allows to happen:
        ```
        const CTextBlock cctb("Hello");  // declare constant object
        char *pc = &cctb[0];             // call the const operator[] to get a  pointer to cctb's data
        *pc = 'J';                       // cctb now has the value "Jello"
        ```
    - *logical constness* <br>
        A `const` member function *might modify some of the bits in the object* on which it's invoked, 
        but *only in ways that clients cannot detect*. <br>
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
- **Avoiding Duplication in `const` and Non-`const` Member Functions** <br>
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
    but there's no direct way to do that. <br>
    Instead, we cast `*this` from its native type of `TextBlock &` to `const TextBlock &`.
    Yes, we use a cast to add const! So we have two casts: <br>
    one to add const to `*this` (so that our call to `operator[]` will call the `const` version), 
    the second to remove the `const` from the `const` `operator[]`'s return value. <br>
    The cast that adds `const` is just forcing a safe conversion 
    (from a non-`const` object to a `const` one), 
    so we use a `static_cast` for that. <br>
    The one that removes `const` can be accomplished only via a `const_cast`, 
    so we don't really have a choice there. <br>
    Even more worth knowing is that trying to do things the other way around 
    (avoiding duplication by ~~having the `const` version call the non-`const` version~~)
    is **not** something you want to do. 
    Remember, a `const` member function promises never to change the logical state of its object, 
    but a non-`const` member function makes no such promise. <br>
    If you were to call a non-`const` function from a `const` one, 
    you'd run the risk that the object you'd promised not to modify would be changed. <br>
    That's why having a `const` member function call a non-`const` one is wrong: 
    the object could be changed. <br>
    In fact, to get the code to compile, you'd have to use a `const_cast` to get rid of the `const` on `*this`,
    a clear sign of trouble. <br>
    The reverse calling sequence (the one we used above) is safe: 
    the non-`const` member function can do whatever it wants with an object, 
    so calling a `const` member function imposes no risk. <br>
    That's why a `static_cast` works on `*this` in that case: there's no `const`-related danger. 

### 📌 Item 4: Make sure that objects are initialized before they're used










































---

## 🌱 Effective Modern C++: 42 Ways to Improve Your of C++11 and C++14

### 📌 Item 2























---

## 🌱 Effective STL: 50 Specific Ways to Improve Your Use of the Standard Template Library

### 📌 Item 2