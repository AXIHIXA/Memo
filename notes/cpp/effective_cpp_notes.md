# *`Effective C++`* Notes

- Notes of reading *`Effective C++ Digital Collection: 140 Ways to Improve Your Programming`*






## 🌱 Effective C++: 55 Specific Ways to Improve Your Programs and Designs

### 📌 Item 2: Prefer `const`s, `enum`s, and `inline`s to `#define`s

- The *"`enum` hack"* : Use `enum`s instead of `static` `const` class data member
    - Version a)
    ```
    // GamePlayer.h
    class GamePlayer 
    {
        static const int NumTurns = 5;   // constant declaration & in-class initialization 
        int scores[NumTurns];            // use of constant
    };

    // GamePlayer.cpp
    const int GamePlayer::NumTurns;      // definition of NumTurns; see below for why no value is given
    ```
    Usually, C++ requires that you provide a definition for anything you use, 
    but class-specific constants that are static and of integral type 
    (e.g., integers, chars, bools) are an exception. 
    As long as you don't take their address, you can declare them and use them without providing a definition. 
    If you do take the address of a class constant, 
    or if your compiler incorrectly insists on a definition even if you don't take the address, 
    you provide a separate definition in implementation file. 
- An example of a nonsense marco: 
```
// call f with the maximum of a and b
// even if everything is properly parenthesised, there can still be problems! 
#define CALL_WITH_MAX(a, b) f((a) > (b) ? (a) : (b))

int a = 5, b = 0;
CALL_WITH_MAX(++a, b);       // a is incremented twice
CALL_WITH_MAX(++a, b + 10);  // a is incremented once
```





















## 🌱 Effective Modern C++: 42 Ways to Improve Your of C++11 and C++14

### 📌 Item 2

























## 🌱 Effective STL: 50 Specific Ways to Improve Your Use of the Standard Template Library

### 📌 Item 2