# _Effective C++_ Notes

- Notes of reading <u>Scott Meyers</u>'s _Effective C++_ series:
    1. *[`Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_01_effective_cpp.md)*
    2. ***[`More Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_02_more_effective_cpp.md)***
    3. *[`Effective STL`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_03_effective_stl.md)*
    4. *[`Effective Modern C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_04_effective_modern_cpp.md)*






---

## 🌱 _More Effective C++_

### 🎯 Chapter 1. Basics

### 📌 Item 1: Distinguish between pointers and references


First, recognize that there is **no** null reference. 
A reference must always refer to some object. 
As a result, if you have a variable whose purpose is to refer to another object, 
but it is possible that there might not be an object to refer to,
you should make the variable a pointer, because then you can set it to null. 
On the other hand, if the variable must always refer to an object, 
i.e., if your design does **not** allow for the possibility that the variable is null, 
you should probably make the variable a reference.


The following code is undefined behavior (compilers can generate output to do anything they like):
```c++
char * pc = nullptr;  // set pointer to null
char & rc = *pc;      // make reference to refer to de-referenced null pointer
```
Because a reference must refer to an object, C++ requires that references be initialized:
```c++
std::string & rs;        // error! References must be initialized

std::string s("xyzzy");
std::string & rs = s;    // okay, rs refers to s

// Pointers are subject to no such restriction:
std::string * ps;        // uninitialized pointer: valid but risky
```
The fact that there is no such thing as a null reference implies that 
it can be more efficient to use references than to use pointers. 
That’s because there’s no need to test the validity of a reference before using it:
```c++
void printDouble(const double & rd)
{
    std::cout << rd << '\n';
}
```
Pointers, on the other hand, should generally be tested against null:
```c++
void printDouble(const double * pd)
{
    if (pd)
    {
        std::cout << pd << '\n';
    }
}
```
Another important difference between pointers and references is that
pointers may be reassigned to refer to different objects (low-high-level `const`ness).
A reference, however, always refers to the object with which it is initialized (fixed high-level `const`ness): 
```c++
std::string s1("Nancy");
std::string s2("Clancy");
std::string & rs = s1;     // rs refers to s1
std::string * ps = &s1;    // ps points to s1
rs = s2;                   // rs still refers to s1, 
                           // but s1’s value is now "Clancy"
ps = &s2;                  // ps now points to s2; s1 is unchanged
```
In general, you should use a pointer 
whenever you need to take into account the possibility 
that there’s nothing to refer to (in which case you can set the pointer to null) 
or whenever you need to be able to refer to different things at different times 
(in which case you can change where the pointer points). 
You should use a reference whenever you know there will always be an object to refer to, 
and you also know that once you’re referring to that object, 
you’ll never want to refer to anything else.


There is one other situation in which you should use a reference, 
and that’s when you’re implementing certain operators. 
The most common example is `operator[]`. 
This operator typically needs to return lvalue (something that can be used as the target of an assignment):
```c++
std::vector<int> vec(10);
vec[5] = 10;
```
If `std::vector<int>::operator[]` returned a pointer, this last statement would have to be written this way:
```c++
*vec[5] = 10;
```
But this makes it look like `vec` is a vector of pointers, which it’s not. 
For this reason, you’ll almost always want `operator[]` to return a reference. 
(For an interesting exception to this rule, see Item 30.)
References, then, are the feature of choice 
when you know you have something to refer to 
and when you’ll never want to refer to anything else. 
They’re also appropriate when implementing operators whose syntactic requirements make the use of pointers undesirable. 
In all other cases, stick with pointers.






### 📌 Item 2: Prefer C++-style casts






### 📌 Item 3: Never treat arrays polymorphically

- Polymorphism does **not** work involving pointer arithmetic. 
  Array operations always involve pointer arithmetic, so arrays **do** not work with polymorphism. 


One of the most important features of inheritance is that 
you can manipulate derived class objects 
through pointers and references to base class objects. 
Such pointers and references are said to behave _polymorphically_, as if they had multiple types. 


C++ also allows you to manipulate _arrays_ of derived class objects through base class pointers and references. 
This is **no** feature at all, because it almost **never** works the way you want it to.


For example, suppose you have a class `BST` (for binary search trees) 
and a second class, `AVL`, that inherits from `BST`:
```c++
template <typename T>
class BST { ... };

template <typename T>
class AVL : public BST<T> { ... };

// Note that for parameter types,
// `const BST<T> []` is essentially same as a `const BST<T> *`!
template <typename T>
void printBSTArray(std::ostream & cout, const BST<T> array[], int numElements)
{
    for (int i = 0; i < numElements; ++i)
    {
        cout << array[i] << '\n';
    }
}

BST bstArray[10];
printBSTArray(cout, bstArray, 10);  // fine

AVL avlArray[10];
printBSTArray(cout, avlArray, 10);  // works fine?
```
Compilers will accept this function call without complaint.
But, `array[i]` is really just shorthand for the expression `*(array + i)`. 
We know that when passed as to function parameters, 
`array` is implicitly cast to a pointer to the beginning of the array, 
but how far away from the memory location pointed to by `array` is the memory location pointed to by `array + i`? 
The distance between them is `i * sizeof(an object in the array)`, 
because there are `i` objects between `array[0]` and `array[i]`. 
In order for compilers to emit code that walks through the array correctly, 
they must be able to determine the size of the objects in the array. 
This is easy for them to do. 
The parameter `array` is declared to be of type array-of-`BST`, 
so each element of the array must be a `BST`, and the distance between `array` and `array + i` must be `i * sizeof(BST)`.


At least that’s how your compilers look at it. 
But if you’ve passed an array of `AVL` objects to `printBSTArray`, 
your compilers are probably wrong. 
In that case, they’d assume each object in the array is the size of a `BST`, 
but each object would actually be the size of an `AVL`. 
Derived classes usually have more data members than their base classes, 
so derived class objects are usually larger than base class objects. 
We thus expect a `AVL` object to be larger than a `BST` object. 
If it is, the pointer arithmetic generated for `printBSTArray` will be **wrong** for arrays of `AVL` objects, 
and there’s no telling what will happen when `printBSTArray` is invoked on a `AVL`array. 
Whatever does happen, it’s a good bet it won’t be pleasant.


The problem pops up in a different guise if you try to 
delete an array of derived class objects through a base class pointer. 
Here’s one way you might innocently attempt to do it:
```c++
void deleteArray(std::ostream & lout, BST array[])
{
    lout << "Deleting array at address " << static_cast<void *>(array) << '\n';
    delete [] array;
}

AVL * avlArray = new AVL[50];
deleteArray(std::cout, avlArray);
```
You can’t see it, but there’s pointer arithmetic going on here, too. 
When an array is deleted, a destructor for each element of the array must be called (see Item 8). 
When compilers see the statement `delete [] array;`,
they must generate code that does something like this:
```c++
// destruct the objects in *array in the inverse order in which they were constructed
for (int i = the number of elements in the array - 1; 0 <= i; --i)
{
    array[i].BST::~BST();
}
```
Just as this kind of loop failed to work when you wrote it, 
it will fail to work when your compilers write it, too. 
The language specification says 
**the result of deleting an array of derived class objects through a base class pointer is undefined**, 
but we know what that really means:
executing the code is almost certain to lead to grief. 


Polymorphism and pointer arithmetic simply don’t mix. 
Array operations almost always involve pointer arithmetic, so arrays and polymorphism don’t mix.


Note that you’re unlikely to make the mistake of treating an array polymorphically 
if you avoid having a concrete class (like `AVL`) inherit from another concrete class (such as `BST`).
As Item 33 explains, designing your software so that concrete classes never inherit from one another has many benefits. 
I encourage you to turn to Item 33 and read all about them. 






### 📌 Item 4: Avoid gratuitous default constructors






### 🎯 Chapter 2. Operators

### 📌 Item 5: Be wary of user-defined conversion functions






### 📌 Item 6: Distinguish between prefix and postfix forms of increment and decrement operators






### 📌 Item 7: Never overload `&&`, `||`, or `,`






### 📌 Item 8: Understand the different meanings of `new` and `delete`






### 🎯 Chapter 3. Exceptions

### 📌 Item 9: Use destructors to prevent resource leaks






### 📌 Item 10: Prevent resource leaks in constructors






### 📌 Item 11: Prevent exceptions from leaving destructors






### 📌 Item 12: Understand how throwing an exception differs from passing a parameter or calling a virtual function






### 📌 Item 13: Catch exceptions by reference






### 📌 Item 14: Use exception specifications judiciously






### 📌 Item 15: Understand the costs of exception handling






### 🎯 Chapter 4. Efficiency

### 📌 Item 16: Remember the 80-20 rule






### 📌 Item 17: Consider using lazy evaluation






### 📌 Item 18: Amortize the cost of expected computations






### 📌 Item 19: Understand the origin of temporary objects






### 📌 Item 20: Facilitate the return value optimization






### 📌 Item 21: Overload to avoid implicit type conversions






### 📌 Item 22: Consider using `op=` instead of stand-alone `op`






### 📌 Item 23: Consider alternative libraries






### 📌 Item 24: Understand the costs of virtual functions, multiple inheritance, virtual base classes, and RTTI






### 🎯 Chapter 5. Techniques

### 📌 Item 25: Virtualizing constructors and non-member functions






### 📌 Item 26: Limiting the number of objects of a class






### 📌 Item 27: Requiring or prohibiting heap-based objects






### 📌 Item 28: Smart pointers

- The originals in this Item is for C++98 and is already outdated. 
  Refer to Effective Modern C++ Chapter 4 Smarter Pointers for details. 






### 📌 Item 29: Reference counting






### 📌 Item 30: Proxy classes






### 📌 Item 31: Making functions virtual with respect to more than one object






### 🎯 Chapter 6. Miscellany

### 📌 Item 32: Program in the future tense






### 📌 Item 33: Make non-leaf classes abstract


Suppose you’re working on a project whose software deals with animals. 
Most animals can be treated pretty much the same, but lizards and chickens require special handling. 
That being the case, the obvious way to relate the classes for animals, lizards, and chickens is like this:
```c++
class Animal 
{
public:
    Animal & operator=(const Animal & rhs);
    // ...
};

class Lizard : public Animal 
{
public:
    Lizard & operator=(const Lizard & rhs);
    // ...
};

class Chicken : public Animal 
{
public:
    Chicken & operator=(const Chicken & rhs);
    // ...
};
```
Consider this code:
```c++
Lizard liz1;
Lizard liz2;
Animal * pAnimal1 = &liz1;
Animal * pAnimal2 = &liz2;
*pAnimal1 = *pAnimal2;      // partial assignment! 
```
There are two problems here. 
1. **Partial assignment**. 
   The assignment operator invoked on the last line is that of the `Animal` class, 
   even though the objects involved are of type `Lizard`. 
   As a result, only the `Animal` part of `liz1` will be modified. 
   This is a _partial assignment_. 
   After the assignment, `liz1`’s Animal members have the values they got from `liz2`, 
   but `liz1`’s Lizard members remain unchanged. 
2. **Assignment operations through de-referenced pointers are valid and commonly-seen**. 
   It’s common to make assignments to objects through pointers, 
   especially for experienced C programmers who have moved to C++. 
   That being the case, we’d like to make the assignment behave in a more reasonable fashion. 
   As Item 32 points out, our classes should be easy to use correctly and difficult to use incorrectly, 
   and the classes in the hierarchy above are easy to use incorrectly.


One approach to the problem is to make the assignment operators `virtual`. 
If `Animal::operator=` were `virtual`, the assignment would invoke the `Lizard::operator=`, 
which is certainly the correct one to call. 
However, look what happens if we declare the assignment operators `virtual`:
```c++
class Animal 
{
public:
    virtual Animal & operator=(const Animal & rhs);
    // ...
};

class Lizard : public Animal 
{
public:
    virtual Lizard & operator=(const Animal & rhs);  // NOT Lizard & rhs!
    // ...
};

class Chicken : public Animal 
{
public:
    virtual Chicken & operator=(const Animal & rhs);  // NOT Chicken & rhs!
    // ...
};
```
The rules of C++ force us to 
**declare identical parameter types for a `virtual` function in every class in which it is declared.** 
That means `Lizard::operator=` and `Lizard::operator=` classes 
must be prepared to accept any kind of `Animal` object on the right-hand side of an assignment. 
That, in turn, means we have to confront the fact that code like the following is legal:
```c++
Lizard liz;
Chicken chick;
Animal * pAnimal1 = &liz;
Animal * pAnimal2 = &chick;
*pAnimal1 = *pAnimal2;       // assigning a chicken to a lizard!
```
This is a _mixed-type assignment_: 
a `Lizard` is on the left and a `Chicken` is on the right. 
Mixed-type assignments aren’t usually a problem in C++, 
because the language’s strong typing generally renders them illegal. 
By making `Animal::operator=` `virtual`, however, 
we opened the door to such mixed-type operations.


This puts us in a difficult position. 
We’d like to allow same-type assignments through pointers, 
but we’d like to forbid mixed-type assignments through those same pointers. 
In other words, we want to _allow_ this,
```c++
Animal * pAnimal1 = &liz1;
Animal * pAnimal2 = &liz2;
*pAnimal1 = *pAnimal2;       // assign a lizard to a lizard
```
but we want to _prohibit_ this:
```c++
Animal * pAnimal1 = &liz;
Animal * pAnimal2 = &chick;
*pAnimal1 = *pAnimal2;       // assign a chicken to a lizard
```
Distinctions such as these can be made only at runtime, 
because sometimes assigning `*pAnimal2` to `*pAnimal1` is valid, 
sometimes it’s not. 
We thus enter the murky world of type-based runtime errors. 
In particular, we need to signal an error inside `operator=` 
if we’re faced with a mixed-type assignment, 
but if the types are the same, 
we want to perform the assignment in the usual fashion.


We can use a `dynamic_cast` to implement this behavior.
Here’s how to do it for `Lizard::operator=`:
```c++
Lizard & Lizard::operator=(const Animal & rhs)
{
    // make sure rhs is really a lizard
    const Lizard & rhs_liz = dynamic_cast<const Lizard &>(rhs);
    // proceed with a normal assignment of rhs_liz to *this;
    // ...
}
```
This function assigns `rhs` to `*this` only if `rhs` is really a `Lizard`. 
If it’s not, the function propagates the `std::bad_cast` exception 
thrown by `dynamic_cast` when the cast to a reference fails. 


Even without worrying about exceptions, this function seems needlessly complicated and expensive:
the `dynamic_cast` must consult a `std::type_info` structure (see Item 24). 
In the common case where one `Lizard` object is assigned to another:
```c++
Lizard liz1, liz2;
liz1 = liz2;
// no need to perform a dynamic_cast: this assignment must be valid
```
We can handle this case without paying for the complexity or cost of a `dynamic_cast` 
by adding to `Lizard` the conventional assignment operator:
```c++
class Lizard : public Animal
{
public:
    virtual Lizard & operator=(const Animal & rhs);
    Lizard & operator=(const Lizard & rhs);
    // ...
};

// calls operator= taking a const Lizard &
Lizard liz1, liz2;
liz1 = liz2;

// calls operator= taking a const Animal &
Animal * pAnimal1 = &liz1;
Animal * pAnimal2 = &liz2;
*pAnimal1 = *pAnimal2;
```
In fact, given this latter `operator=`, 
it’s simplicity itself to implement the former one in terms of it:
```c++
Lizard & Lizard::operator=(const Animal & rhs)
{
    return operator=(dynamic_cast<const Lizard &>(rhs));
}
```






### 📌 Item 34: Understand how to combine C++ and C in the same program






### 📌 Item 35: Familiarize yourself with the language standard



