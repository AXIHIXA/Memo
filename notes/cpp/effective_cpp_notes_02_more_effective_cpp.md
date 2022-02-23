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

- Polymorphism does **not** work with pointer arithmetic. 
  Array operations always involve pointer arithmetic, so they **do** not work with polymorphism. 


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
if you avoid having a concrete (non-abstract) class (like `AVL`) inherit from another concrete class (such as `BST`).
As Item 33 explains, designing your software so that concrete classes never inherit from one another has many benefits. 
I encourage you to turn to Item 33 and read all about them. 






### 📌 Item 4: Avoid gratuitous default constructors


A default constructor (i.e., a constructor that can be called with no arguments)
is the C++ way of saying you can get something for nothing. 
Constructors initialize objects, 
so default constructors initialize objects without any information from the place where the object is being created. 
Sometimes this makes perfect sense. 
Objects that act like numbers, for example, may reasonably be initialized to zero or to undefined values. 
Objects that act like pointers may reasonably be initialized to null or to undefined values. 
Data structures like linked lists, hash tables, maps, and the like may reasonably be initialized to empty containers. 


Not all objects fall into this category. 
For many objects, there is no reasonable way to 
perform a complete initialization in the absence of outside information. 
For example, an object representing an entry in an address book 
makes no sense unless the name of the thing being entered is provided. 
In some companies, all equipment must be tagged with a corporate ID number, 
and creating an object to model a piece of equipment in such companies is nonsensical 
unless the appropriate ID number is provided.


In a perfect world, 
classes in which objects could reasonably be created from nothing would contain default constructors 
and classes in which information was required for object construction would not. 
If a class lacks a default constructor, there are restrictions on how you can use that class.
Consider a class for company equipment 
in which the corporate ID number of the equipment is a mandatory constructor argument:
```c++
class EquipmentPiece 
{
public:
    EquipmentPiece(int IDNumber);
    // ...
};
```
Because `EquipmentPiece` lacks a default constructor, its use may be problematic in three contexts. 
The first is the _creation of arrays_. 
There is, in general, no way to specify constructor arguments for objects in arrays, 
so it is not usually possible to create arrays of `EquipmentPiece` objects:
```c++
EquipmentPiece bestPieces[10];                         // Error! No EquipmentPiece::EquipmentPiece()
EquipmentPiece * bestPieces = new EquipmentPiece[10];  // Error! Same problem
```
There are three ways to get around this restriction. 
A solution for non-heap arrays is to provide the necessary arguments 
at the point where the array is defined:
```c++
EquipmentPiece bestPieces[] = 
        { 
            EquipmentPiece(ID1),
            EquipmentPiece(ID2),
            EquipmentPiece(ID3),
            ...,
            EquipmentPiece(ID10)
        };
```
Unfortunately, there is **no** way to extend this strategy to heap arrays.
A more general approach is to use an array of pointers instead of an array of objects:
```c++
using PEP = EquipmentPiece *;    // a PEP is a pointer to an EquipmentPiece
PEP bestPieces[10];              // fine, no constructors called
PEP * bestPieces = new PEP[10];  // also fine
```
Each pointer in the array can then be made to point to a different `EquipmentPiece` object:
```c++
for (std::size_t i = 0; i < 10; ++i)
{
    bestPieces[i] = new EquipmentPiece(someIDNumber);
}
```
There are two disadvantages to this approach. 
First, you have to remember to `delete` all the objects pointed to by the array. 
If you forget, you have a resource leak. 
Second, the total amount of memory you need increases, 
because you need the space for the pointers as well as the space for the `EquipmentPiece` objects.
You can avoid the space penalty if you allocate the raw (uninitialized) memory for the array, 
then use _placement `new`_ (see Item 8) to construct the `EquipmentPiece` objects in the memory:
```c++
// Allocate enough raw memory for an array of 10 EquipmentPiece objects; 
// See Item 8 for details on the operator new[] function
void * rawMemory = operator new[](10 * sizeof(EquipmentPiece));

// Make bestPieces point to it so it can be treated as an EquipmentPiece array
EquipmentPiece * bestPieces = static_cast<EquipmentPiece *>(rawMemory);

// Construct the EquipmentPiece objects in the memory using placement new (see Item 8)
for (std::size_t i = 0; i < 10; ++i)
{
    new (bestPieces + i) EquipmentPiece(someIDNumber);
}
```
Notice that you still have to provide a constructor argument for each `EquipmentPiece` object. 
This technique (as well as the array-of-pointers idea) 
allows you to create arrays of objects when a class lacks a default constructor; 
it doesn’t show you how to bypass required constructor arguments. 
There is no way to do that. 
If there were, it would defeat the purpose of constructors, 
which is to guarantee that objects are initialized.


The downside to using placement `new`, 
aside from the fact that most programmers are unfamiliar with it (which will make maintenance more difficult), 
is that you must manually call destructors on the objects
in the array when you want them to go out of existence, 
then you must manually deallocate the raw memory by calling `operator delete[]` (see Item 8):
```c++
// Destruct the objects in bestPieces 
// in the inverse order in which they were constructed
for (std::size_t i = 9; 0 <= i; --i)
{
    bestPieces[i].~EquipmentPiece();
}

// deallocate the raw memory
operator delete[](rawMemory);
```
If you forget this requirement and use the normal array-deletion syntax,
your program will behave _unpredictably_. 
That’s because `delete`ing a pointer that didn’t come from the `new` operator is _undefined behavior_:
```c++
// undefined! bestPieces didn’t come from the new operator
delete [] bestPieces;
```
The second problem with classes lacking default constructors is that
they are ineligible for use with many template-based container classes. 
That’s because it’s a common requirement for such templates that the
type used to instantiate the template provide a default constructor. 
This requirement almost always grows out of the fact that inside the template, 
an array of the template parameter type is being created. 
For example, a template for an `Array` class might look something like this:
```c++
template <typename T>
class Array 
{
public:
    Array(std::size_t size);
    // ...
    
private:
    T * data;
};

template <typename T>
Array<T>::Array(std::size_t size)
{
    // calls T::T() for each element of the array
    data = new T[size]; 
    // ... 
}
```
In most cases, careful template design can eliminate the need for a default constructor. 
For example, `std::vector` template has **no** requirement that its type parameter have a default constructor. 
Unfortunately, many templates are not carefully designed. 
That being the case, classes without default constructors will be incompatible with many templates. 


The final consideration in the to-provide-a-default-constructor-or-not-to-provide-a-default-constructor dilemma 
has to do with virtual base classes. 
Virtual base classes lacking default constructors are a pain to work with. 
That’s because the arguments for virtual base class constructors must be provided 
by the most derived class of the object being constructed. 
As a result, a virtual base class lacking a default constructor requires that _all_ classes derived from that class
must understand the meaning of and provide for the virtual base class’s constructors’ arguments. 
Authors of derived classes neither expect nor appreciate this requirement. 


Because of the restrictions imposed on classes lacking default constructors,
some people believe all classes should have them, 
even if a default constructor doesn’t have enough information to fully initialize objects of that class. 
For example, adherents to this philosophy might modify `EquipmentPiece` as follows:
```c++
class EquipmentPiece 
{
public:
    explicit EquipmentPiece(int IDNumber = UNSPECIFIED);
    // ...
    
private:
    // magic ID number value meaning no ID was specified
    static constexpr int UNSPECIFIED;  
};
```
This allows `EquipmentPiece` objects to be created like this:
```c++
EquipmentPiece e;  // now okay
```
Such a transformation almost always complicates the other member functions of the class, 
because there is no longer any guarantee that 
the fields of an `EquipmentPiece` object have been meaningfully initialized.
Assuming it makes no sense to have an `EquipmentPiece` without an ID field, 
most member functions must check to see if the ID is present. 
If it’s not, they’ll have to figure out how to stumble on anyway.
Often it’s not clear how to do that, and many implementations choose a solution that offers nothing but expediency: 
they throw an exception or they call a function that terminates the program. 
When that happens, it’s difficult to argue that the overall quality of the software has been improved 
by including a default constructor in a class where none was warranted.


Inclusion of meaningless default constructors affects the efficiency of classes, too. 
If member functions have to test to see if fields have truly been initialized, 
clients of those functions have to pay for the time those tests take. 
Furthermore, they have to pay for the code that goes into those tests, 
because that makes executables and libraries bigger.
They also have to pay for the code that handles the cases where the tests fail. 
All those costs are avoided if a class’s constructors ensure that 
all fields of an object are correctly initialized. 
Often default constructors can’t offer that kind of assurance, 
so it’s best to avoid them in classes where they make no sense. 
That places some limits on how such classes can be used, 
but it also guarantees that when you do use such classes, 
you can expect that the objects they generate are fully initialized and are efficiently implemented. 






### 🎯 Chapter 2. Operators

### 📌 Item 5: Be wary of user-defined conversion functions


C++ allows compilers to perform implicit conversions between types. 
E.g., C++ allows implicit conversions 
from `char` to `int`, from `short` to `double`, from `int` to `short`, and from `double` to `char`. 


You can’t do anything about such conversions, because they’re hardcoded into the language. 
When you add your own types, however, you have more control, 
because you can choose whether to provide the functions compilers are allowed to use for implicit type conversions.


Two kinds of functions allow compilers to perform user-defined implicit conversions:
1. Non-`explicit` Single-argument Constructors; 
2. Implicit Type Conversion Operators. 


A single-argument constructor is a constructor that may be called with only one argument. 
Such a constructor may declare a single parameter or it may declare multiple parameters, 
with each parameter after the first having a default value. 
```c++
class Name
{
public:
    // converts std::string to Name
    Name(const std::string & s); 
    // ...
};

class Rational
{
public:
    // converts int to Rational
    Rational(int numerator = 0, int denominator = 1);
    // ...
};
```
An implicit type conversion operator is simply a member function with a strange-looking name: 
the word `operator` followed by a type specification. 
You are **not** allowed to specify a type for the function’s return value, 
because the type of the return value is basically just the name of the function. 
For example, to allow `Rational` objects to be implicitly converted to `double`s 
(which might be useful for mixed-mode arithmetic involving `Rational` objects), 
you might define class `Rational` like this:
```c++
class Rational
{
public:
    // ...

    // converts Rational to double
    operator double() const; 
};

// r has the value 1/2
Rational r(1, 2);    
// converts r to a double, then does multiplication
double d = 0.5 * r;  
```
You usually **don’t** want to provide type conversion functions of _any_ ilk. 


The fundamental problem is that such functions often end up 
being called when you neither want nor expect them to be. 
The result can be incorrect and unintuitive program behavior 
that is maddeningly difficult to diagnose.


Let us deal first with implicit type conversion operators, 
as they are the easiest case to handle. 
Suppose you have a class for rational numbers similar to the one above, 
and you’d like to print `Rational` objects as if they were a built-in type. 
That is, you’d like to be able to do this:
```c++
Rational r(1, 2);
std::cout << r << '\n';  // should print "1/2"
```
Further suppose you forgot to write an `operator<<` for `Rational` objects. 
You would probably expect that the attempt to print `r` would fail,
because there is no appropriate `operator<<` to call. 
You would be mistaken. 
Your compilers, faced with a call to a function called `operator<<` that takes a `Rational`, 
would find that no such function existed, 
but they would then try to find an acceptable sequence of implicit type conversions 
they could apply to make the call succeed. 
The rules defining which sequences of conversions are acceptable are complicated,
but in this case your compilers would discover they could make the call succeed 
by implicitly converting `r` to a `double` by calling `Rational::operator double`. 
The result of the code above would be to print `r` as a floating point number, 
**not** as a rational number. 
This is hardly a disaster, 
but it demonstrates the disadvantage of implicit type conversion operators: 
Their presence can lead to the wrong function being called 
(i.e., one other than the one intended).


The solution is to replace the operators with equivalent functions 
that don’t have the syntactically magic names. 
For example, to allow conversion of a `Rational` object to a `double`, 
replace `operator double` with a function called something like `asDouble`: 
```c++
class Rational 
{
public:
    // ...
    
    // converts Rational to double
    double asDouble() const;  
};
```
Such a member function must be called explicitly:
```c++
Rational r(1, 2);
std::cout << r << '\n';             // Error! No operator<< for Rationals
std::cout << r.asDouble() << '\n';  // Fine, prints r as a double
```
In most cases, the inconvenience of having to call conversion functions explicitly is more 
than compensated for by the fact that unintended functions can no longer be silently invoked.
E.g., `std::string` type contains **no** implicit conversion 
from a `std::string` object to a C-style `char *`. 
Instead, there’s an explicit member function, 
`std::string::c_str`, that performs that conversion. 


Implicit conversions via single-argument constructors are more difficult to eliminate. 
Furthermore, the problems these functions cause are in many cases
worse than those arising from implicit type conversion operators.


As an example, consider a class template for array objects. 
These arrays allow clients to specify upper and lower index bounds:
```c++
template <typename T>
class Array 
{
public:
    Array(std::size_t lowerIndexBound, std::size_t upperIndexBound);
    Array(std::size_t size);
    T & operator[](std::size_t index);
    // ...
};
```
The first constructor in the class allows clients to specify a range of array indices. 
As a two-argument constructor, this function is ineligible for use as a type-conversion function. 
The second constructor, which allows clients to define `Array` objects 
by specifying only the number of elements in the array 
(in a manner similar to that used with built-in arrays), is different. 
It can be used as a type conversion function, and that can lead to problems. 
The following code
```c++
bool operator==(const Array<int> & lhs, const Array<int> & rhs);

Array<int> a(10);
Array<int> b(10);

for (std::size_t i = 0; i < 10; ++i)
{
    if (a == b[i])  // oops! "a" should be "a[i]"
    {
        // do something for when a[i] and b[i] are equal;
    } 
    else
    {
        // do something for when they’re not;
    }
}
```
is essentially the same as the following via user-defined implicit conversions:
```c++
for (std::size_t i = 0; i < 10; ++i)
{
    if (a == static_cast<Array<int>>(b[i])) ...
}
```
The drawbacks to implicit type conversion operators can be avoided by
declaring single-parameter constructors `explicit`. 
Compilers are prohibited from invoking `explicit` constructors 
for purposes of implicit type conversion. 
Explicit conversions are still legal.
```c++
template <typename T>
class Array 
{
public:
    explicit Array(std::size_t size); 
    // ...
};

// okay, explicitly calling the explicit constructor
Array<int> a(10);               
Array<int> b(10);

// error! no such implicit conversion
if (a == b[i]) ...              

// okay, conversion is explicit (but the logic of the code is suspect)
if (a == Array<int>(b[i])) ...
if (a == static_cast<Array<int>>(b[i])) ...
if (a == (Array<int>) b[i]) ...
```
A sad story when `explicit` constructors were not introduced to C++ standard.
The implicit conversion only allows one user-defined conversion 
(conversion constructor and conversion operator), 
otherwise the _implicit conversion sequence_ could be infinitely long. 
Knowing this fact, we write the following ugly (but working) code: 
```c++
template <typename T>
class Array
{
public:
    class Size
    {
    public:
        Size(std::size_t size) : mSize(size) {}
        std::size_t size() const { return mSize; }

    private:
        std::size_t mSize;
    };

    Array(std::size_t lowerIndexBound, std::size_t upperIndexBound);

    // Now the conversion from int to Array
    // requires two adjacent user-defined conversions, 
    // blocking the path of implicit conversion
    Array(Size size);
    
    // ...
};

// Good, 
// user calling Array<int>::Array(Array::Size) with argument of type int, 
// just implicitly convert int into Array::Size with one user-defined conversion. 
Array<int> a(10);

bool operator==(const Array<int> & lhs, const Array<int> & rhs);

Array<int> a(10);
Array<int> b(10);

for (int i = 0; i < 10; ++i)
{
    // Oops! "a" should be "a[i]". 
    // This is now an error! 
    if (a == b[i]) ...
}
```
The use of the `Array::Size` in this example might look like a special-purpose hack,
but it’s actually a specific instance of a more general technique. 
Classes like `Array::Size` are often called _proxy classes_ (see Item 30),
because each object of such a class stands for (is a proxy for) some other object. 
An `Array::Size` object is really just a stand-in for the integer used to specify the size of the `Array` being created. 
Proxy objects can give you control over aspects of your software’s behavior, 
in this case implicit type conversions, that is otherwise beyond your grasp, 
so it’s well worth your while to learn how to use them. 






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

Future-tense thinking simply adds a few additional considerations:
- Provide complete classes, e.g., virtual destructor, even if some parts aren’t currently used.
  When new demands are made on your classes, you’re less likely to have to go back and modify them.
- Design your interfaces to facilitate common operations and prevent common errors. 
  Make the classes easy to use correctly, hard to use incorrectly. 
  For example, prohibit copying and assignment for classes where those operations make no sense. 
  Prevent partial assignments (see Item 33).
- If there is no great penalty for generalizing your code, generalize it.
  For example, if you are writing an algorithm for tree traversal, 
  consider generalizing it to handle any kind of directed acyclic graph.






### 📌 Item 33: Make non-leaf classes abstract

- Never write non-leaf concrete base classes to avoid problems 
  like partial assignment via dereferenced polymorphic pointers.
  Make non-leaf classes abstract 
  (e.g., by adding pure virtual destructors and implement them outside the class).


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
    Lizard & operator=(const Animal & rhs) override;  // NOT const Lizard & rhs!
    // ...
};

class Chicken : public Animal 
{
public:
    Chicken & operator=(const Animal & rhs) override;  // NOT const Chicken & rhs!
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
    Lizard & operator=(const Animal & rhs) override;
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
This function attempts to cast `rhs` to be a `Lizard`. 
If the cast succeeds, the normal class assignment operator is called. 
Otherwise, a `std::bad_cast` exception is thrown.


Checking types at runtime and using `dynamic_casts` are expensive. 
For one thing, some compilers lack support for `dynamic_cast`, so code that uses it, 
though theoretically portable, is not necessarily portable in practice. 
More importantly, it requires that clients of `Lizard` and `Chicken` be prepared to catch `std::bad_cast` exceptions 
and do something sensible with them each time they perform an assignment. 
There just aren’t that many programmers who are willing to program that way. 
If they don’t, it’s not clear we’ve gained a whole lot over our original situation
where we were trying to guard against partial assignments.


Given this rather unsatisfactory state of affairs regarding virtual assignment operators, 
it makes sense to regroup and try to find a way 
to prevent clients from making problematic assignments in the first place. 
If such assignments are rejected during compilation, we don’t have to worry about them doing the wrong thing.


The easiest way to prevent such assignments is to make `operator=` `private` in `Animal`. 
That way, lizards can be assigned to lizards and chickens can be assigned to chickens, 
but partial and mixed-type assignments are forbidden:
```c++
class Animal
{
private:
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

// fine
Lizard liz1, liz2;
liz1 = liz2;

// also fine
Chicken chick1, chick2;
chick1 = chick2;

// error! attempt to call private Animal::operator=
Animal * pAnimal1 = &liz1;
Animal * pAnimal2 = &chick1;
*pAnimal1 = *pAnimal2; 
```
Unfortunately, `Animal` is a concrete class, 
and this approach also makes assignments between `Animal` objects illegal:
```c++
// error! attempt to call private Animal::operator=
Animal animal1, animal2;
animal1 = animal2;
```
Moreover, it makes it impossible to implement the `Lizard` and `Chicken` assignment operators correctly, 
because assignment operators in derived classes are responsible for calling assignment operators in their base classes:
```c++
Lizard & Lizard::operator=(const Lizard & rhs)
{
    if (this == &rhs)
    {
        return *this;
    }

    // Error! 
    // Attempt to call private function. 
    // But Lizard::operator= must call this function to
    // assign the Animal parts of *this!
    Animal::operator=(rhs); 
    // ...
}
```
We can solve this latter problem by declaring `Animal::operator=` `protected`,
but the conundrum of allowing assignments between `Animal` objects 
while preventing partial assignments of `Lizard` and `Chicken` objects through Animal pointers remains.


The easiest thing is to eliminate the need to allow assignments between `Animal` objects, 
and the easiest way to do that is to make `Animal` an abstract class. 
As an abstract class, `Animal` can’t be instantiated, 
so there will be no need to allow assignments between `Animals`. 
Of course, this leads to a new problem, 
because our original design for this system presupposed that `Animal` objects were necessary.
There is an easy way around this difficulty.
Instead of making `Animal` itself abstract, we create a new class `AbstractAnimal`, 
say, consisting of the common features of `Animal`, `Lizard`, and `Chicken` objects, and we make _that_ class abstract. 
Then we have each of our concrete classes inherit from `AbstractAnimal`. 
The revised hierarchy looks like this,
```c++
class AbstractAnimal
{
public:
    virtual ~AbstractAnimal() = 0;

protected:
    AbstractAnimal & operator=(const AbstractAnimal & rhs);
    // ...
};

AbstractAnimal::~AbstractAnimal() = default;

class Animal : public AbstractAnimal
{
public:
    Animal & operator=(const Animal & rhs);
    // ...
};

class Lizard : public AbstractAnimal
{
public:
    Lizard & operator=(const Lizard & rhs);
    // ...
};

class Chicken : public AbstractAnimal
{
public:
    Chicken & operator=(const Chicken & rhs);
    // ...
};
```
This design gives you everything you need. 
Homogeneous assignments are allowed for lizards, chickens, and animals; 
partial assignments and heterogeneous assignments are prohibited; 
and derived class assignment operators may call the assignment operator in the base class. 
Furthermore, none of the code written in terms of the Animal,
Lizard, or Chicken classes requires modification, because these
classes continue to exist and to behave as they did before AbstractAnimal
was introduced. Sure, such code has to be recompiled, but
that’s a small price to pay for the security of knowing that assignments
that compile will behave intuitively and assignments that would behave
unintuitively won’t compile.


For all this to work, `AbstractAnimal` must be abstract: 
it must contain at least one pure virtual function. 
In most cases, coming up with a suitable function is not a problem, 
but on rare occasions you may find yourself facing the need to create a class like `AbstractAnimal` 
in which none of the member functions would naturally be declared pure virtual. 
In such cases, the conventional technique is to make the destructor a pure virtual function; 
that’s what’s shown above. 
In order to support polymorphism through pointers correctly, 
base classes need virtual destructors anyway, 
so the only cost associated with making such destructors pure virtual is the inconvenience of 
having to implement them outside their class definitions. 


If the notion of implementing a pure virtual function strikes you as odd, you just haven’t been getting out enough. 
Declaring a function pure virtual **doesn’t** mean ~~it has no implementation~~, it means: 
1. The current class is abstract; 
2. Any concrete class inheriting from the current class 
   must declare the function as an _impure_ virtual function (i.e., without the `= 0`). 


Most pure virtual functions are never implemented, but pure virtual destructors are a special case. 
They must be implemented because they are called whenever a derived class destructor is invoked. 
Furthermore, they often perform useful tasks, such as releasing resources (see Item 9) or logging messages. 
Implementing pure virtual functions may be uncommon in general, 
but for pure virtual destructors, it’s not just common, it’s mandatory. 


You may have noticed that this discussion of assignment through base class pointers 
is based on the assumption that concrete derived classes like `Lizard` contain data members. 
If there are no data members in a derived class, you might point out, there is no problem, 
and it would be safe to have a data-less concrete class inherit from another concrete class. 
However, just because a class has no data now is no reason to conclude that it will have no data in the future. 
If it might have data members in the future, 
all you’re doing is postponing the problem until the data members are added, 
in which case you’re merely trading short-term convenience for long-term grief (see also Item 32).


Replacement of a concrete base class like `Animal` with an abstract base class like `AbstractAnimal` 
yields benefits far beyond simply making the behavior of `operator=` easier to understand. 
It also reduces the chances that you’ll try to treat arrays polymorphically, 
the unpleasant consequences of which are examined in Item 3. 
The most significant benefit of the technique, however, occurs at the design level,
because replacing concrete base classes with abstract base classes 
forces you to explicitly recognize the existence of useful abstractions.
That is, it makes you create new abstract classes for useful concepts,
even if you aren’t aware of the fact that the useful concepts exist.
If you have two concrete classes `C1` and `C2` and you’d like `C2` to publicly inherit from `C1`, 
you should transform that two-class hierarchy into a three-class hierarchy 
by creating a new abstract class `A` and having both `C1` and `C2` publicly inherit from it:
```
C1 <- C2

A <- C1
  <- C2
```
The primary value of this transformation is that it forces you to identify the abstract class `A`. 
Clearly, `C1` and `C2` have something in common; that’s why they’re related by public inheritance. 
With this transformation, you must identify what that something is. 
Furthermore, you must formalize the something as a class in C++, 
at which point it becomes more than just a vague something, 
it achieves the status of a formal abstraction, 
one with well-defined member functions and well-defined semantics.


Although every class represents some kind of abstraction, 
so we should **not** create two classes for every concept in our hierarchy, 
one being abstract (to embody the abstract part of the abstraction) 
and one being concrete (to embody the object-generation part of the abstraction)? 
If we do, we will end up with a huge hierarchy with too many classes. 
which is difficult to understand, hard to maintain, and expensive to compile.
That is not the goal of object-oriented design.


The goal is to identify useful abstractions 
and to force them (and only them) into existence as abstract classes.
The need for an abstraction in one context may be coincidental, 
but the need for an abstraction in more than one context is usually meaningful. 
Useful abstractions, then, are those that are needed in more than one context. 


This is precisely why the transformation from concrete base class to abstract base class is useful: 
it forces the introduction of a new abstract class 
only when an existing concrete class is about to be used as a base class, 
i.e., when the class is about to be (re)used in a new context. 
Such abstractions are useful, because they have, through demonstrated need, shown themselves to be so.


The first time a concept is needed, 
we can’t justify the creation of both an abstract class (for the concept) 
and a concrete class (for the objects corresponding to that concept), 
but the second time that concept is needed, we can justify the creation of both the abstract and the concrete classes. 
The transformation I’ve described simply mechanizes this process,
and in so doing it forces designers and programmers to represent explicitly those abstractions that are useful, 
even if the designers and programmers are not consciously aware of the useful concepts. 
It also happens to make it a lot easier to bring sanity to the behavior of assignment operators.


Let’s consider a brief example. 
Suppose you’re working on an application that deals with moving information between computers on a network 
by breaking it into packets and transmitting them according to some protocol. 
All we’ll consider here is the class or classes for representing packets. 
We’ll assume such classes make sense for this application.


Suppose you deal with only a single kind of transfer protocol and only a single kind of packet. 
Perhaps you’ve heard that other protocols and packet types exist, 
but you’ve never supported them, nor do you have any plans to support them in the future. 
Should you make an abstract class for packets (for the concept that a packet represents) 
as well as a concrete class for the packets you’ll actually be using? 
If you do, you could hope to add new packet types later without changing the base class for packets. 
That would save you from having to recompile packet-using applications if you add new packet types. 
But that design requires two classes, and right now you need only one (for the particular type of packets you use). 
Is it worth complicating your design now to allow for future extension that may never take place?


There is no unequivocally correct choice to be made here, 
but experience has shown it is nearly impossible to design good classes for concepts we do not understand well. 
If you create an abstract class for packets, how likely are you to get it right, 
especially since your experience is limited to only a single packet type? 
Remember that you gain the benefit of an abstract class for packets 
only if you can design that class so that future classes can inherit from it without its being changed in any way. 
(If it needs to be changed, you have to recompile all packet clients, and you’ve gained nothing.)


It is unlikely you could design a satisfactory abstract packet class 
unless you were well versed in many different kinds of packets and in the varied contexts in which they are used. 
Given your limited experience in this case, 
the advice would be **not** to define an abstract class for packets, 
adding one later only if you find a need to inherit from the concrete packet class.


The transformation here is merely one way to identify the need for abstract classes, not the only way.


As is often the case in such matters, brash reality sometimes intrudes on the peaceful ruminations of theory. 
Third-party C++ class libraries are proliferating with gusto, 
and what are you to do if you find yourself wanting to create a concrete class 
that inherits from a concrete class in a library to which you have only read access?


You can’t modify the library to insert a new abstract class, so your choices are both limited and unappealing:
- Derive your concrete class from the existing concrete class, 
  and put up with the assignment-related problems we examined at the beginning of this Item. 
  You’ll also have to watch out for the array-related pitfalls described in Item 3.
- Try to find an abstract class higher in the library hierarchy that does most of what you need, 
  then inherit from that class. 
  Of course, there may not be a suitable class, and even if there is, 
  you may have to duplicate a lot of effort that has already been put into the implementation 
  of the concrete class whose functionality you’d like to extend.
- Make do with what you’ve got. 
  Use the concrete class that’s in the library and modify your software so that the class suffices. 
  Write non-member functions to provide the functionality you’d like to but can't add to the class. 
  The resulting software may not be as clear, as efficient, as maintainable, or as extensible as you’d like,
  but at least it will get the job done. 
- Implement your new class in terms of the library class you’d like to inherit from. 
  For example, you could have an object of the library class as a data member, 
  then reimplement the library class’s interface in your new class.
  This strategy requires that you be prepared to update your class 
  each time the library vendor updates the class on which you’re dependent. 
  It also requires that you be willing to forgo the ability to redefine virtual functions declared in the library class, 
  because you can’t redefine virtual functions unless you inherit them.
```c++
// this is the library class
class Window
{
public:
    virtual void resize(int newWidth, int newHeight);

    virtual void repaint() const;

    int width() const;

    int height() const;
};

// this is the class you wanted to inherit from Window
class SpecialWindow
{
public:
    // ...
    
    // pass-through implementations of non-virtual functions
    int width() const
    {
        return w.width();
    }

    int height() const
    {
        return w.height();
    }

    // new implementations of "inherited" virtual functions
    virtual void resize(int newWidth, int newHeight);

    virtual void repaint() const;

private:
    Window w;
};
```
None of these choices is particularly attractive, 
so you have to apply some engineering judgment and choose the poison you find least un-appealing. 


Still, the general rule remains: **Non-leaf classes should be abstract**.
You may need to bend the rule when working with outside libraries,
but in code over which you have control, adherence to it will yield dividends 
in the form of increased reliability, robustness, comprehensibility, and extensibility throughout your software.






### 📌 Item 34: Understand how to combine C++ and C in the same program


In many ways, the things you have to worry about when making a program 
out of some components in C++ and some in C 
are the same as those you have to worry about when cobbling together a C program 
out of object files produced by more than one C compiler. 
There is **no** way to combine such files 
unless the different compilers agree on implementation-dependent features 
like the size of `long`s, 
the mechanism by which parameters are passed from caller to callee, 
and whether the caller or the callee orchestrates the passing. 
These pragmatic aspects of mixed-compiler software development 
are quite properly ignored by language standardization efforts, 
so the only reliable way to know that object files from compilers A and B 
can be safely combined in a program 
is to obtain assurances from the vendors of A and B 
that their products produce compatible output. 
This is as true for programs made up of C++ and C 
as it is for purely-C++ or purely-C programs, 
so before you try to mix C++ and C in the same program,
make sure your C++ and C compilers generate compatible object files.


Having done that, there are four other things you need to consider:
1. Name Mangling; 
2. Initialization of Statics; 
3. Dynamic Memory Allocation;
4. Data Structure Compatibility.


#### Name Mangling

_Name Mangling_ is the process through which your C++ compilers give each function in your program a unique name. 
In C, this process is unnecessary because you can’t overload function names.  
But, nearly all C++ programs have at least a few functions with the same name. 
(E.g., `<iostream>` declares several versions of `operator<<` and `operator>>`.) 
Overloading is incompatible with most linkers, 
because linkers generally take a dim view of multiple functions with the same name. 
Name mangling is a concession to the realities of linkers; 
in particular, to the fact that linkers usually insist on all function names being unique. 
As long as you stay within the confines of C++, name mangling is not likely to concern you. 
If you have a function name `drawLine` that a compiler mangles into `xyzzy`, 
you’ll always use the name `drawLine`, 
and you’ll have little reason to care that the underlying object files happen to refer to `xyzzy`.
It’s a different story if `drawLine` is in a C library. 
In that case, your C++ source file probably includes a header file that contains a declaration like this,
```c++
void drawLine(int x1, int y1, int x2, int y2);
```
and your code contains calls to `drawLine` in the usual fashion. 
Each such call is translated by your compilers 
into a call to the mangled name of that function, 
so when you write this,
```c++
// call to unmangled function name
drawLine(a, b, c, d);
```
your object files contain a function call that corresponds to this:
```c++
// call to mangled function mame
xyzzy(a, b, c, d);
```
But if `drawLine` is a C function, 
the object file that contains the compiled version of `drawLine` 
contains a function called `drawLine`; 
**no** name mangling has taken place. 
When you try to link the object files comprising your program together, you’ll get an error, 
because the linker is looking for a function called `xyzzy`, and there is no such function.


To solve this problem, you need a way to tell your C++ compilers not to mangle certain function names. 
You never want to mangle the names of functions written in other languages like C.
After all, if you call a C function named `drawLine`, it’s literally called `drawLine`, 
and your object code should contain a reference to that name, not to some mangled version of that name.


To suppress name mangling, use C++’s `extern "C"` directive:
```c++
// declare a function called drawLine; don’t mangle its name
#ifdef __cplusplus
extern "C"
{
#endif
void drawLine(int x1, int y1, int x2, int y2);
#ifdef __cplusplus
};
#endif
```
Technically, `extern "C"` means the function has _C linkage_.
The best way to view `extern "C"` is not as an assertion that the associated function _is written in C_, 
but as a statement that the function should be called as _if it were written in C_.


For example, if you were so unfortunate as to have to write a function in assembler, 
you could declare it `extern "C"`, too:
```c++
// this function is in assembler — don’t mangle its name
extern "C" void twiddleBits(unsigned char bits);
```
There is, by the way, no such thing as a “standard” name mangling algorithm. 
Different compilers are free to mangle names in different ways, and different compilers do. 
This is a good thing. 
If all compilers mangled names the same way, you might be lulled into thinking they all generated compatible code. 
The way things are now, if you try to mix object code from incompatible C++ compilers, 
there’s a good chance you’ll get an error during linking, 
because the mangled names won’t match up. 
This implies you’ll probably have other compatibility problems, too, 
and it’s better to find out about such incompatibilities sooner than later,

#### Initialization of Statics

Once you’ve mastered name mangling, you need to deal with the fact that in C++, 
lots of code can get executed before and after `main`. 
In particular, the constructors of static class objects and objects at global, namespace, and file scope 
are usually called _before_ the body of `main` is executed. 
This process is known as _static initialization_. 
This is in direct opposition to the way we normally think about C++ and C programs,
in which we view `main` as the entry point to execution of the program. 
Similarly, objects that are created through static initialization must
have their destructors called during _static destruction_; 
that process typically takes place _after_ `main` has finished executing.


To resolve the dilemma that `main` is supposed to be invoked first, 
yet objects need to be constructed before main is executed, 
many compilers insert a call to a special compiler-written function at the beginning of `main`, 
and it is this special function that takes care of static initialization. 
Similarly, compilers often insert a call to another special function at the end of `main` 
to take care of the destruction of static objects.
Code generated for `main` often looks as if `main` had been written like this:
```c++
int main(int argc, char * argv[])
{
    // generated by the implementation
    performStaticInitialization();

    // the statements you put in main go here
    
    // generated by the implementation
    performStaticDestruction();
}
```
Now don’t take this too literally. 
The functions `performStaticInitialization` and `performStaticDestruction` usually have much more cryptic names, 
and they may even be generated `inline`, in which case you won’t see any functions for them in your object files. 


The important point is this: 
If a C++ compiler adopts this approach to the initialization and destruction of static objects, 
such objects will be **neither** initialized **nor** destroyed unless `main` is written in C++.
Because this approach to static initialization and destruction is common, 
you should try to write main in C++ if you write any part of a software
system in C++.


Sometimes it would seem to make more sense to write `main` in C 
if most of a program is in C and C++ is just a support library. 
Nevertheless, there’s a good chance the C++ library contains static objects 
(even if it doesn’t now, it probably will in the future), 
so it’s still a good idea to write `main` in C++ if you possibly can. 
That doesn’t mean you need to rewrite your C code, however. 
Just rename the `main` you wrote in C to be `realMain`, 
then have the C++ version of `main` call `realMain`: 
```c++
// implement this function in C
extern "C"
int realMain(int argc, char * argv[]);

// write this in C++
int main(int argc, char * argv[])
{
    return realMain(argc, argv);
}
```
If you do this, it’s a good idea to put a comment above `main` explaining what is going on. 


If you can not write `main` in C++, you’ve got a problem, 
because there is no other portable way to ensure that constructors and destructors for static objects are called. 
This doesn’t mean all is lost, it just means you’ll have to work a little harder. 
Compiler vendors are well acquainted with this problem, 
so almost all provide some extralinguistic mechanism 
for initiating the process of static initialization and static destruction. 
For information on how this works with your compilers,
dig into your compilers’ documentation or contact their vendors. 

#### Dynamic Memory Allocation

For dynamic memory allocation, 
the C++ parts of a program use `new` and `delete` expressions, 
and the C parts of a program use `malloc` (and its variants) and `free`. 
As long as memory that came from `new` is deallocated via `delete` and
memory that came from `malloc` is deallocated via `free`, all is well.
However, calling `free` on a `new`ed pointer yields _undefined behavior_, 
as does `delete`ing a `malloc`ed pointer. 
The only thing to remember is to segregate rigorously 
your `new`s and `delete`s from your `malloc`s and `free`s.


Sometimes this is easier said than done. 
Consider the `strdup` function, which, though standard in neither C nor C++,
is nevertheless widely available:
```c++
// return a copy of the string pointed to by ps
char * strdup(const char * ps);
```
If a memory leak is to be avoided, the memory allocated inside `strdup`
must be deallocated by `strdup`’s caller. 
But how is the memory to be deallocated? 
By using `delete`? By calling `free`? 
If the `strdup` you’re calling is from a C library, it’s the latter. 
If it was written for a C++ library, it’s probably the former. 
What you need to do after calling `strdup`, then, 
varies not only from system to system, but also from compiler to compiler. 
To reduce such portability headaches, try to avoid calling functions that are neither in the standard library 
(see Item 35) nor available in a stable form on most computing platforms.

#### Data Structure Compatibility

Which brings us at long last to passing data between C++ and C programs.
There’s no hope of making C functions understand C++ features,
so the level of discourse between the two languages 
must be limited to those concepts that C can express. 
Thus, it should be clear there’s no portable way 
to pass objects or to pass pointers to member functions to routines written in C. 
C does understand normal pointers,
however, so, provided your C++ and C compilers produce compatible output, 
functions in the two languages can safely 
exchange pointers to objects and pointers to non-member or static functions. 
Naturally, `struct`s and variables of built-in types (e.g., `int`s, `char`s, etc.) 
can also freely cross the C++/C border.


Because the rules governing the layout of a `struct` in C++ are consistent with those of C, 
it is safe to assume that 
a structure definition that compiles in both languages 
is laid out the same way by both compilers.
Such `struct`s can be safely passed back and forth between C++ and C.
If you add non-`virtual` functions to the C++ version of the `struct`, 
its memory layout should not change, 
so objects of a `struct` (or `class`) containing only non-`virtual` functions 
should be compatible with their C brethren 
whose structure definition lacks only the member function declarations. 
Adding `virtual` functions ends the game, because the addition of virtual functions to a class 
causes objects of that type to use a different memory layout (see Item 24). 
Having a `struct` inherit from another `struct` (or `class`) usually changes its layout, too, 
so `struct`s with base `struct`s (or `class`es) are also poor candidates for exchange with C functions.


From a data structure perspective, it boils down to this: 
It is safe to pass data structures from C++ to C and from C to C++ 
provided the definition of those structures compiles in both C++ and C. 
Adding non-`virtual` member functions to the C++ version of a `struct` 
that’s otherwise compatible with C will probably not affect its compatibility, 
but almost any other change to the struct will.

#### Summary

If you want to mix C++ and C in the same program, remember the following simple guidelines:
- Make sure the C++ and C compilers produce compatible object files.
- Declare functions to be used by both languages `extern "C"`.
- If at all possible, write `main` in C++.
- Always use `delete` with memory from `new`; always use `free` with memory from `malloc`.
- Limit what you pass between the two languages to data structures that compile under C; 
  the C++ version of `struct`s may contain non-`virtual` member functions.






### 📌 Item 35: Familiarize yourself with the language standard



