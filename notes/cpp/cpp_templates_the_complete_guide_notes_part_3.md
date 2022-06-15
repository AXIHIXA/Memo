# _C++ Templates: The Complete Guide_ Notes

- _**C++ Templates**: The Complete Guide_ Second Edition by
    - David Vandevoorde
    - Nicolai M. Josuttis
    - Douglas Gregor
- Contents
  - [Part I The Basics](./cpp_templates_the_complete_guide_notes.md)
  - [Part II Templates in Depth](./cpp_templates_the_complete_guide_notes_part_2.md)
  - **[Part III Templates and Design](./cpp_templates_the_complete_guide_notes_part_3.md)**






## 🌱 Part III Templates and Design


### 🎯 Chapter 18 The Polymorphic Power of Templates


_Polymorphism_ is the ability to associate different specific behaviors with a single generic notation. 
Polymorphism literally refers to the condition of having many forms or shapes (from the Greek polymorphos). 


Polymorphism is also a cornerstone of the object-oriented programming paradigm, 
which in C++ is supported mainly through class inheritance and virtual functions. 
Because these mechanisms are (at least in part) handled at run time, we talk about _dynamic polymorphism_. 
This is usually what is thought of when talking about plain polymorphism in C++. 


However, templates also allow us to associate different specific behaviors with a single generic notation, 
but this association is generally handled at compile time, 
which we refer to as _static polymorphism_. 
In this chapter, we review the two forms of polymorphism and discuss which form **is** appropriate in which situations.
Note that Chapter 22 will discuss some ways to deal with polymorphism 
after introducing and discussing some design issues in between. 


#### 📌 18.1 Dynamic Polymorphism

Historically, C++ started with supporting polymorphism 
only through the use of inheritance combined with virtual functions.
Strictly speaking, macros can also be thought of as an early form of static polymorphism. 
However, they are left out of consideration because they are mostly orthogonal to the other language mechanisms.


The art of polymorphic design in this context consists of 
identifying a _common set of capabilities_ among related object types 
and declaring them as virtual function interfaces in a common base class. 
```c++
class GeoObj
{
public:
    virtual ~GeoObj() = default;
    virtual void draw() const = 0;
    Coord centroid() const = 0;
};

class Circle : public GeoObj
{
public:
    ~Circle() override = default;
    virtual void draw() const override;
    Coord centroid() const override;
};

class Line : public GeoObj
{
public:
    ~Line() override = default;
    virtual void draw() const override;
    Coord centroid() const override;
};
```
After creating concrete objects, client code can manipulate these objects 
through references or pointers to the common base class 
by using the _virtual function dispatch_ mechanism. 
Calling a virtual member function through a pointer or reference to a base class sub-object 
results in an invocation of the appropriate member of the specific (“most-derived”) concrete object.
```c++
void draw(GeoObj const & obj)
{
    obj.draw();
}

Coord distance(GeoObj const & a, GeoObj const & b)
{
    return distance(a.centroid(), b.centroid());
}

void foo()
{
    Line l;
    Circle c;
    draw(l);   // draw(GeoObj &) => Line::draw()
    draw(c);   // draw(GeoObj &) => Circle::draw()
    distance(l, c);
}
```

#### 📌 18.2 Static Polymorphism


Templates can also be used to implement polymorphism. 
However, they **don't** rely on the factoring of common behavior in base classes. 
Instead, the commonality is implicit in that: 
the different “shapes” of an application must support operations using common syntax 
(i.e., the relevant functions must have the same names). 
Concrete classes are defined independently from each other. 
The polymorphic power is then enabled when templates are instantiated with the concrete classes.
```c++
// concrete geometric object class Circle
// not derived from any class
class Circle
{
public:
    void draw() const;
    Coord centroid() const;
};

// concrete geometric object class Line
// not derived from any class
class Line
{
public:
    void draw() const;
    Coord centroid() const;
};

// GeoObj is template parameter
template <typename GeoObj>
void draw(GeoObj const & obj)
{
    obj.draw();
}

// GeoObj1, GeoObj2 are template parameters
template <typename GeoObj1, typename GeoObj2>
Coord distance(GeoObj1 const & a, GeoObj2 const & b)
{
    return distance(a.centroid(), b.centroid());
}

// nearly identical client code with dynamic polymorphism for this example
void foo()
{
    Line l;
    Circle c;
    draw(l);   // draw<Line>(GeoObj &) => Line::draw()
    draw(c);   // draw<Circle>(GeoObj &) => Circle::draw()
    distance(l, c);
}
```
As with `draw`, `GeoObj` can **no longer** be used as a concrete parameter type for `distance`. 
Instead, we provide for two template parameters, `GeoObj1` and `GeoObj2`, 
which enables different combinations of geometric object types to be accepted for the distance computation. 


However, heterogeneous collections can **no longer** be handled transparently. 
This is where the static part of static polymorphism imposes its constraint: 
All types must be determined at compile time. 
Instead, we can easily introduce different collections for different geometric object types. 
There is **no longer** a requirement that the collection be limited to pointers, 
which can have significant advantages in terms of performance and type safety. 


#### 📌 18.3 Dynamic versus Static Polymorphism


##### Terminology

- Polymorphism implemented via inheritance is _bounded and dynamic_:
  - _Bounded_ means that the interfaces of the types participating in the polymorphic behavior 
    are predetermined by the design of the common base class 
    (other terms for this concept are _invasive_ and _intrusive_). 
  - _Dynamic_ means that the binding of the interfaces is done at run time (dynamically).
- Polymorphism implemented via templates is _unbounded and static_:
  - _Unbounded_ means that the interfaces of the types participating in the polymorphic behavior 
    are **not** predetermined (other terms for this concept are _noninvasive_ and _nonintrusive_).
  - _Static_ means that the binding of the interfaces is done at compile time (statically).


In C++ parlance, _dynamic polymorphism_ and _static polymorphism_ are shortcuts 
for _bounded dynamic polymorphism_ and _unbounded static polymorphism_. 
In other languages, other combinations exist (e.g., Smalltalk provides unbounded dynamic polymorphism). 
However, in the context of C++, the more concise terms 
_dynamic polymorphism_ and _static polymorphism_ do not cause confusion. 

##### Strengths and Weaknesses

Dynamic polymorphism in C++ exhibits the following strengths:
- Heterogeneous collections are handled elegantly.
- The executable code size is potentially smaller 
  (because only one polymorphic function is needed, 
  whereas distinct template instances must be generated to handle different types).
- Code can be entirely compiled, 
  hence **no** implementation source must be published
  (distributing template libraries usually requires distribution of 
  the source code of the template implementations).


In contrast, the following can be said about static polymorphism in C++:
- Collections of built-in types are easily implemented. 
  More generally, the interface commonality need **not** be expressed through a common base class. 
- Generated code is potentially faster 
  (because **no** runtime virtual pointer indirection is needed 
  and non-virtual functions can be inlined much more often). 
- Concrete types that provide only partial interfaces can still be used 
  as long as only the partial interface is invoked. 


Static polymorphism is often regarded as more _type safe_ than dynamic polymorphism 
because all the bindings are checked at compile time. 
For example, there is little danger of inserting an object of the wrong type 
in a container instantiated from a template. 
However, in a container expecting pointers to a common base class, 
there is a possibility that these pointers unintentionally end up
pointing to complete objects of different types.


In practice, template instantiations can also cause some grief 
when different semantic assumptions hide behind identical-looking interfaces. 
For example, surprises can occur when a template that assumes an associative `operator+` 
is instantiated for a type that is not associative with respect to that operator. 
In practice, this kind of semantic mismatch occurs less often with inheritance-based hierarchies,
presumably because the interface specification is more explicitly specified.

##### Combining Both Forms

You could combine both forms of polymorphism. 
For example, you could derive different kinds of geometric objects from a common base class 
to be able to handle heterogeneous collections of geometric objects. 
However, you can still use templates to write code for a certain kind of geometric object. 


The combination of inheritance and templates is further described in Chapter 21. 
We will see (among other things) how the virtuality of a member function can be parameterized, 
and how an additional amount of flexibility is afforded to static polymorphism
using the inheritance-based_Curiously Recurring Template Pattern (CRTP)_.


#### 📌 18.4 Using Concepts

```c++
// C++20
template <typename T>
concept GeoObj = requires (T x)
{
    {x.draw()} -> void;
    {x.centeriod()} -> Coord;
};

template <GeoObj T>
void draw(T const & obj)
{
    t.draw();
}
```


#### 📌 18.5 New Forms of Design Patterns


N/A


#### 📌 18.6 Generic Programming


Static polymorphism leads to the concept of _generic programming_. 
However, there is **no** single agreed-on definition of _generic programming_ 
(just as there is **no** single agreed-on definition of _object-oriented programming_).
> Generic programming is programming with generic parameters 
> to finding the most abstract representation of efficient algorithms.   
> ...  
> Generic programming is a sub-discipline of computer science 
> that deals with finding abstract representations of 
> efficient algorithms, data structures, and other software concepts, 
> and with their systematic organization.  
> ...  
> Generic programming focuses on representing families of domain concepts. 


In the context of C++, generic programming is sometimes defined as _programming with templates_ 
(whereas object-oriented programming is thought of as _programming with virtual functions_). 
In this sense, just about any use of C++ templates could be thought of as an instance of generic programming. 
However, practitioners often think of generic programming as having an additional essential ingredient: 
Templates have to be designed in a framework for the purpose of enabling a multitude of useful combinations. 


By far the most significant contribution in this area is the _Standard Template Library (STL)_, 
which later was adapted and incorporated into the C++ standard library). 
The STL is a framework that provides a number of useful operations, called _algorithms_, 
for a number of linear data structures for collections of objects, called _containers_. 
Both algorithms and containers are templates. 
However, the key is that the algorithms are **not** member functions of the containers. 
Instead, the algorithms are written in a generic way so that they can be used by any container 
(and linear collection of elements). 
To do this, the designers of STL identified an abstract concept of _iterators_ 
that can be provided for any kind of linear collection.
Essentially, the collection-specific aspects of container operations 
have been factored out into the iterators' functionality. 


As a consequence, we can implement an operation such as computing the maximum value in a sequence 
**without** knowing the details of how values are stored in that sequence:
```c++
template <typename Iterator>
Iterator max_element(Iterator begin, Iterator end)
{
    // Use only certain Iterator operations 
    // to traverse all elements of the collection 
    // to find the element with the maximum value, 
    // and return its position as Iterator. 
}
```






### 🎯 Chapter 19 Implementing Traits


_Traits_ (or _traits templates_) are C++ programming devices 
that greatly facilitate the management of the sort of extra parameters 
that come up in the design of industrial-strength templates. 
In this chapter, we show a number of situations in which they prove useful 
and demonstrate various techniques that will enable you to write robust and powerful devices of your own.


#### 📌 19.1 An Example: Accumulating a Sequence

##### 19.1.1 Fixed Traits

Assume that the values of the sum we want to compute are stored in an array, 
and we are given a pointer to the first element to be accumulated 
and a pointer one past the last element to be accumulated. 
We wish to write a template that will work for many types. 


Most examples in this section use ordinary pointers for the sake of simplicity.
Clearly, an industrial-strength interface may prefer to use iterator parameters
following the conventions of the C++ standard library. 
We revisit this aspect of our example later.
```c++
template <typename T>
T accum(T const * beg, T const * end)
{
    T total {};  // assume this actually creates a zero value
    
    while (beg != end)
    {
        total += *beg;
        ++beg;
    }
    
    return total;
}
```
We use _value initialization_ (with an empty initializer) here as introduced in Section 5.2. 
It means that the local object total is initialized either by its default constructor 
or by zero (which means `nullptr` for pointers and `false` for Boolean values).
```c++
char name[] = "templates";
std::size_t length = sizeof(name) - 1;

std::cout << accum(name, name + length) / length << '\n';  // -5
```
The problem here is that our template was instantiated for the type `char`, 
which turns out to be _too small_ a range for the accumulation of even relatively small values. 
Clearly, we could resolve this by introducing an additional template parameter `AccT` 
that describes the type used for the variable `total` (and hence the return type). 
However, this would put an extra burden on all users of our template: 
They would have to specify an extra type in every invocation of our template. 
In our example, we may therefore need to write the following:
```c++
std::cout << accum<int>(name, name + length) / length << '\n';  // 108
```
This is not an excessive constraint, but it can be avoided.


The template `AccumulationTraits` is called a _traits template_
because it holds a _trait_ (characteristic) of its parameter type.
(In general, there could be more than one trait and more than one parameter.)
We chose **not** to provide a generic definition of this template
because there **isn't** a great way to select a good accumulation type
when we don't know what the type is. 
```c++
template <typename T>
struct AccumulationTraits;

template <>
struct AccumulationTraits<char>
{
    using AccT = int;
};

template <>
struct AccumulationTraits<short>
{
    using AccT = int;
};

template <>
struct AccumulationTraits<int>
{
    using AccT = long;
};

template <>
struct AccumulationTraits<float>
{
    using AccT = double;
};

template <typename T>
auto accum(T const * beg, T const * end)
{
    // return type is traits of the element type
    using AccT = typename AccumulationTraits<T>::AccT;

    // assume this actually creates a zero value
    AccT total {}; 
    
    while (beg != end)
    {
        total += *beg;
        ++beg;
    }
    
    return total;
}

std::cout << accum(name, name + length) / length << '\n';  // 108
```

##### 19.1.2 Value Traits

So far, we have seen that traits represent additional type information related to a given “main” type. 
In this section, we show that this extra information need **not** be limited to types. 
Constants and other classes of values can be associated with a type as well.
```c++
template <typename T>
struct AccumulationTraits;

template <>
struct AccumulationTraits<char>
{
    using AccT = int;
    static constexpr AccT zero = 0;
};

template <>
struct AccumulationTraits<short>
{
    using AccT = int;
    static constexpr AccT zero = 0;
};

template <>
struct AccumulationTraits<int>
{
    using AccT = long;
    static constexpr AccT zero = 0;
};

template <>
struct AccumulationTraits<float> 
{
    using Acct = double;
    static constexpr double zero = 0.0;
};

template <typename T>
auto accum(T const * beg, T const * end)
{
    // return type is traits of the element type
    using AccT = typename AccumulationTraits<T>::AccT;

    // init total by trait value
    AccT total = AccumulationTraits<T>::zero; 
    
    while (beg != end)
    {
        total += *beg;
        ++beg;
    }
    
    return total;
}
```
However, **neither** `const` **nor** `constexpr` permit non-literal types to be initialized in-class. 
For example, a user-defined arbitrary-precision `BigInt` type might **not** be a literal type, 
because typically it has to allocate components on the heap, 
which usually precludes it from being a literal type, 
or just because the required constructor is **not** `constexpr`. 
The following specialization is then an error:
```c++
class BigInt
{
    BigInt(long long);
};

template <>
struct AccumulationTraits<BigInt>
{
    using AccT = BigInt;

    // ERROR: not a literal type
    static constexpr BigInt zero {0};
};
```
The straightforward alternative is **not** to define the value trait in its class:
```c++
template <>
struct AccumulationTraits<BigInt>
{
    using AccT = BigInt;
    static const BigInt zero;
};

const BigInt AccumulationTraits<BigInt>::zero {0};
```
Although this works, it has the disadvantage of being more verbose (code must be added in two places), 
and it is potentially less efficient because compilers are typically unaware of definitions in other files. 


In C++17, this can be addressed using _inline variables_: 
```c++
template <>
struct AccumulationTraits<BigInt> 
{
    using AccT = BigInt;
    inline static BigInt const zero {0};  // OK since C++17
};
```
An alternative that works prior to C++17 is to use _inline member functions_ 
for value traits that **won't** always yield integral values. 
Again, such a function can be declared `constexpr` if it returns a literal type.
Most modern C++ compilers can “see through” calls of simple inline functions.
Additionally, the use of `constexpr` makes it possible to use the value traits 
in contexts where the expression must be a constant (e.g., in a template argument). 
```c++
template <typename T>
struct AccumulationTraits;

template <>
struct AccumulationTraits<char>
{
    using AccT = int;

    static constexpr AccT zero()
    {
        return 0;
    }
};

template <>
struct AccumulationTraits<short>
{
    using AccT = int;

    static constexpr AccT zero()
    {
        return 0;
    }
};

template <>
struct AccumulationTraits<int>
{
    using AccT = long;

    static constexpr AccT zero()
    {
        return 0;
    }
};

template <>
struct AccumulationTraits<unsigned int>
{
    using AccT = unsigned long;

    static constexpr AccT zero()
    {
        return 0;
    }
};

template <>
struct AccumulationTraits<float>
{
    using AccT = double;

    static constexpr AccT zero()
    {
        return 0;
    }
};

template <>
struct AccumulationTraits<BigInt>
{
    using AccT = BigInt;

    static BigInt zero()
    {
        return 0;
    }
};
```
Clearly, traits can be more than just extra types. 
In our example, they can be a mechanism to provide all the necessary information 
that `accum` needs about the element type for which it is called. 
This is the key to the notion of traits: 
Traits provide an avenue to _configure_ concrete elements (mostly types) for generic computations.

##### 19.1.3 Parameterized Traits

The use of traits in `accum` in the previous sections is called _fixed_, 
because once the decoupled trait is defined, it can **not** be replaced in the algorithm. 
There may be cases when such overriding is desirable. 
For example, we may happen to know that 
a set of `float` values can safely be summed into a variable of the same type, 
and doing so may buy us some efficiency. 


We can address this problem by adding a template parameter `AT` for the trait itself
having a default value determined by our traits template:
```c++
template <typename T, typename AT = AccumulationTraits<T>>
auto accum(T const * beg, T const * end)
{
    typename AT::AccT total = AT::zero();
    
    while (beg != end)
    {
        total += *beg;
        ++beg;
    }
    
    return total;
}
```
In this way, many users can omit the extra template argument, 
but those with more exceptional needs can specify an alternative to the preset accumulation type. 
Presumably, most users of this template would never have to provide the second template argument explicitly 
because it can be configured to an appropriate default for every type deduced for the first argument. 


#### 📌 19.2 Traits versus Policies and Policy Classes


So far we have equated _accumulation with summation_. 
However, we can imagine other kinds of accumulations. 
For example, we could multiply the sequence of given values. 
Or, if the values were strings, we could concatenate them. 
Even finding the maximum value in a sequence could be formulated as an accumulation problem. 
In all these alternatives, the only `accum` operation that needs to change is `total += *beg`. 
This operation can be called a _policy_ of our accumulation process.
```c++
class SumPolicy
{
public:
    template <typename T1, typename T2>
    static void accumulate(T1 & total, T2 const & value)
    {
        total += value;
    }
};

class MultPolicy
{
public:
    template <typename T1, typename T2>
    static void accumulate(T1 & total, T2 const & value)
    {
        total *= value;
    }
};

template <typename T,
          typename Policy = SumPolicy,
          typename Traits = AccumulationTraits<T>>
auto accum(T const * beg, T const * end)
{
    using AccT = typename Traits::AccT;
    AccT total = Traits::zero();
    
    while (beg != end)
    {
        Policy::accumulate(total, *beg);
        ++beg;
    }
    
    return total;
}

void foo()
{
    int num[] {1, 2, 3, 4, 5};
    std::cout << accum<int, MultPolicy>(num, num + 5) << '\n';  // 0
}
```
In this version of `accum`, `SumPolicy` is a _policy class_ 
that implements one or more policies for an algorithm through an agreed-upon interface.
We could generalize this to a policy parameter, 
which could be a class (as discussed) or a pointer to a function. 


However, the output of this program **isn't** what we would like (outputs `0`). 
The problem here is caused by our choice of initial value: 
Although `0` works well for summation, it does **not** work for multiplication 
(a zero initial value forces a zero result for accumulated multiplications). 
This illustrates that different traits and policies may interact, 
underscoring the importance of careful template design. 


In this case, we may recognize that the initialization of an accumulation loop is a part of the accumulation policy. 
This policy may or may not make use of the trait `zero`. 
Other alternatives are not to be forgotten: 
**Not** everything must be solved with traits and policies. 
For example, the `std::accumulate` function of the C++ standard library 
takes the initial value and the accumulation policy as the third and fourth (function call) argument. 

##### 19.2.1 Traits and Policies: What's the Difference?

We therefore use the following definitions:
- **Traits** represent natural additional properties of a template parameter; 
- **Policies** represent configurable behavior for generic functions and types (often with some commonly used defaults). 


For traits, we make the following observations:
- Traits can be useful as _fixed traits_ (i.e., **without** being passed through template parameters).
- Traits parameters usually have very natural default values (which are rarely overridden, or simply can **not** be overridden).
- Traits parameters tend to depend tightly on one or more main parameters. 
- Traits mostly combine types and constants rather than member functions. 
- Traits tend to be collected in _traits templates_.


For policy classes, we make the following observations:
- Policy classes **don't** contribute much if they aren't passed as template parameters. 
- Policy parameters need **not** have default values and are often specified explicitly 
  (although many generic components are configured with commonly used default policies).
- Policy parameters are mostly orthogonal to other parameters of a template.
- Policy classes mostly combine member functions.
- Policies can be collected in plain classes or in class templates. 

##### 19.2.2 Member Templates versus Template Template Parameters

To implement an accumulation policy, 
we chose to express `SumPolicy` and `MultPolicy` as ordinary classes with a member template. 
An alternative consists of designing the policy class interface using class templates, 
which are then used as template template arguments.
```c++
template <typename T1, typename T2>
class SumPolicy
{
public:
    static void accumulate(T1 & total, T2 const & value)
    {
        total += value;
    }
};

template <typename T,
          template <typename, typename> class Policy = SumPolicy,
          typename Traits = AccumulationTraits<T>>
auto accum(T const * beg, T const * end)
{
    using AccT = typename Traits::AccT;
    AccT total = Traits::zero();
    
    while (beg != end)
    {
        Policy<AccT, T>::accumulate(total, *beg);
        ++beg;
    }
    
    return total;
}
```
The same transformation can be applied to the traits parameter. 
(Other variations on this theme are possible: 
For example, instead of explicitly passing the `AccT` type to the policy type,
it may be advantageous to pass the accumulation trait 
and have the policy determine the type of its result from a traits parameter.)


The major advantage of accessing policy classes through template template parameters is that 
it makes it easier to have a policy class carry with it some state information (i.e., static data members) 
with a type that depends on the template parameters. 
(In our first approach, the static data members would have to be embedded in a member class template.)


However, a downside of the template template parameter approach is that 
policy classes must now be written as templates, 
with the exact set of template parameters defined by our interface. 
This can make the expression of the traits themselves more verbose and less natural than a simple non-template class. 

##### 19.2.3 Combining Multiple Policies and/or Traits

N/A

##### 19.2.4 Accumulation with General Iterators

```c++
template <typename Iter>
auto accum(Iter start, Iter end)
{
    using VT = typename std::iterator_traits<Iter>::value_type;
    VT total {};  // assume this actually creates a zero value
    
    while (start != end)
    {
        total += *start;
        ++start;
    }
    
    return total;
}
```

##### 📌 19.3 Type Functions

The initial traits example demonstrates that we can define behavior that depends on types. 
Traditionally, in C and C++, we define functions that could more specifically be called _value functions_: 
They take some values as arguments and return another value as a result. 
With templates, we can additionally define _type functions_: 
functions that takes some type as arguments and produce a type or a constant as a result.


A very useful built-in type function is `sizeof`, 
which returns a constant describing the size (in bytes) of the given type argument. 
Class templates can also serve as type functions. 
The parameters of the type function are the template parameters, 
and the result is extracted as a member type or member constant. 
For example, the `sizeof` operator could be given the following interface:
```c++
template <typename T>
struct TypeSize 
{
    static constexpr std::size_t value = sizeof(T);
};

void foo()
{
    std::cout << TypeSize<int>::value << '\n';  // 4
}
```
This may not seem very useful, since we have the built-in sizeof operator available,
but note that `TypeSize<T>` is a type, and it can therefore be passed as a class template argument itself. 
Alternatively, `TypeSize` is a template and can be passed as a template template argument. 


In what follows, we develop a few more general-purpose type functions that can be used as traits classes in this way.

##### 19.3.1 Element Types

```c++
template <typename Container>
struct ElementT
{
    using Type = typename Container::value_type;
};

template <typename T, std::size_t N>
struct ElementT<T [N]> 
{
    using Type = T;
};

template <typename T>
struct ElementT<T []> 
{
    using Type = T;
};

std::vector<bool> s;  // typename ElementT<decltype(s)>::Type is bool
int arr[42];          // typename ElementT<decltype(arr)>::Type is int
```
Note that we should provide partial specializations for all possible array types (see Section 5.4 for details). 


It is usually advisable to provide member type definitions for class template type parameters 
so that they can be accessed more easily in generic code (like the standard container templates do):
```c++
template <typename T1, typename T2, ...>
class X
{
public:
    using ... = T1;
    using ... = T2;
    ...
};
```
In this case, the type `ElementT` is called a _traits class_ 
because it is used to access a trait of the given container type `Container` 
(in general, more than one trait can be collected in such a class). 


As a convenience, we can create an alias template for type functions to bypass verbose `typename`s:
```c++
template <typename T>
using ElementType = typename ElementT<T>::type;
```

##### 19.3.2 Transformation Traits

In addition to providing access to particular aspects of a main parameter type, 
traits can also perform transformations on types, 
such as adding or removing references or `const` and `volatile` qualifiers.

```c++
/// <type_traits>

// Const-volatile modifications.

/// remove_const
template <typename _Tp>
struct remove_const
{ typedef _Tp     type; };

template <typename _Tp>
struct remove_const<_Tp const>
{ typedef _Tp     type; };

/// remove_volatile
template <typename _Tp>
struct remove_volatile
{ typedef _Tp     type; };

template <typename _Tp>
struct remove_volatile<_Tp volatile>
{ typedef _Tp     type; };

/// remove_cv
template <typename _Tp>
struct remove_cv
{
    typedef typename remove_const<typename remove_volatile<_Tp>::type>::type     type;
};

/// add_const
template <typename _Tp>
struct add_const
{ typedef _Tp const     type; };

/// add_volatile
template <typename _Tp>
struct add_volatile
{ typedef _Tp volatile     type; };

/// add_cv
template <typename _Tp>
struct add_cv
{
    typedef typename add_const<typename add_volatile<_Tp>::type>::type     type;
};

/// Alias template for remove_const
template <typename _Tp>
using remove_const_t = typename remove_const<_Tp>::type;

/// Alias template for remove_volatile
template <typename _Tp>
using remove_volatile_t = typename remove_volatile<_Tp>::type;

/// Alias template for remove_cv
template <typename _Tp>
using remove_cv_t = typename remove_cv<_Tp>::type;

/// Alias template for add_const
template <typename _Tp>
using add_const_t = typename add_const<_Tp>::type;

/// Alias template for add_volatile
template <typename _Tp>
using add_volatile_t = typename add_volatile<_Tp>::type;

/// Alias template for add_cv
template <typename _Tp>
using add_cv_t = typename add_cv<_Tp>::type;

// Reference transformations.

/// remove_reference
template<typename _Tp>
struct remove_reference
{ typedef _Tp   type; };

template <typename _Tp>
struct remove_reference<_Tp&>
{ typedef _Tp   type; };

template <typename _Tp>
struct remove_reference<_Tp&&>
{ typedef _Tp   type; };

template <typename _Tp, bool = __is_referenceable<_Tp>::value>
struct __add_lvalue_reference_helper
{ typedef _Tp   type; };

template <typename _Tp>
struct __add_lvalue_reference_helper<_Tp, true>
{ typedef _Tp&   type; };

/// add_lvalue_reference
template <typename _Tp>
struct add_lvalue_reference
: public __add_lvalue_reference_helper<_Tp>
{ };

template <typename _Tp, bool = __is_referenceable<_Tp>::value>
struct __add_rvalue_reference_helper
{ typedef _Tp   type; };

template <typename _Tp>
struct __add_rvalue_reference_helper<_Tp, true>
{ typedef _Tp&&   type; };

/// add_rvalue_reference
template <typename _Tp>
struct add_rvalue_reference
: public __add_rvalue_reference_helper<_Tp>
{ };

/// Alias template for remove_reference
template <typename _Tp>
using remove_reference_t = typename remove_reference<_Tp>::type;

/// Alias template for add_lvalue_reference
template <typename _Tp>
using add_lvalue_reference_t = typename add_lvalue_reference<_Tp>::type;

/// Alias template for add_rvalue_reference
template <typename _Tp>
using add_rvalue_reference_t = typename add_rvalue_reference<_Tp>::type;
```
There are two things to note with the definition of `RemoveCVT`. 
1. It is making use of both `RemoveConstT` and the related `RemoveVolatileT`, 
   first removing the volatile (if present) and then passing the resulting type to `RemoveConstT`.
2. It is using _metafunction forwarding_ to inherit the `Type` member from `RemoveConstT` 
   rather than declaring its own `Type` member 
   that is identical to the one in the `RemoveConstT` specialization. 

Here, metafunction forwarding is used simply to reduce the amount of typing in the definition of `RemoveCVT`.
However, metafunction forwarding is also useful 
when the metafunction is not defined for all inputs, 
a technique that will be discussed further in Section 19.4. 
```c++
/// <bits/c++config.h>

#define _GLIBCXX_NOEXCEPT_PARM , bool _NE
#define _GLIBCXX_NOEXCEPT_QUAL noexcept (_NE)

/// <type_traits>

// decay

/// is_array
template <typename>
struct is_array
: public false_type { };

template <typename _Tp, std::size_t _Size>
struct is_array<_Tp [_Size]>
: public true_type { };

template <typename _Tp>
struct is_array<_Tp []>
: public true_type { };

/// is_function
template <typename>
struct is_function
: public false_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile & _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile && _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

// Decay trait for arrays and functions, used for perfect forwarding
// in make_pair, make_tuple, etc.
template <typename _Up,
          bool _IsArray = is_array<_Up>::value,
          bool _IsFunction = is_function<_Up>::value>
struct __decay_selector;

// NB: DR 705.
template<typename _Up>
struct __decay_selector<_Up, false, false>
{ typedef typename remove_cv<_Up>::type __type; };

template<typename _Up>
struct __decay_selector<_Up, true, false>
{ typedef typename remove_extent<_Up>::type* __type; };

template<typename _Up>
struct __decay_selector<_Up, false, true>
{ typedef typename add_pointer<_Up>::type __type; };

/// decay
template <typename _Tp>
class decay
{
public:
    typedef typename __decay_selector<__remove_type>::__type type;
    
private:
    typedef typename remove_reference<_Tp>::type __remove_type;
};
```
Note that the second partial specialization matches any function type that uses C-style varargs. 
Strictly speaking, 
```c++
template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

// Comma prior to the second ellipsis is optional
template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes..., ...) _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };

// Same as the previous one, re-definition of the same template
template <typename _Res, typename ... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) _GLIBCXX_NOEXCEPT_QUAL>
: public true_type { };
```
the comma prior to the second ellipsis `...` is optional but is provided here for clarity. 
Due to the ellipsis being optional, 
the function type in the first partial specialization is actually syntactically ambiguous: 
It can be parsed as either `_Res(_ArgTypes...)` (a C-style varargs parameter) 
or `_Res(_ArgTypes... name)` (a parameter pack). 
The second interpretation is picked because `Args` is an unexpanded parameter pack. 
We can explicitly add the comma in the (rare) cases where the other interpretation is desired.

##### 19.3.3 Predicate Traits

```c++
/// <type_traits>

/// integral_constant
template <typename _Tp, _Tp __v>
  struct integral_constant
  {
    static constexpr _Tp                  value = __v;
    typedef _Tp                           value_type;
    typedef integral_constant<_Tp, __v>   type;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
  };

template <typename _Tp, _Tp __v>
  constexpr _Tp integral_constant<_Tp, __v>::value;

/// The type used as a compile-time boolean with true value.
typedef integral_constant<bool, true>     true_type;

/// The type used as a compile-time boolean with false value.
typedef integral_constant<bool, false>    false_type;

/// is_same
template <typename, typename>
  struct is_same
  : public false_type { };

template <typename _Tp>
  struct is_same<_Tp, _Tp>
  : public true_type { };

template <typename _Tp, typename _Up>
  inline constexpr bool is_same_v = is_same<_Tp, _Up>::value;
```
_Tag dispatch_ technique as introduced in Section 20.2:
```c++
template <typename T>
void fooImpl(std::true_type, T) {}

template <typename T>
void fooImpl(std::false_type, T) {}

template <typename T>
void foo(T t) { fooImpl(std::is_same_v<T, int>, t); }
```

##### 19.3.4 Result Type Traits

Another example of type functions that deal with multiple types are _result type traits_.
They are very useful when writing operator templates. 
To motivate the idea, let's write a function template that allows us to add two `Array` containers:
```c++
template <typename T>
Array<T> operator+(Array<T> const &, Array<T> const &);
```

This would be nice, but because the language allows us to add a `char` value to an `int` value, 
we really would prefer to allow such mixed-type operations with arrays too. 
We are then faced with determining what the return type of the resulting template should be
```c++
template <typename T1, typename T2>
struct PlusResultT
{
    using Type = decltype(declval<T1>() + declval<T2>());
};

template <typename T1, typename T2>
using PlusResult = typename PlusResultT<T1, T2>::Type;

template <typename T1, typename T2>
Array<PlusResult<T1, T2>> operator+(Array<T1> const &, Array<T2> const &);
```
However, for the purpose of our motivating example, `decltype` actually preserves _too much_ information. 
For example, our formulation of `PlusResultT` may produce a reference type, b
ut most likely our `Array` class template is **not** designed to handle reference types. 
More realistically, an overloaded `operator+` might return a value of `const` class type:
```c++
class Integer {};
Integer const operator+(Integer const &, Integer const &);
```
Adding two `Array<Integer>` values will result in an array of `Integer const`, 
which is most likely **not** what we intended. 
In fact, what we want is to transform the result type by removing references and qualifiers, 
as discussed in the previous section: 
```c++
template <typename T1, typename T2>
Array<std::remove_cv<std::remove_reference<PlusResult<T1, T2>>>>
operator+(Array<T1> const &, Array<T2> const &);
```
At this point, the array addition operator properly computes the result type 
when adding two arrays of (possibly different) element types. 
However, our formulation of `PlusResultT` places an undesirable restriction on the element types `T1` and `T2`:
Because the expression `T1() + T2()` attempts to value-initialize values of types `T1` and `T2`, 
both of these types must have an accessible, non-deleted, default constructor (or be non-class types). 
The `Array` class itself might **not** otherwise require value-initialization of its element type,
so this is an additional, unnecessary restriction. 

##### `declval`

```c++
/// <type_traits> (#include-d by <utility>)
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

template <typename _Tp>
struct __declval_protector
{
    static const bool __stop = false;
};

// declval should appear only in unevaluated contexts, 
// where its body will never be executed. 
template <typename _Tp>
auto declval() noexcept -> decltype(__declval<_Tp>(0))
{
    static_assert(__declval_protector<_Tp>::__stop, "declval() must not be used!");
    return __declval<_Tp>(0);
}
```
```c++
/// Standard specification
/// <utility>

template <class T>
typename std::add_rvalue_reference<T>::type declval() noexcept;
```
- Converts any type `T` to a reference type, 
  making it possible to use member functions in `decltype`, `sizeof`, ..., expressions 
  **without** the need to go through constructors.
- `declval` is commonly used in templates 
  where acceptable template parameters may have **no** constructor in common, 
  but have the same member function whose return type is needed.
- Note that `declval` can only be used in _unevaluated contexts_ and is **not** required to be defined. 
  It is an error to evaluate an expression that contains this function. 
  Formally, the program is ill-formed if this function is odr-used.
- Can **not** be called and thus never returns a value.
  The return type is `T &&` unless `T` is (possibly `cv`-qualified) `void`, in which case the return type is `T`. 


- For referenceable types, the return type is always an rvalue reference to the type,
  which allows `declval` to work even with types that could **not** normally be returned from a function, 
  such as abstract class types (classes with pure virtual functions) or array types. 
  The transformation from `T` to `T &&` otherwise has **no** practical effect 
  on the behavior of `declval<T>()` when used as an expression:
  Both are rvalues (if `T` is an object type), while lvalue reference types are unchanged due to reference collapsing. 
  - Still, the difference between the return types `T` and `T &&` is discoverable 
    by direct use of `decltype`. 
    However, given `declval`'s limited use, this is not of practical interest.
- The `noexcept` exception specification documents that `declval` itself 
  does **not** cause an expression to be considered to throw exceptions. 
  It becomes useful when `declval` is used in the context of the `noexcept` operator (Section 19.7.2).


#### 📌 19.4 SFINAE-Based Traits


The SFINAE principle turns potential errors 
during the formation of invalid types and expressions 
during template argument deduction (which would cause the program to be ill-formed) 
into simple deduction failures, 
allowing overload resolution to select a different candidate. 
While originally intended to avoid spurious failures with function template overloading, 
SFINAE also enables remarkable compile-time techniques 
that can determine if a particular type or expression is valid.
This allows us to write traits that determine, 
for example, whether a type has a specific member, supports a specific operation, or is a class.
The two main approaches for SFINAE-based traits are to 
SFINAE out functions overloads 
and to SFINAE out partial specializations.

##### 19.4.1 SFINAE Out Function Overloads

The usual approach to implement a SFINAE-base trait with function overloading
is to declare two overloaded function templates named `test` with different return types:
```c++
template <typename T>
struct IsDefaultConstructibleT
{
private:
    // test() trying substitute call of a default constructor for T passed as U :
    template <typename U, typename = decltype(U())>
    static char test(void *);

    // test() fallback:
    template <typename>
    static long test(...);

public:
    static constexpr bool value = std::is_same_v<decltype(test<T>(nullptr)), char>;
};
```
Our “return value” value depends on which overloaded test member is selected.
- The first overload is designed to match only if the requested check succeeds.
- The second overload is the fallback. 
  It always matches the call, but because it matches "with ellipsis" (i.e., a vararg parameter), 
  any other match would be preferred (see Section C.2).
  The fallback declaration can sometimes be a plain member function declaration 
  instead of a member function template. 
```c++
struct S 
{
    S() = delete;
};

std::cout << IsDefaultConstructibleT<int>::value << '\n';  // yields true
std::cout << IsDefaultConstructibleT<S>::value << '\n';    // yields false
```

##### Alternative Implementation Strategies for SFINAE-based Traits

The key to the approach always consisted in 
declaring two overloaded function templates returning different return types:
```c++
template <...> static char test(void *);
template <...> static long test(...);
```
On old platforms before C++11 (no `nullptr` and no `constexpr`):
```c++
enum { value = sizeof(test<...>(0)) == 1 };
```
On some platforms, it might happen that `sizeof(char) == sizeof(long)`.
```c++
// either
using Size1T = char;
using Size2T = struct { char a[2]; };

// or
using Size1T = char (&)[1];
using Size2T = char (&)[2];
```
```c++
template <...> static Size1T test(void *);
template <...> static Size2T test(...);
```
Note also that the type of the call argument passed to `func`() **doesn't** matter. 
All that matters is that the passed argument matches the expected type. 
```c++
template <...> static Size1T test(int);
template <...> static Size2T test(...);

enum { value = sizeof(test<...>(42)) == 1 };
```

##### Making SFINAE-based Traits Predicate Traits

A predicate trait, which returns a Boolean value, 
should return a value derived from `std::true_type` or `std::false_type`. 
This way, we can also solve the problem that on some platforms `sizeof(char) == sizeof(long)`. 
```c++
template <typename T>
struct IsDefaultConstructibleHelper
{
public:
    using Type = decltype(test<T>(nullptr));

private:
    // test() trying substitute call of a default constructor for T passed as U:
    template <typename U, typename = decltype(U())>
    static std::true_type test(void *);

    // test() fallback:
    template <typename>
    static std::false_type test(...);
};

template <typename T>
struct IsDefaultConstructibleT : IsDefaultConstructibleHelper<T>::Type {};
```


#### 📌 19.4.2 SFINAE Out Partial Specializations


```c++
// helper to ignore any number of template parameters:
template <typename...> using VoidT = void;

// primary template:
template <typename, typename = VoidT<>>
struct IsDefaultConstructibleT : std::false_type {};

// partial specialization (may be SFINAE'd away):
template <typename T>
struct IsDefaultConstructibleT<T, VoidT<decltype(T())>> : std::true_type {};
```


#### 📌 19.4.3 Using Generic Lambdas for SFINAE


Whichever technique we use, some boilerplate code is always needed to define traits: 
- Overloading and calling two `test` member functions; 
- Implementing multiple partial specializations. 


Next, we will show how in C++17, we can minimize this boilerplate
by specifying the condition to be checked in a generic lambda. 
```c++
// helper: checking validity of f(args...) for F f and Args ... args:
template <typename F, 
          typename ... Args,
          typename = decltype(std::declval<F>()(std::declval<Args &&>()...))>
std::true_type isValidImpl(void *);

// fallback if helper SFINAE'd out:
template <typename F, typename ... Args>
std::false_type isValidImpl(...);

// define a lambda that takes a lambda f and returns whether calling f with args is valid
inline constexpr auto isValid = [](auto f)
{
    return [](auto && ... args)
    {
        return decltype(isValidImpl<decltype(f), decltype(args) && ...>(nullptr)) {};
    };
};

// helper template to represent a type as a value
template <typename T>
struct TypeT
{
    using Type = T;
};

// helper to wrap a type as a value
template <typename T>
constexpr auto type = TypeT<T> {};

// Helper to unwrap a wrapped type in unevaluated contexts. 
// No definition needed. 
template <typename T>
T valueT(TypeT<T>);

inline constexpr auto isDefaultConstructible = isValid([](auto x) -> decltype((void) decltype(valueT(x))()) {});

void foo()
{
    std::cout << isDefaultConstructible(type<int>) << '\n';    // true (int is default-constructible)
    std::cout << isDefaultConstructible(type<int &>) << '\n';  // false (references are not default-constructible)
}
```
Let’s start with the definition of `isValid`: 
It is a `constexpr` variable whose type is a lambda's closure type. 
The declaration must necessarily use a placeholder type (`auto` in our code) 
because C++ has no way to express closure types directly. 
Prior to C++17, lambda expressions could not appear in constant-expressions, 
which is why this code is only valid in C++17. 
Since `isValid` has a closure type, it can be _invoked_ (i.e., called), 
but the item that it returns is itself an object of a lambda closure type, 
produced by the inner lambda expression. 


We already know that `isDefaultConstructible` has a lambda closure type
and it is a function object that checks the trait of a type being default-constructible. 
In other words, `isValid` is a _traits factory_: 
A component that generates traits checking objects from its argument. 


The type helper variable template allows us to represent a type as a value. 
A value `x` obtained that way can be turned back into the original type with `decltype(valueT(x))`, 
and that’s exactly what is done in the lambda passed to `isValid` above. 
If that extracted type cannot be default-constructed, `decltype(valueT(x))()` is invalid, 
and we will either get a compiler error, 
or an associated declaration will be "SFINAE'd out"  
(and the latter effect is what we’ll achieve thanks to the details of the definition of `isValid`).


To see how all the pieces work together,
consider what the inner lambda expression in `isValid` becomes 
with `isValid`'s parameter `f` bound to 
the generic lambda argument specified in the definition of `isDefaultConstructible`. 
By performing the substitution in `isValid`'s definition, we get the equivalent of:
(P.S. Just for understanding, this code snippet is invalid in terms of C++17 syntax.)
```c++
constexpr auto isDefaultConstructible = [](auto && ... args) 
{
    // default template argument (3rd template argument) of isValidImpl omitted; see later
    return decltype(isValidImpl<decltype([](auto x) -> decltype((void) decltype(valueT(x))())), 
                                decltype(args) && ...>
                               (nullptr)
                   ) {};
};
```
If we look back at the first declaration of `isValidImpl()` above, 
we note that it includes a default template argument of the form
```c++
decltype(std::declval<F>()(std::declval<Args &&>()...))>
```
which attempts to invoke a value of the type of its first template argument, 
which is the closure type of the lambda in the definition of `isDefaultConstructible`, 
with values of the types of the arguments (`decltype(args) && ...`) passed to `isDefaultConstructible`. 
As there is only one parameter `x` in the lambda,
`args` must expand to only one argument; 
in our `static_assert` examples above, 
that argument has type `TypeT<int>` or `TypeT<int &>`. 
In the `TypeT<int &>` case, `decltype(valueT(x)) `is `int &` which makes `decltype(valueT(x))()` invalid, 
and thus the substitution of the default template argument in the first declaration of `isValidImpl` 
fails and is SFINAE’d out. 
That leaves just the second declaration (which would otherwise be a lesser match),
which produces a `std::false_type` value.
Overall, `isDefaultConstructible` produces `std::false_type` when `type<int &>` is passed. 
If instead `type<int>` is passed, the substitution does not fail, 
and the first declaration of `isValidImpl` is selected, producing a `std::rue_type` value.


Recall that for SFINAE to work, substitutions have to happen _in the immediate context_ of the substituted templates. 
In this case, the substituted templates are the first declaration of `isValidImpl` 
and the call operator of the generic lambda passed to `isValid`. 
Therefore, the construct to be tested has to appear in the return type of that lambda, not in its body!


Our `isDefaultConstructible` trait is a little different from previous traits implementations 
in that it requires function-style invocation instead of specifying template arguments. 
That is arguably a more readable notation, but the prior style can be obtained also:
```c++
template <typename T>
using IsDefaultConstructibleT = decltype(isDefaultConstructible(std::declval<T>()));
```
Since this is a traditional template declaration, however, it can only appear in namespace scope, 
whereas the definition of `isDefaultConstructible` could conceivably have been introduced in block scope. 


So far, this technique might not seem compelling 
because both the expressions involved in the implementation and the style of use 
are more complex than the previous techniques. 
However, once `isValid` is in place and understood, many traits can be implement with just one declaration. 
For example, testing for access to a member named first, 
is rather clean (see Section 19.6.4 for the complete example):
```c++
constexpr auto hasFirst = isValid([](auto x) -> decltype((void) valueT(x).first) {});
```

#### 📌 19.4.4 SFINAE-Friendly Traits


A type trait should be able to answer a particular query 
**without** causing the program to become ill-formed. 
SFINAE-based traits address this problem by carefully trapping potential problems within a SFINAE context, 
turning those would-be errors into negative results. 


However, some traits presented thus far do not behave well in the presence of errors.
```c++
template <typename T1, typename T2>
struct PlusResultT
{
    using Type = decltype(std::declval<T1>() + std::declval<T2>());
};

// For simplicity, the return value just uses `PlusResultT<T1, T2>::Type`. 
// In practice, the return type should also be computed using `RemoveReferenceT` and `RemoveCVT` 
// to avoid that references are returned.
template <typename T1, typename T2>
using PlusResult = typename PlusResultT<T1, T2>::Type;
```
In this definition, the `+` is used in a context that is **not** protected by SFINAE. 
Therefore, if a program attempts to evaluate `PlusResultT` for types that do **not** have a suitable `operator+`, 
the evaluation of `PlusResultT` itself will cause the program to become ill-formed. 
```c++
class A {};

class B {};

void addAB(Array<A> arrayA, Array<B> arrayB)
{
    // ERROR: fails in instantiation of PlusResultT<A, B>
    auto sum = arrayA + arrayB;
}
```
The practical problem is not that this failure occurs with code that is clearly ill-formed like this 
(there is no way to add an array of `A` to an array of `B`) 
but that it occurs during template argument deduction for `operator+`, 
deep in the instantiation of `PlusResultT<A, B>`.


This has a remarkable consequence: 
It means that the program may **fail** to compile
even if we add a specific overload to adding `A` and `B` arrays, 
because C++ does **not** specify whether the types in a function template are actually instantiated 
if another overload would be better:
```c++
// declare generic + for arrays of different element types:
template <typename T1, typename T2>
Array<typename PlusResultT<T1, T2>::Type>
operator+(Array<T1> const &, Array<T2> const &);

// overload + for concrete types:
Array<A> operator+(Array<A> const & arrayA, Array<B> const & arrayB);

void addAB(Array<A> const & arrayA, Array<B> const & arrayB) 
{
    // ERROR?: depends on whether the compiler instantiates PlusResultT<A, B>
    auto sum = arrayA + arrayB; 
}
```
If the compiler can determine that the second declaration of `operator+` is a better match 
without performing deduction and substitution on the first (template) declaration of `operator+`, 
it will accept this code.


However, while deducing and substituting a function template candidate, 
anything that happens during the instantiation of the definition of a class template 
is **not** part of the _immediate context_ of that function template substitution, 
and SFINAE does **not** protect us from attempts to form invalid types or expressions there. 
Instead of just discarding the function template candidate, 
an error is issued right away. 


To solve this problem, we have to make the `PlusResultT` _SFINAE-friendly_,
which means to make it more resilient by giving it a suitable definition 
even when its `decltype` expression is ill-formed.
```c++
// primary template:
template <typename, typename, typename = std::void_t<>>
struct HasPlusT : std::false_type {};

// partial specialization (may be SFINAE’d away):
template <typename T1, typename T2>
struct HasPlusT<T1, T2, std::void_t<decltype(std::declval<T1>() + std::declval<T2>())>>
        : std::true_type {};

// primary template, used when HasPlusT yields true
template <typename T1, typename T2, bool = HasPlusT<T1, T2>::value>
struct PlusResultT
{ 
    using Type = decltype(std::declval<T1>() + std::declval<T2>());
};

// partial specialization, used otherwise
template <typename T1, typename T2>
struct PlusResultT<T1, T2, false> {};
```


#### 📌 19.5 `IsConvertibleT`


```c++
template <typename FROM, 
          typename TO, 
          bool = std::is_void_v<TO> || std::is_array_v<TO> || std::is_function_v<TO>>
struct IsConvertibleHelper
{
public:
    using Type = std::bool_constant<std::is_void_v<TO> && std::is_void_v<FROM>>;
};

template <typename FROM, typename TO>
struct IsConvertibleHelper<FROM, TO, false>
{
private:
    // test() trying to call the helper aux(TO) for a FROM passed as F :
    static void aux(TO);

    template <typename F,
              typename T,
              typename = decltype(aux(std::declval<F>()))>
    static std::true_type test(void *);

    // test() fallback:
    template <typename, typename>
    static std::false_type test(...);

public:
    using Type = decltype(test<FROM, TO>(nullptr));
};

template <typename FROM, typename TO>
struct IsConvertibleT : IsConvertibleHelper<FROM, TO>::Type {};

template <typename FROM, typename TO>
using IsConvertible = typename IsConvertibleT<FROM, TO>::Type;

template <typename FROM, typename TO>
constexpr bool isConvertible = IsConvertibleT<FROM, TO>::value;
```


#### 📌 19.6 Detecting Members


Another foray into SFINAE-based traits involves creating a trait (or, rather, a set of traits) 
that can determine whether a given type `T` has a member of a given name `X` (a type or a non-type member).

##### 19.6.1 Detecting Member Types

Determine whether a given type `T` has a member type `size_type`:
```c++
// primary template:
template <typename, typename = std::void_t<>>
struct HasSizeTypeT : std::false_type {};

// partial specialization (may be SFINAE’d away):
template <typename T>
struct HasSizeTypeT<T, std::void_t<typename std::remove_reference_t<T>::size_type>> : std::true_type {};


std::cout << HasSizeTypeT<int>::value << '\n';  // false

struct CX 
{
    using size_type = std::size_t;
};

std::cout << HasSizeType<CX>::value << '\n';    // true
```
As usual for predicate traits, 
we define the general case to be derived from `std::false_type`, 
because by default a type doesn’t have the member `size_type`.
In this case, we only need one construct , `typename T::size_type`,
which is valid if and only if type `T` has a member type `size_type`, 
If, for a specific `T`, the construct is invalid (i.e., type `T has` no member type `size_type`), 
SFINAE causes the partial specialization to be discarded, and we fall back to the primary template. 
Otherwise, the partial specialization is valid and preferred.

##### Dealing with Reference Types

Without the `remove_reference` trait, this member-type-detection trait may fail on references: 
```c++
struct CX 
{
    using size_type = std::size_t;
};

struct CXR 
{
    using size_type = char &;
};

std::cout << HasSizeType<CX>::value << '\n';      // OK: prints true
std::cout << HasSizeTypeT<CXR>::value << '\n';    // OK: prints true

std::cout << HasSizeTypeT<CX &>::value << '\n';   // OOPS: prints false
std::cout << HasSizeTypeT<CXR &>::value << '\n';  // OOPS: prints false
```
It is true that a reference type has **not** members per se, 
but whenever we use references, 
the resulting expressions have the underlying type, 
and so perhaps it would be preferable to consider the underlying type in that case. 
Here, that could be achieved by using the `std::remove_reference` trait. 

##### Injected Class Names

It’s also worth noting that our traits technique to detect member types
will also produce a `true` value for injected class names: 
```c++
struct size_type {};

struct Sizeable : size_type {};

static_assert(HasSizeTypeT<Sizeable>::value, "Compiler bug: Injected class name missing");
```
The latter static assertion succeeds 
because `size_type` introduces its own name as a member type, 
and that name is inherited. 
If it didn’t succeed, we would have found a defect in the compiler.

##### 19.6.2 Detecting Arbitrary Member Types

Defining a trait such as `HasSizeTypeT` raises the question of 
how to parameterize the trait to be able to check for any member type name.

Unfortunately, this can currently be achieved only via macros, 
because there is no language mechanism to describe a "potential" name. 
The closest we can get for the moment without using macros is to use generic lambdas, 
as illustrated in Section 19.6.4. 

The following macro would work.
Each use of `DEFINE_HAS_TYPE(MemberType)` defines a new `HasType_MemberType` trait: 
```c++
#ifndef DEFINE_HAS_TYPE
#define DEFINE_HAS_TYPE(MemType) \
template <typename, typename = std::void_t<>> \
struct HasTypeT_##MemType : std::false_type {}; \
template <typename T> \
struct HasTypeT_##MemType<T, std::void_t<typename T::MemType>> : std::true_type {}  // ; intentionally skipped
#endif  // DEFINE_HAS_TYPE

DEFINE_HAS_TYPE(value_type);
DEFINE_HAS_TYPE(char_type);

int main()
{
    std::cout << HasTypeT_value_type<int>::value << '\n';               // false
    std::cout << HasTypeT_value_type<std::vector<int>>::value << '\n';  // true
    std::cout << HasTypeT_value_type<std::iostream>::value << '\n';     // false
    std::cout << HasTypeT_char_type<std::iostream>::value << '\n';      // true
}
```

##### 19.6.3 Detecting Non-type Members

```c++
#ifndef DEFINE_HAS_MEMBER
#define DEFINE_HAS_MEMBER(Member) \
template <typename, typename = std::void_t<>> \
struct HasMemberT_##Member : std::false_type {}; \
template <typename T> \
struct HasMemberT_##Member<T, std::void_t<decltype(&T::Member)>> : std::true_type {}  // ; intentionally skipped
#endif  // DEFINE_HAS_MEMBER

DEFINE_HAS_MEMBER(size);
DEFINE_HAS_MEMBER(first);

int main()
{
    std::cout << HasMemberT_size<int>::value << '\n';                   // false
    std::cout << HasMemberT_size<std::vector<int>>::value << '\n';      // true
    std::cout << HasMemberT_first<std::pair<int, int>>::value << '\n';  // true
}
```
Here, we use SFINAE to disable the partial specialization when `&T::Member` is not valid. 
For that construct to be valid, all of the following must be true:
- Member must unambiguously identify a member of `T` 
  (e.g., it can not be an overloaded member function name, 
  or the name of multiple inherited members of the same name);
- The member must be accessible;
- The member must be a non-type, none-numerator member (otherwise the prefix `&` would be invalid);
- If `T::Member` is a static data member, its type must not provide 
  an `operator&`that makes `&T::Member` invalid 
  (e.g., by making that operator inaccessible). 


It would not be difficult to modify the partial specialization 
to exclude cases where `&T::Member` is not a pointer-to-member type 
(via `std::is_member_pointer`, which amounts to excluding static data members). 
Similarly, a pointer-to-member function could be excluded or required 
to limit the trait to data members or member functions. 

##### Detecting Member Functions

Note that the `HasMember` trait only checks whether a _single_ member with the corresponding name exists. 
The trait also will fail if two members exists, which might happen if we check for overloaded member functions.
```c++
DEFINE_HAS_MEMBER(begin);

std::cout << HasMemberT_begin<std::vector<int>>::value << '\n'; // false
```
Fortunately, SFINAE principle protects against attempts 
to create both invalid _types_ and _expressions_ in a function template declaration, 
allowing the overloading technique above to extend to testing whether arbitrary expressions are well-formed. 


That is, we can simply check whether we can call a function of interest in a specific way 
and that can succeed even if the function is overloaded. 
As with the `IsConvertibleT` trait in Section 19.5, 
the trick is to formulate the expression that checks whether we can call `begin()` 
inside a `decltype` expression (or other unevaluated contexts) 
for the default value of an additional function template parameter:
```c++
// primary template:
template <typename, typename = std::void_t<>>
struct HasBeginT : std::false_type {};

// partial specialization (may be SFINAE’d away):
template <typename T>
struct HasBeginT<T, std::void_t<decltype(std::declval<T>().begin())>> : std::true_type {};
```
Here, we use `decltype(std::declval<T>().begin())` to test whether, 
given a value/object of type `T` (using `std::declval` to avoid any constructor being required), 
calling a member `begin()` is valid.
**Except** that, `decltype(call-expression)` does not require a non-reference, non-void return type to be complete, 
unlike call expressions in other contexts. 
Using `decltype(std::declval<T>().begin(), 0)` instead does add the requirement 
that the return type of the call is complete, 
because the returned value is no longer the result of the `decltype` operand. 

##### Detecting Other Expressions

We can use the technique above for other kinds of expressions and even combine multiple expressions. 
For example, we can test whether, given types `T1` and `T2`,
there is a suitable `<` operator defined for values of these types:
```c++
// primary template:
template <typename, typename, typename = std::void_t<>>
struct HasLessT : std::false_type {};

// partial specialization (may be SFINAE’d away):
template <typename T1, typename T2>
struct HasLessT<T1, T2, std::void_t<decltype(std::declval<T1>() < std::declval<T2>())>> : std::true_type {};

void foo()
{
    std::cout << HasLessT<int, char>::value << '\n';                                   // true
    std::cout << HasLessT<std::string, std::string>::value << '\n';                    // true
    std::cout << HasLessT<std::string, int>::value << '\n';                            // false
    std::cout << HasLessT<std::string, char *>::value << '\n';                         // true
    std::cout << HasLessT<std::complex<double>, std::complex<double>>::value << '\n';  // false
}
```
As always, the challenge is to define a valid expression for the condition to check 
and use decltype to place it in a SFINAE context, 
where it will cause a fallback to the primary template if the expression isn’t valid:
```c++
decltype(std::declval<T1>() < std::declval<T2>())
```
Traits that detect valid expressions in this manner are fairly robust: 
They will evaluate `true` only when the expression is well-formed 
and will correctly return `false` when the `<` operator is ambiguous, deleted, or inaccessible.


P.S. Prior to C++11's expansion of SFINAE to cover arbitrary invalid expressions, 
the techniques for detecting the validity of specific expressions 
centered on introducing a new overload for the function being tested (e.g., `<`) 
that had an overly permissive signature and a strangely sized return type 
to behave as a fallback case. 
However, such approaches were prone to ambiguities and caused errors due to access control violations. 


we can use this trait to require that a template parameter `T` supports `operator<`:
```c++
template <typename T>
class C
{
    static_assert(HasLessT<T>::value, "Class C requires comparable elements");
    ...
};
```

```c++
template <typename, typename = std::void_t<>>
struct HasVariousT : std::false_type {};

// partial specialization (may be SFINAE’d away):
template <typename T>
struct HasVariousT<T, std::void_t<decltype(std::declval<T>().begin()),
                                  typename T::difference_type,
                                  typename T::iterator>>
        : std::true_type
{
};
```

##### Check If A Type Is Hashable

Whether hashable as well as customized hash functions (DIRTY!): 
```c++
#include <boost/container_hash/hash.hpp>
#include <fmt/core.h>
#include <fmt/format.h>


template <typename T, typename = std::void_t<>>
struct is_hashable : std::false_type {};


template <typename T>
struct is_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>> : std::true_type {};


template <typename T>
constexpr bool is_hashable_v = is_hashable<T>::value;


template <>
struct std::hash<std::tuple<int, int>>
{
    std::size_t operator()(const std::tuple<int, int> & t) const
    {
        std::size_t seed {0};
        boost::hash_combine(seed, std::get<0>(t));
        boost::hash_combine(seed, std::get<1>(t));
        return seed;
    }
};


template <>
struct std::hash<std::pair<int, int>> : std::hash<std::tuple<int, int>>
{
    using std::hash<std::tuple<int, int>>::operator();
};


int main(int argc, char * argv[])
{
    fmt::print(FMT_STRING("{}\n"), is_hashable_v<int>);                   // true
    fmt::print(FMT_STRING("{}\n"), is_hashable_v<std::tuple<int, int>>);  // true
    fmt::print(FMT_STRING("{}\n"), is_hashable_v<std::pair<int, int>>);   // true
    fmt::print(FMT_STRING("{}\n"), is_hashable_v<std::vector<int>>);      // false

    return EXIT_SUCCESS;
}
```
References: 
- [cpprefernece `std::unordered_map`](https://en.cppreference.com/w/cpp/container/unordered_map)
- [cppreference `std::hash`](https://en.cppreference.com/w/cpp/utility/hash)
- [`Boost.ContainerHash` "Combining hash values"](https://www.boost.org/doc/libs/1_79_0/libs/container_hash/doc/html/hash.html#combine)
- [StackOverflow "Check if a type is hashable"](https://stackoverflow.com/questions/12753997/check-if-type-is-hashable)

##### 19.6.4 Using Generic Lambdas to Detect Members

The `isValid` lambda introduced in Section 19.4.3 
provides a more compact technique to define traits that check for members, 
helping is to avoid the use of macros to handle members if arbitrary names.


The following example illustrates how to define traits 
checking whether a data or type member such as `first` or `size_type` exists 
or whether `operator<` is defined for two objects of different types:
```c++
// define to check for data member first:
constexpr auto hasFirst = isValid([](auto x) -> decltype((void) valueT(x).first) {});
std::cout << "hasFirst: " << hasFirst(type<std::pair<int, int>>) << '\n';  // true

// define to check for member type size_type:
constexpr auto hasSizeType = isValid([](auto x) -> typename decltype(valueT(x))::size_type {});

struct CX
{
    using size_type = std::size_t;
};

std::cout << "hasSizeType: " << hasSizeType(type<CX>) << '\n';  // true

if constexpr(!hasSizeType(type<int>))
{
    std::cout << "int has no size_type\n";
}

// define to check for <:
constexpr auto hasLess = isValid([](auto x, auto y) -> decltype(valueT(x) < valueT(y)) {});

std::cout << hasLess(42, type<char>) << '\n';                        // true
std::cout << hasLess(type<std::string>, type<std::string>) << '\n';  // true
std::cout << hasLess(type<std::string>, type<int>) << '\n';          // false
std::cout << hasLess(type<std::string>, "hello") << '\n';            // true
```
Note again that `hasSizeType` uses `std::decay` to remove the references from the passed `x` 
because you can’t access a type member from a reference. 
If you skip that, the traits will always yield `false`
because the second overload of `isValidImpl` is used. 


To be able to use the common generic syntax, taking types as template parameters,
we can again define additional helpers. 
```c++
constexpr auto hasFirst = isValid([](auto && x) -> decltype((void) &x.first) {});

template <typename T>
using HasFirstT = decltype(hasFirst(std::declval<T>()));

constexpr auto hasSizeType = isValid([](auto && x) -> typename std::decay_t<decltype(x)>::size_type {});

template <typename T>
using HasSizeTypeT = decltype(hasSizeType(std::declval<T>()));

constexpr auto hasLess = isValid([](auto && x, auto && y) -> decltype(x < y) {});

template <typename T1, typename T2>
using HasLessT = decltype(hasLess(std::declval<T1>(), std::declval<T2>()));
```


#### 📌 19.7 Other Traits Techniques

##### 19.7.1 If-Then-Else

```c++
template <bool COND, typename TrueType, typename FalseType>
struct IfThenElseT
{
    using Type = TrueType;
};

// partial specialization: false yields third argument
template <typename TrueType, typename FalseType>
struct IfThenElseT<false, TrueType, FalseType>
{
    using Type = FalseType;
};

template <bool COND, typename TrueType, typename FalseType>
using IfThenElse = typename IfThenElseT<COND, TrueType, FalseType>::Type;

template <auto N>
struct SmallestIntT
{
    using Type = IfThenElse<N <= std::numeric_limits<char>::max(),
                            char,
                            IfThenElse<N <= std::numeric_limits<short>::max(),
                                       short,
                                       IfThenElse<N <= std::numeric_limits<int>::max(),
                                                  int,
                                                  IfThenElse<N <= std::numeric_limits<long>::max(), 
                                                             long,
                                                             IfThenElse<N <= std::numeric_limits<long long>::max(),
                                                                        long long, 
                                                                        void  //then fallback
                                                                        >
                                                             >
                                                  >
                                       >
                            >;
};

template <auto N>
using SmallestInt = typename SmallestIntT<N>::Type;
```
Note that, unlike a normal C++ if-then-else statement, 
the template arguments for both the “then” and “else” branches are evaluated before the selection is made, 
so **neither** branch may ~~contain ill-formed code~~, otherwise, the program is likely to be ill-formed.
For example, consider a trait that yields the corresponding unsigned type for a given signed type. 
There is a standard trait `std::make_unsigned` which does this conversion, 
but it requires that the passed type is a signed integral type and no bool; 
otherwise its use results in undefined behavior.
So it might be a good idea to implement a trait that yields the corresponding unsigned type 
if this is possible and the passed type otherwise 
(thus, avoiding undefined behavior if an inappropriate type is given). 
The naive implementation does **not** work:
```c++
// ERROR: undefined behavior if T is bool or no integral type:
template <typename T>
struct UnsignedT 
{
    using Type = IfThenElse<std::is_integral<T>::value && !std::is_same<T, bool>::value,
                            typename std::make_unsigned<T>::type, 
                            T>;
};
```
The instantiation of `UnsignedT<bool>` is _still undefined behavior_, 
because the compiler will still attempt to form the type from `typename std::make_unsigned<T>::type`. 
To address this problem, we need to add an additional level of indirection:
```c++
// yield T when using member Type:
template <typename T>
struct IdentityT 
{
    using Type = T;
};

// to make unsigned after IfThenElse was evaluated:
template <typename T>
struct MakeUnsignedT 
{
    using Type = typename std::make_unsigned<T>::type;
};

template <typename T>
struct UnsignedT 
{
    using Type = typename IfThenElse<std::is_integral<T>::value && !std::is_same<T, bool>::value,
                                     MakeUnsignedT<T>,
                                     IdentityT<T>>::Type;
};
```
In this definition of `UnsignedT`, the type arguments to `IfThenElse` are both instances of type functions themselves. 
However, the type functions are **not** actually evaluated before `IfThenElse` selects one. 
Instead, `IfThenElse` selects the type function instance (of either `MakeUnsignedT` or `IdentityT`). 
The `::Type` then evaluates the selected type function instance to produce `Type`. 


It is worth emphasizing here that this relies entirely on the fact that 
the not-selected wrapper type in the `IfThenElse` construct is **never** fully instantiated. 
In particular, the following does **not** work:
```c++
// DOES NOT WORK!
template <typename T>
struct UnsignedT
{
    using Type = IfThenElse<std::is_integral<T>::value && !std::is_same<T, bool>::value,
                            typename MakeUnsignedT<T>::Type,
                            T>;
};
```
Instead, we have to apply the `::Type` for `MakeUnsignedT<T>` _later_, 
which means that we need the `IdentityT` helper to also apply `::Type` later for `T` in the _else_ branch.


This also means that in this context, we can **not** use something like
```c++
template <typename T>
using Identity = typename IdentityT<T>::Type;
```
We can declare such an alias template, and it may be useful elsewhere, 
but we can **not** make effective use of it in the definition of `IfThenElse`
because any use of `Identity<T>` immediately causes 
the full instantiation of `IdentityT<T>` to retrieve its `Type` member. 
The `IfThenElseT` template is available in the C++ standard library as 
[`std::conditional`](https://en.cppreference.com/w/cpp/types/conditional). 
With it, the `UnsignedT` trait would be defined as follows:
```c++
template <typename T>
struct IdentityT 
{
    using Type = T;
};

template <typename T>
struct MakeUnsignedT 
{
    using Type = typename std::make_unsigned<T>::type;
};

template <typename T>
struct UnsignedT 
{
    using Type = typename std::conditional_t<std::is_integral<T>::value && !std::is_same<T, bool>::value,
                                             MakeUnsignedT<T>,
                                             IdentityT<T>>::Type;
};
```

##### 19.7.2 Detecting Non-throwing Operations

It is occasionally useful to determine whether a particular operation can throw an exception. 
For example, a _move constructor_ should be marked `noexcept`,
indicating that it does not throw exceptions, whenever possible. 
However, whether a move constructor for a particular class throws exceptions 
often depends on whether the move constructors of its members and bases throw.
```c++
template <typename T1, typename T2>
class Pair
{
public:
    Pair(Pair && other)
    noexcept(std::is_nothrow_move_constructible_v<T1> && 
             std::is_nothrow_move_constructible_v<T2>)
            : first(std::forward<T1>(other.first)),
              second(std::forward<T2>(other.second))
    {
    }

private:
    T1 first;
    T2 second;
};
```
`Pair`’s move constructor can throw exceptions
when the move operations of either `T1` or `T2` can throw.
we can express this property by using a computed `noexcept` exception specification in `Pair`’s move constructor.
```c++
// NOT SFINAE-FRIENDLY!
template <typename T>
struct IsNothrowMoveConstructibleT : std::bool_constant<noexcept(T(std::declval<T &&>()))>
{};
```
Here, the operator version of `noexcept` is used, 
which determines whether an expression is non-throwing.
Because the result is a Boolean value, we can pass it directly to define the base class `std::bool_constant`, 
which is used to define `std::true_type` and `std::false_type`.
(P.S. In C++11 and C++14, we have to specify the base class as 
`std::integral_constant<bool, ...>` instead of `std::bool_constant<...>`.)


However, this implementation should be improved because it is **not** SFINAE-friendly: 
If the trait is instantiated with a type that does not have a usable move or copy constructor, 
which makes the expression `T(std::declval<T &&>())` invalid, 
the entire program will be ill-formed:
```c++
class E 
{
public:
    E(E &&) = delete;
};

std::cout << IsNothrowMoveConstructibleT<E>::value << '\n';  // compile-time ERROR
```
Instead of aborting the compilation, the type trait should yield a value of `false`.


As discussed in Section 19.4.4, 
we have to check whether the expression computing the result is valid before it is evaluated. 
Here, we have to find out whether move construction is valid before checking whether it is `noexcept`:
```c++
// primary template:
template <typename T, typename = std::void_t<>>
struct IsNothrowMoveConstructibleT : std::false_type
{};

// partial specialization (may be SFINAE’d away):
template <typename T>
struct IsNothrowMoveConstructibleT<T, std::void_t<decltype(T(std::declval<T>()))>>
        : std::bool_constant<noexcept(T(std::declval<T>()))>
{};
```
If the substitution of `std::void_t` in the partial specialization is valid,
that specialization is selected, 
and the `noexcept` expression in the base class specifier can safely be evaluated. 
Otherwise, the partial specialization is discarded without instantiating it, 
and the primary template is instantiated instead (producing a `std::false_type` result).


Note that there is **no** way to check whether a move constructor throws without being able to call it directly. 
That is, it is not enough that the move constructor is public and not deleted, 
it also requires that the corresponding type is no abstract class
(references or pointers to abstract classes work fine). 
For this reason, the type trait is named `IsNothrowMoveConstructible` instead of `HasNothrowMove-Constructor`. 
For anything else, we’d need compiler support.

##### 19.7.3 Traits Convenience

One common complaint with type traits is their relative verbosity, 
because each use of a type trait typically requires a trailing `::Type` and, 
in a dependent context, a leading `typename` keyword, both of which are boilerplate. 
When multiple type traits are composed, this can force some awkward formatting, 
as in our running example of the array `operator+`, 
if we would implement it correctly, ensuring that no constant or reference type is returned:
```c++
template <typename T1, typename T2>
Array<typename RemoveCVT<typename RemoveReferenceT<typename PlusResultT<T1, T2>::Type>::Type>::Type> 
operator+(Array<T1> const &, Array<T2> const &);
```
By using alias templates and variable templates, 
we can make the usage of the traits, yielding types or values respectively more convenient. 
However, note that in some contexts these shortcuts are not usable, 
and we have to use the original class template instead. 
We already discussed one such situation in our `UnsignedT` example, 
but a more general discussion follows.

##### Alias Templates and Traits

There are downsides to using alias templates for type traits:
1. Alias templates can **not** be specialized (as noted in Section 16.3), 
   and since many of the techniques for writing traits depend on specialization, 
   an alias template will likely need to redirect to a class template anyway.
2. Some traits are meant to be specialized by users, 
   such as a trait that describes whether a particular addition operation is commutative, 
   and it can be confusing to specialize the class template when most uses involve the alias template.
3. The use of an alias template _will always instantiate the type_ 
   (e.g., the underlying class template specialization),
   which makes it harder to avoid instantiating traits that don’t make sense for a given type 
   (as discussed in Section 19.7.1).


Another way to express this last point, 
is that alias templates can **not** be used with _metafunction forwarding_ (see Section 19.3.2).


Because the use of alias templates for type traits has both positive and negative aspects, 
we recommend using them as we have in this section and as it is done in the C++ standard library: 
Provide both class templates with a specific naming convention (we have chosen the `T` suffix and the `Type` type member) 
and alias templates with a slightly different naming convention (we dropped the `T` suffix), 
and have each alias template defined in terms of the underlying class template. 
This way, we can use alias templates wherever they provide clearer code, 
but fall back to class templates for more advanced uses.


Note that for historical reasons, the C++ standard library has a different convention. 
The type traits class templates yield a type in type and have no specific suffix (many were introduced in C++11). 
Corresponding alias templates (that produce the type directly) started being introduced in C++14, 
and were given a `_t` suffix, since the un-suffixed name was already standardized.

##### Variable Templates and Traits

Traits returning a value require a trailing `::value` (or similar member selection)
to produce the result of the trait. 
In this case, `constexpr` variable templates offer a way to reduce this verbosity.


Again, for historical reasons, the C++ standard library has a different convention.
The traits class templates producing a result in value have no specific suffix,
and many of them were introduced in the C++11 standard. 
The corresponding variable templates that directly produce the resulting value
were introduced in C++17 with a `_v` suffix.


The C++ standardization committee is further bound by a long-standing tradition
that all standard names consist of lowercase characters and optional underscores to separate them. 
That is, a name like `isSame` or `IsSame` is unlikely to ever be seriously considered for standardization. 


#### 📌 19.8 Type Classification


It is sometimes useful to be able to know whether a template parameter 
is a built-in type, a pointer type, a class type, and so on. 
In the following sections, we develop a suite of type traits that allow us to determine various properties of a given type. 
As a result, we will be able to write code specific for some types:
```c++
if (IsClassT<T>::value) {}

if constexpr (IsClass<T>) {}

// primary template for the general case
template <typename T, bool = IsClass<T>>
class C {};

// partial specialization for class types
template <typename T>
class C<T, true> {};
```
Furthermore, expressions such as `IsPointerT<T>::value` 
will be Boolean constants that are valid non-type template arguments. 
In turn, this allows the construction of more sophisticated and more powerful templates 
that specialize their behavior on the properties of their type arguments. 

##### 19.8.1 Determining Fundamental Types

```c++
/// <type_traits>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

// composite type categories. 

/// is_arithmetic
template<typename _Tp>
  struct is_arithmetic
  : public __or_<is_integral<_Tp>, is_floating_point<_Tp>>::type
  { };

/// is_fundamental
template<typename _Tp>
  struct is_fundamental
  : public __or_<is_arithmetic<_Tp>, is_void<_Tp>,
         is_null_pointer<_Tp>>::type
  { };

// Primary type categories.

template<typename>
struct __is_null_pointer_helper
: public false_type { };

template<>
struct __is_null_pointer_helper<std::nullptr_t>
: public true_type { };

/// is_null_pointer (LWG 2247).
template<typename _Tp>
struct is_null_pointer
: public __is_null_pointer_helper<typename remove_cv<_Tp>::type>::type
{ };

/// __is_nullptr_t (extension).
template<typename _Tp>
struct __is_nullptr_t
: public is_null_pointer<_Tp>
{ };

template<typename>
struct __is_void_helper
: public false_type { };

template<>
struct __is_void_helper<void>
: public true_type { };

/// is_void
template<typename _Tp>
struct is_void
: public __is_void_helper<typename remove_cv<_Tp>::type>::type
{ };

/// is_integral
/// is_integral_helper: brute-force enumeration
template<typename _Tp>
struct is_integral
: public __is_integral_helper<typename remove_cv<_Tp>::type>::type
{ };

template<typename>
struct __is_floating_point_helper
: public false_type { };

template<>
struct __is_floating_point_helper<float>
: public true_type { };

template<>
struct __is_floating_point_helper<double>
: public true_type { };

template<>
struct __is_floating_point_helper<long double>
: public true_type { };

#if !defined(__STRICT_ANSI__) && defined(_GLIBCXX_USE_FLOAT128) && !defined(__CUDACC__)
template<>
struct __is_floating_point_helper<__float128>
: public true_type { };
#endif

/// is_floating_point
template<typename _Tp>
struct is_floating_point
: public __is_floating_point_helper<typename remove_cv<_Tp>::type>::type
{ };
```

##### 19.8.2 Determining Compound Types

```c++
/// <type_traits>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

// array

/// is_array
template<typename>
struct is_array
: public false_type { };

template<typename _Tp, std::size_t _Size>
struct is_array<_Tp[_Size]>
: public true_type { };

template<typename _Tp>
struct is_array<_Tp[]>
: public true_type { };

// pointer

template<typename>
  struct __is_pointer_helper
  : public false_type { };

template<typename _Tp>
  struct __is_pointer_helper<_Tp*>
  : public true_type { };

/// is_pointer
template<typename _Tp>
  struct is_pointer
  : public __is_pointer_helper<typename remove_cv<_Tp>::type>::type
  { };

// reference

/// is_lvalue_reference
template<typename>
  struct is_lvalue_reference
  : public false_type { };

template<typename _Tp>
  struct is_lvalue_reference<_Tp&>
  : public true_type { };

/// is_rvalue_reference
template<typename>
  struct is_rvalue_reference
  : public false_type { };

template<typename _Tp>
  struct is_rvalue_reference<_Tp&&>
  : public true_type { };

// pointer to member

template<typename>
struct __is_member_object_pointer_helper
: public false_type { };

template<typename _Tp, typename _Cp>
struct __is_member_object_pointer_helper<_Tp _Cp::*>
: public __not_<is_function<_Tp>>::type { };

/// is_member_object_pointer
template<typename _Tp>
struct is_member_object_pointer
: public __is_member_object_pointer_helper<
typename remove_cv<_Tp>::type>::type
{ };

template<typename>
struct __is_member_function_pointer_helper
: public false_type { };

template<typename _Tp, typename _Cp>
struct __is_member_function_pointer_helper<_Tp _Cp::*>
: public is_function<_Tp>::type { };

/// is_member_function_pointer
template<typename _Tp>
struct is_member_function_pointer
: public __is_member_function_pointer_helper<
typename remove_cv<_Tp>::type>::type
{ };
```

##### 19.8.3 Identifying Function Types

```c++
/// <c++config.h>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

#define _GLIBCXX_NOEXCEPT_PARM , bool _NE
#define _GLIBCXX_NOEXCEPT_QUAL noexcept (_NE)

/// <type_traits>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

/// is_function
template<typename>
  struct is_function
  : public false_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) const _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) const & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) const && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) const _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) const & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) const && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) volatile _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) volatile & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) volatile && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) volatile _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) volatile & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) volatile && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) const volatile _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) const volatile & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes...) const volatile && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) const volatile _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) const volatile & _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };

template<typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
  struct is_function<_Res(_ArgTypes......) const volatile && _GLIBCXX_NOEXCEPT_QUAL>
  : public true_type { };
```

##### 19.8.4 Determining Class Types && 19.8.5 Determining Enumeration Types

Unlike the other compound types we have handled so far,
we have **no** partial specialization patterns that match class types specifically.
**Nor** is it feasible to enumerate all class types, as it is for fundamental types. 
Instead, we need to use an indirect method to identify class types, 
by coming up with some kind of type or expression that is valid for all class types (and not other type). 
With that type or expression, we can apply the SFINAE trait techniques.


The most convenient property of class types to utilize in this case is that 
only class types can be used as the basis of pointer-to-member types. 
That is, in a type construct of the form `X Y::*`, `Y` can only be a class type. 
The following formulation of `IsClassT` exploits this property 
(and picks `int` arbitrarily for type `X`):
```c++
// primary template: by default no class
template <typename T, typename = std::void_t<>>
struct IsClassT : std::false_type {};

// classes can have pointer-to-member
template <typename T>
struct IsClassT<T, std::void_t<int T::*>> : std::true_type {};
```
The C++ language specifies that the type of a lambda expression is a 
“unique, unnamed non-union class type.” 
For this reason, lambda expressions yield `true` when examining whether they are class type objects:
```c++
auto l = [] {};
static_assert(IsClassT<decltype(l)>::value, "");  // succeeds
```
Note also that the expression `int T::*` is also valid for union types 
(they are also class types according to the C++ standard). 


The C++ standard library provides the traits `std::is_class` and `std::is_union`. 
However, these traits require special compiler support 
because distinguishing class and struct types from union types
can **not** currently be done using any standard core language techniques. 


P.S. `libstdc++` implements this as black magic.
```c++
/// <type_traits>
/// g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

/// is_enum
template<typename _Tp>
  struct is_enum
  : public integral_constant<bool, __is_enum(_Tp)>
  { };

/// is_union
template<typename _Tp>
  struct is_union
  : public integral_constant<bool, __is_union(_Tp)>
  { };

/// is_class
template<typename _Tp>
  struct is_class
  : public integral_constant<bool, __is_class(_Tp)>
  { };
```


#### 📌 19.9 Policy Traits

##### 19.9.1 Read-Only Parameter Types

Consider a function that maps an intended argument type `T` 
onto the optimal parameter-type-for-copy-operations `T` or `T const &`. 


As a first approximation, 
the primary template can use by-value passing for types no larger than two pointers 
and by reference-to-const for everything else:
```c++
template <typename T>
struct RParam 
{
    using Type = std::conditional_t<sizeof(T) <= 2 * sizeof(void *), 
                                    T,
                                    T const &>;
};
```
On the other hand, container types for which `sizeof` returns a small value may involve expensive copy constructors, 
so we may need many specializations and partial specializations, such as the following:
```c++
template <typename T>
struct RParam<Array<T>> 
{
    using Type = Array<T> const &;
};
```
Because such types are common in C++, 
it may be safer to mark only small types with trivial copy and move constructors as by value types 
and then selectively add other class types when performance considerations dictate it:
```c++
template <typename T>
struct RParam 
{
    using Type = typename std::conditional<sizeof(T) <= 2 * sizeof(void *) && 
                                                   std::is_trivially_copy_constructible_v<T> && 
                                                   std::is_trivially_move_constructible_v<T>, 
                                           T,
                                           T const &>::type;
};
```
Either way, the policy can now be centralized in the traits template definition, 
and clients can exploit it to good effect. 
For example, let’s suppose we have two classes,
with one class specifying that calling by value is better for read-only arguments:
```c++
class MyClass1
{
public:
    MyClass1() {}

    MyClass1(MyClass1 const &)
    {
        std::cout << "MyClass1(MyClass1 const &)\n";
    }
};

class MyClass2
{
public:
    MyClass2() {}

    MyClass2(MyClass2 const &)
    {
        std::cout << "MyClass2(MyClass2 const &)\n";
    }
};

// pass MyClass2 objects with RParam by value
template <>
class RParam<MyClass2>
{
public:
    using Type = MyClass2;
};
```
Now, we can declare functions that use `RParam` for read-only arguments and call these functions:
```c++
// function that allows parameter passing by value or by reference
template <typename T1, typename T2>
void foo(typename RParam<T1>::Type p1, typename RParam<T2>::Type p2)
{
    // ...
}

void bar()
{
    MyClass1 mc1;
    MyClass2 mc2;
    foo<MyClass1, MyClass2>(mc1, mc2);  // can NOT be called with deduction!
}
```
Unfortunately, there are some significant **downsides** to using `RParam`. 
1. The function declaration is significantly messier. 
2. A function like `foo` can **not** be called with argument deduction 
   because the template parameter appears only in the qualifiers of the function parameters. 
   Call sites must therefore specify explicit template arguments.


An unwieldy workaround for this option is the use of an inline wrapper function template
that provides perfect forwarding, but it assumes the inline function will be elided by the compiler. 
For example:
```c++
// function that allows parameter passing by value or by reference
template <typename T1, typename T2>
void foo_core(typename RParam<T1>::Type p1, typename RParam<T2>::Type p2)
{
    // ...
}

// wrapper to avoid explicit template parameter passing
template <typename T1, typename T2>
void foo(T1 && p1, T2 && p2)
{
    foo_core<T1, T2>(std::forward<T1>(p1), std::forward<T2>(p2));
}

void bar()
{
    MyClass1 mc1;
    MyClass2 mc2;
    foo(mc1, mc2);
}
```


#### 📌 19.10 In the Standard Library


With C++11, type traits became an intrinsic part of the C++ standard library. 
They comprise more or less all type functions and type traits discussed in this chapter.
However, for some of them, such as the trivial operation detection traits 
and as discussed `std::is_union`, there are **no** known in-language solutions. 
Rather, the compiler provides intrinsic support for those traits. 
Also, compilers start to support traits even if there are in-language solutions to shorten compile time. 


For this reason, if you need type traits, 
we recommend that you use the ones from the C++ standard library whenever available. 
They are all described in detail in Appendix D. 


Note that (as discussed) some traits have potentially surprising behavior
(at least for the naive programmer). 
In addition to the general hints we give in Section 11.2.1 on and Section D.1.2, 
also consider the specific descriptions we provide in Appendix D. 


The C++ standard library also defines some policy and property traits:
- The class template `std::char_traits` is used as a policy traits parameter by the string and I/O stream classes.
- To adapt algorithms easily to the kind of standard iterators for which they are used,
  a very simple `std::iterator_traits` property traits template is provided (and used in standard library interfaces).
- The template `std::numeric_limits` can also be useful as a property traits template.
- Memory allocation for the standard container types is handled using policy traits classes. 
  Since C++98, the template `std::allocator` is provided as the standard component for this purpose. 
  With C++11, the template `std::allocator_traits` was added 
  to be able to change the policy/behavior of allocators
  (switching between the classical behavior and scoped allocators; 
  the latter could not be accommodated within the pre-C++11 framework). 






### 🎯 Chapter 20 Overloading on Type Properties

#### 📌 20.1 Algorithm Specialization

One of the common motivations behind overloading of function templates 
is to provide more specialized versions of an algorithm 
based on knowledge of the types involved.
The design and optimization approach of introducing 
more specialized variants of a generic algorithm 
is called _algorithm specialization_. 
```c++
/// with C++20 concepts utilities

template <std::input_iterator InputIterator>
void advance(InputIterator & x, typename InputIterator::difference_type n)
{
    // linear time
    while (0 < n)
    {
        ++x;
        --n;
    }
}

template <std::random_access_iterator RandomAccessIterator>
void advance(RandomAccessIterator & x, typename RandomAccessIterator::difference_type n)
{
    // constant time
    x += n;
}
```

#### 📌 20.2 Tag Dispatching

```c++
template <typename Iterator, typename Distance>
void advanceIter(Iterator & x, Distance n)
{
    advanceIterImpl(x, n, typename std::iterator_traits<Iterator>::iterator_category());
}

template <typename Iterator, typename Distance>
void advanceIterImpl(Iterator & x, Distance n, std::input_iterator_tag)
{
    while (n > 0)
    {
        ++x;
        --n;
    }
}

template <typename Iterator, typename Distance>
void advanceIterImpl(Iterator & x, Distance n, std::random_access_iterator_tag)
{
    x += n;
}
```

#### 📌 20.3 Enabling/Disabling Function Templates

```c++
/// <type_traits>
/// g++ (Ubuntu 11.1.0-1ubuntu1~20.04) 11.1.0

// Primary template.
/// Define a member typedef @c type only if a boolean constant is true.
template <bool, typename T = void>
struct enable_if {};

// Partial specialization for true.
template <typename T>
struct enable_if<true, T>
{
    typedef T type;
};

/// Alias template for enable_if
template <bool Cond, typename T = void>
using enable_if_t = typename enable_if<Cond, T>::type;
```

#### 📌 20.4 Class Specialization

#### 20.4.1 Enabling/Disabling Class Templates

```c++
template <typename T, typename = std::void_t<>>
struct is_less_comparable : public std::false_type {};

template <typename T>
struct is_less_comparable<T, std::void_t< decltype(std::declval<T>() < std::declval<T>()) >> : public std::true_type {};

template <typename T>
constexpr bool is_less_comparable_v = is_less_comparable<T>::value;

template <typename T, typename = std::void_t<>>
struct is_hashable : public std::false_type {};

template <typename T>
struct is_hashable<T, std::void_t< decltype( std::declval< std::hash<T>()(std::declval<T>()) >() ) >> : public std::true_type {};

template <typename T>
constexpr bool is_hashable_v = is_hashable<T>::value;

template <typename K, typename V, typename = void>
class Dictionary
{ 
    // vector implementation
};

template <typename K, typename V>
class Dictionary<K, V, std::enable_if_t<is_less_comparable_v<K> && !is_hashable_v<K>>>
{
    // tree implementation
};

template <typename K, typename V>
class Dictionary<K, V, std::enable_if_t<is_hashable_v<K>>>
{
    // hash table implementation
};
```

#### 20.4.2 Tag Dispatching for Class Templates

```c++
// construct a set of match() overloads for the types in Types…:
template <typename ... Types>
struct MatchOverloads;

// basis case: nothing matched:
template <>
struct MatchOverloads<>
{
    static void match(...);
};

// recursive case: introduce a new match() overload:
template <typename T1, typename ... Rest>
struct MatchOverloads<T1, Rest...> : public MatchOverloads<Rest...>
{
    using MatchOverloads<Rest...>::match;  // collect overloads from bases
    
    static T1 match(T1);  // introduce overload for T1
};

// find the best match for T in Types...
template <typename T, typename ... Types>
struct BestMatchInSetT
{
    using Type = decltype(MatchOverloads<Types...>::match(declval<T>()));
};

template <typename T, typename ... Types>
using BestMatchInSet = typename BestMatchInSetT<T, Types...>::Type;

// primary template (intentionally undefined):
template <typename Iterator,
        typename Tag =
        BestMatchInSet<
                typename std::iterator_traits<Iterator>
                ::iterator_category,
                std::input_iterator_tag,
                std::bidirectional_iterator_tag,
                std::random_access_iterator_tag>>
class Advance;

// general, linear-time implementation for input iterators:
template <typename Iterator>
class Advance<Iterator, std::input_iterator_tag>
{
public:
    using DifferenceType = typename std::iterator_traits<Iterator>::difference_type;

    void operator()(Iterator & x, DifferenceType n) const
    {
        while (n > 0)
        {
            ++x;
            --n;
        }
    }
};

// bidirectional, linear-time algorithm for bidirectional iterators:
template <typename Iterator>
class Advance<Iterator, std::bidirectional_iterator_tag>
{
public:
    using DifferenceType = typename std::iterator_traits<Iterator>::difference_type;

    void operator()(Iterator & x, DifferenceType n) const
    {
        if (n > 0)
        {
            while (n > 0)
            {
                ++x;
                --n;
            }
        }
        else
        {
            while (n < 0)
            {
                --x;
                ++n;
            }
        }
    }
};

// bidirectional, constant-time algorithm for random access iterators:
template <typename Iterator>
class Advance<Iterator, std::random_access_iterator_tag>
{
public:
    using DifferenceType = typename std::iterator_traits<Iterator>::difference_type;

    void operator()(Iterator & x, DifferenceType n) const
    {
        x += n;
    }
};
```

#### 📌 20.5 Instantiation-Safe Templates

```c++
template <typename T1, typename T2>
class HasLess
{
private:
    template <typename T>
    struct Identity;

    template <typename U1, typename U2>
    static std::true_type
    test(Identity<decltype(std::declval<U1>() < std::declval<U2>())> *);

    template <typename U1, typename U2>
    static std::false_type
    test(...);

public:
    static constexpr bool value = decltype(test<T1, T2>(nullptr))::value;
};

template <typename T1, typename T2, bool HasLess>
class LessResultImpl
{
public:
    using Type = decltype(std::declval<T1>() < std::declval<T2>());
};

template <typename T1, typename T2>
class LessResultImpl<T1, T2, false> {};

template <typename T1, typename T2>
class LessResultT : public LessResultImpl<T1, T2, HasLess<T1, T2>::value> {};

template <typename T1, typename T2>
using LessResult = typename LessResultT<T1, T2>::Type;

template <typename T>
std::enable_if_t<std::is_convertible_v<LessResult<T const &, T const &>, bool>, T const &>
min(T const & x, T const & y)
{
    if (y < x) 
    {
        return y;
    }
    
    return x;
}

struct X1 {};

bool operator<(X1 const &, X1 const &)
{
    return true;
}

struct X2 {};

bool operator<(X2, X2)
{
    return true;
}

struct X3 {};

bool operator<(X3 &, X3 &)
{
    return true;
}

struct X4 {};

struct BoolConvertible
{
    // implicit conversion to bool
    operator bool() const
    {
        return true;
    }
};

struct X5 {};

BoolConvertible operator<(X5 const &, X5 const &)
{
    return BoolConvertible();
}

struct NotBoolConvertible
{ 
    // no conversion to bool
};

struct X6 {};

NotBoolConvertible operator<(X6 const &, X6 const &)
{
    return NotBoolConvertible();
}

struct BoolLike
{
    // explicit conversion to bool
    explicit operator bool() const
    {
        return true;
    } 
};

struct X7 {};

BoolLike operator<(X7 const &, X7 const &)
{
    return BoolLike();
}

void foo()
{
    min(X1(), X1());  // OK:    X1 can be passed to min()
    min(X2(), X2());  // OK:    X2 can be passed to min()
    min(X3(), X3());  // ERROR: X3 cannot be passed to min()
    min(X4(), X4());  // ERROR: X4 can be passed to min()
    min(X5(), X5());  // OK:    X5 can be passed to min()
    min(X6(), X6());  // ERROR: X6 cannot be passed to min()
    min(X7(), X7());  // UNEXPECTED ERROR: X7 cannot be passed to min()
                      //                   std::is_convertible_to_v<BoolLike, bool> == false!
}
```
`X7` illustrates some of the subtleties of  implementing instantiation-safe templates. 
In particular, if `X7` is passed to the non-instantiation-safe `min`, instantiation will succeed. 
However, the instantiation-safe `min` rejects it because `BoolLike` is **not** implicitly convertible to `bool`. 
The distinction here is particularly subtle: 
An `explicit` conversion to `bool` can be used _implicitly_ in certain contexts, 
including in Boolean conditions for 
control-flow statements (`if`, `while`, `for`, and `do while`), 
the built-in `!`, `&&,` and `||` operators, 
and the ternary operator `?:`. 
In these contexts, the value is said to be contextually converted to `bool`. 


However, our insistence on having a general, implicit conversion to `bool` has the effect 
that our instantiation-safe template is _over-constrained_; 
that is, its specified requirements (in the `enable_if`) are stronger than its actual requirements
(what the template needs to instantiate properly). 
If, on the other hand, we had entirely forgotten the conversion-to-bool requirement, 
our `min` template would have been _under-constrained_, 
and it would have allowed some template arguments that could cause an instantiation failure (such as `X6`). 


To fix the instantiation-safe `min`, we need a trait to determine 
whether a type `T`is contextually convertible to `bool`. 
The control-flow statements are _not_ helpful in defining this trait,
because statements can **not** occur within a SFINAE context, 
**nor** are the logical operations, which can be overloaded for an arbitrary type. 
Fortunately, the ternary operator `?:` is an expression and is not overloadable, 
so it can be exploited to test whether a type is contextually convertible to `bool`:
```c++
template <typename T>
class IsContextualBoolT
{
private:
    template <typename U>
    struct Identity;

    template <typename U>
    static std::true_type
    test(Identity<decltype(declval<U>() ? 0 : 1)> *);

    template <typename U>
    static std::false_type
    test(...);

public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <typename T>
constexpr bool IsContextualBool = IsContextualBoolT<T>::value;

template <typename T>
std::enable_if_t<IsContextualBool<LessResult<T const &, T const &>, bool>, T const &>
min(T const & x, T const & y)
{
    if (y < x) 
    {
        return y;
    }
    
    return x;
}
```






### 🎯 Chapter 21 Templates and Inheritance

A priori, there might be no reason to think that templates and inheritance interact in interesting ways. 
If anything, we know from Chapter 13 that deriving from dependent base classes 
forces us to deal carefully with unqualified names. 
However, it turns out that some interesting techniques combine these two features, 
including the _Curiously Recurring Template Pattern (CRTP)_ and *mixin*s. 

#### 📌 21.1 The Empty Base Class Optimization (EBCO)

C++ classes are often "empty," which means that
their internal representation does **not** require any bits of memory at run time. 
This is the case typically for classes that contain only 
type members, non-virtual function members, and static data members.
Non-static data members, virtual functions, and virtual base classes, 
on the other hand, do require some memory at running time.
Even empty classes, however, have **non-zero** size. 
```c++
class S {};  // sizeof(S) == 1
```

#### 21.1.1 Layout Principles

Zero-sized classes are avoided for various reasons, 
one if for the compatibility with arrays (pointer arithmetic). 
```c++
ZeroSized z[10];
std::cout << &z[2] - &z[0] << '\n';  // compute distance between pointers/addresses
```
Normally, the difference in the previous example is obtained by 
dividing the number of bytes between the two addresses by the size of the type to which it is pointing, 
but when that size is zero this is clearly not satisfactory.


However, even though there are **no** zero-size types in C++, 
the C++ standard does specify that when an empty class is used as a base class, 
**no** space needs to be allocated for it 
> provided that it does not cause it to be allocated to the same address 
> as another object or subobject of the same type. 


Let's look at some examples to clarify what this _empty base class optimization (EBCO)_ means in practice. 
Consider the following program:
```c++
class Empty
{
    using Int = int;  // type alias members don't make a class nonempty
};

class EmptyTwo : public Empty {};

class EmptyThree : public EmptyToo {};

void foo()
{
    std::cout << "sizeof(Empty): " << sizeof(Empty) << '\n';            // 1
    std::cout << "sizeof(EmptyTwo): " << sizeof(EmptyTwo) << '\n';      // 1
    std::cout << "sizeof(EmptyThree): " << sizeof(EmptyThree) << '\n';  // 1
}
```
If your compiler implements the EBCO, it will print the same size for every class,
but none of these classes has size zero. 
```
┌────────────┐ ┐       ┐          ┐
│            │ ├ Empty ├ EmptyTwo ├ EmptyThree
└────────────┘ ┘       ┘          ┘
```
This means that within class `EmptyTwo`, the class `Empty` is not given any space. 
Note also that an empty class with optimized empty bases (and no other bases) is also empty. 
This explains why class `EmptyThree` can also have the same size as class `Empty`. 


If your compiler does not implement the EBCO, it will print different sizes.
```
┌────────────┐ ┐       ┐          ┐ 
│            │ ├ Empty │          │
├────────────┤ ┘       ├ EmptyTwo │
│            │         │          ├ EmptyThree
├────────────┤         ┘          │ 
│            │                    │
└────────────┘                    ┘
```
Consider an example that runs into a constraint of the EBCO:
```c++
class Empty
{
    using Int = int;  // type alias members don't make a class nonempty
};

class EmptyTwo : public Empty {};

class NonEmpty : public Empty, public EmptyTwo {};

void foo()
{
    std::cout << "sizeof(Empty): " << sizeof(Empty) << '\n';        // 1
    std::cout << "sizeof(EmptyTwo): " << sizeof(EmptyTwo) << '\n';  // 1
    std::cout << "sizeof(NonEmpty): " << sizeof(NonEmpty) << '\n';  // 2
}
```
It may come as a surprise that class `NonEmpty` is **not** an empty class. 
After all, it does not have any members and neither do its base classes. 
However, the base classes `Empty` and `EmptyTwo` of `NonEmpty` can **not** be allocated to the same address 
because this would cause the base class `Empty` of `EmptyTwo` to end up at the same address 
as the base class `Empty` of class `NonEmpty`. 
In other words, two subobjects of the same type would end up at the same offset, 
and this is **not** permitted by the object layout rules of C++. 
It may be conceivable to decide that one of the `Empty` base subobjects is placed at offset “0 bytes” 
and the other at offset “1 byte,” 
but the complete `NonEmpty` object still can **not** have a size of `1` byte 
because in an array of two `NonEmpty` objects, an `Empty` subobject of the first element 
can **not** end up at the same address as an `Empty` subobject of the second element. 
```
┌────────────┐ ┐                  ┐
│            │ ├ Empty            │
├────────────┤ ┤       ┐          ├ NonEmpty
│            │ ├ Empty ├ EmptyTwo │
└────────────┘ ┘       ┘          ┘
```
The rationale for the constraint on the EBCO stems from the fact that
it is desirable to be able to compare whether two pointers point to the same object. 
Because pointers are nearly always internally represented as just addresses, 
we must ensure that two different addresses (i.e., pointer values) correspond to two different objects.


The constraint may not seem very significant. 
However, in practice, it is often encountered because 
many classes tend to inherit from a small set of empty classes
that define some common type aliases. 
When two subobjects of such classes are used in the same complete object, the optimization is inhibited.


Even with this constraint, the EBCO is an important optimization for template libraries 
because a number of techniques rely on the introduction of base classes 
simply for the purpose of introducing new type aliases 
or providing extra functionality without adding new data. 
Several such techniques will be described in this chapter. 

#### 21.1.2 Members as Base Classes

The EBCO has **no** equivalent for data members because (among other things) 
it would create some problems with the representation of pointers to members. 
As a result, it is sometimes desirable to implement as a (private) base class 
that would at first sight be thought of as a member variable. 
However, this is not without its challenges. 


The problem is most interesting in the context of templates 
because template parameters are often substituted with empty class types, 
but in general we can **not** rely on this rule. 
If nothing is known about a template type parameter, the EBCO cannot easily be exploited. 
Indeed, consider the following trivial example:
```c++
template <typename T1, typename T2>
class MyClass
{
private:
    T1 a;
    T2 b;
};
```
It is entirely possible that one or both template parameters are substituted by an empty class type. 
If this is the case, then the representation of `MyClass<T1, T2>` may be suboptimal
and may waste a word of memory for every instance of a `MyClass<T1, T2>`. 


This can be avoided by making the template arguments base classes instead:
```c++
template <typename T1, typename T2>
class MyClass : private T1, private T2 {};
```
However, this straightforward alternative has its own set of problems: 
- It **doesn’t** work when `T1` or `T2` is substituted with a nonclass type or with a union type; 
- It also **doesn’t** work when the two parameters are substituted with the same type 
  (although this can be addressed fairly easily by adding another layer of inheritance); 
- The class may be `final`, in which case attempts to inherit from it will cause an error. 


Even if we satisfactorily addressed these problems, a very serious problem persists: 
Adding a base class can fundamentally modify the interface of the given class. 
For our `MyClass` class, this may not seem significant because there are very few interface elements to affect, 
but as we see later in this chapter, inheriting from a template parameter can affect 
whether a member function is virtual. 
Clearly, this approach to exploiting the EBCO is fraught with all kinds of trouble. 


A more practical tool can be devised for the common case when a template parameter 
is known to be substituted by class types only and when another member of the class template is available. 
The main idea is to “merge” the potentially empty type parameter with the other member using the EBCO. 
For example, instead of writing
```c++
template <typename CustomClass>
class Optimizable 
{
private:
    CustomClass info;  // might be empty
    void * storage;
};
```
a template implementer would use the following:
```c++
template <typename CustomClass>
class Optimizable 
{
private:
    BaseMemberPair<CustomClass, void *> infoAndStorage;
};
```
Even without seeing the implementation of the template `BaseMemberPair`,
it is clear that its use makes the implementation of `Optimizable` more verbose. 
However, various template library implementers have reported that the performance gains 
(for the clients of their libraries) do justify the added complexity.
We explore this idiom further in our discussion of tuple storage in Section 25.1.1. 


The implementation of `BaseMemberPair` can be fairly compact:
```c++
template <typename Base, typename Member>
class BaseMemberPair : private Base
{
public:
    // constructor
    BaseMemberPair(Base const & b, Member const & m) : Base(b), mem(m) {}

    // access base class data via first()
    Base const & base() const
    {
        return static_cast<Base const &>(*this);
    }

    Base & base()
    {
        return static_cast<Base &>(*this);
    }

    // access member data via second()
    Member const & member() const
    {
        return this->mem;
    }

    Member & member()
    {
        return this->mem;
    }

private:
    Member mem;
};
```
An implementation needs to use the member functions `base` and `member` 
to access the encapsulated (and possibly storage-optimized) data members. 


#### 📌 21.2 The Curiously Recurring Template Pattern (CRTP)


Another pattern is the _Curiously Recurring Template Pattern (CRTP)_. 
This oddly named pattern refers to a general class of techniques that consists of 
passing a derived class as a template argument to one of its own base classes. 
In its simplest form, C++ code for such a pattern looks as follows:
```c++
template <typename Derived>
class CuriousBase 
{
    // ...
};

class Curious : public CuriousBase<Curious> 
{
    // ...
};
```
Our first outline of CRTP shows a nondependent base class: 
The class `Curious` is not a template and is therefore immune to
some of the name visibility issues of dependent base classes. 
However, this is **not** an intrinsic characteristic of CRTP. 
Indeed, we could just as well have used the following alternative outline:
```c++
template <typename Derived>
class CuriousBase 
{
    // ...
};

template <typename T>
class Curious : public CuriousBase<Curious<T>> 
{
    // ...
};
```
By passing the derived class down to its base class via a template parameter, 
the base class can customize its own behavior to the derived class 
**without** requiring the use of virtual functions. 
This makes CRTP useful to factor out implementations that can only be member functions 
(e.g., constructor, destructors, and subscript operators) 
or are dependent on the derived class’s identity.


A simple application of CRTP consists of keeping track of 
how many objects of a certain class type were created. 
This is easily achieved by 
incrementing an integral static data member in every constructor 
and decrementing it in the destructor.
However, having to provide such code in every class is tedious, 
and implementing this functionality via a single (non-CRTP) base class 
would confuse the object counts for different derived classes. 
Instead, we can write the following template:
```c++
template <typename CountedType>
class ObjectCounter
{
public:
    // return number of existing objects:
    static std::size_t live()
    {
        return count;
    }
    
protected:
    // default constructor
    ObjectCounter()
    {
        ++count;
    }

    // copy constructor
    ObjectCounter(ObjectCounter<CountedType> const &)
    {
        ++count;
    }

    // move constructor
    ObjectCounter(ObjectCounter<CountedType> &&)
    {
        ++count;
    }

    // destructor
    ~ObjectCounter()
    {
        --count;
    }

private:
    inline static std::size_t count = 0;  // number of existing objects
};
```
Note that we use `inline` to be able to define and initialize the count member inside the class structure. 
Before C++17, we had to define it outside the class template:
```c++
template <typename CountedType>
class ObjectCounter
{
private:
    static std::size_t count; // number of existing objects
    // ...
};

// initialize counter with zero:
template <typename CountedType>
std::size_t ObjectCounter<CountedType>::count = 0;
```
If we want to count the number of live (i.e., not yet destroyed) objects for a certain class type, 
it suffices to derive the class from the `ObjectCounter` template. 
For example, we can define and use a counted string class along the following lines:
```c++
template <typename CharT>
class MyString : public ObjectCounter<MyString<CharT>>
{
    // ...
};

void foo()
{
    MyString<char> s1, s2;
    MyString<wchar_t> ws;
    std::cout << "num of MyString<char>: " << MyString<char>::live() << '\n';
    std::cout << "num of MyString<wchar_t>: " << ws.live() << '\n';
}
```

#### 21.2.1 The Barton-Nackman Trick

In 1994, John J. Barton and Lee R. Nackman presented a template technique that
they called _restricted template expansion_. 
The technique was motivated in part by the fact that at the time, 
function template overloading was severely limited and namespaces were not available in most compilers. 


To illustrate this, suppose we have a class template `Array` for which we want to
define the equality operator `operator==`. 
One possibility is to declare the operator as a member of the class template, 
but this is **not** good practice because the first argument (binding to the `this` pointer) 
is subject to conversion rules that differ from the second argument. 
Because `operator==` is meant to be symmetrical with respect to its arguments, 
it is preferable to declare it as a namespace scope function. 
An outline of a natural approach to its implementation may look like the following:
```c++
template <typename T>
class Array
{
public:
    // ...
};

template <typename T>
bool operator==(Array<T> const & a, Array<T> const & b)
{
    // ...
}
```
However, if function templates can **not** be overloaded, this presents a problem: 
No other `operator==` template can be declared in that scope, 
and yet it is likely that such a template would be needed for other class templates. 
Barton and Nackman resolved this problem by defining the operator in the class as a normal friend function:
```c++
template <typename T>
class Array
{
public:
    friend bool operator==(Array<T> const & a, Array<T> const & b)
    {
        return areEqual(a, b);
    }

private:
    static bool areEqual(Array<T> const & a, Array<T> const & b);
};
```
Suppose this version of `Array` is instantiated for type `float`. 
The friend operator function is then declared as a result of that instantiation, 
but note that this function itself is **not** an instantiation of a function template.
It is a normal nontemplate function that gets injected in the global scope 
as a side effect of the instantiation process (i.e., a templified entity). 
Because it is a nontemplate function, it could be overloaded with other declarations of `operator==` 
even before overloading of function templates was added to the language. 
Barton and Nackman called this _restricted template expansion_
because it avoided the use of a template `operator==(T, T)` 
that applied to all types `T` (in other words, unrestricted expansion). 


Because `operator==(Array<T> const &, Array<T> const &)` is defined inside a class definition, 
it is implicitly considered to be an `inline` function,
and we therefore decided to delegate the implementation to a static member function `areEqual`, 
which doesn’t need to be `inline`.


Name lookup for friend function definitions has changed since 1994, 
so the Barton-Nackman trick is not as useful in standard C++. 
At the time of its invention, friend declarations would be visible 
in the enclosing scope of a class template when that template was instantiated
via a process called _friend name injection_. 
Standard C++ instead finds friend function declarations via argument-dependent lookup 
(ADL; see Section 13.2.2 for the specific details). 
This means that at least one of the arguments of the function call 
must already have the class containing the friend function as an associated class. 
The friend function would **not** be found if the arguments were of an unrelated class type 
that could be converted to the class containing the friend. 
For example:
```c++
class S {};

template <typename T>
class Wrapper
{
public:
    friend void foo(Wrapper<T> const &) {}
    
    // implicit conversion from T to Wrapper<T>
    Wrapper(T obj) : object(obj) {}
    
private:
    T object;
};

void bar()
{
    S s;
    Wrapper<S> w(s);
    foo(w);              // OK: Wrapper<S> is a class associated with w
    foo(s);              // ERROR: Wrapper<S> is not associated with s
}
```
Here, the call `foo(w)` is valid because the function `foo` is a friend declared in
`Wrapper<S>` which is a class associated with the argument `w`.
However, in the call `foo(s)`, the friend declaration of function `foo(Wrapper<S> const &)`
is **not** visible because the class `Wrapper<S>` in which it is defined 
is **not** associated with the argument `s` of type `S`. 
Hence, even though there is a valid implicit conversion from type `S` to type `Wrapper<S>`
(through the constructor of `Wrapper<S>`), 
this conversion is **never** considered because the candidate function `foo` is **not** found in the first place. 
At the time Barton and Nackman invented their trick, 
friend name injection would have made the friend `foo` visible and the call `foo(s)` would succeed. 


In modern C++, the only advantages to defining a friend function in a class template 
over simply defining an ordinary function template are syntactic: 
friend function definitions have access to the private and protected members of their enclosing class 
and don’t need to restate all of the template parameters of enclosing class templates. 
However, friend function definitions can be useful when combined with the Curiously Recurring Template Pattern (CRTP), 
as illustrated in the operator implementations described in the following section. 

#### 21.2.2 Operator Implementations

When implementing a class that provides overloaded operators, 
it is common to provide overloads for a number of different (but related) operators. 
For example, a class that implements the equality operator `operator==` will likely 
also implement the inequality operator `operator!=`, 
and a class that implements the less-than operator `operator<` 
will likely implement the other relational operators as well (`>`, `<=`, `>=`). 
In many cases, the definition of only one of these operators is actually interesting, 
while the others can simply be defined in terms of that one operator. 
For example, the inequality operator for a class `X` is likely to be defined in terms of the equality operator:
```c++
bool operator!=(X const & x1, X const & x2)
{
    return !(x1 == x2);
}
```
Given the large number of types with similar definitions of `!=`,
it is tempting to generalize this into a template:
```c++
template <typename T>
bool operator!=(T const & x1, T const & x2)
{
    return !(x1 == x2);
}
```
In fact, the C++ standard library contains similar such definitions as part of the `<utility>` header.
However, these definitions (for `!=`, `>`, `<=`, and `>=`)
were relegated to the namespace `std::rel_ops` during standardization, 
when it was determined that they caused problems when made available in namespace `std`. 
Indeed, having these definitions visible makes it appear that
any type has an `!=` operator (which may fail to instantiate), 
and that operator will always be an exact match for both of its arguments. 
While the first problem can be overcome through the use of SFINAE techniques (see Section 19.4), 
such that this `!=` definition will only be instantiated for types with a suitable `operator==`, 
the second problem remains: 
The general `!=` definition above will be preferred over user-provided definitions that require, 
for example, a derived-to-base conversion, which may come as a surprise.


An alternative formulation of these operator templates based on CRTP 
allows classes to opt in to the general operator definitions, 
providing the benefits of increased code reuse **without** the side effects of an overly general operator:
```c++
template <typename Derived>
class EqualityComparable
{
public:
    friend bool operator!=(Derived const & x1, Derived const & x2)
    {
        return !(x1 == x2);
    }
};

class X : public EqualityComparable<X>
{
public:
    friend bool operator==(X const & x1, X const & x2)
    {
        // implement logic for comparing two objects of type X
    }
};

void foo()
{
    X x1, x2;
    
    if (x1 != x2) {}
}
```
Here, we have combined CRTP with the Barton-Nackman trick.
`EqualityComparable` uses CRTP to provide an `operator!=` for its derived class 
based on the derived class’s definition of `operator==`. 
It actually provides that definition via a friend function definition (the Barton-Nackman trick),
which gives the two parameters to `operator!=` equal behavior for conversions.


CRTP can be useful when factoring behavior into a base class
while retaining the identity of the eventual derived class. 
Along with the Barton-Nackman trick, CRTP can provide general definitions 
for a number of operators based on a few canonical operators. 
These properties have made CRTP with the Barton-Nackman trick a favorite technique 
for authors of C++ template libraries.

#### 21.2.3 Facades

The use of CRTP and the Barton-Nackman trick to define some operators is a convenient shortcut. 
We can take this idea further, such that the CRTP base class 
defines most or all of the public interface of a class 
in terms of a much smaller (but easier to implement) interface exposed by the CRTP derived class. 
This pattern, called the _facade_ pattern, is particularly useful when defining new types 
that need to meet the requirements of some existing interface, 
e.g., numeric types, iterators, containers, and so on. 


To illustrate the facade pattern, we will implement a facade for iterators, 
which drastically simplifies the process of writing an iterator 
that conforms to the requirements of the standard library. 
The required interface for an iterator type (particularly a random access iterator) is quite large. 
The following skeleton for class template `IteratorFacade` demonstrates the requirements for an iterator interface:
```c++
template <typename Derived, typename Value, typename Category,
          typename Reference = Value &, typename Distance = std::ptrdiff_t>
class IteratorFacade
{
public:
    using value_type = typename std::remove_const<Value>::type;
    using reference = Reference;
    using pointer = Value *;
    using difference_type = Distance;
    using iterator_category = Category;

    // input iterator interface:
    reference operator*() const
    {
        return asDerived().dereference();
    }
    
    pointer operator->() const { /* ... */ }
    
    Derived & operator++()
    {
        asDerived().increment();
        return asDerived();
    }
    
    Derived operator++(int)
    {
        Derived result(asDerived());
        asDerived().increment();
        return result;
    }
    
    friend bool operator==(IteratorFacade const & lhs, IteratorFacade const & rhs)
    {
        return lhs.asDerived().equals(rhs.asDerived());
    }
        
    /* ... */
    
    // bidirectional iterator interface:
    Derived & operator--() { /* ... */ }

    Derived operator--(int) { /* ... */ }

    // random access iterator interface:
    reference operator[](difference_type n) const { /* ... */ }

    Derived & operator+=(difference_type n) { /* ... */ }

    /* ... */
    
    friend difference_type operator-(IteratorFacade const & lhs, IteratorFacade const & rhs) { /* ... */ }

    friend bool operator<(IteratorFacade const & lhs, IteratorFacade const & rhs) { /* ... */ }
    
    /* ... */
    
private:
    Derived & asDerived()
    {
        return *static_cast<Derived *>(this);
    }
    
    Derived const & asDerived() const
    {
        return *static_cast<Derived const *>(this);
    }
};
```

#### Defining a Linked-List Iterator

```c++
template <typename T>
class ListNode
{
public:
    ~ListNode()
    {
        delete next;
    }
    
    T value;
    ListNode<T> * next = nullptr;
};

template <typename T>
class ListNodeIterator : public IteratorFacade<ListNodeIterator<T>, T, std::forward_iterator_tag>
{
public:
    ListNodeIterator(ListNode<T> * current = nullptr) : current(current) {}
    
    T & dereference() const
    {
        return current->value;
    }

    void increment()
    {
        current = current->next;
    }

    bool equals(ListNodeIterator const & other) const
    {
        return current == other.current;
    }
    
private:
    ListNode<T> * current = nullptr;
};
```

#### Hiding the interface

One downside to our implementation of `ListNodeIterator` is that 
we were required to expose as a public interface the operations `dereference`, `advance`, and `equals`. 
To eliminate this as-public requirement, we can rework `IteratorFacade` 
to perform all of its operations on the derived CRTP class through a separate _access_ class, 
which we call `IteratorFacadeAccess`:
```c++
// ‘friend’ this class to allow IteratorFacade access to core iterator operations: 
class IteratorFacadeAccess
{
    // only IteratorFacade can use these definitions
    template <typename Derived, typename Value, typename Category, typename Reference, typename Distance>
    friend class IteratorFacade;

    // required of all iterators:
    template <typename Reference, typename Iterator>
    static Reference dereference(Iterator const & i)
    {
        return i.dereference();
    }

    // ... 
    
    // required of bidirectional iterators:
    template <typename Iterator>
    static void decrement(Iterator & i)
    {
        return i.decrement();
    }

    // required of random-access iterators:
    template <typename Iterator, typename Distance>
    static void advance(Iterator & i, Distance n)
    {
        return i.advance(n);
    }
    
    // ... 
};

template <typename T>
class ListNodeIterator : public IteratorFacade<ListNodeIterator<T>, T, std::forward_iterator_tag>
{
public:
    friend class IteratorFacadeAccess;
    
    ListNodeIterator(ListNode<T> * current = nullptr) : current(current) {}

private:
    T & dereference() const
    {
        return current->value;
    }
    
    void increment()
    {
        current = current->next;
    }
    
    bool equals(ListNodeIterator const & other) const
    {
        return current == other.current;
    }
    
    ListNode<T> * current = nullptr;
};
```
This class provides static member functions for each of the core iterator operations,
calling the corresponding (nonstatic) member function of the provided iterator. 
All of the static member functions are _private_, 
with the access only granted to `IteratorFacade` itself. 
Therefore, our `ListNodeIterator` can make `IteratorFacadeAccess` a friend 
and keep private the interface needed by the facade.

#### Iterator Adapters

Our `IteratorFacade` makes it easy to build an iterator _adapter_ 
that takes an existing iterator and exposes a new iterator 
that provides some transformed view of the underlying sequence. 
```c++
struct Person
{
    friend std::ostream & operator<<(std::ostream & strm, Person const & p)
    {
        return strm << p.lastName << ", " << p.firstName;
    }

    std::string firstName;
    std::string lastName;
};

template <typename Iterator, typename T>
class ProjectionIterator
        : public IteratorFacade<
                ProjectionIterator<Iterator, T>,                             // derived
                T,                                                           // value_type
                typename std::iterator_traits<Iterator>::iterator_category,  // iterator_category
                T &,                                                         // reference_type
                typename std::iterator_traits<Iterator>::difference_type>    // difference_type
{
private:
    using Base = typename std::iterator_traits<Iterator>::value_type;
    using Distance = typename std::iterator_traits<Iterator>::difference_type;
    
public:
    friend class IteratorFacadeAccess;
    
    // implement core iterator operations for IteratorFacade
    ProjectionIterator(Iterator iter, T Base::* member) : iter(iter), member(member) {}
    
private:
    T & dereference() const
    {
        return (*iter).*member;
    }
    
    void increment()
    {
        ++iter;
    }
    
    bool equals(ProjectionIterator const & other) const
    {
        return iter == other.iter;
    }
    
    void decrement()
    {
        --iter;
    }
    
    Iterator iter;
    T Base::* member;
};

template <typename Iterator, typename Base, typename T>
auto project(Iterator iter, T Base::* member)
{
    return ProjectionIterator<Iterator, T>(iter, member);
}

std::vector<Person> authors = { {"David", "Vandevoorde"},
                                {"Nicolai", "Josuttis"},
                                {"Douglas", "Gregor"} };
std::copy(project(authors.begin(), &Person::firstName),
          project(authors.end(), &Person::firstName),
          std::ostream_iterator<std::string>(std::cout, "\n"));
```
Each projection iterator stores two values: 
`iter`, which is the iterator into the  underlying sequence (of `Base` values), 
and `member`, a pointer-to-data member describing which member to project through. 
With that in mind, we consider the template arguments provided to the `IteratorFacade` base class. 
The first is the `ProjectionIterator` itself (to enable CRTP). 
The second (`T`) and fourth (`T &`) arguments are the value and reference types of our projection iterator, 
defining this as a sequence of `T` values.
The third and fifth arguments merely pass through the category and difference types of the underlying iterator. 
Therefore, our projection iterator will be an input iterator when `Iterator` is an input iterator, 
a bidirectional iterator when `Iterator` is a bidirectional iterator, and so on. 
A `project` function makes it easy to build projection iterators.

#### 📌 21.3 Mixins

```c++
class Point
{
public:
    Point() : x(0.0), y(0.0) {}
    Point(double x, double y) : x(x), y(y) {}

    double x, y;
};

class Polygon
{
public:
    // public operations here

private:
    std::vector<Point> points;
};
```
This `Polygon` class would be more useful if the user could extend 
the set of information associated with each `Point` to include application-specific data
such as the color of each point, or perhaps associate a label with each point. 
One option to make this extension possible would be to parameterize `Polygon` 
based on the type of the point.
Users could conceivably create their own `Point`-like data type 
that provided the same interface as `Point` but includes other application-specific data, 
using inheritance:
```c++
template <typename P>
class Polygon
{
public:
    // public operations here

private:
    std::vector<P> points;
};

class LabeledPoint : public Point
{
public:
    LabeledPoint() : Point(), label("") { }
    
    LabeledPoint(double x, double y) : Point(x, y), label("") {}
    
    std::string label;
};
```
This implementation has its shortcomings. 
For one, it requires that the type `Point` be exposed to the user so that the user can derive from it. 
Also, the author of `LabeledPoint` needs to be careful to provide exactly the same interface as `Point` 
(e.g., inheriting or providing all of the same constructors as `Point`), 
or `LabeledPoint` will not work with `Polygon`. 
This constraint becomes more problematic if `Point` changes from one version of the `Polygon` template to another: 
The addition of a new `Point` constructor could require each derived class to be updated.


_Mixins_ provide an alternative way to customize the behavior of a type without inheriting from it. 
Mixins essentially invert the normal direction of inheritance, 
because the new classes are “mixed in” to the inheritance hierarchy as base classes of a class template
rather than being created as a new derived class. 
This approach allows the introduction of new data members and other operations 
without requiring any duplication of the interface.


A class template that supports mixins will typically accept 
an arbitrary number of extra classes from which it will derive.
Now, we can “mix in” base classes:
```c++
template <typename ... Mixins>
class Point : public Mixins ...
{
public:
    Point() : Mixins()..., x(0.0), y(0.0) {}

    Point(double x, double y) : Mixins()..., x(x), y(y) {}

    double x, y;
};

class Label
{
public:
    Label() : label("") {}
    
    std::string label;
};

using LabeledPoint = Point<Label>;

class Color
{
public:
    unsigned char red = 0, green = 0, blue = 0;
};

using MyPoint = Point<Label, Color>;
```
With this mixin-based `Point`, 
it becomes easy to introduce additional information to `Point` without changing its interface, 
so `Polygon` becomes easier to use and evolve. 
Users need only apply the implicit conversion from the `Point` specialization to their mixin class 
(`Label` or `Color`, above) to access that data or interface. 
Moreover, the `Point` class can even be entirely hidden, 
with the mixins provided to the `Polygon` class template itself:
```c++
template <typename ... Mixins>
class Polygon
{
public:
    // public operations here

private:
    std::vector<Point<Mixins...>> points;
};
```
Mixins are useful in cases where a template needs some small level of customization, 
such as decorating internally stored objects with user-specified data, 
without requiring the library to expose and document those internal data types and their interfaces.

#### 21.3.1 Curious Mixins

Mixins can be made more powerful by combining them with the Curiously Recurring Template Pattern (CRTP). 
Here, each of the mixins is actually a class template that will be provided with the type of the derived class, 
allowing additional customization to that derived class. 
A CRTP-mixin version of `Point` would be written as follows:
```c++
template <template <typename> ... Mixins>
class Point : public Mixins<Point> ...
{
public:
    Point() : Mixins<Point>()..., x(0.0), y(0.0) {}

    Point(double x, double y) : Mixins<Point>()..., x(x), y(y) {}

    double x, y;
};
```
This formulation requires some more work for each class that will be mixed in, 
so classes such as `Label` and `Color` will need to become class templates. 
However, the mixed-in classes can now tailor their behavior 
to the specific instance of the derived class they’ve been mixed into. 
For example, we can mix the previously-discussed `ObjectCounter` template into `Point` 
to count the number of points created by `Polygon` 
and compose that mixin with other application-specific mixins as well.

#### 21.3.2 Parameterized Virtuality

Mixins also allow us to indirectly parameterize other attributes of the derived class,
such as the virtuality of a member function.
```c++
class NotVirtual {};

class Virtual
{
public:
    virtual void foo() {}
};

template <typename ... Mixins>
class Base : public Mixins ...
{
public:
    // the virtuality of foo() depends on its declaration
    // (if any) in the base classes Mixins...
    void foo()
    {
        std::cout << "Base::foo()" << '\n';
    }
};

template <typename ... Mixins>
class Derived : public Base<Mixins...>
{
public:
    void foo()
    {
        std::cout << "Derived::foo()" << '\n';
    }
};

void bar()
{
    std::unique_ptr<Base<NotVirtual>> p1 = std::make_unique<Derived<NotVirtual>>();
    p1->foo();  // calls Base::foo()
    
    std::unique_ptr<Base<Virtual>> p2 = std::make_unique<Derived<Virtual>>();
    p2->foo();  // calls Derived::foo()
}
```
This technique can provide a tool to design a class template 
that is usable both to instantiate concrete classes and to extend using inheritance. 
However, it is rarely sufficient just to sprinkle virtuality on some member functions 
to obtain a class that makes a good base class for more specialized functionality. 
This sort of development method requires more fundamental design decisions. 
It is therefore usually more practical to design two different tools (class or class template hierarchies)
than to try to integrate them all into one template hierarchy.

#### 📌 21.4 Named Template Arguments

Various template techniques sometimes cause a class template to
end up with many different template type parameters. 
However, many of these parameters often have reasonable default values. 
A natural way to define such a class template may look as follows:
```c++
template <typename Policy1 = DefaultPolicy1,
          typename Policy2 = DefaultPolicy2,
          typename Policy3 = DefaultPolicy3,
          typename Policy4 = DefaultPolicy4>
class BreadSlicer {};
```
Presumably, such a template can often be used with the default template argument values using the syntax `BreadSlicer<>`. 
However, if a nondefault argument must be specified, 
all preceding arguments must be specified too (even though they may have the default value). 


Clearly, it would be attractive to be able to use a construct akin to `BreadSlicer<Policy3 = Custom>` rather than
`BreadSlicer<DefaultPolicy1, DefaultPolicy2, Custom>`, 
as is the case right now. In what follows we develop a technique to enable almost exactly that. 


Our technique consists of placing the default type values in a base class 
and overriding some of them through derivation. 
Instead of directly specifying the type arguments, we provide them through helper classes. 
For example, we could write `BreadSlicer<Policy3_is<Custom>>`. 
Because each template argument can describe any of the policies, the defaults cannot be different. 
In other words, at a high level, every template parameter is equivalent:
```c++
// name default policies as P1, P2, P3, P4
class DefaultPolicies
{
public:
    using P1 = DefaultPolicy1;
    using P2 = DefaultPolicy2;
    using P3 = DefaultPolicy3;
    using P4 = DefaultPolicy4;
};

// class to define a use of the default policy values
// avoids ambiguities if we derive from DefaultPolicies more than once
class DefaultPolicyArgs : virtual public DefaultPolicies
{};

template <typename Policy>
class Policy1_is : virtual public DefaultPolicies
{
public:
    using P1 = Policy;  // overriding type alias
};

template <typename Policy>
class Policy2_is : virtual public DefaultPolicies
{
public:
    using P2 = Policy;  // overriding type alias
};

template <typename Policy>
class Policy3_is : virtual public DefaultPolicies
{
public:
    using P3 = Policy;  // overriding type alias
};

template <typename Policy>
class Policy4_is : virtual public DefaultPolicies
{
public:
    using P4 = Policy;  // overriding type alias
};

// PolicySelector<A, B, C, D> creates A, B, C, D as base classes
// Discriminator<> allows having even the same base class more than once
template <typename Base, int D>
class Discriminator : public Base {};

template <typename Setter1, typename Setter2,
          typename Setter3, typename Setter4>
class PolicySelector : public Discriminator<Setter1, 1>,
                       public Discriminator<Setter2, 2>,
                       public Discriminator<Setter3, 3>,
                       public Discriminator<Setter4, 4>
{};

template <typename PolicySetter1 = DefaultPolicyArgs,
          typename PolicySetter2 = DefaultPolicyArgs,
          typename PolicySetter3 = DefaultPolicyArgs,
          typename PolicySetter4 = DefaultPolicyArgs>
class BreadSlicer
{
    using Policies = PolicySelector<PolicySetter1,
                                    PolicySetter2,
                                    PolicySetter3,
                                    PolicySetter4>;
    // use Policies::P1, Policies::P2, ... to refer to the various policies
};

// Defined as PolicySelector<Policy3_is<CustomPolicy>,
//                           DefaultPolicyArgs,
//                           DefaultPolicyArgs,
//                           DefaultPolicyArgs>
BreadSlicer<Policy3_is<CustomPolicy>> bc;
```






### 🎯 Chapter 22 Bridging Static and Dynamic Polymorphism

```c++
// bloats source code. 
template <typename Operator>
void forEachInt(int * b, int * e, Operator f) {}

// does not work with lambdas. 
void forEachInt(int * b, int * e, void (*f)(int)) {}

// works with lambdas and does not bloat source code, but slow. 
void forEachInt(int * b, int * e, std::function<void(int)> f) {}
```
The template argument to `std::function` is a function type 
that describes the parameter types the function object will receive 
and the return type that it should produce, 
much like a function pointer describes the parameter and result types.


This formulation of `forEachInt` provides some aspects of static polymorphism, 
i.e., the ability to work with an unbounded set of types including function pointers,
lambdas, and arbitrary classes with a suitable `operator()`, 
while itself remaining a nontemplate function with a single implementation.
It does so using a technique called _type erasure_, 
which bridges the gap between static and dynamic polymorphism.

The `std::function` type is effectively a generalized form of a C++ function pointer, 
providing the same fundamental operations:
- It can be used to invoke a function without the caller knowing anything about the function itself;
- It can be copied, moved, and assigned;
- It can be initialized or assigned from another function (with a compatible signature);
- It has a “null” state that indicates when no function is bound to it.


However, **unlike** a C++ function pointer, a `std::function` 
can also store a lambda or any other function object with a suitable `operator()`, 
all of which may have different types.


In the remainder of this chapter, we will build our own generalized function pointer class template, `FunctionPtr`, 
to provide these same core operations and capabilities and that can be used in place of `std::function`. 
```c++
template<typename R, typename ... Args>
class FunctorBridge
{
public:
    virtual ~FunctorBridge() = default;
    virtual FunctorBridge* clone() const = 0;
    virtual R invoke(Args... args) const = 0;
    virtual bool equals(FunctorBridge const * fb) const = 0;
};

template <typename Functor, typename R, typename ... Args>
class SpecificFunctorBridge : public FunctorBridge<R, Args...>
{
public:
    template <typename FunctorFwd>
    explicit SpecificFunctorBridge(FunctorFwd && functor) : functor(std::forward<FunctorFwd>(functor)) {}

    SpecificFunctorBridge * clone() const override
    {
        return new SpecificFunctorBridge(functor);
    }

    R invoke(Args ... args) const override
    {
        return functor(std::forward<Args>(args)...);
    }

    bool equals(FunctorBridge<R, Args...> const * fb) const override
    {
        if (auto specFb = dynamic_cast<SpecificFunctorBridge const*>(fb))
        {
            return functor == specFb->functor;
        }

        // functors with different types are never equal.
        return false;
    }

private:
    Functor functor;
};

// primary template:
template <typename Signature>
class FunctionPtr;

// partial specialization:
template <typename R, typename ... Args>
class FunctionPtr<R (Args ...)>
{
public:
    friend bool operator==(FunctionPtr const & f1, FunctionPtr const & f2)
    {
        if (!f1 || !f2) 
        {
            return !f1 && !f2;
        }

        return f1.bridge->equals(f2.bridge);
    }

    friend bool operator!=(FunctionPtr const & f1, FunctionPtr const & f2)
    {
        return !(f1 == f2);
    }

    // constructors:
    FunctionPtr() : bridge(nullptr) {}

    FunctionPtr(FunctionPtr const & other) : bridge(nullptr)
    {
        if (other.bridge)
        {
            bridge = other.bridge->clone();
        }
    }

    FunctionPtr(FunctionPtr & other) : FunctionPtr(static_cast<FunctionPtr const &>(other)) {}

    FunctionPtr(FunctionPtr && other) noexcept : bridge(other.bridge)
    {
        other.bridge = nullptr;
    }

    // construction from arbitrary function objects:
    template <typename F>
    explicit FunctionPtr(F && f) : bridge(nullptr)
    {
        using Functor = std::decay_t<F>;
        using Bridge = SpecificFunctorBridge<Functor, R, Args...>;
        bridge = new Bridge(std::forward<F>(f));
    }

    // assignment operators:
    FunctionPtr & operator=(FunctionPtr const & other)
    {
        if (&other != this)
        {
            FunctionPtr tmp(other);
            swap(*this, tmp);
        }

        return *this;
    }

    FunctionPtr & operator=(FunctionPtr && other) noexcept
    {
        if (&other != this)
        {
            delete bridge;
            bridge = other.bridge;
            other.bridge = nullptr;
        }

        return *this;
    }

    // construction and assignment from arbitrary function objects:
    template <typename F>
    FunctionPtr & operator=(F && f)
    {
        FunctionPtr tmp(std::forward<F>(f));
        swap(*this, tmp);
        return *this;
    }

    // destructor:
    ~FunctionPtr()
    {
        delete bridge;
    }

    friend void swap(FunctionPtr & fp1, FunctionPtr & fp2)
    {
        std::swap(fp1.bridge, fp2.bridge);
    }

    explicit operator bool() const
    {
        return bridge == nullptr;
    }

    // invocation:
    R operator()(Args ... args) const
    {
        return bridge->invoke(std::forward<Args>(args)...);
    }

private:
    FunctorBridge<R, Args...> * bridge;
};
```
**Performance Considerations**. 
Type erasure provides some of the advantages of both static polymorphism and dynamic polymorphism, but not all. 
In particular, the performance of generated code using type erasure hews more closely to that of dynamic polymorphism, 
because both use dynamic dispatch via virtual functions. 
Thus, some of the traditional advantages of static polymorphism, 
such as the ability of the compiler to inline calls, may be lost. 
Whether this loss of performance will be perceptible is application-dependent, 
but it’s often easy to tell by considering how much work is performed 
in the function being called relative to the cost of a virtual function call: 
If the two are close (e.g., using `FunctionPtr` to simply add two integers), 
type erasure is likely to execute far more slowly than a static-polymorphic version. 
If, on the other hand, the function call performs a significant amount of work, 
such as querying a database, sorting a container, or updating a user interface, 
the overhead of type erasure is unlikely to be measurable. 






### 🎯 Chapter 23 Metaprogramming


_Metaprogramming_ consists of “programming a program.” 
In other words, we lay out code that the programming system executes 
to generate new code that implements the functionality we really want. 
Usually, the term metaprogramming implies a reflexive attribute: 
The metaprogramming component is part of the program for which it generates a bit of code 
(i.e., an additional or different bit of the program). 


Why would metaprogramming be desirable? 
As with most other programming techniques, the goal is to achieve more functionality with less effort, 
where effort can be measured as code size, maintenance cost, and so forth. 
What characterizes metaprogramming is that some user-defined computation happens at translation time. 
The underlying motivation is often performance 
(things computed at translation time can frequently be optimized away) 
or interface simplicity (a metaprogram is generally shorter than what it expands to) or both. 

#### 📌 23.1 The State of Modern C++ Metaprogramming

#### 23.1.1 Value Metaprogramming

Since C++14, a compile-time function to compute a square root is easily written as follows: 
```c++
template <typename T>
constexpr T sqrt(T x)
{
    // handle cases where x and its square root are equal 
    // as a special case to simplify the iteration criterion for larger x:
    if (x <= 1)
    {
        return x;
    }
    
    // repeatedly determine in which half of a [lo, hi] interval 
    // the square root of x is located,
    // until the interval is reduced to just one value:
    T lo = 0, hi = x;
    
    while (true)
    {
        auto mid = (hi + lo) / 2, mid2 = mid * mid;
        
        if (hi <= lo + 1 || mid2 == x)
        {
            // mid must be the square root:
            return mid;
        }
        
        //continue with the higher/lower half-interval:
        if (mid2 < x)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
}
```
This algorithm searches for the answer by repeatedly halving an interval known to contain the square root of `x` 
(the roots of 0 and 1 are treated as special cases to keep the convergence criterion simple). 
This `sqrt` function can be evaluated at compile or run time:
```c++
static_assert(sqrt(25) == 5, "");   // OK (evaluated at compile time)
static_assert(sqrt(40) == 6, "");   // OK (evaluated at compile time)
std::array<int, sqrt(40) + 1> arr;  // declares std::array<int, 7> (compile time)
long long l = 53478LL;
std::cout << sqrt(l) << '\n';       // prints 231 (evaluated at run time)
```
This function’s implementation may not be the most efficient at run time 
(where exploiting peculiarities of the machine often pays off), 
but because it is meant to perform compile-time computations, 
absolute efficiency is less important than portability. 
Note that no advanced “template magic” is in sight in that square root example, 
only the usual template argument deduction for a function template. 
The code is “plain C++” and is not particularly challenging to read.


Value metaprogramming (i.e., programming the computation of compile-time values) 
as we did above is occasionally quite useful, 
but there are two additional kinds of metaprogramming that can be performed with modern C++: 
type metaprogramming and hybrid metaprogramming.

#### 23.1.2 Type Metaprogramming

```c++
template <typename _Tp>
  struct remove_all_extents
  { typedef _Tp     type; };

template <typename _Tp, std::size_t _Size>
  struct remove_all_extents<_Tp [_Size]>
  { typedef typename remove_all_extents<_Tp>::type     type; };

template <typename _Tp>
  struct remove_all_extents<_Tp []>
  { typedef typename remove_all_extents<_Tp>::type     type; };

remove_all_extents<int []>       // int
remove_all_extents<int [5][10]>  // int
remove_all_extents<int [][10]>   // int
remove_all_extents<int (*) []>   // int (*) [5]  
```

#### 23.1.3 Hybrid Metaprogramming

```c++
template <typename T, std::size_t N>
struct DotProduct
{
    static inline T result(T * a, T * b)
    {
        return *a * *b + DotProduct<T, N - 1>::result(a + 1, b + 1);
    }
};

// partial specialization as end criteria
template <typename T>
struct DotProduct<T, 0>
{
    static inline T result(T *, T *)
    {
        return T {};
    }
};

template <typename T, std::size_t N>
auto dotProduct(std::array<T, N> const & x, std::array<T, N> const & y)
{
    return DotProduct<T, N>::result(x.begin(), y.begin());
}
```
Given that in `std::array` and `std::tuple` we have flexible counterparts to array types and (simple) struct types, 
it is natural to wonder whether a counterpart to simple union types would also be useful for hybrid computation. 
The answer is “yes.” 
The C++ standard library introduced a `std::variant` template for this purpose in C++17, 
and we develop a similar component in Chapter 26.


Because `std::tuple` and `std::variant`, like struct types, are heterogeneous types, 
hybrid metaprogramming that uses such types is sometimes called _heterogeneous metaprogramming_.

#### 23.1.4 Hybrid Metaprogramming for Unit Types

```c++
template <unsigned N, unsigned D = 1>
struct Ratio
{
    static constexpr unsigned num = N;  // numerator
    static constexpr unsigned den = D;  // denominator
    using Type = Ratio<num, den>;
};

// implementation of adding two ratios:
template <typename R1, typename R2>
struct RatioAddImpl
{
private:
    static constexpr unsigned den = R1::den * R2::den;
    static constexpr unsigned num = R1::num * R2::den + R2::num * R1::den;

public:
    typedef Ratio<num, den> Type;
};

// using declaration for convenient usage:
template <typename R1, typename R2>
using RatioAdd = typename RatioAddImpl<R1, R2>::Type;

using R1 = Ratio<1, 1000>;
using R2 = Ratio<2, 3>;
using RS = RatioAdd<R1, R2>;                     // RS has type Ratio<2003, 3000>
std::cout << RS::num << '/' << RS::den << '\n';  // prints 2003/3000
using RA = RatioAdd<Ratio<2, 3>, Ratio<5, 7>>;   // RA has type Ratio<29, 21>
std::cout << RA::num << '/' << RA::den << '\n';  // prints 29/21

// duration type for values of type T with unit type U:
template <typename T, typename U = Ratio<1>>
class Duration
{
public:
    using ValueType = T;
    using UnitType = typename U::Type;

public:
    constexpr Duration(ValueType v = 0) : val(v) {}

    constexpr ValueType value() const { return val; }

private:
    ValueType val;
};

// adding two durations where unit type might differ:
template <typename T1, typename U1, typename T2, typename U2>
auto constexpr operator+(Duration<T1, U1> const & lhs, Duration<T2, U2> const & rhs)
{
    // resulting type is a unit with 1 a nominator and
    // the resulting denominator of adding both unit type fractions
    using VT = Ratio<1, RatioAdd<U1, U2>::den>;
    
    // resulting value is the sum of both values
    // converted to the resulting unit type:
    auto val = lhs.value() * VT::den / U1::den * U1::num +
               rhs.value() * VT::den / U2::den * U2::num;
    
    return Duration<decltype(val), VT>(val);
}

int x = 42;
int y = 77;
auto a = Duration<int, Ratio<1, 1000>>(x);   // x milliseconds
auto b = Duration<int, Ratio<2, 3>>(y);      // y 2/3 seconds
auto c = a + b;                              // computes resulting unit type 1/3000 seconds
                                             // and generates run-time code for c = a*3 + b*2000
auto d = Duration<double, Ratio<1,3>>(7.5);  // 7.5 1/3 seconds
auto e = Duration<int, Ratio<1>>(4);         // 4 seconds
auto f = d + e;                              // computes resulting unit type 1/3 seconds
                                             // and generates code for f = d + e*3
```
In addition, the compiler can even perform the value computation at compile-time
if the values are known at compile time, because `operator+` for durations is `constexpr`.


The C++ standard library class template `std::chrono` uses this approach with several refinements, 
such as using predefined units (e.g., `std::chrono::milliseconds`),
supporting duration literals (e.g., `10ms`), and dealing with overflow. 


#### 📌 23.2 The Dimensions of Reflective Metaprogramming


Previously, we described value metaprogramming based on `constexpr` evaluation 
and type metaprogramming based on recursive template instantiation. 
These two options, both available in modern C++, 
clearly involve different methods driving the computation. 
It turns out that value metaprogramming can also be driven in terms of recursive template instantiation, 
and, prior to the introduction of `constexpr` functions in C++11, that was the mechanism of choice. 


#### 📌 23.3 The Cost of Recursive Instantiation


The following metaprogram results in significant instantiations because the compiler
instantiates both the positive branch and the negative branch of the tri operator used to compute `Sqrt::value`: 
```c++
// primary template to compute sqrt(N)
template <int N, int LO = 1, int HI = N>
struct Sqrt
{
    // compute the midpoint, rounded up
    static constexpr auto mid = (LO + HI + 1) / 2;
    
    // search a not too large value in a halved interval
    static constexpr auto value = (N < mid * mid) ?
                                  Sqrt<N, LO, mid - 1>::value : 
                                  Sqrt<N, mid, HI>::value;
};

// partial specialization for the case when LO equals HI
template <int N, int M>
struct Sqrt<N, M, M>
{
    static constexpr auto value = M;
};
```
The solution is to refrain from the instantiation of the negative branch: 
```c++
// primary template to compute sqrt(N)
template <int N, int LO = 1, int HI = N>
struct Sqrt
{
    // compute the midpoint, rounded up
    static constexpr auto mid = (LO + HI + 1) / 2;
    
    // search a not too large value in a halved interval
    static constexpr auto value = IfThenElse<N < mid * mid), 
                                             Sqrt<N, LO, mid - 1>::value, 
                                             Sqrt<N, mid, HI>::value>::value;
};

// partial specialization for the case when LO equals HI
template <int N, int M>
struct Sqrt<N, M, M>
{
    static constexpr auto value = M;
};
```
The key change here is the use of the `IfThenElse` template, which was introduced in Section 19.7.1. 
The `IfThenElse` template is a device that selects between two types based on a given Boolean constant. 
If the constant is true, the first type is type-aliased to `Type`; 
otherwise, `Type` stands for the second type. 
At this point it is important to remember that defining a type alias for a class template instance 
does not cause a C++ compiler to instantiate the body of that instance. 


#### 📌 23.4 Computational Completeness

Nothing significant. 


#### 📌 23.5 Recursive Instantiation versus Recursive Template Arguments

```c++
template <typename T, typename U>
struct Doublify {};

template <int N>
struct Trouble
{
    using LongType = Doublify<typename Trouble<N - 1>::LongType,
                              typename Trouble<N - 1>::LongType>;
};

template <>
struct Trouble<0>
{
    using LongType = double;
};

Trouble<10>::LongType ouch;
```
The use of `Trouble<10>::LongType` not only triggers the recursive instantiation of
`Trouble<9>`, `Trouble<8>`, ..., `Trouble<0>`, 
but it also instantiates `Doublify` over increasingly complex types.
The complexity of the type description of the expression `Trouble<N>::LongType` grows exponentially with `N`. 
In general, such a situation stresses a C++ compiler even more than do 
recursive instantiations that do not involve recursive template arguments. 
One of the problems here is that a compiler keeps a representation of the mangled name for the type. 
This mangled name encodes the exact template specialization in some way, 
and early C++ implementations used an encoding that is roughly proportional to the length of the template-id. 
These compilers then used well over 10,000 characters for `Trouble<10>::LongType`. 


Newer C++ implementations take into account the fact that nested template-ids are fairly common in modern C++ programs 
and use clever compression techniques to reduce considerably the growth in name encoding 
(e.g., a few hundred characters for `Trouble<10>::LongType`). 
These newer compilers also avoid generating a mangled name 
if none is actually needed because no low-level code is actually generated for the template instance. 
Still, all other things being equal, it is probably preferable to organize recursive instantiation in such a way 
that template arguments need not also be nested recursively. 


#### 📌 23.6 Enumeration Values versus Static Constants


In the early days of C++, enumeration values (the enum hack) were the only mechanism 
to create “true constants” (called constant-expressions) as named members in class declarations. 
With them, we could, for example, define a `Pow3` metaprogram to compute powers of 3 as follows:
```c++
// primary template to compute 3 to the Nth
template <int N>
struct Pow3
{
    enum
    {
        value = 3 * Pow3<N - 1>::value
    };
};

// full specialization to end the recursion
template <>
struct Pow3<0>
{
    enum
    {
        value = 1
    };
};
```
The standardization of C++98 introduced the concept of in-class static constant initializers, 
so that our `Pow3` metaprogram could look as follows:
```c++
// primary template to compute 3 to the Nth
template <int N>
struct Pow3
{
    static int const value = 3 * Pow3<N - 1>::value;
};

// full specialization to end the recursion
template <>
struct Pow3<0>
{
    static int const value = 1;
};
```
However, there is a drawback with this version: Static constant members are lvalues. 
So, if we have a declaration such as `void foo(int const &);`,
and we pass it the result of a metaprogram: `foo(Pow3<7>::value);`,
a compiler must pass the _address_ of `Pow3<7>::value`, 
and that forces the compiler to instantiate and allocate the definition for the static member. 
As a result, the computation is no longer limited to a pure “compile-time” effect.


Enumeration values aren’t lvalues (i.e., they don’t have an address). 
So, when we pass them by reference, no static memory is used. 
It’s almost exactly as if you passed the computed value as a literal. 
The first edition of this book therefore preferred the use of enumerator constants for this kind of applications.


C++11, however, introduced `constexpr` static data members, 
and those are not limited to integral types. 
They do not solve the address issue raised above, 
but in spite of that shortcoming they are now a common way to produce results of metaprograms. 
They have the advantage of having a correct type (as opposed to an artificial enum type), 
and that type can be deduced when the static member is declared with the `auto` type specifier. 
C++17 added `inline` static data members, 
which do solve the address issue raised above, 
and can be used in conjunction with `constexpr`. 






### 🎯 Chapter 24 Typelists

#### 📌 24.1 Anatomy of a Typelist



#### 📌


#### 📌





#### 📌


#### 📌


#### 📌


#### 📌










### 🎯

#### 📌


#### 📌


#### 📌

#### 📌





### 🎯

#### 📌


#### 📌


#### 📌

#### 📌

