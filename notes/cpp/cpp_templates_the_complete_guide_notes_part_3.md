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
struct HasType_##MemType : std::false_type {}; \
template <typename T> \
struct HasType_##MemType<T, std::void_t<typename T::MemType>> : std::true_type {}  // ; intentionally skipped
#endif  // DEFINE_HAS_TYPE

DEFINE_HAS_TYPE(value_type);
DEFINE_HAS_TYPE(char_type);

int main()
{
    std::cout << HasType_value_type<int>::value << '\n';               // false
    std::cout << HasType_value_type<std::vector<int>>::value << '\n';  // true
    std::cout << HasType_value_type<std::iostream>::value << '\n';     // false
    std::cout << HasType_char_type<std::iostream>::value << '\n';      // true
}
```

##### 19.6.3 Detecting Non-type Members








##### 19.6.5 Check If A Type Is Hashable

Whether hashable as well as customized hash functions: 
```c++
#include <pair>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <fmt/core.h>
#include <fmt/format.h>


template <typename T, typename = std::void_t<>>
struct IsHashable : std::false_type {};


template <typename T>
struct IsHashable<T, std::void_t<decltype( std::declval<std::hash<T>>()( std::declval<T>() ) )>> : std::true_type {};


template <typename T>
constexpr bool isHashableV = IsHashable<T>::value;


template <>
struct std::hash<std::tuple<int, int>>
{
    std::size_t operator()(const std::tuple<int, int> & tup) const
    {
        std::size_t seed {0};
        boost::hash_combine(seed, std::get<0>(tup));
        boost::hash_combine(seed, std::get<1>(tup));
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
    fmt::print(FMT_STRING("{}\n"), isHashableV<int>);                   // true
    fmt::print(FMT_STRING("{}\n"), isHashableV<std::tuple<int, int>>);  // true
    fmt::print(FMT_STRING("{}\n"), isHashableV<std::pair<int, int>>);   // true
    fmt::print(FMT_STRING("{}\n"), isHashableV<std::vector<int>>);      // false

    return EXIT_SUCCESS;
}
```
References: 
- [cpprefernece `std::unordered_map`](https://en.cppreference.com/w/cpp/container/unordered_map)
- [cppreference `std::hash`](https://en.cppreference.com/w/cpp/utility/hash)
- [`Boost.ContainerHash` `boost::hash_combine`](https://www.boost.org/doc/libs/1_79_0/libs/container_hash/doc/html/hash.html#combine)
- [StackOverflow "Check if a type is hashable"](https://stackoverflow.com/questions/12753997/check-if-type-is-hashable)



### 🎯

#### 📌 


22 Except that decltype(call-expression) does not require a nonreference, non-
void return type to be complete, unlike call expressions in other contexts. Using
decltype(std::declval<T>().begin(), 0) instead does add the
requirement that the return type of the call is complete, because the returned value
is no longer the result of the decltype operand.
23 Prior to C++11's expansion of SFINAE to cover arbitrary invalid expressions, the
techniques for detecting the validity of specific expressions centered on
introducing a new overload for the function being tested (e.g., <) that had an
overly permissive signature and a strangely sized return type to behave as a
fallback case. However, such approaches were prone to ambiguities and caused
errors due to access control violations.
24 In C++11 and C++14, we have to specify the base class as
std::integral_constant<bool,…> instead of
std::bool_constant<…>.25
The C++ standardization committee is further bound by a long-standing tradition
that all standard names consist of lowercase characters and optional underscores to
separate them. That is, a name like isSame or IsSame is unlikely to ever be
seriously considered for standardization (except for concepts, where this spelling
style will be used).
26 The use of “primary” vs. “composite” type categories should not be confused with
the distinction between “fundamental” vs. “compound” types. The standard
describes fundamental types (like int or std::nullptr_t) and compound
types (like pointer types and class types). This is different from composite type
categories (like arithmetic), which are categories that are the union of primary type
categories (like floating-point).
27 Specifically, when a function type is marked const, it refers to a qualifier on the
object pointed to by the implicit parameter this, whereas the const on a
const type refers to the object of that actual type.
28 The latest count is 48.
29 Most compilers support intrinsic operators like __is_union to help standard
libraries implement various traits templates. This is true even for some traits that
could technically be implemented using the techniques from this chapter, because
the intrinsics can improve compilation performance.
30 The first edition of this book described enumeration type detection in this way.
However, it checked for an implicit conversion to an integral type, which sufficed
for the C++98 standard. The introduction of scoped enumeration types into the
language, which do not have such an implicit conversion, complicates the
detection of enumeration types.
31 A copy or move constructor is called trivial if, in effect, a call to it can be replaced
by a simple copy of the underlying bytes.