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
However, they **don’t** rely on the factoring of common behavior in base classes. 
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

template <typename GeoObj1, typename GeoObj2>
Coord distance(GeoObj1 const & a, GeoObj2 const & b)
{
    return distance(a.centroid(), b.centroid());
}

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





4 For a detailed discussion of polymorphism terminology, see also Sections 6.5 to
6.7 of [CzarneckiEiseneckerGenProg].
5 GCC 7, for example, provides the option -fconcepts.






### 🎯

#### 📌 
