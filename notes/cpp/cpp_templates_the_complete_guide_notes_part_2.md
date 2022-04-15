# _C++ Templates: The Complete Guide_ Notes

- _**C++ Templates**: The Complete Guide_ Second Edition by
    - David Vandevoorde
    - Nicolai M. Josuttis
    - Douglas Gregor
- Contents
  - [Part I The Basics](./cpp_templates_the_complete_guide_notes.md)
  - **[Part II Templates in Depth](./cpp_templates_the_complete_guide_notes_part_2.md)**
  - [Part III Templates and Design](./cpp_templates_the_complete_guide_notes_part_3.md)






## 🌱 Part II Templates in Depth



### 🎯 Chapter 12 Fundamentals in Depth


In this chapter we review some of the fundamentals
introduced in the first part of this book in depth:
the declaration of templates,
the restrictions on template parameters,
the constraints on template arguments,
and so forth.


#### 📌 12.1 Parameterized Declarations


C++ currently supports four fundamental kinds of templates:
- Class templates;
- Function templates;
- Variable templates;
- Alias templates.


Each of these template kinds can appear in namespace scope,
but also in class scope.
In class scope, they become
- Nested class templates;
- Member function templates;
- Static data member templates;
- Member alias templates.


Such templates are declared much like
ordinary classes, functions, variables, and type aliases
(or their class member counterparts)
except for being introduced
by a _parameterization clause_ of the form
```
template <template-parameter-list>
...
```
Note that C++17 introduced another construct that is introduced with such a parameterization clause:
_deduction guides_ (see Section 2.9 and Section 15.12.1 ).
Those **aren't** called templates (e.g., they are **not** instantiated),
but the syntax was chosen to be reminiscent of function templates.


We'll come back to the actual template parameter declarations in a later section.
First, some examples illustrate the four kinds of templates.
They can occur in namespace scope (including global namespace scope) as follows:
```c++
// a namespace scope class template
template <typename T>
class Data
{
public:
    static constexpr bool copyable = true;
};

// a namespace scope function template
template <typename T>
void log(T x) {}

// a namespace scope variable template (since C++14)
template <typename T> 
T pi = 3.141592653589793238462643383279502884L;

// a namespace scope variable template (since C++14)
template <typename T> 
bool dataCopyable = Data<T>::copyable;

// a namespace scope alias template
template <typename T> 
using DataList = Data<T *>;
```
Note that in this example, the static data member `Data<T>::copyable` is **not** a variable template,
even though it is indirectly parameterized through the parameterization of the class template Data.
However, a variable template can appear in class scope (as the next example will illustrate),
and in that case it is a static data member template.


The following example shows the four kinds of templates as class members
that are defined within their parent class:
```c++
class Collection
{
public:
    // an in-class member class template definition
    template <typename T>
    class Node {};

    // an in-class (and therefore implicitly inline)
    // member function template definition
    template <typename T>
    T * alloc() {}

    // a static member variable template (since C++14)
    template <typename T>
    static T pi = 3.141592653589793238462643383279502884L;

    // a member alias template
    template <typename T>
    using NodePtr = Node<T> *;
};
```
Note that in C++17, variables (including static data members)
and variable templates can be `inline`,
which means that their definition can be repeated across translation units.
This is redundant for variable templates,
which can always be defined in multiple translation units.
Unlike member functions, however, a static data member being defined in its enclosing class
does **not** make it inline:
The keyword `inline` must be specified in all cases.


Finally, the following code demonstrates how member templates
that are **not** alias templates can be defined out-of-class:
```c++
// a namespace scope class template
template <typename T> 
class List 
{
public:
    // because a template constructor is defined
    List() = default;
    
    // another member class template,
    // without its definition
    template <typename U> 
    class Handle;

    // a member function template (constructor)
    template <typename U> 
    List(List<U> const &);

    // a member variable template (since C++14)
    template <typename U> 
    static U pi;
};

// out-of-class member class template definition
template <typename T> 
template <typename U>
class List<T>::Handle {};

// out-of-class member function template definition
template <typename T> 
template <typename U>
List<T>::List (List<U> const & b) {}

// out-of-class static data member template definition
template <typename T> 
template <typename U>
U List<T>::pi = 3.141592653589793238462643383279502884L;
```
Member templates defined outside their enclosing class
may need multiple `template<...>` parameterization clauses:
one for every enclosing class template and one for the member template itself.
The clauses are listed starting from the outermost class template.


Note also that a constructor template
(a special kind of member function template)
**disables** the implicit declaration of the default constructor
(because it is only implicitly declared if **no** other constructor is declared).
Adding a defaulted declaration
```c++
List() = default;
```
ensures an instance of `List<T>` is default-constructible
with the semantics of an implicitly declared constructor.

##### Union Templates

_Union templates_ are possible too (and they are considered a kind of class template):
```c++
template <typename T>
union AllocChunk 
{
    T object;
    unsigned char bytes[sizeof(T)];
};
```

##### Default Call Arguments

Function templates can have default call arguments just like ordinary function declarations:
```c++
template <typename T>
void report_top(Stack<T> const & s, int number = 10);

// T {} is zero for built-in types
template <typename T>
void fill(Array<T> & a, T const & v = T {}); 
```
The latter declaration shows that a default call argument could depend on a template parameter.
It also can be defined as (the only way possible before C++11, see Section 5.2)
```c++
template <typename T>
void fill(Array<T> & a, T const & v = T()); 
```
When the `fill` function is called, the default argument is **not** instantiated
if a second function call argument is supplied.
This ensures that no error is issued if the default call argument
can **not** be instantiated for a particular `T`:
```c++
// no default constructor
class Value 
{
public:
    explicit Value(int); 
};

void init(Array<Value> & a)
{
    Value zero(0);
    fill(array, zero);  // OK: default constructor not used
    fill(array);        // ERROR: undefined default constructor for Value is used
}
```

##### Non-template Members of Class Templates

In addition to the four fundamental kinds of templates declared inside a class,
you can also have ordinary class members parameterized by being part of a class template.
They are occasionally (erroneously) also referred to as _member templates_.
Although they can be parameterized, such definitions **aren't** quite first-class templates.
Their parameters are entirely determined by the template of which they are members.
For example:
```c++
template <int I>
class CupBoard
{
    class Shelf;                // ordinary class in class template
    void open();                // ordinary function in class template
    enum Wood : unsigned char;  // ordinary enumeration type in class template
    static double totalWeight;  // ordinary static data member in class template
};
```
The corresponding definitions only specify a parameterization clause for the parent class templates,
but **not** for the member itself, because it is **not** a template
(i.e., no parameterization clause is associated with the name appearing after the last `::`):
```c++
// definition of ordinary class in class template
template <int I> 
class CupBoard<I>::Shelf {};

// definition of ordinary function in class template
template <int I>
void CupBoard<I>::open() {}

// definition of ordinary enumeration type class in class template
template <int I> 
enum CupBoard<I>::Wood 
{
    MAPLE, 
    CHERRY, 
    OAK
};

// definition of ordinary static member in class template
template <int I> 
double CupBoard<I>::totalWeight = 0.0;
```
Since C++17, the static `totalWeight` member can be initialized
inside the class template using `inline`:
```c++
template <int I>
class CupBoard
{
    inline static double totalWeight = 0.0;
    ...
};
```
Although such parameterized definitions are commonly called _templates_,
the term doesn't quite apply to them.
A term that has been occasionally suggested for these entities is _temploid_.
Since C++17, the C++ standard does define the notion of a _templated entity_,
which includes templates and temploids as well as,
recursively, any entity defined or created in templated entities
(that includes, e.g., a friend function defined inside a class template
or the closure type of a lambda expression appearing in a template).
Neither _temploid_ nor _templated entity_ has gained much traction so far,
but they may be useful terms to communicate more precisely about C++ templates in the future.

##### 12.1.1 Virtual Member Functions

Member function templates can **not** be declared virtual.
This constraint is imposed because the usual implementation of the virtual function call mechanism
uses a _fixed-size virtual table_ with one entry per virtual function.
However, the number of instantiations of a member function template
is **not** fixed until the entire program has been translated.
Hence, supporting virtual member function templates would require support
for a whole new kind of mechanism in C++ compilers and linkers.


In contrast, the ordinary members of class templates can be virtual
because their number is fixed when a class is instantiated:
```c++
template <typename T>
class Dynamic 
{
public:
    // OK: one destructor per instance of Dynamic<T>
    virtual ~Dynamic(); 
    
    // ERROR: unknown number of instances of copy
    // given an instance of Dynamic<T>
    template <typename U>
    virtual void copy(U const &);
};
```

##### 12.1.2 Linkage of Templates

Every template must have a name, and that name must be unique within its scope,
except that function templates can be overloaded (see Chapter 16).
Note especially that, unlike class types,
class templates can **not** share a name with a different kind of entity:
```c++
// OK: 
// class names and non-class names 
// are in a different "space"
int C;

class C;

// ERROR: conflict with variable X
int X;

template <typename T>
class X;

// ERROR: conflict with struct S
struct S;

template <typename T>
class S; 
```
Template names have linkage, but they can **not** have C linkage.
Nonstandard linkages may have an implementation-dependent meaning
(however, we don't know of an implementation that supports nonstandard name linkages for templates):
```c++
// this is the default: 
// the linkage specification could be left out
extern "C++" template <typename T>
void normal();

// ERROR: 
// templates can not have C linkage
extern "C" template <typename T>
void invalid();

// nonstandard, 
// but maybe some compiler will someday support
// linkage compatible with Java generics
extern "Java" template <typename T>
void javaLink();
```
Templates usually have external linkage.
The only exceptions are namespace scope function templates with the `static` specifier,
templates that are direct or indirect members of an unnamed namespace (which have internal linkage),
and member templates of unnamed classes (which have no linkage).
For example:
```c++
// refers to the same entity as a declaration 
// of the same name (and scope) in another file
template <typename T>
void external(); 

// unrelated to a template with the same name in another file
template <typename T>
static void internal();

// redeclaration of the previous declaration
template <typename T>
static void internal();

namespace
{

// also unrelated to a template with the same name in another file,
// even one that similarly appears
template <typename>
void otherInternal(); 

}  // namespace anonymous

namespace
{

// redeclaration of the previous template declaration
template <typename>
void otherInternal();

}  // namespace anonymous

struct
{
    // no linkage: cannot be redeclared
    template <typename T>
    void f(T) {} 
} x;
```
Note that since the latter member template has no linkage,
it must be defined within the unnamed class
because there is **no** way to provide a definition outside the class.


As of C++17, templates can **not** be declared in function scope or local class scope,
but _generic lambdas_ (see Section 15.10.6),
which have associated closure types that contain member function templates,
can appear in local scopes, which effectively implies a kind of local member function template.


The linkage of an instance of a template is that of the template.
For example, a function `internal<void>`
instantiated from the template `internal` declared above
will have internal linkage.
This has an interesting consequence in the case of variable templates:
```c++
template <typename T> 
T zero = T {};
```
All instantiations of `zero` have external linkage,
even something like `zero<int const>`.
That's perhaps counter-intuitive given that
```c++
int const zero = int {};
```
has internal linkage because it is declared with a `const` type.
Similarly, all instantiations of the template
```c++
template <typename T> 
int const max_volume = 11;
```
have external linkage,
despite all those instantiations also having type `int const`.

##### 12.1.3 Primary Templates

Normal declarations of templates declare _primary templates_.
Such template declarations are declared **without**
adding template arguments in angle brackets after the template name:
```c++
// OK: primary template
template <typename T>
class Box;

// ERROR: does not specialize
template <typename T>
class Box<T>;

// OK: primary template
template <typename T>
void translate(T);

// ERROR: not allowed for functions
template <typename T>
void translate<T>(T);

// OK: primary template
template <typename T> 
constexpr T zero = T {};

// ERROR: does not specialize
template <typename T> 
constexpr T zero<T> = T {}; 
```
Non-primary templates occur when declaring _partial specializations_ of class or variable templates.
Those are discussed in Chapter 16.
Function templates must always be primary templates (thus can **not** be partially specialized).


#### 📌 12.2 Template Parameters


There are three basic kinds of template parameters:
1. Type parameters (these are by far the most common);
2. Non-type parameters;
3. Template template parameters.


Any of these basic kinds of template parameters
can be used as the basis of a _template parameter pack_ (see Section 12.2.4).


Template parameters are declared in the introductory parameterization clause of a template declaration.
An exception since C++14 are the implicit template type parameters for a generic lambda.
See Section 15.10.6.

Such declarations do **not** necessarily need to be named:
```c++
// X is parameterized by a type and an integer
template <typename, int>
class X; 
```
A parameter name is, of course, required if the parameter is referred to later in the template.
Note also that a template parameter name can be referred to in a subsequent parameter declaration (but not before):
```c++
template <typename T, 
          T Root, 
          template <T> class Buf> 
class Structure;
```

##### 12.2.1 Type Parameters


Type parameters are introduced with either the keyword `typename` or the keyword `class`:
The two are entirely equivalent.
The keyword `class` does **not** imply that the substituting argument should be a class type.
It could be any accessible type.


The keyword must be followed by a simple identifier,
and that identifier must be followed by one of the following:
- A comma `,` to denote the start of the next parameter declaration;
- A closing angle bracket `>` to denote the end of the parameterization clause;
- An equal sign `=` to denote the beginning of a default template argument.


Within a template declaration, a type parameter acts much like a type alias.
For example, it is **not** possible to use an elaborated name of the form `class T` when `T` is a template parameter,
even if `T` were to be substituted by a class type:
```c++
template <typename Allocator>
class List 
{
    class Allocator * allocptr;  // ERROR: use "Allocator * allocptr"
    friend class Allocator;      // ERROR: use "friend Allocator"
    ...
};
```

##### 12.2.2 Non-type Parameters

Non-type template parameters stand for constant values
that can be determined at compile or link time (i.e. satisfies `constexpr`).


Template template parameters do **not** denote types either.
However, they are distinct from non-type parameters.
This oddity is historical:
Template template parameters were added to the language
_after_ type parameters and non-type parameters.


The type of such a parameter
(in other words, the type of the value for which it stands)
must be one of the following:
- An integer type or an enumeration type;
- A pointer type;
  - As of C++17,
    only "pointer to object" and "pointer to function" types are permitted,
    which **excludes** types like `void *`.
    However, all compilers appear to accept `void *` also.
- A pointer-to-member type;
- An lvalue reference type
  (both references to objects and references to functions are acceptable);
- `std::nullptr_t`;
- A type containing `auto` or `decltype(auto)` `(since C++17)` (see Section 15.10.1).


All other types are excluded as of C++17,
although floating-point types may be added in the future (see Section 17.2).


The declaration of a non-type template parameter can
in some cases also start with the keyword `typename`:
```c++
template <typename T,                         // a type parameter
          typename T::Allocator * Allocator>  // a non-type parameter
class List;
```
or with the keyword `class`:
```c++
class X {};

// a non-type parameter of pointer type
// keyword class is optional here
template <class X *>
class Y;

// a non-type parameter of pointer type
template <X *>
class Z;
```
The two cases are easily distinguished
because the first is followed by a simple identifier
and then one of a small set of tokens
(`=` for a default argument,
`,` to indicate that another template parameter follows,
or a closing `>` to end the template parameter list).
Keyword `typename` in the first non-type parameter
because it is a nested name dependent on template parameters
(see Section 5.1 and Section 13.3.2).


Function and array types can be explicitly specified as non-type template parameters,
but they are implicitly adjusted to the pointer type to which they decay:
```c++
template <int buf[5]> class Lexer;     // buf is really an int*
template <int * buf> class Lexer;      // OK: this is a redeclaration
template <int fun()> struct FuncWrap;  // fun really has pointer to function type
template <int (*)()> struct FuncWrap;  // OK: this is aredeclaration
```
Non-type template parameters are declared much like variables,
but they can **not** have non-type specifiers like `static`, `mutable`, and so forth.
They can have `const` and `volatile` qualifiers,
but if such a qualifier appears at the outermost level of the parameter type,
it is simply ignored:
```c++
template <int const length> class Buffer;  // const is useless here
template <int length> class Buffer;        // same as previous declaration
```
Finally, non-reference non-type parameters are always prvalues when used in expressions.
Their address can **not** be taken, and they can **not** be assigned to.
A non-type parameter of lvalue reference type, on the other hand, can be used to denote an lvalue:
```c++
template <int & Counter>
struct LocalIncrement 
{
    // OK: reference to an integer
    LocalIncrement() { Counter = Counter + 1; } 
    ~LocalIncrement() { Counter = Counter - 1; }
};
```
Rvalue references are **not** permitted.

##### 12.2.3 Template Template Parameters

Template template parameters are placeholders for class or alias templates.
They are declared much like class templates,
but the keywords `struct` and `union` can **not** be used:
```c++
template <template <typename X> class C>     // OK
void f(C<int> * p);

template <template <typename X> struct C>    // ERROR: struct not valid here
void f(C<int> * p);

template <template <typename X> union C>     // ERROR: union not valid here
void f(C<int> * p);
```
C++17 allows the use of `typename` instead of `class`:
That change was motivated by the fact that template template parameters can be substituted
not only by class templates but also by alias templates
(which instantiate to arbitrary types).
So, in C++17, our example above can be written instead as
```c++
template <template <typename X> typename C>  // OK since C++17
void f(C<int> * p);
```
In the scope of their declaration, template template parameters
are used just like other class or alias templates.


The parameters (`typename T`, `typename A`)
of template template parameters (`class Container`)
can have default template arguments (`A = MyAllocator`).
These default arguments apply
when the corresponding parameters are **not** specified
in uses of the template template parameter:
```c++
template <template <typename T, 
                    typename A = MyAllocator> 
          class Container>
class Adaptation 
{
    // implicitly equivalent to Container<int, MyAllocator>
    Container<int> storage;
    ...
};
```
`T` and `A` are the names of the template parameter
of the template template parameter `Container`.
These names be used _only_ in the declaration
of other parameters of that template template parameter.
The following contrived template illustrates this concept:
```c++
template <template <typename T, T *> 
          class Buf>
class Lexer 
{
    // ERROR: 
    // names of the template parameters 
    // of a template template parameter 
    // can not be used here 
    static T * storage; 
    ...
};
```
Usually however, the names of the template parameters of a template template parameter
are **not** needed in the declaration of other template parameters
and are therefore often left unnamed altogether.
For example, our earlier `Adaptation` template could be declared as follows:
```c++
template <template <typename, 
                    typename = MyAllocator> 
          class Container>
class Adaptation 
{
    // implicitly equivalent to Container<int, MyAllocator>
    Container<int> storage;
    ...
};
```

##### 12.2.4 Template Parameter Packs

Since C++11, any kind of template parameter can be turned into a _template parameter pack_
by introducing an ellipsis `...` prior to the template parameter name or,
if the template parameter is unnamed, where the template parameter name would occur:
```c++
// declares a template parameter pack named Types
template <typename ... Types> 
class Tuple;
```
A template parameter pack behaves like its underlying template parameter,
but with a crucial difference:
While a normal template parameter matches exactly one template argument,
a template parameter pack can match _any number of_ template arguments.
This means that the `Tuple` class template declared above accepts
any number of (possibly distinct) types as template arguments:
```c++
using IntTuple = Tuple<int>;             // OK: one template argument 
using IntCharTuple = Tuple<int, char>;   // OK: two template arguments
using IntTriple = Tuple<int, int, int>;  // OK: three template arguments
using EmptyTuple = Tuple<>;              // OK: zero template arguments
```
Similarly, template parameter packs of non-type and template template parameters
can accept any number of non-type or template template arguments, respectively:
```c++
// OK: declares a non-type template parameter pack
template <typename T, unsigned ... Dimensions>
class MultiArray;

// OK: 3x3 matrix
using TransformMatrix = MultiArray<double, 3, 3>;

// OK: declares a template template parameter pack
template <typename T, template <typename, typename> ... Containers>
void testContainers(); 
```
The `MultiArray` example requires all non-type template arguments to be of the same type `unsigned`.
C++17 introduced the possibility of deduced non-type template arguments,
which allows us to work around that restriction to some extent (see Section 15.10.1):
```c++
template <auto ... vs>
struct Values {};

Values<1, 2, 3> beginning;
Values<1, 'x', nullptr> triplet;

// homogeneous pack of non-type parameters
template <auto v1, decltype(v1) ... vs>
struct HomogeneousValues {};
```
Primary class templates, primary variable templates, and alias templates
may have _at most one_ template parameter pack and,
if present, the template parameter pack must be _the last_ template parameter.
Function templates have a weaker restriction:
Multiple template parameter packs are permitted,
as long as each template parameter subsequent to a template parameter pack
either has a default value (see the next section) or can be deduced (see Chapter 15):
```c++
// ERROR: template parameter pack is not the last template parameter
template <typename ... Types, typename Last>
class LastType;

// OK: template parameter pack is followed by a deducible template parameter
template <typename ... TestTypes, typename T>
void runTests(T value); 

template <unsigned ...> 
struct Tensor;

// OK: the tensor dimensions can be deduced
template <unsigned ... dims1, unsigned ... dims2>
auto compose(Tensor<dims1...>, Tensor<dims2...>);
```
The last example is the declaration of a function with a deduced return type `(since C++14)`.
See also Section 15.10.1.


Declarations of _partial specializations_ of class/variable templates (see Chapter 16)
_can_ have multiple parameter packs, unlike their primary template counterparts.
That is because partial specialization are selected through a deduction process
that is nearly identical to that used for function templates.
```c++
template <typename ...> 
struct Typelist;

template <typename X, typename Y> 
struct Zip;

// OK: partial specialization uses deduction 
// to determine the Xs and Ys substitutions
template <typename ... Xs, typename ... Ys>
struct Zip<Typelist<Xs...>, Typelist<Ys...>>;
```
A type parameter pack can **not** be expanded in its own parameter clause.
For example:
```c++
// ERROR: Ts cannot be expanded in its own parameter list
template <typename ... Ts, Ts... vals> 
struct StaticValues {};
```
However, nested templates can create similar valid situations:
```c++
template <typename ... Ts> struct ArgList 
{
    template<Ts ... vals> 
    struct Vals {};
};

ArgList<int, char, char>::Vals<3, 'x', 'y'> tada;
```
A template that contains a template parameter pack is called a _variadic template_
because it accepts a variable number of template arguments.
Chapter 4 and Section 12.4 describe the use of variadic templates.

##### 12.2.5 Default Template Arguments

Any kind of template parameter that is **not** a template parameter pack
can be equipped with a default argument,
although it must match the corresponding parameter in kind
(e.g., a type parameter can **not** have a non-type default argument).
A default argument can **not** depend on its own parameter,
because the name of the parameter is **not** in scope until after the default argument.
However, it may depend on previous parameters:
```c++
template <typename T, 
          typename Allocator = std::allocator<T>>
class List;
```
A template parameter for a class template, variable template, or alias template
can have a default template argument
only if default arguments were also supplied for the subsequent parameters.
(A similar constraint exists for default function call arguments.)
The subsequent default values are usually provided in the same template declaration,
but they could also have been declared in a previous declaration of that template.
The following example makes this clear:
```c++
// OK
template <typename T1, 
          typename T2, 
          typename T3,
          typename T4 = char, 
          typename T5 = char>
class Quintuple; 

// OK: T4 and T5 already have defaults
template <typename T1, 
          typename T2, 
          typename T3 = char,
          typename T4, 
          typename T5>
class Quintuple;

// ERROR: 
// T1 can not have a default argument
// because its successor T2 does not have one
template <typename T1 = char, 
          typename T2, 
          typename T3,
          typename T4, 
          typename T5>
class Quintuple; 
```
Default template arguments for template parameters of function templates
do **not** require subsequent template parameters to have a default template argument,
as those can still be determined by template argument deduction (see Chapter 15):
```c++
// OK: if not explicitly specified, R will be void
template <typename R = void, typename T>
R * addressof(T & value); 
```
Default template arguments can **not** be repeated:
```c++
template <typename T = void>
class Value;

// ERROR: repeated default argument
template <typename T = void>
class Value; 
```
A number of contexts do **not** permit default template arguments:
- Partial specializations
```c++
template <typename T>
class C;

// ERROR
template <typename T = int>
class C<T *>;
```
- Parameter packs
```c++
template <typename ... Ts = int> 
struct X;
```
- The out-of-class definition of a member of a class template:
```c++
template<typename T> 
struct X
{
    T f();
};

// ERROR
template <typename T = int> 
T X<T>::f() {}
```
- A friend class template declaration:
```c++
struct S 
{
    template <typename = void> 
    friend struct F;
};
```
- A friend function template declaration **unless** it is a definition
  and **no** declaration of it appears anywhere else in the translation unit:
```c++
struct S
{
    // ERROR: not adefinition
    template <typename = void> 
    friend void f(); 
    
    // OK so far
    template <typename = void> 
    friend void g() {}
};

// ERROR: 
// g was given a default template argument when defined.  
// No other declaration may exist here. 
template <typename> 
void g();
```


#### 📌 12.3 Template Arguments


When instantiating a template, template parameters are substituted by template arguments.
The arguments can be determined using several different mechanisms:
- **Explicit template arguments**:  
  A template name can be followed by explicit template arguments enclosed in angle brackets.
  The resulting name is called a _template-id_.
- **Injected class name**:  
  Within the scope of a class template `X` with template parameters `P1`, `P2`, ...,
  the name `X` can be equivalent to the template-id `X<P1, P2, ...>`.
  See Section 13.2.3 for details.
- **Default template arguments**:  
  Explicit template arguments can be omitted from template instances
  if default template arguments are available.
  However, for a class or alias template,
  even if all template parameters have a default value,
  the (possibly empty) angle brackets must be provided.
- **Argument deduction**:  
  Function template arguments that are **not** explicitly specified
  may be deduced from the types of the function call arguments in a call.
  This is described in detail in Chapter 15.
  Deduction is also done in a few other situations.
  If all the template arguments can be deduced,
  **no** angle brackets need to be specified after the name of the function template.
  C++17 also introduces the ability to deduce class template arguments
  from the initializer of a variable declaration or functional-notation type conversion.
  See Section 15.12 for a discussion.

##### 12.3.1 Function Template Arguments

Template arguments for a function template can be specified explicitly,
deduced from the way the template is used,
or provided as a default template argument.
For example:
```c++
template <typename T>
T max(T a, T b)
{
    return b < a ? a : b;
}

max<double>(1.0, -3.0);  // explicitly specify template argument
max(1.0, -3.0);          // template argument is implicitly deduced to be double
max<int>(1.0, 3.0);      // the explicit <int> inhibits the deduction;
                         // hence the result has type int
```
Some template arguments can **never** be deduced
because their corresponding template parameter does **not** appear in a function parameter type
or for some other reason (see Section 15.2).
The corresponding parameters are typically placed at the beginning of the list of template parameters,
so they can be specified explicitly while allowing the other arguments to be deduced:
```c++
// SrcT can be deduced, but DstT can not
template <typename DstT, typename SrcT>
DstT implicit_cast(SrcT const & x) 
{
    return x;
}

int main()
{
    double value = implicit_cast<double>(-1);
}
```
If we had reversed the order of the template parameters in this example
(in other words, if we had written `template<typename SrcT, typename DstT>`),
a call of `implicit_cast` would have to specify _both_ template arguments explicitly.


Moreover, such parameters can't usefully
be placed after a template parameter pack
or appear in a partial specialization,
because there would be **no** way to explicitly specify or deduce them.
```c++
// useless declaration,
// because N can not be specified or deduced
template <typename ... Ts, int N>
void f(double (&)[N + 1], Ts ... ps);
```
Because function templates can be overloaded,
explicitly providing all the arguments for a function template
may **not** be sufficient to identify a single function:
In some cases, it identifies a _set_ of functions.
The following example illustrates a consequence of this observation:
```c++
template <typename Func, typename T>
void apply(Func f, T x)
{
    f(x);
}

template <typename T> 
void single(T);

template <typename T> 
void multi(T);

template <typename T> 
void multi(T *);

int main()
{
    apply(&single<int>, 3);  // OK
    apply(&multi<int>, 7);   // ERROR: ambiguous call: no single multi<int>
}
```
In this example, the first call to `apply` works
because the type of the expression `&single<int>` is unambiguous.
As a result, the template argument value for the `Func` parameter is easily deduced.
In the second call, however, `&multi<int>` could be one of two different types
and therefore `Func` can **not** be deduced in this case.


Furthermore, it is possible that substituting template arguments in a function template
results in an attempt to construct an invalid C++ type or expression.
Consider the following overloaded function template (`RT1` and `RT2` are unspecified types):
```c++
template <typename T> 
RT1 test(typename T::X const *);

template <typename T> 
RT2 test(/* ... */);
```
The expression `test<int>` makes **no** sense
for the first of the two function templates
because type `int` has **no** member type `X`.
However, the second template has no such problem.
Therefore, the expression `&test<int>` identifies the address of a single function.
The fact that the substitution of `int` into the first template fails does not make the expression invalid.
This SFINAE principle is an important ingredient to make the overloading of function templates practical
and is discussed in Section 8.4 and Section 15.7.

##### 12.3.2 Type Arguments

Template type arguments are the "values" specified for template type parameters.
Any type (including `void`, function types, reference types, etc.) can be used as a template argument,
as long as their substitution for the template parameters must lead to valid constructs:
```c++
// requires that the unary * be applicable to T
template <typename T>
void clear(T p)
{
    *p = 0;
}

int main()
{
    int a;
    clear(a);  // ERROR: int doesn't support the unary *
}
```

##### 12.3.3 Non-type Arguments

Non-type template arguments are the values substituted for non-type parameters.
Such a value must be one of the following things:
- Another non-type template parameter that has the right type.
- A compile-time constant value of integer (or enumeration) type.
  This is acceptable only if the corresponding parameter has a type
  that matches that of the value or a type
  to which the value can be implicitly converted **without** narrowing.
  For example, a `char` value can be provided for an `int` parameter,
  but `500` is **not** valid for an 8-bit `char` parameter.
- The name of an external variable or function
  preceded by the built-in unary address-of operator `operator&`.
  For functions and array variables, `operator&` can be left out.
  Such template arguments match non-type parameters of a pointer type.
  C++17 relaxed this requirement to permit any constant-expression
  that produces a pointer to a function or variable.
- The previous kind of argument but **without** a leading `operator&`
  is a valid argument for a non-type parameter of reference type.
  C++17 relaxed the constraint to permit any constant-expression `glvalue` for a function or variable.
- A pointer-to-member constant.  
  In other words, an expression of the form `&ClassType::nonStaticMember`.
  This matches non-type parameters of pointer-to-member type _only_.
  In C++17, any constant-expression evaluating to a matching pointer-to-member constant is permitted.
- A null pointer constant is a valid argument for a non-type parameter of pointer or pointer-to-member type.


For non-type parameters of integral type,
implicit conversions to the parameter type are considered.
With the introduction of `constexpr` conversion functions in C++11,
this means that the argument before conversion can have a class type.


Prior to C++17, when matching an argument to a parameter that is a pointer or reference,
user-defined conversions (constructors for one argument and conversion operators)
and derived-to-base conversions are **not** considered,
even though in other circumstances they would be valid implicit conversions.
Implicit conversions that make an argument more `const` and/or more `volatile` are fine.
Here are some valid examples of non-type template arguments:
```c++
template <typename T, T nonTypeParam>
class C;

// integer type
C<int, 33> * c1;

// address of an external variable
int a;
C<int*, &a> * c2;

// Name of a function. 
// Overload resolution selects f(int) in this case. 
// The & is implied. 
void f();
void f(int);

C<void (*)(int), f> * c3;

// function template instantiations are functions
template <typename T> 
void templ_func();

C<void (), &templ_func<double>> * c4; 

struct X 
{
    static bool b;
    int n;
    constexpr operator int() const { return 42; }
};

// static class members are acceptable variable/function names 
C<bool &, X::b> * c5;

// an example of a pointer-to-member constant 
C<int X::*, &X::n> * c6;

// OK. 
// X is first converted to int via a constexpr conversion function, 
// and then to long via a standard integer conversion. 
C<long, X {}> * c7; 
```
A general constraint of template arguments is that
a compiler or a linker must be able to express their value when the program is being built.
Values that aren't known until a program is run (e.g., the address of local variables)
**aren't** compatible with the notion that templates are instantiated when the program is built.


There are some constant values that are, **not** currently valid:
- Floating-point numbers;
- String literals;
- Null pointer constants `(until C++11)`


One of the problems with string literals is that
two identical literals can be stored at two distinct addresses.
An alternative (but cumbersome) way to express templates instantiated over constant strings
involves introducing an additional variable to hold the string:
```c++
template <char const * str>
class Message {};

extern char const hello[] = "Hello World!";
char const hello11[] = "Hello World!";

void foo()
{
    static char const hello17[] = "Hello World!";
    Message<hello> msg03;    // OK in all versions
    Message<hello11> msg11;  // OK since C++11
    Message<hello17> msg17;  // OK since C++17
}
```
The requirement is that a non-type template parameter declared as reference or pointer
can be a constant expression with:
- External linkage;
- Internal linkage `(since C++11)`;
- Any linkage `(since C++17)`.

Here are a few other invalid examples:
```c++
template <typename T, T nontypeParam>
class C;

struct Base 
{
    int i;
};

struct Derived : public Base {}; 

Base base;
Derived derived;

C<Base*, &derived> * err1;  // ERROR: derived-to-base conversions are not considered
C<int &, base.i> * err2;    // ERROR: fields of variables aren't considered to be variables

int a[10];
C<int *, &a[0]> * err3;     // ERROR: aren't acceptable either
```

##### 12.3.4 Template Template Arguments

A template template argument must generally be a class template or alias template
with parameters that exactly match the parameters of the template template parameter it substitutes.
Prior to C++17, default template arguments of a template template argument were _ignored_
(but if the template template parameter has default arguments,
they are considered during the instantiation of the template).
C++17 relaxed the matching rule to just require that
the template template parameter be _at least as specialized (see Section 16.2.2) as_
the corresponding template template argument.


This makes the following example invalid prior to C++17:
```c++
// Container expects one parameter
template<typename T1, 
         typename T2,
         template <typename> class Container>
class Rel {};

// ERROR before C++17: 
// std::list<T, Alloc = std::allocator<T>> has more than template parameter
// Default template arguments are not considered during substitution
Rel<int, double, std::list> rel;
```
The problem in this example is that the `std::list` template has more than one parameter.
The second parameter (which describes an allocator) has a default value,
but prior to C++17, that is **not** considered when matching `std::list` to the `Container` parameter.


Variadic template template parameters are an **exception**
to the pre-C++17 "exact match" rule described above
and offer a solution to this limitation:
They enable more general matching against template template arguments.
A template template parameter pack can match zero or more template parameters
of the same kind (type, non-type, template) in the template template argument:
```c++
// Container expects any number of type parameters
template <typename T1, 
          typename T2,
          template <typename ...> class Container>
class Rel {};

// OK: 
// std::list has two template arguments,
// but can be used with one argument. 
Rel<int, double, std::list> rel;
```
Template parameter packs can only match template arguments
of the same kind (type, non-type, template).
For example, the following class template can be instantiated
with any class template or alias template having only template type parameters,
because the template type parameter pack passed there as `TT`
can match zero or more template type parameters:
```c++
template <template <typename... > class TT>
class AlmostAnyTmpl {};

AlmostAnyTmpl<std::vector> withVector;  // two type parameters
AlmostAnyTmpl<std::map> withMap;        // four type parameters

// ERROR: 
// A template type parameter pack
// doesn't match a non-type template parameter
AlmostAnyTmpl<std::array> withArray;
```
Prior to C++17, only the keyword `class` could be used to declare a template template parameter.
But this does **not** indicate that
only class templates declared with the keyword `class`
were allowed as substituting arguments.
`struct`, `union`, and alias templates are all valid arguments
for a template template parameter.
This is similar to the observation that any type can be used as an argument
for a template type parameter declared with the keyword `class`.

##### 12.3.5 Equivalence

Two sets of template arguments are equivalent
when values of the arguments are identical one-for-one.
For type arguments, type aliases **don't** matter:
It is the type ultimately underlying the type alias declaration that is compared.
For integer non-type arguments, the value of the argument is compared.
How that value is expressed **doesn't** matter:
```c++
// Note NO template definition is needed 
// to establish template equivalence!
template <typename T, int I>
class Mix;

using Int = int;

// p2 has the same type as p1
Mix<int, 3 * 3> * p1;
Mix<Int, 4 + 5> * p2;
```
As is clear from this example,
**no** template definition is needed
to establish the equivalence of the template argument lists.


In template-dependent contexts,
the "value" of a template argument can **not** always be established definitely,
and the rules for equivalence become a little more complicated.
```c++
template <int N> struct I {};
template <int M, int N> void f(I<M + N>);  // #1
template <int N, int M> void f(I<N + M>);  // #2
template <int M, int N> void f(I<N + M>);  // #3 ERROR
```
`#1` and `#2` are _equivalent_ and declare the same function template `f`,  
because these two templates are the same by swapping the naming of `N` and `M`.
The expressions `M + N` and `N + M` in those two declarations are called _equivalent_.

Declaration `#3` is subtly different: The order of the operands is inverted.
That makes the expression `N + M` in `#3` **not** equivalent to either of the other two expressions.
However, because the expression will produce the same result for any values of the template parameters involved,
those expressions are called _functionally equivalent_.
It is an **error** for templates to be declared in ways that differ only because
the declarations include functionally equivalent expressions that are not actually equivalent.


However, such an error need **not** be diagnosed by your compiler.
That's because some compilers may
internally represent `N + 1 + 1` in exactly the same way as `N + 2`,
whereas other compilers may not.
Rather than impose a specific implementation approach,
the standard allows either one and requires programmers to be careful in this area.


A function generated from a function template is **never** equivalent to an ordinary function
even though they may have the same type and the same name.
This has two important consequences for class members:
1. A function generated from a member function template
   **never** overrides a virtual function.
2. A constructor generated from a constructor template is
   **never** a copy or move constructor.
   However, a constructor template can be a default constructor.
   Similarly, an assignment generated from an assignment template
   is **never** a copy-assignment or move-assignment operator.


This can be good and bad:
- It can happen that a template constructor or assignment operator
  is a better match than the predefined copy/move constructor or assignment operator,
  although a template version is provided for initialization of other types only.
  See Section 6.2 for details.
- It is not easy to "templify" a copy/move constructor,
  for example, to be able to constrain its existence.
  See Section 6.4 for details.


#### 📌 12.4 Variadic Templates


Variadic templates are templates that contain at least one template parameter pack.
The term variadic is borrowed from C's variadic functions,
which accept a variable number of function arguments.
Variadic templates also borrowed from C the use of the ellipsis to denote zero or more arguments
and are intended as a typesafe replacement for C's variadic functions for some applications.


Variadic templates are useful when a template's behavior
can be generalized to any number of arguments.
The `Tuple` class template introduced in Section 12.2.4 is one such type,
because a tuple can have any number of elements,
all of which are treated the same way.
We can also imagine a simple `print` function
that takes any number of arguments and displays each of them in sequence.


When template arguments are determined for a variadic template,
each template parameter pack in the variadic template will match
a sequence of zero or more template arguments (an _argument pack_).
```c++
template <typename ... Types>
class Tuple {};

int main() 
{
    Tuple<> t0;            // Types contains an empty list
    Tuple<int> t1;         // Types contains int
    Tuple<int, float> t2;  // Types contains int and float
}
```
Because a template parameter pack represents a list of template arguments
rather than a single template argument,
it must be used in a context where the same language construct
applies to all of the arguments in the argument pack.
One such construct is the `sizeof...` operation,
which counts the number of arguments in the argument pack:
```c++
template <typename ... Types>
class Tuple
{
public:
    static constexpr std::size_t length = sizeof...(Types);
};

int a1[Tuple<int>::length];               // array of one integer
int a3[Tuple<short, int, long>::length];  // array of three integers
```

##### 12.4.1 Pack Expansions

The `sizeof...` expression is an example of a _pack expansion_.
A pack expansion is a construct that expands an argument pack into separate arguments.
`sizeof...` is one form of pack expansion.
Other forms pack expansions are identified by an ellipsis `...` to the right of an element in the list.
```c++
template <typename ...>
class Tuple {};

template <typename ... Types>
class MyTuple : public Tuple<Types ...> {};

MyTuple<int, float> t2;  // inherits from Tuple<int, float>
```
Note that you **can't** access the individual elements of a parameter pack directly by name,
because names for individual elements are **not** defined in a variadic template.
If you need the types, the only thing you can do is to pass them (recursively) to another class or function.


Each pack expansion has a pattern,
which is the type or expression that will be repeated for each argument in the argument pack
and typically comes before the ellipsis that denotes the pack expansion.
Our prior examples have had only trivial patterns (the name of the parameter pack)
but patterns can be arbitrarily complex.
```c++
template <typename ... Types>
class PtrTuple : public Tuple<Types * ...> {};

PtrTuple<int, float> t3;  // Inherits from Tuple<int *, float *>
```

##### 12.4.2 Where Can Pack Expansions Occur?

Our examples thus far have focused on the use of pack expansions
to produce a sequence of template arguments.
In fact, pack expansions can be used essentially anywhere
in the language where the grammar provides a comma-separated list,
including:
- In the list of base classes.
- In the list of base class initializers in a constructor.
- In a list of call arguments (the pattern is the argument expression).
- In a list of initializers (e.g., in a braced initializer list).
- In the template parameter list of a class, function, or alias template.
- In the list of exceptions that can be thrown by a function
  (deprecated in C++11 and C++14, and disallowed in C++17).
- Within an attribute, if the attribute itself supports pack expansions
  (although no such attribute is defined by the C++ standard).
- When specifying the alignment of a declaration.
- When specifying the capture list of a lambda.
- In the parameter list of a function type.
- In using declarations `(since C++17)`.


We've already mentioned `sizeof...` as a pack-expansion mechanism
that does **not** actually produce a list.
C++17 also adds _fold expressions_,
which are another mechanism that does **not** produce a comma-separated list.
```c++
template <typename ... Mixins>
class Point : public Mixins...  // base class pack expansion
{
public:
    Point() : Mixins()... {}  // base class initializer pack expansion

    template <typename Visitor>
    void visitMixins(Visitor visitor)
    {
        visitor(static_cast<Mixins &>(*this)...);  // call argument pack expansion
    }

private:
    double x, y, z;
};

struct Color { char red, green, blue; };
struct Label { std::string name; };

Point<Color, Label> p;  // inherits from both Color and Label
```
A pack expansion can also be used within a template parameter list
to create a non-type or template parameter pack:
```c++
template <typename ... Ts>
struct Values 
{
    template <Ts ... vs>
    struct Holder {};
};

int i;
Values<char, int, int *>::Holder<'a', 17, &i> valueHolder;
```
Note that once the type arguments for `Values` have been specified,
the non-type argument list for `Values::Holder` has a fixed length;
the parameter pack `vs` is thus **not** a variable-length parameter pack.

`Values` is a non-type template parameter pack for which
each of the actual template arguments can have a different type,
as specified by the types provided for the template type parameter pack `Ts`.
Note that the ellipsis in the declaration of `Values` plays a dual role,
both declaring the template parameter as a template parameter pack
and declaring the type of that template parameter pack as a pack expansion.

##### 12.4.3 Function Parameter Packs

A _function parameter pack_ is a function parameter that matches zero or more function call arguments.
Like a template parameter pack, a function parameter pack is introduced
using an ellipsis `...` prior to (or in the place of) the function parameter name.
A function parameter pack must be expanded by a pack expansion whenever it is used.
Template parameter packs and function parameter packs together are referred to as _parameter packs_.


Unlike template parameter packs, function parameter packs are always pack expansions,
so their declared types must include at least one parameter pack.
```c++
template <typename ... Mixins>
class Point : public Mixins...
{
public:
    Point(Mixins... mixins) : Mixins(mixins)... {}
    
private:
    double x, y, z;
};

struct Color { char red, green, blue; };
struct Label { std::string name; };

Point<Color, Label> p({0x7f, 0, 0x7f}, {"center"});
```
A function parameter pack for a function template may depend on
template parameter packs declared in that template,
which allows the function template to accept an arbitrary number of call arguments
without losing type information:
```c++
template <typename ... Types>
void print(Types ... values);

int main()
{
    std::string welcome("Welcome to ");
    print(welcome, "C++", 2011, '\n');  //calls print<std::string, char const *, int, char>
}
```
There is a syntactic ambiguity between an unnamed function parameter pack
appearing at the end of a parameter list and a C-style `vararg` parameter:
```c++
template <typename T> 
void c_style(int, T ...);

template <typename ... T> 
void pack(int, T ...);
```
In the first case, the `T ...` is treated as `T, ...`:
an unnamed parameter of type `T` followed by a C-style `vararg` parameter.
In the second case, the `T ...` construct is treated as a function parameter pack
because `T` is a valid expansion pattern.
The disambiguation can be forced by adding a comma before the ellipsis
(which ensures the ellipsis is treated as a C-style `vararg` parameter)
or by following the `...` by an identifier,
which makes it a named function parameter pack.
Note that in generic lambdas, a trailing `...` will be treated as denoting a parameter pack
if the type that immediately precedes it (with **no** intervening comma) contains `auto`.

##### 12.4.4 Multiple and Nested Pack Expansions

The pattern of a pack expansion can be arbitrarily complex
and may include multiple, nested, distinct parameter packs.
When instantiating a pack expansion containing multiple parameter packs,
all of the parameter packs must have the _same length_.
```c++
template <typename F, typename ... Types>
void forwardCopy(F f, Types const & ... values) 
{
    f(Types(values)...);
}
```
```c++
template <typename ... OuterTypes>
class Nested 
{
    template <typename ... InnerTypes>
    void f(InnerTypes const & ... innerValues) 
    {
        g(OuterTypes(InnerTypes(innerValues)...)...);
    }
};
```

##### 12.4.5 Zero-Length Pack Expansions

The syntactic interpretation of pack expansions can be a useful tool
for understanding how an instantiation of a variadic template
will behave with different numbers of arguments.
However, the syntactic interpretation often **fails** in the presence of _zero-length argument packs_.
To illustrate this, consider the Point class template syntactically substituted with zero arguments:
```c++
template <typename ... Mixins>
struct Point : public Mixins...
{
    Point(Mixins... mixins) : Mixins(mixins)... {}
};
```
```c++
template <>
struct Point : 
{
    Point() : {}
};
```
The code as written above is ill-formed,
since the template parameter list is now empty
and the empty base class and base class initializer lists
each have a stray colon character.


Pack expansions are actually semantic constructs,
and the substitution of an argument pack of any size does **not** affect how the pack expansion
(or its enclosing variadic template) is parsed.
Rather, when a pack expansion expands to an empty list,
the program behaves (semantically) as if the list were **not** present.
The instantiation `Point<>` ends up having **no** base classes,
and its default constructor has **no** base class initializers but is otherwise well-formed.
This semantic rules holds even when the syntactic interpretation of zero-length pack expansion
would be well-defined (but different) code:
```c++
template <typename T, typename ... Types>
void g(Types ... values)
{
    T v(values...);
}
```
The variadic function template `g` creates a value `v`
that is direct-initialized from the sequence of values it is given.
If that sequence of values is empty, the declaration of `v` looks
syntactically like a function declaration `T v()`.
However, since substitution into a pack expansion is semantic
and can **not** affect the kind of entity produced by parsing,
`v` is initialized with zero arguments (value-initialization).


There is a similar restriction on members of class templates and nested classes within class templates:
If a member is declared with a type that does not appear to be a function type,
but after instantiation the type of that member is a function type,
the program is ill-formed because the semantic interpretation of the member
has changed from a data member to a member function.

##### 12.4.6 Fold Expressions

A recurring pattern in programming is the _fold_ of an operation on a sequence of values.
For example, a *right fold o*f a function fn over a sequence
`x[1], x[2], ..., x[n-1], x[n]` is given by `fn(x[1], fn(x[2], fn(..., fn(x[n - 1], x[n])...)))`.
While exploring a new language feature,
the C++ committee ran into the need to deal with such constructs
for the special case of a logical binary operator (i.e., `&&` or `||`) applied to a pack expansion.
Without an extra feature, we might write the following code to achieve that for the `&&` operator:
```c++
bool and() 
{ 
    return true; 
}

template <typename T>
bool and(T cond) 
{  
    return cond; 
}

template <typename T, typename ... Ts>
bool and(T cond, Ts ... conds) 
{
    return cond && and(cond...);
}
```
With C++17, a new feature called _fold expressions_ was added (see Section 4.2).
It applies to all binary operators **except** `.,` `->`, and `[]`.
Given an unexpanded expression pattern _pack_ and a non-pattern expression _value_,
C++17 allows us to write for a _right fold_ of the operator `op` (_binary right fold_):
```c++
(pack op ... op value)
```
or for a _left fold_ of the operator `op` (_binary left fold_):
```c++
(value op ... op pack)
```
Note that the parentheses are required here.


The fold operation applies to the sequence
that results from expanding the pack
and adding value as either the last element of the sequence (for a right fold)
or the first element of the sequence (for a left fold).


With this feature available:
```c++
template <typename ... T> bool g() 
{
    return and(trait<T>()...);
}
```
turns into
```c++
template <typename ... T> 
bool g() 
{
    return (trait<T>() && ... && true);
}
```
Fold expressions are pack expansions.
Note that if the pack is empty,
the type of the fold expression can still be determined from the non-pack operand (`value` in the forms above).


However, the designers of this feature also wanted an option to leave out the `value` operand.
Two other forms are therefore available in C++17:
The _unary right fold_
```c++
(pack op ...)
```
and the _unary left fold_
```c++
(... op pack)
```
Again, the parentheses are required.
Clearly this creates a problem for empty expansions:
How do we determine their type and value?
The answer is that an empty expansion of a unary fold is generally an error,
with three exceptions:
- An empty expansion of a unary fold of `&&` produces the value `true`.
- An empty expansion of a unary fold of `||` produces the value `false`.
- An empty expansion of a unary fold of the comma operator `,` produces a `void` expression.


Note that this will create surprises if you overload one of these special operators
in a somewhat unusual way:
```c++
struct BooleanSymbol {};

BooleanSymbol operator||(BooleanSymbol, BooleanSymbol);

template <typename ... BTs> 
void symbolic(BTs ... ps) 
{
    BooleanSymbol result = (ps || ...);
}
```
Suppose we call `symbolic` with types that are derived from `BooleanSymbol`.
For all expansions, the result will produce a `BooleanSymbol` value
**except** for the empty expansion, which will produce a bool `value`.


Because overloading these three special operators is unusual,
this problem is fortunately rare (but subtle).
The original proposal for fold expressions included empty expansion values
for more common operators like `+` and `*`, which would have caused more serious problems.


We therefore generally caution against the use of unary fold expressions,
and recommend using binary fold expressions instead
(with an explicitly specified empty expansion value).


#### 📌 12.5 Friends


The basic idea of friend declarations is a simple one:
Identify classes or functions that have a privileged connection
with the class in which the friend declaration appears.
Matters are somewhat complicated, by two facts:
1. A friend declaration may be the only declaration of an entity.
2. A friend function declaration can be a definition.

##### 12.5.1 Friend Classes of Class Templates

Friend class declarations can **not** be definitions and therefore are rarely problematic.
In the context of templates, the only new facet of friend class declarations
is the ability to name a particular instance of a class template as a friend:
```c++
template <typename T>
class Node;

template <typename T>
class Tree 
{
    friend class Node<T>;
};
```
Note that the class template must be visible at the point
where one of its instances is made a friend of a class or class template.
With an ordinary class, there is **no** such requirement:
```c++
template <typename T>
class Tree 
{
    friend class Factory;  // OK even if first declaration of Factory
    friend class Node<T>;  // ERROR if Node is not visible
};
```
Section 13.2.2 has more to say about this.


One application is the declaration of other class template instantiations to be friends:
```c++
template <typename T>
class Stack 
{
public:
    // to get access to private members of Stack<U> for any type U:
    template <typename> friend class Stack;
    
    // assign stack of elements of type U
    template <typename U>
    Stack<T> & operator= (Stack<U> const &);

    ...
};
```
C++11 also added syntax to make a template parameter a friend:
```c++
template <typename T>
class Wrap 
{
    friend T;
    ...
};
```
This is valid for any type `T` but is ignored if `T` is **not** actually a class type.

##### 12.5.2 Friend Functions of Class Templates

An instance of a function template can be made a friend
by making sure the name of the friend function is followed by angle brackets.
The angle brackets can contain the template arguments,
but if the arguments can be deduced,
the angle brackets can be left empty:
```c++
template <typename T1, typename T2>
void combine(T1, T2);

class Mixer 
{
    // OK: T1 = int &, T2 = int &
    friend void combine<>(int &, int &);
    
    // OK: T1 = int, T2 = int
    friend void combine<int, int>(int, int);
    
    // OK: T1 = char T2 = int
    friend void combine<char>(char, int);
    
    // OK: definition of a specialization
    friend void combine<>(long, long) {}
    
    // ERROR: does not match combine template
    friend void combine<char>(char &, int);
};
```
Note that we can **not** define a template instance (at most, we can define a specialization),
and hence a friend declaration that names an instance can **not** be a definition.
If the name is **not** followed by angle brackets, there are two possibilities:
1. If the name isn't qualified (in other words, it doesn't contain `::`),
   it **never** refers to a template instance.
   If no matching non-template function is visible at the point of the friend declaration,
   the friend declaration is the first declaration of that function.
   The declaration could also be a definition.
2. If the name is qualified (it contains `::`),
   the name _must_ refer to a previously declared function or function template.
   A matching function is preferred over a matching function template.
   However, such a friend declaration can **not** be a definition.
   An example may help clarify the various possibilities:
```c++
void multiply(void *);

template <typename T>
void multiply(T);

class Comrades 
{
    // defines a new function ::multiply(int)
    friend void multiply(int) {}

    // refers to the ordinary function above,
    // not to the multiply<void *> instance
    friend void ::multiply(void *);
    
    // refers to an instance of the template
    friend void ::multiply(int);

    // qualified names can also have angle brackets,
    // but a template must be visible
    friend void ::multiply<double *>(double *);

    // ERROR: a qualified friend can not be a definition
    friend void ::error() {}
};
```
In our previous examples, we declared the friend functions in an ordinary class.
The same rules apply when we declare them in class templates,
but the template parameters may participate in identifying the function that is to be a friend:
```c++
template <typename T>
class Node 
{
    Node<T> * allocate();
    ...
};

template <typename T>
class List 
{
    friend Node<T> * Node<T>::allocate();
    ...
};
```
A friend function may also be defined within a class template,
in which case it is only instantiated when it is actually used.
This typically requires the friend function
to use the class template itself in the type of the friend function,
which makes it easier to express functions on the class template
that can be called as if they were visible in namespace scope
(name found via ADL):
```c++
template <typename T>
class Creator 
{
    // Every T instantiates a different function ::feed
    friend void feed(Creator<T>) {}
};

int main()
{
    Creator<void> one;
    feed(one);  // Instantiates ::feed(Creator<void>)
    
    Creator<double> two;
    feed(two);  // Instantiates ::feed(Creator<double>)
}
```
In this example, every instantiation of `Creator` generates a different function.
Note that even though these functions are generated as part of the instantiation of a template,
the functions themselves are ordinary functions, `not` instances of a template.
However, they are considered _templated entities_ (see Section 12.1)
and their definition is instantiated only when used.
Also note that because the body of these functions is defined inside a class definition,
they are implicitly inline.
Hence, it is **not** an error for the same function to be generated in two different translation units.
Section 13.2.2 Section 21.2.1 have more to say about this topic.

##### 12.5.3 Friend Templates

Usually when declaring a friend that is an instance of a function or a class template,
we can express exactly which entity is to be the friend.
Sometimes it is useful to express that all instances of a template are friends of a class.
This requires a _friend template_:
```c++
class Manager 
{
    template <typename T>
    friend class Task;
    
    template <typename T>
    friend void Schedule<T>::dispatch(Task<T> *);
    
    template <typename T>
    friend int ticket() 
    {
        return ++Manager::counter;
    }
    
    static int counter;
};
```
Just as with ordinary friend declarations, a friend template can be a definition
only if it names an unqualified function name that is **not** followed by angle brackets
(except defining specializations of existing templates).


A friend template can declare only primary templates and members of primary templates.
Any partial specializations and explicit specializations associated with a primary template
are automatically considered friends too.






### 🎯 Chapter 13 Names in Templates


When a C++ compiler encounters a name,
it must "look it up" to identify the entity being referred.
From an implementer's point of view, C++ is a hard language in this respect.
Consider the C++ statement `x * y;`.
If `x` and `y` are the names of variables, this statement is a multiplication,
but if `x` is the name of a type, then the statement declares `y` as a pointer to an entity of type `x`.


This small example demonstrates that C++ (like C) is a _context-sensitive_ language:
A construct can **not** always be understood without knowing its wider context.
How does this relate to templates?
Well, templates are constructs that must deal with multiple wider contexts:
1. The context in which the template appears;
2. The context in which the template is instantiated;
3. The contexts associated with the template arguments for which the template is instantiated.


#### 📌 13.1 Name Taxonomy


C++ classifies names in a large variety of ways.
Fortunately, you can gain good insight into most C++ template issues
by familiarizing yourself with two major naming concepts:
1. A name is a _qualified name_ if the scope to which it belongs
   is explicitly denoted using a scope-resolution operator (`::`) or a member access operator (`.` or `->`).
   For example, `this->count` is a qualified name,
   but `count` is not (even though the plain `count` might actually refer to a class member).
2. A name is a _dependent name_ if it depends in some way on a template parameter.
   For example, `std::vector<T>::iterator` is usually a dependent name if `T` is a template parameter,
   but it is a non-dependent name if `T` is a known type alias (such as the `T` from `using T = int`).

- **Identifier**:  
  A name that consists solely of an uninterrupted sequences of letters, underscores `_`, and digits.
  It can **not** start with a digit, and some identifiers are reserved for the implementation:
  You should **not** introduce them in your programs
  (as a rule of thumb, avoid leading underscores and double underscores).
  The concept of "letter" should be taken broadly and includes special _universal character names_
  (UCNs) that encode glyphs from non-alphabetical languages.
- **Operator-function-id**:  
  The keyword `operator` followed by the symbol for an operator.
  for example, `operator new` and `operator []`.
- **Conversion-function-id**:  
  Used to denote a user-defined implicit conversion operator.
  For example, `operator int &`,
  which could also be obfuscated as `operator int bitand`.
- **Literal-operator-id**:   
  Used to denote a user-defined literal operator.
  For example, `operator ""_km`, which will be used when writing a literal such as `100_km`.
- **Template-id**:  
  The name of a template followed by template arguments enclosed in angle brackets.
  For example, `List<T, int, 0>`.
  A template-id may also be an `operator-function-id` or a `literal-operator-id`
  followed by template arguments enclosed in angle brackets.
  For example, `operator+<X<int>>`.
- **Unqualified-id**:  
  The generalization of an identifier.
  It can be any of the above
  (`identifier`, `operator-function-id`, `conversion-function-id`, `literal-operator-id`, or `template-id`)
  or a "destructor name"
  (e.g., notations like `~Data` or `~List<T, T, N>`).
- **Qualified-id**:  
  An `unqualified-id` that is qualified with the name of a class, enum, or namespace,
  or just with the global scope resolution operator.
  Note that such a name itself can be qualified.
  Examples are `::X`, `S::x`, `Array<T>::y`, and `::N::A<T>::z`.
- **Qualified Name**:  
  This term is not defined in the standard,
  but we use it to refer to names that undergo _qualified lookup_.
  Specifically, this is a `qualified-id` or an `unqualified-id`
  that is used after an explicit member access operator (`.` or `->`).
  Examples are `S::x`, `this->f`, and `p->A::m`.
  However, just `class_mem` in a context that is implicitly equivalent to `this->class_mem` is **not** a qualified name:
  The member access must be explicit.
- **Unqualified Name**:  
  An `unqualified-id` that is **not** a qualified name.
  This is not a standard term but corresponds to names that undergo
  what the standard calls unqualified lookup.
- **Name**:  
  Either a qualified or an unqualified name.
- **Dependent Name**:  
  A name that depends in some way on a template parameter.
  Typically, a qualified or unqualified name that explicitly contains a template parameter is dependent.
  Furthermore, a qualified name that is qualified by a member access operator (`.` or `->`)
  is typically dependent if the type of the expression on the left of the access operator is type-dependent,
  a concept that is discussed in Section 13.3.6.
  In particular, `b` in `this->b` is generally a dependent name when it appears in a template.
  Finally, a name that is subject to _Argument-Dependent Lookup (ADL)_ (described in Section 13.2),
  such as ident in a call of the form `ident(x, y)` or `+` in the expression `x + y`,
  is a dependent name if and only if any of the argument expressions is type-dependent.
- **Non-dependent Name**:  
  A name that is **not** a dependent name by the above description.


#### 📌 13.2 Looking Up Names


Qualified names are looked up in the scope implied by the qualifying construct.
If that scope is a class, then base classes may also be searched.
However, enclosing scopes of the qualified scope are **not** considered when looking up qualified names.
```c++
int x;

class B 
{
public:
    int i;
};

class D : public B {};

void f(D * pd)
{
    pd->i = 3;  // finds B::i
    D::x = 2;   // ERROR: No x in D and its children scopes
}
```
In contrast, unqualified names are typically looked up in successively more enclosing scopes.  
In member function definitions,
the scope of the class and its base classes is searched before any other enclosing scopes.
This is called _ordinary lookup_.
```c++
extern int count;               // #1

int lookup_example(int count)   // #2
{
    if (count < 0) 
    {
        int count = 1;          // #3
        lookup_example(count);  // refers to #3
    }
    
    return count + ::count;     // refers to #2 + #1
    
}
```


#### 📌 13.2.1 Argument-Dependent Lookup


Unqualified names sometimes undergo _Argument-Dependent Lookup (ADL)_ in addition to ordinary lookup.
In C++98/C++03, this was also called _Koenig Lookup_ (or _Extended Koenig Lookup_)
after Andrew Koenig, who first proposed a variation of this mechanism.


Suppose we need to apply function template `max` to a type defined in another namespace:
```c++
template <typename T>
inline T max(T a, T b)
{
    return b < a ? a : b;
}

namespace BigMath 
{

class BigNumber {};

bool operator<(BigNumber const &, BigNumber const &);

}  // namespace BigMath

using BigMath::BigNumber;

void g(BigNumber const & a, BigNumber const & b)
{
    BigNumber x = ::max(a, b);
}
```
The problem here is that the `max` template is unaware of the `BigMath` namespace.
Ordinary lookup would **not** find the `operator<` applicable to values of type `BigNumber`.
Without ADL, this greatly reduces the applicability of templates in the presence of C++ namespaces.


ADL applies primarily to unqualified names of non-member functions
in a function call or operator invocation.
ADL does **not** happen if ordinary lookup finds
- The name of a member function;
- The name of a variable;
- The name of a type;
- The name of a block-scope function declaration.
  ADL is also **inhibited** if the name of the function to be called
  is enclosed in parentheses.


Otherwise, if the name is followed by a list of argument expressions enclosed in parentheses,
ADL proceeds by looking up the name in _associated namespaces_ and _associated classes_ of the call arguments.
ADL then looks up the name in all the associated namespaces as if
the name had been qualified with each of these namespaces in turn,
**except** that using directives are **ignored**.


The precise definition of the set of _associated namespaces_ and _associated classes_ for a given type
is determined by the following rules:
- **Built-in types**:
  - Empty set.
- **Pointer types** and **array types**:
  - Associated namespaces and classes: That of the underlying type.
- **Enumeration** types:
  - Associated namespace: Namespace in which the enumeration is declared.
- **Class members**:
  - Associated class: The enclosing class
- **Class types** (including union types):
  - Associated classes:
    The type itself, the enclosing class, and any direct and indirect base classes.
  - Associated namespaces:
    The namespaces in which the associated classes are declared.
  - If the class is a class template instance,
    then the types of the template type arguments and the classes and namespaces
    in which the template template arguments are declared are also included.
- **Function types**:
  - Associated namespaces and classes:
    The namespaces and classes associated with all the parameter types
    and those associated with the return type.
- **Pointer-to-member-of-class-`X` types**,
  - Associated namespaces and classes:
    Those associated with `X` in addition to those associated with the type of the member.
    (If it is a pointer-to-member-function type, then the parameter and return types can contribute too.)

```c++
namespace X 
{

template <typename T> 
void f(T);
    
}  // namespace X

namespace N 
{
    using namespace X;
    
    enum E { e1 };
    
    void f(E) 
    {
        std::cout << "N::f(N::E)\n";
    }
}  // namespace N

void f(int)
{
    std::cout << "::f(int)\n";
}

int main()
{
    ::f(N::e1);  // Qualified function name, no ADL
    f(N::e1);    // Ordinary lookup finds ::f(int) and ADL finds N::f(E), 
                 // the latter is preferred. 
                 // Note that using directives are ignored during ADL, 
                 // so X::f(T) is not found. 
}
```
Note that in this example,
the using directive in namespace `N` is **ignored**when ADL is performed.
Hence `X::f()` is **never** even a candidate for the call.

##### 13.2.2 Argument-Dependent Lookup of Friend Declarations

A friend function declaration can be the first declaration of the nominated function.
(And it could also be the only declaration if the friend function is defined in friend declaration.)
If this is the case, then the function is assumed to be declared in the nearest namespace scope
(which may be the global scope) enclosing the class containing the friend declaration.
However, such a friend declaration is **not** directly visible in that scope:
```c++
template <typename T>
class C 
{
    friend void f();
    friend void f(C<T> const &);
};

void g(C<int> * p)
{ 
    f();    // ERROR: f() can not be found
    f(*p);  // OK: f(C<int> const &) visible here via ADL
}
```
If friend declarations were visible in the enclosing namespace,
then instantiating a class template may make visible the declaration of ordinary functions.
This would lead to surprising behavior:
The call `f()` would result in a compilation error
unless an instantiation of the class `C` occurred earlier in the program!


On the other hand, it can be useful to declare (and define) a function in a friend declaration only
(see Section 21.2.1 for a technique that relies on this behavior).
Such a function can be found when the class of which they are a friend
is among the associated classes considered by ADL.


Reconsider our last example.
The call `f()` has **no** associated classes or namespaces because there are no arguments:
It is an invalid call in our example.
However, the call `f(*p)` does have the associated class `C<int>`
and the global namespace is also associated
(because this is the namespace in which the class template `C` is declared).
Therefore, the second friend function declaration could be found
provided the class `C<int>` was actually fully instantiated prior to the call.


To ensure this, it is assumed that a call involving a lookup for friends in associated classes
actually causes the class to be instantiated (if not done already).
Although this was clearly intended by those who wrote the C++ standard,
it is **not** clearly spelled out in the standard.


The ability of Argument-Dependent Lookup to find friend declarations and definition
is sometimes referred to as _friend name injection_.
However, this term is somewhat misleading,
because it is the name of a pre-standard C++ feature
that did in fact "inject" the names of friend declarations into the enclosing scope,
making them visible to normal name lookup.
In our example above, this would mean that both calls would be well-formed.

##### 13.2.3 Injected Class Names

The name of a class is injected into the class scope of that class itself
and is therefore accessible as an unqualified name in that scope.
However, it is **not** accessible as a qualified name
because this is the notation used to denote the constructors.
```c++
int C;

class C 
{
public:
    static int f() 
    {
        return sizeof(C);
    }

private:
    int i[2];
};

int f()
{
    return sizeof(C);
}

int main()
{
    std::cout << "C::f() = " << C::f() << '\n'
              << " ::f() = " << ::f() << '\n';
}
```
The member function `C::f()` returns the size of type `C`,
whereas the function `::f()` returns the size of the variable `C`
(in other words, the size of an `int` object).


Class templates also have injected class names.
However, they're stranger than ordinary injected class names:
They can be followed by template arguments (in which case they are injected class _template_ names),
But, when they are **not** followed by template arguments,
depending on the context, they could represent:
- An alias for the class type being defined
- The template type with its parameters as its arguments
  (specialization arguments for a partial specialization).
```c++
template <template <typename> class TT>
class X {};

template <typename T>
class C
{
    C * a;        // OK: same as "C<T> * a;"
    C<void> & b;  // OK 
    X<C> c;       // OK: C without a template argument list denotes the template C
    X<::C> d;     // OK: ::C is not the injected class name and therefore always denotes the template
};
```
The injected class name for a variadic template has an additional wrinkle:
If the injected class name were directly formed
by using the variadic template's template parameters as the template arguments,
the injected class name would contain template parameter packs that have **not** been expanded.
Therefore, when forming the injected class name for a variadic template,
the template argument that corresponds to a template parameter pack
is a pack expansion whose pattern is that template parameter pack:
```c++
template <int I, typename ... Ts> 
class V 
{
    V * a;         // OK: same as "V<I, Ts...> * a;"
    V<0, void> b;  // OK
};
```

##### 13.2.4 Current Instantiations

The injected class name of a class or class template is effectively
an alias for the type being defined.
Inside a class template or a nested class within a class template,
each template instantiation produces a different type.
This property means that the injected class name refers to
the same instantiation of the class template
rather than some other specialization of that class template.  
The same holds for nested classes of class templates.


Within a class template,
the injected class name or any type that is equivalent to the injected class name
is said to refer to a _current instantiation_.
Types that depend on a template parameter (i.e., _dependent types_)
but do **not** refer to a current instantiation
are said to refer to an _unknown specialization_,
which may be instantiated from the same class template
or some entirely different class template.
```c++
template <typename T> 
class Node
{
    using Type = T;
    
    Node * next;            // Node refers to the current instantiation
    Node<Type> * previous;  // Node<Type> refers to the current instantiation too
    Node<T *> * parent;     // Node<T *> refers to an unknown instantiation
};
```
In the presence of nested classes and class templates.
The injected class names of enclosing classes and class templates (or types equivalent to them)
refer to a current instantiation,
while the names of other nested classes or class templates do **not**:
```c++
template <typename T>
class C
{
    using Type = T;
    
    struct I
    {
        C * c;         // C refers to a current instantiation
        C<Type> * c2;  // C<Type> refers to a current instantiation
        I * i;         // I refers to a current instantiation
    };
    
    struct J
    {
        C * c;         // C refers to a current instantiation
        C<Type> * c2;  // C<Type> refers to a current instantiation
        I * i;         // I refers to an unknown instantiation because I does not enclose J
        J * j;         // J refers to a current instantiation
    };
};
```
When a type refers to a current instantiation,
the contents of that instantiated class are guaranteed to be instantiated
from the class template or nested class thereof that is currently being defined.


This hints another way to determine whether a type `X` within the definition of a class template
refers to a current instantiation or an unknown specialization:
If another programmer can write an explicit specialization (described in detail in Chapter 16)
such that `X` refers to that specialization, then `X` refers to an unknown specialization.


For example, consider the instantiation of the type `C<int>::J` in the context of the above example.
We could not explicitly specialize `C` or `J` for those defined in `J`,
so, the references to `J` and `C<int>` within `J` refer to a current instantiation.
On the other hand, one could write an explicit specialization for `C<int>::I` as follows:
```c++
template <> 
struct C<int>::I 
{
    // definition of the specialization
};
```
Here, the specialization of `C<int>::I` provides a completely different definition
than the one that was visible from the definition of `C<T>::J`,
so the `I` inside the definition of `C<T>::J` refers to an unknown specialization.


#### 📌 13.3 Parsing Templates


Two fundamental activities of compilers for most programming languages are
- _Tokenization_ (also called _Scanning_ or _Lexing_);
- Parsing.


The tokenization process reads the source code as a sequence of characters
and generates a sequence of tokens from it.
A parser will then find known patterns in the token sequence
by recursively reducing tokens or previously found patterns into grammar.

##### 13.3.1 Context Sensitivity in Non-templates

Tokenizing is easier than parsing.
Parsing is a subject for which a solid theory has been developed,
and many useful languages are not hard to parse using this theory.
However, the theory works best for _Context-Free Grammar/Languages (CFG)_,
and we have already noted that C++ is _context sensitive_.


To handle this, a C++ compiler will couple a symbol table to the tokenizer and parser:
When a declaration is parsed, it is entered in the symbol table.
When the tokenizer finds an identifier,
it looks it up and annotates the resulting token if it finds a type.


For example, if the C++ compiler sees
```c++
x *
```
the tokenizer looks up `x`.
If it finds a type,
the parser receives from the tokenizer
```
identifier, type, x, symbol, *
```
concludes that a declaration has started.
However, if `x` is **not** found to be a type,
then the parser receives from the tokenizer
```c++
identifier, non-type, x, symbol, *
```
and the construct can be parsed validly only as a multiplication.
The details of these principles are dependent on the particular implementations.


Another example of context sensitivity:
```c++
X<1>(0)
```
If `X` is the name of a class template,
then the previous expression casts the integer `0` to the type `X<1>` generated from that template.
If X is **not** a template, then the previous expression is equivalent to
```c++
(X < 1) > 0
```
In other words, `X` is compared with `1`,
and the result of that comparison is then compared with `0`.
Although code like this is rarely used, it is valid C++ (and valid C, for that matter).
A C++ parser will therefore look up names appearing before a `<` and treat the `<` as an angle bracket
only if the name is known to be that of a template.
Otherwise, the `<` is treated as an ordinary less-than operator.


This form of context sensitivity is an unfortunate consequence of
having chosen angle brackets to delimit template argument lists.
Here is another such consequence:
```c++
template <bool B>
class Invert 
{
public:
    static bool const result = !B;
};

void g()
{
    bool test = Invert<(1 > 0)>::result;  // parentheses required!
}
```
If the parentheses in `Invert<(1 > 0)>` were omitted,
the greater-than symbol would be mistaken for the closing of the template argument list.
This would make the code invalid because the compiler would read it to be equivalent to
```c++
((Invert<1>))0>::result
```
Note the double parentheses to avoid parsing `(Invert<1>)0` as a cast operation,
yet another source of syntactic ambiguity.


The tokenizer isn't spared problems with the angle-bracket notation either.
For example, in
```
List<List<int>> a;
~~~~~~~~~~~~~^~~~~  no space between right angle brackets
```
the two `>` characters combine into a right-shift token `>>`
and hence are **never** treated as two separate tokens by the tokenizer.
This is a consequence of the maximum munch tokenization principle:
A C++ implementation must collect as many consecutive characters as possible into a token.
Specific exceptions were introduced to address
tokenization issues described in this section.


As mentioned in Section, since C++11, the C++ standard specifically calls out this case,
where a nested template-id is closed by a right-shift token `>>`,
and, within the parser, treats the right shift as being equivalent to
two separate right angle brackets `>` and `>` to close two template-ids at once.


The 1998 and 2003 versions of the C++ standard did **not** support this "angle bracket hack".
However, the need to introduce a space between the two consecutive right angle brackets
was such a common stumbling block for beginning template users
that the committee decided to codify this hack in the 2011 standard.


This change silently changes the meaning of some admittedly contrived programs:
```c++
template <int I> 
struct X 
{
    static int const c = 2;
};

template <> 
struct X<0> 
{
    typedef int c;
};

template <typename T> 
struct Y
{
    static int const c = 3;
};

static int const c = 4;

int main()
{
    std::cout << (Y<X<1> >::c >::c>::c) << ' ';
              << (Y<X< 1>>::c >::c>::c) << '\n';
}
```
This is a valid C++98 program that outputs `0 3`
It is also a valid C++11 program,
but there the angle bracket hack makes the two parenthesized expressions equivalent,
and the output is `0 0`.
Some compilers that provide a C++98 or C++03 mode keep the C++11 behavior in those modes
and thus print `0 0` even when formally compiling C++98/C++03 code.


A similar problem existed because of the existence of the digraph `<:`
as an alternative for the source character `[` (which is not available on some traditional keyboards):
```c++
template <typename T> 
struct G {};

struct S;
G<::S> gs;  // valid since C++11, but an error before that
```
Before C++11, that last line of code was equivalent to `G[:S> gs;`,
which is clearly invalid.
Another "lexical hack" was added to address that problem:
When a compiler sees the characters `<::` not immediately followed by `:` or `>`,
the leading pair of characters `<:` is **not** treated as a digraph token equivalent to `[`.
This is therefore an exception to the aforementioned maximum munch principle.


This _digraph hack_ can make previously valid (but somewhat contrived) programs invalid:
```c++
#define F(X) X ## :

int a[] = {1, 2, 3}; 
int i = 1;

int n = a F(<::)i];  // valid in C++98/C++03, but not in C++11
```

##### 13.3.2 Dependent Names of Types And `typename` Prefix

The problem with names in templates is that they can **not** always be sufficiently classified.
In particular, one template can **not** look into another template
because the contents of that other template can be made invalid by an explicit specialization.
The following contrived example illustrates this:
```c++
template <typename T>
class Trap
{
public:
    enum { x };          // #1 x is not a type here
};

template <typename T>
class Victim
{
public:
    void poof()
    {
        Trap<T>::x * y;  // #2 declaration or multiplication?
    }

    int y;
};

template <>
class Trap<void>
{
public:
    using x = int;       // #3 x is a type here
};

void boom(Victim<void> & bomb)
{
    bomb.poof();
}
```
As the compiler is parsing line `#2`,
it must decide whether it is seeing a declaration or a multiplication.
This decision in turn depends on whether `Trap<T>::x` is a type name.


`Trap<T>` is a _dependent name_ because it depends on the template parameter `T`.
Moreover, `Trap<T>` refers to an _unknown specialization_,
which means that the compiler can **not** safely look inside the primary template
to determine whether the name `Trap<T>::x` is a type or not.
Actually, another specification `Trap<void>` just contradicts with the primary template
on what `Trap<T>::x` is.


However, as illustrated by the example,
name lookup into an unknown specialization is still a problem.
The language definition resolves this problem by specifying that
in general a dependent qualified name does **not** denote a type
unless that name is prefixed with the keyword `typename`.
If it turns out that the name is not the name of a type after template argument substitution,
the program is invalid and your C++ compiler should complain at instantiation time.
Note that this use of `typename` differs from the use to denote template type parameters.
Unlike type parameters, you can **not** equivalently replace `typename` with `class`.


The `typename` prefix to a name is _required_ when the name satisfies all of the following conditions:
1. It is qualified and **not** itself followed by `::` to form a more qualified name.
2. It is **not** part of an `elaborated-type-specifier`,
   i.e., a type name that starts with one of the keywords `class`, `struct`, `union`, or `enum`.
3. It is **not** used in a _list of base class specifications_
   or in a _list of member initializers_ introducing a constructor definition.
   Syntactically, only type names are permitted within these contexts,
   so a qualified name is always assumed to name a type.
4. It is dependent on a template parameter.
5. It is a member of an unknown specialization,
   meaning that the type named by the qualifier refers to an unknown specialization.


Furthermore, the `typename` prefix is _**not** allowed_
unless at least the first two previous conditions hold.

##### 13.3.3 Dependent Names of Templates And `template` Prefix

A problem very similar to the one encountered in the previous section occurs
when a name of a template is dependent.
In general, a C++ compiler is required to treat a `<` following the name of a template
as the beginning of a template argument list.
Otherwise, it is a less-than operator.
As is the case with type names, a compiler has to assume that a dependent name does **not** refer to a template
unless the programmer provides extra information using the keyword `template`:
```c++
template <typename T>
class Shell
{
public:
    template <int N>
    class In
    {
    public:
        template <int M>
        class Deep
        {
        public:
            virtual void f() {}
        };
    };
};

template <typename T, int N>
class Weird
{
public:
    void case1(typename Shell<T>::template In<N>::template Deep<N> * p)
    {
        p->template Deep<N>::f();  // inhibit virtual call
    }

    void case2(typename Shell<T>::template In<N>::template Deep<N> & p)
    {
        p.template Deep<N>::f();   // inhibit virtual call
    }
};
```
This example shows how all the operators that can qualify a name (`::`, `->`, and `.`)
may need to be followed by the keyword `template`.

Specifically, this is the case whenever
the type of the name or expression preceding the qualifying operator
is dependent on a template parameter and refers to an unknown specialization,
and the name that follows the operator is a `template-id`
(in other words, a template name followed by template arguments in angle brackets).
```c++
dependent-name-to-unknown-specification ::/->/. template template-id
```
For example, in the expression
```c++
p.template Deep<N>::f()
```
the type of `p` depends on the template parameter `T`.
Consequently, a C++ compiler can `not` look up `Deep` to see if it is a template,
and we must explicitly indicate that `Deep` is the name of a template by inserting the prefix `template`.


Without the `template` prefix,
`Deep` will be parsed as non-template member of `p`,
and `p.Deep<N>::f()` will be parsed as `((p.Deep) < N) > f()`.
Note also that this may need to happen multiple times within a qualified name because qualifiers
themselves may be qualified with a dependent qualifier.


If the keyword `template` is omitted in cases such as these,
the opening and closing angle brackets are parsed as less-than and greater-than operators.
As with the `typename` keyword, one can safely add the `template` prefix
to indicate that the following name is a template-id,
even if the `template` prefix is **not** strictly needed.

##### 13.3.4 Dependent Names in Using Declarations

Using declarations can bring in names from two places: namespaces and classes.
The namespace case is not relevant in this context
because there are **no** such things as _namespace templates_.
Using declarations that bring in names from classes, on the other hand,
can bring in names only from a base class to a derived class.
Such using declarations behave like "symbolic links" or "shortcuts"
in the derived class to the base declaration,
thereby allowing the members of the derived class to access the nominated name
as if it were actually a member declared in that derived class.
```c++
class BX 
{
public:
    void f(int);
    void f(char const *);
    void g();
};

class DX : private BX 
{
public:
    using BX::f;
};
```
By now you can probably perceive the problem when a using declaration brings in a name from a dependent class.
Although we know about the name, we **don't** know whether it's the name of a type, a template, or something else:
```c++
template <typename T>
class BXT 
{
public:
    using Mystery = T;
    
    template <typename U>
    struct Magic {};
};

template <typename T>
class DXTT : private BXT<T> 
{
public:
    using typename BXT<T>::Mystery;
    
    Mystery * p;  // would be a syntax error without the earlier typename
};
```
Again, if we want a dependent name to be brought in by a using declaration to denote a type,
we must explicitly say so by inserting the keyword `typename`.
Strangely, the C++ standard does **not** provide for a similar mechanism to mark such dependent names as templates.
```c++
template <typename T>
class DXTM : private BXT<T>
{
public:
    using BXT<T>::template Magic;  // ERROR: not standard
    Magic<T> * plink;              // SYNTAX ERROR: Magic is not a known template
};
```
The standardization committee has not been inclined to address this issue.
However, C++11 alias templates do provide a partial workaround:
```c++
template <typename T>
class DXTM : private BXT<T>
{
public:
    template <typename U> 
    using Magic = typename BXT<T>::template Magic<U>;  // alias template
    
    Magic<int> * plink;                                // OK
}
```
This is a little unwieldy, but it achieves the desired effect for the case of class templates.
The case of function templates (arguably less common) remains **unaddressed**, unfortunately.

##### 13.3.5 ADL and Explicit Template Arguments

```c++
namespace N 
{

class X {};

template <int I> 
void select(X *) {}

}  // namespace N

void g(N::X * xp)
{
    select<3>(xp);
}
```
In this example, we may expect that the template `select` is found through ADL in the call `select<3>(xp)`.
However, this is **not** the case because
a compiler can **not** decide that `xp` is a function call argument
until it has decided that `<3>` is a template argument list.
Furthermore, a compiler can **not** decide that `<3>` is a template argument list
until it has found `select` to be a template.
Because this chicken-and-egg problem can not be resolved,
the expression is parsed as `(select < 3) > (xp)`, which makes no sense.


This example may give the impression that ADL is disabled for `template-id`s, but it is **not**.
The code can be fixed by introducing a declaration of function template `select` that is visible at the call:
```c++
void g(N::X * xp)
{
    template <typename T> 
    void select();
    
    select<3>(xp);
}
```
Even though it **doesn't** make any sense for the call `select<3>(xp)`,
the presence of this function template ensures that `select<3>` will be parsed as a `template-id`.
ADL will then find the function template `N::select`, and the call will succeed.

##### 13.3.6 Dependent Expressions

Like names, expressions themselves can be dependent on template parameters.
An expression that depends on a template parameter can behave differently from one instantiation to the next.
For example, selecting a different overloaded function or producing a different type or constant value.
In contrast, expressions that do not depend on a template parameter provide the same behavior in all instantiations.


An expression can be dependent on a template parameter in several ways:
- **Type-Dependent Expressions**: Type of the expression itself can vary from one instantiation to the next;
- **Value-Dependent Expressions**: Produce different constant values from one instantiation to the next.


_Type-dependent expressions_ are those whose type can vary from one instantiation to the next.
For example, an expression that refers to a function parameter whose type is that of a template parameter:
```c++
template <typename T> 
void typeDependent1(T x)
{
    // the expression type-dependent, 
    // because the type of x can vary
    x;
}
```
Expressions that have type-dependent subexpressions are generally type-dependent themselves.
For example, calling a function `f` with the argument `x`:
```c++
template <typename T> 
void typeDependent1(T x)
{
    // the expression type-dependent, 
    // because the type of x can vary
    f(x);
}
```
Here, note that type of `f(x)` can vary from one instantiation to the next
both because `f` might resolve to a template whose result type depends on the argument type
and because _two-phase lookup_ (discussed in Section 14.3.1) might find
completely different functions named `f` in different instantiations.


Expressions that produce different constant values from one instantiation to the next
are called _value-dependent expressions_.
The simplest of which are those that refer to a non-type template parameter of non-dependent type:
```c++
template <int N> 
void valueDependent1()
{
    // the expression is value-dependent but not type-dependent,
    // because N has a fixed type but a varying constant value
    N;
}
```
Like type-dependent expressions, an expression is generally value-dependent
if it is composed of other value-dependent expressions,
so `N + N` or `f(N)` are also value-dependent expressions.


Some operations, such as `sizeof`, have a known result type,
so they can turn a type-dependent operand into a value-dependent expression that is `not` type-dependent:
```c++
template <typename T> 
void valueDependent2(T x)
{
    sizeof(x);  // the expression is value-dependent but not type-dependent
}
```
The `sizeof` operation always produces a value of type `std::size_t` regardless of its input,
so a `sizeof` expression is **never** type-dependent, even if its subexpression is type-dependent.
However, the resulting constant value will vary from one instantiation to the next,
so `sizeof(x)` is a value-dependent expression.


What if we apply `sizeof` on a value-dependent expression?
```c++
template <typename T> 
void maybeDependent(T const & x)
{
    sizeof(sizeof(x));
}
```
Here, the inner `sizeof` expression is value-dependent, as noted above.
However, the outer sizeof expression always computes the size of a `std::size_t`,
so both its type and constant value are consistent across all instantiations of the template,
despite the innermost expression `(x)` being type-dependent.
Any expression that involves a template parameter is an _instantiation-dependent_ expression,
even if both its type and constant value are invariant across valid instantiations.
However, an instantiation-dependent expression may turn out to be invalid when instantiated.
For example, instantiating `maybeDependent` with an incomplete class type will trigger an error,
because sizeof can **not** be applied to such types.


Type-dependence, value-dependence, and instantiation-dependence
can be thought of as a series of increasingly more inclusive classifications of expressions.
Any type-dependent expression is also considered to be value-dependent,
because an expression whose type that varies from one instantiation to the next
will naturally have its constant value vary from one instantiation to the next.
Similarly, an expression whose type or value varies from one instantiation to the next
depends on a template parameter in some way,
so both type-dependent expressions and value-dependent expressions are instantiation-dependent.


As one proceeds from the innermost context (type-dependent expressions) to the outermost context,
more of the behavior of the template is determined when the template is parsed
and therefore can **not** vary from one instantiation to the next.
For example, consider the call `f(x)`:
If `x` is type-dependent,
then `f` is a dependent name that is subject to two-phase lookup (Section 14.3.1),
whereas if `x` is value-dependent but not type-dependent,
`f` is a non-dependent name for which name lookup can be completely determined
at the time that the template is parsed.

##### 13.3.7 Compiler Errors

A C++ compiler is permitted (but **not** required!) to diagnose errors at the time the template is parsed
when all of the instantiations of the template would produce that error.
Let's expand on the `f(x)` example from the previous section to explore this further:
```c++
void f() {}

template <int x> 
void nondependentCall()
{
    // x is value-dependent, so f is non-dependent. 
    // This call will never succeed
    f(x);
}
```
Here, the call `f(x)` will produce an error in every instantiation
because `f` is a non-dependent name and the only visible `f` accepts zero arguments, not one.
A C++ compiler can produce an error when parsing the template
or may wait until the first template instantiation:
Commonly used compilers differ even on this simple example.
One can construct similar examples with expressions that are instantiation-dependent but not value-dependent:
```c++
template <int N>
void instantiationDependentBound()
{
    constexpr int x = sizeof(N);
    constexpr int y = sizeof(N) + 1;
    int array[x - y];  // negative size in all instantiations
}
```


#### 📌 13.4 Inheritance and Class Templates


Class templates can inherit or be inherited from.
For many purposes, there is nothing significantly different between the template and non-template scenarios.
However, there is one important subtlety
when deriving a class template from a base class referred to by a dependent name.
Let's first look at the somewhat simpler case of non-dependent base classes.

##### 13.4.1 Non-dependent Base Classes

In a class template, a non-dependent base class is one with a complete type
that can be determined without knowing the template arguments.
In other words, the name of this base is denoted using a non-dependent name:
```c++
template <typename>
class Base
{
public:
    using T = int;
    
    int basefield;
};

// not a template case really
class D1 : public Base<Base<void>>
{
public:
    void f()
    {
        basefield = 3;
    }
};

// usual access to non-dependent base
template <typename T>
class D2 : public Base<double>
{
public:
    void f()
    {
        // usual access to inherited member
        basefield = 7;
    }
    
    // T is Base<double>::T (aka int), 
    // not the template parameter!
    T strange;
};
```
Non-dependent bases in templates behave very much like bases in ordinary non-template classes,
but there is a slightly unfortunate surprise:
When an unqualified name is looked up in the templated derivation,
the non-dependent bases are considered _before_ the list of template parameters.
This means that in the previous example,
the member `strange` of the class template `D2` always has the type `T`
corresponding to `Base<double>::T` (aka `int`).
For example, the following function is **not** valid C++ (assuming the previous declarations):
```c++
void g(D2<int *> & d2, int * p)
{
    // ERROR: type mismatch!
    d2.strange = p;
}
```

##### 13.4.2 Dependent Base Classes

In the previous example, the base class is fully determined.
It does **not** depend on a template parameter.
This implies that a C++ compiler can look up non-dependent names in those base classes
_as soon as_ the template definition is seen.
An alternative (**not** allowed by the C++ standard) would consist in delaying the lookup of such names
until the template is instantiated.
The disadvantage of this alternative approach is that it also delays
any error messages resulting from missing symbols until instantiation.
Hence, the C++ standard specifies that a non-dependent name appearing in a template
is looked up _as soon as_ it is encountered:
```c++
template <typename>
class Base
{
public:
    using T = int;

    int basefield;
};

template <typename T>
class DD : public Base<T>  // dependent base
{
public:
    void f()
    {
        basefield = 0;     // #1 ERROR: use of undeclared identifier basefield
    }
};

template <>
class Base<bool>           // explicit specialization
{
public:
    enum
    {
        basefield = 42     // #2
    };
};

void g(DD<bool> & d)
{
    d.f();                 // #3
}
```
At point `#1` we find our reference to a non-dependent name `basefield`:
It must be looked up _right away_.
Suppose we look it up in the template `Base` and bind it to the `int` member that we find therein.
However, shortly after this we override the generic definition of `Base` with an explicit specialization.
As it happens, this specialization changes the meaning of the `basefield` member
to which we already committed!
So, when we instantiate the definition of `DD::f` at point `#3`,
we find that we too eagerly bound the non-dependent name at point `#1`.
There is **no** modifiable `basefield` in `DD<bool>` that was specialized at point `#2`,
and an error message should have been issued.


To circumvent this problem, standard C++ says that
non-dependent names are **not** looked up in dependent base classes.
(But they are still looked up as soon as they are encountered.)
This is part of the two-phase lookup rules that distinguish
between a first phase when template definitions are first seen
and a second phase when templates are instantiated (see Section 14.3.1).


So, a standard C++ compiler will emit a diagnostic at point `#1`.
To correct the code, it suffices to make the name `basefield` dependent
because dependent names can be looked up only at the time of instantiation,
and at that time the concrete base instance that must be explored will be known.
For example, at point `#3`, the compiler will know that
the base class of `DD<bool>` is `Base<bool>`
and that this has been explicitly specialized by the programmer.
In this case, our preferred way to make the name dependent is as follows:
```c++
// Variation 1:
template <typename T>
class DD1 : public Base<T> 
{
public:
    // lookup delayed
    void f() { this->basefield = 0; }
};
```
An alternative consists in introducing a dependency using a qualified name:
```c++
// Variation 2:
template <typename T>
class DD2 : public Base<T> 
{
public:
    void f() { Base<T>::basefield = 0; }
};
```
Care must be taken with this solution,
because if the unqualified non-dependent name is used to form a virtual function call,
then the qualification **inhibits** the virtual call mechanism and the meaning of the program changes.
Nonetheless, there are situations when the first variation can **not** be used and this alternative is appropriate:
```c++
template <typename T>
class B
{
public:
    enum E
    {
        e1 = 6, 
        e2 = 28, 
        e3 = 496
    };

    virtual void zero(E e = e1) {}

    virtual void one(E & e) {}
};

template <typename T>
class D : public B<T>
{
public:
    void f()
    {
        typename D<T>::E e;  // this->E would not be valid syntax
        this->zero();        // D<T>::zero() would inhibit virtuality
        one(e);              // one is dependent because its argument is dependent
    }
};
```
Note how we used `D<T>::E` instead of `B<T>::E` in this example.
In this case, either one works.
In _multiple-inheritance_ cases, however, we may **not** know which base class provides the desired member
(in which case using the derived class for qualification works)
or multiple base classes may declare the same name
(in which case we may have to use a specific base class name for disambiguation).


Note that the name one in the call `one(e)` is dependent on the template parameter
simply because the type of one of the call's explicit arguments is dependent.
Implicitly-used default arguments with a type that depends on a template parameter do **not** count
because the compiler can **not** verify this until it already has decided the lookup (a chicken-and-egg problem).
To avoid subtlety, we prefer to use the `this->` prefix in all situations that allow it,
even for non-template code.
If you find that the repeated qualifications are cluttering up your code,
you can bring a name from a dependent base class in the derived class once and for all:
```c++
// Variation 3:
template <typename T>
class DD3 : public Base<T> 
{
public:
    using Base<T>::basefield;    // #1 dependent name now in scope
    void f() { basefield = 0; }  // #2 fine
};
```
The lookup at point `#2` succeeds and finds the using declaration of point `#1`.
However, the using declaration is **not** verified until instantiation time and our goal is achieved.
There are some subtle limitations to this scheme.
For example, if multiple bases are derived from,
the programmer must select exactly which one contains the desired member.


When searching for a qualified name within the current instantiation,
the C++ standard specifies that name lookup
first search in the current instantiation and in all non-dependent bases,
similar to the way it performs unqualified lookup for that name.
If any name is found, then the qualified name refers to a member of a current instantiation
and will **not** be a dependent name.
However, the lookup is nonetheless repeated when the template is instantiated,
and if a different result is produced in that context, the program is ill-formed.
If no such name is found, and the class has any dependent bases,
then the qualified name refers to a member of an unknown specialization.
For example:
```c++
class NonDep
{
public:
    using Type = int;
};

template <typename T>
class Dep
{
public:
    using OtherType = T;
};

template <typename T>
class DepBase : public NonDep, public Dep<T>
{
public:
    void f()
    {
        // Finds NonDep::Type. 
        // typename keyword is optional
        typename DepBase<T>::Type t;

        // Finds nothing. 
        // DepBase<T>::OtherType is a member of an unknown specialization
        typename DepBase<T>::OtherType * ot;
    }
};
```






### 🎯 Chapter 14 Instantiation


Template instantiation is the process that
generates types, functions, and variables from generic template definitions.
The term _instantiation_ is sometimes also used to refer to the creation of objects from types.
In this book, however, it always refers to template instantiation.


#### 📌 14.1 On-Demand Instantiation


When a C++ compiler encounters the use of a template specialization,
it will create that specialization by substituting the required arguments for the template parameters.
The term _specialization_ is used in the general sense of an entity that is a specific instance of a template.
It does **not** refer to the _explicit specialization_ mechanism described in Chapter 16.


This is done automatically and requires **no** direction from the client code
(or from the template definition, for that matter).
This _on-demand instantiation_ feature is sometimes also called
_implicit instantiation_ or _automatic instantiation_.


On-demand instantiation implies that the compiler often needs access to the full definition
(in other words, not just the declaration)
of the template and some of its members at the point of use.
Consider the following tiny source code file:
```c++
// #1 declaration only
template <typename T>
class C;

// #2 fine: definition of C<int>
C<int> * p = nullptr;


template <typename T>
class C
{
public:
    void f();       // #3 member declaration
};                  // #4 class template definition completed

void g(C<int> & c)  // #5 use class template declaration only
{
    // #6 use class template definition,
    // will need definition of C::f in this translation unit
    c.f();    
}

// required definition due to #6
template <typename T>
void C<T>::f() {}
```
At point `#1` in the source code, only the declaration of the template is available,
**not** the definition (such a declaration is sometimes called a _forward declaration_).
As is the case with ordinary classes,
we do **not** need the definition of a class template to be visible to declare pointers or references to this type,
as was done at point `#2`.
For example, the type of the parameter of function `g`
does **not** require the full definition of the template `C`.
However, as soon as a component needs to know the size of a template specialization
or if it accesses a member of such a specialization,
the entire class template definition is required to be visible.
This explains why at point `#6` in the source code, the class template definition must be seen.
Otherwise, the compiler can **not** verify that the member exists and is accessible
(not private or protected).
Furthermore, the member function definition is needed too,
since the call at point `#6` requires `C<int>::f` to exist.


Here is another expression that needs the instantiation:
```c++
C<void> * p = new C<void>;
```
In this case, instantiation is needed so that the compiler can determine the size of `C<void>`,
which the new-expression needs to determine how much storage to allocate.
You might observe that for this particular template,
the type of the argument substituted for `T`
will **not** influence the size of the template
because in any case, `C<X>` is an empty class
(without explicit specializations, without data member in the primary template).
However, a compiler is **not** required to avoid
instantiation by analyzing the template definition
(and all compilers do perform the instantiation in practice).
Furthermore, instantiation is also needed in this example
to determine whether `C<void>` has an accessible default constructor
and to ensure `C<void>` does not declare member `operator new` or `operator delete`.


The need to access a member of a class template is **not**
always very explicitly visible in the source code.
For example, C++ overload resolution requires visibility
into class types for parameters of candidate functions:
```c++
template <typename T>
class C
{
public:
    // a constructor that can be called with a single parameter
    // may be used for implicit conversions
    C(int) {}
};

void candidate(C<double>) {}  // #1

void candidate(int) {}        // #2

int main()
{
    // both previous function declarations can be called
    candidate(42);
}
```
The call `candidate(42)` will resolve to the overloaded declaration at point `#2`.
However, the declaration at point `#1` could also be instantiated
to check whether it is a viable candidate for the call
(it is in this case because the one-argument constructor
can implicitly convert `42` to an rvalue of type `C<double>`).
Note that the compiler is allowed (but **not** required) to perform this instantiation
if it can resolve the call **without** it
(as could be the case in this example because an implicit conversion would not be selected over an exact match).
Note also that the instantiation of `C<double>` could trigger an error, which may be surprising.


#### 📌 14.2 Lazy Instantiation


Requirements on templates are not fundamentally different from that when using non-template classes.  
Many uses require a class type to be complete (see Section 10.3.1).
For the template case, the compiler will instantiate the template.
But a compiler should be "lazy" when instantiating templates,
only instantiating parts that are needed right away.

##### 14.2.1 Partial and Full Instantiation

The compiler sometimes **doesn't** need to substitute the complete definition of a template.
```c++
template <typename T> 
T f (T p) 
{ 
    return 2 * p; 
}

decltype(f(2)) x = 2;
```
In this example, the type indicated by `decltype(f(2))` does **not** require
the complete instantiation of the function template `f`.
A compiler is therefore only permitted to substitute the _declaration_ of `f`,
but **not** its "body".
This is sometimes called _partial instantiation_.


Similarly, if an instance of a class template is referred to
without the need for that instance to be a complete type,
the compiler should **not** perform a complete instantiation of that class template instance:
```c++
template <typename T> 
class Q 
{
    using Type = typename T::Type;
};

Q<int> * p = 0;  // OK: the body of Q<int> is not substituted
```
Here, the full instantiation of `Q<int>` would trigger an error,
because `T::Type` doesn't make sense when `T` is `int`.
But because `Q<int>` need **not** be complete in this example,
**no** full instantiation is performed and the code is okay (albeit suspicious).


Variable templates also have a "full" vs. "partial" instantiation distinction:
```c++
template <typename T> 
T v = T::default_value();

decltype(v<int>) s;  // OK: initializer of v<int> not instantiated
```
A full instantiation of `v<int>` would elicit an error,
but that is **not** needed if we only need the type of the variable template instance.


Alias templates do **not** have this distinction:
There are **no** two ways of substituting them.


In C++, when speaking about "template instantiation"
without being specific about full or partial instantiation, the former is intended.
That is, instantiation is full instantiation by default.

##### 14.2.2 Instantiated Components

When a class template is implicitly (fully) instantiated,
each declaration of its members is instantiated as well,
but the corresponding definitions are **not** (i.e., the member are _partially_ instantiated).
There are a few exceptions to this.
1. **Anonymous unions**:
   The members of that union's definition are also instantiated.
   Anonymous unions are always special in this way:
   Their members can be considered to be members of the enclosing class.
   An anonymous union is primarily a construct that says that some class members share the same storage.
2. **Virtual member functions**:
   Their definitions _may or may not_ be instantiated as a result of instantiating a class template.
   Many implementations will instantiate the definition
   because the internal structure that enables the virtual call mechanism
   requires the virtual functions actually to exist as linkable entities.


Default function call arguments are considered separately when instantiating templates.
Specifically, they are **not** instantiated unless there is a call to that function (or member function)
that actually makes use of the default argument.
If, on the other hand, the function is called only with explicit arguments that override the default,
then the default arguments are **not** instantiated.


Similarly, exception specifications and default member initializers are not instantiated unless they are needed.


Let's put together some examples that illustrate some of these principles:
```c++
template <typename T>
class Safe {};

template <int N>
class Danger
{
    int arr[N];                    // OK here, although would fail for N <= 0
};

template <typename T, int N>
class Tricky
{
public:
    void noBodyHere(Safe<T> = 3);  // OK until usage of default value results in an error

    void inclass()
    {
        Danger<N> noBoomYet;       // OK until inclass is used with N <= 0
    }

    struct Nested 
    {
        Danger<N> pfew;            // OK until Nested is used with N <= 0
    };  
    
    union                          // due anonymous union:
    {
        Danger<N> anonymous;       // OK until Tricky is instantiated with N <= 0
        int align;
    };

    void unsafe(T (* p)[N]);       // OK until Tricky is instantiated with N <= 0

    void error()
    {
        Danger<-1> boom;           // Always ERROR (which not all compilers detect)
    }
};
```
A standard C++ compiler will examine these template definitions
to check the syntax and general semantic constraints.
While doing so, it will "assume the best" when checking constraints involving template parameters.
For example, the parameter `N` in the member `Danger::arr` could be zero or negative (which would be invalid),
but it is assumed that this isn't the case.
GCC allow zero-length arrays as extensions and may therefore accept this code even when `N` ends up being `0`.


The definitions of `inclass`, struct `Nested`, and the anonymous union are thus not a problem.
For the same reason, the declaration of the member `unsafe(T (* p)[N])` is not a problem,
as long as `N` is an unsubstituted template parameter.


The default argument specification (`Safe<T> = 3`) on the declaration of the member `noBodyHere`
is suspicious because the template `Safe` isn't initializable with an integer,
but the assumption is that either the default argument won't actually be needed for the generic definition of `Safe<T>`
or that `Safe<T>` will be specialized (see Chapter 16) to enable initialization with an integer value.
However, the definition of the member function `error` is an error even when the template is not instantiated,
because the use of `Danger<-1>` requires a complete definition of the class `Danger<-1>`,
and generating that class runs into an attempt to define an array with negative size.
Interestingly, while the standard clearly states that this code is invalid,
it also allows compilers **not** to diagnose the error when the template instance is not actually used.
That is, since `Tricky<T, N>::error` is **not** used for any concrete `T` and `N`,
a compiler is **not** required to issue an error for this case.
For example, GCC and Visual C++ do **not** diagnose this error at the time of this writing.


Now let's analyze what happens when we add the following definition:
```c++
Tricky<int, -1> inst;
```
This causes the compiler to (fully) instantiate `Tricky<int, -1>`
by substituting `int` for `T` and `-1` for `N` in the definition of template `Tricky`.
Not all the member definitions will be needed,
but the default constructor and the destructor (both implicitly declared in this case) are definitely called,
and hence their definitions must be available somehow
(which is the case in our example, since they are implicitly generated).
As explained above, the members of `Tricky<int, -1>` are partially instantiated
(i.e., their declarations are substituted):
That process can potentially result in errors.
For example, the declaration of `unsafe(T (* p)[N])` creates an array type with a negative of number elements,
and that is an error.
Similarly, the member anonymous now triggers an error, because type `Danger<-1>` can not be completed.
In contrast, the definitions of the members `inclass` and `struct Nested` are not yet instantiated,
and thus no errors occur from their need for the complete type `Danger<-1>`
(which contains an invalid array definition).


When instantiating a template, the definitions of virtual members should also be provided.
Otherwise, linker errors are likely to occur:
```c++
template <typename T>
class VirtualClass
{
public:
    virtual ~VirtualClass() {}

    virtual T vmem();  // Likely ERROR if instantiated without definition
};

int main()
{
    VirtualClass<int> inst;
}
```
Finally, a note about `operator->`. Consider:
```c++
template <typename T>
class C 
{
public:
    T operator->();
};
```
Normally, `operator->` must return a pointer type or another class type to which `operator->` applies.
This suggests that the completion of `C<int>` triggers an error,
because it declares a return type of `int` for `operator->`.
However, certain natural class template definitions trigger these kinds of definitions,
typical examples are smart pointer templates (e.g., `std::unique_ptr<T>`).
Thus, the language rule is more flexible:
Only when a user-defined `operator->` _is actually selected by overload resolution_,
it is required to return a type to which another `operator->` applies.
This is true even outside templates (although the relaxed behavior is less useful in those contexts).
Hence, the declaration here triggers **no** error, even though `int` is substituted for the return type.


#### 📌 14.3 The C++ Instantiation Model

Template instantiation is the process of obtaining a regular type, function, or variable
from a corresponding template entity by appropriately substituting the template parameters.

##### 14.3.1 Two-Phase Lookup

In Chapter 13 we saw that dependent names can **not** be resolved when parsing templates.
Instead, they are looked up again at the point of instantiation.
Non-dependent names, however, are looked up early so that many errors can be diagnosed when the template is first seen.
This leads to the concept of _two-phase lookup_ (_two-stage lookup_, _two-phase name lookup_):
The first phase is the parsing of a template, and the second phase is its instantiation:
1. During the first phase, while _parsing_ a template:
  - Non-dependent names are looked up using both the _ordinary lookup rules_ and _Argument-Dependent Lookup (ADL)_;
    - Non-dependent names are **not** looked up in dependent base classes (Section 13.4.2).
  - Dependent unqualified names (all unqualified names are dependent) are looked up using the ordinary lookup rules,
    but the result of the lookup is **not** considered complete
    until an additional lookup is performed in the second phase (when the template is instantiated).
2. During the second phase, while _instantiating_ a template at a point called the _Point of Instantiation (POI)_:
  - Dependent qualified names are looked up
    with the template parameters replaced with the template arguments for that specific instantiation;
  - An additional ADL is performed for the unqualified dependent names
    that were looked up using ordinary lookup in the first phase.


For unqualified names (all unqualified names are dependent),
the initial ordinary lookup (while not complete) is used to decide whether the name is a template:
```c++
namespace N 
{

template <typename> void g() {}
enum E { e };

}  // namespace N

template <typename> void f() {}

template <typename T> 
void h(T P) 
{
    f<int>(p);  // #1
    g<int>(p);  // #2 ERROR
}

int main() 
{
    h(N::e);
}
```
In line `#1`, when seeing the name `f` followed by a `<`,
the compiler has to decide whether that `<` is an angle bracket or a less-than sign.
That depends on whether `f` is known to be the name of a template or not.
In this case, ordinary lookup finds the declaration of `f`,
which is indeed a template, and so parsing succeeds with angle brackets.


Line `#2` produces an error because **no** template `g` is found using ordinary lookup.
The `<` is thus treated as a less-than sign, which is a syntax error in this example.
If we could get past this issue, we'd eventually find the template `N::g` using ADL
when instantiating `h` for `T = N::E` (since `N` is a namespace associated with `E`),
but we can **not** get that far until we successfully parse the generic definition of `h`.

##### 14.3.2 Points of Instantiation

A _Point of Instantiation (POI)_ is created when a code construct refers to a template specialization
in such a way that the definition of the corresponding template needs to be instantiated to create that specialization.
The POI is a point in the source where the substituted template could be inserted:
```c++
class MyInt
{
public:
    MyInt(int i);
};

MyInt operator-(MyInt const &);

bool operator>(MyInt const &, MyInt const &);

using Int = MyInt;

template <typename T>
void f(T i)
{
    if (i > 0)
    {
        g(-i);
    }
}

// #1
void g(Int)
{
    // #2
    f<Int>(42);  // point of call
    // #3
}
// #4
```
When a C++ compiler sees the call `f<Int>(42)`,
it knows the template `f` will need to be instantiated
for `T` substituted with `MyInt`:
A POI is created.


Points `#2` and `#3` are very close to the point of call,
but they can **not** be POIs because C++ does **not** allow us to insert the definition of `::f<Int>(Int)` there.
The essential difference between point `#1` and point `#4` is that at point `#4` the function `g(Int)` is visible,
and hence the template-dependent call `g(-i)` can be resolved.
However, if point `#1` were the POI, then that call could **not** be resolved because `g(Int)` is **not** yet visible.
Fortunately, C++ defines the POI "for a reference to a function template specialization"
to be _immediately after the nearest namespace scope declaration or definition that contains that reference_.
In our example, this is point `#4`.


You may wonder why this example involved the type `MyInt` rather than simple `int`.
The answer lies in the fact that the second lookup performed at the POI
(after instantiating `f`, at `#4`) is only an ADL.
Because `int` has **no** associated namespace,
the POI lookup would therefore **not** take place and would **not** find function `g`.
Hence, if we were to replace the type alias declaration for `Int` with
```c++
using Int = int;
```
the previous example would no longer compile.
The following example suffers from a similar problem:
```c++
template <typename T>
void f1(T x)
{
    g1(x);  // #1
}

void g1(int) {}

int main()
{
    f1(7);  // ERROR: g1 not found!
}
// #2 POI for f1<int>(int)
```
The call `f1(7)` creates a POI for `f1<int>(int)` just outside of `main` at point `#2`.
In this instantiation, the key issue is the lookup of function `g1`.
When the definition of the template `f1` is first encountered,
it is noted that the unqualified name `g1` is dependent
because it is the name of a function in a function call with dependent arguments
(the type of the argument `x` depends on the template parameter `T`).
Therefore, `g1` is looked up at point `#1` using ordinary lookup rules.
However, **no** `g1` is visible at this point.
At point `#2`, the POI, the function is looked up again via ADL in associated namespaces and classes,
but the only argument type is `int`, and it has **no** associated namespaces and classes.
Therefore, `g1` is **never** found even though ordinary lookup at the POI would have found `g1`.


The point of instantiation for variable templates is handled similarly to that of function templates.
Surprisingly, this is not clearly specified in the standard at the time of this writing.
However, it is not expected to be a controversial issue.


For class template specializations, the situation is different:
```c++
template <typename T>
class S 
{
public:
    T m;
};

// #1
unsigned long h()
{
    // #2
    return static_cast<std::size_t>(sizeof(S<int>));
    // #3
}
// #4
```
Again, the function scope points `#2` and `#3` can **not** be POIs
because a definition of a namespace scope class `S<int>` can **not** appear there.
Generally, templates can generally **not** appear in function scope,
yet the call operator of generic lambdas are a subtle exception to that observation.


If we were to follow the rule for function template instances, the POI would be at point `#4`,
but then the expression `sizeof(S<int>)` is invalid
because the size of `S<int>` can **not** be determined until point `#4` is reached.
Therefore, the POI for a reference to a generated class instance is defined to be
_the point immediately before the nearest namespace scope declaration or definition
that contains the reference to that instance_.
In our example, this is point `#1`.


When a template is actually instantiated, the need for additional instantiations may appear:
```c++
template <typename T>
class S
{
public:
    using I = int;
};

// #1
template <typename T>
void f()
{
    S<char>::I var1 = 41;
    typename S<T>::I var2 = 42;
}

int main()
{
    f<double>();
}
// #2 : #2a , #2b
```
Our preceding discussion already established that the POI for `f<double>` is at point `#2`.
The function template `f` also refers to the class specialization `S<char>` with a POI that is therefore at point `#1`.
It references `S<T>` too, but because this is still dependent, we can **not** really instantiate it at this point.
However, if we instantiate `f<double>` at point `#2`,
we notice that we also need to instantiate the definition of `S<double>`.
Such secondary or transitive POIs are defined slightly differently.
For function templates, the secondary POI is exactly the same as the primary POI.
For class entities, the secondary POI immediately precedes (in the nearest enclosing namespace scope) the primary POI.
In our example, this means that the POI of `f<double>` can be placed at point `#2b`,
and just before it is the secondary POI for `S<double>` at `#2a`.
Note how this differs from the POI for `S<char>`.


A translation unit often contains multiple POIs for the same instance.
For class template instances, only the first POI in each translation unit is retained,
and the subsequent ones are ignored (they are **not** really considered POIs).
For instances of function and variable templates, all POIs are retained.
In either case, the _One-Definition Rule (ODR)_ requires
that the instantiations occurring at any of the retained POIs be equivalent,
but a C++ compiler does not need to verify and diagnose violations of this rule.
This allows a C++ compiler to pick just one non-class POI to perform the actual instantiation
without worrying that another POI might result in a different instantiation.


In practice, most compilers delay the actual instantiation of most function templates
to the end of the translation unit.
This effectively moves the POIs of the corresponding template specializations to the end of the translation unit,
which is permitted by the C++ standard as an alternative POI.


Some instantiations can **not** be delayed,
including cases where instantiation is needed to determine a deduced return type
(see Section 15.10.1 and Section 15.10.4)
and cases where the function is `constexpr` and must be evaluated to produce a constant result.


Some compilers instantiate inline functions when they're first used
to potentially inline the call right away.
In modern compilers the inlining of calls is typically handled
by a mostly language-independent component of the compiler dedicated to optimizations
(a "back end" or "middle end").
However, C++ "front ends" (the C++-specific part of the C++ compiler)
that were designed in the earlier days of C++ may also have the ability to expand calls inline
because older back ends were too conservative when considering calls for inline expansion.

##### 14.3.3 The Inclusion Model

Whenever a POI is encountered, the definition of the corresponding template must somehow be accessible.
For class specializations, this means that
the class template definition must have been seen earlier in the translation unit.
This is also needed for the POIs of function templates and variable templates
(and member functions and static data members of class templates),
and typically template definitions are simply added to header files that are `#included` into the translation unit,
even when they're non-type templates.
This source model for template definitions is called the _inclusion model_,
and it is the only automatic source model for templates supported by the current C++ standard.


Although the inclusion model encourages programmers to place all their template definitions in header files
so that they are available to satisfy any POIs that may arise,
it is also possible to explicitly manage instantiations using
_explicit instantiation declarations_ and _explicit instantiation definitions_ (see Section 14.5).
Doing so is logistically not trivial and most of the time
programmers will prefer to rely on the automatic instantiation mechanism instead.
One challenge for an implementation with the automatic scheme is to
deal with the possibility of having POIs for the same specialization of a function or variable templates
(or the same member function or static data member of a class template instance) across different translation units.
We discuss approaches to this problem next.


#### 📌 14.4 Implementation Schemes


In this section we review some ways in which C++ implementations support the inclusion model.
All these implementations rely on two classic components:
a compiler and a linker.
The compiler translates source code to object files,
which contain machine code with symbolic annotations (cross-referencing other object files and libraries).
The linker creates executable programs or libraries
by combining the object files and resolving the symbolic cross-references they contain.


In what follows, we assume such a model,
even though it is entirely possible (but not popular) to implement C++ in other ways,
such as _C++ interpreters_ like [Cling](https://root.cern/cling/).


When a class template specialization is used in multiple translation units,
a compiler will repeat the instantiation process in every translation unit.
This poses very few problems because class definitions do not directly create low-level code.
They are used only internally by a C++ implementation
to verify and interpret various other expressions and declarations.
In this regard, the multiple instantiations of a class definition are **not** materially different
from the multiple (header-file) inclusions of a class definition in various translation units.


However, if you instantiate a (non-inline) function template, the situation may be different.
If you were to provide multiple definitions of an ordinary non-inline function, you would violate the ODR.
Assume, for example, that you compile and link a program consisting of two definitions of one ordinary function:
```c++
/// "a.cpp"
int main() {}

/// "b.cpp"
int main() {}
```
C++ compilers will compile each module separately without any problems
because indeed they are valid C++ translation units.
However, your linker will most likely protest if you try to link the two together:
Duplicate definitions are **not** allowed.


In contrast, consider the template case:
```c++
/// "t.hpp"
/// common header (inclusion model)
template <typename T>
class S
{
public:
    void f();
};

template <typename T>
void S::f() {}

void helper(S<int> *);

/// "a.cpp"
#include "t.hpp"

void helper(S<int> * s)
{
    s->f();  // #1 first point of instantiation of S::f
    
}

/// "b.cpp"
#include "t.hpp"

int main()
{
    S<int> s;
    helper(&s);
    s.f();   // #2 second point of instantiation of S::f
}
```
If the linker treats instantiated member functions of class templates
just like it does for ordinary functions or member functions,
the compiler needs to ensure that it generates code at only one of the two POIs:
at points `#1` or `#2`, but **not** both.
To achieve this, a compiler has to carry information from one translation unit to the other,
and this is something C++ compilers were **never** required to do prior to the introduction of templates.
In what follows, we discuss the three broad classes of solutions that have been used by C++ implementers.


Note that the same problem occurs with all linkable entities produced by template instantiation:
instantiated function templates and member function templates,
as well as instantiated static data members and instantiated variable templates.

##### 14.4.1 Greedy Instantiation

The first C++ compilers that popularized greedy instantiation were produced by a company called Borland.
It has grown to be by far the most commonly used technique among the various C++ systems.


Greedy instantiation assumes that the linker is aware that linkable template instantiations
may in fact appear in duplicate across the various object files and libraries.
The compiler will typically mark these entities in a special way.
When the linker finds multiple instances, it keeps one and discards all the others.
There is not much more to it than that.


Greedy instantiation has some serious drawbacks:
- The compiler may be wasting time on generating and optimizing `N` instantiations,
  of which only one will be kept.
- Linkers typically do **not** check that two instantiations are identical
  because some insignificant differences in generated code can validly occur
  for multiple instances of one template specialization.
  - These small differences should **not** cause the linker to fail.
    These differences could result from tiny differences
    in the state of the compiler at the instantiation times.
  - However, this often also results in the linker **not** noticing more substantial differences,
    such as when one instantiation was compiled with strict floating-point math rules.  
    whereas the other was compiled with relaxed, higher-performance floating-point math rules.
  - Current systems have grown to detect certain other differences.
    For example, they might report if one instantiation has associated debugging information and another does not.
- The sum of all the object files could potentially be much larger than with alternatives
  because the same code may be duplicated many times.


Greedy instantiation has the following merits:
- The traditional source-object dependency is preserved.
  In particular, one translation unit generates but one object file,
  and each object file contains compiled code for all the linkable definitions
  in the corresponding source file, which includes the instantiated definitions.
- All function template instances are candidates for inlining
  without resorting to expensive “link-time” optimization mechanisms
  (and, in practice, function template instances are often small functions that benefit from inlining).
  The other instantiation mechanisms treat _inline_ function template instances specially
  to ensure they can be expanded inline.
  However, greedy instantiation allows even non-inline function template instances to be expanded inline.


The linker mechanism that allows duplicate definitions of linkable entities
is also typically used to handle
- Duplicate _spilled inlined functions_:
  - When a compiler is unable to "inline" every call
    to a function that you marked with the keyword `inline`,
    a separate copy of the function is emitted in the object file.
    This may happen in multiple object files.
- _Virtual function dispatch tables_:
  - Virtual function calls are usually implemented as indirect calls
    through a table of pointers to functions.


If this mechanism is not available,
the alternative is usually to emit these items with internal linkage,
at the expense of generating larger code.
The requirement that an inline function have a single address makes it difficult
to implement that alternative in a standard-conforming way.

##### 14.4.2 Queried Instantiation

In the mid-1990s, a company called _Sun Microsystems_ (later acquired by Oracle)
released a reimplementation of its C++ compiler (version 4.0)
with a new and interesting solution of the instantiation problem,
which we call _queried instantiation_.
Queried instantiation is conceptually remarkably simple and elegant,
and yet it is chronologically the most recent class of instantiation schemes that we review here.
This scheme maintains a database shared by the compilations of all translation units.
This database keeps track of which specializations have been instantiated and on what source code they depend.
The generated specializations themselves are typically stored with this information in the database.
Whenever a point of instantiation for a linkable entity is encountered,
one of three things can happen:
1. No specialization is available:
   In this case, instantiation occurs, and the resulting specialization is entered in the database.
2. A specialization is available
   but is out of date because source changes have occurred since it was generated.
   Here, too, instantiation occurs, but the resulting specialization replaces the old one.
3. An up-to-date specialization is available in the database.
   Nothing needs to be done.


Although conceptually simple, this design presents a few implementation challenges:
- It is **not trivial** to maintain correctly the dependencies of the database contents
  with respect to the state of the source code.
  Although it is not incorrect to mistake the third case for the second,
  doing so increases the amount of work done by the compiler (and hence overall build time).
- It is quite common to compile multiple source files concurrently.
  Hence, an industrial-strength implementation needs to provide
  the appropriate amount of concurrency control in the database.


Unfortunately, the use of a database may also present some problems to the programmer.
The origin of most of these problems lies in that fact that
the traditional compilation model inherited from most C compilers no longer applies:
A single translation unit no longer produces a single standalone object file.
Assume that you wish to link your final program.
This link operation needs not only the contents of each of the object files associated with your various translation units,
but also the object files stored in the database.
Similarly, if you create a binary library, you need to ensure that the tool that creates that library
(typically a linker or an archiver) is aware of the database contents.
More generally, any tool that operates on object files may need to be made aware of the contents of the database.
Many of these problems can be alleviated by not storing the instantiations in the database,
but instead by emitting the object code in the object file that caused the instantiation in the first place.


Libraries present yet another challenge.
A number of generated specializations may be packaged in a library.
When the library is added to another project,
that project's database may need to be made aware of the instantiations that are already available.
If not, and if the project creates some of its own points of instantiation
for the specializations present in the library,
duplicate instantiation may occur.
A possible strategy to deal with such situations
is to use the same linker technology that enables greedy instantiation:
Make the linker aware of generated specializations and have it weed out duplicates
(which should nonetheless occur much less frequently than with greedy instantiation).
Various other subtle arrangements of sources, object files, and libraries
can lead to frustrating problems such as missing instantiations
because the object code containing the required instantiation
was not linked in the final executable program.


Ultimately, queried instantiation did **not** survive in the marketplace,
and even Sun's compiler now uses greedy instantiation.

##### 14.4.3 Iterated Instantiation

The first compiler to support C++ templates was Cfront 3.0,
a direct descendant of the compiler that Bjarne Stroustrup wrote to develop the language.
Do **not** let this phrase mislead you into thinking that Cfront was an academic prototype:
It was used in industrial contexts and formed the basis of many commercial C++ compiler offerings.
Release 3.0 appeared in 1991 but was plagued with bugs.
Version 3.0.1 followed soon thereafter and made templates usable.

An inflexible constraint on Cfront was that it had to be very portable from platform to platform,
and this meant that it
1. Used the C language as a common target representation across all target platforms;
2. Used the local target linker.


In particular, this implied that the linker was **not** aware of templates.
In fact, Cfront emitted template instantiations as ordinary C functions,
and therefore it had to avoid duplicate instantiations.
Although the Cfront source model was different from the standard inclusion model,
its instantiation strategy can be adapted to fit the inclusion model.
As such, it also merits recognition as the first incarnation of _iterated instantiation_.


The Cfront iteration can be described as follows:
1. Compile the sources **without** instantiating any required linkable specializations;
2. Link the object files using a _prelinker_;
3. The prelinker invokes the linker and parses its error messages to determine
   whether any are the result of missing instantiations.
   If so, the prelinker invokes the compiler on sources that contain the needed template definitions,
   with options to generate the missing instantiations.
4. Repeat step 3 if any definitions are generated.


The need to iterate step 3 is prompted by the observation that
the instantiation of one linkable entity may lead to the need
for another such entity that was not yet instantiated.
Eventually the iteration will “converge”, and the linker will succeed in building a complete program.


The drawbacks of the original Cfront scheme are quite severe:
- The perceived time to link is augmented not only by the prelinker overhead
  but also by the cost of every required recompilation and relinking.
  Some users of Cfront-based systems reported link times of “a few days”
  compared with “about an hour” with the alternative schemes reported earlier.
- Diagnostics (errors, warnings) are delayed until link time.
  This is especially painful when linking becomes expensive
  and the developer must wait hours just to find out about a typo in a template definition.
- Special care must be taken to remember where the source containing a particular definition is located (step 1).
  Cfront in particular used a central repository,
  which had to deal with some of the challenges of the central database in the queried instantiation approach.
  In particular, the original Cfront implementation was **not** engineered to support concurrent compilations.


The iteration principle was subsequently refined both by the Edison Design Group's (EDG) implementation and by HP's aC++,
eliminating some of the drawbacks of the original Cfront implementation.
In practice, these implementations work quite well, and,
although a build “from scratch” is typically more time consuming than the alternative schemes,
subsequent build times are quite competitive.
Still, relatively few C++ compilers use iterated instantiation anymore.
HP's aC++ also added greedy instantiation made that the default mechanism.


#### 📌 14.5 Explicit Instantiation


It is possible to create explicitly a point of instantiation for a template specialization.
The construct that achieves this is called an _explicit instantiation directive_.
Syntactically, it consists of the keyword `template` followed by a declaration of the specialization to be instantiated.
```c++
template <typename T>
void f(T) {}

// four valid explicit instantiations:
template void f<int>(int);
template void f<>(float);
template void f(long);
template void f(char);
```
Note that every instantiation directive is valid.
Template arguments can be deduced (see Chapter 15).


Members of class templates can also be explicitly instantiated in this way:
```c++
template <typename T>
class S 
{
public:
    void f() {}
};

template void S<int>::f();
template class S<void>;
```
Furthermore, all the members of a class template specialization can be explicitly instantiated
by explicitly instantiating the class template specialization.
Because these explicit instantiation directives ensure that
a definition of the named template specialization (or member thereof) is created,
the explicit instantiation directives above are more accurately referred to as _explicit instantiation definitions_.
A template specialization that is explicitly instantiated should **not** be explicitly specialized,
and vice versa, because that would imply that the two definitions could be different (thus violating the ODR).

##### 14.5.1 Manual Instantiation

Many C++ programmers have observed that
automatic template instantiation has a nontrivial negative impact on build times.
This is particularly true with compilers that implement greedy instantiation (Section 14.4.1),
because the same template specializations may be instantiated and optimized in many different translation units.


A technique to improve build times consists in
_manually instantiating_ those template specializations that the program requires in a single location
and _inhibiting_ the instantiation in all other translation units.
One portable way to ensure this _inhibition_ is to **not** provide the template definition
except in the translation unit where it is explicitly instantiated.
In the 1998 and 2003 C++ standards,
this was the only portable way to inhibit instantiation in other translation units.
```c++
/// translation unit 1:
// no definition: prevents instantiation in this translation unit
template <typename T>
void f(); 

void g()
{
    f<int>();
}

/// translation unit 2:
template <typename T>
void f()
{
    // implementation
}

// manual instantiation
template void f<int>();

void g();

int main()
{
    g();
}
```
In the first translation unit, the compiler can **not** see the definition of the function template `f`,
so it can **not** produce an instantiation of `f<int>`.
The second translation unit provides the definition of `f<int>` via an explicit instantiation definition.  
Without it, the program would **fail** to link.


Manual instantiation has a clear disadvantage:
We must carefully keep track of which entities to instantiate.
For large projects this quickly becomes an excessive burden,
hence we do **not** recommend it.
We have worked on several projects that initially underestimated this burden,
and we came to regret our decision as the code matured.


However, manual instantiation also has a few advantages
because the instantiation can be tuned to the needs of the program.
Clearly, the overhead of large headers is avoided,
as is the overhead of repeatedly instantiating the same templates
with the same arguments in multiple translation units.
Moreover, the source code of template definition can be kept hidden,
but then **no** additional instantiations can be created by a client program.


Some of the burden of manual instantiation can be alleviated
by placing the template definition into a third source file,
conventionally with the extension `.tpp`.
For our function `f`, this breaks down into:
```c++
/// "f.hpp"
// no definition: prevents instantiation
template <typename T>
void f(); 

/// "f.tpp"
#include "f.hpp"

//definition
template <typename T>
void f()
{
    // implementation
}

// "f.cpp"
#include "f.tpp"

// manual instantiation
template void f<int>();
```
This structure provides some flexibility.
One can include only `f.hpp` to get the declaration of `f`, with **no** automatic instantiation.
Explicit instantiations can be manually added to `f.cpp` as needed.
Or, if manual instantiations become too onerous, one can also include `f.tpp` to enable automatic instantiation.

##### 14.5.2 Explicit Instantiation Declarations

A more targeted approach to the elimination of redundant automatic instantiations
is the use of an _explicit instantiation declaration_,
which is an explicit instantiation directive prefixed by the keyword `extern`.
An explicit instantiation declaration _generally_ suppresses automatic instantiation
of the named template specialization,
because it declares that the named template specialization will be defined somewhere in the program
(by an explicit instantiation definition).
We say _generally_, because there are many **exceptions** to this:
- Inline functions can still be instantiated for the purpose of expanding them inline
  (but no separate object code is generated).
- Variables with deduced `auto` or `decltype(auto)` types
  and functions with deduced return types
  can still be instantiated to determine their types.
- Variables whose values are usable as constant-expressions
  can still be instantiated so their values can be evaluated.
- Variables of reference types can still be instantiated
  so the entity they reference can be resolved.
- Class templates and alias templates can still be instantiated to check the resulting types.


Using explicit instantiation declarations,
we can provide the template definition for `f` in the header (`t.hpp`),
then suppress automatic instantiation for commonly used specializations, as follows:
```c++
/// "t.hpp"
template <typename T> 
void f() {}

// declared but not defined
extern template void f<int>();
extern template void f<float>(); 

/// "t.cpp"
// definition
template void f<int>();
template void f<float>();
```
Each explicit instantiation declaration must be paired with a corresponding explicit instantiation definition,
which must follow the explicit instantiation declaration.
Omitting the definition will result in a linker error.


Explicit instantiation declarations can be used to improve compile or link times
when certain specializations are used in many different translation units.
Unlike with manual instantiation, which requires manually updating
the list of explicit instantiation definitions each time a new specialization is required,
explicit instantiation declarations can be introduced as an optimization at any point.
However, the compile-time benefits may not be as significant as with manual instantiation,
both because some redundant automatic instantiation is likely to occur
and because the template definitions are still parsed as part of the header.


An interesting part of this optimization problem is to determine exactly
which specializations are good candidates for explicit instantiation declarations.
Low-level utilities such as the common Unix tool `nm` can be useful
in identifying which automatic instantiations actually made it into the object files that comprise a program.


#### 📌 14.6 Compile-Time `if` Statements


As introduced in Section 8.5,
C++17 added a new statement kind that turns out to be remarkably useful when writing templates:
compile-time `if`.
It also introduces a new wrinkle in the instantiation process.


The following example illustrates its basic operation:
```c++
template <typename T>
bool f(T p)
{
    if constexpr (sizeof(T) <= sizeof(long long))
    {
        return 0 < p;
    }
    else
    {
        return 0 < p.compare(0);
    }
}

bool g(int n)
{
    return f(n);  // OK
}
```
Prior to C++17 and its `constexpr if` statements,
avoiding such errors required explicit template specialization or overloading (see Chapter 16)
to achieve similar effects.
The example above, in C++14, might be implemented as follows:
```c++
template <bool>
struct Dispatch
{
    template <typename T>
    static bool f(T p)
    {
        return 0 < p.compare(0);
    }
};
    
template <>
struct Dispatch<true>
{
    template <typename T>
    static bool f(T p)
    {
        return 0 < p;
    }
};

template <typename T>
bool f(T p)
{
    return Dispatch<sizeof(T) <= sizeof(long long)>::f(p);
}

bool g(int n)
{
    return f(n);  // OK
}
```
Clearly, the `constexpr if` alternative expresses our intention far more clearly and concisely.
However, it requires implementations to refine the unit of instantiation:
Whereas previously function definitions were always instantiated as a whole,
now it must be possible to inhibit the instantiation of parts of them.


Another very handy use of `constexpr if` is expressing the recursion needed to handle function parameter packs.
To generalize the example, introduced in Section 8.5:
```c++
template <typename Head, typename ... Remainder>
void f(Head && h, Remainder && ... r)
{
    doSomething(std::forward<Head>(h));
    
    if constexpr (sizeof...(r) != 0)
    {
        // handle the remainder recursively (perfectly forwardingthe arguments):
        f(std::forward<Remainder>(r)...);
    }
}
```
Without `constexpr if` statements, this requires an additional overload of the `f` template
to ensure that recursion terminates.
```c++
template <typename Head, typename ... Remainder>
void f(Head && h, Remainder && ... r)
{
    doSomething(std::forward<Head>(h));
    f(std::forward<Remainder>(r)...);
}

// explicit specialization of f for end of recursion
template <typename Head>
void f(Head && h)
{
    doSomething(std::forward<Head>(h));
}
```
Even in non-template contexts, `constexpr if` statements have a somewhat unique effect:
```c++
void h();  // no definition!

void g() 
{
    if constexpr (sizeof(int) == 1) 
    {
        h();
    }
}
```
On most platforms, the condition in `g` is `false` and the call to `h` is therefore discarded.
As a consequence, `h` need **not** necessarily be defined at all (unless it is used elsewhere, of course).
Had the keyword `constexpr` been omitted in this example,
a lack of a definition for `h` would often elicit an error at link time.
Optimization may nonetheless mask the error.
With `constexpr if` the problem is guaranteed not to exist.


#### 📌 14.7 In the Standard Library


The C++ standard library includes a number of templates that are only commonly used with a few basic types.
For example, the `std::basic_string` class template is most commonly used with `char`
(because `std::string` is a type alias of `std::basic_string<char>`) or `wchar_t`,
although it is possible to instantiate it with other character-like types.
Therefore, it is common for standard library implementations to introduce explicit instantiation declarations
for these common cases.
```c++
namespace std
{

template <typename charT, 
          typename traits = char_traits<charT>,
          typename Allocator = allocator<charT>>
class basic_string
{
    // ...
};

extern template class basic_string<char>;

extern template class basic_string<wchar_t>;

}  // namespace std
```
The source files implementing the standard library will then contain the
corresponding explicit instantiation definitions,
so that these common implementations can be shared among all users of the standard library.
Similar explicit instantiations often exist for the various stream classes,
such as `std::basic_iostream`, `std::basic_istream`, and so on.






### 🎯 Chapter 15 Template Argument Deduction


#### 📌 15.1 The Deduction Process


The basic deduction process
compares the types of an argument of a function call with the corresponding parameterized type of a function template
and attempts to conclude the correct substitution for one or more of the deduced parameters.
Each argument-parameter pair is analyzed independently,
and if the conclusions differ in the end, the deduction process fails:
```c++
template <typename T>
T max (T a, T b)
{
    return b < a ? a : b;
}

auto g = max(1, 1.0);  // ERROR
```
Here the first call argument is of type `int`,
so the parameter `T` of our original `max` template is tentatively deduced to be `int`.
The second call argument is a `double`, however, and so `T` should be `double` for this argument:
This conflicts with the previous conclusion.
Note that we say that “the deduction process fails”,
not that “the program is invalid”.
After all, it is possible that the deduction process would succeed for another template named `max`
(function templates can be overloaded much like ordinary functions. See Section 1.5 and Chapter 16).


If all the deduced template parameters are consistently determined,
the deduction process can still fail if substituting the arguments
in the rest of the function declaration results in an invalid construct:
```c++
template <typename T>
typename T::ElementT at(T a, int i)
{
    return a[i];
}

void f(int * p)
{
    int x = at(p, 7);
}
```
Here `T` is concluded to be `int *`
(there is only one parameter type where `T` appears,
so there are obviously no analysis conflicts).
However, substituting `int *` for `T` in the return type `T::ElementT` is clearly invalid C++,
and the deduction process fails.
In this case, deduction failure follows the SFINAE principle (see Section 8.4):
If there were another function for which deduction succeeds, the code could be valid.


We still need to explore how argument-parameter matching proceeds.
We describe it in terms of
matching a type `A` (derived from the call argument type)
to a parameterized type `P` (derived from the call parameter declaration).
If the call parameter is declared with a reference declarator,
`P` is taken to be the type referenced, and `A` is the type of the argument.
Otherwise, `P` is the declared parameter type,
and `A` is obtained from the type of the argument
by _decaying_ array and function types to pointer types
(_decay_ is the term used to refer to the implicit conversion of function and array types to pointer types),
ignoring top-level `const` and `volatile` qualifiers:
```c++
// parameterized type P is T
template <typename T>
void f(T) {}

// parameterized type P is T
template <typename T>
void g(T &) {}

double arr[20];
int const seven = 7;

f(arr);    // T = double *
g(arr);    // T = double [20]
f(seven);  // T = int
g(seven);  // T = const int
f(7);      // T = int
g(7);      // T = int, ERROR: can't bind 7 to int &
```
The fact that **no** decay occurs for arguments bound to reference parameters
can be surprising when the arguments are string literals.
Reconsider our `max` template declared with references:
```c++
template <typename T>
T const & max(T const & a, T const & b);
```
It would be reasonable to expect that for the expression `max("Apple", "Pie")`,
`T` is deduced to be `char const *`.
However, the type of `"Apple"` is `char const[6]`, and the type of `"Pie"` is `char const[4]`.
**No** array-to-pointer decay occurs (because the deduction involves reference parameters),
and therefore `T` would have to be both `char[6]` and `char[4]` for deduction to succeed.
That is, of course, impossible.
See Section 7.4 for a discussion about how to deal with this situation.


#### 📌 15.2 Deduced Contexts


Parameterized types that are considerably more complex than just `T`
can be matched to a given argument type:
```c++
template <typename T>
void f1(T *);

template <typename E, int N>
void f2(E (&)[N]);

template <typename T1, typename T2, typename T3>
void f3(T1 (T2::*)(T3 *));

class S
{
public:
    void f(double *);
};

void g(int *** ppp)
{
    bool b[42];
    f1(ppp);    // deduces T to be int **
    f2(b);      // deduces E to be bool and N to be 42
    f3(&S::f);  // deduces T1 = void, T2 = S, and T3 = double
}
```
Complex type declarations are built from more elementary constructs
(pointer, reference, array, and function declarators; pointer-to-member declarators; template-ids; and so forth),
and the matching process proceeds from the top-level construct and recurses through the composing elements.
It is fair to say that most type declaration constructs can be matched in this way,
and these are called _deduced contexts_.
However, a few constructs are
[Non-Deduced Contexts](https://en.cppreference.com/w/cpp/language/template_argument_deduction#Non-deduced_contexts).
For example:
- **Qualified type names**.  
  For example, a type name like `Q<T>::X` will **never** be used to deduce a template parameter `T`.
- **Non-type expressions that are not just a non-type parameter**.  
  For example, a type name like `S<I + 1>` will **never** be used to deduce `I`.
  **Neither** will `T` be deduced by matching against a parameter of type `int (&)[sizeof(S<T>)]`.
- ...


These limitations should come as no surprise because the deduction would,
in general, not be unique (or even finite),
although this limitation of qualified type names is sometimes easily overlooked.
A non-deduced context does not automatically imply that the program is in error
or even that the parameter being analyzed can not participate in type deduction.
To illustrate this, consider the following, more intricate example:
```c++
template <int N>
class X
{
public:
    using I = int;

    void f(int) {}
};

template <int N>
void fppm(void (X<N>::* p)(typename X<N>::I));

int main()
{
    // fine: N deduced to be 33
    fppm(&X<33>::f);
}
```
In the function template `fppm`, the subconstruct `X<N>::I` is a non-deduced context.
However, the member-class component `X<N>` of the pointer-to-member type is a deducible context,
and when the parameter `N`, which is deduced from it, is plugged in the non-deduced context,
a type compatible with that of the actual argument `&X<33>::f` is obtained.
The deduction therefore succeeds on that argument-parameter pair.


Conversely, it is possible to deduce contradictions for a parameter type entirely built from deduced contexts.
For example, assuming suitably declared class templates `X` and `Y`:
```c++
template <typename, typename> 
class X {};

template <typename> 
class Y {};

template <typename T>
void f(X<Y<T>, Y<T>>) {}

void g()
{
    f(X<Y<int>, Y<int>>());   // OK
    f(X<Y<int>, Y<char>>());  // ERROR: deduction fails
}
```
The problem with the second call to the function template `f` is that
the two arguments deduce different arguments for the parameter `T`, which is not valid.
In both cases, the function call argument is a temporary object
obtained by calling the default constructor of the class template `X`.


#### 📌 15.3 Special Deduction Situations


There are several situations in which the pair `(A, P)` used for deduction
is **not** obtained from the arguments to a function call and the parameters of a function template.
The first situation occurs when the address of a function template is taken.
In this case, `P` is the type of the function template declaration,
and `A` is the function type underlying the pointer that is initialized or assigned to:
```c++
template <typename T>
void f(T, T) {}

void (*pf)(char, char) = &f;
```
In this example, `P` is `void (T, T)` and `A` is `void (char, char)`.
Deduction succeeds with `T` substituted with `char`,
and `pf` is initialized to the address of the specialization `f<char>`.


Similarly, function types are used for `P` and `A` for a few other special situations:
- Determining a partial ordering between overloaded function templates;
- Matching an explicit specialization to a function template;
- Matching an explicit instantiation to a template;
- Matching a friend function template specialization to a template;
- Matching a placement `operator delete` or `operator delete[]`
  to a corresponding placement `operator new` template or `operator new[]` template.


Some of these topics, along with the use of template argument deduction for class template partial specializations,
are further developed in Chapter 16.


Another special situation occurs with conversion function templates:
```c++
class S 
{
public:
    template <typename T> 
    operator T&();
};
```
In this case, the pair `(P, A)` is obtained as if it involved:
- an argument of the conversion-target type;
- a parameter type that is the return type of the conversion function.
  The following code illustrates one variation:
```c++
void f(int (&)[20]) {}

void g(S s)
{
    f(s);
}
```
Here we are attempting to convert `S` to `int (&)[20]`.
Type `A` is therefore `int [20]` and type `P` is `T`.
The deduction succeeds with `T` substituted with `int [20]`.


Some special treatment is also needed for the deduction of the `auto` placeholder type.
That is discussed in Section 15.10.4.


#### 📌 15.4 Initializer Lists


When the argument of a function call is an initializer list,
that argument **doesn't** have a specific type,
so in general no deduction will be performed from that given pair `(A, P)` because there is no `A`:
```c++
template <typename T> 
void f(T t) {}

int main() 
{
    f({1, 2, 3});  // ERROR: can not deduce T from an initializer list
}
```
However, if the parameter type `P`, after removing references and top-level cv,
is equivalent to `std::initializer_list<P′>` for some type `P′` that has a deducible pattern,
deduction proceeds by comparing `P′` to the type of each element in the initializer list,
succeeding only if all of the elements have the same type:
```c++
template <typename T> 
void f(std::initializer_list<T> il) {}

int main() 
{
    f({1, 2, 3});  // OK: T deduced to int
    f({1, '2'});   // ERROR: T is deduced to both in and char
}
```
Similarly, if the parameter type `P` is a reference to an array type
with element type `P′` for some type `P′` that has a deducible pattern,
deduction proceeds by comparing `P′` to the type of each element in the initializer list,
succeeding only if all of the elements have the same type.
Furthermore, if the bound has a deducible pattern (i.e., just names a non-type template parameter),
then that bound is deduced to the number of elements in the list.
```c++
template <typename T, std::size_t N>
void f(T (&&)[N])
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

int main(int argc, char * argv[])
{
    f({1, 2, 3});  // T = int; N = 3
}
```


#### 📌 15.5 Parameter Packs


```c++
template <typename First, typename ... Rest>
void f(First first, Rest ... rest) {}

void g(int i, double j, int * k)
{
    f(i, j, k);  // deduces First to int, Rest to {double, int *}
}
```
Deduction for the first function parameter is simple, since it does not involve any parameter packs.
Deduction determines the value of the parameter pack `Rest` to be the sequence `{double, int *}`.
Substituting the results of that deduction and the deduction for the first function parameter
yields the function type `void (int, double, int*)`, which matches the argument types at the call site.


Values for multiple template parameters and parameter packs
can be determined from each of the argument types:
```c++
template <typename T, typename ... Rest>
void h1(std::pair<T, Rest> const & ...) {}

template <typename ... Ts, typename ... Rest>
void h2(std::pair<Ts, Rest> const & ...) {}

void foo(std::pair<int, float> pif,
         std::pair<int, double> pid,
         std::pair<double, double> pdd)
{
    h1(pif, pid);  // OK: deduces T to int, Rest to {float,double}
    h2(pif, pid);  // OK: deduces Ts to {int, int}, Rest to {float, double}
    
    h1(pif, pdd);  // ERROR: T deduced to int from the 1st arg, but to double from the 2nd
    h2(pif, pdd);  // OK: deduces Ts to {int, double}, Rest to {float, double}
}
```
Deduction for parameter packs is **not** limited to function parameter packs
where the argument-parameter pairs come from call arguments.
In fact, this deduction is used wherever a pack expansion
is at the end of a function parameter list or a template argument list.
If a pack expansion occurs anywhere else in a function parameter list or template argument list,
that pack expansion is considered a non-deduced context.
```c++
template <typename ... Types>
bool f1(std::tuple<Types ...>, std::tuple<Types ...>) {}

template <typename ... Types1, typename ... Types2>
bool f2(std::tuple<Types1 ...>, std::tuple<Types2 ...>) {}

void bar(std::tuple<short, int, long> sv,
         std::tuple<unsigned short, unsigned, unsigned long> uv)
{
    f1(sv, sv);  // OK: Types is deduced to {short, int, long}.
    f2(sv, sv);  // OK: Types1 is deduced to {short, int, long}, Types2 is deduced to {short, int, long}.
    f1(sv, uv);  // ERROR: Types is deduced to {short, int, long} from the 1st arg, but to {unsigned short, unsigned, unsigned long} from the 2nd.
    f2(sv, uv);  // OK: Types1 is deduced to {short, int, long}, Types2 is deduced to {unsigned short, unsigned, unsigned long}.
}
```

##### 15.5.1 Literal Operator Templates

C++11 allows user-defined literals (could be `constexpr`) as follows:
- Cooked: Pre-processed by processor.
```c++
T operator "" _suffix(unsigned long long);
T operator "" _suffix(long double);
T operator "" _suffix(char);
T operator "" _suffix(wchar_t);
T operator "" _suffix(char16_t);
T operator "" _suffix(char32_t);
T operator "" _suffix(char const *, std::size_t);
T operator "" _suffix(wchar_t const *, std::size_t);
T operator "" _suffix(char16_t const *, std::size_t);
T operator "" _suffix(char32_t const *, std::size_t);
```
- Raw: Only suffix numeric types!
```c++
T operator "" _suffix(const char *);

// literal operator template
template <char ...>
T operator "" _suffix();
```
When the compiler identifies a user-defined literal and has to call the appropriate user-defined literal operator,
it will pick the overload from the overload set according to the following rules:
- **For integral literals**, it calls in the following order:
  1. The operator that takes an `unsigned long long`,
  2. The _raw_ literal operator that takes a `const char *`,
  3. the literal operator template.
- **For floating-point literals**, it calls in the following order:
  1. The operator that takes a `long double`,
  2. The _raw_ literal operator that takes a `const char *`,
  3. The literal operator template.
- **For character literals**, it calls the appropriate _cooked_ operator,
  depending on the character type (`char`, `wchar_t`, `char16_t`, and `char32_t`).
- **For string literals**, it calls the appropriate _cooked_ operator,
  that takes a pointer to the string of characters and the size,
  depending on the string type.


Literal operator templates have their argument determined in a unique way.
```c++
template <char...>
int operator "" _B7();  // #1

int a = 121_B7;         // #2
```
Here, the initializer for `#2` contains a user-defined literal,
which is turned into a call to the literal operator template `#2`
with the template argument list `<'1', '2', '1'>`.
Thus, an implementation of the literal operator such as
```c++
template <char ... cs>
int operator "" _B7()
{
    std::array<char, sizeof...(cs)> chars {cs...};

    for (char c: chars)
    {
        std::cout << '\'' << c << '\'';
    }

    std::cout << '\n';
    return ...;
}
```
will output `'1' '2' '1' '.' '5'` for `121.5_B7`.
Note that raw literal operators are _only supported for valid numeric (integral and floating-point) literals_:
```c++
auto b = 01.3_B7;     // OK: deduces {'0', '1', '.', '3'}
auto c = 0xFF00_B7;   // OK: deduces {'0', 'x', 'F', 'F', '0', '0'}
auto d = 0815_B7;     // ERROR: invalid digit '8' in octal constant
auto e = hello_B7;    // ERROR: use of undeclared identifier hello_B7
auto f = "hello"_B7;  // ERROR: no matching operator "" _B7
```
See Section 25.6 for an application of the feature to compute integral literals at compile time.


#### 📌 15.6 Rvalue References

##### 15.6.1 Reference Collapsing

Programmers are **not** allowed to _directly_ declare a “reference to a reference”:
```c++
int const & r = 42;
int const & & ref2ref = i;  // ERROR: reference to reference is invalid
```
However, when composing types through the substitution of
template parameters, type aliases, or `decltype` constructs,
such situations are permitted:
```c++
using RI = int &;
int i = 42;
RI r = i;
R const & rr = r;  // OK: rr has type int &. 
                   // const qualifier on a reference type RI (aka int &) has no effect. 
```
The rules that determine the type resulting from such a composition
are known as the _reference collapsing_ rules:
1. Any `const` or `volatile` qualifiers applied on top of the inner reference are simply **discarded**.
   I.e., only the bottom-level cv is retained.
2. The two references are reduced to a single reference.
  - Only rvalue reference of rvalue references are rvalue references.
  - All other compositions yields lvalue references.
```c++
using RCI = int const &;
RCI volatile && r = 42;   // OK: r has type int const &, top-level volatile is discarded
using RRI = int &&;
RRI const && rr = 42;     // OK: rr has type int &&, top-level const is discarded
```

##### 15.6.2 Forwarding References

Template argument deduction behaves in a special way when a function parameter is a _forwarding reference_
(an rvalue reference to a template parameter of that function template).
In this case, template argument deduction considers **not** just the type of the function call argument
but also whether that argument is an lvalue or an rvalue.
- **When the argument is an lvalue**,
  the type determined by template argument deduction is an lvalue reference to the argument type,
  and the reference collapsing rules (see above) ensure that the substituted parameter will be an lvalue reference.
- **Otherwise**,
  the type deduced for the template parameter is simply the argument type (**not** a reference type),
  and the substituted parameter is an rvalue reference to that type.
```c++
template <typename T> 
void f(T && p) {}

void g()
{
    int i;
    int const j = 0;
    f(i);  // T = int &
    f(j);  // T = const int &
    f(2);  // T = int
}
```
The deduction of `T` as a reference type can have some interesting effects on the instantiation of the template.
For example, a local variable declared with type `T` will, after instantiation for an lvalue,
have reference type and will therefore require an initializer:
```c++
template <typename T> 
void f(T &&)
{
    T x;  // ERROR for lvalue arguments
}
```
Solution:
```c++
template <typename T> 
void f(T &&)
{
    std::remove_reference_t<T> x;  // ERROR again for array arguments...
}
```

##### 15.6.3 Perfect Forwarding

Forwarding reference parameters accept almost any argument (bit fields are an exception)
and captures its salient properties (both its _type_ and its _value category_).
```c++
class C {};

void g(C &) {}
void g(C const &) {}
void g(C &&) {}

template <typename T>
void forwardToG(T && x)
{
    g(static_cast<T &&>(x));
}

void foo()
{
    C v;
    C const c;
    
    forwardToG(v);             // g(C &)
    forwardToG(c);             // g(const C &)
    forwardToG(C());           // g(C &&)
    forwardToG(std::move(v));  // g(C &&)
}
```
The use of `static_cast` within the function `forwardToG` requires some additional explanation.
In each instantiation of `forwardToG`,
the type of expression `x` could either be lvalue reference type or rvalue reference type.
Regardless, the value category of expression `x` will always be an lvalue.


Treating a parameter of rvalue reference type as an lvalue is intended as a safety feature,
because anything with a name (like a parameter) can easily be referenced multiple times in a function.
If each of those references could be implicitly treated as an rvalue,
its value could be _destroyed_ unbeknownst to the programmer.
Therefore, one must explicitly state when a named entity should be treated as an rvalue.
For this purpose, the C++ standard library function `std::move` treats any value as an rvalue
(or, more precisely, an xvalue. See Appendix B for details).


The `static_cast` casts `x` to its original type and value category.
The type `T &&` will either collapse to an lvalue reference
(if the original argument was an lvalue causing `T` to be an lvalue reference)
or will be an rvalue reference (if the original argument was an rvalue),
so the result of the `static_cast` has the same type and value category as the original argument,
thereby achieving perfect forwarding.


The C++ standard library provides a function template `std::forward` in header `<utility>`
that should be used in place of `static_cast` for perfect forwarding.
Using that utility template better documents the programmer's intent
than the arguably opaque `static_cast` constructs shown above
and prevents errors such as omitting one `&`.
```c++
template <typename T>
void forwardToG(T && x)
{
    g(std::forward<T>(x));
}
```
Perfect forwarding is **not** “perfect” and has several failure cases.  
Refer to [Effective Modern C++ Notes Item 30](./effective_cpp_notes_04_effective_modern_cpp.md#-item-30-familiarize-yourself-with-perfect-forwarding-failure-cases).
- Perfect forwarding fails when template type deduction fails or when it deduces the wrong type.
  The kinds of arguments that lead to perfect forwarding failure are:
  - braced initializers (decreed failure, create a temp var via auto);
  - null pointers expressed as `0` or `NULL` (deduces "wrong" type `int`, use `nullptr`);
  - declaration-only integral const static data members (no address, not referencable, provide definition);
  - template and overloaded function names (fails, explicitly cast to desired function pointer type);
  - bitfields (address not aligned, not non-const referencable, create a temp copy).


For example, it does **not** distinguish whether an lvalue is a bit-field lvalue,
**nor** does it capture whether the expression has a specific constant value.
The latter causes problems particularly when we’re dealing with the null pointer constant `NULL`,
which is a value of integral type that evaluates to the constant value zero.
Since the constant value of an expression is **not** captured by perfect forwarding,
overload resolution in the following example will behave differently
for the direct call to `g` than for the forwarded call to `g`:
```c++
void g(int);
void g(int *);

template <typename T> 
void forwardToG(T && x)
{
    g(std::forward<T>(x));
}

void foo()
{
    g(NULL);           // g(int *) 
    forwardToG(NULL);  // g(int) !
}
```
This is yet another reason to use `nullptr` (introduced in C++11) instead of null pointer constants:
```c++
g(nullptr);           // g(int *) 
forwardToG(nullptr);  // g(int *)
```

##### Perfect Forwarding for Variadic Templates

```c++
template <typename ... Ts> 
void forwardToG(Ts && ... xs)
{
    g(std::forward<Ts>(xs)...);
}
```
All of our examples of perfect forwarding have focused on forwarding the function arguments
while maintaining their _precise_ type (**not** decayed!) and value categories.
The same problem occurs when forwarding the return value of a call to another function,
with precisely the same type and value category.
```c++
// C++11
template <typename ... Ts>
auto forwardToG(Ts && ... xs) -> decltype(g(std::forward<Ts>(xs)...))
{
    return g(std::forward<Ts>(xs)...);  // forward all xs to g()
}

// C++14
template <typename ... Ts>
decltype(auto) forwardToG(Ts && ... xs)
{
    return g(std::forward<Ts>(xs)...);  // forward all xs to g()
}
```

##### 15.6.4 Deduction Surprises

The results of the special deduction rule for rvalue references are very useful for perfect forwarding.
However, they can come as a surprise, because function templates typically generalize the types in the function signature
**without** affecting what kinds of arguments (lvalue or rvalue) it allows:
```c++
void int_lvalues(int &);

template <typename T>
void lvalues(T &);

void int_rvalues(int &&);

template <typename T>
void anything(T &&);  // SURPRISE: accepts lvalues and rvalues of any type
```
Programmers who are simply abstracting a concrete function like `int_rvalues` to its template equivalent
would likely be surprised by the fact that the function template `anything` accepts lvalues.
Fortunately, this deduction behavior only applies
when the function parameter is written specifically with the form `T &&`,
is part of a function template,
and the named template parameter is declared by that function template.
Therefore, this deduction rule does **not** apply in any of the following situations:
```c++
template <typename T>
class X
{
public:
    X(X &&);     // X is not a template parameter

    X(T &&);     // this constructor is not a function template

    template <typename Other>
    X(X<U> &&);  // X<U> is not a template parameter

    template <typename U>
    X(U, T &&);  // T is a template parameter from an outer template
}
```
One can use a combination of SFINAE (see Section 8.4 and Section 15.7)
and type traits such as `std::enable_if` (see Section 6.3 and Section 20.3)
to restrict the template to rvalues:
```c++
template <typename T>
std::enable_if_t<!std::is_lvalue_reference_v<T>>
rvalues(T &&) {}
```


#### 📌 15.7 SFINAE (Substitution Failure Is Not An Error)


The SFINAE (Substitution Failure Is Not An Error) principle is an important aspect of template argument deduction
that prevents unrelated function templates from causing errors during overload resolution.
SFINAE also applies to the substitution of partial class template specializations (see Section 16.4).
```c++
template <typename T, unsigned N>
T * begin(T (& array)[N])
{
    return array;
}

template <typename Container>
typename Container::iterator begin(Container & c)
{
    return c.begin();
}

int main()
{
    std::vector<int> v;
    int a[10];
    ::begin(v);  // OK: only container begin matches, because the first deduction fails
    ::begin(a);  // OK: only array begin matches, because the second substitution fails
}
```

##### 15.7.1 Immediate Context

SFINAE protects against attempts to form invalid types or expressions,
including errors due to ambiguities or access control violations,
that occur within the _immediate context_ of the function template substitution.


The _immediate context_ includes many things,
including various kinds of lookup, alias template substitutions, overload resolution, etc.
Arguably the term is a bit of a misnomer,
because some of the activities it includes are not closely tied to the function template being substituted.


SFINAE errors are only the failures in the types and expressions in the _immediate context_ of:
- The function type;
- Its template parameter types;
- Its [explicit specifier](https://en.cppreference.com/w/cpp/language/explicit)


Specifically, the following errors are **not** covered by SFINAE (and thus result in compilation errors):
- Anything that happens _during the instantiation_ of
  - The definition of a class template (i.e., its "body" and list of base classes);
  - The definition of a function template ("body" and, in the case of a constructor, its constructor-initializers);
  - The initializer of a variable template;
  - A default argument;
  - A default member initializer;
  - An exception specification;
- Any implicit definition of special member functions triggered by the substitution process.


So if substituting the template parameters of a function template declaration
requires the _instantiation_ of the body of a class template because a member of that class is being referred to,
an error during that instantiation is **not** in the _immediate context_ of the function template substitution,
and is therefore a real error (even if another function template matches without error):
```c++

template <typename T>
class Array
{
public:
    using iterator = T *;
};

template <typename T>
void f(Array<T>::iterator first, Array<T>::iterator last);

template <typename T>
void f(T *, T *);

int main()
{
    f<int &>(0, 0);  // ERROR: substituting int & for T in the first function template
}
// Instantiation of Array<int &> fails
// Error happens in the instantiation of Array, so not SFINAE
```
The main difference between a SFINAE error and a hard error and  is _where the failure occurs_.
- Consider `typename std::enable_if<Cond>::type`.
  The _instantiation_ of `std::enable_if` _always succeed_.
  An error will happen when accessing its `type` member for an already-instantiated `std::enable_if<false>` class.
- In this example, the failure occurs _in_ the _instantiation_ of `Array<int &>`,
  which actually occurs in the context of the class template `Array`,
  although `Array` was triggered from `f`'s context.
  Therefore, the SFINAE principle does **not** apply, and the compiler will produce an error.


Here is a C++14 example relying on deduced return types (see Section 15.10.1)
that involves an error during the instantiation of a function template definition:
```c++
template <typename T> 
auto f(T p) 
{
    return p->m;
}

int f(...);

template <typename T> 
auto g(T p) -> decltype(f(p));

int main()
{
    g(42);
}
```
The call `g(42)` deduces `T` to be i`n`t.
Making that substitution in the declaration of `g` requires us to determine the type of `f(p)`
(where `p` is now known to be of type `int`) and therefore to determine the return type of `f`.
There are two candidates for `f`.
The non-template candidate is a match, but not a very good one because it matches with an ellipsis parameter.
Unfortunately, the template candidate has a deduced return type,
and so we must instantiate its definition to determine that return type.
That instantiation **fails** because `p->m` is **not** valid when `p` is an `int`,
and since the failure is outside the _immediate context_ of the substitution
(because it’s in a subsequent instantiation of a function definition),
the failure produces an error.
Because of this, we recommend avoiding deduced return types
if they can easily be specified explicitly.


SFINAE was originally intended to eliminate surprising errors
due to unintended matches with function template overloading,
as with the container `begin` example.
However, the ability to detect an invalid expression or type enables remarkable compile-time techniques,
allowing one to determine whether a particular syntax is valid.
These techniques are discussed in Section 19.4.


See especially Section 19.4.4 for an example of making a type trait _SFINAE-friendly_
to avoid problems due to the _immediate context_ issue.


#### 📌 15.8 Limitations of Deduction


##### 15.8.1 Allowable Argument Conversions

Normally, template deduction attempts to find a substitution of the function template parameters
that make the parameterized type `P` identical to type `A`.
However, when this is **not** possible, the following differences are tolerable:
- _Qualification conversion_ (adding `cv`-qualifiers).
  - Only bottom-level cv for reference and pointer parameter types.
  - Regular types only have top-level cv, which is decayed.
- _Derived-to-base conversion_ (for regular types or pointer types),  
  **unless** deduction occurs for a conversion operator template.
```c++
template <typename T>
class B {};

template <typename T>
class D : public B<T> {};

template <typename T> 
void f(B<T> *) {}

void g(D<long> dl)
{
    f(&dl);  // deduction succeeds with T substituted with long
}
```
If `P` does **not** contain _a template parameter in a deduced context_,
then all implicit conversion are permissible:
```c++
template <typename T>
void f(T, typename T::X) {}

struct V
{
    struct X { X(double) {} };

    V() = default;
};

// OK:
// T is deduced to V through the first parameter,
// which causes the second parameter to have type V::X (not deduced because already clear),
// which can be constructed from a double value
f(V {}, 7.0);
```

##### 15.8.2 Class Template Arguments

Prior to C++17, template argument deduction applied
exclusively to function templates and member function templates.
The arguments for a class template were **not** deduced:
```c++
template <typename T>
class S
{
public:
    S(T b) : a {b} {}

private:
    T a;
};

// ERROR before C++17: 
// the class template parameter T was not deduced
// from the constructor call argument 12
S x(12);
```

##### 15.8.3 Default Call Arguments

Default function call arguments can be specified in function templates
just as they are in ordinary functions.
The default function call argument can depend on a template parameter.
Such a dependent default argument is instantiated only if **no** explicit argument is provided:
```c++
template <typename T>
void init(T * loc, T const & val = T())
{
    *loc = val;
}

class S
{
public:
    S(int, int);
};

S s(0, 0);

int main()
{
    // T() is invalid for T = S, 
    // but the default call argument T() needs no instantiation
    // because an explicit argument is given
    init(&s, S(7, 42));
}
```
Even when a default call argument is **not** dependent,
it can **not** be used to deduce template arguments:
```c++
template <typename T>
void f (T x = 42) {}

f<int>();  // OK: T = int
f();       // ERROR: can not deduce T from default call argument
```

##### 15.8.4 Exception Specifications

Exception specifications are also only instantiated when they are needed.
This means that they do **not** participate in template argument deduction.
```c++
// #1
template <typename T>
void f(T, int) noexcept(nonexistent(T()));

// #2 (C-style vararg function)
template <typename T>
void f(T, ...);

void test(int i)
{
    // ERROR: chooses #1 , but the expression nonexistent(T()) is ill-formed
    f(i, i);
}
```
The noexcept specification in function `#1` tries to call a nonexistent function.
Normally, such an error directly within the declaration of the function template
would trigger a template argument deduction failure (SFINAE),
allowing the call `f(i, i)` to succeed by selecting function `#2`.
However, because exception specifications do **not** participate in template argument deduction,
overload resolution selects `#1` and the program becomes ill-formed
when the noexcept specification is later instantiated.


The same rules apply to exception specifications that list the potential exception types:
```c++
// #1
template <typename T>
void g(T, int) throw(typename T::Nonexistent);

// #2
template <typename T>
void g(T, ...);

void test(int i)
{
    // ERROR: chooses #1, but the type T::Nonexistent is ill-formed
    g(i, i);
}
```
However, dynamic exception specifications have been deprecated since C++11 and were removed in C++17.


#### 📌 15.9 Explicit Function Template Arguments


It is possible to explicitly specify it following the function template name.
This is useful when a function template argument can **not** be deduced:
```c++
template <typename T>
void compute(T p) {}

template <typename T>
T default_value()
{
    return T {};
}

int main(int argc, char * argv[])
{
    compute<double>(2);    // 2 implicitly converted to double; 
                           // not possible without explicit specification
    default_value<int>();  // non-deductible
    
    return EXIT_SUCCESS;
}
```
Once a template argument is explicitly specified,
its corresponding parameter is no longer subject to deduction.
That, allows conversions to take place on the function call parameter
that would **not** be possible in a deduced call.
In the example above, the argument `2` in the call `compute<double>(2)`
will be implicitly converted to `double`.


It is possible to explicitly specify some template arguments while having others be deduced.
However, the explicitly specified ones are always matched left-to-right with the template parameters.
Therefore, parameters that can **not** be deduced (or that are likely to be specified explicitly)
should be specified first.
```c++
template <typename Out, typename In>
Out convert(In p) { /* ... */ }

int main()
{
    // the type of parameter p is deduced,
    // but the return type is explicitly specified
    auto x = convert<double>(42);
}
```
It is occasionally useful to specify an empty template argument list
to ensure the selected function is a template instance
while still using deduction to determine the template arguments:
```c++
int f(int) {}  

template <typename T>
T f(T) {}

int main()
{
    auto x = f(42);    // f(int)
    auto y = f<>(42);  // f(T) [with T = int]
}
```
In the context of friend function declarations,
the presence of an explicit template argument list has an interesting effect:
```c++
void f();

template <typename>
void f();

namespace N
{

class C
{
    friend int f();    // OK

    friend int f<>();  // ERROR: return type conflict
};

}  // namespace N
```
When a plain identifier is used to name a friend function,
that function is only looked up within the nearest enclosing scope,
and if it is not found there, a new entity is _declared_ in that scope
(but it remains "invisible" except when looked up via ADL).
That is what happens with our first friend declaration above:
No `f` is declared within namespace `N`, and so a new `N::f` is "invisibly" declared.


However, when the identifier naming the friend is followed by a template argument list,
a template must be visible through normal lookup at that point,
and normal lookup will go up any level of (enclosing) scopes.
So, our second declaration above will find the global function template `f`,
but the compiler will then issue an error because the return types do **not** match
(since no ADL is performed here, the declaration of `f` one line above is not found).


Explicitly specified template arguments are substituted using SFINAE principles:
If the substitution leads to an error in the immediate context of that substitution,
the function template is discarded, but other templates may still succeed:
```c++
template <typename T> 
typename T::EType f() {}  // #1

template <typename T> 
T f() {}                  // #2

int main() 
{
    auto x = f<int *>();
}
```
```c++
template <typename T> 
void f(T) {}

template <typename T> 
void f(T, T) {}

int main() 
{
    auto x = f<int *>;  // ERROR: Ambiguous. Two possible f<int *> here
}
```
is invalid because `f<int *>` does **not** identify a single function in that case.


Variadic function templates can also be used with explicit template arguments.
```c++
template <typename ... Ts>
void f(Ts ... ps);

void foo()
{
    f<double, double, int>(1, 2, 3);
}
```
A pack can be partially explicitly specified and partially deduced.
```c++
template <typename ... Ts>
void f(Ts ... ps);

void foo()
{
    f<double, int>(1, 2, 3);
}
```


#### 📌 15.10 Deduction from Initializers and Expressions


##### 15.10.1 `auto` And `decltype`

Refer to Effective Modern C++ for details.

##### Deducible Non-type Parameters

Prior to C++17, non-type template arguments had to be declared with a specific type.
That type could be a template parameter type:
```c++
// Prior to C++17
template <typename T, T V> 
struct S {};

S<int, 42> s;
```
C++17 added the ability to declare non-type template parameters
whose actual types are deduced from the corresponding template argument:
```c++
template <auto V> 
struct S {};

S<42> s;
```
Here the type of `V` for `S<42>` is deduced to be `int` because `42` has type `int`.
Had we written `S<42U>` instead, the type of `V` would have been deduced to be `unsigned int`.


Note that the general constraints on the type of non-type template parameters remain in effect:
```c++
S<3.14> d;  // ERROR: floating-point non-type argument not allowed
```
A template definition with that kind of deducible non-type parameter
often also needs to express the actual type of the corresponding argument.
That is easily done using the `decltype` construct:
```c++
template <auto V> 
struct Value 
{
    using ArgType = decltype(V);
};
```
`auto` non-type template parameters are also useful
to parameterize templates on members of classes:
```c++
template <typename>
struct PMClassT {};

template <typename C, typename M>
struct PMClassT<M C::*>
{
    using Type = C;
};

template <typename PM> 
using PMClass = typename PMClassT<PM>::Type;

template <auto PMD>
class CounterHandle
{
public:
    CounterHandle(PMClass<decltype(PMD)> & c) : c(c) {}

    void incr() { ++(c.*PMD); }
    
private:
    PMClass<decltype(PMD)> & c;
};

struct S
{
    int i;
};

int main()
{
    S s {41};
    CounterHandle<&S::i> h(s);
    h.incr();  // increases s.i
    
    return EXIT_SUCCESS;
}
```
Prior to C++17, we also have to specify a pointer-member-type
```c++
OldCounterHandle<int S::*, &S::i>
```
This feature can also be used for non-type parameter packs:
```c++
template <auto ... VS>
struct Values {};

Values<1, 2, 3> beginning;
Values<1, 'x', nullptr> triplet;
```
The triplet example shows that each non-type parameter element of the pack
can be deduced to a distinct type.


If we want to force a homogeneous pack of non-type template parameters,
that is possible too:
```c++
template <auto V1, decltype(V1) ... VRest> 
struct HomogeneousValues {};
```
However, the template argument list can **not** be empty in that particular case.

##### 15.10.2 Expressing the Type of an Expression with `decltype`

The `decltype` keyword allows a programmer to express the _precise_ type of an expression or declaration.
However, programmers should be careful about a subtle difference in what `decltype` produces,
depending on whether the passed argument is a declared entity or an expression:
- If `e` is the name of an entity (such as a variable, function, enumerator, or data member) or a class member access,
  `decltype(e)` yields the declared type of that entity or the denoted class member.
- Otherwise, if `e` is any other expression,
  `decltype(e)` produces a type that reflects the type and value category of that expression as follows:
  - If `e` is an `lvalue` of type `T`, `decltype(e)` produces `T &`.
  - If `e` is an `xvalue` of type `T`, `decltype(e)` produces `T &&`.
  - If `e` is a `prvalue` of type `T`, `decltype(e)` produces `T`.
```c++
void g(std::string && s)
{
    // check the type of s: 
    std::is_lvalue_reference_v<decltype(s)>;        // false
    std::is_rvalue_reference_v<decltype(s)>;        // true
    std::is_same_v<decltype(s), std::string &>;     // false
    std::is_same_v<decltype(s), std::string &&>;    // true

    // check the value category of s used as expression: 
    std::is_lvalue_reference_v<decltype((s))>;      // true
    std::is_rvalue_reference_v<decltype((s))>;      // false
    std::is_same_v<decltype((s)), std::string &>;   // true
    std::is_same_v<decltype((s)), std::string &&>;  // false
}
```

##### 15.10.3 `decltype(auto)`

Since C++17, `decltype(auto)` can also be used for deducible non-type parameters.
```c++
template <decltype(auto) Val> 
class S {};

constexpr int c = 42;
extern int v = 42;

S<c> sc;    // #1 produces S<42>
S<(v)> sv;  // #2 produces S<static_cast<int &>(v)>
```
`decltype` as call parameter is non-deduced context:
```c++
template <auto V> 
int f(decltype(V) p) {}

int r1 = f<42>(42);  // OK
int r2 = f(42);      // ERROR: decltype(V) is a non-deduced context
```
In this case, `decltype(V)` is a non-deduced context:
There is **no** unique value of `V` that matches the argument `42`
(e.g., `decltype(7)` produces the same type as `decltype(42)`).
Therefore, the non-type template parameter must be specified explicitly to be able to call this function.

##### 15.10.4 Special Situations for `auto` Deduction

`auto` with initializer list:
```c++
template <typename T>
void deduceT (T) {}

deduceT({2, 3, 4});  // ERROR
deduceT({1});        // ERROR
```
```c++
template <typename T>
void deduceInitList(std::initializer_list<T>);

deduceInitList({2, 3, 5, 7});  // OK: T deduced as int
```
Copy-initializing an `auto` variable with an initializer list
is therefore defined in terms of that more specific parameter:
```c++
auto primes = {2, 3, 5, 7};  // OK: primes is std::initializer_list<int>
deduceT(primes);             // OK: T deduced as std::initializer_list<int>
```
Before C++17, the corresponding direct-initialization of `auto` variables was also handled that way,
but this was changed in C++17 to better match the behavior expected by most programmers:
```c++
auto oops {0, 8, 15};  // ERROR in C++17
auto val {2};          // OK: val has type int in C++17
```
Prior to C++17, both initializations were valid,
initializing both `oops` and `val` of type `initializer_list<int>`.


It is invalid to return a braced initializer list for a function with a _deducible placeholder type_:
```c++
auto subtleError() 
{
    return {1, 2, 3};
}
```
That is because an initializer list in function scope
is an object that _points to_ an underlying stack array object
(that expires when the function returns).
Allowing the construct would thus encourage what is in effect a _dangling reference_.


Another special situation occurs when multiple variable declarations share the same `auto`:
```c++
auto first = container.begin(), last = container.end();
```
In such cases, deduction is performed independently for each declaration.
In other words, there is an invented template type parameter `T1` for first
and another invented template type parameter `T2` for last.
Only if both deductions succeed, and the deductions for `T1` and `T2` are the same type,
are the declarations well-formed. This can produce some interesting cases:
```c++
char c;
auto * cp = &c, d = c;  // OK
auto e = c, f = c + 1;  // ERROR
```
A somewhat parallel special situation can also occur with placeholders for deduced return types.
```c++
auto f(bool b) 
{
    if (b) 
    {
        return 42.0;  // deduces return type double
    }
    else 
    {
        return 0;     // deduces return type double
    }
}
```
If the returned expression calls the function recursively,
deduction can **not** occur and the program is invalid
unless a prior deduction already determined the return type:
```c++
auto f(int n)
{
    if (n > 1)
    {
        return n * f(n - 1);  // ERROR: type of f(n-1) unknown
    }
    else
    {
        return 1;
    }
}
```
but the following otherwise equivalent code is fine:
```c++
auto f(int n)
{
    if (n <= 1)
    {
        return 1;             // return type is deduced to be int
    }
    else
    {
        return n * f(n - 1);  // OK: type of f(n-1) is int and so istype of n*f(n-1)
    }
}
```
Deduced return types have another special case with no counterpart
in deduced variable types or deduced non-type parameter types:
```c++
auto f1() {}           // OK: return type is void
auto f2() { return; }  // OK: return type is void
```
However, if the return type pattern can **not** match `void`, such cases are invalid:
```c++
auto * f3() {}         //ERROR: auto * can not deduce as void
```
Any use of a function template with a deduced return type
requires the _immediate instantiation of that template_
to determine the return type with certainty.
This instantiation is **not** in the _immediate context of the function call_
and is thus **not** covered by SFINAE:
```c++
template <typename T, typename U>
auto addA(T t, U u) -> decltype(t + u)
{
    return t + u;
}

void addA(...) {}

template <typename T, typename U>
auto addB(T t, U u) -> decltype(auto)
{
    return t + u;
}

void addB(...) {}

struct X {};

// OK: AddResultA is void
using AddResultA = decltype(addA(X(), X()));

// ERROR: 
// Instantiation of addB<X, X> is ill-formed because there is no operator+(const X &, const X &). 
// This error is NOT covered by SFINAE so results in compiling error (even though there is another match)! 
using AddResultB = decltype(addB(X(), X()));
```
Here, the use of `decltype(auto)` rather than `decltype(t + u)` for `addB`
causes an error during overload resolution:
The function body of the `addB` template must be fully instantiated to determine its return type.
That instantiation **isn’t** in the immediate context of the call to `addB`,
and therefore **doesn’t** fall under the SFINAE filter but results in an outright error.


Deduced return types should be used with care,
that they **shouldn’t** be called in the signatures
of other function templates that would count on SFINAE properties.

##### 15.10.5 Structured Bindings

Syntactically, a structured binding must always have an `auto` type
optionally extended by `cv` qualifiers, `&` or `&&`,
but **not** a `*` pointer declarator or some other declarator construct.
It is followed by a bracketed list containing at least one identifier.
That in turn has to be followed by an initializer.


Three different kinds of entities can initialize a structured binding:
1. **Simple class type, where all the non-static data members are public**.
  - All the nonstatic data members have to be public
    (either all directly in the class itself
    or all in the same unambiguous public base class.
    No anonymous unions may be involved).
  - The number of bracketed identifiers must equal the number of members.
  - Using one of these identifiers within the scope of the structured bindings
    amounts to using the corresponding member of the object,
    with all the associated properties.
    - E.g., if the corresponding member is a bit field, it is invalid to take its address.
2. **Arrays**.
  - The bracketed initializers are just shorthand for the corresponding elements of the unnamed array variable.
  - The number of array elements must equal the number of bracketed initializers.
   ```c++
   double pt[3];
   auto & [x, y, z] = pt;
   x = 3.0; y = 4.0; z = 0.0;
   ```
   ```c++
   auto f() -> int (&)[2]; 
   auto [x, y] = f();       // #1
   auto & [r, s] = f();     // #2
   ```
   Line `#1` is special:
   Ordinarily, the entity `e` would be deduced like this:
   ```c++
   auto e = f();
   ```
   However, that would deduce the _decayed_ pointer to the array,
   which is **not** what happens when performing the structured binding of an array.
   Instead, `e` is deduced to be a variable of type corresponding to the type of the initializer.
   Then that array is _copied_ from the initializer, element by element:
   That is a somewhat unusual concept for built-in arrays.
   The other two places where built-in arrays are copied are lambda captures and generated copy constructors.
   Finally, `x` and `y` become aliases for the expressions `e[0]` and `e[1]`, respectively.   
   Line `#2`, does **not** involve array copying and follows the usual rules for `auto`.
   So the hypothetical `e` is declared as follows:
   ```c++
   auto & e = f();
   ```
   which yields a reference to an array,
   and `x` and `y` again become aliases for the expressions `e[0]` and `e[1]`.
3. **`std::tuple`-like classes**.
  - Decomposed through a template-based protocol using `get`.
  - Let `E` be `std::remove_reference_t<decltype((e))>` with `e` declared as above.
    If the expression `std::tuple_size<E>::value` is a valid integral constant expression,
    it must equal the number of bracketed identifiers.
  - Denote the bracketed identifiers by `n0`, `n1`, `n2`, and so forth.
    - If `e` has any member named `get`, then the behavior is as if these identifiers are declared as:
      - If `e` was deduced to have a reference type
      ```c++
      std::tuple_element<i, E>::type & ni = e.get<i>();
      ```
      - Otherwise
      ```c++
      std::tuple_element<i, E>::type && ni = e.get<i>();
      ```
    - If `e` has **no** member `get`, then the corresponding declarations become one of the following:
    ```c++
    std::tuple_element<i, E>::type & ni = get<i>(e);
    std::tuple_element<i, E>::type && ni = get<i>(e);
    ```
    where `get` is only looked up in associated classes and namespaces.
    (In all cases, `get` is assumed to be a template and therefore the `<` follows is an angle bracket.)
  - The `std::tuple`, `std::pair`, and `std::array` templates all implement this protocol
   ```c++
   std::tuple<bool, int> bi{true, 42};
   auto [b, i] = bi;
   int r = i;
   // initializes r to 42
   ```
However, it is not difficult to add specializations of `std::tuple_size`, `std::tuple_element`,
and a function template or member function template `get`
that will make this mechanism work for an arbitrary class or enumeration type:
```c++
enum M {};

template <>
class std::tuple_size<M>
{
public:
    // map M to a pair of values
    static unsigned const value = 2; 
};

template <>
class std::tuple_element<0, M>
{
public:
    // the first value will have type int
    using type = int;
    
};

template <>
class std::tuple_element<1, M>
{
public:
    // the second value will have type double
    using type = double;
};

template <int>
auto get(M);

template <>
auto get<0>(M)
{
    return 42;
}

template <>
auto get<1>(M)
{
    return 7.0;
}

// as if: int && i = 42; double && d = 7.0;
auto [i, d] = M(); 
```
In addition, note that the third case above (using the tuple-like protocol)
performs an actual initialization of the bracketed initializers
and the bindings are actual reference variables.
They are **not** just aliases for another expression
(unlike the first two cases using simple class types and arrays).


That’s of interest because that reference initialization could go wrong.
For example, it might throw an exception, and that exception is now unavoidable.

##### 15.10.6 Generic Lambdas

Within templates, lambdas can become fairly verbose
due to the need to spell out the parameter and result types.
```c++
// C++11
template <typename Iter>
Iter findNegative(Iter first, Iter last)
{
    return std::find_if(first, last, [](typename std::iterator_traits<Iter>::value_type value)
    {
        return value < 0;
    });
}

// C++14
template <typename Iter>
Iter findNegative(Iter first, Iter last)
{
    return std::find_if(first, last, [](auto value)
    {
        return value < 0;
    });
}
```
An `auto` in a parameter of a lambda is handled similarly to
an `auto` in the type of a variable with an initializer:
It is replaced by an invented template type parameter `T`.


However, unlike in the variable case, the deduction **isn’t** performed immediately
because the argument **isn’t** known at the time the lambda is created.
Instead, the lambda itself becomes generic (if it wasn’t already),
and the invented template type parameter is added to its template parameter list.
Thus, the lambda above can be invoked with any argument type,
so long as that argument type supports the `< 0` operation whose result is convertible to `bool`.
For example, this lambda could be called with either an `int` or a `float` value.


C++ compiler translates a lambda expression into a _closure_ or _closure object_,
and the class is called a _closure type_.
The closure type has a function call operator,
and hence the closure is a function object.
```c++
[] (int i) 
{
    return i < 0;
}
```
```c++
// leaving out the conversion function to a pointer-to-function value
class SomeCompilerSpecificNameX
{
public:
    // only callable by the compiler
    SomeCompilerSpecificNameX(); 
    
    bool operator()(int i) const
    {
        return i < 0;
    }
};
```

This translation model of lambdas is actually used in the specification of the C++ language,
making it both a convenient and an accurate description of the semantics.
Captured variables become data members,
the conversion of a non-capturing lambda to a function pointer is modeled
as a conversion function in the class, and so on.


Because lambdas are function objects,
whenever rules for function objects are defined, they also apply to lambdas.
E.g., if you check the type category for a lambda, `std::is_class` will yield `true`.


If the lambda were to capture local variables,
those captures would be modeled as initializing members of the associated class type:
```c++
int x, y;

[x, y](int i) 
{
    return x < i && i < y;
}
```
```c++
class SomeCompilerSpecificNameY
{
public:
    // only callable by compiler
    SomeCompilerSpecificNameY(int x, int y) : _x(x), _y(y) {}

    bool operator()(int i) const
    {
        return i > _x && i < _y;
    }

private:
    int _x, _y;
};
```
For a generic lambda, the function call operator becomes a member function template:
```c++
[] (auto i) 
{
    return i < 0;
}
```
```c++
class SomeCompilerSpecificNameZ
{
public:
    // only callable by compiler
    SomeCompilerSpecificNameZ() = default;
    
    template <typename T>
    auto operator()(T i) const
    {
        return i < 0;
    }
};
```
The member function template is instantiated when the closure is invoked,
which is usually **not** at the point where the lambda expression appears:
```c++
template <typename F, typename ... Ts>
void invoke(F f, Ts ... ps)
{
    f(ps...);
}

int main()
{
    invoke([](auto x, auto y)
           {
               std::cout << x + y << '\n';
           },
           21, 21);
}
```
Here the lambda expression appears in `main`, and that’s where an associated closure is created.
However, the call operator of the closure **isn’t** instantiated at that point.
Instead, the `invoke` function template is instantiated with the closure type
as the first parameter type and `int` (the type of `21`) as a second and third parameter type.
That instantiation of invoke is called with a _copy_ of the closure
(which is still a closure associated with the original lambda),
and it instantiates the `operator()` template of the closure to satisfy the instantiated call `f(ps...)`.


#### 📌 15.11 Alias Templates


Alias templates are transparent with respect to deduction.
That means that wherever an alias template appears with some template arguments,
that alias’s definition (i.e., the type to the right of the `=`) is substituted with the arguments,
and the resulting pattern is what is used for the deduction.
They can be used to clarify and simplify code, but have **no** effect on how deduction operates.
```c++
template <typename T, typename Cont>
class Stack {};

template <typename T>
using DequeStack = Stack<T, std::deque<T>>;

template <typename T, typename Cont>
void f1(Stack<T, Cont>) {}

template <typename T>
void f2(DequeStack<T>) {}

template <typename T>
void f3(Stack<T, std::deque<T>>) {}

void test(DequeStack<int> intStack)
{
    f1(intStack);  // OK: T deduced to int, Cont deduced to std::deque<int>
    f2(intStack);  // OK: T deduced to int
    f3(intStack);  // OK: T deduced to int
}
```
For the purposes of template argument deduction, template aliases are transparent:
They can be used to clarify and simplify code but have **no** effect on how deduction operates.
Note that this is possible because alias templates can **not** be specialized
(see Chapter 16 for details on the topic of template specialization).
Suppose the following were possible:
```c++
template <typename T> 
using A = T;

template <> 
using A<int> = void;  // ERROR, but suppose it were possible...
```
Then we would **not** be able to match `A<T>` against type `void` and conclude that `T` must be `void`
because both `A<int>` and `A<void>` are equivalent to `void`.
The fact that this is **not** possible guarantees that
each use of an alias can be generically expanded according to its definition,
which allows it to be transparent for deduction.


#### 📌 15.12 Class Template Argument Deduction


C++17 introduces a new kind of deduction:
Deducing the template parameters of a class type
from the arguments specified in an initializer of a variable declaration
or a functional-notation type conversion:
```c++
template <typename T1, typename T2, typename T3 = T2>
class C
{
public:
    // constructor for 0, 1, 2, or 3 arguments:
    C(T1 x = T1 {}, T2 y = T2 {}, T3 z = T3 {});
};

C c1(22, 44.3, "hi");  // OK in C++17: T1 is int, T2 is double, T3 is char const *
C c2(22, 44.3);        // OK in C++17: T1 is int, T2 and T3 are double
C c3("hi", "guy");     // OK in C++17: T1, T2, and T3 are char const *
C c4;                  // ERROR: T1 and T2 are undefined
C c5("hi");            // ERROR: T2 is undefined
```
Note that _all_ parameters must be determined by the deduction process or from default arguments.
It is **not** possible to explicitly specify a few arguments and deduce others:
```c++
C<std::string> c10("hi","my", 42);           // ERROR: only T1 explicitly specified, T2 not deduced
C<> c11(22, 44.3, 42);                       // ERROR: neither T1 nor T2 explicitly specified
C<std::string, std::string> c12("hi","my");  // OK: T1 and T2 are deduced, T3 has default
```

##### 15.12.1 Deduction Guides

```c++
template <typename T>
class S
{
public:
    S(T b) : a {b} {}

private:
    T a;
};

template <typename T>
S(T) -> S<T>;     // deduction guide

S x {12};         // OK since C++17, same as: S<int> x {12};
S y(12);          // OK since C++17, same as: S<int> y(12);
auto z = S {12};  // OK since C++17, same as: auto z = S<int> {12};
```
Note in particular the addition of a new template-like construct called a _deduction guide_.
It looks a little like a function template,
but it differs syntactically from a function template in a few ways:
- The part that looks like a trailing return type can **not** be written as a traditional return type.
  We call the type it designates (`S<T>` in our example) as the _guided type_.
- There is **no** leading `auto` keyword to indicate that a trailing return type follows.
- The name of a deduction guide must be the unqualified name of a class template declared earlier in the same scope.
- The guided type of the guide must be a `template-id` whose template name corresponds to the guide name.
- It can be declared with the `explicit` specifier.


In the declaration `S x(12);` the specifier `S` is called a _placeholder class type_.
Note the distinction between a _placeholder type_,
which is `auto` or `decltype(auto)` and can resolve to any kind of type.
A _placeholder class type_,
which is a template name and can only resolve to a class type
that is an instance of the indicated template.


When such a placeholder is used, the name of the variable being declared
must follow immediately and that in turn must be followed by an initializer:
```c++
S * p = &x;  // ERROR: syntax not permitted
```
With the guide as written in the example,
the declaration `S x(12);` deduces the type of the variable
by treating the deduction guides associated with class `S` as an overload set
and attempting overload resolution with the initializer against that overload set.
In this case, the set has only one guide in it,
and it successfully deduces `T` to be `int` and the guide’s guided type to be `S<int>`.
That guided type is therefore selected as the type of the declaration.


As with ordinary function template deduction,
SFINAE could apply if, for example, substituting the deduced arguments in the guided type failed.
That is not the case in this simple example.


Note that in the case of multiple declarators following a class template name requiring deduction,
the initializer for each of those declarators has to produce the same type:
```c++
S s1(1), s2(2.0);  // ERROR: deduces S both as S<int> and S<double>
```
This is similar to the constraints when deducing the C++11 placeholder type `auto`.


In the previous example, there is an implicit connection
between the deduction guide we declared and the constructor `S(T b)` declared in class `S`.
However, such a connection is **not** required,
which means that deduction guides can be used with aggregate class templates:
```c++
template <typename T>
struct A
{
    T val;
};

template <typename T> 
A(T) -> A<T>;  // deduction guide
```
**Without** the deduction guide, we are _always required_ (even in C++17) to specify explicit template arguments.
```c++
A<int> a1 {42};    // OK
A<int> a2(42);     // ERROR: not aggregate initialization
A<int> a3 = {42};  // OK

A a4 = {42};       // OK
A a5(42);          // ERROR: not aggregate initialization
A a6 = 42;         // ERROR: not aggregate initialization
```

##### 15.12.2 Implicit Deduction Guides

Quite often, a deduction guide is desirable for every constructor in a class template.
There is thus an _implicit_ mechanism for the deduction.
It is equivalent to introducing for every _constructor_
and _constructor template of the primary class template_
an implicit deduction guide as follows.
(Chapter 16 introduces the ability to “specialize” class templates in various ways.
Such specializations do **not** participate in class template argument deduction.)
- The template parameter list for the implicit guide
  consists of the template parameters for the class template.
  - In the constructor template case,
    it is followed by the template parameters of the constructor template.
    The template parameters from the constructor template retain any default arguments.
- The "function-like" parameters of the guide
  are copied from the constructor or constructor template.
- The guided type of the guide is the name of the template
  with arguments that are the template parameters taken from the class template.
```c++
template <typename T>
class S
{
public:
    S(T b) : a {b} {}

private:
    T a;
};

// implicitly defined deduction guide from constructors:
// template <typename T>
// S(T b) -> S<T>;
```
The template parameter list is typename `T`,
the function-like parameter list becomes just `(T b)`,
and the guided type is then `S<T>`.
Thus, we obtain a guide that’s equivalent to the user-declared guide we wrote earlier:
That guide was therefore **not** required to achieve our desired effect!
That is, with just the simple class template as originally written (and **no** deduction guide),
we can validly write `S x(12);` with the expected result that `x` has type `S<int>`.


Deduction guides have an unfortunate ambiguity:
```c++
S s1 {12};  // s1 has type S<int>
S x {s1};
S y(s1);
```
We already saw that `s1` has type `S<int>`, but what should the type of `x` and `y` be?
The two types that arise intuitively are `S<S<int>>` and `S<int>`.
The C++ standard determines controversially that it should be `S<int>` in both cases.
Why is this controversial? Consider a similar example with a `std::vector` type:
```c++
std::vector v {1, 2, 3};  // std::vector<int>
std::vector w2 {v, v};    // std::vector<std::vector<int>>
std::vector w1 {v};       // std::vector<int>!
```
In other words, a braced initializer with one element
deduces differently from a braced initializer with multiple elements.
Often, the one-element outcome is what is desired, but the inconsistency is somewhat subtle.
In generic code, however, it is easy to miss the subtlety:
```c++
template <typename T, typename ... Ts>
auto f(T p, Ts ... ps)
{
    std::vector v {p, ps...};  // type depends on pack length!
}
```
Here it is easy to forget that if `T` is deduced to be a vector type,
the type of `v` will be _fundamentally different_ depending on whether `ps` is an empty pack.


The addition of implicit template guides themselves was **not** without controversy.
The main argument against their inclusion is that the feature automatically adds interfaces to existing libraries.
To understand this, consider once more our simple class template `S` above.
Its definition has been valid since templates were introduced in C++.
Suppose, however, that the author of `S` expands library causing `S` to be defined in a more elaborate way:
```c++
template <typename T>
struct ValueArg
{
    using Type = T;
};

template <typename T>
class S
{
public:
    using ArgType = typename ValueArg<T>::Type;

    S(ArgType b) : a {b} {}
    
private:
    T a;
};
```
Prior to C++17, transformations like these (which are **not** uncommon) did **not** affect existing code.
However, in C++17 they disable implicit deduction guides.
To see this, let’s write a deduction guide corresponding to the one
produced by the implicit deduction guide construction process outlined above:
The template parameter list and the guided type are unchanged,
but the function-like parameter is now `typename ValueArg<T>::Type`:
```c++
// before: ok
template <typename T>
S(T b) -> S<T>;

// after: non-deduced context!
template <typename T>
S(typename ValueArg<T>::Type) -> S<T>;
```
Recall from Section 15.2 that a name qualifier like `ValueArg<T>::` is **not** a deduced context.
So a deduction guide of this form is useless and will **not** resolve a declaration like `S x(12);`.
In other words, a library writer performing this kind of transformation is likely to break client code in C++17.


What is a library writer to do given that situation?
Our advice is to carefully consider for each constructor
whether you want to offer it as a source for an implicit deduction guide
for the remainder of the library’s lifetime.
If **not**, replace each instance of a deducible constructor parameter of type `X`
by something like `typename ValueArg<X>::Type`.
There is unfortunately no simpler way to “opt out” of implicit deduction guides.
(P.S. Hope we could have `= delete;` specifier for deduction guides in the future.)


#### 📌 15.12.3 Other Subtleties

##### Class Template Argument Deduction Disabled for Injected Class Names

In order to maintain backward compatibility,
class template argument deduction is **disabled**
if the name of the template is an _injected class name_.
```c++
template <typename T>
struct X
{
    template <typename Iter>
    X(Iter b, Iter e) {}

    template <typename Iter>
    auto f(Iter b, Iter e)
    {
        return X(b, e);  // What is this?
    }
};
```
This code is valid C++14:
The `X` in `X(b, e)` is the injected class name and is equivalent to `X<T>` in this context (see Section 13.2.3).
The rules for class template argument deduction, however, would naturally make that `X` equivalent to `X<Iter>`.
Thus class template argument deduction is **disabled** in such cases.

##### Implicit Deduction Guides Disabled for Forwarding References of Class Template Parameters

```c++
template <typename T>
struct Y
{
    Y(T const &);

    Y(T &&);
};

void g(std::string s)
{
    Y y = s;
}
```
Clearly, the intent here is that we deduce `T` to be `std::string`
through the implicit deduction guide associated with the copy constructor.
However, that would **not** happen:
```c++
template <typename T>
Y(T const &) -> Y(T);  // #1, implicitly defined by compiler

template <typename T>
Y(T &&) -> Y(T);       // #2, implicitly defined by compiler
```
Forwarding reference `T &&` behaves specially during template argument deduction:
It causes `T` to be deduced to a _reference type_ if the corresponding call argument is an `lvalue`.


In our example above, the argument in the deduction process is the expression `s`, which is an `lvalue`.
Implicit guide `#1` deduces `T` to be `std::string`
but requires the argument to be adjusted from `std::string` to `std::string const`.
Guide `#2`would normally deduce `T` to be a reference type `std::string &`
and produce a parameter of that same type (because of the reference collapsing rule),
which is a better match because no const must be added for type adjustment purposes.


This outcome would likely result in instantiation errors
(when the class template parameter is used in contexts that do **not** permit reference types)
or, worse, silent production of misbehaving instantiations (e.g., producing dangling references).


The C++ standardization committee therefore decided to **disable** the special deduction rule for `T &&`
when performing deduction for implicit deduction guides if the `T` was originally a class template parameter
(as opposed to a constructor template parameter. For those, the special deduction rule remains).
The example above thus deduces `T` to be `std::string`, as would be expected.

##### The `explicit` Keyword

A deduction guide can be declared with the keyword `explicit`.
It is then considered only for direct-initialization cases, **not** for copy-initialization cases.
```c++
template <typename T, typename U>
struct Z
{
    Z(T const &);

    Z(T &&);
};

template <typename T> 
Z(T const &) -> Z<T, T &>;    // #1

template <typename T> 
explicit Z(T &&) -> Z<T, T>;  // #2

Z z1 = 1;  // only considers #1; same as: Z<int, int&> z1 = 1;
Z z2 {2};  // prefers #2; same as: Z<int, int> z2 {2};
```

##### Copy Construction and Initializer Lists

```c++
template <typename ... Ts>
struct Tuple
{
    Tuple(Ts ...);

    Tuple(Tuple<Ts ...> const &);
};
```
To understand the effect of the implicit guides,
let’s write them as explicit declarations:
```c++
template <typename ... Ts> 
Tuple(Ts ...) -> Tuple<Ts ...>;

template <typename ... Ts> 
Tuple(Tuple<Ts ...> const &) -> Tuple<Ts ...>;
```
Now consider some examples:
```c++
auto x = Tuple {1, 2};
```
This clearly selects the first guide (as it has **two** arguments) and therefore the first constructor:
`x` is therefore a `Tuple<int, int>`.
Let’s continue with some examples that use syntax that is suggestive of copying `x`:
```c++
Tuple a = x;
Tuple b(x);
```
For both `a` and `b`, both guides match.
The first guide selects type `Tuple<Tuple<int, int>>`,
whereas the guide associated with the copy constructor produces `Tuple<int, int>`.
Fortunately, the second guide is a better match (more specialized than the first one),
and therefore both `a` and `b` are copy-constructed from `x`.


Now, consider some examples using braced initializer lists:
```c++
Tuple c {x, x};
Tuple d {x};
```
The first of these examples `x` can only match the first guide,
and so produces `Tuple<Tuple<int, int>, Tuple<int, int>>`.
That is entirely intuitive and **not** a surprise.
That would suggest that the second example should deduce `d` to be of type `Tuple<Tuple<int>>`.
Instead, it is treated as a _copy construction_ (because the second implicit guide is preferred).
This also happens with functional-notation casts:
```c++
auto e = Tuple {x};
```
Here, `e` is deduced to be a `Tuple<int, int>`, **not** a `Tuple<Tuple<int, int>>`.

##### Guides Are for Deduction Only

Deduction guides are **not** function templates:
They are only used to deduce template parameters and are **not** “called”.
That means that the difference between passing arguments by reference or by value
is **not** important for guiding declarations.
```c++
template <typename T>
struct X {};

template <typename T>
struct Y
{
    Y(X<T> const &);

    Y(X<T> &&);
};

template <typename T>
Y(X<T>) -> Y<T>;
```
Note how the deduction guide does **not** quite correspond to the two constructors of `Y`.
However, that does **not** matter, because the guide is only used for deduction.
Given a value `x` of type `X<T>`, no matter it is  `lvalue` or `rvalue`,
it will select the deduced type `Y<T>`.
Then, initialization will perform overload resolution on the constructors of `Y<T>`
to decide which one to call (which will depend on whether `x` is an `lvalue` or an `rvalue`).






### 🎯 Chapter 16 Specialization and Overloading


#### 📌 16.1 When “Generic Code” Doesn’t Quite Cut It


```c++
template <typename T>
class Array
{
public:
    Array(Array<T> const &);

    Array<T> & operator=(Array<T> const &);

    void exchangeWith(Array<T> * b)
    {
        T * tmp = data;
        data = b->data;
        b->data = tmp;
    }

    T & operator[](std::size_t k)
    {
        return data[k];
    }

private:
    T * data;
};

template <typename T>
inline void exchange(T * a, T * b)
{
    T tmp(*a);
    *a = *b;
    *b = tmp;
}
```
For simple types, the generic implementation of `exchange` works well.
However, for types with expensive copy operations,
the generic implementation may be much more expensive,
both in terms of machine cycles and in terms of memory usage,
than an implementation that is tailored to the particular, given structure.
In our example, the generic implementation requires
one call to the copy constructor of `Array<T>` and two calls to its copy-assignment operator.
For large data structures these copies can often involve copying relatively large amounts of memory.
However, the functionality of `exchange` could presumably often be replaced
just by swapping the internal data pointers,
as is done in the member function `exchangeWith`.

##### 16.1.1 Transparent Customization

In our previous example, the member function `exchangeWith` provides
an efficient alternative to the generic `exchange` function,
but the need to use a different function is inconvenient in several ways:
1. Users of the `Array` class have to remember an extra interface and must be careful to use it when possible;
2. Generic algorithms can generally **not** discriminate between various possibilities.
```c++
template <typename T>
void genericAlgorithm(T * x, T * y)
{
    exchange(x, y);  // How do we select the right algorithm?
}
```
Because of these considerations, C++ templates provide ways to customize function templates and class templates transparently.
For function templates, this is achieved through the overloading mechanism.
For example, we can write an overloaded set of `quickExchange` function templates as follows:
```c++
// #1
template <typename T>
void quickExchange(T * a, T * b)
{
    T tmp(*a);
    *a = *b;
    *b = tmp;
}

// #2
template <typename T>
void quickExchange(Array<T> * a, Array<T> * b)
{
    a->exchangeWith(b);
}

void demo(Array<int> * p1, Array<int> * p2)
{
    int x = 42;
    int y = -7;
    quickExchange(&x, &y);  // uses #1
    quickExchange(p1, p2);  // uses #2
}
```

##### 16.1.2 Semantic Transparency

Although both the generic algorithm and the one customized for `Array<T>` types
end up swapping the values that are being pointed to,
the _side effects_ of the operations are very different.
```c++
struct S 
{
    int x;
};

S s1;
S s2;

void distinguish(Array<int> a1, Array<int> a2)
{
    int * p = &a1[0];
    int * q = &s1.x;
    a1[0] = s1.x = 1;
    a2[0] = s2.x = 2;
    quickExchange(&a1, &a2);  // *p == 1 after this (still)
    quickExchange(&s1, &s2);  // *q == 2 after this
}
```
This example shows that a pointer `p` into the first `Array`
becomes a pointer into the _second_ array after `quickExchange` is called.
However, the pointer into the non-Array `s1` remains pointing into `s1` even after the exchange operation:
Only the values that were pointed to were exchanged.
The difference is significant enough that it may confuse clients of the template implementation.


However, the original generic `exchange` template can still have a useful optimization for `Array<T>`s:
```c++
template <typename T>
void exchange(Array<T> * a, Array<T> * b)
{
    T * p = &(*a)[0];
    T * q = &(*b)[0];
    
    for (std::size_t k = a->size(); k-- != 0;)
    {
        exchange(p++, q++);
    }
}
```
The advantage of this version over the generic code is that **no** (potentially) large temporary `Array<T>` is needed.
The `exchange` template is called recursively so that good performance is achieved
even for types such as `Array<Array<char>>`.
Note also that the more specialized version of the template is **not** declared `inline`
because it does a considerable amount of work of its own,
whereas the original generic implementation is `inline`
because it performs only a few operations (each of which is potentially expensive).


#### 📌 16.2 Overloading Function Templates


Two function templates with the same name can coexist,
even though they may be instantiated so that both have identical parameter types:
```c++
template <typename T>
int f(T)
{
    return 1;
}

template <typename T>
int f(T *)
{
    return 2;
}

void foo()
{
    f<int *>(static_cast<int *>(nullptr));  // int f(T) [with T = int *]
    f<int>(static_cast<int *>(nullptr));    // int f(T *) [with T = int]
}
```

##### 16.2.1 Signatures

Two functions can coexist in a program if they have distinct signatures.
We define the signature of a function as the following information:
1. The unqualified name of the function
   (or the name of the function template from which it was generated);
2. The class or namespace scope of that name,  
   and, if the name has internal linkage, the translation unit in which the name is declared;
3. The `cv` and value category qualification of the function
   (if it is a member function with such a qualifier);
4. The types of the function parameters
   (before template parameters are substituted if the function is generated from a function template);
5. Its **return type**, if the function is generated from a function template;
6. The template parameters and the template arguments,
   if the function is generated from a function template.


In principle, the following templates and their instantiations could coexist in the same program:
```c++
template <typename T1, typename T2>
void f1(T1, T2) {}

template <typename T1, typename T2>
void f1(T2, T1) {}

template <typename T>
long f2(T) {}

template <typename T>
char f2(T) {}
```
However, they can **not** always be used when they’re _declared in the same scope_
because instantiating both creates an overload ambiguity:
```c++
void foo()
{
    f1<char, char>('a', 'b');  // ERROR: ambiguous
    f2(42);                    // ERROR: ambiguous
}
```
If the templates appear in different translation units,
then the two instantiations can actually exist in the same program
(and the linker should **not** complain about duplicate definitions
because the signatures of the instantiations are distinct):
```c++
/// "a.cpp"
template <typename T1, typename T2>
void f1(T1, T2) {}

int main()
{
    f1('a', 'b');
    return EXIT_SUCCESS;
}

/// "b.cpp"
template <typename T1, typename T2>
void f1(T2, T1) {}

int main()
{
    f1('a', 'b');
    return EXIT_SUCCESS;
}
```

##### 16.2.2 Partial Ordering of Overloaded Function Templates

The function generated from the more specialized template is preferred:
```c++
template <typename T>
void f(T)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

template <typename T>
void f(T *)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

void foo()
{
    f(0);                                 // void f(T) [with T = int]
    f(NULL);                              // void f(T) [with T = long int]
    f(nullptr);                           // void f(T) [with T = std::nullptr_t]
    f(static_cast<int *>(nullptr));       // void f(T *) [with T = int]
    f(reinterpret_cast<int *>(nullptr));  // ERROR: reinterpret_cast from std::nullptr_t to int * is not allowed
}
```

##### 16.2.3 Formal Ordering Rules

We describe the exact procedure to determine 
whether one function template participating in an overload set 
is more specialized than the other. 


Note that these are _partial_ ordering rules: 
It is possible that given two templates, **neither** can be considered more specialized than the other. 
If overload resolution must select between two such templates, 
**no** decision can be made, and the program contains an ambiguity error.


Assume we are comparing two identically named function templates `f1` and `f2`. 
Overload resolution is decided as follows: 
- Function call parameters that are covered by a default argument 
  and ellipsis parameters that are not used 
  might or might not be ignored in what follows. 
  - Depends on whether they are needed during the template argument deduction process specified as follows. 
- We then synthesize two lists of call argument types of `f1` and `f2`
  (or for conversion function templates, a return type) 
  by substituting every template parameter as follows: 
  1. Replace each template type parameter with a unique invented type. 
  2. Replace each template template parameter with a unique invented class template. 
  3. Replace each non-type template parameter with a unique invented value of the appropriate type. 
     (Types, templates, and values that are invented in this context are distinct from 
     any other types, templates, or values that 
     either the programmer used or the compiler synthesized in other contexts.)
- If `f1`'s call argument type list is an exact match on `f2` in terms of template argument deduction, 
  but **not** vice versa, then `f1` is more specialized than `f2`.
  If no deduction succeeds or both succeed, there is **no** ordering between `f1` and `f2`. 
```c++
// #1
// Call argument type list: A1 *, A1 const *
// Not an exact match on #2
template <typename T>
void t(T *, T const * = nullptr, ...) {}

// #2
// Call argument type list: A2 const *, A2 *
// Not an exact match on #1
template <typename T>
void t(T const *, T *, T * = nullptr) {}

void example(int * p)
{
    t(p, p);  // ERROR: ambiguous
}
```
The synthesized lists of argument types are `(A1 *, A1 const *)` and `(A2 const *, A2 *)`.
Template argument deduction of `(A1 *, A1 const *)` versus `#2` succeeds with the substitution of `T = A1 const`, 
but the resulting match is **not** exact because a qualification adjustment is needed to call 
```c++
t<A1 const>(A1 const *, A1 const *, A1 const * = nullptr)
```
with arguments of types `(A1 *, A1 const *)`. 
Similarly, **no** exact match can be found by deducing template arguments for the `#1` 
from the argument type list `(A2 const *, A2 *)`. 
Therefore, there is **no** ordering relationship between the two templates, and the call is ambiguous.


The formal ordering rules generally result in the intuitive selection of function templates. 
However, under several cases, the rules do **not** select the intuitive choice. 
It is therefore possible that the rules will be revised to accommodate those examples in the future. 

##### 16.2.4 Templates and Non-templates

Function templates can be overloaded with non-template functions. 
All else being equal, the non-template function is preferred during overload resolution. 
```c++
template <typename T>
void f(T) {}

void f(int &) {}

void foo()
{
    int x = 7;
    f(x);       // void f(int &)
}
```
Overload resolution prefers better match over specialization. 
When `const` and reference qualifiers differ, priorities for overload resolution can change.
```c++
template <typename T>
void f(T) {}

void f(const int &) {}

void foo()
{
    int x = 7;
    const int y = x;
    
    f(x);             // void f(T) [with T = int]
    f(y);             // void f(const int &)
}
```
For this reason, it’s a good idea to declare the member function template as
```c++
template <typename T>
void f(const T &);
```
Nevertheless, this effect can easily occur accidentally and cause surprising behavior 
when member functions are defined that accept the same arguments as copy or move constructors. 
```c++
class C
{
public:
    C() = default;
    C(C const &) { std::cout << __PRETTY_FUNCTION__ << '\n'; }
    C(C &&) { std::cout << __PRETTY_FUNCTION__ << '\n'; }

    template <typename T>
    C(T &&) { std::cout << __PRETTY_FUNCTION__ << '\n'; }
};

void foo()
{
    C x;
    C const c;
    
    C x2 {x};             // C::C(T &&) [with T = C &]
    C x3 {std::move(x)};  // C(C &&)
    C x4 {c};             // C(const C &)
}
```
Thus, the member function template is a better match for copying a `C` than the copy constructor. 
And for `std::move(c)`, which yields type `C const &&` 
(a type that is possible but usually **doesn’t** have meaningful semantics), 
the member function template also is a better match than the move constructor. 


For this reason, usually you have to partially disable such member function templates 
when they might hide copy or move constructors via SFINAE. 
Note providing a non-const copy constructor is **no** solution 
because forwarding reference is still a better match for derived class objects. 
This is explained in Section 6.4.

##### 16.2.5 Variadic Function Templates

```c++
template <typename T>
void f(T *)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

template <typename ... Ts>
void f(Ts ...)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

template <typename ... Ts>
void f(Ts * ...)
{
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

void foo()
{
    f(0, 0.0);                                                       // void f(Ts ...) [with Ts = {int, double}]
    f(static_cast<int *>(nullptr), static_cast<double *>(nullptr));  // void f(Ts * ...) [with Ts = {int, double}]
    f(static_cast<int *>(nullptr));                                  // void f(T *) [with T = int]
}
```
When applying the formal ordering rules described in Section 16.2.3 to a variadic template, 
each template _parameter pack_ is replaced by a _single_ made-up type, class template, or value. 
For example, this means that the synthesized argument types for the second and third function templates 
are `(A1)` and `(A2 *)`, respectively, where `A1` and `A2` are unique, made-up types. 
Deduction of the second template against the third’s list of argument types 
succeeds by substituting the single-element sequence `(A2* )` for the parameter pack `Ts`. 
However, there is no way to make the pattern `Ts *` of the third template’s parameter pack match the non-pointer type `A1`, 
so the third function template (which accepts pointer arguments) 
is considered more specialized than the second function template (which accepts any arguments).


#### 📌 16.3 Explicit Specialization


The ability to overload function templates, 
combined with the partial ordering rules to select the “best” matching function template, 
allows us to add more specialized templates to a generic implementation 
to tune code transparently for greater efficiency. 


However, class templates and variable templates can **not** be overloaded. 
Instead, another mechanism was chosen to enable transparent customization of class templates: 
_explicit specialization_. 
The standard term _explicit specialization_ refers to 
a language feature that we call _full specialization_ instead. 
It provides an implementation for a template with template parameters that are fully substituted: 
**No** template parameters remain. 


Class templates, function templates, and variable templates can be fully specialized.
So can members of class templates that may be defined outside the body of a class definition
(i.e., member functions, nested classes, static data members, and member enumeration types).


Alias templates are the only form of template that can **not** be specialized, 
either by a full specialization or a partial specialization. 
This restriction is necessary to make the use of template aliases 
transparent to the template argument deduction process (Section 15.11). 


In a later section, we will describe _partial specialization_. 
This is similar to full specialization, 
but instead of fully substituting the template parameters, 
some parameterization is left in the alternative implementation of a template. 
Full specializations and partial specializations are both equally “explicit” in our source code, 
which is why we avoid the term _explicit specialization_ in our discussion. 


**Neither** full **nor** partial specialization introduces a totally new template or template instance. 
Instead, these constructs provide alternative definitions for instances 
that are already implicitly declared in the generic (or unspecialized) template. 
This is a relatively important conceptual observation, and it is a key difference with overloaded templates.

##### 16.3.1 Full Class Template Specialization

A full specialization is introduced with a sequence of three tokens: 
```c++
template <>
```
The same prefix is also needed to declare full function template specializations. 
Earlier designs of the C++ language did **not** include this prefix, 
but the addition of member templates required additional syntax to disambiguate complex specialization cases. 
```c++
template <typename T>
class S
{
public:
    void info() {}
};

template <>
class S<void>
{
public:
    void msg() {}
};
```
The implementation of the full specialization 
does **not** need to be related in any way to the generic definition:
This allows us to have member functions of different names (`info` versus `msg`). 
The connection is solely determined by the name of the class template.


The list of specified template arguments must correspond to the list of template parameters. 
For example, it is **not** valid to specify a non-type value for a template type parameter. 
However, template arguments for parameters with default template arguments are optional. 
```c++
template <typename T>
class Types
{
public:
    using I = int;
};

// #1
template <typename T, typename U = typename Types<T>::I>
class S;

// #2, aka S<void, int>
template <>
class S<void>
{
public:
    void f();
};

// #3
template <>
class S<char, char>;

// ERROR: 0 can not substitute type parameter U
template <>
class S<char, 0>;

void foo()
{
    S<int> * pi;       // OK: uses #1, no definition needed
    S<int> e1;         // ERROR: uses #1, but no definition available
    S<void> * pv;      // OK: used #2, no definition needed
    S<void, int> sv;   // OK: uses #2, definition available
    S<void, char> e2;  // ERROR: uses #1, but no definition available
    S<char, char> e3;  // ERROR: uses #3, but no definition available at present
}

// definition for #3, comes too late
template <>
class S<char, char> {};
```
Declarations of full specializations (and of templates) 
do **not** necessarily have to be definitions. 
However, when a full specialization is declared, 
the generic definition is **never** used for the given set of template arguments. 
Hence, if a definition is needed but **none** is provided, the program is in **error**. 


For class template specialization, it is sometimes useful to “forward declare” types 
so that mutually dependent types can be constructed. 
A _full specialization declaration_ is identical to a normal class declaration 
in this way (it is **not** a template declaration). 
The only differences are the syntax and the fact that the declaration must match a previous template declaration. 
Because it is **not** a template declaration, the members of a full class template specialization 
can be defined using the ordinary out-of-class member definition syntax 
(in other words, the `template <>` prefix can **not** be specified):
```c++

template <typename T>
class S;

template <>
class S<char **>
{
public:
    // templated entity, not a template!
    void print() const;
};

// the following definition can not be preceded by template<>
void S<char **>::print() const {}
```
```c++
template <typename T>
class Outside
{
public:
    template <typename U>
    class Inside {};
};


template <>
class Outside<void>
{
    // there is no special connection 
    // between the following nested class
    // and the one defined in the generic template
    template <typename U>
    class Inside
    {
    private:
        static int count;
    };
};

// the following definition can not be preceded by template <> 
template <typename U>
int Outside<void>::Inside<U>::count = 1;
```
A full specialization is a replacement for the instantiation of a certain generic template, 
and it is **not** valid to have both the explicit and the generated versions of a template present in the same program. 
An attempt to use both in the same file is usually caught by a compiler:
```c++
template <typename T>
class Invalid {};

// Use of class template
// causes the instantiation of Invalid<double>,
Invalid<double> x1;

// Manual instantiation (explicit instantiation definition)
// also causes the instantiation of Invalid<double>
template class Invalid<double>;

// ERROR: Invalid<double> already instantiated
template <>
class Invalid<double>;
```
Unfortunately, if the uses occur in different translation units, 
the problem may **not** be caught so easily. 
The following invalid C++ example consists of two files and compiles and links on many implementations, 
but it is invalid and dangerous:
```c++
/// "a.cpp"
template <typename T>
class Danger
{
public:
    enum { max = 10 };
};

// uses generic value
char buffer[Danger<void>::max];

extern void clear(char *);

int main()
{
    clear(buffer);
}

/// "b.cpp"
template <typename T>
class Danger;

template <>
class Danger<void>
{
public: 
    enum { max = 100 };
};

void clear(char * buf)
{
    // mismatch in array bound:
    for (int k = 0; k < Danger<void>::max; ++k)
    {
        buf[k] = '\0';
    }
}
```
Care must be taken to ensure that the declaration of the specialization 
is visible to all the users of the generic template. 
In practical terms, this means that a declaration of the specialization 
should normally follow the declaration of the template in its header file. 
When the generic implementation comes from an external source 
(such that the corresponding header files should **not** be modified), 
this is **not** necessarily practical, 
but it may be worth creating a header including the generic template 
followed by declarations of the specializations to avoid these hard-to-find errors. 
In general, it is better to **avoid** specializing templates coming from an external source 
**unless** it is clearly marked as being designed for that purpose. 

##### 16.3.2 Full Function Template Specialization

The syntax and principles behind (explicit) full function template specialization
are much the same as those for full class template specialization, 
but overloading and argument deduction come into play. 


The full specialization declaration can omit explicit template arguments 
when the template being specialized can be determined via argument deduction 
(using as argument types the parameter types provided in the declaration) and partial ordering. 
```c++
// #1
template <typename T>
void f(T) {}

// #2
template <typename T>
void f(T *) {}

// #3, OK: specialization of #1, template argument emitted
template <>
void f(int) {}

// #4, OK: specialization of #2, template argument emitted
template <>
void f(int *) {}
```
A full function template specialization can **not** include default call arguments. 
However, any default arguments that were specified for the template being specialized 
remain applicable to the explicit specialization:
```c++
template <typename T>
int f(T, T x = 42)
{
    return x;
}

// ERROR
template <> 
int f(int, int = 35)
{
    return 0;
}
```
(That is because a full specialization provides an _alternative definition_, 
but **not** an alternative declaration. 
At the point of a call to a function template, 
the call is entirely resolved based on the function template. 
The only difference happens when the compiler instantiate the function template, 
and find there is one full specialization.)


A full function template specialiation only provides an alternative definition but **not** an alternative declaration.
For this reason, the signature (including the return type) must match exactly:
```c++
template <typename T> 
auto foo();

// ERROR
template <> 
int foo<int>() 
{ 
    return 42; 
}

// OK
template <> 
auto foo<int>() 
{ 
    return 42; 
}
```


A full specialization is in many ways similar to a normal declaration
(or rather, a normal redeclaration). 
In particular, it does **not** declare a template, 
and therefore only _one definition_ of a non-inline full function template specialization should appear in a program. 
However, we must still ensure that a declaration of the full specialization
follows the template to prevent attempts at using the function generated from the template.
The declarations for a template `g` and one full specialization 
would therefore typically be organized in two files as follows:
- The interface file contains the definitions of primary templates and partial specializations, 
  but declares only the full specializations:
```c++
#ifndef TEMPLATE_G_HPP
#define TEMPLATE_G_HPP

// template definition should appear in header file:
template <typename T>
int g(T, T x = 42)
{
    return x;
}

// specialization declaration inhibits instantiations of the template;
// definition should not appear here to avoid multiple definition errors
template <> 
int g(int, int y);

#endif  // TEMPLATE_G_HPP
```
- The corresponding implementation file defines the full specialization:
```c++
#include "template_g.hpp"

template <> 
int g(int, int y)
{
    return y / 2;
}
```
Alternatively, the specialization could be made `inline`,
in which case its definition can be (and should be) placed in the header file.

##### 16.3.3 Full Variable Template Specialization

Variable templates can also be fully specialized. 
```c++
template <typename T> 
constexpr std::size_t SZ = sizeof(T);

template <> 
constexpr std::size_t SZ<void> = 0;
```
The specialization can provide an initializer that is distinct from that resulting from the template. 
A variable template specialization is **not** required to have a type matching that of the template being specialized:
```c++
template <typename T> 
typename T::iterator null_iterator;

// BitIterator doesn’t match T::iterator, and that is fine
template <> 
BitIterator null_iterator<std::bitset<100>>;
```

##### 16.3.4 Full Member Specialization

Not only member templates, but also ordinary static data members and member functions of class templates, can be fully specialized. 
The syntax requires `template <>` prefix for every enclosing class template. 
If a member template is being specialized, a `template <>` must also be added to denote that it is being specialized. 
```c++
template <typename T>
class Outer
{
public:
    template <typename U>
    class Inner
    {
    private:
        static int count;
    };

    static int code;

    void print() const {}
};

template <typename T>
int Outer<T>::code = 6;

template <typename T>
template <typename U>
int Outer<T>::Inner<U>::count = 7;

template <>
class Outer<bool>
{
public:
    template <typename U>
    class Inner
    {
    private:
        static int count;
    };

    void print() const {}
};

template <>
int Outer<void>::code = 12;

template <>
void Outer<void>::print() const {}
```
These definitions are used over the generic ones for class `Outer<void>`, 
but other members of class `Outer<void>` are still generated from the primary template. 
Note that after these declarations, it is no longer valid to provide an explicit specialization for `Outer<void>`.


Just as with full function template specializations, 
we need a way to declare the specialization of an ordinary member of a class template
without specifying a definition (to prevent multiple definitions). 
Although _non-defining out-of-class declarations_ are **not** allowed in C++ 
for member functions and static data members of ordinary classes, 
they are fine _when specializing members of class templates_.  
The previous definitions could be declared with
```c++
template <>
int Outer<void>::code;

template <>
void Outer<void>::print() const;
```
The attentive reader might point out that the non-defining declaration of the full specialization of `Outer<void>::code` 
looks exactly like a definition to be initialized with a default constructor. 
This is indeed so, but such declarations are always interpreted as non-defining declarations. 
For a full specialization of a static data member with a type that can only be initialized using a default constructor, 
we must resort to initializer list syntax:
```c++
class DefaultInitOnly
{
public:
    DefaultInitOnly() = default;
    DefaultInitOnly(DefaultInitOnly const &) = delete;
};

template <typename T>
class Statics
{
private:
    static T sm;
};

// declaration
template <>
DefaultInitOnly Statics<DefaultInitOnly>::sm;

// a definition calling the default constructor
template <>
DefaultInitOnly Statics<DefaultInitOnly>::sm {};
```
Prior to C++11, this was **not** possible. 
Default initialization was thus **not** available for such specializations. 
Typically, an initializer copying a default value was used:
```c++
template <>
DefaultInitOnly Statics<DefaultInitOnly>::sm = DefaultInitOnly();
```
Unfortunately, for our example that was **not** possible 
because the copy constructor is deleted. 
However, C++17 introduced _mandatory copy-elision_ rules, 
which make that alternative valid, 
because **no** copy constructor invocation is involved anymore. 


The member template `Outer<T>::Inner` can also be specialized for a given template argument
**without** affecting the other members of the specific instantiation of `Outer<T>`,
for which we are specializing the member template. 
Again, because there is one enclosing template, we will need one `template <>` prefix.
```c++
template <>
template <typename X>
class Outer<wchar_t>::Inner
{
public:
    static long count;  // member type changed
};

template <>
template <typename X>
long Outer<wchar_t>::Inner<X>::count;
```
The template `Outer<T>::Inner` can also be fully specialized, 
but only for a given instance of `Outer<T>`. 
We now need two `template <>` prefixes: 
one because of the enclosing class 
and one because we’re fully specializing the (inner) template:
```c++
template <>
template <>
class Outer<char>::Inner<wchar_t>
{
public:
    enum { count = 1 };
};

// the following is not valid C++:
// template <> cannot follow a template parameter list
template <typename X>
template <>
class Outer<X>::Inner<void>;  // ERROR
```
Contrast this with the specialization of the member template of `Outer<bool>`. 
Because the latter is already fully specialized, there is **no** enclosing template, 
and we need only one `template <>` prefix:
```c++
template <>
class Outer<bool>::Inner<wchar_t>
{
public:
    enum { count = 2 };
};
```


#### 📌 16.4 Partial Class Template Specialization


```c++
// primary template
template <typename T>
class List
{
public:
    void append(T const &);
    inline std::size_t length() const;
};

// partial specialization
template <typename T>
class List<T *>
{
public:
    inline void append(T * p)
    {
        impl.append(p);
    }

    inline std::size_t length() const
    {
        return impl.length();
    }

private:
    List<void *> impl;
};

template <>
class List<void *>
{
    void append(void * p);
    inline std::size_t length() const;
};
```
All member functions of `List`s of pointers are forwarded (through easily inlineable functions)
to the implementation of `List<void *>`. 
This is an effective way to combat _code bloat_ (of which C++ templates are often accused).


There exist several limitations on the parameter and argument lists of partial specialization declarations. 
Some of them are as follows: 
1. The arguments of the partial specialization must match in kind (type, non-type, or template)
   the corresponding parameters of the primary template. 
2. The parameter list of the partial specialization can **not** have default arguments. 
   The default arguments of the primary class template are used instead. 
3. The non-type arguments of the partial specialization should be 
   either non-dependent values or plain non-type template parameters. 
   They can **not** be more complex dependent expressions like `2 * N` 
   (where `N` is a template parameter). 
4. The list of template arguments of the partial specialization 
   should **not** be identical (ignoring renaming) to the list of parameters of the primary template. 
5. If one of the template arguments is a pack expansion, 
6. it must come at the end of a template argument list.
```c++
// primary template
template <typename T, int I = 3>
class S;

// ERROR: parameter kind mismatch
template <typename T>
class S<int, T>;

// ERROR: no default arguments allowed for specializations
template <typename T = int>
class S<T, 10>;

// ERROR: no non-type expressions allowed for specializations
template <int I>
class S<int, I * 2>;

// ERROR: no significant difference from primary template
template <typename U, int K>
class S<U, K>;

template <typename ... Ts>
class Tuple;

// ERROR: pack expansion not at the end
template <typename Tail, typename... Ts>
class Tuple<Ts ..., Tail>;

// OK: pack expansion is at the end of a nested template argument list
template <typename Tail, typename... Ts>
class Tuple<Tuple<Ts ...>, Tail>; 
```
Every partial specialization, like every full specialization, 
is associated with the primary template. 
When a template is used, the primary template is always the one that is looked up, 
but then the arguments are also matched against those of the associated specializations 
(using template argument deduction, as described in Chapter 15) 
to determine which template implementation is picked. 


Just as with function template argument deduction, the SFINAE principle applies here: 
If, while attempting to match a partial specialization an invalid construct is formed, 
that specialization is silently abandoned and another candidate is examined if one is available. 
If **no** matching specializations is found, the primary template is selected. 
If multiple matching specializations are found, 
the most specialized one (in the sense defined for overloaded function templates) is selected; 
if none can be called most specialized, the program contains an ambiguity error. 


Finally, we should point out that it is entirely possible for a class template partial specialization 
to have more or fewer parameters than the primary template. 
```c++
// partial specialization for any pointer-to-void* member
template <typename C>
class List<void * C::*>
{
public:
    using ElementType = void * C::*;

    void append(ElementType pm);

    inline std::size_t length() const;
};

// Partial specialization for any pointer-to-member-pointer type 
// except pointer-to-void * member, which is handled earlier. 
// (Note that this partial specialization has two template parameters,
// whereas the primary template only has one parameter.)
// This specialization makes use of the prior one to achieve the desired optimization. 
template <typename T, typename C>
class List<T * C::*>
{
public:
    using ElementType = T * C::*;

    inline void append(ElementType pm)
    {
        impl.append((void * C::*) pm);
    }

    inline std::size_t length() const
    {
        return impl.length();
    }

private:
    List<void * C::*> impl;
};
```


#### 📌 16.5 Partial Variable Template Specialization


The syntax is similar to full variable template specialization,
except that `template <>` is replaced by an actual template declaration header, 
and the template argument list following the variable template name must depend on template parameters: 
```c++
template <typename T> 
constexpr std::size_t SZ = sizeof(T);

template <typename T> 
constexpr std::size_t SZ<T &> = sizeof(void *);
```
As with the full specialization of variable templates, 
the type of a partial specialization is **not** required to match that of the primary template:
```c++
template <typename T> 
typename T::iterator null_iterator;

// T * doesn’t match T::iterator, and that is fine
template <typename T, std::size_t N> 
T * null_iterator<T[N]> = nullptr;
```
The rules regarding the kinds of template arguments 
that can be specified for a variable template partial specialization 
are identical to those for class template specializations. 
Similarly, the rules to select a specialization for a given list of concrete template arguments are identical too.






### 🎯 Chapter 17 Future Directions

N/A. 
