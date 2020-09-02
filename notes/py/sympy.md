# `SymPy` Tutorial Notes

- Notes of reading [`SymPy` Tutorial](https://docs.sympy.org/latest/tutorial/index.html)

## 🔱 [Gotchas](https://docs.sympy.org/latest/tutorial/gotchas.html)

### 🌱 Symbols

SymPy can be used in any environment where Python is available. We just import it, like we would any other library: 

```
>>> from sympy import *
```

In Python, variables have no meaning until they are defined. SymPy is no different. Unlike many symbolic manipulation systems you may have used, in SymPy, variables are not defined automatically. To define variables, we must use `symbols`. 

```
>>> x = symbols('x')
>>> x + 1
x + 1
```

`symbols` takes a string of variable names separated by *spaces* or *commas*, and creates Symbols out of them. We can then assign these to variable names. Later, we will investigate some convenient ways we can work around this issue. For now, let us just define the most common variable names, `x`, `y`, and `z`, for use through the rest of this section

```
>>> x, y, z = symbols('x y z')
```

As a final note, we note that the name of a Symbol and the name of the variable it is assigned to need **NOT** have anything to do with one another. 

```
>>> a, b = symbols('b a')
>>> a
b

>>> b
a
```

Usually, the best practice is to assign Symbols to Python variables of the *same name*. 

To avoid confusion, throughout this tutorial, Symbol names and Python variable names will always coincide. Furthermore, the word “Symbol” will refer to a SymPy Symbol and the word “variable” will refer to a Python variable. 

Finally, let us be sure we understand the difference between SymPy Symbols and Python variables. Consider the following: 

```
>>> x = symbols('x')
>>> expr = x + 1
>>> x = 2
>>> expr
x + 1
```

Changing `x` to `2` had no effect on `expr`. This is because `x = 2` changes the Python variable `x` to `2`, but has no effect on the SymPy Symbol `x`, which was what we used in creating `expr`. When we created expr, the Python variable x was a Symbol. After we created, it, we changed the Python variable x to 2. But expr remains the same. This behavior is **NOT** unique to SymPy. All Python programs work this way: if a variable is changed, expressions that were already created with that variable do **NOT** change automatically. 

**Quick Tip**: To change the value of a Symbol in an expression, use `subs`: 

```
>>> x = symbols('x')
>>> expr = x + 1
>>> expr.subs(x, 2)
3
```

### 🌱 Equals signs

As in Python, `=` in SymPy represents assignment. Unlike Python, `==` in SymPy, represents *exact structural equality* testing. This means that `a == b` means that we are asking if `a = b`. We always get a `bool` as the result of `==`. There is a separate object, called `Eq`, which can be used to create symbolic equalities: 

```
>>> x + 1 == 4
False

>>> Eq(x + 1, 4)
Eq(x + 1, 4)
```

There is one additional caveat about `==` as well. Suppose we want to know if `(x + 1)**2 = x**2 + 2*x + 1`. We might try something like this: 

```
>>> (x + 1)**2 = x**2 + 2*x + 1
False
```

`==` represents *exact structural equality* testing. *“Exact”* here means that two expressions will compare equal with `==` only if they are exactly equal structurally. Here, `(x + 1)**2` and `x**2 + 2*x + 1` are **NOT** the same symbolically. One is the power of an addition of two terms, and the other is the addition of three terms. 

It turns out that when using SymPy as a library, having `==` test for *exact structural equality* is far more useful than having it represent *symbolic equality*, or having it test for *mathematical equality*. However, as a new user, you will probably care more about the latter two. We have already seen an alternative to representing equalities symbolically, `Eq`. To test if two things are equal, it is best to recall the basic fact that if `a = b`, then `a − b = 0`. Thus, the best way to check if `a = b` is to take `a − b` and simplify it, and see if it goes to `0`. We will learn later that the function to do this is called [`simplify`](https://docs.sympy.org/latest/tutorial/simplification.html#tutorial-simplify). This method is **NOT** infallible — in fact, it can be [theoretically proven](https://en.wikipedia.org/wiki/Richardson's_theorem) that it is impossible to determine if two symbolic expressions are identically equal in general — but for most common expressions, it works quite well. 

```
>>> a = (x + 1)**2
>>> b = x**2 + 2*x + 1
>>> simplify(a - b)
0

>>> c = x**2 - 2*x + 1
>>> simplify(a - c)
4*x
```

There is also a method called `equals` that tests if two expressions are equal by evaluating them *numerically* at random points. 

```
>>> a = cos(x)**2 - sin(x)**2
>>> b = cos(2*x)
>>> a.equals(b)
True
```

### 🌱 Two Final Notes: `^` and `/`

You may have noticed that we have been using `**` for exponentiation instead of the standard `^`. That’s because SymPy follows Python’s conventions. In Python, `^` represents *logical exclusive or*. SymPy follows this convention: 

```
>>> True ^ False
True

>>> True ^ True
False

>>> Xor(x, y)
x ^ y
```

Finally, a small technical discussion on how SymPy works is in order. When you type something like `x + 1`, the SymPy Symbol `x` is added to the Python int `1`. Python’s operator rules then allow SymPy to tell Python that SymPy objects know how to be added to Python ints, and so `1` is automatically converted to the SymPy Integer object. 

Whenever you combine a SymPy object and a SymPy object, or a SymPy object and a Python object, you get a SymPy object, but whenever you combine two Python objects, SymPy never comes into play, and so you get a Python object.

```
>>> type(Integer(1) + 1)
<class 'sympy.core.numbers.Integer'>

>>> type(1 + 1)
<... 'int'>
```

Python ints work much the same as SymPy Integers, but there is one important exception: division. In SymPy, the division of two Integers gives a Rational: 

```
>>> Integer(1)/Integer(3)
1/3

>>> type(Integer(1)/Integer(3))
<class 'sympy.core.numbers.Rational'>
```

To avoid this, we can construct the rational object explicitly: 

```
>>> Rational(1, 2)
1/2
```

This problem also comes up whenever we have a larger symbolic expression with int/int in it. For example: 

```
>>> x + 1/2 
x + 0.5
```

This happens because Python first evaluates `1/2` into `0.5`, and then that is cast into a SymPy type when it is added to `x`. Again, we can get around this by explicitly creating a Rational: 

```
>>> x + Rational(1, 2)
x + 1/2
```

There are several tips on avoiding this situation in the [Gotchas and Pitfalls](https://docs.sympy.org/latest/gotchas.html#gotchas) document. 

## 🔱 [Basic Operations](https://docs.sympy.org/latest/tutorial/basic_operations.html)











## 🔱

### 🌱

#### 📌

