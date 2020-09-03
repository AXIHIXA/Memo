# `SymPy` Tutorial Notes

- Notes of reading [`SymPy` Tutorial](https://docs.sympy.org/latest/tutorial/index.html)

## 🔱 [Gotchas](https://docs.sympy.org/latest/tutorial/gotchas.html)

This will make all further examples pretty print with unicode characters: 

```
init_printing(use_unicode=True)
```

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

`=` in SymPy represents assignment. 

`==` in SymPy represents *exact structural equality* testing. This means that `a == b` means that we are asking if `a = b`. We always get a `bool` as the result of `==`. There is a separate object, called `Eq`, which can be used to create *symbolic equalities*: 

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

```
from sympy import *
from sympy.abc import x, y, z
```

### 🌱 `subs`: Substitution

Substitution replaces all instances of something in an expression with something else. It is done using the `subs` method. For example: 

```
>>> expr = cos(x) + 1
>>> expr.subs(x, y)
cos(y) + 1
```

Substitution is usually done for one of two reasons: 

1. Evaluating an expression at a point. For example, if our expression is `cos(x) + 1` and we want to evaluate it at the point `x = 0`, so that we get `cos(0) + 1`, which is `2`. 
```
>>> expr.subs(x, 0)
2
```
2. Replacing a subexpression with another subexpression. There are two reasons we might want to do this. 
    - If we are trying to build an expression that has some symmetry, such as `x**(x**(x**x))`. To build this, we might start with `x**y`, and replace `y` with `x**y`. We would then get `x**(x**y)`. If we replaced `y` in this new expression with `x**x`, we would get `x**(x**(x**x))`, the desired expression. 
    ```
    >>> expr = x**y
    >>> expr
    x**y

    >>> expr = expr.subs(y, x**y)
    >>> expr
    x**(x**y)

    >>> expr = expr.subs(y, x**x)
    >>> expr
    x**(x**(x**x))
    ```
    - If we want to perform a very controlled simplification, or perhaps a simplification that SymPy is otherwise unable to do. For example, say we have `sin(2*x) + cos(2*x)`, and we want to replace `sin(2*x)` with `2*sin(x)*cos(x)`. As we will learn later, the function `expand_trig` does this. However, this function will also expand `cos(2*x)`, which we may **NOT** want. While there are ways to perform such precise simplification, and we will learn some of them in the [Advanced Expression Manipulation](https://docs.sympy.org/latest/tutorial/manipulation.html#tutorial-manipulation) section, an easy way is to just replace `sin(2*x)` with `2*sin(x)*cos(x)`. 
    ```
    >>> expr = sin(2*x) + cos(2*x)

    >>> expand_trig(expr)
    2*sin(x)*cos(x) + 2*cos(x)**2 - 1

    >>> expr.subs(sin(2*x), 2*sin(x)*cos(x))
    2*sin(x)*cos(x) + cos(2*x)
    ```

`subs` returns a *new expression*. SymPy objects are *immutable*. That means that subs does **NOT** modify it in-place. For example: 

```
>>> expr = cos(x)

>>> expr.subs(x, 0)
1

>>> expr
cos(x)

>>> x
x
```

We see that performing `expr.subs(x, 0)` leaves `expr` *unchanged*. In fact, since SymPy expressions are immutable, **NO** function will change them in-place. All functions will return new expressions. 

To perform *multiple* substitutions at once, pass a list of `(old, new)` pairs to `subs`. 

```
>>> expr = x**3 + 4*x*y - z
>>> expr.subs([(x, 2), (y, 4), (z, 0)])
40
```

It is often useful to combine this with a *list comprehension* to do a large set of similar replacements all at once. For example, say we had `x**4 - 4*x**3 + 4*x**2 - 2*x + 3` and we wanted to replace all instances of `x` that have an even power with `y`, to get `y**4 -4*x**3 + 4*y**2 - 2*x + 3`.

```
>>> expr = x**4 - 4*x**3 + 4*x**2 - 2*x + 3
>>> replacements = [(x**i, y**i) for i in range(5) if i % 2 == 0]
>>> expr.subs(replacements)
-4*x**3 - 2*x + y**4 + 4*y**2 + 3
```

### 🌱 `sympify`: Converting Strings to SymPy Expressions

The `sympify` function (that’s `sympify`, not to be confused with `simplify`) can be used to convert strings into SymPy expressions. 

```
>>> str_expr = "x**2 + 3*x - 1/2"
>>> expr = sympify(str_expr)
>>> expr
x**2 + 3*x - 1/2

>>> expr.subs(x, 2)
19/2
```

**WARNING**: `sympify` uses `eval`. Don’t use it on unsanitized input. 

### 🌱 `evalf`: Numerical evaluation

To evaluate a numerical expression into a floating point number, use `evalf`. 

```
>>> expr = sqrt(8)
>>> expr.evalf()
2.82842712474619
```

SymPy can evaluate floating point expressions to arbitrary precision. By default, 15 digits of precision are used, but you can pass any number as the argument to `evalf`. Let’s compute the first 100 digits of `pi`. 

```
>>> pi.evalf(100)
3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068
```

To numerically evaluate an expression with a Symbol at a point, we might use `subs` followed by `evalf`, but it is more efficient and numerically stable to pass the substitution to `evalf` using the `subs` flag, which takes a dictionary of `Symbol: point` pairs. 

```
>>> expr = cos(2*x)

>>> expr.evalf(subs={x: 2.4})
0.0874989834394464
```

Sometimes there are roundoff errors smaller than the desired precision that remain after an expression is evaluated. Such numbers can be removed at the user’s discretion by setting the `chop` flag to `True`. 

```
>>> one = cos(1)**2 + sin(1)**2
>>> (one - 1).evalf()
-0.e-124

>>> (one - 1).evalf(chop=True)
0

```

### 🌱 `lambdify`: Batch numerical evaluation using NumPy, etc. 

If you want to evaluate an expression at a thousand points, using SymPy would be far slower than it needs to be, especially if you only care about machine precision. Instead, you should use libraries like NumPy and SciPy. 

The easiest way to convert a SymPy expression to an expression that can be numerically evaluated is to use the `lambdify` function. `lambdify` acts like a `lambda` function, except it converts the SymPy names to the names of the given numerical library, usually NumPy. For example: 

```
>>> import numpy 
>>> a = numpy.arange(10) 
>>> expr = sin(x)
>>> f = lambdify(x, expr, "numpy") 
>>> f(a) 
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
       -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])
```

**Warning**: `lambdify` uses `eval`. Don’t use it on unsanitized input. 

You can use other libraries than NumPy. For example, to use the standard library math module, use "math".

```
>>> f = lambdify(x, expr, "math")
>>> f(0.1)
0.0998334166468
```

To use `lambdify` with numerical libraries that it does not know about, pass a dictionary of `sympy_name: numerical_function pairs`. For example: 

```
>>> def mysin(x):
...    """
...    My sine. Note that this is only accurate for small x.
...    """
...    return x

>>> f = lambdify(x, expr, {"sin": mysin})
>>> f(0.1)
0.1
```

## 🔱 [Printing](https://docs.sympy.org/latest/tutorial/printing.html)

As we have already seen, SymPy can pretty print its output using Unicode characters. This is a short introduction to the most common printing options available in SymPy. 

### 🌱 Printers

There are several printers available in SymPy. The most common ones are: 
    - str
    - srepr
    - ASCII pretty printer
    - Unicode pretty printer
    - LaTeX
    - MathML
    - Dot

In addition to these, there are also “printers” that can output SymPy objects to code, such as C, Fortran, Javascript, Theano, and Python. These are not discussed in this tutorial. 

### 🌱 Setting up Pretty Printing

If all you want is the best pretty printing, use the `init_printing()` function. This will automatically enable the best printer available in your environment. 

```
>>> from sympy import init_printing
>>> init_printing() 
```

If you plan to work in an interactive calculator-type session, the `init_session()` function will automatically import everything in SymPy, create some common Symbols, setup plotting, and run `init_printing()`. 

```
>>> from sympy import init_session
>>> init_session() 
```

```
Python console for SymPy 0.7.3 (Python 2.7.5-64-bit) (ground types: gmpy)

These commands were executed:
>>> from __future__ import division
>>> from sympy import *
>>> x, y, z, t = symbols('x y z t')
>>> k, m, n = symbols('k m n', integer=True)
>>> f, g, h = symbols('f g h', cls=Function)
>>> init_printing() # doctest: +SKIP

Documentation can be found at http://www.sympy.org
```

In any case, this is what will happen:
    - In the IPython QTConsole, if LATEX is installed, it will enable a printer that uses LATEX. If LATEX is not installed, but Matplotlib is installed, it will use the Matplotlib rendering engine. If Matplotlib is not installed, it uses the Unicode pretty printer. 
    - In the IPython notebook, it will use MathJax to render LATEX. 
    - In an IPython console session, or a regular Python session, it will use the Unicode pretty printer if the terminal supports Unicode. 
    - In a terminal that does not support Unicode, the ASCII pretty printer is used. 
    - To explicitly **NOT** use LATEX, pass `use_latex=False` to `init_printing()` or `init_session()`. To explicitly **NOT** use Unicode, pass `use_unicode=False`. 

### 🌱 Printing Functions

#### 📌 `str`

To get a string form of an expression, use `str(expr)`. This is also the form that is produced by `print(expr)`. String forms are designed to be easy to read, but in a form that is correct Python syntax so that it can be copied and pasted. The `str()` form of an expression will usually look exactly the same as the expression as you would enter it. 

```
>>> from sympy import *
>>> x, y, z = symbols('x y z')

>>> str(Integral(sqrt(1/x), x))
'Integral(sqrt(1/x), x)'

>>> print(Integral(sqrt(1/x), x))
Integral(sqrt(1/x), x)

```

#### 📌 `srepr`












## 🔱

### 🌱

#### 📌

