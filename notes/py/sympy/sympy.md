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

- There are several printers available in SymPy. The most common ones are: 
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

- In any case, this is what will happen:
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

The srepr form of an expression is designed to show the exact form of an expression. It will be discussed more in the [Advanced Expression Manipulation](https://docs.sympy.org/latest/tutorial/manipulation.html#tutorial-manipulation) section. To get it, use `srepr()`. 

```
>>> srepr(Integral(sqrt(1/x), x))
"Integral(Pow(Pow(Symbol('x'), Integer(-1)), Rational(1, 2)), Tuple(Symbol('x')))"
```

The srepr form is mostly useful for understanding how an expression is built *internally*. 

#### 📌 ASCII Pretty Printer

The ASCII pretty printer is accessed from `pprint()`. If the terminal does not support Unicode, the ASCII printer is used by default. Otherwise, you must pass `use_unicode=False`. 

```
>>> pprint(Integral(sqrt(1/x), x), use_unicode=False)
  /
 |
 |     ___
 |    / 1
 |   /  -  dx
 | \/   x
 |
/

```

`pprint()` prints the output to the screen. If you want the string form, use `pretty()`. 

```
>>> pretty(Integral(sqrt(1/x), x), use_unicode=False)
'  /          \n |           \n |     ___   \n |    / 1    \n |   /  -  dx\n | \\/   x    \n |           \n/            '

>>> print(pretty(Integral(sqrt(1/x), x), use_unicode=False))
  /
 |
 |     ___
 |    / 1
 |   /  -  dx
 | \/   x
 |
/

```

#### 📌 ASCII Pretty Printer

The Unicode pretty printer is also accessed from `pprint()` and `pretty()`. If the terminal supports Unicode, it is used automatically. If `pprint()` is not able to detect that the terminal supports unicode, you can pass `use_unicode=True` to force it to use Unicode. 

```
>>> pprint(Integral(sqrt(1/x), x), use_unicode=True)
```

#### 📌 LATEX

To get the LATEX form of an expression, use `latex()`. 

```
>>> print(latex(Integral(sqrt(1/x), x)))
\int \sqrt{\frac{1}{x}}\, dx
```

The `latex()` function has many options to change the formatting of different things. See [its documentation](https://docs.sympy.org/latest/modules/printing.html#sympy.printing.latex.latex) for more details. 

#### 📌 MathML

There is also a printer to MathML, called `print_mathml()`. It must be imported from `sympy.printing.mathml`.

```
>>> from sympy.printing.mathml import print_mathml
>>> print_mathml(Integral(sqrt(1/x), x))
<apply>
    <int/>
    <bvar>
        <ci>x</ci>
    </bvar>
    <apply>
        <root/>
        <apply>
            <power/>
            <ci>x</ci>
            <cn>-1</cn>
        </apply>
    </apply>
</apply>
```

`print_mathml()` prints the output. If you want the string, use the function `mathml()`. 

#### 📌 Dot

The `dotprint()` function in `sympy.printing.dot` prints output to dot format, which can be rendered with [Graphviz](http://www.graphviz.org/). See the [Advanced Expression Manipulation](https://docs.sympy.org/latest/tutorial/manipulation.html#tutorial-manipulation) section for some examples of the output of this printer. 

Here is an example of the raw output of the `dotprint()` function: 

```
>>> from sympy.printing.dot import dotprint
>>> from sympy.abc import x
>>> print(dotprint(x+2))
digraph{

# Graph style
"ordering"="out"
"rankdir"="TD"

#########
# Nodes #
#########

"Add(Integer(2), Symbol('x'))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
"Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
"Symbol('x')_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];

#########
# Edges #
#########

"Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
"Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
}
```

## 🔱 [Simplification](https://docs.sympy.org/latest/tutorial/simplification.html)

To make this document easier to read, we are going to enable pretty printing. 

```
>>> from sympy import *
>>> from sympy.abc import x, y, z
```

### 🌱 `simplify`: General heuristical simplification

SymPy has dozens of functions to perform various kinds of simplification. There is also one general function called `simplify()` that attempts to apply all of these functions in an intelligent way to arrive at the simplest form of an expression. Here are some examples: 

```
>>> simplify(sin(x)**2 + cos(x)**2)
1

>>> simplify((x**3 + x**2 - x - 1)/(x**2 + 2*x + 1))
x - 1

>>> simplify(gamma(x)/gamma(x - 2))
(x - 2) * (x - 1)
```

But `simplify()` has a pitfall. It just applies all the major simplification operations in SymPy, and uses heuristics to determine the simplest result. But “simplest” is *not a well-defined term*. For example, say we wanted to “simplify” `x**2 + 2*x + 1` into `(x + 1)**2`: 

```
>>> simplify(x**2 + 2*x + 1)
>>> x**2 + 2*x + 1
```

We did not get what we want. There is a function to perform this simplification, called `factor()`, which will be discussed below. 

Another pitfall to `simplify()` is that it can be *unnecessarily slow*, since it tries many kinds of simplifications before picking the best one. If you already know exactly what kind of simplification you are after, it is better to apply the specific simplification function(s) that apply those simplifications. 

Applying specific simplification functions instead of `simplify()` also has the advantage that specific functions have certain guarantees about the form of their output. These will be discussed with each function below. For example, `factor()`, when called on a polynomial with rational coefficients, is guaranteed to factor the polynomial into irreducible factors. `simplify()` has no guarantees. It is entirely heuristical, and, as we saw above, it may even miss a possible type of simplification that SymPy is capable of doing. 

`simplify()` is best when used interactively, when you just want to whittle down an expression to a simpler form. You may then choose to apply specific functions once you see what `simplify()` returns, to get a more precise result. It is also useful when you have no idea what form an expression will take, and you need a catchall function to simplify it. 

### 🌱 Polynomial/Rational Function Simplification

#### 📌 `expand`

`expand()` is one of the most common simplification functions in SymPy. Although it has a lot of scopes, for now, we will consider its function in expanding polynomial expressions. For example: 

```
>>> expand((x + 1)**2)
x**2 + 2*x + 1

>>> expand((x + 2)*(x - 3))
x**2 - x - 6
```

Given a polynomial, `expand()` will put it into a canonical form of a sum of monomials. 

`expand()` may not sound like a simplification function. After all, by its very name, it makes expressions bigger, not smaller. Usually this is the case, but often an expression will become smaller upon calling `expand()` on it due to cancellation. 

```
>>> expand((x + 1)*(x - 2) - (x - 1)*x)
-2
```

#### 📌 `factor`

`factor()` takes a polynomial and factors it into irreducible factors over the rational numbers. For example:

```
>>> factor(x**3 - x**2 + x - 1)
(x - 1)*(x**2 + 1)

>>> factor(x**2*z + 4*x*y*z + 4*y**2*z)
z * (x + 2*y)**2
```

For polynomials, `factor()` is the opposite of `expand()`. `factor()` uses a complete multivariate factorization algorithm over the rational numbers, which means that each of the factors returned by `factor()` is guaranteed to be irreducible. 

If you are interested in the factors themselves, `factor_list` returns a more structured output. 

```
factor_list(x**2*z + 4*x*y*z + 4*y**2*z)
(1, [(z, 1), (x + 2*y, 2)])
```

Note that the input to `factor` and `expand` need **NOT** be polynomials in the strict sense. They will intelligently factor or expand any kind of expression (though note that the factors may not be irreducible if the input is no longer a polynomial over the rationals).

```
>>> expand((cos(x) + sin(x))**2)
sin(x)**2 + 2*sin(x)*cos(x) + cos(x)**2

>>> factor(cos(x)**2 + 2*cos(x)*sin(x) + sin(x)**2)
(sin(x) + cos(x))**2
```

#### 📌 `collect`

`collect()` collects common powers of a term in an expression. For example

```
>>> expr = x*y + x - 3 + 2*x**2 - z*(x**2) + x**3
>>> expr
x**3 - (x**2)*z + 2*(x**2) + x*y + x - 3

>>> collected_expr = collect(expr, x)
>>> collected_expr
x**3 + (x**2)*(2 - z) + x*(y + 1) - 3
```

`collect()` is particularly useful in conjunction with the `.coeff()`method. `expr.coeff(x, n)` gives the coefficient of `x**n` in `expr`:

```
>>> collected_expr.coeff(x, 2)
2 - z
```

#### 📌 `cancel`

`cancel()` will take any rational function and put it into the standard *canonical form*, `p/q`, where `p` and `q` are expanded polynomials with no common factors, and the leading coefficients of `p` and `q` do not have denominators (i.e., are integers). 

```
>>> cancel((x**2 + 2*x + 1)/(x**2 + x))
(x + 1)/x
```

```
>>> expr = 1/x + (3*x/2 - 2)/(x - 4)
>>> expr
1/x + (3*x/2 - 2)/(x - 4)

>>> cancel(expr)
(3*x**2 - 2*x - 8)/(2*x**2 - 8*x)
```

```
>>> expr = (x*y**2 - 2*x*y*z + x*z**2 + y**2 - 2*y*z + z**2)/(x**2 - 1)
>>> expr
(x*y**2 - 2*x*y*z + x*z**2 + y**2 - 2*y*z + z**2)/(x**2 - 1)

>>> cancel(expr)
(y**2 - 2*y*z + z**2)/(x - 1)
```

Note that since `factor()` will completely factorize both the numerator and the denominator of an expression, it can also be used to do the same thing: 

```
>>> factor(expr)
(y - z)**2/(x - 1)
```

However, if you are only interested in making sure that the expression is in canceled form, `cancel()` is *more efficient* than `factor()`.

#### 📌 `apart`

`apart()` performs a [partial fraction decomposition](https://en.wikipedia.org/wiki/Partial_fraction_decomposition) on a rational function.

```
>>> expr = (4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x)
>>> expr
(4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x)

>>> apart(expr)
(2*x - 1)/(x**2 + x + 1) - 1/(x + 4) + 3/x
```

### 🌱 Trigonometric Simplification

**Note**: SymPy follows Python’s naming conventions for inverse trigonometric functions, which is to append an `a` to the front of the function’s name. For example, the inverse cosine, or arc cosine, is called `acos()`. 

```
>>> acos(x)
acos(x)

>>> cos(acos(x))
x

>>> asin(1)
pi/2
```

#### 📌 `trigsimp`

To simplify expressions using trigonometric identities, use `trigsimp()`. 

```
>>> trigsimp(sin(x)**2 + cos(x)**2)
1

>>> trigsimp(sin(x)**4 - 2*cos(x)**2*sin(x)**2 + cos(x)**4)
cos(4*x)/2 + 1/2

>>> trigsimp(sin(x)*tan(x)/sec(x))
   2
sin(x)**2
```

`trigsimp()` also works with hyperbolic trig functions. 

```
>>> trigsimp(cosh(x)**2 + sinh(x)**2)
cosh(2*x)

>>> trigsimp(sinh(x)/tanh(x))
cosh(x)
```

Much like `simplify()`, `trigsimp()` applies various trigonometric identities to the input expression, and then uses a heuristic to return the “best” one.

#### 📌 `expand_trig`

To expand trigonometric functions, that is, apply the sum or double angle identities, use `expand_trig()`. 

```
>>> expand_trig(sin(x + y))
sin(x)*cos(y) + sin(y)*cos(x)

>>> expand_trig(tan(2*x))
2*tan(x)/(1 - tan(x)**2)
```

Because `expand_trig()` tends to make trigonometric expressions larger, and `trigsimp()` tends to make them smaller, these identities can be applied in reverse using `trigsimp()`

```
>>> trigsimp(sin(x)*cos(y) + sin(y)*cos(x))
sin(x + y)
```

### 🌱 Powers

Before we introduce the power simplification functions, a mathematical discussion on the identities held by powers is in order. There are three kinds of identities satisfied by exponents: 

1. `(x**a) * (x**b) == x**(a + b)`
    - Is always true. 
2. `(x**a) * (y**a) == (x*y)**a`
    - Sufficient to hold when `x, y >= 0` and `a in R`. Counterexample: `x = y = -1`, `a = 1/2`, `sqrt(x)*sqrt(y) != sqrt(x*y)`. 
3. `(x**a)**b == x**(a*b)`
    - Sufficient to hold when `b in Z`. Counterexample: `x = -1`, `a = 2` and `b = 1/2`, `sqrt(x**2) != x` and `sqrt(1/x) != 1/sqrt(x)`. 

This is important to remember, because by default, SymPy will **NOT** perform simplifications if they are not true in general. 

- In order to make SymPy perform simplifications involving identities that are only true under certain assumptions, we need to put assumptions on our Symbols. We will undertake a full discussion of the assumptions system later, but for now, all we need to know are the following.
    - By default, SymPy Symbols are assumed to be *complex* (elements of `C`). That is, a simplification will **NOT** be applied to an expression with a given Symbol unless it holds for *all complex numbers*.
    - Symbols can be given different assumptions by passing the assumption to `symbols()`. For the rest of this section, we will be assuming that `x` and `y` are *positive*, and that `a` and `b` are *real*. We will leave `z`, `t`, and `c` as *arbitrary complex* Symbols to demonstrate what happens in that case. 
```
>>> x, y = symbols('x y', positive=True)
>>> a, b = symbols('a b', real=True)
>>> z, t, c = symbols('z t c')
```

**Note**: In SymPy, `sqrt(x)` is just a shortcut to `x**Rational(1, 2)`. They are *exactly the same* object. 

```
>>> sqrt(x) == x**Rational(1, 2)
True
```

#### 📌 `powsimp`

`powsimp()` applies identities 1 and 2 from above, from left to right.

```
>>> powsimp(x**a*x**b)
x**(a + b)

>>> powsimp(x**a*y**a)
(x*y)**a
```

Notice that `powsimp()` **refuses** to do the simplification if it is not valid. 

```
>>> powsimp(t**c*z**c)
t**c*z**c
```

If you know that you want to apply this simplification, but you don’t want to mess with assumptions, you can pass the `force=True` flag. This will force the simplification to take place, regardless of assumptions.

```
>>> powsimp(t**c*z**c, force=True)
(t*z)**c
```

Note that in some instances, in particular, when the exponents are integers or rational numbers, and identity 2 holds, it will be applied automatically.

```
>>> (z*t)**2
t**2 * z**2

>>> sqrt(x*y)
sqrt(x) * sqrt(y)
```

This means that it will be **impossible** to undo this identity with `powsimp()`, because even if `powsimp()` were to put the bases together, they would be automatically split apart again. 

```
>>> powsimp(z**2*t**2)
t**2 * z**2

>>> powsimp(sqrt(x)*sqrt(y))
sqrt(x) * sqrt(y)
```

#### 📌 `expand_power_exp` / `expand_power_base`

`expand_power_exp()` and `expand_power_base()` apply identities 1 and 2 from right to left, respectively. 

```
>>> expand_power_exp(x**(a + b))
x**a * x**b
```

```
>>> expand_power_base((x*y)**a)
x**a * y**a
```

As with `powsimp()`, identity 2 is **NOT** applied if it is not valid.

```
>>> expand_power_base((z*t)**c)
(t*z)**c
```

And as with `powsimp()`, you can force the expansion to happen without fiddling with assumptions by using `force=True`.

```
expand_power_base((z*t)**c, force=True)
t**c * z**c
```

As with identity 2, identity 1 is applied automatically if the power is a number, and hence can **NOT** be undone with `expand_power_exp()`.

```
>>> x**2*x**3
x**5

>>> expand_power_exp(x**5)
x**5
```

#### 📌 `powdenest`

`powdenest()` applies identity 3, from left to right.

```
>>> powdenest((x**a)**b)
x**(a*b)
```

As before, the identity is **NOT** applied if it is not true under the given assumptions. 

```
>>> powdenest((z**a)**b)
(z**a)**b
```

And as before, this can be manually overridden with `force=True`.

```
>>> powdenest((z**a)**b, force=True)
z**(a*b)
```

### 🌱 Exponentials and logarithms

**Note**: In SymPy, as in Python and most programming languages, `log` is the *natural logarithm*, also known as `ln`. SymPy automatically provides an alias `ln = log` in case you forget this.

```
>>> ln(x)
log(x)
```

Logarithms have similar issues as powers. There are two main identities

1. `log(x * y) == log(x) + log(y)`
2. `log(x**n) == n * log(x)`

Neither identity is true for arbitrary *complex* `x` and `y`, due to the branch cut in the complex plane for the complex logarithm. However, sufficient conditions for the identities to hold are if `x`and `y` are *positive* and `n` is *real*. 

```
>>> x, y = symbols('x y', positive=True)
>>> n = symbols('n', real=True)
```

As before, `z` and `t` will be Symbols with no additional assumptions.

Note that the identity `log(x / y) == log(x) − log(y)` is a special case of identities 1 and 2 by `log(x * y) == log(x * 1/y) == log(x) + log(y**(−1)) == log(x) − log(y)`, and thus it also holds if `x` and `y` are *positive*, but may **NOT** hold in general.

We also see that `log(e * x) == x` comes from `log(e * x) = x * log(e) == x`, and thus holds when `x` is *real* (and it can be verified that it does **NOT** hold in general for *arbitrary complex* `x`, for example, `log(e*x + 2*pi*i) == log(e*x) == x != x + 2*pi*i)`. 

#### 📌 `expand_log`

To apply identities 1 and 2 from left to right, use expand_log(). As always, the identities will not be applied unless they are valid. 

```
>>> expand_log(log(x*y))
log(x) + log(y)

>>> expand_log(log(x/y))
log(x) - log(y)

>>> expand_log(log(x**2))
2⋅* log(x)

>>> expand_log(log(x**n))
n⋅* log(x)

>>> expand_log(log(z*t))
log(t⋅* z)
```

As with `powsimp()` and `powdenest()`, `expand_log()` has a force option that can be used to ignore assumptions. 

```
>>> expand_log(log(z**2))
log(z**2)

>>> expand_log(log(z**2), force=True)
2*log(z)
```

#### 📌 `logcombine`

To apply identities 1 and 2 from right to left, use logcombine().

```
>>> logcombine(log(x) + log(y))
log(x*y)

>>> logcombine(n*log(x))
log(x**n)

>>> logcombine(n*log(z))
n*log(z)
```

`logcombine()` also has a force option that can be used to ignore assumptions.

```
logcombine(n*log(z), force=True)
log(z**n)
```

### 🌱 Special Functions

SymPy implements dozens of special functions, ranging from functions in combinatorics to mathematical physics.

An extensive list of the special functions included with SymPy and their documentation is at the Functions Module page.

For the purposes of this tutorial, let’s introduce a few special functions in SymPy.

Let’s define `x`, `y`, and `z` as regular, complex Symbols, removing any assumptions we put on them in the previous section. We will also define `k`, `m`, and `n`. 

```
>>> x, y, z = symbols('x y z')
>>> k, m, n = symbols('k m n')
```

#### 📌 `factorial`

The factorial function is `factorial`. `factorial(n)` represents `n! = 1 * 2 * ... * (n − 1) * n`. `n!` represents the number of permutations of `n` distinct items. 

```
>>> factorial(n)
n!
```

#### 📌 `binomial`

The binomial coefficient function is `binomial`. `binomial(n, k)` represents `nCk`, the number of ways to choose `k` items from a set of `n` distinct items. It is pronounced “`n` choose `k`”. 

```
>>> binomial(n, k)
binomial(n, k)
```

#### 📌 `gamma`

The factorial function is closely related to the gamma function, `gamma`. `gamma(z)` represents 

![](https://github.com/AXIHIXA/Memo/blob/master/notes/py/sympy/gamma_func.gif) 

which for positive integer `z` is the same as `(z − 1)!`. 

```
>>> gamma(z)
gamma(z)
```

#### 📌 `hyper`

The generalized hypergeometric function is `hyper`. `hyper([a_1, ..., a_p], [b_1, ..., b_q], z)` is `pFq`. 

```
>>> hyper([1, 2], [3], z)
hyper([1, 2], [3], z)
```

#### 📌 `rewrite`

A common way to deal with special functions is to rewrite them in terms of one another. This works for any function in SymPy, not just special functions. To rewrite an expression in terms of a function, use `expr.rewrite(function)`. For example,

```
>>> tan(x).rewrite(sin)
2*sin(x)**2/sin(2*x)

>>> factorial(x).rewrite(gamma)
Γ(x + 1)
```

For some tips on applying more targeted rewriting, see the [Advanced Expression Manipulation](https://docs.sympy.org/latest/tutorial/manipulation.html#tutorial-manipulation) section. 

#### 📌 `expand_func`

To expand special functions in terms of some identities, use `expand_func()`. For example

```
>>> expand_func(gamma(x + 3))
x*(x + 1)*(x + 2)*Γ(x)
```

#### 📌 `hyperexpand`

To rewrite hyper in terms of more standard functions, use `hyperexpand()`. 

```
>>> hyperexpand(hyper([1, 1], [2], z))
-log(1 - z)
───────────────
     z
```

`hyperexpand()` also works on the more general Meijer G-function (see [its documentation](https://docs.sympy.org/latest/modules/functions/special.html#sympy.functions.special.hyper.meijerg) for more information).

#### 📌 `combsimp`

To simplify combinatorial expressions, use `combsimp()`. 

```
>>> n, k = symbols('n k', integer = True)
>>> combsimp(factorial(n)/factorial(n - 3))
n*(n - 2)*(n - 1)

>>> combsimp(binomial(n+1, k+1)/binomial(n, k))
(n + 1)/(k + 1)
```

#### 📌 `gammasimp`

To simplify expressions with gamma functions or combinatorial functions with non-integer argument, use `gammasimp()`. 

```
>>> gammasimp(gamma(x)*gamma(1 - x))
pi/sin(pi*x)
```

### 🌱 Example: Continued Fractions

Let’s use SymPy to explore continued fractions. A continued fraction is an expression of the form

![](https://github.com/AXIHIXA/Memo/blob/master/notes/py/sympy/continued%20_fractions.gif)


where `a0`, ..., `an` are integers, and `a1`, ..., `an`are positive. A continued fraction can also be infinite, but infinite objects are more difficult to represent in computers, so we will only examine the finite case here. 

A continued fraction of the above form is often represented as a list `[a0; a1, ..., an]`. Let’s write a simple function that converts such a list to its continued fraction form. The easiest way to construct a continued fraction from a list is to work backwards. Note that despite the apparent symmetry of the definition, the first element, `a0`, must usually be handled differently from the rest. 

```
>>> def list_to_frac(lst):
...    expr = Integer(0)
...    for i in reversed(lst[1:]):
...        expr += i
...        expr = 1/expr
...    return l[0] + expr

>>> list_to_frac([x, y, z])
x + (1/(y + (1/z)))
```

We use `Integer(0)` in `list_to_frac` so that the result will always be a SymPy object, even if we only pass in Python ints. 

```
>>> list_to_frac([1, 2, 3, 4])
43/30
```

Every finite continued fraction is a rational number, but we are interested in symbolics here, so let’s create a symbolic continued fraction. The `symbols()` function that we have been using has a shortcut to create numbered symbols. `symbols('a0:5')` will create the symbols `a0`, `a1`, ..., `a4`. 

```
>>> syms = symbols('a0:5')
>>> syms
(a0, a1, a2, a3, a4)

>>> a0, a1, a2, a3, a4 = syms
>>> frac = list_to_frac(syms)
>>> frac
             1
a0 + ─────────────────────────
               1
     a1 + ──────────────────
                  1
          a2 + ───────────
                    1
               a3 + ────
                    a4
```

This form is useful for understanding continued fractions, but lets put it into standard rational function form using `cancel()`. 

```
>>> frac = cancel(frac)
```

## 🔱 [Calculus](https://docs.sympy.org/latest/tutorial/calculus.html)

This section covers how to do basic calculus tasks such as derivatives, integrals, limits, and series expansions in SymPy. If you are not familiar with the math of any part of this section, you may safely skip it.

```
>>> from sympy import *
>>> x, y, z = symbols('x y z')
```

### 🌱 Derivatives

To take derivatives, use the `diff` function. 

```
>>> diff(cos(x), x)
-sin(x)

>>> diff(exp(x**2), x)
(2*x*e)**(x**2)
```

`diff` can take *multiple derivatives* at once. To take multiple derivatives, pass the *variable as many times* as you wish to differentiate, or pass a *number* after the variable. For example, both of the following find the third derivative of `x**4`. 

```
>>> diff(x**4, x, x, x)
24*x

>>> diff(x**4, x, 3)
24*x
```

You can also take derivatives with respect to many variables at once. Just pass each derivative in order, using the same syntax as for single variable derivatives. For example, each of the following will compute `\dfrac{\partial^7}{\partial x \partial y \partial z^4} e^{xyz}`. 

```
>>> expr = exp(x*y*z)

>>> diff(expr, x, y, y, z, z, z, z)ℯ
x**3*y**2*(x**3*y**3*z**3 + 14*x**2*y**2*z**2 + 52*x*y*z + 48)*exp(x*y*z)

>>> diff(expr, x, y, 2, z, 4)
x**3*y**2*(x**3*y**3*z**3 + 14*x**2*y**2*z**2 + 52*x*y*z + 48)*exp(x*y*z)

>>> diff(expr, x, y, y, z, 4)
x**3*y**2*(x**3*y**3*z**3 + 14*x**2*y**2*z**2 + 52*x*y*z + 48)*exp(x*y*z)
```

`diff` can also be called as a method. The two ways of calling `diff` are exactly the same, and are provided only for convenience. 

```
>>> expr.diff(x, y, y, z, 4)
x**3*y**2*(x**3*y**3*z**3 + 14*x**2*y**2*z**2 + 52*x*y*z + 48)*exp(x*y*z)
```

To create an unevaluated derivative, use the `Derivative` class. It has the same syntax as `diff`. 

```
>>> deriv = Derivative(expr, x, y, y, z, 4)
>>> deriv
Derivative(exp(x*y*z), x, (y, 2), (z, 4))
```

To evaluate an unevaluated derivative, use the `doit` method. 

```
>>> deriv.doit()
x**3*y**2*(x**3*y**3*z**3 + 14*x**2*y**2*z**2 + 52*x*y*z + 48)*exp(x*y*z)
```

These unevaluated objects are useful for delaying the evaluation of the derivative, or for printing purposes. They are also used when SymPy does not know how to compute the derivative of an expression (for example, if it contains an undefined function, which are described in the [Solving Differential Equations](https://docs.sympy.org/latest/tutorial/solvers.html#tutorial-dsolve) section).

Derivatives of unspecified order can be created using tuple `(x, n)` where `n` is the order of the derivative with respect to `x`. 

```
>>> m, n, a, b = symbols('m n a b')
>>> expr = (a*x + b)**m
>>> expr.diff((x, n))
Derivative((a*x + b)**m, (x, n))
```

### 🌱 Integrals

To compute an integral, use the `integrate` function. There are two kinds of integrals, definite and indefinite. 

To compute an *indefinite integral*, that is, an antiderivative, or primitive, just pass the variable after the expression. 

```
>>> integrate(cos(x), x)
sin(x)
```

Note that SymPy does **NOT** include the constant of integration. If you want it, you can add one yourself, or rephrase your problem as a differential equation and use `dsolve` to solve it, which does add the constant (see [Solving Differential Equations](https://docs.sympy.org/latest/tutorial/solvers.html#tutorial-dsolve)).

To compute a *definite integral*, pass the *limit tuple*  `(integration_variable, lower_limit, upper_limit)`. For example, to compute `\int_0^\infty e^{-x} \, \mathrm{d} x`, we would do

```
>>> integrate(exp(-x), (x, 0, oo))
1
```

**Quick Tip**: ∞ in SymPy is `oo` (that’s the lowercase letter “oh” twice). This is because `oo` looks like ∞, and is easy to type. 

As with indefinite integrals, you can pass *multiple limit tuples* to perform a *multiple integral*. For example

```
>>> integrate(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo))
pi
```

If `integrate` is unable to compute an integral, it returns an unevaluated `Integral` object.

```
>>> expr = integrate(x**x, x)
>>> expr
>>> Integral(x**x, x)
```

As with `Derivative`, you can create an unevaluated integral using `Integral`. To later evaluate this integral, call `doit`. 

```
>>> expr = Integral(log(x)**2, x)
>>> expr
Integral(log(x)**2, x)

>>> expr.doit()
x*log(x)**2 - 2*x*log(x) + 2*x
```

`integrate` uses powerful algorithms that are always improving to compute both definite and indefinite integrals, including heuristic pattern matching type algorithms, a partial implementation of the [Risch Algorithm](https://en.wikipedia.org/wiki/Risch_algorithm), and an algorithm using [Meijer G-functions](https://en.wikipedia.org/wiki/Meijer_G-function) that is useful for computing integrals in terms of special functions, especially definite integrals. 

### 🌱 Limits

SymPy can compute symbolic limits with the `limit` function. The syntax to compute the limit of `f(x)` when `x` approaches `x0` is `limit(f(x), x, x0)`. 

```
>>> limit(sin(x)/x, x, 0)
1
```

`limit` should be used instead of `subs` whenever the point of evaluation is a *singularity*. Even though SymPy has objects to represent `oo`, using them for evaluation is **NOT** reliable because they do not keep track of things like rate of growth. Also, things like `oo − oo` and `oo * oo` return `nan` (not-a-number). For example

```
>>> expr = x**2/exp(x)

>>> expr.subs(x, oo)
nan

>>> limit(expr, x, oo)
0
```

Like `Derivative` and `Integral`, `limit` has an unevaluated counterpart, `Limit`. To evaluate it, use `doit`. 

```
>>> expr = Limit((cos(x) - 1)/x, x, 0)

>>> expr
Limit((cos(x) - 1)/x, x, 0)

>>> expr.doit()
0
```

To evaluate a limit at *one side only*, pass `'+'` or `'-'` as a fourth argument to `limit`. For example 

```
>>> limit(1/x, x, 0, '+')
oo

>>> limit(1/x, x, 0, '-')
-oo
```

### 🌱 Series Expansion

SymPy can compute asymptotic series expansions of functions around a point. To compute the expansion of `f(x)` around the point `x = x0` terms of order `x**n`, use `f(x).series(x, x0, n)`. `x0` and `n` can be omitted, in which case the defaults `x0 = 0` and `n = 6` will be used. 

```
>>> expr = exp(sin(x))
>>> expr.series(x, 0, 4)
1 + x + x**2/2 + O(x**4)
```

The `O(x**4)` term at the end represents the Landau order term at `x = 0` (not to be confused with big O notation used in computer science, which generally represents the Landau order term at `x = oo`). It means that all x terms with power greater than or equal to `x**4` are omitted. Order terms can be created and manipulated outside of `series`. They automatically absorb higher order terms. 

```
>>> x + x**3 + x**6 + O(x**4)
x + x**3 + O(x**4)

>>> x*O(1)
O(x)
```

If you do not want the order term, use the `removeO` method.

```
>>> expr.series(x, 0, 4).removeO()
x**2/2 + x + 1
```

The O notation supports arbitrary limit points (other than 0):

```
>>> exp(x - 6).series(x, x0=6)
-5 + (x - 6)**2/2 + (x - 6)**3/6 + (x - 6)**4/24 + (x - 6)**5/120 + x + O((x - 6)**6, (x, 6))
```

### 🌱 Finite Differences

So far we have looked at expressions with analytic derivatives and primitive functions respectively. But what if we want to have an expression to estimate a derivative of a curve for which we lack a closed form representation, or for which we don’t know the functional values for yet. One approach would be to use a finite difference approach.

The simplest way the differentiate using finite differences is to use the `differentiate_finite` function: 

```
>>> f, g = symbols('f g', cls=Function)
>>> differentiate_finite(f(x)*g(x))
-f(x - 1/2)*g(x - 1/2) + f(x + 1/2)*g(x + 1/2)
```

If you already have a `Derivative` instance, you can use the `as_finite_difference` method to generate approximations of the derivative to arbitrary order: 

```
>>> f = Function('f')
>>> dfdx = f(x).diff(x)
>>> dfdx.as_finite_difference()
-f(x - 1/2) + f(x + 1/2)
```

here the first order derivative was approximated around x using a minimum number of points (2 for 1st order derivative) evaluated equidistantly using a step-size of 1. We can use arbitrary steps (possibly containing symbolic expressions):

```
>>> f = Function('f')
>>> d2fdx2 = f(x).diff(x, 2)
>>> h = Symbol('h')
>>> d2fdx2.as_finite_difference([-3*h, -h, 2*h])
f(-3*h)/(5*h**2) - f(-h)/(3*h**2) + 2*f(2*h)/(15*h**2)
```

If you are just interested in evaluating the weights, you can do so manually: 

```
>>> finite_diff_weights(2, [-3, -1, 2], 0)[-1][-1]
[1/5, -1/3, 2/15]
```

note that we only need the last element in the last sublist returned from `finite_diff_weights`. The reason for this is that the function also generates weights for lower derivatives and using fewer points (see the documentation of `finite_diff_weights` for more details).

If using `finite_diff_weights` directly looks complicated, and the `as_finite_difference` method of `Derivative` instances is not flexible enough, you can use `apply_finite_diff` which takes `order`, `x_list`, `y_list` and `x0` as parameters: 

```
>>> x_list = [-3, 1, 2]
>>> y_list = symbols('a b c')
>>> apply_finite_diff(1, x_list, y_list, 0)
-3*a/20 - b/4 + 2*c/5
```

## 🔱 [Solvers](https://docs.sympy.org/latest/tutorial/solvers.html)

```
>>> from sympy import *
>>> x, y, z = symbols('x y z')
```

### 🌱 A Note about Equations

Recall from the [Gotchas](https://docs.sympy.org/latest/tutorial/gotchas.html#tutorial-gotchas-equals) section of this tutorial that symbolic equations in SymPy are **NOT** represented by `=` or `==`, but by `Eq`. 

```
>>> Eq(x, y)
x = y
```

However, there is an even easier way. In SymPy, any expression not in an `Eq` is automatically assumed to equal 0 by the solving functions. Since `a = b` iff. `a − b = 0`, this means that instead of using `x == y`, you can just use `x - y`. For example

```
>>> solveset(Eq(x**2, 1), x)
{-1, 1}

>>> solveset(Eq(x**2 - 1, 0), x)
{-1, 1}

>>> solveset(x**2 - 1, x)
{-1, 1}
```

This is particularly useful if the equation you wish to solve is already equal to 0. Instead of typing `solveset(Eq(expr, 0), x)`, you can just use `solveset(expr, x)`. 

### 🌱 Solving Equations Algebraically

The main function for solving algebraic equations is `solveset`. The syntax for `solveset` is 
```
solveset(equation, variable=None, domain=S.Complexes) 
```
Where `equations` may be in the form of `Eq` instances or expressions that are assumed to be equal to zero.

Please note that there is another function called `solve` which can also be used to solve equations. The syntax is 
```
solve(equations, variables)
```
However, it is recommended to use `solveset` instead. 

When solving a single equation, the output of `solveset` is a `FiniteSet` or an `Interval` or `ImageSet` of the solutions. 

```
>>> solveset(x**2 - x, x)
{0, 1}

>>> solveset(x - x, x, domain=S.Reals)
Reals

>>> solveset(sin(x) - 1, x, domain=S.Reals)
ImageSet(Lambda(_n, 2*_n*pi + pi/2), Integers)  # 2*n*pi + 1/2*pi, n in Z
```

If there are no solutions, an `EmptySet` is returned and if it is not able to find solutions then a `ConditionSet` is returned. 

```
>>> solveset(exp(x), x)     # No solution exists
EmptySet

>>> solveset(cos(x) - x, x)  # Not able to find solution
ConditionSet{x | x ∊ ℂ ∧ (-x + cos(x) = 0)}
```

- In the `solveset` module, the linear system of equations is solved using `linsolve`. In future we would be able to use linsolve directly from `solveset`. Following is an example of the syntax of `linsolve`.
    - List of Equations Form:
    ```
    linsolve([x + y + z - 1, x + y + 2*z - 3], (x, y, z))
    {(-y - 1, y, 2)}
    ```
    - Augmented Matrix Form: 
    ```
    linsolve(Matrix(([1, 1, 1, 1], [1, 1, 2, 3])), (x, y, z))
    {(-y - 1, y, 2)}
    ```
    - `A*x = b` Form
    ```
    >>> M = Matrix(((1, 1, 1, 1), (1, 1, 2, 3)))
    >>> system = A, b = M[:, :-1], M[:, -1]

    >>> linsolve(system, x, y, z)
    {(-y - 1, y, 2)}
    ```

**Note**: The order of solution corresponds the order of given symbols. 

- In the `solveset` module, the non linear system of equations is solved using `nonlinsolve`. Following are examples of `nonlinsolve`. 
    1. When only real solution is present: 
    ```
    >>> a, b, c, d = symbols('a, b, c, d', real=True)

    >>> nonlinsolve([a**2 + a, a - b], [a, b])
    {(-1, -1), (0, 0)}

    >>> nonlinsolve([x*y - 1, x - 2], x, y)
    {(2, 1/2)}
    ```
    2. When only complex solution is present: 
    ```
    >>> nonlinsolve([x**2 + 1, y**2 + 1], [x, y])
    {(-i, -i), (-i, i), (i, -i), (i, i)}
    ```
    3. When both real and complex solution are present:
    ```
    >>> from sympy import sqrt
    >>> system = [x**2 - 2*y**2 -2, x*y - 2]
    >>> vars = [x, y]
    >>> nonlinsolve(system, vars)
    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}
    ```
    ```
    >>> system = [exp(x) - sin(y), 1/y - 3]
    >>> nonlinsolve(system, vars)
    {({2⋅n⋅pi + log(sin(1/3)) | n in Zℤ}, 1/3)}
    ```
    4. When the system is positive-dimensional system (has infinitely many solutions): 
    ```
    >>> nonlinsolve([x*y, x*y - x], [x, y])
    {(0, y)}

    >>> system = [a**2 + a*c, a - b]

    >>> nonlinsolve(system, [a, b])
    {(0, 0), (-c, -c)}
    ```
- **Note**: 
    1. The order of solution corresponds the order of given symbols. 
    2. Currently `nonlinsolve` doesn’t return solution in form of `LambertW` (if there is solution present in the form of `LambertW`). `solve` can be used for such cases: 
    ```
    >>> solve([x**2 - y**2/exp(x)], [x, y], dict=True)
    [{x: 2*LambertW(-y/2)}, {x: 2*LambertW(y/2)}]
    ```
    3. Currently `nonlinsolve` is not properly capable of solving the system of equations having trigonometric functions. `solve` can be used for such cases (but does not give all solution): 
    ```
    >>> solve([sin(x + y), cos(x - y)], [x, y])
    [(-3*pi/4, 3*pi/4), (-pi/4, pi/4), (pi/4, 3*pi/4), (3*pi/4, pi/4)]
    ```

`solveset` reports each solution only once. To get the solutions of a polynomial including multiplicity use `roots`. 

```
>>> solveset(x**3 - 6*x**2 + 9*x, x)
{0, 3}

>>> roots(x**3 - 6*x**2 + 9*x, x)
{0: 1, 3: 2}
```

The output `{0: 1, 3: 2}` of `roots` means that `0` is a root of multiplicity 1 and `3` is a root of multiplicity 2. 

**Note**: Currently solveset is not capable of solving Equations solvable by LambertW (Transcendental equation solver). `solve` can be used for such cases: 

```
>>> solve(x*exp(x) - 1, x )
[W(1)]
```

### 🌱 Solving Differential Equations

To solve differential equations, use `dsolve`. First, create an undefined function by passing `cls=Function` to the `symbols` function. 

```
f, g = symbols('f g', cls=Function)
```

`f` and `g` are now undefined functions. We can call `f(x)`, and it will represent an unknown function. 

```
>>> f(x)
f(x)
```

Derivatives of `f(x)` are unevaluated.

```
>>> f(x).diff(x)
Derivative(f(x), x)
```

To represent the differential equation `f''(x) − 2f'(x) + f(x) = sin(x)`, we would thus use

```
>>> diffeq = Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), sin(x))
>>> diffeq
Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), sin(x))
```

To solve the ODE, pass it and the function to solve for to `dsolve`. 

```
>>> dsolve(diffeq, f(x))
Eq(f(x), (C1 + C2*x)*exp(x) + cos(x)/2)
```

`dsolve` returns an instance of `Eq`. This is because in general, solutions to differential equations cannot be solved explicitly for the function. 

```
>>> dsolve(f(x).diff(x)*(1 - sin(f(x))) - 1, f(x))
x - f(x) - cos(f(x)) = C₁
```

The arbitrary constants in the solutions from `dsolve` are symbols of the form `C1`, `C2`, `C3`, and so on.

## 🔱 [Matrices](https://docs.sympy.org/latest/tutorial/matrices.html)

To make a matrix in SymPy, use the `Matrix` object. A matrix is constructed by providing a list of row vectors that make up the matrix. For example, to construct the matrix `[[1, -1], [3, 4], [0, 2]]`, use

```
>>> Matrix([[1, -1], [3, 4], [0, 2]])
Matrix([
[1, -1],
[3,  4],
[0,  2]])
```

To make it easy to make column vectors, a list of elements is considered to be a *column vector*. 

```
>>> Matrix([1, 2, 3])
Matrix([
[1],
[2],
[3]])
```

Matrices are manipulated just like any other object in SymPy or Python. 

```
>>> M = Matrix([[1, 2, 3], [3, 2, 1]])
>>> N = Matrix([0, 1, 1])
>>> M*N
Matrix([
[5],
[3]])
```

One important thing to note about SymPy matrices is that, unlike every other object in SymPy, they are *mutable*. This means that they can be modified in place, as we will see below. The downside to this is that `Matrix` can **NOT** be used in places that require immutability, such as inside other SymPy expressions or as keys to dictionaries. If you need an immutable version of `Matrix`, use `ImmutableMatrix`. 

### 🌱 Basic Operations

#### 📌 Shape

Here are some basic operations on `Matrix`. To get the shape of a matrix use `shape`

```
>>> M = Matrix([[1, 2, 3], [-2, 0, 4]])
Matrix([
[ 1, 2, 3],
[-2, 0, 4]])

>>> M.shape
(2, 3)
```

#### 📌 Accessing Rows and Columns

Matrices are stored in row-major. `M[i]` gets the `i`-th element of the whole flattened matrix. 

To get an individual row or column of a matrix, use `row` or `col`. For example, `M.row(0)` will get the first row. `M.col(-1)` will get the last column. 

```
>>> M.row(0)
[1  2  3]

>>> M.col(-1)
[3]
[ ]
[6]
```

#### 📌 Deleting and Inserting Rows and Columns

To delete a row or column, use `row_del` or `col_del`. These operations will modify the Matrix *in place*. 

```
>>> M.col_del(0)
>>> M
[2  3]
[⎢   ]⎥
[0  4]

>>> M.row_del(1)
>>> M
[2  3]
```

To insert rows or columns, use `row_insert` or `col_insert`. These operations do **NOT** operate in place. 

```
>>> M
[2  3]

>>> M = M.row_insert(1, Matrix([[0, 4]]))
>>> M
[2  3]
[    ]
[0  4]

>>> M = M.col_insert(0, Matrix([1, -2]))
>>> M
[1   2  3]
[        ]
[-2  0  4]
```

Unless explicitly stated, the methods mentioned below do not operate in place. In general, a method that does not operate in place will return a new `Matrix` and a method that does operate in place will return `None`. 

### 🌱 Basic Methods

As noted above, simple operations like addition and multiplication are done just by using `+`, `*`, and `**`. To find the inverse of a matrix, just raise it to the `-1` power. 

```
>>> M = Matrix([[1, 3], [-2, 3]])
>>> N = Matrix([[0, 3], [0, 7]])

>>> M + N
[1   6 ]⎤
[      ]      ⎥
[-2  10]⎦

>>> M*N
[0  24]
[     ]     ⎥
[0  15]

>>> 3*M
[3   9]
[     ]     ⎥
[-6  9]

>>> M**2
[-5  12]
[      ]      ⎥
[-8  3 ]

>>> M**-1
[1/3  -1/3]
[         ]         ⎥
[2/9  1/9 ]⎦

>>> N**-1
Traceback (most recent call last):
...
NonInvertibleMatrixError: Matrix det == 0; not invertible. 
```

To take the transpose of a Matrix, use `T`. 

```
>>> M = Matrix([[1, 2, 3], [4, 5, 6]])
>>> M
[1  2  3]
[       ]       ⎥
[4  5  6]

>>> M.T
[1  4]
[    ]    ⎥
[2  5]
[    ]    ⎥
[3  6]
```

### 🌱 Matrix Constructors

Several constructors exist for creating common matrices. To create an identity matrix, use `eye`. `eye(n)` will create an `n*n` identity matrix. 

```
>>> eye(3)
[1  0  0]
[       ]
[0  1  0]
[       ]
[0  0  1]

>>> eye(4)
[1  0  0  0]
[          ]
[0  1  0  0]
[          ]
[0  0  1  0]
[          ]
[0  0  0  1]
```

To create a matrix of all zeros, use `zeros`. `zeros(n, m)` creates an `n*m` matrix of `0`s. 

```
>>> zeros(2, 3)
[0  0  0]
[       ]
[0  0  0]
```

Similarly, `ones` creates a matrix of ones. 

```
>>> ones(3, 2)
[1  1]
[    ]
[1  1]
[    ]
[1  1]
```

To create diagonal matrices, use `diag`. The arguments to `diag` can be either numbers or matrices. A number is interpreted as a `1*1` matrix. The matrices are stacked diagonally. The remaining elements are filled with `0`s. 

```
>>> diag(1, 2, 3)
[1  0  0]
[       ]
[0  2  0]
[       ]
[0  0  3]

>>> diag(-1, ones(2, 2), Matrix([5, 7, 5]))
[-1  0  0  0]
[           ]
[0   1  1  0]
[           ]
[0   1  1  0]
[           ]
[0   0  0  5]
[           ]
[0   0  0  7]
[           ]
[0   0  0  5]
```

### 🌱 Advanced Methods

#### 📌 Determinant

To compute the determinant of a matrix, use `det`. 

```
>>> M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
>>> M
[1  0   1]
[        ]        ⎥
[2  -1  3]
[        ]        ⎥
[4  3   2]

>>> M.det()
-1
```

#### 📌 RREF

To put a matrix into reduced row echelon form, use `rref`. `rref` returns a tuple of two elements. The first is the reduced row echelon form, and the second is a tuple of indices of the pivot columns.

```
>>> M = Matrix([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
>>> M
[1   0   1   3 ]
[              ]
[2   3   4   7 ]
[              ]
[-1  -3  -3  -4]

>>> M.rref()
 [1  0   1    3 ]         
 [              ]         
([0  1  2/3  1/3], (0, 1))
 [              ]         
 [0  0   0    0 ]         
```

**Note**: The first element of the tuple returned by rref is of type `Matrix`. The second is of type `tuple`. 

#### 📌 Nullspace

To find the nullspace of a matrix, use `nullspace`. `nullspace` returns a `list` of column vectors that span the nullspace of the matrix. 

```
>>> M = Matrix([[1, 2, 3, 0, 0], [4, 10, 0, 0, 1]])
>>> M
⎡1  2   3  0  0⎤
⎢              ⎥
⎣4  10  0  0  1⎦

>>> M.nullspace()
⎡⎡-15⎤  ⎡0⎤  ⎡ 1  ⎤⎤
⎢⎢   ⎥  ⎢ ⎥  ⎢    ⎥⎥
⎢⎢ 6 ⎥  ⎢0⎥  ⎢-1/2⎥⎥
⎢⎢   ⎥  ⎢ ⎥  ⎢    ⎥⎥
⎢⎢ 1 ⎥, ⎢0⎥, ⎢ 0  ⎥⎥
⎢⎢   ⎥  ⎢ ⎥  ⎢    ⎥⎥
⎢⎢ 0 ⎥  ⎢1⎥  ⎢ 0  ⎥⎥
⎢⎢   ⎥  ⎢ ⎥  ⎢    ⎥⎥
⎣⎣ 0 ⎦  ⎣0⎦  ⎣ 1  ⎦⎦
```

#### 📌 Columnspace

To find the columnspace of a matrix, use `columnspace`. `columnspace` returns a `list` of column vectors that span the columnspace of the matrix. 

```
>>> M = Matrix([[1, 1, 2], [2 ,1 , 3], [3 , 1, 4]])
>>> M
⎡1  1  2⎤
⎢       ⎥
⎢2  1  3⎥
⎢       ⎥
⎣3  1  4⎦

>>> M.columnspace()
⎡⎡1⎤  ⎡1⎤⎤
⎢⎢ ⎥  ⎢ ⎥⎥
⎢⎢2⎥, ⎢1⎥⎥
⎢⎢ ⎥  ⎢ ⎥⎥
⎣⎣3⎦  ⎣1⎦⎦
```

#### 📌 Eigenvalues, Eigenvectors, and Diagonalization

To find the eigenvalues of a matrix, use `eigenvals`. `eigenvals` returns a dictionary of `eigenvalue: algebraic multiplicity` pairs (similar to the output of [roots](https://docs.sympy.org/latest/tutorial/solvers.html#tutorial-roots)). 

```
>>> M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
>>> M
⎡3  -2  4   -2⎤
⎢             ⎥
⎢5  3   -3  -2⎥
⎢             ⎥
⎢5  -2  2   -2⎥
⎢             ⎥
⎣5  -2  -3  3 ⎦

>>> M.eigenvals()
{-2: 1, 3: 1, 5: 2}
```

This means that `M` has eigenvalues -2, 3, and 5, and that the eigenvalues -2 and 3 have algebraic multiplicity 1 and that the eigenvalue 5 has algebraic multiplicity 2. 

To find the eigenvectors of a matrix, use `eigenvects`. `eigenvects` returns a list of tuples of the form `(eigenvalue:algebraic multiplicity, [eigenvectors])`. 

```
>>> M.eigenvects()
          [0]            [1]            [1]  [0 ]   
          [ ]            [ ]            [ ]  [  ]   
          [1]            [1]            [1]  [-1]   
[(-2, 1, [[ ]]), (3, 1, [[ ]]), (5, 2, [[ ], [  ]])]
          [1]            [1]            [1]  [0 ]   
          [ ]            [ ]            [ ]  [  ]   
          [1]            [1]            [0]  [1 ]   
```

This shows us that, for example, the eigenvalue 5 also has geometric multiplicity 2, because it has two eigenvectors. Because the algebraic and geometric multiplicities are the same for all the eigenvalues, `M` is diagonalizable. 

To diagonalize a matrix, use `diagonalize`. `diagonalize` returns a tuple `(P, D)`, where `D` is diagonal and `M = PDP^{−1}`.

```
>>> P
⎡0  1  1  0 ⎤
⎢           ⎥
⎢1  1  1  -1⎥
⎢           ⎥
⎢1  1  1  0 ⎥
⎢           ⎥
⎣1  1  0  1 ⎦

>>> D
⎡-2  0  0  0⎤
⎢           ⎥
⎢0   3  0  0⎥
⎢           ⎥
⎢0   0  5  0⎥
⎢           ⎥
⎣0   0  0  5⎦

>>> P*D*P**-1
⎡3  -2  4   -2⎤
⎢             ⎥
⎢5  3   -3  -2⎥
⎢             ⎥
⎢5  -2  2   -2⎥
⎢             ⎥
⎣5  -2  -3  3 ⎦

>>> P*D*P**-1 == M
True
```

Note that since `eigenvects` also includes the eigenvalues, you should use it instead of eigenvals if you also want the eigenvectors. However, as computing the eigenvectors may often be costly, `eigenvals` should be preferred if you only wish to find the eigenvalues.

If all you want is the characteristic polynomial, use `charpoly`. This is more efficient than `eigenvals`, because sometimes symbolic roots can be expensive to calculate. 

**Quick Tip**: `lambda` is a reserved keyword in Python, so to create a Symbol called λ, while using the same names for SymPy Symbols and Python variables, use `lamda` (without the b). It will still pretty print as λ. 

```
>>> lamda = symbols('lamda')
>>> p = M.charpoly(lamda)
>>> factor(p.as_expr())
(λ - 5)**2 * (λ - 3)⋅* (λ + 2)
```

### 🌱 Possible Issues

#### 📌 Zero Testing


If your matrix operations are failing or returning wrong answers, the common reasons would likely be from zero testing. If there is an expression not properly zero-tested, it can possibly bring issues in finding pivots for gaussian elimination, or deciding whether the matrix is inversible, or any high level functions which relies on the prior procedures.

Currently, the SymPy’s default method of zero testing `_iszero` is only guaranteed to be accurate in some limited domain of numerics and symbols, and any complicated expressions beyond its decidability are treated as `None`, which behaves similarly to logical `False`.

The list of methods using zero testing procedures are as follows:

`echelon_form` , `is_echelon` , `rank` , `rref` , `nullspace` , `eigenvects` , `inverse_ADJ` , `inverse_GE` , `inverse_LU` , `LUdecomposition` , `LUdecomposition_Simple` , `LUsolve`

They have property `iszerofunc` opened up for user to specify zero testing method, which can accept any function with single input and boolean output, while being defaulted with `_iszero`.

Here is an example of solving an issue caused by undertested zero. 

```
>>> from sympy import *
>>> q = Symbol("q", positive = True)
>>> m = Matrix([[-2*cosh(q/3),      exp(-q),            1],
...             [      exp(q), -2*cosh(q/3),            1],
...             [           1,            1, -2*cosh(q/3)]])

>>> m.nullspace()
[]
```

You can trace down which expression is being underevaluated, by injecting a custom zero test with warnings enabled. 

```
>>> import warnings

>>> def my_iszero(x):
...    try:
...        result = x.is_zero
...   except AttributeError:
...        result = None
...        
...    # Warnings if evaluated into None
...    if result is None:
...        warnings.warn("Zero testing of {} evaluated into None".format(x))
...    return result


>>> m.nullspace(iszerofunc=my_iszero) 
__main__:9: UserWarning: Zero testing of 4*cosh(q/3)**2 - 1 evaluated into None
__main__:9: UserWarning: Zero testing of (-exp(q) - 2*cosh(q/3))*(-2*cosh(q/3) - exp(-q)) - (4*cosh(q/3)**2 - 1)**2 evaluated into None
__main__:9: UserWarning: Zero testing of 2*exp(q)*cosh(q/3) - 16*cosh(q/3)**4 + 12*cosh(q/3)**2 + 2*exp(-q)*cosh(q/3) evaluated into None
__main__:9: UserWarning: Zero testing of -(4*cosh(q/3)**2 - 1)*exp(-q) - 2*cosh(q/3) - exp(-q) evaluated into None
[]
```

In this case, `(-exp(q) - 2*cosh(q/3))*(-2*cosh(q/3) - exp(-q)) - (4*cosh(q/3)**2 - 1)**2` should yield zero, but the zero testing had failed to catch. possibly meaning that a stronger zero test should be introduced. For this specific example, rewriting to exponentials and applying simplify would make zero test stronger for hyperbolics, while being harmless to other polynomials or transcendental functions. 

## 🔱 [Advanced Expression Manipulation](https://docs.sympy.org/latest/tutorial/manipulation.html)

### 🌱 Understanding Expression Trees

### 🌱 Recursing through an Expression Tree

### 🌱 Prevent expression evaluation
















































## 🔱

### 🌱

#### 📌

