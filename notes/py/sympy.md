# `matplotlib` Tutorial Notes

- Notes of reading [`SymPy` Tutorial](https://docs.sympy.org/latest/tutorial/index.html)

## [Gotchas](https://docs.sympy.org/latest/tutorial/gotchas.html)

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

Usually, the best practice is to assign Symbols to Python variables of the *same name*, although there are exceptions: Symbol names can contain characters that are not allowed in Python variable names, or may just want to avoid typing long names by assigning Symbols with long names to single letter Python variables. 

To avoid confusion, throughout this tutorial, Symbol names and Python variable names will always coincide. Furthermore, the word “Symbol” will refer to a SymPy Symbol and the word “variable” will refer to a Python variable. 



### 🌱

#### 📌

