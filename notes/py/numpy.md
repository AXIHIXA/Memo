# `NumPy` Tutorial Notes

- Notes of reading:
    - [`Numpy User Guide`](https://numpy.org/doc/1.18/numpy-user.pdf)
    - [`Numpy Tutorial - Tutorialspoint`](https://www.tutorialspoint.com/numpy/index.htm)
    
### 🌱 SETTING UP

- Python library w/ homogeneous multidimensional array object `ndarray`, difference with bulit-in Python sequences: 
    - Fixed size at creation
        - Changing size will create a new array and delete the original
    - All elements be of the same data type, and thus same size
        - Exception: array of (Python, NumPy, ...) objects 
    - Faster operations
        - e.g. `c = a * b` in NumPy expands into precompiles index-based C loop
            - vectorization
            - boardcasting
    - Good eco-system

### 🌱 QUICKSTART TUTORIAL

- Main object: homogeneous multidimensional array `numpy.ndarray`, alias `numpy.array`
    - table of elements (usually numbers)
    - all of the same type
    - indexed by a tuple of non-negative integers 
    - In NumPy, dimensions are called `axes`
- Attributes: 
    - `ndarray.ndim` 
        - the number of axes (dimensions) of the array.
    - `ndarray.shape` 
        - the dimensions of the array. This is a tuple of integers indicating the size of the array in each dimension. For a matrix with n rows and m columns, shape will be `(n, m)`. The length of the shape tuple is therefore the number of axes, `ndim`.
    - `ndarray.size` 
        - the total number of elements of the array. This is equal to the product of the elements of shape. 
    - `ndarray.dtype` 
        - an object describing the type of the elements in the array. One can create or specify dtype’s using standard Python types. Additionally NumPy provides types of its own. `numpy.int32`, `numpy.int16`, and `numpy.float64` are some examples.
    - `ndarray.itemsize` 
        - the size in bytes of each element of the array. For example, an array of elements of type `float64` has `itemsize` 8 (=64/8), while one of type `complex32` has `itemsize` 4 (=32/8). It is equivalent to `ndarray.dtype.itemsize`.
    - `ndarray.data` 
        - the buffer containing the actual elements of the array. Normally, we won’t need to use this attribute because we will access the elements in an array using indexing facilities.
- Array creation
    - `numpy.array`
        - from regular Python list or tuple
            - must feed single sequence rather than multiple numeric arguments
        - `dtype` is deduced from that of elements in sequence
            - can also be explicitly specified at creation time
        ```
        >>> import numpy as np
        >>> a = np.array([2, 3, 4])
        >>> a
        array([2, 3, 4])
        >>> a.dtype
        dtype('int64')
        >>> b = np.array([1.2, 3.5, 5.1])
        >>> b.dtype
        dtype('float64')
        
        >>> c = np.array([[1, 2], [3, 4]], dtype=complex)
        >>> c
        array([[ 1.+0.j, 2.+0.j],
        [ 3.+0.j, 4.+0.j]])
        ```
        - transform sequence of sequence into 2D arrays, sequence of sequence of sequence into 3D arrays, and so on
    - `numpy.zeros`, `numpy.ones`, `numpy.empty`
        - create arrays with initial placeholder content, or 
        - initial content is random and depends on the state of the memory
        - `dtype` by default is `float64`
        ```
        >>> np.zeros((3, 4))
        array([[ 0., 0., 0., 0.],
        [ 0., 0., 0., 0.],
        [ 0., 0., 0., 0.]])
        >>> np.ones((2, 3, 4), dtype=np.int16) # dtype can also be specified
        array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
        [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
        >>> np.empty((2, 3)) # uninitialized, output may vary
        array([[ 3.73603959e-262, 6.02658058e-154, 6.55490914e-260],
        [ 5.30498948e-313, 3.14673309e-307, 1.00000000e+000]])
        ```
    - `numpy.zeros_like`, `numpy.ones_like`, `numpy.empty_like`
    - `numpy.arange`, `numpy.linspace`
        - analogous to range that returns arrays instead of lists
        ```
        >>> np.arange(10, 30, 5)
        array([10, 15, 20, 25])
        >>> np.arange(0, 2, 0.3) # it accepts float arguments
        array([ 0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
        ```
        - use `numpy.linspace` with floating point arguments (floating point steps will generate inaccurate results due to finite floating point precision)
        ```
        >>> from numpy import pi
        >>> np.linspace(0, 2, 9) # 9 numbers from 0 to 2
        array([ 0. , 0.25, 0.5 , 0.75, 1. , 1.25, 1.5 , 1.75, 2. ])
        >>> x = np.linspace( 0, 2*pi, 100 ) # useful to evaluate function at lots of points
        >>> f = np.sin(x)
        ```
    - `numpy.random.RandomState.rand`, `numpy.random.RandomState.randn` 
    - `fromfunction`, `fromfile`
- Printing arrays
    - schema
        - the last axis is printed from left to right
        - the second-to-last is printed from top to bottom
        - the rest are also printed from top to bottom, with each slice separated from the next by an empty line
    ```
    >>> a = np.arange(6) # 1d array
    >>> print(a)
    [0 1 2 3 4 5]
    >>>
    >>> b = np.arange(12).reshape(4,3) # 2d array
    >>> print(b)
    [[ 0 1 2]
     [ 3 4 5]
     [ 6 7 8]
     [ 9 10 11]]
    >>>
    >>> c = np.arange(24).reshape(2,3,4) # 3d array
    >>> print(c)
    [[[ 0 1 2 3]
      [ 4 5 6 7]
      [ 8 9 10 11]]
    [[12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]]]
    ```
    - automatically skips the central part of the big arrays and only prints the corners
        - use `numpy.set_printoptions` to override this rule
    ```
    >>> np.set_printoptions(threshold=sys.maxsize) # sys module should be imported
    ```
- Basic operations


### 🌱 Introduction
