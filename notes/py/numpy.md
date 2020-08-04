# `numpy` Tutorial Notes

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
    - From regular Python list or tuple
        - using `array` function
        - type of resulting array is deduced from that of elements in sequence
        ```
        >>> import numpy as np
        >>> a = np.array([2,3,4])
        >>> a
        array([2, 3, 4])
        >>> a.dtype
        dtype('int64')
        >>> b = np.array([1.2, 3.5, 5.1])
        >>> b.dtype
        dtype('float64')
        ```

### 🌱 Introduction
