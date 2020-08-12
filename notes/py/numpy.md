# `NumPy` Tutorial Notes

- Notes of reading:
    - [`Numpy Tutorial - Tutorialspoint`](https://www.tutorialspoint.com/numpy/index.htm)
    - [`NumPy ver. Stable Manual`](https://numpy.org/doc/stable/index.html)

### 🌱 `ndarray` Object

- `ndarray` describes the collection of items of the *same type*
    - Items in the collection can be accessed using a zero-based index
    - Every item in an ndarray takes the same size of block in the memory
    - Each element in `ndarray` is an object of data-type object (called `dtype`)
    - Any item extracted from `ndarray` object (by *slicing*) is represented by a `Python` object of one of array scalar types
![](https://www.tutorialspoint.com/numpy/images/ndarray.jpg)
- [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html?highlight=numpy%20array#numpy.array)
    - Signature
    ```
    numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0)
    ```
    - Parameters
        - `object`: `array_like`. Any object exposing the array interface, whose `__array__` method returns an array, or any (nested) sequence. 
        - `dtype`: `data-type`, *optional*. Desired data type of array, if not given then use min type to hold the data. 
        - `copy`: `bool`, *optional*. By default (`True`), the object is copied. 
        - `order`: `{'K', 'A', 'C', 'F'}`, *optional*. Specify the memory layout of the array. If object is not an array, the newly created array will be in C order (row major) unless `'F'` is specified, in which case it will be in Fortran order (column major). If object is an array, the following holds: 
            - `'K'`: unchanged if `copy=False`, else F & C order preserved, otherwise most similar order
            - `'A'`: unchanged if `copy=False`, else F order if input is F and not C, otherwise C order
            - `'C'`: C order
            - `'F'`: F order
        - `subok`: `bool`, *optional*. If `True`, then sub-classes will be passed-through; otherwise, the returned array will be forced to be a base-class array (default). 
        - `ndimin`: `int`, *optional*. Specifies minimum dimensions of resultant array. Ones will be pre-pended to the shape as needed to meet this requirement. 
    - Returns: 
        - `out`: `ndarray`. An array object satisfying the specified requirements. 

### 🌱 Data Types

- Scalar Data Types
    - `numpy.bool_`: Boolean (`True` or `False`) stored as a byte
    - `numpy.int_`: Default integer type (same as `C` long; normally either `numpy.int64` or `numpy.int32`)
    - `numpy.intc`: Identical to `C` `int` (normally `numpy.int32` or `numpy.int64`)
    - `numpy.intp`: Integer used for indexing (same as `C` `ssize_t`; normally either `numpy.int32` or `numpy.int64`)
    - `numpy.int8`: Byte (-128 to 127)
    - `numpy.int16`: Integer (-32768 to 32767)
    - `numpy.int32`: Integer (-2147483648 to 2147483647)
    - `numpy.int64`: Integer (-9223372036854775808 to 9223372036854775807)
    - `numpy.uint8`: Unsigned integer (0 to 255)
    - `numpy.uint16`: Unsigned integer (0 to 65535)
    - `numpy.uint32`: Unsigned integer (0 to 4294967295)
    - `numpy.uint64`: Unsigned integer (0 to 18446744073709551615)
    - `numpy.float_`: Shorthand for `numpy.float64`
    - `numpy.float16`: Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    - `numpy.float32`: Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    - `numpy.float64`: Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    - `numpy.complex_`: Shorthand for `numpy.complex128`
    - `numpy.complex64`: Complex number, represented by two 32-bit floats (real and imaginary components)
    - `numpy.complex128`: Complex number, represented by two 64-bit floats (real and imaginary components)
- Each built-in data type has a character code that uniquely identifies it
    - `b`: boolean
    - `i`: (signed) integer
    - `u`: unsigned integer
    - `f`: floating-point
    - `c`: complex-floating point
    - `m`: timedelta
    - `M`: datetime
    - `O`: (Python) objects
    - `S`, `a`: (byte-)string
    - `U`: Unicode
    - `V`: raw data (void)
- Data Type Objects (`dtype`)
    - A data type object describes interpretation of fixed block of memory corresponding to an array, depending on the following aspects
        - Type of data (`int`, `float` or `Python` *object*)
        - Size of data
        - Byte order (little-endian or big-endian)
        - In case of *structured type*, the names of *fields*, data type of each field and part of the memory block taken by each field
        - If data type is a *subarray*, its shape and data type
    - Byte order is decided by prefixing `<` (little-endian) or `>` (big-endian) to data type
    - `dtype` object construction
        - constructor signature
        ```
        numpy.dtype(object, align, copy)
        ```
        - parameters:
            - `object`: To be converted to data type object
            - `align`: If `True`, adds padding to the field to make it similar to `C` `struct`
            - `copy`: Makes a new copy of `dtype` object. If `False`, the result is reference to builtin data type object
    ```
    >>> # using array-scalar type
    >>> import numpy as np
    >>> dt = np.dtype(np.int32)
    >>> dt
    int32
    
    >>> # int8, int16, int32, int64 can be replaced by equivalent string 'i1', 'i2','i4', etc.
    >>> dt = np.dtype('i4')
    >>> dt
    int32
    
    >>> # using endian notation
    >>> dt = np.dtype('>i4')
    >>> dt
    >i4
    
    >>> # first create structured data type
    >>> dt = np.dtype([('age', np.int8)])
    >>> dt
    [('age', 'i1')]
    
    >>> # now apply it to ndarray object
    >>> a = np.array([(10, ), (20, ), (30, )], dtype=dt)
    >>> a
    [(10,) (20,) (30,)]
    
    >>> # file name can be used to access content of age column
    >>> a['age']
    [10 20 30]
    
    >>> # The following examples define a structured data type called student 
    >>> # with a string field 'name', an integer field 'age' and a float field 'marks'
    >>> # This dtype is applied to ndarray object
    >>> student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
    >>> student
    [('name', 'S20'), ('age', 'i1'), ('marks', '<f4')])
    
    >>> a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
    >>> a
    [('abc', 21, 50.0), ('xyz', 18, 75.0)]
    ```

### 🌱 Array Attributes

- `ndarray.shape`
    - returns a tuple consisting of array dimensions
    - It can also be used to resize the array
```
>>> import numpy as np
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> a.shape
(2, 3)

>>> # this resizes the ndarray
>>> a.shape = (3, 2)
>>> a
[[1 2]
 [3 4]
 [5 6]]

>>> # reshape function
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> b = a.reshape(3, 2)
>>> b
[[1 2]
 [3 4]
 [5 6]]
```
- `ndarray.ndim`
    - returns the number of array dimensions
```
>>> import numpy as np
>>> a = np.arange(24)
>>> a
[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
>>> a.ndim
1

>>> b = a.reshape(2, 4, 3)
>>> b
[[[ 0, 1, 2]
  [ 3, 4, 5]
  [ 6, 7, 8]
  [ 9, 10, 11]]
  
 [[12, 13, 14]
  [15, 16, 17]
  [18, 19, 20]
  [21, 22, 23]]]
```
- `ndarray.itemsize`
    - returns the length of each element of array in bytes
```
>>> # dtype of array is int8 (1 byte)
>>> import numpy as np
>>> x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
>>> x.itemsize
1

>>> # dtype of array is now float32 (4 bytes)
>>> x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
>>> x.itemsize
4
```
- `ndarray.flags`
    - `C_CONTIGUOUS (C)`: The data is in a single, `C`-style row-major contiguous segment
    - `F_CONTIGUOUS (F)`: The data is in a single, `Fortran`-style column-major contiguous segment
    - `OWNDATA (O)`: The array owns the memory it uses or borrows it from another object
    - `WRITEABLE (W)`: The data area can be written to. Setting this to `False` locks the data, making it read-only
    - `ALIGNED (A)`: The data and all elements are aligned appropriately for the hardware
    - `UPDATEIFCOPY (U)`: This array is a copy of some other array. When this array is deallocated, the base array will be updated with the contents of this array
```
>>> import numpy as np
>>> x = np.array([1, 2, 3, 4, 5])
>>> x.flags
C_CONTIGUOUS : True
F_CONTIGUOUS : True
OWNDATA : True
WRITEABLE : True
ALIGNED : True
WRITEBACKIFCOPY : False
UPDATEIFCOPY : False
```

### 🌱 Array Creation Routines

- [`numpy.empty`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html#numpy.empty)
    - Creates an *uninitialized* array of specified `shape` and `dtype`
    - Signature
    ```
    numpy.empty(shape, dtype=float, order='C')
    ```
    - Parameters
        - `shape`: `int` or `Tuple[int]`. shape of an empty array
        - `dtype`: `data-type`, *optional*. Desired output data-type for the array, e.g, `numpy.int8`. Default is `numpy.float64`.
        - `order`: `{'C', 'F'}`, *optional*. Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory. 
    - Returns: 
        - `out`: `ndarray`. Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to `None`. 
    ```
    >>> import numpy as np
    >>> x = np.empty([3, 2], dtype=int)
    >>> x
    [[22649312   1701344351]
     [1818321759 1885959276]
     [16779776   156368896]]
    ```
- [`numpy.empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html#numpy.empty_like)
    - Return a new *uninitialized* array with the same shape and type as a given array. 
    - Signature: 
    ```
     numpy.empty_like(a, dtype=None, order='K', subok=True, shape=None)
    ```
    - Parameters: 
        - `a`: `array_like`. The shape and data-type of `a` define these same attributes of the returned array.
        - `dtype`: `data-type`, *optional*. Overrides the data type of the result.
        - `order`: `{'C', 'F', 'A', 'K'}`, *optional*. Overrides the memory layout of the result. `‘C’` means C-order, `‘F’` means F-order, `‘A’` means `‘F’` if prototype is Fortran contiguous, `‘C’` otherwise. `‘K’` means match the layout of prototype as closely as possible.
        - `subok`: `bool`, *optional*. If `True`, then the newly created array will use the sub-class type of `a`, otherwise it will be a base-class array. Defaults to `True`.
        - `shape`: `int` or `Sequence[int]`, *optional*. Overrides the shape of the result. If `order='K'` and the number of dimensions is unchanged, will try to keep order, otherwise, `order='C'` is implied. 
    - Returns: 
        - `out`: `ndarray`.  Array of uninitialized (arbitrary) data with the same shape and type as prototype.
    ```
    >>> a = ([1, 2, 3], [4, 5, 6])                         # a is array-like
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    # uninitialized
           [          0,           0, -1073741821]])

    >>> a = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000], # uninitialized
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    ```
- [`numpy.eye`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html#numpy.eye)
    - Return a 2-D array with ones on the diagonal and zeros elsewhere.
    - Signature: 
    ```
    numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C')
    ```
    - Parameters: 
        - `N`: `int`. Number of rows in the output.
        - `M`: `int`, *optional*. Number of columns in the output. If `None`, defaults to `N`. 
        - `k`: `int`, *optional*. Index of the diagonal: `0` (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
        - `dtype`: `data-type`, *optional*. Data-type of the returned array.
        - `order`: `{'C', 'F'}`, *optional*. Whether the output should be stored in row-major (C-style) or column-major (Fortran-style) order in memory.
    - Returns: 
        - `I`: `ndarray` of shape `(N, M)`. An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.
    ```
    >>> np.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])

    >>> np.eye(3, k=1)
    array([[0.,  1.,  0.],
           [0.,  0.,  1.],
           [0.,  0.,  0.]])
    ```
- [`numpy.identity`](https://numpy.org/doc/stable/reference/generated/numpy.identity.html#numpy.identity)
    - Return the identity array. The identity array is a square array with ones on the main diagonal. 
    - Signature: 
    ```
    numpy.identity(n, dtype=None)
    ```
    - Parameters: 
        - `n`: `int`. Number of rows (and columns) in `n x n` output.
        - `dtype`: `data-type`, *optional*. Data-type of the output. Defaults to `float`.
    - Returns: 
        - `out`: `ndarray`. `n x n` array with its main diagonal set to one, and all other elements `0`.
    ```
    >>> np.identity(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])
    ```
- [`numpy.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html#numpy.zeros)
    - Return a new array of given shape and type, filled with zeros.
    - Signature
    ```
    numpy.zeros(shape, dtype=float, order='C')
    ```
    - Parameters
        - `shape`: `int` or `Tuple[int]`. Shape of an empty array, e.g., `(2, 3)` or `2`. 
        - `dtype`: `data-type`, *optional*. Desired output data type. e.g, `numpy.int8`. Default is `numpy.float64`.
        - `order`: `{'C', 'F'}`, *optional*. Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
    - Returns: 
        - `out`: `ndarray`. Array of zeros with the given shape, dtype, and order.
    ```
    >>> import numpy as np
    >>> x = np.zeros(5)
    >>> x
    [ 0. 0. 0. 0. 0.]

    >>> x = np.zeros((5, ), dtype=np.int)
    >>> x
    [0 0 0 0 0]

    >>> # custom type
    >>> x = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
    >>> x
    [[(0, 0) (0, 0)]
     [(0, 0) (0, 0)]]
    ```
- [`numpy.zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html)
    - Return an array of zeros with the same shape and type as a given array.
    - Signature: 
    ```
    numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
    ```
    - Parameters: 
        - `a`: `array_like`. The shape and data-type of `a` define these same attributes of the returned array.
        - `dtype`: `data-type`, *optional*. Overrides the data type of the result.
        - `order`: `{'C', 'F', 'A', 'K'}`, *optional*. Overrides the memory layout of the result. 
        - `subok`: `bool`, *optional*. If `True`, then the newly created array will use the sub-class type of `a`, otherwise it will be a base-class array. Defaults to `True`.
        - `shape`: `int` or `Sequence[int]`, *optional*. Overrides the shape of the result. If `order='K'` and the number of dimensions is unchanged, will try to keep order, otherwise, `order='C'` is implied.
    - Returns: 
        - `out`: `ndarray`. Array of zeros with the same shape and type as `a`. 
    ```
    >>> x = np.arange(6).reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
           
    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])

    >>> np.zeros_like(y)
    array([0.,  0.,  0.])
    ```
- [`numpy.ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html#numpy.ones)
    - Return a new array of given shape and type, filled with ones.
    - Signature
    ```
    numpy.ones(shape, dtype=float, order='C')
    ```
    - Parameters
        - `shape`: `int` or `Tuple[int]`. Shape of an empty array, e.g., `(2, 3)` or `2`. 
        - `dtype`: `data-type`, *optional*. Desired output data type. e.g, `numpy.int8`. Default is `numpy.float64`.
        - `order`: `{'C', 'F'}`, *optional*. Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
    - Returns: 
        - `out`: `ndarray`. Array of ones with the given shape, dtype, and order.
    ```
    >>> import numpy as np
    >>> x = np.ones(5)
    >>> x
    [ 1. 1. 1. 1. 1.]

    >>> x = np.ones([2, 2], dtype=int)
    >>> x
    [[1 1]
     [1 1]]
    ```
- [`numpy.ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html)
    - Return an array of zeros with the same shape and type as a given array.
    - Signature: 
    ```
    numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
    ```
    - Parameters: 
        - `a`: `array_like`. The shape and data-type of `a` define these same attributes of the returned array.
        - `dtype`: `data-type`, *optional*. Overrides the data type of the result.
        - `order`: `{'C', 'F', 'A', 'K'}`, *optional*. Overrides the memory layout of the result. 
        - `subok`: `bool`, *optional*. If `True`, then the newly created array will use the sub-class type of `a`, otherwise it will be a base-class array. Defaults to `True`.
        - `shape`: `int` or `Sequence[int]`, *optional*. Overrides the shape of the result. If `order='K'` and the number of dimensions is unchanged, will try to keep order, otherwise, `order='C'` is implied.
    - Returns: 
        - `out`: `ndarray`. Array of zeros with the same shape and type as `a`. 
    ```
    >>> x = np.arange(6).reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> np.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])
           
    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])

    >>> np.ones_like(y)
    array([1.,  1.,  1.])
    ```
- [`numpy.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html#numpy.full)
    - RReturn a new array of given shape and type, filled with `fill_value`. 
    - Signature
    ```
    numpy.full(shape, fill_value, dtype=None, order='C')[source]
    ```
    - Parameters
        - `shape`: `int` or `Tuple[int]`. Shape of an empty array, e.g., `(2, 3)` or `2`. 
        - `fill_value`: `scalar` or `array_like`, Fill value.
        - `dtype`: `data-type`, *optional*. The desired data-type for the array. The default, `None`, means `np.array(fill_value).dtype`.
        - `order`: `{'C', 'F'}`, *optional*. Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
    - Returns: 
        - `out`: `ndarray`. Array of `fill_value` with the given shape, dtype, and order.
    ```
    >>> np.full((2, 2), np.inf)
    array([[inf, inf],
           [inf, inf]])

    >>> np.full((2, 2), 10)
    array([[10, 10],
           [10, 10]])
    
    >>> np.full((2, 2), [1, 2])
    array([[1, 2],
           [1, 2]])
    ```
- [`numpy.full_like`](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy.full_like)
    - Return a full array with the same shape and type as a given array. 
    - Signature: 
    ```
    numpy.full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None)
    ```
    - Parameters: 
        - `a`: `array_like`. The shape and data-type of `a` define these same attributes of the returned array.
        - `fill_value`: `scalar`. Fill value.
        - `dtype`: `data-type`, *optional*. Overrides the data type of the result.
        - `order`: `{'C', 'F', 'A', 'K'}`, *optional*. Overrides the memory layout of the result. 
        - `subok`: `bool`, *optional*. If `True`, then the newly created array will use the sub-class type of `a`, otherwise it will be a base-class array. Defaults to `True`.
        - `shape`: `int` or `Sequence[int]`, *optional*. Overrides the shape of the result. If `order='K'` and the number of dimensions is unchanged, will try to keep order, otherwise, `order='C'` is implied.
    - Returns: 
        - `out`: `ndarray`. Array of `fill_value` with the same shape and type as `a`. 
    ```
    >>> x = np.arange(6, dtype=int)

    >>> np.full_like(x, 1)
    array([1, 1, 1, 1, 1, 1])

    >>> np.full_like(x, 0.1)
    array([0, 0, 0, 0, 0, 0])

    >>> np.full_like(x, 0.1, dtype=np.double)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    >>> np.full_like(x, np.nan, dtype=np.double)
    array([nan, nan, nan, nan, nan, nan])

    >>> y = np.arange(6, dtype=np.double)

    >>> np.full_like(y, 0.1)
    array([0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
    ```

### 🌱 Array from Existing Data

- `numpy.array`
- [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html?highlight=numpy%20asarray#numpy.asarray)
    - Convert the input to an array. Similar to `numpy.array` except for the fact that it has fewer parameters. This routine is useful for converting `Python` sequence into `ndarray`
    - Signature
    ```
    numpy.asarray(a, dtype=None, order=None)
    ```
    - Parameters
        - `a`: `array_like`. Input data, in any form that can be converted to an array. This includes `List`, `List[Tuple]`, `Tuple`, `Tuple[Tuples]`, `Tuple[List[np.ndarray]]`.
        - `dtype`: `data-type`, *optional*. By default, the data-type is inferred from the input data.
        - `order`: `{'C', 'F'}`, *optional*. Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.
    - Returns: 
        - `out`: `ndarray`. Array interpretation of `a`. **NO** copy is performed if the input is already an `ndarray` with matching dtype and order. If `a` is a subclass of ndarray, a base class `ndarray` is returned. 

    ```
    >>> # convert list to ndarray
    >>> import numpy as np
    >>> x = [1, 2, 3]
    >>> np.asarray(x)
    [1 2 3]

    >>> # dtype is set
    >>> np.asarray(x, dtype=float)
    [ 1. 2. 3.]

    >>> # ndarray from tuple
    >>> x = (1, 2, 3)
    >>> np.asarray(x)
    [1 2 3]

    >>> # ndarray from list of tuples
    >>> x = [(1, 2, 3), (4, 5)]
    >>> np.asarray(x)
    [(1, 2, 3) (4, 5)]
    ```
- `numpy.frombuffer`
    - This function interprets a buffer as 1D array. Any object that exposes the buffer interface is used as parameter to return an `ndarray`
    - signature
    ```
    numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)
    ```
    - parameters
        - `buffer`: Any object that exposes buffer interface
        - `dtype`: Data type of returned ndarray. Defaults to `float`
        - `count`: The number of items to read, default `-1` means all data
        - `offset`: The starting position to read from. Default is `0`
```
>>> import numpy as np
>>> s = 'Hello World'
>>> a = np.frombuffer(s, dtype='S1')
>>> a
['H' 'e' 'l' 'l' 'o' ' ' 'W' 'o' 'r' 'l' 'd']
```
- `numpy.fromiter`
    - This function builds an ndarray object from any iterable object. A new 1D array is returned by this function
- signature
    ```
    numpy.fromiter(buffer, dtype=float, count=-1)
    ```
    - parameters
        - `buffer`: Any iterable object
        - `dtype`: Data type of returned ndarray. Defaults to `float`
        - `count`: The number of items to be read from iterator. Default is `-1` which means all data to be read
```
>>> # create list object using range function
>>> import numpy as np
>>> lst = range(5)
>>> lst
[0, 1, 2, 3, 4]

>>> # use iterator to create ndarray
>>> it = iter(list)
>>> np.fromiter(it, dtype=float)
[0. 1. 2. 3. 4.]
```

### 🌱 Array from Numerical Ranges

- [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html?highlight=numpy%20arange#numpy.arange)
    - Return evenly spaced values within a given interval.
    - Values are generated within the half-open interval `[start, stop)`. For integer arguments the function is equivalent to the `Python` built-in range function, but returns an `ndarray` rather than a generator.
    - When using a non-integer step, such as 0.1, the results will often **NOT** be consistent. It is better to use `numpy.linspace` for these cases.   
    - Signature
    ```
    numpy.arange(start=0, stop, step=1, dtype=None)
    ```
    - Parameters
        - `start`: `number`, *optional*. Start of interval. The interval includes this value. The default start value is `0`.
        - `stop`: `number`. End of interval. The interval does **NOT** include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
        - `step`: `number`, *optional*. Spacing between values. For any output `out`, this is the distance between two adjacent values, `out[i + 1] - out[i]`. The default step size is `1`. If step is specified as a position argument, `start` *must also be given*. 
        - `dtype`: `data-type`. The type of the output array. If dtype is not given, infer the data type from the other input arguments.
    - Returns: 
        - `argange`: `ndarray`. Array of evenly spaced values. For floating point arguments, the length of the result is `ceil((stop - start) / step)`. Because of floating point overflow, this rule may result in the last element of out being greater than stop.
    ```
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> x
    [0 1 2 3 4]

    >>> # dtype set
    >>> x = np.arange(5, dtype=float)
    >>> x
    [0. 1. 2. 3. 4.]

    >>> # start and stop parameters set
    >>> x = np.arange(10, 20, 2)
    >>> x
    [10 12 14 16 18]
    ```
- [`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
    - This function is similar to `numpy.arange` function. In this function, instead of step size, the *number* of evenly spaced values between the interval `[start, stop]`
    - Signature
    ```
    numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    ```
    - Parameters
        - `start`: `array_like`. The starting value of the sequence
        - `stop`: `array_like`. The end value of the sequence, *included* in the sequence if `endpoint=True`
        - `num`: `int`, *optional*. The number of evenly spaced samples to be generated. Default is `50`
        - `endpoint`: `bool`, *optional*. `True` by default, hence the stop value is included in the sequence. If `False`, it is **NOT** included
        - `retstep`: `bool`, *optional*. If `True`, returns a tuple of (samples, step between the consecutive numbers)
        - `dtype`: `data-type`, *optional*. Data type of output `ndarray`
        - `axis`: `int`, *optional*. The axis in the result to store the samples. Relevant only if `start` or `stop` are `array-like`. By default (`0`), the samples will be along a new axis inserted at the beginning. Use `-1` to get an axis at the end.
    - Returns: 
        - `samples`: `ndarray`. There are num equally spaced samples in the closed interval `[start, stop]` or the half-open interval `[start, stop)` (depending on whether `endpoint` is `True` or `False`).
        - `step`: `float`, *optional*. Only returned if `retstep=True`. Size of spacing between samples.
    ```
    >>> import numpy as np
    >>> x = np.linspace(10, 20, 5)
    >>> x
    [10. 12.5 15. 17.5 20.]

    >>> # endpoint set to false
    >>> x = np.linspace(10, 20, 5, endpoint=False)
    >>> x
    [10. 12. 14. 16. 18.]

    >>> # find retstep value
    >>> x = np.linspace(1, 2, 5, retstep=True)
    >>> x
    (array([ 1. , 1.25, 1.5 , 1.75, 2. ]), 0.25)
    ```
- [`numpy.logspace`](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html#numpy.logspace)
    - Return numbers spaced evenly on a log scale. In linear space, the sequence starts at `base ** start` and ends with `base ** stop`. 
    - Signature
    ```
    numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
    ```
    - Parameters
        - `start`: `array_like`. `base ** start` is the starting value of the sequence.
        - `stop`: `array_like`. `base ** stop` is the final value of the sequence, unless `endpoint=False`. In that case, `num + 1` values are spaced over the interval in log-space, of which all but the last (a sequence of length num) are returned.
        - `num`: `int`, *optional*. Number of samples to generate. Default is `50`.
        - `endpoint`: `bool`, *optional*. If `True`, `stop` is the last sample. Otherwise, it is **NOT** included. Default is `True`.
        - `base`: `float`, *optional*. The base of the log space. The step size between the elements in `ln(samples) / ln(base)` (or `log_base(samples)`) is uniform. Default is `10.0`.
        - `dtype`: `data-type`. The type of the output array. If dtype is not given, infer the data type from the other input arguments.
        - `axis`: `int`, *optional*. The axis in the result to store the samples. Relevant only if start or stop are array-like. By default (`0`), the samples will be along a new axis inserted at the beginning. Use `-1` to get an axis at the end.
    - Returns: 
        - `samples`: `ndarray`. `num` samples equally spaced on a log scale.
    ```
    >>> # default base is 10
    >>> import numpy as np
    >>> a = np.logspace(1.0, 2.0, num=10)
    >>> a
    [ 10.         12.91549665 16.68100537 21.5443469  27.82559402
      35.93813664 46.41588834 59.94842503 77.42636827 100.        ]
      
    >>> # set base of log space to 2
    >>> a = np.logspace(1, 10, num=10, base=2)
    >>> a
    [ 2. 4. 8. 16. 32. 64. 128. 256. 512. 1024.]
    ```
- [`numpy.geomspace`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html#numpy.geomspace)  
    - Return numbers spaced evenly on a log scale (a geometric progression). This is similar to `logspace`, but with endpoints specified *directly*. Each output sample is a constant multiple of the previous.
    - Signature: 
    ```
    numpy.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)
    ```
    - Parameters
        - `start`: `array_like`. The starting value of the sequence.
        - `stop`: `array_like`. The final value of the sequence, unless `endpoint=False`. In that case, `num + 1` values are spaced over the interval in log-space, of which all but the last (a sequence of length num) are returned.
        - `num`: `int`, *optional*. Number of samples to generate. Default is `50`.
        - `endpoint`: `bool`, *optional*. If `True`, `stop` is the last sample. Otherwise, it is **NOT** included. Default is `True`.
        - `base`: `float`, *optional*. The base of the log space. The step size between the elements in `ln(samples) / ln(base)` (or `log_base(samples)`) is uniform. Default is `10.0`.
        - `dtype`: `data-type`. The type of the output array. If dtype is not given, infer the data type from the other input arguments.
        - `axis`: `int`, *optional*. The axis in the result to store the samples. Relevant only if start or stop are array-like. By default (`0`), the samples will be along a new axis inserted at the beginning. Use `-1` to get an axis at the end.
    - Returns: 
        - `samples`: `ndarray`. `num` samples equally spaced on a log scale.
- [`numpy.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid)
    - Return coordinate matrices from coordinate vectors. Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields over N-D grids, given one-dimensional coordinate arrays `x1, x2, ..., xn`.
    - Signature: 
    ```
    numpy.meshgrid(*xi, copy=True, sparse=False, indexing='xy')
    ```
    - Parameters: 
        - `x1, x2, ..., xn`: `array_like`. 1-D arrays representing the coordinates of a grid. 
        - `indexing`: `{'xy', 'ij'}`, *optional*. Cartesian (`'xy'`, default) or matrix (`'ij'`) indexing of output. 
        - `sparse`: `bool`, *optional*. If `True` a sparse grid is returned in order to conserve memory. Default is `False`. 
        - `copy`: `bool`, *optional*. If `False`, a *view* into the original arrays are returned in order to conserve memory.  Default is `True`. Please note that `sparse=False`, `copy=False` will likely return *non-contiguous* arrays.  Furthermore, more than one element of a broadcast array may refer to a single memory location. If you need to write to the arrays, make copies first. 
    - Returns: 
        - `X1, X2, ..., XN`: `ndarray`. For vectors `x1, x2, ..., xn` with lengths `Ni = len(xi)`, return `(N1, N2, N3, ..., Nn)` shaped arrays if `indexing='ij'` or `(N2, N1, N3, ..., Nn)` shaped arrays if `indexing='xy'` with the elements of `xi` repeated to fill the matrix along the first dimension for `x1`, the second for `x2` and so on. 
    - Notes: 
        - This function supports both indexing conventions through the indexing keyword argument. Giving the string `'ij'` returns a meshgrid with matrix indexing, while `'xy'` returns a meshgrid with Cartesian indexing. In the 2-D case with inputs of length `M` and `N`, the outputs are of shape `(N, M)` for `'xy'` indexing and `(M, N)` for `'ij'` indexing. In the 3-D case with inputs of length `M`, `N` and `P`, outputs are of shape `(N, M, P)` for `'xy'`indexing and `(M, N, P)` for `'ij'` indexing.
    - Examples: 
    ```
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    
    >>> xv, yv = np.meshgrid(x, y)

    >>> xv
    array([[0. , 0.5, 1. ],
           [0. , 0.5, 1. ]])

    >>> yv
    array([[0.,  0.,  0.],
           [1.,  1.,  1.]])

    >>> xv, yv = np.meshgrid(x, y, sparse=True)  # make sparse output arrays

    >>> xv
    array([[0. ,  0.5,  1. ]])

    >>> yv
    array([[0.],
           [1.]])
    ```
    
### 🌱 Building Matrices

- [`numpy.diag`](https://numpy.org/doc/stable/reference/generated/numpy.diag.html#numpy.diag)
    - Extract a diagonal or construct a diagonal array. Whether it returns a copy or a view depends on what version of numpy you are using. 
    - Signature: 
    ```
    numpy.diag(v, k=0)
    ```
    - Parameters: 
        - `v`: `array_like`. If `v` is a 2-D array, return a *copy* of its `k`-th diagonal. If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th diagonal. 
        - `k`: `int`, *optional*. Diagonal in question. The default is `0`. Use `k > 0` for diagonals above the main diagonal, and `k < 0` for diagonals below the main diagonal. 
    - Returns: 
        - `out`: `ndarray`. The extracted diagonal or constructed diagonal array. 
    ```
    >>> x = np.arange(9).reshape(3, 3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])

    >>> np.diag(x, k=1)
    array([1, 5])

    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    ```
- [`numpy.diagflat`](https://numpy.org/doc/stable/reference/generated/numpy.diagflat.html#numpy.diagflat)
    - Create a 2-D array with the flattened input as a diagonal. 
    - Signature: 
    ```
    numpy.diagflat(v, k=0)
    ```
    - Parameters: 
        - `v`: `array_like`. Input data, which is flattened and set as the k-th diagonal of the output. 
        - `k`: `int`, *optional*. Diagonal to set; `0`, the default, corresponds to the "main" diagonal, a positive (negative) k giving the number of the diagonal above (below) the main. 
    - Returns: 
        - `out`: `ndarray`. The 2-D output array. 
    ```
    >>> np.diagflat([[1, 2], [3, 4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat([1, 2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])
    ```
- [`numpy.tri`](https://numpy.org/doc/stable/reference/generated/numpy.tri.html#numpy.tri)
    - An array with ones at and below the given diagonal and zeros elsewhere. 
    - Signature: 
    ```
    numpy.tri(N, M=None, k=0, dtype=<class 'float'>)
    ```
    - Parameters: 
        - `N`: `int`. Number of rows in the array. 
        - `M`: `int`, *optional*. Number of columns in the array. By default, `M` is taken equal to `N`. 
        - `k`: `int`, *optional*. The sub-diagonal at and below which the array is filled. `k = 0` is the main diagonal, while `k < 0` is below it, and `k > 0` is above. The default is `0`. 
        - `dtype``data-type`, *optional*. Data type of the returned array. The default is `float`. 
    - Returns: 
        - `tri`: `ndarray` of shape `(N, M)`. Array with its lower triangle filled with ones and zero elsewhere; in other words `T[i,j] == 1` for `j <= i + k`, `0` otherwise. 
    ```
    >>> np.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])
    
    >>> np.tri(3, 5, -1)
    array([[0.,  0.,  0.,  0.,  0.],
           [1.,  0.,  0.,  0.,  0.],
           [1.,  1.,  0.,  0.,  0.]])
    ```
- [`numpy.tril`](https://numpy.org/doc/stable/reference/generated/numpy.tril.html#numpy.tril)
    - Lower triangle of an array. Return a *copy* of an array with elements *above* the `k`-th diagonal zeroed. 
    - Signature: 
    ```
    numpy.tril(m, k=0)
    ```
    - Parameters: 
        - `m`: `array_like`, shape `(M, N)`. Input array. 
        - `k`: `int`, *optional*. Diagonal above which to zero elements. `k = 0` (the default) is the main diagonal, `k < 0` is below it and `k > 0` is above. 
    - Returns: 
        - `tril`: `ndarray`, shape `(M, N)`. Lower triangle of `m`, of same shape and data-type as `m`. 
    ```
    >>> np.tril([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
    array([[ 0,  0,  0], 
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])
    ```
- [`numpy.triu`](https://numpy.org/doc/stable/reference/generated/numpy.triu.html#numpy.triu)
    - Upper triangle of an array. Return a *copy* of a matrix with the elements *below* the `k`-th diagonal zeroed. 
    - Signature: 
    ```
    numpy.triu(m, k=0)
    ```
    - Examples: 
    ```
    >>> np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])
    ```

### 🌱 Basic Indexing & Slicing

- Contents of `ndarray` object can be accessed and modified by *indexing* or *slicing* 
    - items in `ndarray` object follows *zero-based index*, can be indexed using `x[obj]` syntax 
    - three types of indexing methods
        - *field access* 
        - *basic slicing* 
        - *advanced indexing* 
- *Field Access* 
    - If the ndarray object is a *structured array*, the *fields* of the array can be accessed by indexing the array with strings, dictionary-like
    - Indexing `x['field-name']` returns a new `view` to the array, which is of the same shape as `x` (except when the field is a sub-array) but of data type `x.dtype['field-name']` and contains only the part of the data in the specified field
    ```
    >>> x = np.zeros((2, 2), dtype=[('a', np.int32), ('b', np.float64, (3, 3))])
    >>> x['a'].shape
    (2, 2)
    >>> x['a'].dtype
    dtype('int32')
    >>> x['b'].shape
    (2, 2, 3, 3)
    >>> x['b'].dtype
    dtype('float64')
    ```
- *Basic Slicing* 
    - Basic slicing occurs when `obj` is
        - a *slice object*, or 
        - an *integer*, or 
        - a *tuple* of *slice objects* and *integers*
        - *Ellipsis* `...` and `np.newaxis` objects can be interspersed with these as well
    - All arrays generated by *basic slicing* are always *views* of the original array
        - Care must be taken when extracting a small portion from a large array which becomes useless after the extraction, because the small portion extracted contains a reference to the large original array whose memory will not be released until all arrays derived from it are garbage-collected. In such cases an explicit `copy()` is recommended.
    - You may use slicing to set values in the array, but (unlike lists) you can **never** grow the array. The size of the value to be set in `x[obj] = value` must be (broadcastable) to the same shape as `x[obj]`
    - a *slice object* is constructed by giving `start`, `stop`, `step` parameters to the built-in `slice` function, or by `(start:stop:step)` syntax directly
    ```
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> s = slice(2, 7, 2)
    >>> a[s]
    [2 4 6]

    >>> a[2:7:2]
    [2 4 6]
    
    >>> a[5]
    5
    
    >>> a[2:]
    [2 3 4 5 6 7 8 9]
    
    >>> a[2:5]
    [2 3 4]
    ```
    - Note
        - Remember that a slicing tuple can always be constructed as `obj` and used in the `x[obj]` notation. Slice objects can be used in the construction in place of the `[start:stop:step]` notation. 
        - For example, `x[1:10:5,::-1]` can also be implemented as 
        ```
        obj = (slice(1, 10, 5), slice(None, None, -1))
        x[obj]
        ```
        - This can be useful for constructing generic code that works on arrays of arbitrary dimension.
    - If the number of objects in the selection tuple is less than `x.ndim`, then `:` is assumed for any subsequent dimensions
    ```
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
    >>> a
    [[1 2 3]
     [3 4 5]
     [4 5 6]]

    >>> # slice items starting from index
    >>> a[1:]
    [[3 4 5]
     [4 5 6]]
    ```
    - Slicing can also include *ellipsis* `...` to make a selection tuple of the same length as the `ndim` of an array. There may only be a *single* ellipsis present.
    ```
    >>> # array to begin with
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
    >>> a
    [[1 2 3]
     [3 4 5]
     [4 5 6]]
    
    >>> # this returns array of items in the second column
    >>> a[..., 1]
    [2 4 5]
    
    >>> # Now we will slice all items from the second row
    >>> a[1, ...]
    [3 4 5]
    
    >>> # Now we will slice all items from column 1 onwards
    >>> a[..., 1:]
    [[2 3]
     [4 5]
     [5 6]]
    ```
    - Each `numpy.newaxis` object in the selection tuple serves to expand the dimensions of the resulting selection by one unit-length dimension. The added dimension is the position of the newaxis object in the selection tuple
        - The `newaxis` object can be used in all slicing operations to create an axis of length one. `newaxis` is an alias for `None`, and `None` can be used in place of this with the same result
        ```
        >>> x = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        >>> x.shape
        (2, 3, 1)
        >>> x[:,np.newaxis,:,:].shape
        (2, 1, 3, 1)
        >>> x[:,None,:,:].shape
        (2, 1, 3, 1)
        ```

### 🌱 Advanced Indexing

- Advanced indexing is triggered when the selection object `obj` is:
    - a *non-tuple sequence object*, or 
    - an `ndarray` (whose `dtype` is `int` or `bool`), or
    - a *tuple* having one more *sequence object* or `ndarray` (whose `dtype` is `int` or `bool`)
- Warning
    - `x[(1, 2, 3), ]` is different from `x[(1, 2, 3)]`. The former triggers *advanced indexing*, while the latter one equals to `x[1, 2, 3]` and triggers *basic indexing*
    - Also recognize that `x[[1, 2, 3]]` will trigger *advanced indexing*, whereas due to the deprecated Numeric compatibility mentioned above, `x[[1, 2, slice(None)]]` will trigger *basic slicing* 
- Offical documents says that advanced indexing generates *copies*, however, as tested, advanced indexing can also be used to set values! WEIRD THING HERE. 
- two types of advanced indexing: 
    - *Integer Array Indexing* 
    - *Boolean Array Indexing* 
- Integer Array Indexing
    - Index is a tuple of `x.ndim` `ndarray`s, all in same shape (same as output shape) (or boardcastable shape). For the following
    ```
    res = x[a_1, ..., a_N]  # N == x.ndim
                            # M == res.ndim
                            # res.shape == a_1.shape == ... == a_k.shape
    ```
    - We have: every scalar element in `res` is selected from `x`, with the `k-th` coordinate given by `a_k`'s cooresponding element, i.e. 
    ```
    res[i_1, ..., i_M] == x[a_1[i_1, ..., i_M], a_2[i_1, ..., i_M], ..., a_N[i_1, ..., i_M]]
    ```
    - **Example 1**: From each row, a specific element should be selected. The row index is just `[0, 1, 2]` and the column index specifies the element to choose for the corresponding row, here `[0, 1, 0]`
    ```
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> x[[0, 1, 2], [0, 1, 0]]
    [1 4 5]
    ```
    - **Example 2**: from a `4 * 3` array the corner elements should be selected using advanced indexing. Thus all elements for which the column is one of `[0, 2]` and the row is one of `[0, 3]` need to be selected. To use advanced indexing one needs to select all elements explicitly.
    ```
    >>> x = np.array([[ 0,  1,  2], 
                      [ 3,  4,  5], 
                      [ 6,  7,  8], 
                      [ 9, 10, 11]])
    >>> rows = np.array([[0, 0], [3, 3]])
    >>> cols = np.array([[0, 2], [0, 2]])
    >>> x[rows, cols]
    [[ 0,  2],
     [ 9, 11]]
    ```
    - **Example 2.1**: we can make use of *broadcasting* to generate `rows`, `cols` from simpler `ndarray`s
    ```
    >>> x = np.array([[ 0,  1,  2], 
                      [ 3,  4,  5], 
                      [ 6,  7,  8], 
                      [ 9, 10, 11]])
    >>> rows = np.array([0, 3])
    >>> rows[:, np.newaxis]
    [[0],
     [3]]
    
    >>> cols = np.array([0, 2])
    >>> x[rows[:, np.newaxis], cols]
    [[ 0,  2],
     [ 9, 11]]
    ```
    - **Example 2.2**: use of `np.ix_` function
        - `numpy.ix_(*args)`: take `N` 1D arrays, return tuple of `N` arrays s.t. they can automatically broadcast into proper index array
    ```
    >>> x = np.array([[ 0,  1,  2], 
                      [ 3,  4,  5], 
                      [ 6,  7,  8], 
                      [ 9, 10, 11]])
    >>> rows = np.array([0, 3])
    >>> cols = np.array([0, 2])
    >>> idx = np.ix_(rows, cols)
    >>> idx
    (array([[0],
            [3]]),
            
     array([[0, 2]]))
    
    >>> x[idx]
    [[ 0,  2],
     [ 9, 11]]
    ```
- Combining Advanced And Basic Indexing
    - **Example 3**: Advanced and basic indexing can be combined by using one *slice* `:` or *ellipsis* `...` with an index array. The following example uses slice for row and advanced index for column. The result is the same when slice is used for both. But advanced index results in copy and may have different memory layout.
    ```
    >>> x = np.array([[ 0,  1,  2], 
                      [ 3,  4,  5], 
                      [ 6,  7,  8], 
                      [ 9, 10, 11]])
                      
    >>> x[1:4, 1:3]
    [[ 4  5]
     [ 7  8]
     [10 11]]
    
    >>> x[1:4, [1, 2]]
    [[ 4  5]
     [ 7  8]
     [10 11]]
    ```
- Boolean Array Indexing
    - Occurs when `obj` is array object of `bool` type, such as may be returned from comparison operators
    - Boolean array indexing `x[obj]` is practically identical to Integer array indexing `x[obj.nonzero()]`. However, Boolean Array Indexing is faster when `obj.shape == x.shape`. 
        - `numpy.nonzero`
            - `numpy.nonzero(a)` will return a tuple of `a.ndim` `ndarray`s, let `r_1, ..., r_N`. Then, `a`'s `i-th` non-zero element (counted in row-major, `C`-style order) is `a[r_1[i], ..., r_N[i]]`. 
            - Offical expression: Returns a tuple of arrays, one for each dimension of `a`, containing the indices of the non-zero elements in that dimension. The values in `a` are always tested and returned in row-major, `C`-style order.
    - If `obj.ndim == x.ndim`, `x[obj]` returns a 1D array filled with the elements of `x` corresponding to the `True` values of `obj`. The search order will be row-major, `C`-style. 
        - If `obj` has True values at entries that are outside of the bounds of `x`, then an `IndexError` will be raised.
        - If `obj` is smaller than `x`, it is identical to filling it with `False`.
    - **Example 4**: In this example, items greater than 5 are returned as a result of Boolean indexing
    ```
    >>> x = np.array([[ 0,  1,  2], 
                      [ 3,  4,  5], 
                      [ 6,  7,  8], 
                      [ 9, 10, 11]])
    >>> x[x > 5]
    [6 7 8 9 10 11]
    ```
    - **Example 5**: In this example, `NaN` (Not a Number) elements are omitted
    ```
    >>> a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
    >>> a[~np.isnan(a)]
    [ 1. 2. 3. 4. 5.]
    ```
    - **Example 6**: The following example adds a constant to all negative elements
    ```
    >>> x = np.array([1., -1., -2., 3])
    >>> x[x < 0] += 20
    >>> x
    array([  1.,  19.,  18.,   3.])
    ```
    - If an index includes a Boolean array, the result will be identical to inserting `obj.nonzero()` into the same position and using the integer array indexing mechanism. `x[a_1, boolean_array, a_2]` is equivalent to `x[(ind_1,) + boolean_array.nonzero() + (ind_2,)]`.
    - **Example 7**: 
    ```
    >>> x = np.array([[0, 1], [1, 1], [2, 2]])
    >>> rowsum = x.sum(-1)
    >>> x[rowsum <= 2, :]
    [[0, 1],
     [1, 1]]
    ```

### 🌱 Broadcasting

- The term *broadcasting* refers to the ability of *NumPy* to treat arrays of different shapes during arithmetic operations
- Arithmetic operations on arrays are usually done elementwise
    - If two arrays are of exactly the same shape, then these operations are smoothly performed
    - If the dimensions of two arrays are dissimilar, the smaller array is *broadcast* to the size of the larger array so that they have compatible shapes
- *Broadcasting* is possible if the following rules are satisfied: 
    - Array with smaller `ndim` than the other is prepended with `1` in its shape 
    - Size in each dimension of the output shape is maximum of the input sizes in that dimension 
    - An input can be used in calculation, if its size in a particular dimension: *matches* the output size, or is *exactly `1`* 
    - If an input has a dimension size of `1`, the first data entry in that dimension is used for all calculations along that dimension 
- A set of arrays is said to be *broadcastable* if the above rules produce a valid result and one of the following is true: 
    - Arrays have exactly the same shape
    - Arrays have the same number of dimensions and the length of each dimension is either a common length or `1`
    - Array having too few dimensions can have its shape prepended with a dimension of length `1`, so that the above stated property is true

### 🌱 Iterating Over Array

- Iterator object `numpy.nditer`: 
    - Efficient multidimensional iterator object using which it is possible to iterate over an array. 
    - Each element of an array is visited using Python’s standard Iterator interface.
    - **Example 1**
    ```
    >>> import numpy as np
    >>> a = np.arange(0, 60, 5).reshape(3, 4)
    >>> a
    [[ 0  5 10 15]
     [20 25 30 35]
     [40 45 50 55]]

    >>> for x in np.nditer(a):
    ...     print(x, end=' ')
    0 5 10 15 20 25 30 35 40 45 50 55
    ```
    - **Example 2**: The order of iteration is chosen to match the *memory layout* of an array, **without** considering a particular ordering. This can be seen by iterating over the transpose of the above array: 
    ```
    >>> a = np.arange(0, 60, 5).reshape(3, 4)
    >>> b = a.T
    >>> b
    [[ 0 20 40]
     [ 5 25 45]
     [10 30 50]
     [15 35 55]]

    >>> for x in np.nditer(b):
    ...     print(x, end=' ')
    0 5 10 15 20 25 30 35 40 45 50 55
    ```
- Iteration Order
    - **Example 3**: If the same elements are stored using `F`-style order, the iterator chooses the more efficient way of iterating over an array.
    ```
    >>> a = np.arange(0, 60, 5).reshape(3, 4)
    >>> a
    [[ 0  5 10 15]
     [20 25 30 35]
     [40 45 50 55]]
    
    >>> b = a.T
    >>> b
    [[ 0 20 40]
     [ 5 25 45]
     [10 30 50]
     [15 35 55]]
    
    >>> c = b.copy(order='C')
    >>> c
    [[ 0 20 40]
     [ 5 25 45]
     [10 30 50]
     [15 35 55]]
    
    >>> for x in np.nditer(c):
    ...     print(x, end=' ')
    0 20 40 5 25 45 10 30 50 15 35 55
    
    >>> d = b.copy(order='F')
    >>> d
    [[ 0 20 40]
     [ 5 25 45]
     [10 30 50]
     [15 35 55]]
    
    >>> for x in np.nditer(d):
    ...     print(x, end=' ')
    0 5 10 15 20 25 30 35 40 45 50 55
    ```
    - **Example 4**: It is possible to force nditer object to use a specific order by explicitly mentioning it.
    ```
    >>> a = np.arange(0, 60, 5).reshape(3, 4)
    >>> a
    [[ 0  5 10 15]
     [20 25 30 35]
     [40 45 50 55]]
     
    >>> for x in np.nditer(a, order='C'):
    ...     print(x, end=' ')
    0 5 10 15 20 25 30 35 40 45 50 55
    
    >>> for x in np.nditer(a, order='F'):
    ...     print(x, end=' ')
    0 20 40 5 25 45 10 30 50 15 35 55
    ```
- Modifying Array Values
    - The `nditer` object has another optional parameter called `op_flags`. Its default value is `'read-only'`, but can be set to `'read-write'` or `'write-only'` mode. This will enable modifying array elements using this iterator.
    - **Example 5**: 
    ```
    >>> a = np.arange(0, 60, 5).reshape(3, 4)
    >>> a
    [[ 0  5 10 15]
     [20 25 30 35]
     [40 45 50 55]]
    
    >>> for x in np.nditer(a, op_flags=['readwrite']):
    ...     x[...]= 2 * x
    
    >>> a
    [[ 0  10  20  30]
     [ 40 50  60  70]
     [ 80 90 100 110]]
    ```
- External Loop
    - The `nditer` class constructor has a `flags` parameter, which can take the following values:
        - `c_index`: `C`-order index can be tracked
        - `f_index`: `Fortran`-order index is tracked
        - `multi-index`: Type of indexes with one per iteration can be tracked
        - `external_loop`: Causes values given to be one-dimensional arrays with multiple values instead of zero-dimensional array
    - **Example 6**: In the following example, one-dimensional arrays corresponding to each column is traversed by the iterator.
    ```
    >>> a = np.arange(0, 60, 5).reshape(3, 4)
    >>> a
    [[ 0  5 10 15]
     [20 25 30 35]
     [40 45 50 55]]
     
    >>> for x in np.nditer(a, flags=['external_loop'], order='F'):
    ...     print(x, end=' ')
    [ 0 20 40] [ 5 25 45] [10 30 50] [15 35 55]
    ```
- Broadcasting Iteration
    - If two arrays are *broadcastable*, a combined `nditer` object is able to iterate upon them concurrently.
    - **Example 7**: Assuming that an array `a` has dimension `3 * 4`, and there is another array `b` of dimension `1 * 4`, the iterator of following type is used (array `b` is broadcast to size of `a`).
    ```
    >>> a = np.arange(0, 60, 5).reshape(3, 4)
    >>> a
    [[ 0  5 10 15]
     [20 25 30 35]
     [40 45 50 55]]
    
    >>> b = np.array([1, 2, 3, 4], dtype=int)
    >>> b
    [1 2 3 4]
    
    >>> for x, y in np.nditer([a, b]):
    ...     print('{}:{}'.format(x, y), end=' ')
    0:1 5:2 10:3 15:4 20:1 25:2 30:3 35:4 40:1 45:2 50:3 55:4
    ```

### 🌱 Array Manipulation

- Several routines are available for manipulation of elements in `ndarray`: 
    - Changing Shape
        - [`numpy.atleast_1d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d), [`numpy.atleast_2d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html#numpy.atleast_2d), [`numpy.atleast_3d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html#numpy.atleast_3d)
            - Convert inputs to arrays with at least 1/2/3 dimension. Scalar inputs are converted to 1/2/3-dimensional arrays, whilst higher-dimensional inputs are preserved. 
            - Signature: 
            ```
            numpy.atleast_1d(*arys)
            ```
            - Parameters: 
                - `*arys`: `array_like`s. One or more input arrays. 
            - Returns: 
                - `ret`: `ndarray`. An array, or `list` of arrays, each with `a.ndim >= 1/2/3`. Copies are made only if necessary. 
            ```
            >>> np.atleast_1d(1.0)
            array([1.])

            >>> x = np.arange(9.0).reshape(3, 3)
            >>> np.atleast_1d(x)
            array([[0., 1., 2.],
                   [3., 4., 5.],
                   [6., 7., 8.]])

            >>> np.atleast_1d(x) is x
            True

            >>> np.atleast_1d(1, [3, 4])
            [array([1]), array([3, 4])]
            ```
        - [`numpy.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html?highlight=numpy%20reshape#numpy.reshape), [`numpy.ndarray.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape)
            - Gives a new shape to an array **without** changing its data.
            - Signature: 
            ```
            numpy.reshape(a, newshape, order='C')
            ```
            - Parameters: 
                - `a`: `array_like`. Array to be reshaped
                - `newshape`: `int` or `Tuple[int]`. The new shape should be compatible with the original shape. If `int`, then the result will be a 1D array of that length. One shape dimension can be `-1`. In this case, the value is inferred from the length of the array and remaining dimensions. 
                - `order`: `{'C', 'F', 'A'}`, *optional*. 
            - Returns: 
                - `reshaped_array`: `ndarray`. This will be a *new view* object if possible; otherwise, it will be a copy. Note there is **NO** guarantee of the memory layout (C or Fortran contiguous) of the returned array.
            - Notes: 
                - It is **NOT** always possible to change the shape of an array **without** copying the data. If you want an error to be raised when the data is copied, you should assign the new shape to the shape attribute of the array:
                ```
                >>> a = np.zeros((10, 2))

                # A transpose makes the array non-contiguous
                >>> b = a.T

                # Taking a view makes it possible to modify the shape 
                # without modifying the initial object.
                >>> c = b.view()
                >>> c.shape = (20)
                Traceback (most recent call last):
                   ...
                AttributeError: Incompatible shape for in-place modification. Use 
                `.reshape()` to make a copy with the desired shape.
                ```
            - Examples
            ```
            >>> a = np.array([[1, 2, 3], [4, 5, 6]])
            
            >>> a.base
            None
            
            >>> b = np.reshape(a, (3, -1))  # the unspecified value is inferred to be 2
            >>> b
            array([[1, 2],
                   [3, 4],
                   [5, 6]])
                   
            >>> b.base is a
            True
            ```
        - `numpy.ndarray.flat`
            - A 1-D iterator over the array. This is a `numpy.flatiter` instance, which acts similarly to, but is not a subclass of, `Python`’s built-in `iter` object. 
            ```
            >>> x = np.arange(1, 7).reshape(2, 3)
            >>> x
            array([[1, 2, 3],
                   [4, 5, 6]])

            >>> x.flat[3]
            4

            >>> x.T
            array([[1, 4],
                   [2, 5],
                   [3, 6]])

            >>> x.T.flat[3]
            5

            >>> type(x.flat)
            <class 'numpy.flatiter'>

            >>> x.flat = 3
            >>> x
            array([[3, 3, 3],
                   [3, 3, 3]])

            >>> x.flat[[1, 4]] = 1
            >>> x
            array([[3, 1, 3],
                   [3, 1, 3]])
            ```
        - [`numpy.ndarray.flatten`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten)
            - Return a *copy* of the array collapsed into one dimension.
            - Signature: 
            ```
            ndarray.flatten(order='C')
            ```
            - Parameters: 
                - `order`: `{'C', 'F', 'A', 'K'}`, *optional*. 
            - Returns: 
                - `y`: `ndarray`. A copy of the input array, flattened to one dimension.
            ```
            >>> a = np.arange(8).reshape(2, 4)
            >>> a
            [[0 1 2 3]
             [4 5 6 7]]
             
            >>> b = a.flatten()
            >>> b
            [0 1 2 3 4 5 6 7]
            >>> b.flags
            C_CONTIGUOUS : True
            F_CONTIGUOUS : True
            OWNDATA : True
            WRITEABLE : True
            ALIGNED : True
            WRITEBACKIFCOPY : False
            UPDATEIFCOPY : False
            
            >>> a.flatten(order='F')
            [0 4 1 5 2 6 3 7]
            ```
        - [`numpy.ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html?highlight=numpy%20ravel)
            - This function returns a *view* of original array that is flattened into contiguous 1D (whenever possible. A copy is made only if needed). The returned array will have the same type as that of the input array. 
            - Signature: 
            ```
            ndarray.ravel(a, order='C')
            ```
            - Parameters: 
                - `a`: `array_like`. Input array. The elements in a are read in the order specified by order, and packed as a 1-D array. 
                - `order`: `{'C', 'F', 'A', 'K'}`, *optional*. 
            ```
            >>> a = np.arange(8).reshape(2, 4)
            >>> a
            [[0 1 2 3]
             [4 5 6 7]]
             
            >>> a.ravel()
            [0 1 2 3 4 5 6 7]
            
            >>> a.ravel(order='F')
            [0 4 1 5 2 6 3 7]
            ```
    - Transpose Operations
        - [`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose)
            - Reverse or permute the axes of an array; returns the modified array. For an array a with two axes, this function gives the matrix transpose. 
            - Signature
            ```
            numpy.transpose(a, axes=None)
            ```
            - Parameters: 
                - `a`: `array_like`. The array to be transposed
                - `axes`: `Tuple[int]` or `List[int]`, *optional*. If specified, it must be a `tuple` or `list` which contains a permutation of `[0, 1 , .., N - 1]` where `N == a.ndim`. The `i`-th axis of the returned array will correspond to the axis numbered `axes[i]` of the input. If not specified, defaults to `range(a.ndim)[::-1]`, which reverses the order of the axes.
            - Returns: 
                - `p`: `ndarray`. `a` with its axes permuted. A *view* is returned whenever possible. 
            - Notes: 
                - Use `np.transpose(a, np.argsort(axes))` to invert the transposition of tensors when using the axes keyword argument.
                - Transposing a 1-D array returns an *unchanged* view of the original array.
            ```
            >>> a = np.arange(12).reshape(3, 4)
            >>> a
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
            
            >>> b = np.transpose(a)
            >> b
            [[ 0,  4,  8],
             [ 1,  5,  9],
             [ 2,  6, 10],
             [ 3,  7, 11]]
            
            >>> a
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
            
            >>> b.base is a.base
            True
            ```
        - [`numpy.ndarray.T`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html?highlight=numpy%20ndarray%20t#numpy.ndarray.T)
            - This *attribute* belongs to `ndarray` class. It behaves similar to `self.transpose()`, i.e. is actually a *view* to original array but with its own transposed index mapping.  
            ```
            >>> a = np.arange(12).reshape(3, 4)
            >>> a
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
            
            >>> b = a.T
            >>> b
            [[ 0,  4,  8],
             [ 1,  5,  9],
             [ 2,  6, 10],
             [ 3,  7, 11]]
            
            >>> a
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
            
            >>> b.base is a.base
            True
            ```
        - [`numpy.moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html)
            - Move axes of an array to new positions. Other axes remain in their original order. 
            - Signature: 
            ```
            numpy.moveaxis(a, source, destination)
            ```
            - Parameters: 
                - `a`: `ndarray`. The array whose axes should be reordered.
                - `source`: `int` or `Sequence[int]`. Original positions of the axes to move. These must be unique. 
                - `destination`: `int` or `Sequence[int]`. Destination positions for each of the original axes. These must also be unique. 
            - Returns
                - `result`: `ndarray`. Array with moved axes. This array is a *view* of the input array. 
            ```
            >>> x = np.zeros((3, 4, 5))

            >>> np.moveaxis(x, 0, -1).shape
            (4, 5, 3)

            >>> np.moveaxis(x, -1, 0).shape
            (5, 3, 4)
            
            >>> np.transpose(x).shape
            (5, 4, 3)

            >>> np.swapaxes(x, 0, -1).shape
            (5, 4, 3)

            >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
            (5, 4, 3)

            >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
            (5, 4, 3)
            ```
        - [`numpy.rollaxis`](https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html)
            - Roll the specified axis backwards, until it lies in a given position. This function continues to be supported for backward compatibility, but you should prefer `moveaxis`. 
                - e.g. roll axis `2` backwards to `0` will have index dimention mapping `0 1 2 -> 0 2 1 -> 2 0 1`
            - Signature: 
            ```
            numpy.rollaxis(a, axis, start=0)
            ```
            - Parameters: 
                - `a`: `ndarray`. Input array
                - `axis`: `int`. Axis to roll backwards. The positions of the other axes do not change relative to one another. 
                - `start`: `int`, *optional*. `When start <= axis`, the axis is rolled back until it lies in this position. When `start > axis`, the axis is rolled until it lies before this position. The default, `0`, results in a “complete” roll. The following table describes how negative values of start are interpreted: `0` by default leading to the complete roll: 
                    - `-(a.ndim + 1)`: raise `AxisError`
                    - `-a.ndim`: 0
                    - `...`: `...`
                    - `-1`: `a.ndim - 1`
                    - `-1`: `0`
                    - `...`: `...`
                    - `a.ndim`: `a.ndim`
                    - `a.ndim + 1`: raise `AxisError`
            - Returns: 
                - `res`: `ndarray`. A view of a is always returned `(since NumPy 1.10.0)`. 
            ```
            >>> a = np.arange(8).reshape(2, 2, 2)
            >>> a
            array([[[0, 1],
                    [2, 3]],

                   [[4, 5],
                    [6, 7]]])
            
            >>> b = np.rollaxis(a, 2)
            >>> b.base is a.base
            True
            >>> b
            array([[[0, 2],
                    [4, 6]],

                   [[1, 3],
                    [5, 7]]])
                    
            >>> c = np.rollaxis(a, 2, 1)
            >>> c
            array([[[0, 1],
                    [4, 5]],

                   [[2, 3],
                    [6, 7]]])
            ```
        - [`numpy.swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html)
            - Interchange two axes of an array. 
            - Signature
            ```
            numpy.swapaxes(a, axis1, axis2)
            ```
            - Parameters: 
                - `a`: `array_like`. Input array whose axes are to be swapped
                - `axis1`: `int`. First axis
                - `axis2`: `int`. Second axis
            - Returns: 
                - `a_swapped`: `ndarray`. If `a` is an `ndarray`, then a *view* is returned; otherwise, a new array is created.
            ```
            >>> a = np.arange(8).reshape(2, 2, 2)
            >>> a
            array([[[0, 1],
                    [2, 3]],

                   [[4, 5],
                    [6, 7]]])
            
            >>> b = np.swapaxes(a, 2, 0)
            >>> b
            array([[[0, 4],
                    [2, 6]],

                   [[1, 5],
                    [3, 7]]])
            
            >>> b.base is a.base
            True
            
            >>> for i in range(2):
            ...    for j in range(2):
            ...        for k in range(2):
            ...            if b[i, j, k].base is not a[k, j, i].base: 
            ...                print('hehe')
            
            ```
    - Changing Dimensions
        - `numpy.broadcast`: Produces an object that mimics broadcasting
            - It returns an object that encapsulates the result of broadcasting one array against the other.
            ```
            >>> x = np.array([[1], [2], [3]])
            >>> y = np.array([4, 5, 6])
            >>> b = np.broadcast(x, y)
            
            >>> for (r, c) in b:
            ...     print(r, c)
            1 4
            1 5
            1 6
            2 4
            2 5
            2 6
            3 4
            3 5
            3 6

            >>> out = np.empty(b.shape)
            >>> out.flat = [u + v for (u, v) in b]
            >>> out
            array([[5.,  6.,  7.],
                   [6.,  7.,  8.],
                   [7.,  8.,  9.]])
            ```
        - `numpy.broadcast_to`: Broadcasts an array to a new shape
            - This function broadcasts an array to a new shape. It returns a *read-only view* on the original array. It is typically **NOT** contiguous. The function may throw ValueError if the new shape does not comply with NumPy's broadcasting rules.
            - Signature
            ```
            numpy.broadcast_to(array, shape, subok=False)
            ```
            - Parameters
                - `array`: `array_like`. The array to broadcast.
                - `shape`: `tuple`. The shape of the desired array.
                - `subok`: `bool`, *optional*. If `True`, then sub-classes will be passed-through, otherwise the returned array will be forced to be a base-class array (default).
            ```
            >>> a = np.array([0, 1, 2, 3])
            >>> np.broadcast_to(a, (4, 4))
            [[0 1 2 3]
             [0 1 2 3]
             [0 1 2 3]
             [0 1 2 3]]
            ```
        - `numpy.broadcast_arrays`
            - Broadcast any number of arrays against each other. Return a `list` of arrays (*views* on the original arrays, typically **NOT** contiguous. Furthermore, more than one element of a broadcasted array may refer to a single memory location).
            - Signature
            ```
            numpy.broadcast_arrays(*args, subok=False)
            ```
            - Parameters
                - `*args`: `array_likes`. The arrays to broadcast.
                - `subok`: `bool`, *optional*. If `True`, then sub-classes will be passed-through, otherwise the returned arrays will be forced to be a base-class array (default).
            ```
            >>> x = np.array([[1, 2, 3]])
            >>> y = np.array([[4], [5]])
            >>> np.broadcast_arrays(x, y)
            [array([[1, 2, 3],
                    [1, 2, 3]]),
             array([[4, 4, 4],
                    [5, 5, 5]])]
            ```
            - Here is a useful idiom for getting *contiguous copies* instead of non-contiguous views:
            ```
            >>> [np.array(a) for a in np.broadcast_arrays(x, y)]
            [array([[1, 2, 3],
                    [1, 2, 3]]),
             array([[4, 4, 4],
                    [5, 5, 5]])]
            ```
        - [`numpy.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html?highlight=numpy%20expand_dims#numpy.expand_dims)
            - Expand the shape of an array. Insert a new axis that will appear at the axis position in the expanded array shape. 
            - Signature
            ```
            numpy.expand_dims(a, axis)
            ```
            - Parameters
                - `a`: `array_like`. Input array
                - `axis`: `int` or `Tuple[int]`. Position in the expanded axes where the new axis (or axes) is placed. 
            - Returns: 
                - `result`: `ndarray`. *View* of `a` with the number of dimensions increased.
            - Notes: 
                - Note that some examples may use `None` instead of `np.newaxis`. These are the same objects:
                ```
                >>> np.newaxis is None
                True
                ```
            - Examples: 
            ```
            >>> x = np.array([1, 2])
            >>> x.shape
            (2,)
            
            # equivalent to x[np.newaxis, :] or x[np.newaxis]
            >>> y = np.expand_dims(x, axis=0)
            >>> y
            array([[1, 2]])
            
            >>> y.shape
            (1, 2)
            
            # equivalent to x[:, np.newaxis]
            >>> y = np.expand_dims(x, axis=1)
            >>> y
            array([[1],
                   [2]])

            >>> y.shape
            (2, 1)
          
            # axis may also be a tuple
            >>> x = np.array([[1, 2], [3, 4]])
            >>> x.shape
            (2, 2)
            
            >>> y = np.exapnd_dims(x, axis=(2, 0))
            >>> y.shape
            (1, 2, 1, 2)
            ```
        - [`numpy.squeeze`](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html#numpy.squeeze)
            - Remove single-dimensional entries from the shape of an array. 
            - Signature: 
            ```
            numpy.squeeze(a, axis=None)
            ```
            - Parameters: 
                - `arr`: `array_like`. `Input array
                - `axis`: `None` or `int` or `Tuple[int]`. Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised. 
            - Returns: 
                - `squeezed`: `ndarray`. The input array, but with all or a subset of the dimensions of length `1` removed. This is always `a` itself or a *view* into `a`. Note that if all axes are squeezed, the result is a 0-D array and **NOT** a scalar.
            ```
            >>> x = np.arange(9).reshape(1, 3, 3)
            >>> x
            array([[[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]]])
                    
            >>> y = np.squeeze(x)
            >>> y
            array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])
                   
            >>> y.base is x.base
            True
            ```
    - Joining Arrays
        - [`numpy.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html?highlight=numpy%20concatenate#numpy.concatenate)
            - Join a sequence of arrays along an *existing axis*.
            - Signature: 
            ```
            numpy.concatenate((a1, a2, ...), axis=0, out=None)
            ```
            - Parameters: 
                - `a1, a2, ...`: `Sequence[array_like]`. The arrays must have the *same shape*, except in the dimension corresponding to `axis`.
                - `axis`: `None` or `int`, *optional*. The axis along which the arrays will be joined. If axis is `None`, arrays are flattened before use. 
                - `out`: `ndarray`, *optional*. If provided, the destination to place the result. The shape must be correct, matching that of what concatenate would have returned if no out argument were specified.
            - Returns: 
                - `res`: `ndarray`. The concatenated array.
            ```
            >>> a = np.array([[1, 2], [3, 4]])
            >>> b = np.array([[5, 6]])

            >>> np.concatenate((a, b), axis=0)
            array([[1, 2],
                   [3, 4],
                   [5, 6]])

            >>> np.concatenate((a, b.T), axis=1)
            array([[1, 2, 5],
                   [3, 4, 6]])

            >>> np.concatenate((a, b), axis=None)
            array([1, 2, 3, 4, 5, 6])
            ```
        - [`numpy.stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html#numpy.stack)
            - Join a sequence of arrays along a *new axis*. The `axis`parameter specifies the index of the new axis in the dimensions of the result. For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension. 
            - Signature: 
            ```
            numpy.stack(arrays, axis=0, out=None)
            ```
            - Parameters: 
                - `arrays`: `Sequence[array_like]`. Each array must have the *same shape*.
                - `axis`: `int`, *optional*. The axis in the result array along which the input arrays are stacked.
                - `out`: `ndarray`, *optional*. If provided, the destination to place the result. The shape must be correct, matching that of what stack would have returned if no out argument were specified. 
            - Returns: 
                - `stacked`: `ndarray`. The stacked array has one more dimension than the input arrays. 
            ```
            >>> arrays = [np.empty((3, 4)) for _ in range(10)]
            >>> np.stack(arrays, axis=0).shape
            (10, 3, 4)
            
            >>> np.stack(arrays, axis=1).shape
            (3, 10, 4)

            >>> np.stack(arrays, axis=2).shape
            (3, 4, 10)
            
            >>> a = np.array([1, 2, 3])
            >>> b = np.array([2, 3, 4])
            >>> np.stack((a, b))
            array([[1, 2, 3],
                   [2, 3, 4]])
            
            >>> np.stack((a, b), axis=-1)
            array([[1, 2],
                   [2, 3],
                   [3, 4]])
            ```
        - [`numpy.block`](https://numpy.org/doc/stable/reference/generated/numpy.block.html?highlight=numpy%20block#numpy.block)
            - Assemble an nd-array from nested lists of blocks.
            - Blocks in the innermost lists are concatenated along the last dimension (`-1`), then these are concatenated along the second-last dimension (`-2`), and so on until the outermost list is reached.
            - Blocks can be of any dimension, but will **NOT** be broadcasted using the normal rules. Instead, leading axes of size `1` are inserted, to make `block.ndim` the same for all blocks. This is primarily useful for working with scalars, and means that code like `np.block([v, 1])` is valid, where `v.ndim == 1`.
            - When the nested list is two levels deep, this allows block matrices to be constructed from their components.
            - Signature: 
            ```
            numpy.block(arrays)
            ```
            - Parameters: 
                - `arrays`: nested `list` of `array_like` or `scalars` (but **NOT** `tuples`). If passed a single `ndarray` or `scalar` (a nested list of depth 0), this is returned unmodified (and not copied). Elements shapes must match along the appropriate axes (without broadcasting), but leading `1`s will be prepended to the shape as necessary to make the dimensions match.
            - Returns: 
                - `block_array`: `ndarray`. The array assembled from the given blocks. The dimensionality of the output is equal to the greatest of: 
                    - the dimensionality of all the inputs; 
                    - the depth to which the input list is nested. 
            - Examples: 
                - The most common use of this function is to build a block matrix
                ```
                >>> A = np.eye(2) * 2
                >>> B = np.eye(3) * 3
                >>> np.block([
                ...     [A,               np.zeros((2, 3))],
                ...     [np.ones((3, 2)), B               ]
                ... ])
                array([[2., 0., 0., 0., 0.],
                       [0., 2., 0., 0., 0.],
                       [1., 1., 3., 0., 0.],
                       [1., 1., 0., 3., 0.],
                       [1., 1., 0., 0., 3.]])
                ```
                - With a list of depth 1, block can be used as `numpy.hstack`: 
                ```
                >>> np.block([1, 2, 3])              # hstack([1, 2, 3])
                array([1, 2, 3])

                >>> a = np.array([1, 2, 3])
                >>> b = np.array([2, 3, 4])
                >>> np.block([a, b, 10])             # hstack([a, b, 10])
                array([ 1,  2,  3,  2,  3,  4, 10])

                >>> A = np.ones((2, 2), int)
                >>> B = 2 * A
                >>> np.block([A, B])                 # hstack([A, B])
                array([[1, 1, 2, 2],
                       [1, 1, 2, 2]])
                ```
                - With a list of depth 2, block can be used in place of `numpy.vstack`:
                ```
                >>> a = np.array([1, 2, 3])
                >>> b = np.array([2, 3, 4])
                >>> np.block([[a], [b]])             # vstack([a, b])
                array([[1, 2, 3],
                       [2, 3, 4]])

                >>> A = np.ones((2, 2), int)
                >>> B = 2 * A
                >>> np.block([[A], [B]])             # vstack([A, B])
                array([[1, 1],
                       [1, 1],
                       [2, 2],
                       [2, 2]])
                ```
                - It can also be used in places of `numpy.atleast_1d` and `numpy.atleast_2d`: 
                ```
                >>> a = np.array(0)
                >>> b = np.array([1])

                >>> np.block([a])                    # atleast_1d(a)
                array([0])

                >>> np.block([b])                    # atleast_1d(b)
                array([1])

                >>> np.block([[a]])                  # atleast_2d(a)
                array([[0]])

                >>> np.block([[b]])                  # atleast_2d(b)
                array([[1]])
                ```
        - [`numpy.vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack)
            - Stack arrays in sequence vertically (row wise). 
            - This is equivalent to concatenation along the first axis after 1-D arrays of shape `(N,)` have been reshaped to `(1, N)`. Rebuilds arrays divided by `vsplit`. 
            - This function makes most sense for arrays with up to 3 dimensions. For instance, for pixel-data with a height (first axis), width (second axis), and r/g/b channels (third axis). The functions `concatenate`, `stack` and `block` provide more general stacking and concatenation operations. 
            - Signature: 
            ```
            numpy.vstack(tup)
            ```
            - Parameters: 
                - `tup`: `Sequence[ndarray]`. The arrays must have the same shape along all but the *first axis*. 1-D arrays must have the same length.
            - Returns: 
                - `stacked`: `ndarray`. The array formed by stacking the given arrays, will be at least 2-D. 
            ```
            >>> a = np.array([1, 2, 3])
            >>> b = np.array([2, 3, 4])
            >>> np.vstack((a,b))
            array([[1, 2, 3],
                   [2, 3, 4]])
                   
            >>> a = np.array([[1], [2], [3]])
            >>> b = np.array([[2], [3], [4]])
            >>> np.vstack((a,b))
            array([[1],
                   [2],
                   [3],
                   [2],
                   [3],
                   [4]])
            ```
        - [`numpy.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy.hstack)
            - Stack arrays in sequence horizontally (column wise). 
            - This is equivalent to concatenation along the second axis, except for 1-D arrays where it concatenates along the first axis. Rebuilds arrays divided by `hsplit`. 
            - Signature: 
            ```
            numpy.hstack(tup)
            ```
            - Parameters: 
                - `tup`: `Sequence[ndarray]`. The arrays must have the same shape along all but the *second axis*, except 1-D arrays which can be any length.
            - Returns: 
                - `stacked`: `ndarray`. The array formed by stacking the given arrays. 
            ```
            >>> a = np.array((1, 2, 3))
            >>> b = np.array((2, 3, 4))
            >>> np.hstack((a, b))
            array([1, 2, 3, 2, 3, 4])

            >>> a = np.array([[1], [2], [3]])
            >>> b = np.array([[2], [3], [4]])
            >>> np.hstack((a, b))
            array([[1, 2],
                   [2, 3],
                   [3, 4]])
            ```
        - [`numpy.dstack`](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html#numpy.dstack)
            - Stack arrays in sequence depth wise (along third axis). This is equivalent to concatenation along the third axis after 2-D arrays of shape `(M, N)` have been reshaped to `(M, N, 1)` and 1-D arrays of shape `(N,)` have been reshaped to `(1, N, 1)`. Rebuilds arrays divided by `dsplit`.
            - Signature: 
            ```
            numpy.dstack(tup)
            ```
            - Parameters: 
                - `tup`: `Sequence[ndarray]`. The arrays must have the same shape along all but the *third axis*. 1-D or 2-D arrays must have the same shape. 
            - Returns: 
                - `stacked`: `ndarray`. The array formed by stacking the given arrays, will be at least 3-D. 
            ```
            >>> a = np.array((1, 2, 3))
            >>> b = np.array((2, 3, 4))
            >>> np.hstack((a, b))
            array([[[1, 2],
                    [2, 3],
                    [3, 4]]])

            >>> a = np.array([[1], [2], [3]])
            >>> b = np.array([[2], [3], [4]])
            >>> np.hstack((a, b))
            array([[[1, 2]],
                   [[2, 3]],
                   [[3, 4]]])
            ```
        - [`numpy.column_stack`](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html#numpy.column_stack)
            - Stack 1-D arrays as columns into a 2-D array. 
            - Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array. 2-D arrays are stacked as-is, just like with hstack. 1-D arrays are turned into 2-D columns first. 
            - Signature: 
            ```
            numpy.column_stack(tup)
            ```
            - Parameters: 
                - `tup`: sequence of 1-D or 2-D arrays.. Arrays to stack. All of them must have the same first dimension.  
            - Returns: 
                - `stacked`: 2-D array. The array formed by stacking the given arrays. 
            ```
            >>> a = np.array((1, 2, 3))
            >>> b = np.array((2, 3, 4))

            >>> np.column_stack((a, b))
            array([[1, 2],
                   [2, 3],
                   [3, 4]])
            ```
    - Splitting Arrays
        - `numpy.split`
        - [`numpy.array_split`](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split)
            - Split an array into multiple sub-arrays. Same as `split` except `array_split` allows `indices_or_sections` to be an integer that does **NOT** equally divide the axis. For an array of length `l` that should be split into `n` sections, it returns `l % n` sub-arrays of `size l//n + 1` and the rest of size `l//n`. 
            ```
            >>> x = np.arange(8.0)
            >>> np.array_split(x, 3)
            [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]
            
            >>> x = np.arange(7.0)
            >>> np.array_split(x, 3)
            [array([0.,  1.,  2.]), array([3.,  4.]), array([5.,  6.])]
            ```
        - `numpy.hsplit`
        - `numpy.vsplit`
        - `numpy.dsplit`
    - Adding / Removing Elements
        - `numpy.resize`
        - `numpy.append`
        - `numpy.insert`
        - `numpy.delete`
        - `numpy.unique`

### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























