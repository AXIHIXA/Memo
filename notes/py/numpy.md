# `NumPy` Tutorial Notes

- Notes of reading:
    - [`Numpy Tutorial - Tutorialspoint`](https://www.tutorialspoint.com/numpy/index.htm)

### 🌱 `ndarray` Object

- `ndarray` describes the collection of items of the *same type*
    - Items in the collection can be accessed using a zero-based index
    - Every item in an ndarray takes the same size of block in the memory
    - Each element in `ndarray` is an object of data-type object (called `dtype`)
    - Any item extracted from `ndarray` object (by *slicing*) is represented by a `Python` object of one of array scalar types

![](https://www.tutorialspoint.com/numpy/images/ndarray.jpg)

- The basic ndarray is created using `np.array` function in `NumPy`, creating an `ndarray` from any object exposing array interface, or from any method that returns an array
    - signature
    ```
    numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)
    ```
    - parameters
        - `object`: Any object exposing the array interface method returns an array, or any (nested) sequence
        - `dtype`: *Optional*. Desired data type of array
        - `copy`: *Optional*. By default (true), the object is copied 
        - `order`: `C` (`C`-style, row major) or `F` (`FORTRAN`-style, column major) or `A` (any) (default)
        - `subok`: By default, returned array forced to be a base class array. If true, sub-classes passed through
        - `ndimin`: Specifies minimum dimensions of resultant array

### 🌱 Data Types

- Scalar Data Types
    - `np.bool_`: Boolean (`True` or `False`) stored as a byte
    - `np.int_`: Default integer type (same as `C` long; normally either `np.int64` or `np.int32`)
    - `np.intc`: Identical to `C` `int` (normally `np.int32` or `np.int64`)
    - `np.intp`: Integer used for indexing (same as `C` `ssize_t`; normally either `np.int32` or `np.int64`)
    - `np.int8`: Byte (-128 to 127)
    - `np.int16`: Integer (-32768 to 32767)
    - `np.int32`: Integer (-2147483648 to 2147483647)
    - `np.int64`: Integer (-9223372036854775808 to 9223372036854775807)
    - `np.uint8`: Unsigned integer (0 to 255)
    - `np.uint16`: Unsigned integer (0 to 65535)
    - `np.uint32`: Unsigned integer (0 to 4294967295)
    - `np.uint64`: Unsigned integer (0 to 18446744073709551615)
    - `np.float_`: Shorthand for `np.float64`
    - `np.float16`: Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    - `np.float32`: Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    - `np.float64`: Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    - `np.complex_`: Shorthand for `np.complex128`
    - `np.complex64`: Complex number, represented by two 32-bit floats (real and imaginary components)
    - `np.complex128`: Complex number, represented by two 64-bit floats (real and imaginary components)
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
        np.dtype(object, align, copy)
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

- `np.empty`
    - creates an *uninitialized* array of specified `shape` and `dtype`
    - signature
    ```
    numpy.empty(shape, dtype=float, order='C')
    ```
    - parameters
        - `shape`: int, or tuple of int. shape of an empty array
        - `dtype`: desired output data type. *Optional*
        - `order`: `'C'` for `C`-style row-major array, `'F'` for `FORTRAN`-style column-major array
```
>>> import numpy as np
>>> x = np.empty([3, 2], dtype=int)
>>> x
[[22649312   1701344351]
 [1818321759 1885959276]
 [16779776   156368896]]
```
- `np.zeros`
    - returns a new array of specified size, filled with zeros
        - signature
    ```
    numpy.zeros(shape, dtype=float, order='C')
    ```
    - parameters
        - `shape`: int, or tuple of int. shape of an empty array
        - `dtype`: desired output data type. *Optional*
        - `order`: `'C'` for `C`-style row-major array, `'F'` for `FORTRAN`-style column-major array
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
- `np.ones`
    - returns a new array of specified size, filled with ones
        - signature
    ```
    numpy.ones(shape, dtype=float, order='C')
    ```
    - parameters
        - `shape`: int, or tuple of int. shape of an empty array
        - `dtype`: desired output data type. *Optional*
        - `order`: `'C'` for `C`-style row-major array, `'F'` for `FORTRAN`-style column-major array
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

### 🌱 Array from Existing Data

- `np.array`
- `np.asarray`
    - similar to `numpy.array` except for the fact that it has fewer parameters. This routine is useful for converting `Python` sequence into `ndarray`
    - signature
    ```
    numpy.asarray(a, dtype=None, order=None)
    ```
    - parameters
        - `a`: Input data in any form such as list, list of tuples, tuples, tuple of tuples or tuple of lists
        - `dtype`: *Optional*. Desired data type of array. By default, data type of input data is applied
        - `copy`: *Optional*. By default (true), the object is copied 
        - `order`: `C` (`C`-style, row major) or `F` (`FORTRAN`-style, column major) or `A` (any) (default)
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
- `np.frombuffer`
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
- `np.fromiter`
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

- `np.arange`
    - This function returns an `ndarray` object containing evenly spaced values within a given range
    - signature
    ```
    numpy.arange(start, stop, step, dtype)
    ```
    - parameters
        - `start`: *Optional*. The start of an interval. If omitted, defaults to `0`
        - `end`: The end of an interval (**NOT** including this number)
        - `step`: Spacing between values, default is `1`
        - `dtype`: Data type of resulting ndarray. If not given, data type of input is used
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
- `np.linspace`
    - This function is similar to `numpy.arange` function. In this function, instead of step size, the *number* of evenly spaced values between the interval is specified
    - signature
    ```
    numpy.linspace(start, stop, num, endpoint, retstep, dtype)
    ```
    - parameters
        - `start`: The starting value of the sequence
        - `stop`: The end value of the sequence, included in the sequence if endpoint set to `True`
        - `num`: The number of evenly spaced samples to be generated. Default is `50`
        - `endpoint`: `True` by default, hence the stop value is included in the sequence. If `False`, it is **NOT** included
        - `retstep`: If `True`, returns a tuple of (samples, step between the consecutive numbers)
        - `dtype`: Data type of output `ndarray`
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
- `np.logspace`
    - This function returns an `ndarray` object that contains the numbers that are evenly spaced on a log scale. `start` and `stop` endpoints of the scale are indices of the base, usually `10`
    - signature
    ```
    numpy.logscale(start, stop, num, endpoint, base, dtype)
    ```
    - parameters
        - `start`: The starting point of the sequence is basestart
        - `stop`: The final value of sequence is basestop
        - `num`: The number of values between the range. Default is `50`
        - `endpoint`: If `True`, `stop` is the last value in the range
        - `base`: Base of log space, default is `10`
        - `dtype`: Data type of output array. If not given, it depends upon other input arguments
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
    - Each `np.newaxis` object in the selection tuple serves to expand the dimensions of the resulting selection by one unit-length dimension. The added dimension is the position of the newaxis object in the selection tuple
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
    - a *tuple* having `>= 1` *sequence object* or `ndarray` (whose `dtype` is `int` or `bool`)
- Warning
    - `x[(1, 2, 3), ]` is different from `x[(1, 2, 3)]`. The former triggers *advanced indexing*, while the latter one equals to `x[1, 2, 3]` and triggers *basic indexing*
    - Also recognize that `x[[1, 2, 3]]` will trigger *advanced indexing*, whereas due to the deprecated Numeric compatibility mentioned above, `x[[1, 2, slice(None)]]` will trigger *basic slicing* 
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
    - Example 1: From each row, a specific element should be selected. The row index is just `[0, 1, 2]` and the column index specifies the element to choose for the corresponding row, here `[0, 1, 0]`
    ```
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> x[[0, 1, 2], [0, 1, 0]]
    [1 4 5]
    ```
    - Example 2: from a `4 * 3` array the corner elements should be selected using advanced indexing. Thus all elements for which the column is one of `[0, 2]` and the row is one of `[0, 3]` need to be selected. To use advanced indexing one needs to select all elements explicitly.
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
    - Example 2.1: we can make use of *broadcasting* to generate `rows`, `cols` from simpler `ndarray`s
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
- Combining Advanced And Basic Indexing
    - Example 3: Advanced and basic indexing can be combined by using one *slice* `:` or *ellipsis* `...` with an index array. The following example uses slice for row and advanced index for column. The result is the same when slice is used for both. But advanced index results in copy and may have different memory layout.
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
    - Occurs when `obj` is array object of `Boolean` type, such as may be returned from comparison operators
    - A single boolean index array is practically identical to `x[obj.nonzero()]` where, as described above, `obj.nonzero()` returns a tuple (of length `obj.ndim`) of integer index arrays showing the `True` elements of `obj`. However, it is faster when `obj.shape == x.shape`. 
        - `np.nonzero`
            - `numpy.nonzero(a)` will return a tuple of `a.ndim` `ndarray`s, let `r_1, ..., r_N`. Then, `a`'s `i-th` non-zero element (counted in row-major, `C`-style order) is `a[r_1[i], ..., r_N[i]]`. 
            - Offical expression: Returns a tuple of arrays, one for each dimension of `a`, containing the indices of the non-zero elements in that dimension. The values in `a` are always tested and returned in row-major, `C`-style order.
    - If `obj.ndim == x.ndim`, `x[obj]` returns a 1D array filled with the elements of `x` corresponding to the `True` values of `obj`. The search order will be row-major, `C`-style. If `obj` has `True` values at entries that are outside of the bounds of `x`, then an index error will be raised. If `obj` is smaller than `x` it is identical to filling it with `False`.
    - Example 4: In this example, items greater than 5 are returned as a result of Boolean indexing
    ```
    >>> x = np.array([[ 0,  1,  2], 
                      [ 3,  4,  5], 
                      [ 6,  7,  8], 
                      [ 9, 10, 11]])
    >>> x[x > 5]
    [6 7 8 9 10 11]
    ```
    - Example 5: In this example, `NaN` (Not a Number) elements are omitted
    ```
    >>> a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
    >>> a[~np.isnan(a)]
    [ 1. 2. 3. 4. 5.]
    ```
    - Example 6: The following example adds a constant to all negative elements
    ```
    >>> x = np.array([1., -1., -2., 3])
    >>> x[x < 0] += 20
    >>> x
    array([  1.,  19.,  18.,   3.])
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



























### 🌱 





















### 🌱 



























