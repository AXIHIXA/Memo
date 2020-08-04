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
        - Type of data (integer, float or Python object)
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
    
    >>> a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype=student)
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
>>> a = np.asarray(x)
>>> a
[1 2 3]

>>> # dtype is set
>>> a = np.asarray(x, dtype=float)
>>> a
[ 1. 2. 3.]

>>> # ndarray from tuple
>>> x = (1, 2, 3)
>>> a = np.asarray(x)
>>> a
[1 2 3]

>>> # ndarray from list of tuples
>>> x = [(1, 2, 3), (4, 5)]
>>> a = np.asarray(x)
>>> a
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
>>> x = np.fromiter(it, dtype=float)
>>> x
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

### 🌱 Indexing & Slicing

- Contents of `ndarray` object can be accessed and modified by *indexing* or *slicing* 
    - items in `ndarray` object follows *zero-based index* 
    - three types of indexing methods
        - *field access* 
        - *basic slicing* 
        - *advanced indexing* 

























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



























### 🌱 





















### 🌱 



























