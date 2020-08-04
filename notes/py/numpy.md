# `NumPy` Tutorial Notes

- Notes of reading:
    - [`Numpy User Guide`](https://numpy.org/doc/1.18/numpy-user.pdf)
    - [`Numpy Tutorial - Tutorialspoint`](https://www.tutorialspoint.com/numpy/index.htm)

### 🌱 `ndarray` Object

- `ndarray` describes the collection of items of the *same type*
    - Items in the collection can be accessed using a zero-based index
    - Every item in an ndarray takes the same size of block in the memory
    - Each element in `ndarray` is an object of data-type object (called `dtype`)
    - Any item extracted from `ndarray` object (by *slicing*) is represented by a `Python` object of one of array scalar types

![](https://www.tutorialspoint.com/numpy/images/ndarray.jpg)

- The basic ndarray is created using an `array` function in `NumPy`, creating an `ndarray` from any object exposing array interface, or from any method that returns an array
    - signature
    ```
    np.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)
    ```
    - parameters
        - `object`: Any object exposing the array interface method returns an array, or any (nested) sequence
        - `dtype`: *Optional*. Desired data type of array
        - `copy`: *Optional*. By default (true), the object is copied 
        - `order`: `C` (`C`-style, row major) or `F` (`Fortan / MATLAB`-style, column major) or `A` (any) (default)
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
        - In case of structured type, the names of fields, data type of each field and part of the memory block taken by each field
        - If data type is a subarray, its shape and data type
    - Byte order is decided by prefixing `<` (little-endian) or `>` (big-endian) to data type
    - `dtype` object construction
        - constructor signature
        ```
        np.dtype(object, align, copy)
        ```
        - paramters:
            - `object`: To be converted to data type object
            - `align`: If `True`, adds padding to the field to make it similar to `C` `struct`
            - `copy`: Makes a new copy of `dtype` object. If `False`, the result is reference to builtin data type object
    ```
    >>> # using array-scalar type
    >>> import numpy as np
    >>> dt = np.dtype(np.int32)
    >>> dt
    int32
    
    >>> #int8, int16, int32, int64 can be replaced by equivalent string 'i1', 'i2','i4', etc.
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
    >>> dt = np.dtype([('age',np.int8)])
    >>> a = np.array([(10,),(20,),(30,)], dtype=dt)
    >>> a
    [(10,) (20,) (30,)]
    ```














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





















### 🌱 



























### 🌱 





















### 🌱 



























### 🌱 





















### 🌱 



























