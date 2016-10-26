## Numpy data structures

The main numpy data structure is the
[ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html),
which stands for "n-dimensional array".  There are some other data
structures in Numpy such as the
[matrix](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html),
and the
[recarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html).
But these are mainly used in relatively narrow situations.  Here we
focus only on the widely-useful ndarray data structure.

First, since numpy is a library we need to import it.

```
import numpy as np
```

### Ndarray construction and dtypes

An ndarray is a homogeneous rectangular data structure with an
arbitrary number of axes. Being "homogeneous" means that all data
values in an ndarray must have the same data type. Numpy supports many
data types. Below, we create a 1-dimensional literal ndarray with
double precision floating point (8 byte float) values:

```
x = np.asarray([4, 1, 5, 4, 7, 3, 0], dtype=np.float64)
```

Since the data are all integers, we could have used an integer data type instead:

```
x = np.asarray([4, 1, 5, 4, 7, 3, 0], dtype=np.int64)
```

We could even store the data as single byte values, since none of the
values exceeds 255:

```
x = np.asarray([4, 1, 5, 4, 7, 3, 0], dtype=np.uint8)
```

An ndarray can have multiple axes (dimensions):

```
x = np.zeros((4, 3))

y = np.zeros((4, 3, 2))

z = np.asarray([[1, 2, 3], [2, 3, 4]])
```

To determine the shape of an ndarray, use the shape attribute:

```
x = np.random.normal(size=(3, 2))
x.shape # returns a tuple (3, 2)
```

### Combining and reshaping

The concatenate function concatenates arrays either horizontally or vertically:

```
x = np.random.normal(size=(5, 2))
y = np.random.normal(size=(7, 2))
z = np.concatenate((x, y), axis=0)

x = np.random.normal(size=(5, 2))
y = np.random.normal(size=(5, 3))
z = np.concatenate((x, y), axis=1)
```

Note that concatenate preserves the number of axes, i.e. in the
examples above, we combine arrays with two axes, and the result also
has two axes.  If you want to "stack" ndarray objects to create a new
object with one more axis than the things being combined, use one of
the stack functions:

```
w = np.vstack((np.zeros(5), np.ones(5)))

z = np.dstack((np.zeros(5), np.ones(5)))
```

We can create a new array with the same data as another array, but
with a different shape.  The new array must have the same total number
of elements as the array it is reshaped from.  The reshaped array is
filled in row-wise (i.e. the last index moves fastest):

```
x = np.arange(10)
x = np.reshape(x, (5, 2))
```

See also the tile and repeat functions for additional ways to
construct larger arrays out of components.

### Indexing and slicing

We can index and slice an ndarray just like we index and slice a
Python list.  But note that the index positions start at 0, not 1.

```
w = x[2]
z = x[3:5]
```

In addition, an ndarray supports two types of indexing that core
Python lists do not. We can index with a Boolean array:

```
ii = np.asarray([False, False, True, False, True, False, False])
z = x[ii]
```

We can also index using a list of positions:

```
ix = np.asarray([0, 3, 3, 5])
z = x[ix]
```

Boolean and position-based indexing can be used to select elements
from an ndarray with some desired characteristic.  For example, if we
want to retain only the positive elements of an ndarray, we can use
either of the following approaches:


```
z = z[z> 0]

ix = np.flatnonzero(z > 0)
z = z[ix]
```

Slicing an ndarray with multiple dimensions is straightforward:

```
x = np.random.normal(size=(3, 4))
w = x[1, :]
z = x[1:3, 2:4]
```

We can also use boolean and index vectors with multidimensional arrays:

```
x = np.random.normal(size=(10, 3))
b = x[:, 0] > 0
z = x[b, :]

ix = np.flatnonzero(x[:, 0] > 0)
z = x[ix, :]
```

## Functions and methods in Numpy

Some of the functionality of Numpy is implemented as functions, and
some of it is implemented as methods.  Some commonly-used operations
are implemented as both functions and as methods:

```
x = np.random.normal(size=(4, 3))

s1 = np.sum(x) # a function that sums all elements of the array

s2 = x.sum()   # a method that does the same thing
```

## Reducing (summarization) methods

A reducing operation reduces the number of axes of an array, by
applying a function along one axis that combines the values. Here are
some basic summarization operations along the columns:

```
x = np.random.normal(size=(4, 3))

s = x.sum(0)
m = x.max(0)
n = x.min(0)
p = x.prod(0)
```

The results of all these operations will have shape (3,).  The
argument 0 to the method implies that the input array (x) is reduced
along axis 0 (the first axis).  We can also reduce by applying a
function along the rows:

```
s = x.sum(1)
m = x.max(1)
n = x.min(1)
p = x.prod(1)
```

These values all have shape (4,).

Finally, we can reduce the entire array to a scalar:

```
s = x.sum()
m = x.max()
n = x.min()
p = x.prod()
```

Many of the reducing methods have function analogues:

```
s = np.sum(x, 1)
m = np.max(x, 1)
n = np.min(x, 1)
p = np.prod(x, 1)
```

But some reducing methods only have the function form:

```
q = ma.median(x, 1) # there is no x.median(1)
```

## Basic arithmetic

The standard arithmetic operators (+, -, *, /, **, %) in Numpy all
behave in a pointwise (element-wise) fashion.  The most typical use of
these operators is to combine two ndarray objects that have the same
shape, producing a result that has the same shape as the two operands:

```
x = np.random.normal(size=(4, 2))
y = np.random.normal(size=(4, 2))
z = x + y
```

Note that these operations all have an in-place version that may
have slightly better efficiency.

```
x = np.random.normal(size=(4, 2))
y = np.random.normal(size=(4, 2))
x += y
```

## Broadcasting

Numpy also supports a form of *broadcasting*, allowing objects with
different shapes to be combined in limited situations.  The most
common use-cases for broadcasting are to center or standardize rows or
columns of an ndarray.

```
x = np.random.normal(size=(100, 5))

# Center the columns
xc = x - x.mean(0)

# Standardize the columns
xs = x / x.std(0)

# Center the rows
xc = x - x.mean(1)[:, None]

# Standardize the rows
xs = x / x.std(0)[:, None]
```

Let's try to understand how this works.  The basic rule for
broadcasting is that two arrays can be combined if their shapes agree
in their trailing axes.  So the following pairs of shapes agree and
can be broadcast together:

```
(8, 6)        (6,)
(5, 4)        (4,)
(3, 2, 3)     (2, 3)
```

while the following pairs of shapes do not agree and cannot be
broadcast together:

```
(8, 6)        (8,)
(5, 4)        (6, 4)
(3, 2, 3)     (3, 2, 2)
```

When broadcasting is allowed, the smaller array is repeated to match
the shape of the larger array.

Broadcasting is allowed in one additional situation not discussed
above.  If an axis has length 1, it matches anything.  So for example,
the following shapes match:

```
(8, 6)   (1, 6)
(5, 4)   (5, 1)
```

In this situation, the axis with extent 1 is replicated to match the
same axis in the other array.

Now we return to the centering of columns.  If x has shape (m, n),
then x.mean(0) has shape (n,).  Following the broadcasting rules
above, arrays of shape (m, n) and (n,) can be broadcasted.  The effect
of this is that the values in x.mean(0) are subtracted from each row
of x, centering the data by columns.

When centering rows as shown above, we centered the data by
subtracting x.mean(1)[:, None] from x.  The shape of x.mean(1) is (m,)
but this is not broadcastable with an array of shape (m, n) (the two
shapes do not agree in their trailing dimensions).  The trick here is
to add an axis of extent 1 using a "None" indexer: x.mean(1)[:, None]
has shape (m, 1).  The axis with length 1 is then expanded to length n
by replicating the values along the newly-created second dimension.

## Pointwise functions

Many mathematical functions operate pointwise on ndarray objects:

```
x = np.runif((5, 4))

a = np.log(x)
b = np.sqrt(x)
c = np.cos(x)
d = np.exp(x)
```

## Linear algebra

Numpy includes many standard functions for linear algebra.

```
x = np.random.normal(size=(5, 4))

# x' * x (matrix product)
xtx = np.dot(x.T, x)

# The trace
z = np.trace(xtx)

# The singular value decomposition of x
u,s,vt = np.linalg.svd(x, 0)

# Solve a system of equations
y = np.random.normal(size=3)
r = np.linalg.solve(xtx, y)
```

# Sorting

An array can can be sorted in-place with the sort method:

```
x = np.random.normal(size=10)
x.sort()
```

If you want to preserve the original array, you can use the sort
function:

```
x = np.random.normal(size=10)
y = np.sort(x)
```

You can sort multidimensional arrays in-place along a given axis:

```
x = np.random.normal(size=(5, 2))
x.sort(0)
```

If you want to sort indirectly, use the argsort function.  After the
following code is run, x is unchanged, but x[ix] would be identical to
the result of np.sort(x).

```
x = np.random.normal(size=10)
ix = np.argsort(x)
```

## References and copies

Slicing operations sometimes result in a reference into the parent
array.  In this case, since memory is shared between the slice and its
parent, changing the data in the slice also alters the parent object.
For example, in the following code, after changing an element in y,
the state of x is also changed:

```
x = np.arange(10)
y = x
y[3] = 99
```

This can also happen with slices

```
x = np.arange(10)
y = x[3:8]
y[3] = 99
```

A slice with a stride may also return a view:

```
x = np.arange(10)
y = x[2:10:2]
y[3] = 99
```

But an arbitrary selection of values is a copy:

```
x = np.arange(10)
y = x[[0, 3, 4, 5]]
y[3] = 99
```

References are often a useful way to improve performance, or to
simplify the implementation of complex algorithms.  However, if you do
not want a reference, you can force a copy to be made with the copy
method:

```
x = np.arange(10)
y = x[3:8].copy()
y[3] = 99
```

## Internal structure of ndarray objects

The canonical state of an ndarray is a contiguous block of memory with
the data packed consecutively.  This means that the "stride" (the
number of bytes from the start of one element to the next) is the same
as the "itemsize" (the number of bytes used to store each element).
You can get some information about the memory layout of an array via
some of its attributes:

```
x = np.arange(5)

x.dtype
x.itemsize
x.strides
x.flags
```

If the array was obtained by slicing or reshaping (e.g. transposing)
another array, it may not have this "canonical" layout.  For example,
compare the "strides" attribute of x and y below:

```
x = np.arange(10)
y = x[::2]
```

Strides can also be used to allow a reshaped array (e.g. a transpose)
to share memory with the array it was obtained from:

```
x = np.random.normal(size=(3, 2))
y = x.T
```

## Additional topics

* Counting with bincount and unique

* Time, string and object dtypes

* Searching with searchsorted

* Set operations

* Data type reinterpreation with "view"

* Output parameters

* Einstein summation

## Limitations and future directions

Numpy is arguably the "best in class" for what it is: an interpreted
array processing language that uses contiguous blocks of memory to
store array data.  Compared to other Python array libraries, and
compared to the array processing capabilities in other languages,
Numpy is exceedingly powerful in terms of the range of data types that
it supports, its flexible broadcasting and reshaping capabilities, and
its use of strides and other flexible indexing models to permit
complex operations with minimal data copying and indexing overhead.

However there are some fundamental limitations to Numpy's design, and
in recent years there has been a lot of interest in devising the next
generation or successor to Numpy.  Two main limitations of Numpy are
commonly noted.  One limitation is that since arrays are stored in
contiguous memory chunks, Numpy cannot easily handle very large
arrays.  There are various work-arounds to address this, but it is a
significant problem.

The other major limitation of Numpy's design is that since the code is
executed partially by the Python interpreter and partially by the
Numpy library, it is hard to avoid unecessary copies of data.  For
example, when executing the code below, a new allocation is made to
store the result of x + y, then this new allocation is assigned to x.
The original allocation of x looses a reference count and can be
garbage collected, but it would be better to recycle the original
memory allocation of x and avoid the extra allocation entirely.

```
x = x + y
```

A number of powerful tools have been developed to either augment or
supplant Numpy in numeric data processing.  For example, to address
the issue of excess data copying, the
[Numexpr](https://github.com/pydata/numexpr) package takes expressions
written as strings and evaluates them in a virtual machine that can
apply a variety of accelerations.  For example, by passing "sum(x *
y)" to the Numexpr virtual machine, it can be automatically be
determined that the sum can be calculated in streaming fashion without
explicitly forming x * y.

Other relevant projects include [Numba](http://numba.pydata.org/),
which uses just-in-time compilation to bypass the Python interpreter,
[Cython](http://cython.org), which uses an extended Python-like
language to permit compilation of code to C,
[Dask](http://dask.pydata.org/en/latest/), which defines array-like
data containers that can live in either primary memory or on-disk, and
[Theano](http://deeplearning.net/software/theano/), which generates
code for doing array processing on GPUs (among other things).
Finally, [bcolz](https://github.com/Blosc/bcolz) is a column-oriented
compressed data container, [Pytables](http://www.pytables.org) is an
indexed on-disk data container, and [Pandas](http://pandas.pydata.org)
is a powerful in-memory toolkit for working with inhomogeneous "data
frames".

