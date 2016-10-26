## Numpy data structures

The main numpy data structure is the
[ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html),
which stands for "n-dimensional array".  There are some other data
structures in Numpy such as the
[matrix](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html),
and the
[recarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html).
But these are mainly used in very narrow situations.  Here we focus
only on the widely-used ndarray type.

First, since numpy is a library we need to import it.

```
import numpy as np
```

### Construction and dtypes

An ndarray is a homogeneous rectangular data structure with an
arbitrary number of axes. Being "homogeneous" means that all data
values in an ndarray must have the same data type. Numpy supports many
data types. In the following cell, we create a 1-dimensional literal
ndarray with double precision floating point (8 byte float) values:

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
```

```
x = np.zeros((4, 3, 2))
```

To determine the shape of an ndarray, use the shape attribute:

```
x = np.random.normal(size=(3, 2))
x.shape # returns a tuple (3, 2)
```

### Indexing and slicing

We can index and slice an ndarray just like we index and slice a
Python list:

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

We can do elementwise arithmetic using numpy arrays as long as they
are conformable (or can be broadcast to be conformable, as discussed
below). Note that numerical types are "upcast" (in the example below,
use z.dtype to get the type of z).

```
y = np.asarray([0, 1, 0, -1, 1, 1, -2], dtype=np.float64)
z = x + y
```

Slicing an ndarray with multiple dimensions is straightforward:

```
x = np.random.normal(size=(3, 4))
w = x[1, :]
z = x[1:3, 2:4]
```

## Functions and methods in Numpy

Some of the functionality of Numpy is implemented as functions, and
some of it is implemented as methods.  Some operations are implemented
as both functions and as methods:

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
q = md.median(x, 1) # there is no x.median(1)
```

## Broadcasting

The standard arithmetic operators (+, -, *, /, **, %) in Numpy all
behave in a pointwise (element-wise) fashion.  The most basic use of
these operators is to combine two ndarray objects with the same shape.

```
x = np.random.normal(size=(4, 2))
y = np.random.normal(size=(4, 2))
z = x + y
```

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
x.mean(0) has shape (n,).  Following the broadcasting rules above,
arrays of shape (m, n) and (n,) can be broadcasted.  The effect of
this is that the values in x.mean(0) are subtracted from each row of
x, centering the data by columns.

When centering columns as shown above, we centered the data by
subtracting x.mean(1)[:, None] from x.  The shape of x.mean(1) is (m,)
but this is not broadcastable with (m, n) (they do not agree in the
training dimensions).  The trick here is to add an axis of extent 1:
x.mean(1)[:, None] has shape (m, 1).  The axis with length 1 is
expanded to length n by replicating the values along the second
dimension.

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