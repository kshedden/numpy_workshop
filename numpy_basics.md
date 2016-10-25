## Numpy data structures

The main numpy data structure is the
[ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html),
which stands for "n-dimensional array".  There are some other data
structures in Numpy such as the
[matrix](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html),
and the
[recarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html).
But these are mainly used in very narrow situations.  Here we focus
only on `ndarray`s.

First, since numpy is a library we need to import it.

```
import numpy as np
```

An ndarray is a homogeneous rectangular data structure with an
arbitrary number of axes. Being "homogeneous" means all data values in
an `ndarray` must have the same data type. Numpy supports many data
types. In the following cell, we create a 1-dimensional literal
`ndarray` with double precision floating point (8 byte float) values:

```
x = np.asarray([4, 1, 5, 4, 7, 3, 0], dtype=np.float64)
```

Since the data are all integers, we could have used an integer data type instead:

```
x = np.asarray([4, 1, 5, 4, 7, 3, 0], dtype=np.int64)
```

We can even store them as single byte values, since none of the values
exceeds 255:

```
x = np.asarray([4, 1, 5, 4, 7, 3, 0], dtype=np.uint8)
```

We can index and slice an `ndarray` just like we index and slice a
Python list:

```
w = x[2]
z = x[3:5]
```

In addition, `ndarray`s support two types of indexing that core Python
lists do not. We can index with a Boolean array:

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
are conformable (or can be broadcast to be conformable, but that is a
more advanced topic). Note that numerical types are "upcast" (use
`z.dtype` to get the type of z).

```
y = np.asarray([0, 1, 0, -1, 1, 1, -2], dtype=np.float64)
z = x + y
```

An `ndarray` can have multiple axes (dimensions):

```
x = np.zeros((4, 3))
```

```
x = np.zeros((4, 3, 2))
```

Slicing ndarrays with multiple dimensions is straightforward:

```
x = np.random.normal(size=(3, 4))
x[1, :]
x[1:3, 2:4]
```
