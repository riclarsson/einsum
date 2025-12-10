# einsum
Header-file only [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
for std::mdspan and std::ranges.  Beyond the standard summation, it also supports transformations.

The header requires C++26 or later, but might need modification.  It is implemented against
the [reference implementation by the Kokkos team](https://github.com/kokkos/mdspan).
The required features are an implementation of `std::mdspan` complete with `std::submdspan`.

# How to use

There are multiple ways to use this library.  Below are some examples ranging from trivial to useful.
In all these examples, `x` is a `double`, `a` and `b` and `c` are `std::vector<double>`,
and `A` and `B` and `C` are `std::mdspan` of a `double` with rank 2.

Note that the size and shapes of the sized parameters are set before the call to the methods.

## Summing

```cpp
ein::sum<"", "i">(x, a);  // x is sum(a)
```

## Copying

```cpp
ein::sum<"i", "i">(b, a);  // b = a;
```

## Transform

```cpp
ein::sum<"i", "i">(b, a | std::views::transform([](auto x) {
  return std::exp(-x);
}));  // b = exp(-a);
```

## Matrix-Vector multiplication

```cpp
ein::sum<"i", "ij", "j">(b, A, a);  // b = A * a;
```

## Matrix-Vector multiplication transform

```cpp
ein::sum<"i", "ij", "j">(b, A, a | std::views::transform([](auto x) {
  return std::exp(-x);
}));  // b = A * exp(-a);
```

## Matrix-Matrix multiplication

```cpp
ein::sum<"pi", "im", "pm">(C, A, B);  // C = A * B;
```

## Complicated equation

```cpp
ein::tra<"i", "i", "i", "ji", "">(c, [](auto ai, auto bi, auto ATi) {
  double sumATi{};
  ein::sum<"", "j">(sumATi, ATi);
  return ai * sumATi + bi / sumATi;
} a, b, A);  // c[i] = a[i] * sum(A.T[i]) + b[i] / sum(A.T[i]);
```

# Fixes that might be required before using the header

The implementation is towards the [reference implementation by the Kokkos team](https://github.com/kokkos/mdspan).
This defines `std::mdspan` and `std::submdspan` under `experimental/mdspan`.  You should must change the `#include <experimental/mdspan>`
to point either at the true `<mdspan>` or set up the path to a compatible reference implementation and include that properly.

# Origin and differences

This library is mostly based on the [numpy.einsum](https://numpy.org/devdocs/reference/generated/numpy.einsum.html) operations.
There are a few changes in notation and two severe limitations.

## Notation change

The change is in the way the notation is expressed.
For instance, the way matrix multiplication is written in `numpy` is:

```python
from numpy import einsum
...
c = einsum('ij,jh->ih', A, B)
```

This library writes the same operations as

```cpp
#include <ein.h>
...
ein::sum<"ih", "ij", "jh">(C, A, B);
```

## No type conversion or creation - sizes must be pre-set

The first limitation is that the `numpy` approach allows any numpy-compatible types, however the C++ header only allows `std::mdspan`
for multi-dimensional arrays, and any type with `a[std::size_t{}]`-style index access operator that returns an integer representing
the size when passed to `std::ranges::size()`.  If you want to use unlimited ranges, ensure that they are defined so that the
size of the index is derived from other inputs earlier in the call-chain.

## No extension to all dimensions

The second limitation is that the `numpy` approach allows `...` notation to indicate that the operation is repeated.
The C++ header requires the indices for all dimensions to be set.

# How to optimize

It is recommended to have a look at the `sum_` internal method if you need optimizations of some common patterns.
It is quite trivial to add things like matrix-matrix and matrix-vector multiplication routines that use Lapack in there.
You can see [my original implementation here](https://github.com/atmtools/arts/blob/main/src/core/matpack/matpack_einsum.h),
which makes use of such optimizations (and have extended the type system to also support `Eigen3`, among other things).
