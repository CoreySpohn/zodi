"""Zodiacal and exozodiacal light brightness conventions.

One implementation serves numpy and JAX through the array API standard:
every function computes in the namespace of its array inputs, so numpy
callers stay numpy end to end while JAX callers get functions that jit,
vmap, and differentiate natively. The public physics API (local zodiacal
brightness, exozodi conventions, population quantile functions, and unit
conversions) lands with v0.1; the dispatch and interpolation machinery
underneath it is in place and tested.
"""

__all__: list[str] = []
