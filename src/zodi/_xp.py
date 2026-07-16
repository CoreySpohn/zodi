"""Array-backend dispatch for the single-source numpy/JAX core.

Every public zodi function computes through the array API namespace of its
array arguments: numpy inputs run pure numpy and return numpy arrays, jax
inputs trace natively (tracers pass the dispatch, so functions jit, grad,
and vmap without special casing), and plain Python scalars fall back to
numpy. The library never imports jax; the namespace is picked up
automatically when jax arrays flow in.
"""

import array_api_compat
import numpy as np

__all__ = ["array_namespace"]

_NUMPY_XP = array_api_compat.array_namespace(np.asarray(0.0))


def array_namespace(*args):
    """Return the array API namespace shared by the array arguments.

    numpy arrays are treated as backend-neutral static data: when the
    arguments mix numpy arrays with arrays from one other backend (for
    example a numpy wavelength grid alongside jax spectra), the other
    backend wins and the numpy inputs are lifted into it with ``asarray``
    by the caller. Arrays from two non-numpy backends still raise.

    Args:
        *args: Candidate inputs; anything with a ``shape`` attribute is
            treated as an array, and plain Python scalars are ignored.

    Returns:
        The ``array_api_compat`` namespace of the array arguments, or the
        numpy namespace when none of the arguments is an array.
    """
    arrays = [a for a in args if hasattr(a, "shape")]
    if not arrays:
        return _NUMPY_XP
    try:
        return array_api_compat.array_namespace(*arrays)
    except TypeError:
        foreign = [a for a in arrays if not isinstance(a, (np.ndarray, np.generic))]
        if not foreign:
            raise
        return array_api_compat.array_namespace(*foreign)
