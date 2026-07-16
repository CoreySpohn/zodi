"""The dispatch contract: numpy in -> numpy out, jax in -> jax out."""

import numpy as np
import pytest

from zodi._xp import array_namespace


def test_numpy_arrays_dispatch_to_numpy():
    ns = array_namespace(np.ones(3), np.zeros(2))
    assert "numpy" in ns.__name__


def test_scalars_fall_back_to_numpy():
    ns = array_namespace(1.0, 2)
    assert "numpy" in ns.__name__


def test_no_arguments_falls_back_to_numpy():
    ns = array_namespace()
    assert "numpy" in ns.__name__


def test_mixed_scalars_and_arrays_use_the_array():
    ns = array_namespace(1.0, np.ones(3))
    assert "numpy" in ns.__name__


def test_jax_arrays_dispatch_to_jax():
    jnp = pytest.importorskip("jax.numpy")
    ns = array_namespace(jnp.ones(3))
    assert "jax" in ns.__name__
