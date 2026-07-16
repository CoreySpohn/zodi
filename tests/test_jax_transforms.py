"""jit, vmap, and grad through the shared interpolation (jax leg only)."""

import numpy as np
import pytest

from zodi._interp import interp2d_bilinear_clamped

jax = pytest.importorskip("jax")
jnp = jax.numpy

XG = np.linspace(0.0, 180.0, 19)
YG = np.linspace(0.0, 90.0, 10)
TABLE = (
    1e-7
    * (1.0 + np.cos(np.deg2rad(XG))[:, None] ** 2)
    * (1.0 + 0.5 * np.sin(np.deg2rad(YG))[None, :])
)
X = np.array([12.3, 88.0, 179.9, 250.0])  # last one clamps
Y = np.array([5.0, 42.0, 89.0, 95.0])


def _f(x, y):
    # numpy-static grids + traced queries: the tabulated-physics pattern
    return interp2d_bilinear_clamped(XG, YG, TABLE, x, y)


def test_jit_matches_eager_exactly():
    eager = _f(jnp.asarray(X), jnp.asarray(Y))
    jitted = jax.jit(_f)(jnp.asarray(X), jnp.asarray(Y))
    np.testing.assert_allclose(
        np.asarray(eager), np.asarray(jitted), rtol=1e-13, atol=0.0
    )


def test_jit_with_static_numpy_tables_matches_numpy():
    plain = _f(X, Y)
    jitted = jax.jit(_f)(jnp.asarray(X), jnp.asarray(Y))
    np.testing.assert_allclose(plain, np.asarray(jitted), rtol=1e-13, atol=0.0)


def test_vmap_matches_eager_exactly():
    eager = _f(jnp.asarray(X), jnp.asarray(Y))
    mapped = jax.vmap(_f)(jnp.asarray(X), jnp.asarray(Y))
    np.testing.assert_allclose(
        np.asarray(eager), np.asarray(mapped), rtol=1e-13, atol=0.0
    )


def test_grad_matches_central_difference():
    g = jax.grad(lambda x: _f(x, jnp.asarray(42.0)))(jnp.asarray(55.0))
    eps = 1e-6
    fd = (_f(55.0 + eps, 42.0) - _f(55.0 - eps, 42.0)) / (2 * eps)
    assert abs(float(g) - float(fd)) / abs(float(fd)) < 1e-6
