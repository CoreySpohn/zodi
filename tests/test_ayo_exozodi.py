"""Analytic AYO exozodi units and projected-separation law."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("equinox")
jnp = jax.numpy

from zodi.jax import AYOExozodi  # noqa: E402


def test_analytic_radiance_units():
    """Check AB normalization and photon units from SI primitives.

    A 1e-12 relative tolerance allows float64 unit arithmetic.
    """
    waves = jnp.array([450.0, 550.0, 750.0])
    zero_jy = 10 ** (-0.4 * 48.60) * 1e4 / 1e7 / 1e-26
    expected = zero_jy * 1e-26 / (6.62607015e-34 * np.asarray(waves))
    expected *= 3 * 10 ** (-0.4 * 22) / (np.pi / (180 * 3600)) ** 2
    np.testing.assert_allclose(
        AYOExozodi().photon_radiance(waves, separation_au=1.0),
        expected,
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        AYOExozodi(6.0, 23.0).photon_radiance(waves, separation_au=1.0),
        expected * 2 * 10 ** (-0.4),
        rtol=1e-12,
        atol=0,
    )


def test_projected_separation_law():
    """Check r^-2 with physical separation = distance_pc * angle_arcsec.

    Exact powers of two set the ratios; 1e-12 allows JIT rounding.
    """
    model = AYOExozodi()
    distances = jnp.array([10.0, 20.0, 10.0, 20.0])
    separations = jnp.array([0.1, 0.1, 0.2, 0.05])

    def read(distance, separation):
        return model.photon_radiance(550.0, separation_au=distance * separation)

    base = (
        10 ** (-0.4 * 48.60)
        * 1e-3
        / (6.62607015e-34 * 550)
        * 3
        * 10 ** (-0.4 * 22)
        / (np.pi / (180 * 3600)) ** 2
    )
    expected = jnp.array([1.0, 0.25, 0.25, 1.0]) * base
    np.testing.assert_allclose(
        jax.vmap(read)(distances, separations), expected, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(read))(distances, separations),
        expected,
        rtol=1e-12,
        atol=0,
    )
