"""JAX spectrum helper and AB flux normalization."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("equinox")
jnp = jax.numpy

from zodi.jax.leinert import (  # noqa: E402
    ayo_default_zodi_flux_jy,
    ayo_default_zodi_mag,
    create_zodi_spectrum_jax,
    leinert_zodi_factor,
    leinert_zodi_mag,
    leinert_zodi_spectral_radiance,
    zodi_color_correction,
)
from zodi.units import AB_ZERO_POINT_JY, C_M_PER_S  # noqa: E402


def test_ayo_default_flux_matches_spectrum_helper():
    """Fails if spectrum normalization or AB color differs; float64 tolerance."""
    wavelengths = jnp.array([500.0, 910.0])
    spectrum = create_zodi_spectrum_jax(wavelengths)
    expected = jnp.array([float(ayo_default_zodi_flux_jy(w)) for w in wavelengths])
    np.testing.assert_allclose(spectrum, expected, rtol=3e-14, atol=0)


def test_leinert_helper_units_and_argument_order():
    """Fails if helper defaults, angle order, or spectral units change.

    Expected values use Table 17's (135, 0) cell 179/259 and Table 19's
    500 nm radiance 2.6e-6. Tolerance basis: floating-point (3e-14) for
    logarithmic interpolation and AB conversion, exact table ratios.
    """
    factor = 179.0 / 259.0
    power = 2.6e-6 * factor
    flux_jy = power * (np.pi / (180 * 3600)) ** 2 * 1e6 * (500e-9) ** 2
    flux_jy /= C_M_PER_S * 1e-26
    mag = -2.5 * np.log10(flux_jy / AB_ZERO_POINT_JY)
    np.testing.assert_allclose(leinert_zodi_factor(0.0), factor, rtol=3e-14)
    np.testing.assert_allclose(leinert_zodi_spectral_radiance(500.0), power, rtol=3e-14)
    np.testing.assert_allclose(leinert_zodi_mag(500.0), mag, rtol=3e-14)
    np.testing.assert_allclose(ayo_default_zodi_mag(550.0), 22.0, rtol=3e-14)
    np.testing.assert_allclose(
        zodi_color_correction(700.0, 500.0, False), 2.0e-6 / 2.6e-6, rtol=3e-14
    )
    np.testing.assert_allclose(
        zodi_color_correction(700.0, 500.0),
        2.0e-6 / 2.6e-6 * 700.0 / 500.0,
        rtol=3e-14,
    )
    np.testing.assert_allclose(
        create_zodi_spectrum_jax(jnp.array([500.0, 700.0]), 2.0, 500.0),
        2.0 * np.array([1.0, 2.0e-6 / 2.6e-6 * (700.0 / 500.0) ** 2]),
        rtol=3e-14,
    )
