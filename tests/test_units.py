"""Unit-conversion arbiter: round trips, pinned values, backend parity."""

import ineedvalidation as vv
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from zodi import units


class TestRoundTrips:
    @given(mag=floats(min_value=-5.0, max_value=35.0))
    def test_mag_flux_ratio_round_trip(self, mag):
        ratio = units.mag_to_flux_ratio(mag)
        np.testing.assert_allclose(units.flux_ratio_to_mag(ratio), mag, atol=1e-10)

    @given(
        mag=floats(min_value=5.0, max_value=35.0),
        wavelength_nm=floats(min_value=200.0, max_value=20000.0),
    )
    def test_ab_mag_intensity_round_trip(self, mag, wavelength_nm):
        intensity = units.intensity_from_ab_mag_arcsec2(mag, wavelength_nm)
        back = units.ab_mag_arcsec2_from_intensity(intensity, wavelength_nm)
        np.testing.assert_allclose(back, mag, atol=1e-10)


def test_mag_to_flux_ratio_pinned():
    np.testing.assert_allclose(units.mag_to_flux_ratio(22.0), 10.0**-8.8, rtol=1e-12)


def test_power_to_photon_pinned():
    # 1 W at 500 nm: photon energy h c / lambda = 3.97289e-19 J, so
    # 2.5170582837713546e18 photons per second (exact SI constants)
    got = units.power_to_photon_intensity(1.0, 500.0)
    expected = 500e-9 / (units.H_J_S * units.C_M_PER_S)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(got, 2.5170582837713546e18, rtol=1e-12)


def test_intensity_to_mjy_per_sr_pinned_against_astropy():
    # astropy 7.2.2 reference: (1 W m-2 um-1).to(MJy,
    # spectral_density(1.25 um)) = 521193.89874711254
    got = units.intensity_to_mjy_per_sr(1.0, 1.25)
    np.testing.assert_allclose(got, 521193.89874711254, rtol=1e-12)


def test_photon_ratio_factor():
    np.testing.assert_allclose(units.photon_ratio_factor(1000.0, 500.0), 2.0)


def test_ab_zero_point_matches_oke_gunn():
    np.testing.assert_allclose(units.AB_ZERO_POINT_JY, 3630.7805477010028, rtol=1e-12)


def test_arcsec2_per_sr():
    np.testing.assert_allclose(units.ARCSEC2_PER_SR, 4.254517029615221e10, rtol=1e-12)


def test_backend_parity():
    jnp = pytest.importorskip("jax.numpy")
    mag = np.array([18.0, 22.0, 23.0])
    lam = np.array([500.0, 1250.0, 550.0])
    for fn, args in [
        (units.mag_to_flux_ratio, (mag,)),
        (units.ab_mag_arcsec2_from_intensity, (np.array([1e-7, 1e-8, 1e-6]), lam)),
        (units.intensity_to_mjy_per_sr, (np.array([1e-7, 1e-8, 1e-6]), lam / 1000)),
        (units.power_to_photon_intensity, (np.array([1e-7, 1e-8, 1e-6]), lam)),
    ]:
        a = fn(*args)
        b = fn(*(jnp.asarray(x) for x in args))
        # a few ULPs of round-off from transcendental ops and fma fusion
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=0.0)


@vv.case("zodi-units", "code-verification")
def test_constant_copies_and_hwoutils_crosscheck():
    """Fails if private SI copies drift or a conversion constant loses precision.

    Tolerance basis: floating-point. SI copies agree bitwise. For the
    independently evaluated AB exponent and angular conversions, allow
    16 float64 eps relative (3.553e-15), zero absolute tolerance. This
    covers rounding of the decimal 48.6 and the reciprocal/square steps.
    """
    from zodi import exozodi

    assert (
        np.float64(units.C_M_PER_S).tobytes()
        == np.float64(exozodi._C_M_PER_S).tobytes()
    )
    assert np.float64(units.H_J_S).tobytes() == np.float64(exozodi._H_J_S).tobytes()
    pytest.importorskip("hwoutils", reason="optional independent constants cross-check")
    from hwoutils import constants, conversions

    for own, reference in (
        (units.C_M_PER_S, constants.c),
        (units.H_J_S, constants.h),
        (exozodi._KB_J_PER_K, constants.k_B),
        (units.AB_ZERO_POINT_JY, conversions.AB_ZERO_POINT_JY),
        (units.ARCSEC2_PER_SR, constants.rad2arcsec**2),
        (units.ARCSEC2_PER_SR, 1.0 / constants.arcsec2rad**2),
        # 1 Jy/arcsec^2 -> MJy/sr; undo the Jy -> MJy factor.
        (units.ARCSEC2_PER_SR, conversions.jy_per_arcsec2_to_mjy_per_sr(1.0) * 1e6),
    ):
        np.testing.assert_allclose(
            own, reference, rtol=16 * np.finfo(float).eps, atol=0
        )
