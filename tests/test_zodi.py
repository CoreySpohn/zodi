"""Local zodi model: table reproduction, conventions, and invariants."""

import numpy as np
import pytest

from zodi import tables, units, zodi


def test_position_factor_reference_is_one():
    np.testing.assert_allclose(zodi.position_factor(90.0, 0.0), 1.0, rtol=1e-14)


def test_position_factor_tabulated_nodes():
    np.testing.assert_allclose(
        zodi.position_factor(60.0, 0.0), 505.0 / 259.0, rtol=1e-14
    )
    np.testing.assert_allclose(
        zodi.position_factor(135.0, 45.0), 90.0 / 259.0, rtol=1e-14
    )


def test_position_factor_symmetry_and_clamping():
    np.testing.assert_allclose(
        zodi.position_factor(-60.0, 30.0), zodi.position_factor(60.0, 30.0)
    )
    np.testing.assert_allclose(
        zodi.position_factor(60.0, -30.0), zodi.position_factor(60.0, 30.0)
    )
    np.testing.assert_allclose(
        zodi.position_factor(200.0, 0.0), zodi.position_factor(180.0, 0.0)
    )


def test_clamp_contract_matches_nearest_tabulated_cell():
    """Out-of-grid queries return the nearest tabulated cell.

    Expected values come from the raw Table 17 and Table 19 arrays, not
    from the interpolator. Tolerance basis: exact-algebra (a clamped
    bilinear query on a node reproduces the node).
    """
    lon = list(tables.LON_GRID_DEG)
    beta = list(tables.BETA_GRID_DEG)
    ref = tables.TABLE17_REFERENCE
    # latitude beyond the pole clamps to 90 deg
    np.testing.assert_allclose(
        zodi.position_factor(60.0, 95.0),
        tables.TABLE17_RAW[lon.index(60.0), beta.index(90.0)] / ref,
        rtol=1e-14,
    )
    # longitude difference beyond 180 deg clamps to the 180 deg row
    np.testing.assert_allclose(
        zodi.position_factor(200.0, 30.0),
        tables.TABLE17_RAW[lon.index(180.0), beta.index(30.0)] / ref,
        rtol=1e-14,
    )
    # near-Sun exclusion cell takes the first valid entry along its row
    row = tables.TABLE17_RAW[lon.index(5.0)]
    first_valid = row[np.isfinite(row)][0]
    np.testing.assert_allclose(
        zodi.position_factor(5.0, 0.0), first_valid / ref, rtol=1e-14
    )
    # wavelengths outside Table 19 clamp to the end knots (0.2 and 140 um)
    np.testing.assert_allclose(
        zodi.color_correction(100.0), zodi.color_correction(200.0), rtol=1e-14
    )
    np.testing.assert_allclose(
        zodi.color_correction(2.0e5), zodi.color_correction(1.4e5), rtol=1e-14
    )


def test_specific_intensity_at_location_absolute_units():
    got = zodi.specific_intensity_at_location(90.0, 0.0)
    np.testing.assert_allclose(got, 259e-8, rtol=1e-14)
    photon = zodi.specific_intensity_at_location(90.0, 0.0, photon_units=True)
    np.testing.assert_allclose(
        photon, units.power_to_photon_intensity(259e-8, 500.0), rtol=1e-14
    )


def test_color_correction_at_knots():
    np.testing.assert_allclose(zodi.color_correction(500.0), 1.0, rtol=1e-14)
    np.testing.assert_allclose(
        zodi.color_correction(1200.0), 8.1e-7 / 2.6e-6, rtol=1e-12
    )
    np.testing.assert_allclose(
        zodi.color_correction(2200.0), 1.7e-7 / 2.6e-6, rtol=1e-12
    )


def test_color_correction_photon_units_factor():
    power = zodi.color_correction(1200.0)
    photon = zodi.color_correction(1200.0, photon_units=True)
    np.testing.assert_allclose(photon / power, 1200.0 / 500.0, rtol=1e-14)


def test_anchor_conventions_differ_by_constant_ratio():
    expected = 2.6e-6 / (tables.TABLE17_REFERENCE * tables.TABLE17_UNIT_W_M2_SR_UM)
    for dlon, beta, lam in [(60.0, 0.0, 700.0), (135.0, 30.0, 1250.0)]:
        t17 = zodi.specific_intensity(dlon, beta, lam, anchor="table17")
        t19 = zodi.specific_intensity(dlon, beta, lam, anchor="table19")
        np.testing.assert_allclose(t19 / t17, expected, rtol=1e-12)


def test_specific_intensity_table19_anchor_reproduces_reference_spectrum():
    # at the reference geometry the table19 anchor IS Table 19
    got = zodi.specific_intensity(90.0, 0.0, 1200.0, anchor="table19")
    np.testing.assert_allclose(got, 8.1e-7, rtol=1e-12)


def test_pinned_against_skyscapes_leinert_x64():
    # References computed from skyscapes.background.leinert under jax x64
    # (same tables, same linear log-log interpolation, same AB zero point):
    #   leinert_zodi_mag(550, 0, 135)  = 22.406737315862767
    #   leinert_zodi_mag(550, 30, 90)  = 22.737435065297873
    #   leinert_zodi_spectral_radiance(1250, 0, 60) = 1.4217142030965463e-6
    got = zodi.surface_brightness_ab_mag(135.0, 0.0, 550.0, anchor="table19")
    np.testing.assert_allclose(got, 22.406737315862767, rtol=1e-9)
    got = zodi.surface_brightness_ab_mag(90.0, 30.0, 550.0, anchor="table19")
    np.testing.assert_allclose(got, 22.737435065297873, rtol=1e-9)
    got = zodi.specific_intensity(60.0, 0.0, 1250.0, anchor="table19")
    np.testing.assert_allclose(got, 1.4217142030965463e-06, rtol=1e-9)


def test_zodi_flux_ratio_composition():
    f0 = 1.0e10
    intensity_photon = zodi.specific_intensity(90.0, 0.0, 550.0, photon_units=True)
    expected = intensity_photon / units.ARCSEC2_PER_SR / 1000.0 / f0
    got = zodi.zodi_flux_ratio(90.0, 0.0, 550.0, f0)
    np.testing.assert_allclose(got, expected, rtol=1e-14)


def test_backend_parity():
    jnp = pytest.importorskip("jax.numpy")
    dlon = np.array([60.0, 90.0, 135.0, 200.0])
    beta = np.array([0.0, 15.0, 45.0, 95.0])
    lam = np.array([500.0, 700.0, 1250.0, 2200.0])
    for fn, args in [
        (zodi.position_factor, (dlon, beta)),
        (zodi.specific_intensity_at_location, (dlon, beta)),
        (zodi.color_correction, (lam,)),
        (zodi.specific_intensity, (dlon, beta, lam)),
        (zodi.surface_brightness_ab_mag, (dlon, beta, lam)),
    ]:
        a = fn(*args)
        b = fn(*(jnp.asarray(x) for x in args))
        # a few ULPs of round-off from transcendental ops and fma fusion
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=0.0)
