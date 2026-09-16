"""Exozodi chain: magnitude scalings, latitudinal models, grey-scatter fit."""

import numpy as np
import pytest

from zodi import exozodi


class TestFluxRatioV:
    def test_solar_twin_at_one_zodi_is_the_definition(self):
        got = exozodi.exozodi_flux_ratio_v(1.0, 4.83, 1.0)
        np.testing.assert_allclose(got, 10.0**-8.8, rtol=1e-12)

    def test_magnitude_radius_and_zodi_scalings(self):
        base = exozodi.exozodi_flux_ratio_v(1.0, 4.83, 1.0)
        np.testing.assert_allclose(
            exozodi.exozodi_flux_ratio_v(1.0, 7.33, 1.0), base / 10.0, rtol=1e-12
        )
        np.testing.assert_allclose(
            exozodi.exozodi_flux_ratio_v(1.0, 4.83, 2.0), base / 4.0, rtol=1e-12
        )
        np.testing.assert_allclose(
            exozodi.exozodi_flux_ratio_v(3.0, 4.83, 1.0), 3.0 * base, rtol=1e-12
        )

    def test_reduces_to_stark_c4_at_the_eeid(self):
        # Stark et al. (2014) Eq. C4: at r = sqrt(L) AU the surface
        # brightness is 10**(-0.4 dMV) * (1/L) * 10**(-0.4 x).
        mv = np.array([3.5, 4.83, 6.2, 10.4])
        lum = np.array([3.0, 1.0, 0.3, 0.02])
        c4 = 10.0 ** (-0.4 * (mv - 4.83)) * 10.0**-8.8 / lum
        got = exozodi.exozodi_flux_ratio_v(1.0, mv, np.sqrt(lum))
        np.testing.assert_allclose(got, c4, rtol=1e-12)

    def test_legacy_positional_luminosity_call_is_rejected(self):
        # The removed l_star_lsun argument must not be silently read as r_au.
        with pytest.raises(TypeError):
            exozodi.exozodi_flux_ratio_v(1.0, 4.83, 1.0, 1.0)


class TestFluxRatioBand:
    def test_v_band_recovers_v_form(self):
        got = exozodi.exozodi_flux_ratio_band(1.0, 4.83, 5.0, 5.0)
        np.testing.assert_allclose(got, 10.0**-8.8, rtol=1e-12)

    def test_stellar_color_scaling_is_grey_scattering(self):
        # a star 2.5 mag brighter in the band scatters 10x more exozodi
        base = exozodi.exozodi_flux_ratio_band(1.0, 4.83, 5.0, 5.0)
        brighter = exozodi.exozodi_flux_ratio_band(1.0, 4.83, 5.0, 2.5)
        np.testing.assert_allclose(brighter, 10.0 * base, rtol=1e-12)


class TestInclinationFolding:
    def test_folding_matches_exosims_convention(self):
        np.testing.assert_allclose(exozodi.theta_from_inclination(0.0), 90.0)
        np.testing.assert_allclose(exozodi.theta_from_inclination(90.0), 0.0)
        np.testing.assert_allclose(exozodi.theta_from_inclination(135.0), 45.0)
        np.testing.assert_allclose(exozodi.theta_from_inclination(270.0), 0.0)

    def test_symmetry_about_edge_on(self):
        inc = np.array([30.0, 150.0])
        theta = exozodi.theta_from_inclination(inc)
        np.testing.assert_allclose(theta[0], theta[1])


class TestLatitudinalFactor:
    def test_lindler_normalized_at_zero(self):
        np.testing.assert_allclose(
            exozodi.latitudinal_factor(0.0, model="lindler2006"), 1.0, rtol=1e-12
        )

    def test_stark2014_polynomial_values(self):
        np.testing.assert_allclose(
            exozodi.latitudinal_factor(0.0, model="stark2014"), 1.02, rtol=1e-12
        )
        theta = 30.0
        s = np.sin(np.deg2rad(theta))
        expected = 1.02 - 0.566 * s - 0.884 * s**2 + 0.853 * s**3
        np.testing.assert_allclose(
            exozodi.latitudinal_factor(theta, model="stark2014"), expected, rtol=1e-12
        )

    def test_leinert_profile_at_tabulated_latitudes(self):
        # lon 135 row of Table 17: beta 0 -> 179, beta 90 -> 77
        np.testing.assert_allclose(
            exozodi.latitudinal_factor(0.0, model="leinert"), 1.0, rtol=1e-14
        )
        np.testing.assert_allclose(
            exozodi.latitudinal_factor(90.0, model="leinert"), 77.0 / 179.0, rtol=1e-12
        )
        np.testing.assert_allclose(
            exozodi.latitudinal_factor(45.0, model="leinert"), 90.0 / 179.0, rtol=1e-12
        )

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            exozodi.latitudinal_factor(0.0, model="nope")


class TestBlackbody:
    def test_wien_peak_location(self):
        # Wien: lambda_max = 2898 um K / T = 11.08 um at 261.5 K
        peak = 2898.0 / 261.5
        b = exozodi.blackbody_spectral_radiance
        assert b(peak, 261.5) > b(peak - 3.0, 261.5)
        assert b(peak, 261.5) > b(peak + 3.0, 261.5)

    def test_pinned_value_at_10um(self):
        # Planck B_lambda(10 um, 261.5 K) with exact SI constants:
        # x = h c / (lambda k T) = 5.5019, B = 2 h c^2 / lambda^5 / expm1(x)
        # = 4.877616640635079 W m-2 sr-1 um-1
        got = exozodi.blackbody_spectral_radiance(10.0, 261.5)
        np.testing.assert_allclose(got, 4.877616640635079, rtol=1e-9)


class TestGreyScatterFit:
    def test_closed_form_fit_recovers_synthetic_constants(self):
        lam = np.geomspace(0.2, 100.0, 40)
        star = 1e3 * lam**-2
        f_star, f_thermal = 2.5e-3, 1.7
        target = exozodi.grey_scatter_intensity(lam, star, f_star, f_thermal)
        got_star, got_thermal = exozodi.fit_grey_scatter_constants(lam, star, target)
        np.testing.assert_allclose(got_star, f_star, rtol=1e-10)
        np.testing.assert_allclose(got_thermal, f_thermal, rtol=1e-10)

    def test_scatter_cutoff_excludes_long_wavelengths(self):
        lam = np.array([1.0, 5.0, 20.0])
        star = np.ones(3)
        got = exozodi.grey_scatter_intensity(lam, star, 1.0, 0.0)
        np.testing.assert_allclose(got, [1.0, 1.0, 0.0])

    def test_table19_calibration_with_blackbody_sun_proxy(self):
        # reproduces the EXOSIMS calibration procedure with a 5772 K
        # blackbody standing in for the solar spectrum; the real-spectrum
        # calibration lives in the validation scripts. The UV points
        # (0.2-0.3 um) are excluded: a blackbody badly overpredicts solar
        # UV. In the optical (0.4-0.9 um), where grey scattering
        # dominates, the fit tracks Table 19 to a few percent; the known
        # worst region is the 3.5 um dip between the scattered and
        # thermal components (ratio ~0.5, visible in the EXOSIMS
        # documentation fit figure too).
        from zodi import tables

        lam = tables.TABLE19_WAVELENGTH_UM
        sun_proxy = exozodi.blackbody_spectral_radiance(lam, 5772.0)
        f_star, f_thermal = exozodi.fit_grey_scatter_constants(
            lam, sun_proxy, tables.TABLE19_RADIANCE_W_M2_SR_UM
        )
        assert f_star > 0 and f_thermal > 0
        model = exozodi.grey_scatter_intensity(lam, sun_proxy, f_star, f_thermal)
        ratio = model / tables.TABLE19_RADIANCE_W_M2_SR_UM
        optical = (lam >= 0.4) & (lam <= 0.9)
        np.testing.assert_allclose(ratio[optical], 1.0, atol=0.06)
        beyond_uv = lam >= 0.4
        assert np.all(ratio[beyond_uv] > 0.45)
        assert np.all(ratio[beyond_uv] < 1.3)


class TestBandAverage:
    def test_constant_spectrum_is_unchanged(self):
        lam = np.linspace(0.4, 0.7, 31)
        np.testing.assert_allclose(
            exozodi.band_average(lam, np.full(31, 2.5), np.ones(31)), 2.5, rtol=1e-12
        )

    def test_linear_spectrum_flat_throughput_gives_band_center(self):
        lam = np.linspace(0.4, 0.7, 31)
        values = 3.0 * lam + 1.0
        got = exozodi.band_average(lam, values, np.ones(31))
        np.testing.assert_allclose(got, 3.0 * 0.55 + 1.0, rtol=1e-12)


class TestJez:
    def test_jez0_composition(self):
        got = exozodi.jez0(1e10, 4.83, 1.0, 1.0, 100.0)
        np.testing.assert_allclose(got, 1e10 * 10.0**-8.8 * 100.0, rtol=1e-12)

    def test_scale_jez(self):
        np.testing.assert_allclose(
            exozodi.scale_jez(5.0, 3.0, 2.0, 0.5, 4.0),
            5.0 * 3.0 * 0.5 * 4.0 / 4.0,
            rtol=1e-14,
        )

    def test_scale_jez_at_the_eeid_returns_the_reference_value(self):
        lum = np.array([3.0, 1.0, 0.3, 0.02])
        ref = exozodi.jez0(1e10, 4.83, lum, 1.0, 100.0)
        got = exozodi.scale_jez(ref, 2.0, np.sqrt(lum), 0.7, lum)
        np.testing.assert_allclose(got, ref * 2.0 * 0.7, rtol=1e-12)

    def test_chain_matches_flux_ratio_and_is_luminosity_free(self):
        # jez0 -> scale_jez must equal the general-radius V-band chain times
        # F0 * f_lambda * bandwidth * fbeta, for any bolometric luminosity.
        f0, flam, bw, fbeta, nz = 1e10, 1.3, 100.0, 0.6, 3.0
        mv = np.array([3.5, 6.2, 10.4])
        lum = np.array([3.0, 0.3, 0.02])
        r = np.array([0.5, 1.0, 2.5])
        chain = exozodi.scale_jez(
            exozodi.jez0(f0, mv, lum, flam, bw), nz, r, fbeta, lum
        )
        direct = f0 * flam * bw * fbeta * exozodi.exozodi_flux_ratio_v(nz, mv, r)
        np.testing.assert_allclose(chain, direct, rtol=1e-12)


def test_backend_parity():
    jnp = pytest.importorskip("jax.numpy")
    lam = np.geomspace(0.3, 50.0, 12)
    star = 1e3 * lam**-2
    inc = np.array([0.0, 45.0, 135.0])
    cases = [
        (
            exozodi.exozodi_flux_ratio_v,
            (
                np.array([1.0, 3.0]),
                np.array([4.0, 5.0]),
                np.array([1.0, 1.5]),
            ),
        ),
        (exozodi.theta_from_inclination, (inc,)),
        (lambda t: exozodi.latitudinal_factor(t, model="leinert"), (inc,)),
        (lambda t: exozodi.latitudinal_factor(t, model="stark2014"), (inc,)),
        (exozodi.blackbody_spectral_radiance, (lam, 261.5)),
        (lambda w, s: exozodi.grey_scatter_intensity(w, s, 1e-3, 2.0), (lam, star)),
    ]
    for fn, args in cases:
        a = fn(*args)
        b = fn(*(jnp.asarray(x) for x in args))
        # transcendental ops (10**x, expm1, sin) and XLA fma fusion allow
        # a few ULPs of cross-backend round-off; gather/arithmetic paths
        # are exact
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=0.0)


def test_grad_through_grey_scatter_fit():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    lam = np.geomspace(0.3, 50.0, 12)
    star = 1e3 * lam**-2
    target = exozodi.grey_scatter_intensity(lam, star, 2e-3, 1.5)

    def loss(scale):
        f_star, f_thermal = exozodi.fit_grey_scatter_constants(
            lam, jnp.asarray(star) * scale, jnp.asarray(target)
        )
        return f_star * 1e3 + f_thermal

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert np.isfinite(float(g))
