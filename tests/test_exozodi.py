"""Exozodi chain: magnitude scalings, latitudinal models, grey-scatter fit."""

from functools import partial

import ineedvalidation as vv
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

    @vv.case(
        "exozodi-conventions",
        "code-verification",
        srq="exozodi surface brightness versus stellar type and separation",
    )
    def test_reduces_to_stark_c4_at_the_eeid(self):
        """Stark et al. (2014) Eq. C4 at the EEID.

        At r = sqrt(L) AU the surface brightness is
        10**(-0.4 dMV) * (1/L) * 10**(-0.4 x), typed here from the paper.
        Tolerance basis: exact-algebra (the two sides differ only by
        floating-point evaluation order).
        """
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
    @pytest.mark.float64_only(reason="recovery of fitted grey-scatter coefficients")
    def test_closed_form_fit_recovers_synthetic_constants(self):
        lam = np.geomspace(0.2, 100.0, 40)
        star = 1e3 * lam**-2
        f_star, f_thermal = 2.5e-3, 1.7
        target = exozodi.grey_scatter_intensity(lam, star, f_star, f_thermal)
        got_star, got_thermal = exozodi.fit_grey_scatter_constants(lam, star, target)
        np.testing.assert_allclose(got_star, f_star, rtol=1e-10)
        np.testing.assert_allclose(got_thermal, f_thermal, rtol=1e-10)

    @vv.case("exozodi-conventions", "code-verification")
    def test_scatter_cutoff_excludes_long_wavelengths(self, xp):
        """Fails if backend conversion changes membership on either decision grid.

        With unit stellar flux, the masked stellar column is also the target.
        The normal equations reuse identical reductions, leaving only the
        final products, subtraction, and division. The determinant retains
        over one third of its leading product on either grid, so eight
        spacings at one bound these errors.
        """
        first_above_float32 = np.nextafter(np.float32(10.0), np.float32(11.0))
        lam = np.array(
            [10.0, np.nextafter(10.0, 11.0), 10.0 + 1e-9, first_above_float32, 11.0]
        )
        star = xp.ones(lam.size)
        host_target = xp.asarray([1.0, 0.0, 0.0, 0.0, 0.0])

        def evaluate(wavelength, flux, target):
            intensity = exozodi.grey_scatter_intensity(wavelength, flux, 1.0, 0.0)
            constants = exozodi.fit_grey_scatter_constants(wavelength, flux, target)
            return intensity, constants

        def check(result, target):
            intensity, constants = result
            np.testing.assert_array_equal(intensity, target)
            np.testing.assert_allclose(
                constants, [1.0, 0.0], rtol=0, atol=8 * np.spacing(star.dtype.type(1))
            )

        check(evaluate(lam, star, host_target), host_target)
        if xp is not np:
            jax = pytest.importorskip("jax")
            check(jax.jit(partial(evaluate, lam))(star, host_target), host_target)

            # Both wavelengths just above 10 um round to 10 on the decision grid.
            device_target = xp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
            device_lam = xp.asarray(lam)
            check(evaluate(device_lam, star, device_target), device_target)
            check(jax.jit(evaluate)(device_lam, star, device_target), device_target)

    @pytest.mark.float64_only(reason="grey-scatter calibration to Leinert Table 19")
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


class TestWavelengthUnitGuard:
    def test_host_wavelength_above_um_limit_raises(self, xp):
        """Fails if a JAX argument bypasses the host wavelength limit."""
        lam = np.array([np.nextafter(200.0, np.inf)])
        flux = xp.ones(lam.size)
        cases = (
            (exozodi.blackbody_spectral_radiance, (xp.asarray(261.5),)),
            (exozodi.grey_scatter_intensity, (flux, 1.0, 0.0)),
            (exozodi.fit_grey_scatter_constants, (flux, flux)),
        )
        jax = pytest.importorskip("jax") if xp is not np else None
        for evaluate, args in cases:
            with pytest.raises(ValueError, match="wavelength_um exceeds"):
                evaluate(lam, *args)
            if jax is not None:
                with pytest.raises(ValueError, match="wavelength_um exceeds"):
                    jax.jit(partial(evaluate, lam))(*args)


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

    @vv.case(
        "exozodi-conventions",
        "code-verification",
        srq="exozodi surface brightness versus stellar type and separation",
    )
    def test_chain_matches_flux_ratio_and_is_luminosity_free(self):
        """The cached-intensity chain equals the general-radius chain.

        jez0 -> scale_jez must equal the V-band flux ratio times
        F0 * f_lambda * bandwidth * fbeta for any bolometric luminosity,
        i.e. the 1/L in jez0 cancels against (EEID / r)**2. A chain that
        double counts the luminosity fails by a factor L (0.02 to 3 here).
        Tolerance basis: exact-algebra.
        """
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
    """Check the geometry, magnitude chain, and grey spectrum against float64.

    The V chain allows 12 eps for casting, magnitude subtraction and power,
    products, and division. Folding these integer angles is exact. The
    normalized latitude table uses eight roundings; the signed cubic uses
    16 relative eps for the angle, sine, powers, products, and additions.
    Thermal error counts 12 amplitude and conversion roundings and six in
    the exponent, amplified by x/(1-exp(-x)). The positive grey spectrum
    adds twice the thermal error plus three product and sum roundings.
    """
    jnp = pytest.importorskip("jax.numpy")
    from hwoutils.numerics import eps

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
            12,
        ),
        (exozodi.theta_from_inclination, (inc,), 0),
        (lambda t: exozodi.latitudinal_factor(t, model="leinert"), (inc,), 8),
        (lambda t: exozodi.latitudinal_factor(t, model="stark2014"), (inc,), 16),
    ]
    for fn, args, n_ops in cases:
        expected = fn(*args)
        actual = fn(*(jnp.asarray(x) for x in args))
        np.testing.assert_allclose(
            actual, expected, rtol=n_ops * eps(actual.dtype), atol=0
        )

    thermal = exozodi.blackbody_spectral_radiance(lam, 261.5)
    spectrum = exozodi.grey_scatter_intensity(lam, star, 1e-3, 2.0)
    jlam = jnp.asarray(lam)
    thermal_bound = _planck_error(lam, 261.5, thermal, jlam.dtype)
    actual = exozodi.blackbody_spectral_radiance(jlam, 261.5)
    np.testing.assert_array_less(abs(np.asarray(actual) - thermal), thermal_bound)
    actual = exozodi.grey_scatter_intensity(jlam, jnp.asarray(star), 1e-3, 2.0)
    bound = 2 * thermal_bound + 3 * eps(actual.dtype) * spectrum
    np.testing.assert_array_less(abs(np.asarray(actual) - spectrum), bound)


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


def _planck_error(lam_um, temperature, radiance, dtype):
    """Absolute energy-radiance error at the local Planck condition number."""
    from hwoutils.constants import c, h, k_B, um2m
    from hwoutils.numerics import eps

    x = (h * c / (k_B * um2m)) / (lam_um * temperature)
    sensitivity = x / -np.expm1(-x)
    roundoff = (12 + 6 * sensitivity) * eps(dtype) * radiance
    tiny = np.finfo(dtype).tiny
    return np.where(radiance < tiny, tiny, roundoff)


def test_blackbody_agrees_with_hwoutils(xp, precision):
    """Compare energy radiance to the photon law and the float64 Planck law.

    The logarithmic sensitivity to exponent error is x/(1-exp(-x)). Six
    roundings cover constants, input casts, nm to um, product and quotient.
    Twelve more cover lambda**-5, the occupation factor, and unit conversion.
    Below the smallest normal radiance the backend may flush to zero.
    All float64-reference values in the normal range must remain positive.
    """
    pytest.importorskip("jax")
    from hwoutils.constants import c, h, k_B, nm2m, nm2um, um2m, um2nm
    from hwoutils.conversions import photons_per_nm_per_m2_to_w_per_nm_per_m2
    from hwoutils.planck import planck_photon_radiance

    nm = np.geomspace(300.0, 30000.0, 31)[:, None]
    temp = np.geomspace(100.0, 6000.0, 13)[None, :]
    meters = nm * nm2m
    reference = (2 * h * c**2 * um2m / meters**5) / np.expm1(
        (h * c / k_B) / (meters * temp)
    )
    wavelength_nm = xp.asarray(nm, dtype=precision)
    temperature = xp.asarray(temp, dtype=precision)
    actual = np.asarray(
        exozodi.blackbody_spectral_radiance(wavelength_nm * nm2um, temperature)
    )
    assert actual.dtype == precision
    normal = reference >= np.finfo(precision).tiny
    assert np.all(actual[normal] > 0)
    bound = _planck_error(nm * nm2um, temp, reference, precision)
    np.testing.assert_array_less(abs(actual - reference), bound)

    photons = planck_photon_radiance(wavelength_nm, temperature)
    # Convert on the host so a per-nm intermediate cannot flush a normal
    # per-um result. Each independent law gets its own roundoff allowance.
    energy = (
        photons_per_nm_per_m2_to_w_per_nm_per_m2(
            np.asarray(photons, dtype=np.float64), nm
        )
        * um2nm
    )
    np.testing.assert_array_less(abs(actual - energy), 2 * bound)


@pytest.mark.float64_only(reason="mixed-precision comparison to a float64 forward")
def test_blackbody_float32_inside_float64():
    """No wide intermediates; the relative Wien error scales with x.

    At 0.4 um and 400 K the exponent exceeds the float32 exp overflow
    threshold while the radiance is normal. Use the same conditioned
    bound as the full grid, relative to each individual radiance.
    """
    jnp = pytest.importorskip("jax.numpy")
    from hwoutils.testing import assert_precision_closed, precision_pair

    law = exozodi.blackbody_spectral_radiance
    wave = jnp.asarray([0.4, 10.0, 100.0])
    temperature = jnp.asarray(400.0)
    assert_precision_closed(
        law, wave.astype(jnp.float32), temperature.astype(jnp.float32)
    )
    bound = _planck_error(0.4, 400, 1.0, np.dtype("float32"))
    precision_pair(law, (wave, temperature), bound=bound, scale=0)
