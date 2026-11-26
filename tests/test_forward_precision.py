"""Forward reads in float32 inside a float64 process."""

from functools import partial

import ineedvalidation as vv
import numpy as np
import pytest

from zodi import (
    LocalZodiMultiple,
    ReferenceStar,
    SurfaceBrightnessDust,
    exozodi,
    tables,
    units,
    zodi,
)
from zodi._interp import interp1d_clamped, interp2d_bilinear_clamped

_MIXED_PRECISION = pytest.mark.float64_only(
    reason="float32 trace inside a float64 process"
)


def _assert_forward(fn, args, n_ops):
    pytest.importorskip("jax")
    from hwoutils.leaves import canonical
    from hwoutils.testing import assert_precision_closed, precision_pair, ulps

    wide = canonical(args, precision="float64")
    assert_precision_closed(fn, *canonical(wide, precision="float32"))
    precision_pair(fn, wide, bound=float(ulps(n_ops, np.float32(1))), scale=0)


_WAVE_NM = (100.0, 500.0, 723.25, 200_000.0)
_LON_DEG = (90.0, 83.5, -140.0, 190.0)
_LAT_DEG = (0.0, 23.25, -35.0, 95.0)


def _table19_ops():
    """Convert one log interpolation's error to a relative radiance error."""
    axis = np.log10(tables.TABLE19_WAVELENGTH_UM)
    radiance = np.log10(tables.TABLE19_RADIANCE_W_M2_SR_UM)
    slope = np.max(np.abs(np.diff(radiance) / np.diff(axis)))
    log_error = 6 * np.max(np.abs(radiance)) + 5 * np.max(np.abs(axis)) * slope
    return int(np.ceil(np.log(10.0) * log_error))


@pytest.mark.parametrize("bilinear", [False, True], ids=["linear", "bilinear"])
def test_host_cell_spacing_is_differenced_before_staging(bilinear):
    """Fails if rounding host knots changes cell selection, weights, or clamps.

    The interior knots round in opposite directions. The queries include
    rounded knots still inside the table and points outside both ends.
    Twelve roundings at the maximum ordinate cover the bilinear fractions,
    products and sum, which also bounds the linear case.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from hwoutils.testing import ulps

    step = float(np.spacing(np.float32(1)))
    axis = 1.0 + step * np.array([0.49, 2.51, 5.49, 7.49])
    values = np.array([0.0, 1.0, 0.25, 0.75])
    query = 1.0 + step * np.arange(10)
    expected = np.interp(query, axis, values)
    if bilinear:
        table = values[:, None] + values[None, :]

        def read(x):
            return interp2d_bilinear_clamped(axis, axis, table, x, x[::-1])

        expected = expected + expected[::-1]
        scale = np.max(table)
    else:
        read = partial(interp1d_clamped, axis, values)
        scale = np.max(values)
    x = jnp.asarray(query, dtype=np.float32)
    for run in (read, jax.jit(read)):
        actual = run(x)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=float(ulps(12, np.float32(scale)))
        )


@_MIXED_PRECISION
@pytest.mark.parametrize("wide_input", ["axis", "table"])
def test_interpolation_follows_live_grid_and_table(wide_input):
    """Fails if a narrow query narrows a wide device axis or sampled ordinate.

    Exact binary inputs produce an exact result. The axis case uses knots
    that collapse together at the query's dtype.
    """
    jnp = pytest.importorskip("jax.numpy")

    axis = [1 - 2.0**-26, 1 + 2.0**-26]
    values = [0.0, 1.0]
    expected = 0.5
    if wide_input == "axis":
        axis = jnp.asarray(axis)
    else:
        axis = [0.0, 1.0]
        expected = 1.0 + 2.0**-30
        values = jnp.asarray([0.0, expected])
    query = jnp.asarray(1.0, dtype=np.float32)
    for result in (
        interp1d_clamped(axis, values, query),
        interp2d_bilinear_clamped(
            axis,
            [0.0, 1.0],
            jnp.stack((values, values), axis=1)
            if wide_input == "table"
            else [[0.0, 0.0], [1.0, 1.0]],
            query,
            query,
        ),
    ):
        assert result.dtype == np.float64
        assert result == expected


@pytest.mark.parametrize("read", ["v", "band", "jez", "amplitude", "eeid"])
def test_host_magnitude_difference_survives_float32_staging(read):
    """Fails when separately rounded host magnitudes change nearly equal colors.

    Compare the ratio of the two fluxes with the host-color reference.
    Four float32 roundings cover two powers, their products and the ratio.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from hwoutils.testing import ulps

    magnitude = [24.0000009, 24.0000010]
    model = partial(SurfaceBrightnessDust, reference_star=ReferenceStar(24.0, 1.0))

    def evaluate(level):
        if read == "v":
            return exozodi.exozodi_flux_ratio_v(level, magnitude, 1.0, mv_sun=24.0)
        if read == "band":
            return exozodi.exozodi_flux_ratio_band(
                level, 24.0, 24.0, magnitude, mv_sun=24.0
            )
        if read == "jez":
            return exozodi.jez0(level, magnitude, 1.0, 1.0, 1.0, mv_sun=24.0)
        dust = model(level=level)
        host = dict(host_band_abs_mag=magnitude, host_luminosity_lsun=1.0)
        if read == "eeid":
            return dust.at_eeid(**host)
        return dust.flux_ratio(**host, r_au=1.0)

    expected = 10.0 ** (-0.4 * (magnitude[1] - magnitude[0]))
    level = jnp.asarray(1.0, dtype=np.float32)
    for run in (evaluate, jax.jit(evaluate)):
        actual = run(level)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(
            actual[1] / actual[0], expected, rtol=0, atol=float(ulps(4, np.float32(1)))
        )


@_MIXED_PRECISION
def test_band_flux_retains_precision_of_a_wide_owner():
    """Fails when the dust read discards a wide input while staging host color.

    One host magnitude spacing survives in the float64 ratio. Eight epsilons
    cover the powers and products, well below the float32 rounding error.
    """
    jnp = pytest.importorskip("jax.numpy")
    from hwoutils.testing import ulps

    dust = SurfaceBrightnessDust(
        level=jnp.asarray(1, dtype=np.float32), reference_star=ReferenceStar(24.0, 1.0)
    )
    luminosity = jnp.asarray(1.0, dtype=np.float64)
    magnitude = [24.0000009, 24.0000010]
    host = dict(host_band_abs_mag=magnitude, host_luminosity_lsun=luminosity)
    for actual in (dust.flux_ratio(**host, r_au=1.0), dust.at_eeid(**host)):
        assert actual.dtype == luminosity.dtype
        expected = 10.0 ** (-0.4 * (magnitude[1] - magnitude[0]))
        np.testing.assert_allclose(
            actual[1] / actual[0], expected, rtol=0, atol=float(ulps(8, luminosity))
        )


@_MIXED_PRECISION
@pytest.mark.parametrize(
    "fn, args, spectra",
    [
        pytest.param(
            partial(zodi.position_factor, 83.5), (_LAT_DEG,), 0, id="position"
        ),
        pytest.param(
            partial(zodi.specific_intensity_at_location, photon_units=True),
            (_LON_DEG, _LAT_DEG),
            0,
            id="location",
        ),
        pytest.param(zodi.color_correction, (_WAVE_NM,), 2, id="power-color"),
        pytest.param(
            partial(zodi.color_correction, 700.0, photon_units=True),
            ((525.0, 637.5),),
            2,
            id="photon-color",
        ),
        pytest.param(zodi.ab_color_correction, (_WAVE_NM,), 2, id="ab-color"),
        pytest.param(zodi.ayo_default_ab_mag, (_WAVE_NM,), 2, id="ayo"),
        pytest.param(partial(zodi.ayo_default_ab_mag, 650.0), (22.0,), 2, id="ayo-mag"),
        pytest.param(
            partial(zodi.specific_intensity, 83.5, 23.25),
            (_WAVE_NM,),
            2,
            id="table17-radiance",
        ),
        pytest.param(
            partial(zodi.specific_intensity, wavelength_nm=723.25, anchor="table19"),
            (_LON_DEG, _LAT_DEG),
            1,
            id="table19-radiance",
        ),
        pytest.param(
            partial(zodi.surface_brightness_ab_mag, wavelength_nm=723.25),
            (_LON_DEG, _LAT_DEG),
            2,
            id="leinert-mag",
        ),
        pytest.param(
            partial(zodi.zodi_flux_ratio, 83.5, 23.25, 723.25),
            (1200.0,),
            2,
            id="flux-ratio",
        ),
    ],
)
@vv.case("local-zodi", "code-verification", basis="floating-point")
def test_local_zodi_forward_precision(fn, args, spectra):
    """Keep sightline and spectral reads closed at nodes, interiors and clamps.

    Allow 20 roundings for the positive Table 17 bilinear sum, normalization
    and final unit or magnitude conversion. For each Table 19 radiance,
    allow six roundings at the largest log ordinate for weights and sums,
    plus five at the largest log wavelength for knot/width conversion and
    query log error, propagated through the steepest tabulated log slope.
    Multiplying by ln(10) propagates log-radiance error through the power.
    """
    _assert_forward(
        fn, tuple(np.asarray(arg) for arg in args), 20 + spectra * _table19_ops()
    )


_EXO_WAVE_UM = np.array([4.5, 8.0, 12.0, 24.0])
_HOST = {"host_band_abs_mag": 4.25, "host_luminosity_lsun": 1.5}


def _amplitude_flux(record):
    return record.flux_ratio(**_HOST, r_au=0.75)


def _amplitude_mag(record):
    return record.mag_per_arcsec2_at(**_HOST, r_au=0.75)


@_MIXED_PRECISION
@pytest.mark.parametrize(
    "fn, args, n_ops",
    [
        pytest.param(
            partial(exozodi.exozodi_flux_ratio_v, mv_star=exozodi.MV_SUN, r_au=1.0),
            (2.0,),
            1,
            id="exo-v",
        ),
        pytest.param(
            partial(
                exozodi.exozodi_flux_ratio_band,
                mv_star=4.25,
                vmag_star=5.25,
                band_mag_star=5.75,
            ),
            (2.5,),
            80,
            id="exo-band",
        ),
        pytest.param(
            exozodi.theta_from_inclination,
            (np.array([0.0, 45.0, 190.0]),),
            0,
            id="inclination",
        ),
        pytest.param(
            exozodi.latitudinal_factor,
            (np.array([0.0, 23.25, 95.0]),),
            12,
            id="leinert-latitude",
        ),
        pytest.param(
            partial(exozodi.latitudinal_factor, model="lindler2006"),
            (np.array([0.0, 23.25, 89.0]),),
            64,
            id="lindler-latitude",
        ),
        pytest.param(
            partial(exozodi.latitudinal_factor, model="stark2014"),
            (np.array([0.0, 23.25, 89.0]),),
            64,
            id="stark-latitude",
        ),
        pytest.param(
            partial(exozodi.blackbody_spectral_radiance, _EXO_WAVE_UM),
            (261.5,),
            96,
            id="planck-host-wave",
        ),
        pytest.param(
            partial(
                exozodi.grey_scatter_intensity, _EXO_WAVE_UM, f_star=1e-3, f_thermal=0.5
            ),
            (np.array([10.0, 3.0, 1.0, 0.25]),),
            96,
            id="grey-scatter",
        ),
        pytest.param(
            partial(exozodi.jez0, 1200.0, 4.25, 1.5, bandwidth_nm=20.0),
            (1.2,),
            80,
            id="jez0",
        ),
        pytest.param(
            partial(
                exozodi.scale_jez, nzodi=2.5, r_au=0.75, fbeta=0.7, l_star_lsun=1.5
            ),
            (0.002,),
            8,
            id="jez",
        ),
        pytest.param(
            _amplitude_flux, (SurfaceBrightnessDust(level=3.0),), 80, id="dust-band"
        ),
        pytest.param(
            _amplitude_flux,
            (SurfaceBrightnessDust(level=3.0, luminosity_rule="bolometric"),),
            8,
            id="dust-bolometric",
        ),
        pytest.param(
            partial(SurfaceBrightnessDust.at_eeid, **_HOST),
            (SurfaceBrightnessDust(level=3.0),),
            80,
            id="dust-eeid",
        ),
        pytest.param(
            _amplitude_mag, (SurfaceBrightnessDust(level=3.0),), 80, id="dust-magnitude"
        ),
        pytest.param(
            partial(LocalZodiMultiple.specific_intensity, wavelength_nm=723.25),
            (
                LocalZodiMultiple(
                    level=3.0,
                    helioecliptic_longitude_deg=83.5,
                    ecliptic_latitude_deg=23.25,
                ),
            ),
            20 + 2 * _table19_ops(),
            id="dust-local",
        ),
    ],
)
@vv.case("exozodi-conventions", "code-verification", basis="floating-point")
def test_dust_forward_precision(fn, args, n_ops):
    """Compare dust radiance, geometry and amplitude reads at both precisions.

    Folding the integral angles is exact. For latitude, 12 roundings cover
    the normalized table interpolation. The signed polynomials allow eight
    roundings amplified by at most eight for cancellation on these angles.
    The solar V case uses a unit radius and a power-of-two level, so the
    host magnitude scale rounds just once. Other magnitude chains allow
    80, including two exponent roundings at scale nine amplified by
    ln(10), and the color, square, root and flux arithmetic. Eight suffice
    for simple scalings. The Planck sensitivity x/(1-exp(-x)) is below 13
    at these wavelengths, so 96 covers six exponent roundings plus 18
    for amplitude, conversion and the positive scattered/thermal sum.
    The local amplitude adds one product to the Table 17/19 budget above.
    """
    _assert_forward(fn, args, n_ops)


@_MIXED_PRECISION
@pytest.mark.parametrize(
    "fn, arg, n_ops",
    [
        pytest.param(units.mag_to_flux_ratio, 22.25, 64, id="mag-to-flux"),
        pytest.param(units.flux_ratio_to_mag, 3e-9, 8, id="flux-to-mag"),
        pytest.param(
            partial(units.power_to_photon_intensity, wavelength_nm=500.0),
            2e-6,
            8,
            id="power-to-photons",
        ),
        pytest.param(
            partial(units.photon_ratio_factor, reference_wavelength_nm=550.0),
            723.25,
            4,
            id="photon-ratio",
        ),
        pytest.param(
            partial(units.intensity_to_mjy_per_sr, wavelength_um=0.65),
            2e-6,
            10,
            id="intensity-to-mjy",
        ),
        pytest.param(
            partial(units.ab_mag_arcsec2_from_intensity, wavelength_nm=500.0),
            2e-6,
            16,
            id="intensity-to-ab",
        ),
        pytest.param(
            partial(units.intensity_from_ab_mag_arcsec2, wavelength_nm=500.0),
            22.25,
            80,
            id="ab-to-intensity",
        ),
    ],
)
@vv.case("local-zodi", "code-verification", basis="floating-point")
def test_radiance_conversion_precision(fn, arg, n_ops):
    """Check unit conversions with default host constants and one dynamic input.

    Ratios, wavelength powers and conversions to logs allow 4 to 16
    roundings for constants, products, divisions and the final logarithm.
    The inverse powers allow two roundings at exponent magnitude nine,
    amplified by ln(10), plus the final power and unit arithmetic.
    """
    _assert_forward(fn, (arg,), n_ops)
