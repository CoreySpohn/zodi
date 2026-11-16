"""Typed dust conventions: absolute anchors, forwarding, and traced levels."""

import dataclasses
import subprocess
import sys

import ineedvalidation as vv
import numpy as np
import pytest

from zodi import (
    AbstractDustAmplitude,
    DensityDust,
    LocalZodiMultiple,
    Passband,
    ReferenceStar,
    SurfaceBrightnessDust,
    Viewing,
    exozodi_flux_ratio_v,
    specific_intensity,
)


@vv.case("exozodi-conventions", "code-verification")
def test_solar_calibration_from_primitives():
    """Fails if the default calibration or magnitude denominator changes.

    Tolerance basis: exact-algebra for flux (bitwise), floating-point for
    the logarithm (four float64 eps relative, zero absolute tolerance).
    """
    dust = SurfaceBrightnessDust(level=3)
    host = dict(host_band_abs_mag=4.83, host_luminosity_lsun=1.0)
    expected = 3 * 10.0 ** (-0.4 * 22)
    assert dust.flux_ratio(**host, r_au=1.0) == expected
    assert dust.viewing == Viewing(0.0, 0.0)
    np.testing.assert_allclose(
        dust.mag_per_arcsec2_at(**host, r_au=1.0),
        22 - 2.5 * np.log10(3),
        rtol=4 * np.finfo(float).eps,
        atol=0,
    )


@pytest.mark.parametrize("mv,lum", [(4.83, 1.0), (3.5, 3.0), (7.0, 0.2)])
@vv.case("exozodi-conventions", "code-verification")
def test_band_rule_and_eeid_identity(mv, lum):
    """Fails if band scaling changes or EEID luminosity is counted twice.

    Tolerance basis: exact-algebra (bitwise) for the reused V chain;
    floating-point, four float64 eps, for squaring sqrt(L) in the identity.
    """
    dust = SurfaceBrightnessDust(level=np.array([1.0, 3.0, 0.5]))
    host = dict(host_band_abs_mag=mv, host_luminosity_lsun=lum)
    radius = np.array([0.5, 1.0, 2.5])
    np.testing.assert_array_equal(
        dust.flux_ratio(**host, r_au=radius),
        exozodi_flux_ratio_v(dust.level, mv, radius),
    )
    np.testing.assert_array_equal(
        dust.at_eeid(**host), dust.flux_ratio(**host, r_au=np.sqrt(lum))
    )
    expected = dust.level * 10.0 ** (-0.4 * (mv - 4.83)) * 10.0 ** (-0.4 * 22) / lum
    np.testing.assert_allclose(
        dust.at_eeid(**host), expected, rtol=4 * np.finfo(float).eps, atol=0
    )


@pytest.mark.parametrize("mv,lum", [(4.83, 1.0), (7.0, 0.2)])
@vv.case("exozodi-conventions", "code-verification")
def test_luminosity_rules_and_declared_band_reference(mv, lum):
    """Fails if bolometric scaling silently substitutes band luminosity.

    The K dwarf has M_V=7, L=0.2, hence a different solar bolometric
    correction. Tolerance basis: exact-algebra, eight float64 eps for the
    ratio's reordered products; zero absolute tolerance. No colour model
    is inferred for a caller-declared R-band calibration.
    """
    host = dict(host_band_abs_mag=mv, host_luminosity_lsun=lum, r_au=0.7)
    band = SurfaceBrightnessDust(level=3)
    bolometric = dataclasses.replace(band, luminosity_rule="bolometric")
    ratio = band.flux_ratio(**host) / bolometric.flux_ratio(**host)
    np.testing.assert_allclose(
        ratio,
        10.0 ** (-0.4 * (mv - 4.83)) / lum,
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    if mv == 4.83:
        assert band.flux_ratio(**host) == bolometric.flux_ratio(**host)
    else:
        assert ratio != 1.0
    custom = dataclasses.replace(
        band,
        passband=Passband("R", 650.0, "ab"),
        reference_star=ReferenceStar(4.5, 1.0),
        mag_per_arcsec2=21.0,
        calibration_id="declared-r",
    )
    expected = 3 * 10.0 ** (-0.4 * (mv - 4.5)) * 10.0 ** (-0.4 * 21) / 0.7**2
    np.testing.assert_allclose(custom.flux_ratio(**host), expected, rtol=0, atol=0)


@pytest.mark.parametrize("lon,lat,cell", [(90.0, 30.0, 132), (60.0, 30.0, 183)])
@vv.case("local-zodi", "code-verification")
def test_local_pointing_order_and_exact_multiple(lon, lat, cell):
    """Fails if the wrapper swaps longitude and latitude or changes units.

    Tolerance basis: exact-algebra (bitwise). At 500 nm the colour is one;
    the asymmetric Table 17 cells pin angle order independently of the
    forwarded function, so a swapped-angle implementation fails.
    """
    dust = LocalZodiMultiple(
        level=np.array([1.0, 3.0]),
        ecliptic_latitude_deg=lat,
        solar_longitude_deg=lon,
    )
    wavelength = np.array([500.0, 910.0])
    np.testing.assert_array_equal(
        dust.specific_intensity(wavelength),
        dust.level * specific_intensity(lon, lat, wavelength),
    )
    assert dust.specific_intensity(500.0)[0] == cell * 1e-8
    assert specific_intensity(lat, lon, 500.0) != cell * 1e-8


@pytest.mark.parametrize(
    "kind", ["passband", "viewing", "star", "surface", "local", "density"]
)
@vv.case("exozodi-conventions", "code-verification")
def test_static_metadata_validation(kind):
    """Fails if invalid enums, non-finite or non-host metadata enter a tree.

    Tolerance basis: exact-algebra; categorical checks use exact exceptions.
    Level is intentionally unchecked, including negative and NaN values.
    """
    constructors = {
        "passband": (
            Passband,
            dict(name="V", center_nm=550.0, magnitude_system="vega"),
            ["center_nm"],
        ),
        "viewing": (
            Viewing,
            dict(inclination_deg=0.0, position_angle_deg=0.0),
            ["inclination_deg", "position_angle_deg"],
        ),
        "star": (
            ReferenceStar,
            dict(band_abs_mag=4.83, luminosity_lsun=1.0),
            ["band_abs_mag", "luminosity_lsun"],
        ),
        "surface": (
            SurfaceBrightnessDust,
            dict(level=1),
            ["mag_per_arcsec2", "reference_radius_au"],
        ),
        "local": (
            LocalZodiMultiple,
            dict(level=1, ecliptic_latitude_deg=30.0, solar_longitude_deg=90.0),
            ["ecliptic_latitude_deg", "solar_longitude_deg", "reference_radius_au"],
        ),
        "density": (
            DensityDust,
            dict(level=1, profile_id="flat", calibration_id="profile-v1"),
            ["reference_radius_au"],
        ),
    }
    cls, kwargs, numeric_fields = constructors[kind]
    for field in numeric_fields:
        for value in (float("nan"), float("inf"), -float("inf"), np.array(1.0)):
            with pytest.raises(ValueError, match=field):
                cls(**(kwargs | {field: value}))
    invalid = {
        "passband": [
            dict(magnitude_system="st"),
            dict(magnitude_system=np.array(["vega"])),
            dict(name=""),
        ],
        "viewing": [],
        "star": [],
        "surface": [
            dict(luminosity_rule="unknown"),
            dict(luminosity_rule=np.array(["band"])),
            dict(reference_radius_au="unknown"),
            dict(calibration_id=""),
            dict(reference_star=None),
            dict(passband="V"),
            dict(viewing=(0.0, 0.0)),
        ],
        "local": [dict(calibration_id=""), dict(reference_star="Sun")],
        "density": [dict(profile_id=""), dict(calibration_id=""), dict(profile_id=[])],
    }
    for changes in invalid[kind]:
        with pytest.raises(ValueError):
            cls(**(kwargs | changes))
    if "level" in kwargs:
        level = np.array([-1.0, np.nan, np.inf])
        assert cls(**(kwargs | dict(level=level))).level is level


@vv.case("exozodi-conventions", "code-verification")
def test_frozen_value_metadata_and_density_boundary():
    """Fails if metadata compare by identity or density implies a conversion.

    Tolerance basis: exact-algebra, exact equality and hash consistency.
    """
    first = SurfaceBrightnessDust(level=3)
    second = SurfaceBrightnessDust(
        level=3,
        reference_star=ReferenceStar(4.83, 1.0),
        passband=Passband("V", 550.0, "vega"),
        viewing=Viewing(0.0, 0.0),
    )
    assert isinstance(first, AbstractDustAmplitude)
    assert first == second and hash(first) == hash(second)
    assert first != dataclasses.replace(first, calibration_id="different")
    assert first != dataclasses.replace(first, level=4)
    density = DensityDust(level=3, profile_id="flat", calibration_id="profile-v1")
    local = LocalZodiMultiple(level=3, ecliptic_latitude_deg=30, solar_longitude_deg=90)
    for record in (first, density, local):
        assert record == dataclasses.replace(record)
        assert hash(record) == hash(dataclasses.replace(record))
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.level = 4
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.family = "other"
    for method in ("flux_ratio", "at_eeid", "mag_per_arcsec2_at", "specific_intensity"):
        assert not hasattr(density, method)


@pytest.mark.parametrize("kind", ["band", "bolometric", "local", "density"])
@vv.case("exozodi-conventions", "code-verification")
def test_jax_level_transforms_and_value_based_trace_cache(kind):
    """Fails if metadata become leaves, cache by identity, or levels detach.

    Tolerance basis: floating-point, 64 float64 eps for XLA fusion and
    transcendental roundoff; zero absolute tolerance, including gradients.
    Analytic derivatives of the linear level scaling are independent of AD.
    """
    jax = pytest.importorskip("jax", reason="JAX is an optional array backend")
    jnp = jax.numpy
    host = dict(host_band_abs_mag=7.0, host_luminosity_lsun=0.2)
    if kind in ("band", "bolometric"):
        dust = SurfaceBrightnessDust(level=jnp.asarray(3.0), luminosity_rule=kind)

        def evaluate(record):
            return record.flux_ratio(**host, r_au=0.7)

        scale = 10.0 ** (-0.4 * (7.0 - 4.83)) if kind == "band" else 0.2
        derivative = 10.0 ** (-0.4 * 22) * scale / 0.7**2
    elif kind == "local":
        dust = LocalZodiMultiple(
            level=jnp.asarray(3.0),
            ecliptic_latitude_deg=30,
            solar_longitude_deg=90,
        )

        def evaluate(record):
            return record.specific_intensity(500.0)

        derivative = 132 * 1e-8
    else:
        dust = DensityDust(
            level=jnp.asarray(3.0), profile_id="flat", calibration_id="profile-v1"
        )

        def evaluate(record):
            return record.level * 2

        derivative = 2.0
    leaves = jax.tree_util.tree_leaves(dust)
    assert len(leaves) == 1 and leaves[0] is dust.level
    traces = []

    @jax.jit
    def compiled(record):
        traces.append(None)
        return evaluate(record)

    tolerance = 64 * np.finfo(float).eps
    eager = evaluate(dust)
    assert isinstance(eager, jax.Array)
    np.testing.assert_allclose(compiled(dust), eager, rtol=tolerance, atol=0)
    equal_metadata = dataclasses.replace(
        dust,
        level=jnp.asarray(4.0),
        passband=Passband("V", 550, "vega"),
        viewing=Viewing(0, 0),
        reference_star=(
            ReferenceStar(4.83, 1) if kind in ("band", "bolometric") else None
        ),
    )
    compiled(equal_metadata).block_until_ready()
    assert len(traces) == 1
    compiled(
        dataclasses.replace(equal_metadata, calibration_id="different")
    ).block_until_ready()
    assert len(traces) == 2
    levels = jnp.asarray([0.5, 1.0, 3.0])
    batch = dataclasses.replace(dust, level=levels)
    loop = jnp.stack(
        [evaluate(dataclasses.replace(dust, level=level)) for level in levels]
    )
    np.testing.assert_allclose(jax.vmap(evaluate)(batch), loop, rtol=tolerance, atol=0)
    gradient = jax.grad(evaluate)(dust)
    np.testing.assert_allclose(gradient.level, derivative, rtol=tolerance, atol=0)
    assert jax.tree_util.tree_structure(gradient) == jax.tree_util.tree_structure(dust)
    np.testing.assert_allclose(
        jax.jit(lambda record: record)(dust).level, dust.level, rtol=0, atol=0
    )
    if kind in ("band", "bolometric"):
        for method in ("at_eeid", "mag_per_arcsec2_at"):
            kwargs = host if method == "at_eeid" else host | dict(r_au=0.7)

            def call(record, method=method, kwargs=kwargs):
                return getattr(record, method)(**kwargs)

            np.testing.assert_allclose(
                jax.jit(call)(dust), call(dust), rtol=tolerance, atol=0
            )
        # Host arrays must choose JAX even when level is a Python scalar.
        mixed = SurfaceBrightnessDust(level=3, luminosity_rule=kind)
        assert isinstance(
            mixed.at_eeid(host_band_abs_mag=jnp.asarray(7.0), host_luminosity_lsun=0.2),
            jax.Array,
        )
    elif kind == "local":
        assert isinstance(
            dataclasses.replace(dust, level=3).specific_intensity(jnp.asarray(500.0)),
            jax.Array,
        )


@vv.case("exozodi-conventions", "code-verification")
def test_numpy_only_import_and_frozen_records():
    """Fails if optional JAX registration makes a NumPy install unusable.

    Tolerance basis: exact-algebra, bitwise solar anchor in an interpreter
    where importing JAX is explicitly blocked.
    """
    code = """
import importlib.abc
import sys
class WithoutJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise ModuleNotFoundError('JAX intentionally absent', name='jax')
sys.meta_path.insert(0, WithoutJax())
import dataclasses
import numpy as np
import zodi
assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)
dust = zodi.SurfaceBrightnessDust(level=np.array([3.0]))
assert dataclasses.is_dataclass(dust)
np.testing.assert_array_equal(
    dust.flux_ratio(host_band_abs_mag=4.83, host_luminosity_lsun=1, r_au=1),
    np.array([3 * 10.0**(-0.4 * 22)]),
)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
