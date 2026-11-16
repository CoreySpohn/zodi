"""Three distinct meanings of a dust amplitude, with no implicit conversion.

``SurfaceBrightnessDust`` declares a calibrated scattered surface brightness
(Stark et al. 2014); ``LocalZodiMultiple`` multiplies the local zodiacal
brightness at a pointing; ``DensityDust`` multiplies a named density profile.
Matching their numeric levels does not match their physical definitions.
Conversion requires a profile and its line-of-sight integral, owned by the
profile's implementation, so this module provides no conversion between them.

Records are frozen dataclasses. Each concrete ``family`` is a fixed class
constant. Only ``level`` is array-capable; all calibration metadata are
validated, immutable host values. If JAX is importable, guarded PyTree
registration makes only ``level`` a child, with value-comparable metadata
as static auxiliary data. NumPy-only installations need no JAX.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import partial
from math import isfinite
from numbers import Real
from typing import Any, ClassVar

from zodi._xp import array_namespace
from zodi.exozodi import MAG_1ZODI_V_ARCSEC2, MV_SUN, exozodi_flux_ratio_v
from zodi.units import flux_ratio_to_mag
from zodi.zodi import V_BAND_WAVELENGTH_NM, specific_intensity

__all__ = [
    "AbstractDustAmplitude",
    "DensityDust",
    "LocalZodiMultiple",
    "Passband",
    "ReferenceStar",
    "SurfaceBrightnessDust",
    "Viewing",
]


def _host_float(record, name):
    """Normalize finite real metadata to a hashable host float."""
    value = getattr(record, name)
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite host float")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite host float") from error
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    object.__setattr__(record, name, value)


def _required_string(value, name):
    """Require a nonempty host string without constraining free text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class Passband:
    """Calibration band: name, center in nm, and 'vega' or 'ab' magnitudes.

    The center is descriptive metadata, not a bandpass integration model.
    """

    name: str
    center_nm: float
    magnitude_system: str

    def __post_init__(self):
        """Validate static photometric metadata."""
        _required_string(self.name, "name")
        _host_float(self, "center_nm")
        if not isinstance(self.magnitude_system, str) or self.magnitude_system not in (
            "vega",
            "ab",
        ):
            raise ValueError("magnitude_system must be 'vega' or 'ab'")


@dataclass(frozen=True)
class Viewing:
    """Calibration inclination and position angle in degrees; face-on is 0."""

    inclination_deg: float
    position_angle_deg: float

    def __post_init__(self):
        """Validate static viewing angles."""
        _host_float(self, "inclination_deg")
        _host_float(self, "position_angle_deg")


@dataclass(frozen=True)
class ReferenceStar:
    """Reference absolute magnitude in the declared band and luminosity.

    ``luminosity_lsun`` is bolometric, in L_sun. The solar V reference has
    ``band_abs_mag=MV_SUN`` and ``luminosity_lsun=1``. A different passband
    requires the caller's matching reference magnitude and calibration.
    """

    band_abs_mag: float
    luminosity_lsun: float

    def __post_init__(self):
        """Validate static reference-star quantities."""
        _host_float(self, "band_abs_mag")
        _host_float(self, "luminosity_lsun")


_V_BAND = Passband("V", V_BAND_WAVELENGTH_NM, "vega")
_FACE_ON = Viewing(0.0, 0.0)
_SUN = ReferenceStar(MV_SUN, 1.0)


@dataclass(frozen=True, kw_only=True)
class AbstractDustAmplitude(ABC):
    """Dust level plus static calibration metadata, all keyword-only.

    ``level`` is unchecked and may be an array or tracer. ``reference_star``
    is a ReferenceStar or None. ``reference_radius_au`` is a finite host
    radius in AU or 'eeid', meaning sqrt(L / L_sun) AU. ``passband`` and
    ``viewing`` describe the calibration, and ``calibration_id`` is required
    nonempty free text. Concrete families can supply calibration defaults.

    Metadata compare and hash by value. Entire records also have dataclass
    equality and hashing when their level supports those operations (e.g.
    a scalar); array levels should be compared with array operations.
    """

    level: Any
    calibration_id: str
    reference_star: ReferenceStar | None = None
    reference_radius_au: float | str = 1.0
    passband: Passband = _V_BAND
    viewing: Viewing = _FACE_ON

    @property
    @abstractmethod
    def family(self) -> str:
        """Definition name, fixed as a class constant by each concrete type."""

    def __post_init__(self):
        """Validate metadata without inspecting the dynamic level."""
        _required_string(self.calibration_id, "calibration_id")
        if self.reference_star is not None and not isinstance(
            self.reference_star, ReferenceStar
        ):
            raise ValueError("reference_star must be a ReferenceStar or None")
        if isinstance(self.reference_radius_au, str):
            if self.reference_radius_au != "eeid":
                raise ValueError("reference_radius_au must be a float or 'eeid'")
        else:
            _host_float(self, "reference_radius_au")
        if not isinstance(self.passband, Passband):
            raise ValueError("passband must be a Passband")
        if not isinstance(self.viewing, Viewing):
            raise ValueError("viewing must be a Viewing")


@dataclass(frozen=True, kw_only=True)
class SurfaceBrightnessDust(AbstractDustAmplitude):
    """Stark et al. (2014, ApJ 795, 122) surface-brightness convention.

    Physically, one zodi is a fixed vertical optical depth at the EEID.
    Scattered surface brightness in passband X is proportional to the
    stellar flux density in X at the dust, so at the EEID it is
    ``n I_X,sun (L_X / L_X,sun) / (L / L_sun)``. The 'band' rule uses
    the host and reference absolute magnitudes in that same passband;
    the 'bolometric' rule assumes solar colour for every host and uses
    ``host_luminosity_lsun`` instead of the band luminosity ratio.

    ``mag_per_arcsec2`` calibrates one zodi around the reference star
    (default: solar V, 22 mag arcsec-2 at 1 AU, also the solar EEID).
    ``reference_radius_au`` and ``viewing`` label the calibration; methods
    apply the declared inverse-square law in AU, with no implicit radius
    renormalization, band correction, or inclination correction. A custom
    band/reference requires its own caller-supplied magnitude calibration.
    """

    family: ClassVar[str] = "surface-brightness"
    calibration_id: str = "stark-2014"
    reference_star: ReferenceStar | None = _SUN
    reference_radius_au: float | str = "eeid"
    mag_per_arcsec2: float = MAG_1ZODI_V_ARCSEC2
    luminosity_rule: str = "band"

    def __post_init__(self):
        """Validate the fixed brightness calibration and luminosity rule."""
        super().__post_init__()
        _host_float(self, "mag_per_arcsec2")
        if not isinstance(self.luminosity_rule, str) or self.luminosity_rule not in (
            "band",
            "bolometric",
        ):
            raise ValueError("luminosity_rule must be 'band' or 'bolometric'")
        if self.luminosity_rule == "band" and self.reference_star is None:
            raise ValueError("reference_star is required for the band rule")

    def flux_ratio(self, *, host_band_abs_mag, host_luminosity_lsun, r_au):
        """Return zero-magnitude-referenced brightness per square arcsecond.

        Evaluate ``level * 10**(-0.4 mag_per_arcsec2) * S / r_au**2``,
        where S is the declared band or bolometric host luminosity ratio.
        The band rule calls the existing V-chain algebra, with the declared
        reference magnitude, and is bitwise identical for solar V inputs.
        Host magnitudes must use ``passband.magnitude_system``; luminosity
        is bolometric in L_sun, and radius is in AU. The result is
        dimensionless per arcsec^2 relative to the passband zero-magnitude
        flux density, not relative to the host star.
        """
        xp = array_namespace(self.level, host_band_abs_mag, host_luminosity_lsun, r_au)
        if self.luminosity_rule == "band":
            return exozodi_flux_ratio_v(
                xp.asarray(self.level),
                xp.asarray(host_band_abs_mag),
                xp.asarray(r_au),
                mag_1zodi=self.mag_per_arcsec2,
                mv_sun=self.reference_star.band_abs_mag,
            )
        return (
            xp.asarray(self.level)
            * 10.0 ** (-0.4 * self.mag_per_arcsec2)
            * xp.asarray(host_luminosity_lsun)
            / xp.asarray(r_au) ** 2
        )

    def at_eeid(self, *, host_band_abs_mag, host_luminosity_lsun):
        """Return the flux ratio at sqrt(host_luminosity_lsun) AU."""
        xp = array_namespace(self.level, host_band_abs_mag, host_luminosity_lsun)
        return self.flux_ratio(
            host_band_abs_mag=host_band_abs_mag,
            host_luminosity_lsun=host_luminosity_lsun,
            r_au=xp.sqrt(xp.asarray(host_luminosity_lsun)),
        )

    def mag_per_arcsec2_at(self, *, host_band_abs_mag, host_luminosity_lsun, r_au):
        """Return brightness in the declared system's mag per arcsec^2."""
        return flux_ratio_to_mag(
            self.flux_ratio(
                host_band_abs_mag=host_band_abs_mag,
                host_luminosity_lsun=host_luminosity_lsun,
                r_au=r_au,
            )
        )


@dataclass(frozen=True, kw_only=True)
class LocalZodiMultiple(AbstractDustAmplitude):
    """Multiplier of local zodiacal surface brightness at a fixed pointing.

    ``solar_longitude_deg`` is the helio-ecliptic longitude difference,
    target longitude minus solar longitude (``dlon_deg`` in the local-zodi
    functions), not an absolute solar longitude. ``ecliptic_latitude_deg``
    is ``beta_deg``. Both follow the existing absolute-value symmetry and
    table-edge clamping conventions. Reference star/radius, passband, and
    viewing are metadata; the local model itself is wavelength-resolved.
    """

    family: ClassVar[str] = "local-zodi-multiple"
    calibration_id: str = "leinert-1998-table17"
    ecliptic_latitude_deg: float
    solar_longitude_deg: float

    def __post_init__(self):
        """Validate static pointing angles."""
        super().__post_init__()
        _host_float(self, "ecliptic_latitude_deg")
        _host_float(self, "solar_longitude_deg")

    def specific_intensity(self, wavelength_nm):
        """Return level times local intensity in W m-2 sr-1 um-1.

        Wavelength is in nm. The existing local-zodi function supplies the
        spectrum with its default Table 17 anchoring and power units.
        """
        xp = array_namespace(self.level, wavelength_nm)
        return xp.asarray(self.level) * specific_intensity(
            xp.asarray(self.solar_longitude_deg),
            xp.asarray(self.ecliptic_latitude_deg),
            xp.asarray(wavelength_nm),
        )


@dataclass(frozen=True, kw_only=True)
class DensityDust(AbstractDustAmplitude):
    """Bare multiplier of the density profile named by required profile_id.

    No surface-brightness method or conversion is provided. Converting to
    or from surface brightness needs the profile's line-of-sight integral,
    which lives with the profile's owner.
    """

    family: ClassVar[str] = "density"
    profile_id: str

    def __post_init__(self):
        """Require an explicit density-profile identity."""
        super().__post_init__()
        _required_string(self.profile_id, "profile_id")


def _flatten_amplitude(record):
    """Partition the dynamic level from value-comparable static metadata."""
    static = tuple(
        (field.name, getattr(record, field.name))
        for field in fields(record)
        if field.name != "level"
    )
    return (record.level,), static


def _unflatten_amplitude(record_type, static, children):
    """Reconstruct without interpreting JAX's dynamic placeholder values."""
    return record_type(level=children[0], **dict(static))


def _register_jax_pytrees():
    """Register only when the optional JAX package can be imported."""
    try:
        from jax.tree_util import register_pytree_node
    except ImportError:
        return
    for record_type in (SurfaceBrightnessDust, LocalZodiMultiple, DensityDust):
        register_pytree_node(
            record_type,
            _flatten_amplitude,
            partial(_unflatten_amplitude, record_type),
        )


_register_jax_pytrees()
