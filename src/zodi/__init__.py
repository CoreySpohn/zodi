"""Zodiacal and exozodiacal light brightness conventions.

One implementation serves numpy and JAX through the array API standard:
every function computes in the namespace of its array inputs, so numpy
callers stay numpy end to end while JAX callers get functions that jit,
vmap, and differentiate natively.

The package owns the conventions shared by the exoplanet imaging yield
and exposure-time codes: the Leinert et al. (1998) local zodi tables and
their interpolation (``zodi.zodi``), the Stark et al. (2014) / EXOSIMS
exozodi chain including the grey-scatterer band correction
(``zodi.exozodi``), and the conversions between the surface-brightness
dialects in use across codes (``zodi.units``). Geometry, star catalogs,
spectra, and bandpasses are caller inputs, never package state.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from zodi._version import __version__
from zodi.exozodi import (
    MAG_1ZODI_V_ARCSEC2,
    MV_SUN,
    T_DUST_K,
    blackbody_spectral_radiance,
    exozodi_flux_ratio_band,
    exozodi_flux_ratio_v,
    fit_grey_scatter_constants,
    grey_scatter_intensity,
    jez0,
    latitudinal_factor,
    scale_jez,
    theta_from_inclination,
)
from zodi.zodi import (
    AYO_DEFAULT_ZODI_MAG_V,
    V_BAND_WAVELENGTH_NM,
    ab_color_correction,
    ayo_default_ab_mag,
    color_correction,
    position_factor,
    specific_intensity,
    specific_intensity_at_location,
    surface_brightness_ab_mag,
    zodi_flux_ratio,
)

if TYPE_CHECKING:
    from zodi.amplitude import (
        AbstractDustAmplitude,
        DensityDust,
        LocalZodiMultiple,
        Passband,
        ReferenceStar,
        SurfaceBrightnessDust,
        Viewing,
    )

_AMPLITUDE_NAMES = {
    "AbstractDustAmplitude",
    "DensityDust",
    "LocalZodiMultiple",
    "Passband",
    "ReferenceStar",
    "SurfaceBrightnessDust",
    "Viewing",
}


def __getattr__(name):
    """Load amplitude records, including optional PyTree registration, on demand."""
    if name in _AMPLITUDE_NAMES:
        value = getattr(import_module("zodi.amplitude"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AYO_DEFAULT_ZODI_MAG_V",
    "MAG_1ZODI_V_ARCSEC2",
    "MV_SUN",
    "T_DUST_K",
    "V_BAND_WAVELENGTH_NM",
    "AbstractDustAmplitude",
    "DensityDust",
    "LocalZodiMultiple",
    "Passband",
    "ReferenceStar",
    "SurfaceBrightnessDust",
    "Viewing",
    "__version__",
    "ab_color_correction",
    "ayo_default_ab_mag",
    "blackbody_spectral_radiance",
    "color_correction",
    "exozodi_flux_ratio_band",
    "exozodi_flux_ratio_v",
    "fit_grey_scatter_constants",
    "grey_scatter_intensity",
    "jez0",
    "latitudinal_factor",
    "position_factor",
    "scale_jez",
    "specific_intensity",
    "specific_intensity_at_location",
    "surface_brightness_ab_mag",
    "theta_from_inclination",
    "zodi_flux_ratio",
]
