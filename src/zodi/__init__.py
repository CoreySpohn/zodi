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

from zodi.exozodi import (
    MAG_1ZODI_V_ARCSEC2,
    MV_SUN,
    T_DUST_K,
    band_average,
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
    color_correction,
    position_factor,
    specific_intensity,
    specific_intensity_at_location,
    surface_brightness_ab_mag,
    zodi_flux_ratio,
)

__all__ = [
    "MAG_1ZODI_V_ARCSEC2",
    "MV_SUN",
    "T_DUST_K",
    "band_average",
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
