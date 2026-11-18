"""Analytic exozodiacal radiance in the scalar AYO calculator dialect."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from zodi.exozodi import MAG_1ZODI_V_ARCSEC2, MV_SUN, exozodi_flux_ratio_v
from zodi.units import (
    intensity_from_ab_mag_arcsec2,
    power_to_photon_intensity,
)


class AYOExozodi(eqx.Module):
    """Scalar AYO exozodi with flat AB colour and inverse-square illumination.

    At 1 AU, Fexozodi = level_zodis * 10**(-0.4 * brightness), per arcsec^2,
    where brightness is surface_brightness_mag_arcsec2 for one zodi.
    Its denominator is the AB magnitude-zero flux density F0(lambda), from
    zodi.units.AB_ZERO_POINT_JY, not the host-star flux. Radiance is F0 * Fexozodi
    divided by arcsec2rad**2 and separation_au**2. There is no host colour,
    luminosity, or inclination correction. The analytic AB zero point has
    no table knots; photon colour is 1/lambda (constant F_nu).

    One 22-mag/arcsec^2 zodi is 10^-8.8 relative to a magnitude-zero source.
    The alternative approximately 10^-6.87 ratio uses a solar absolute-V
    magnitude 4.83 source at 10 pc as denominator. Face-on and inclined
    definitions also differ by a factor 0.49; no such conversion is applied
    here. These dialects must not be substituted without conversion.

    separation_au is the projected physical separation dist_pc * sep_arcsec,
    not the orbital radius. The scalar law is singular at zero separation.
    """

    level_zodis: float | Array = 3.0
    surface_brightness_mag_arcsec2: float | Array = MAG_1ZODI_V_ARCSEC2

    def spectral_breaks_nm(self) -> np.ndarray:
        """The analytic flat-AB spectrum has no wavelength breaks."""
        return np.empty(0)

    def photon_radiance(
        self, wavelength_nm: float | Array, *, separation_au: float | Array
    ) -> Array:
        """Return ph s^-1 m^-2 nm^-1 sr^-1 at each wavelength and separation."""
        wave = jnp.asarray(wavelength_nm)
        ratio = exozodi_flux_ratio_v(
            self.level_zodis,
            MV_SUN,
            jnp.asarray(separation_au),
            mag_1zodi=self.surface_brightness_mag_arcsec2,
        )
        intensity = intensity_from_ab_mag_arcsec2(0.0, wave) * ratio
        return power_to_photon_intensity(intensity, wave) / 1000.0
