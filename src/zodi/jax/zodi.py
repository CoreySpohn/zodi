"""Local zodiacal photon radiance, with physics delegated to zodi."""

from abc import abstractmethod
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import um_to_nm
from jax import Array

import zodi
from zodi._interp import interp1d_clamped
from zodi.tables import TABLE19_WAVELENGTH_UM
from zodi.units import intensity_from_ab_mag_arcsec2, power_to_photon_intensity


class AbstractLocalZodi(eqx.Module):
    """Local background radiance in ph s^-1 m^-2 nm^-1 sr^-1.

    Every implementation preserves the wavelength array's shape. Pointing
    angles are degrees; solar_longitude_deg is target minus solar longitude
    (helio-ecliptic longitude difference), as in the zodi tables.
    """

    @abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the model's spectral breakpoints in nm as a host array."""

    @abstractmethod
    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Evaluate photon radiance at wavelengths in nm, without integration."""


@final
class AYOZodi(AbstractLocalZodi):
    """Fixed 22 AB mag/arcsec^2 at 550 nm with Table 19 spectral color.

    Both pointing keywords are accepted and ignored. The reference magnitude
    parameter sets the fixed apparent brightness; zodi owns the color model.
    """

    surface_brightness_mag_arcsec2: float = 22.0
    reference_wavelength_nm: float = 550.0

    def spectral_breaks_nm(self) -> np.ndarray:
        """Table 19 color breakpoints [nm]; the spectrum is piecewise in them."""
        return um_to_nm(TABLE19_WAVELENGTH_UM)

    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Photon radiance per sr; pointing is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        mag = zodi.ayo_default_ab_mag(
            wave,
            mag_v=self.surface_brightness_mag_arcsec2,
            reference_wavelength_nm=self.reference_wavelength_nm,
        )
        return (
            power_to_photon_intensity(intensity_from_ab_mag_arcsec2(mag, wave), wave)
            / 1000.0
        )


@final
class LeinertZodi(AbstractLocalZodi):
    """Pointing-dependent Leinert radiance anchored on absolute Table 19.

    The absolute scale comes from zodi.specific_intensity, anchor='table19'.
    There is no adjustable reference magnitude. At (longitude=90, latitude=0)
    the V-band brightness is approximately 22 AB mag/arcsec^2; AYO's rounded
    22-mag normalization differs slightly from the absolute table value.
    """

    def spectral_breaks_nm(self) -> np.ndarray:
        """Table 19 color breakpoints [nm]; the spectrum is piecewise in them."""
        return um_to_nm(TABLE19_WAVELENGTH_UM)

    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Photon radiance per sr, converted from zodi's per-micron output."""
        return (
            zodi.specific_intensity(
                jnp.asarray(solar_longitude_deg),
                jnp.asarray(ecliptic_latitude_deg),
                jnp.asarray(wavelength_nm),
                photon_units=True,
                anchor="table19",
            )
            / 1000.0
        )


@final
class PrecomputedZodi(AbstractLocalZodi):
    """Linearly interpolated photon radiance supplied in per-steradian units.

    Both pointing keywords are accepted and ignored. Values outside the
    supplied wavelength support are NaN.
    """

    wavelengths_nm: Array = eqx.field(converter=jnp.asarray)
    radiance_phot_per_s_m2_nm_sr: Array = eqx.field(converter=jnp.asarray)

    def __init__(self, wavelengths_nm: Array, radiance_phot_per_s_m2_nm_sr: Array):
        """Store a photon radiance table in ph s^-1 m^-2 nm^-1 sr^-1."""
        self.wavelengths_nm = jnp.asarray(wavelengths_nm)
        self.radiance_phot_per_s_m2_nm_sr = jnp.asarray(radiance_phot_per_s_m2_nm_sr)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the tabulated radiance wavelengths in nm on the host."""
        return np.asarray(self.wavelengths_nm)

    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Photon radiance per sr; pointing is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        radiance = interp1d_clamped(
            self.wavelengths_nm, self.radiance_phot_per_s_m2_nm_sr, wave
        )
        supported = (wave >= self.wavelengths_nm[0]) & (wave <= self.wavelengths_nm[-1])
        return jnp.where(supported, radiance, jnp.nan)
