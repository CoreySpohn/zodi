"""Conversions between the surface-brightness dialects in common use.

The zodi and exozodi literature and codes speak four dialects for the same
quantity: magnitudes per square arcsecond, dimensionless flux ratio per
square arcsecond (relative to a zero-magnitude flux density), spectral
radiance in power units, and photon-rate units. This module is the
conversion table between them, with every constant written out.

Physical constants are the exact SI values (2019 redefinition):
c = 299792458 m/s, h = 6.62607015e-34 J s. The AB zero point follows the
Oke & Gunn convention, 10**(23 - 48.6/2.5) Jy, evaluated exactly rather
than rounded to 3631 Jy.

All functions run on numpy or jax inputs (see ``zodi._xp``).
"""

import numpy as np

from zodi._xp import array_namespace

__all__ = [
    "AB_ZERO_POINT_JY",
    "ARCSEC2_PER_SR",
    "C_M_PER_S",
    "H_J_S",
    "ab_mag_arcsec2_from_intensity",
    "flux_ratio_to_mag",
    "intensity_from_ab_mag_arcsec2",
    "intensity_to_mjy_per_sr",
    "mag_to_flux_ratio",
    "photon_ratio_factor",
    "power_to_photon_intensity",
]

C_M_PER_S = 299792458.0
H_J_S = 6.62607015e-34
AB_ZERO_POINT_JY = 10.0 ** (23.0 - 48.6 / 2.5)
ARCSEC2_PER_SR = float((180.0 / np.pi * 3600.0) ** 2)


def mag_to_flux_ratio(mag_arcsec2):
    """Surface brightness magnitude to dimensionless flux ratio.

    The flux-ratio dialect (EXOSIMS ``fZ``/``fEZ``, jaxedith
    ``Fzodi``/``Fexozodi``, pyEDITH) expresses surface brightness as
    ``10**(-0.4 * mag)`` per square arcsecond, relative to whatever
    zero-magnitude flux density the consumer multiplies back in.

    Args:
        mag_arcsec2: Surface brightness in magnitudes per square arcsecond.

    Returns:
        Flux ratio per square arcsecond (dimensionless).
    """
    xp = array_namespace(mag_arcsec2)
    return 10.0 ** (-0.4 * xp.asarray(mag_arcsec2))


def flux_ratio_to_mag(flux_ratio_arcsec2):
    """Inverse of :func:`mag_to_flux_ratio`.

    Args:
        flux_ratio_arcsec2: Flux ratio per square arcsecond.

    Returns:
        Surface brightness in magnitudes per square arcsecond.
    """
    xp = array_namespace(flux_ratio_arcsec2)
    return -2.5 * xp.log10(xp.asarray(flux_ratio_arcsec2))


def power_to_photon_intensity(intensity_w_m2_sr_um, wavelength_nm):
    """Convert power specific intensity to photon-rate specific intensity.

    Divides by the photon energy ``h c / lambda``, so the returned quantity
    is in ph s-1 m-2 sr-1 um-1 when the input is in W m-2 sr-1 um-1.

    Args:
        intensity_w_m2_sr_um: Specific intensity in W m-2 sr-1 um-1.
        wavelength_nm: Wavelength in nm at which the intensity applies.

    Returns:
        Photon-rate specific intensity in ph s-1 m-2 sr-1 um-1.
    """
    xp = array_namespace(intensity_w_m2_sr_um, wavelength_nm)
    wavelength_m = xp.asarray(wavelength_nm) * 1e-9
    photon_energy_j = H_J_S * C_M_PER_S / wavelength_m
    return xp.asarray(intensity_w_m2_sr_um) / photon_energy_j


def photon_ratio_factor(wavelength_nm, reference_wavelength_nm):
    """Factor converting a power-unit intensity ratio to a photon-unit ratio.

    A ratio of intensities at two wavelengths changes value between power
    and photon units by exactly ``lambda / lambda_ref`` (the photon-energy
    ratio); this is the whole difference between the two conventions noted
    in the EXOSIMS documentation.

    Args:
        wavelength_nm: Numerator wavelength in nm.
        reference_wavelength_nm: Denominator (reference) wavelength in nm.

    Returns:
        The multiplicative factor ``lambda / lambda_ref``.
    """
    xp = array_namespace(wavelength_nm, reference_wavelength_nm)
    return xp.asarray(wavelength_nm) / xp.asarray(reference_wavelength_nm)


def intensity_to_mjy_per_sr(intensity_w_m2_sr_um, wavelength_um):
    """Convert specific intensity from per-wavelength to MJy per steradian.

    Applies ``I_nu = I_lambda * lambda**2 / c``. With ``I_lambda`` in
    W m-2 sr-1 um-1 and the wavelength in um, the numeric factor is
    ``wavelength**2 * 1e14 / c``.

    Args:
        intensity_w_m2_sr_um: Specific intensity in W m-2 sr-1 um-1.
        wavelength_um: Wavelength in um.

    Returns:
        Specific intensity in MJy sr-1.
    """
    xp = array_namespace(intensity_w_m2_sr_um, wavelength_um)
    lam = xp.asarray(wavelength_um)
    return xp.asarray(intensity_w_m2_sr_um) * lam**2 * 1e14 / C_M_PER_S


def ab_mag_arcsec2_from_intensity(intensity_w_m2_sr_um, wavelength_nm):
    """Convert power specific intensity to AB magnitudes per square arcsecond.

    The per-steradian intensity is scaled to one square arcsecond,
    converted to a per-frequency flux density with
    ``f_nu = f_lambda * lambda**2 / c``, and referenced to the AB zero
    point.

    Args:
        intensity_w_m2_sr_um: Specific intensity in W m-2 sr-1 um-1.
        wavelength_nm: Wavelength in nm.

    Returns:
        AB surface brightness in magnitudes per square arcsecond.
    """
    xp = array_namespace(intensity_w_m2_sr_um, wavelength_nm)
    lam_um = xp.asarray(wavelength_nm) / 1000.0
    per_arcsec2 = xp.asarray(intensity_w_m2_sr_um) / ARCSEC2_PER_SR
    flux_jy = per_arcsec2 * lam_um**2 * 1e14 / C_M_PER_S * 1e6
    return -2.5 * xp.log10(flux_jy / AB_ZERO_POINT_JY)


def intensity_from_ab_mag_arcsec2(mag_arcsec2, wavelength_nm):
    """Inverse of :func:`ab_mag_arcsec2_from_intensity`.

    Args:
        mag_arcsec2: AB surface brightness in magnitudes per square
            arcsecond.
        wavelength_nm: Wavelength in nm.

    Returns:
        Specific intensity in W m-2 sr-1 um-1.
    """
    xp = array_namespace(mag_arcsec2, wavelength_nm)
    lam_um = xp.asarray(wavelength_nm) / 1000.0
    flux_jy = AB_ZERO_POINT_JY * 10.0 ** (-0.4 * xp.asarray(mag_arcsec2))
    per_arcsec2 = flux_jy / 1e6 / (lam_um**2 * 1e14 / C_M_PER_S)
    return per_arcsec2 * ARCSEC2_PER_SR
