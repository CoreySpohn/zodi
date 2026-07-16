"""Local zodiacal light surface brightness from the Leinert (1998) tables.

The model is the one shared by the exoplanet yield and exposure-time codes
(following Stark et al. 2014 and the EXOSIMS implementation): the Table 17
position dependence evaluated at 500 nm, times the Table 19 wavelength
dependence expressed as a color ratio,

    I(dlon, beta, lam) = I_17(dlon, beta) * I_19(lam) / I_19(500 nm).

Because Table 19 is tabulated at the (90 deg, 0 deg) reference geometry
where Table 17 reads 259e-8 W m-2 sr-1 um-1 while Table 19 itself reads
2.6e-6 at 0.5 um, the two tables disagree by about 0.4 percent at their
shared anchor point. Codes differ in which table they treat as absolute:
the form above anchors on Table 17 (``anchor="table17"``, the EXOSIMS
convention); anchoring on Table 19 with a normalized position factor
(``anchor="table19"``) reproduces implementations that build the absolute
scale from the wavelength table. Both are exposed and documented so
consumers can match their existing outputs exactly.

Wavelength interpolation is linear in log10-log10 space. Position
interpolation is bilinear, with out-of-range queries clamped to the table
edge and near-Sun exclusion cells filled with the nearest tabulated value
(see ``zodi.tables``).
"""

import numpy as np

from zodi import tables, units
from zodi._interp import interp1d_clamped, interp2d_bilinear_clamped
from zodi._xp import array_namespace

__all__ = [
    "color_correction",
    "position_factor",
    "specific_intensity",
    "specific_intensity_at_location",
    "surface_brightness_ab_mag",
    "zodi_flux_ratio",
]

_LOG_T19_LAM_UM = np.log10(tables.TABLE19_WAVELENGTH_UM)
_LOG_T19_RADIANCE = np.log10(tables.TABLE19_RADIANCE_W_M2_SR_UM)


def _table19_radiance(xp, wavelength_nm):
    """Table 19 spectral radiance at the reference geometry, W m-2 sr-1 um-1."""
    log_lam_um = xp.log10(xp.asarray(wavelength_nm) / 1000.0)
    log_radiance = interp1d_clamped(_LOG_T19_LAM_UM, _LOG_T19_RADIANCE, log_lam_um)
    return 10.0**log_radiance


def position_factor(dlon_deg, beta_deg):
    """Table 17 brightness relative to the (90 deg, 0 deg) reference cell.

    Args:
        dlon_deg: Helio-ecliptic longitude difference (target longitude
            minus solar longitude) in degrees. The table is symmetric, so
            the absolute value is used; values are clamped to [0, 180].
        beta_deg: Ecliptic latitude in degrees; symmetric, clamped to
            [0, 90].

    Returns:
        Dimensionless brightness factor, 1 at (90 deg, 0 deg).
    """
    xp = array_namespace(dlon_deg, beta_deg)
    lon = xp.abs(xp.asarray(dlon_deg))
    lat = xp.abs(xp.asarray(beta_deg))
    value = interp2d_bilinear_clamped(
        tables.LON_GRID_DEG, tables.BETA_GRID_DEG, tables.TABLE17_FILLED, lon, lat
    )
    return value / tables.TABLE17_REFERENCE


def specific_intensity_at_location(dlon_deg, beta_deg, photon_units=False):
    """Table 17 specific intensity at 500 nm for the given viewing geometry.

    Args:
        dlon_deg: Helio-ecliptic longitude difference in degrees.
        beta_deg: Ecliptic latitude in degrees.
        photon_units: If True, return ph s-1 m-2 sr-1 um-1 instead of
            W m-2 sr-1 um-1.

    Returns:
        Specific intensity of the zodiacal light at 500 nm.
    """
    xp = array_namespace(dlon_deg, beta_deg)
    lon = xp.abs(xp.asarray(dlon_deg))
    lat = xp.abs(xp.asarray(beta_deg))
    value = interp2d_bilinear_clamped(
        tables.LON_GRID_DEG, tables.BETA_GRID_DEG, tables.TABLE17_FILLED, lon, lat
    )
    intensity = value * tables.TABLE17_UNIT_W_M2_SR_UM
    if photon_units:
        return units.power_to_photon_intensity(intensity, 500.0)
    return intensity


def color_correction(wavelength_nm, reference_wavelength_nm=500.0, photon_units=False):
    """Table 19 color ratio relative to a reference wavelength.

    Multiplicative factor scaling a zodiacal brightness known at the
    reference wavelength to another wavelength. In photon units the ratio
    gains exactly ``lambda / lambda_ref`` relative to power units; the
    units of the factor must match the units of the quantity being scaled.

    Args:
        wavelength_nm: Target wavelength in nm.
        reference_wavelength_nm: Reference wavelength in nm. The Leinert
            tables anchor at 500 nm (the default); some implementations
            use the V-band effective wavelength 550 nm instead.
        photon_units: If True, return the photon-unit ratio.

    Returns:
        Dimensionless color correction factor.
    """
    xp = array_namespace(wavelength_nm)
    factor = _table19_radiance(xp, wavelength_nm) / _table19_radiance(
        xp, reference_wavelength_nm
    )
    if photon_units:
        return factor * units.photon_ratio_factor(
            wavelength_nm, reference_wavelength_nm
        )
    return factor


def specific_intensity(
    dlon_deg, beta_deg, wavelength_nm, photon_units=False, anchor="table17"
):
    """Zodiacal light specific intensity at the given geometry and wavelength.

    Args:
        dlon_deg: Helio-ecliptic longitude difference in degrees.
        beta_deg: Ecliptic latitude in degrees.
        wavelength_nm: Wavelength in nm.
        photon_units: If True, return ph s-1 m-2 sr-1 um-1 instead of
            W m-2 sr-1 um-1.
        anchor: ``"table17"`` scales the absolute Table 17 value by the
            Table 19 color ratio (EXOSIMS convention); ``"table19"``
            scales the absolute Table 19 spectrum by the normalized
            position factor. The two differ by a constant factor of about
            1.004 (see module docstring).

    Returns:
        Specific intensity of the zodiacal light.
    """
    if anchor not in ("table17", "table19"):
        raise ValueError("anchor must be 'table17' or 'table19'")
    xp = array_namespace(dlon_deg, beta_deg, wavelength_nm)
    if anchor == "table17":
        intensity = specific_intensity_at_location(
            dlon_deg, beta_deg
        ) * color_correction(wavelength_nm)
    else:
        intensity = _table19_radiance(xp, wavelength_nm) * position_factor(
            dlon_deg, beta_deg
        )
    if photon_units:
        return units.power_to_photon_intensity(intensity, wavelength_nm)
    return intensity


def surface_brightness_ab_mag(dlon_deg, beta_deg, wavelength_nm, anchor="table17"):
    """Zodiacal light surface brightness in AB magnitudes per square arcsecond.

    Args:
        dlon_deg: Helio-ecliptic longitude difference in degrees.
        beta_deg: Ecliptic latitude in degrees.
        wavelength_nm: Wavelength in nm.
        anchor: Table anchoring convention; see :func:`specific_intensity`.

    Returns:
        AB surface brightness in magnitudes per square arcsecond.
    """
    intensity = specific_intensity(dlon_deg, beta_deg, wavelength_nm, anchor=anchor)
    return units.ab_mag_arcsec2_from_intensity(intensity, wavelength_nm)


def zodi_flux_ratio(dlon_deg, beta_deg, wavelength_nm, f0_ph_s_m2_nm, anchor="table17"):
    """Local zodi in the flux-ratio dialect (EXOSIMS ``fZ``, jaxedith ``Fzodi``).

    The photon-unit specific intensity per square arcsecond divided by the
    zero-magnitude flux density of the observing band, following the
    Stark/EXOSIMS convention
    ``fZ = I_photon / arcsec2 / (F0 / dlambda)``.

    Args:
        dlon_deg: Helio-ecliptic longitude difference in degrees.
        beta_deg: Ecliptic latitude in degrees.
        wavelength_nm: Wavelength in nm.
        f0_ph_s_m2_nm: Zero-magnitude flux density of the band in
            ph s-1 m-2 nm-1 (the caller's photometric system; EXOSIMS
            supplies ``mode["F0"] / mode["deltaLam"]``).
        anchor: Table anchoring convention; see :func:`specific_intensity`.

    Returns:
        Surface brightness as a dimensionless flux ratio per square
        arcsecond.
    """
    xp = array_namespace(dlon_deg, beta_deg, wavelength_nm, f0_ph_s_m2_nm)
    intensity_photon = specific_intensity(
        dlon_deg, beta_deg, wavelength_nm, photon_units=True, anchor=anchor
    )
    per_arcsec2_per_nm = intensity_photon / units.ARCSEC2_PER_SR / 1000.0
    return per_arcsec2_per_nm / xp.asarray(f0_ph_s_m2_nm)
