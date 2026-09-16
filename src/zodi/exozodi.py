"""Exozodiacal light conventions following Stark et al. (2014) and EXOSIMS.

One zodi of exozodiacal dust has the optical depth of the solar zodiacal
cloud at 1 AU, placed at the Earth-equivalent instellation distance
(EEID, ``sqrt(L)`` AU) of the target star, where it receives the same
bolometric insolation. For a solar twin its V-band surface brightness
there is ``x = 22`` mag arcsec-2. Stark et al. (2014, Eq. C4) gives the
surface brightness at the EEID,

    I_EZ(EEID) = n_EZ * F0V * 10**(-0.4 (MV - MV_sun)) * 10**(-0.4 x) / L

and, for a radially flat optical depth, the surface brightness at any
other radius follows the ``1/r**2`` illumination falloff measured from
the EEID, ``I_EZ(r) = I_EZ(EEID) * (EEID / r)**2``. Because
``EEID**2 = L``, the luminosity cancels:

    I_EZ(r) = n_EZ * F0V * 10**(-0.4 (MV - MV_sun)) * 10**(-0.4 x)
              * f_lambda * f(theta) / r**2

with ``r`` in AU, ``L`` in solar luminosities, and ``MV_sun = 4.83``.
The stellar V-band luminosity enters through the magnitude term; the
bolometric luminosity only sets where the EEID lies. (For a radial
profile ``tau ~ r**-gamma`` the exponent becomes ``2 + gamma``; this
module adopts the flat profile.) Rescaling from 1 AU while also keeping
the ``1/L`` of Eq. C4, as in the EXOSIMS Fundamental Concepts
documentation, counts the luminosity twice.

The band correction ``f_lambda`` rests on one physical assumption: the
dust is a GREY SCATTERER, so the scattered component of the exozodi
spectrum is proportional to the spectrum of the illuminating star. Two
flavors of the same assumption are in production use:

- pyEDITH scales by the star's apparent color directly,
  ``f_lambda = 10**(-0.4 (m_band - m_V))`` (scattered light only).
- EXOSIMS adds the dust's thermal emission: the exozodi spectrum is
  modeled as ``F_star(lam) * f_star + B_lam(261.5 K) * f_thermal`` with
  the scattered term truncated beyond 10 um, the two constants calibrated
  once against the local zodi wavelength dependence (Leinert Table 19),
  and ``f_lambda`` computed as the ratio of bandpass-averaged intensities
  between the observing band and V.

This module implements both flavors. Because the calibration model is
linear in ``(f_star, f_thermal)``, the fit is a closed-form least-squares
solve (:func:`fit_grey_scatter_constants`); the constants therefore adapt
to whatever spectral units and normalization the caller's star spectrum
uses, rather than being magic numbers tied to one code's spectrum
handling.

Inclination handling follows the EXOSIMS convention: the latitudinal
factor is evaluated at ``theta = 90 deg - inclination`` after folding the
inclination into [0, 90] deg, with three published models available (see
:func:`latitudinal_factor`).
"""

import numpy as np

from zodi import tables
from zodi._interp import interp1d_clamped
from zodi._xp import array_namespace

__all__ = [
    "MAG_1ZODI_V_ARCSEC2",
    "MAX_WAVELENGTH_UM",
    "MV_SUN",
    "T_DUST_K",
    "band_average",
    "blackbody_spectral_radiance",
    "exozodi_flux_ratio_band",
    "exozodi_flux_ratio_v",
    "fit_grey_scatter_constants",
    "grey_scatter_intensity",
    "jez0",
    "latitudinal_factor",
    "scale_jez",
    "theta_from_inclination",
]

# Absolute V magnitude of the Sun adopted by Stark et al. (2014) and EXOSIMS.
MV_SUN = 4.83
# V-band surface brightness of 1 zodi at the EEID (Stark et al. 2014).
MAG_1ZODI_V_ARCSEC2 = 22.0
# Local zodi dust temperature reported by Leinert et al. (1998).
T_DUST_K = 261.5

# Wavelength arguments in this module are in um. A value above this bound is
# almost certainly a nm value passed by mistake (the zodi and units modules
# take nm), since the Leinert Table 19 spectrum ends at 140 um.
MAX_WAVELENGTH_UM = 200.0

_C_M_PER_S = 299792458.0
_H_J_S = 6.62607015e-34
_KB_J_PER_K = 1.380649e-23


def _check_wavelength_um(xp, wavelength_um):
    """Reject a um wavelength argument that looks like nm.

    Only concrete numpy inputs are checked; JAX inputs (which may be
    tracers under ``jit``) pass through unchecked.
    """
    if xp is not array_namespace():
        return
    if np.nanmax(np.asarray(wavelength_um)) > MAX_WAVELENGTH_UM:
        raise ValueError(
            f"wavelength_um exceeds {MAX_WAVELENGTH_UM} um; exozodi functions take "
            "um (the zodi and units modules take nm)"
        )


def exozodi_flux_ratio_v(
    nzodi, mv_star, r_au, *, mag_1zodi=MAG_1ZODI_V_ARCSEC2, mv_sun=MV_SUN
):
    """V-band exozodi surface brightness in the flux-ratio dialect.

    The Stark et al. (2014) chain before band and inclination corrections,
    ``n * 10**(-0.4 (MV - MV_sun)) * 10**(-0.4 x) / r**2``, dimensionless
    per square arcsecond relative to the V-band zero-magnitude flux
    density. It depends on the star only through its V-band luminosity.
    Evaluating at the EEID, ``r_au = sqrt(L)``, recovers Stark et al.
    (2014) Eq. C4.

    Args:
        nzodi: Exozodi level in zodis.
        mv_star: Absolute V magnitude of the star.
        r_au: Circumstellar radius in AU at which to evaluate.
        mag_1zodi: V surface brightness of one zodi at the EEID in
            mag arcsec-2.
        mv_sun: Absolute V magnitude of the Sun.

    Returns:
        Flux ratio per square arcsecond.
    """
    xp = array_namespace(nzodi, mv_star, r_au)
    return (
        xp.asarray(nzodi)
        * 10.0 ** (-0.4 * (xp.asarray(mv_star) - mv_sun))
        * 10.0 ** (-0.4 * mag_1zodi)
        / xp.asarray(r_au) ** 2
    )


def exozodi_flux_ratio_band(
    nzodi,
    mv_star,
    vmag_star,
    band_mag_star,
    mag_1zodi=MAG_1ZODI_V_ARCSEC2,
    mv_sun=MV_SUN,
):
    """Exozodi flux ratio in an arbitrary band via the stellar apparent color.

    The pyEDITH formulation of the grey-scatterer assumption: the V-band
    surface brightness is scaled by the target star's color,
    ``10**(-0.4 (m_band - m_V))``, so the exozodi spectrum follows the
    stellar spectrum exactly (no thermal term). There is no explicit
    luminosity or radius scaling, matching
    ``pyEDITH.astrophysical_scene.calc_exozodi_flux``: the V-band factor
    equals :func:`exozodi_flux_ratio_v` at ``r = 1`` AU. The Stark et al.
    (2014) Eq. C4 value at the EEID is this divided by the bolometric
    luminosity in solar units.

    Args:
        nzodi: Exozodi level in zodis.
        mv_star: Absolute V magnitude of the star.
        vmag_star: Apparent V magnitude of the star.
        band_mag_star: Apparent magnitude of the star in the observing
            band (same photometric system as ``vmag_star``).
        mag_1zodi: V surface brightness of one zodi in mag arcsec-2.
        mv_sun: Absolute V magnitude of the Sun.

    Returns:
        Flux ratio per square arcsecond, relative to the observing band's
        zero-magnitude flux density.
    """
    xp = array_namespace(nzodi, mv_star, vmag_star, band_mag_star)
    return (
        xp.asarray(nzodi)
        * 10.0 ** (-0.4 * mag_1zodi)
        * 10.0 ** (-0.4 * (xp.asarray(mv_star) - mv_sun))
        * 10.0 ** (-0.4 * (xp.asarray(band_mag_star) - xp.asarray(vmag_star)))
    )


def theta_from_inclination(inclination_deg):
    """Fold an orbital inclination into the latitudinal-model angle.

    Follows ``EXOSIMS.ZodiacalLight.calc_fbeta``: inclinations above
    180 deg wrap, values above 90 deg reflect (the dust disk is symmetric),
    and the model angle is ``theta = 90 deg - inclination`` so a face-on
    system (inclination 0) looks through the dust pole (theta 90 deg) and
    an edge-on system (90 deg) looks along the midplane (theta 0).

    Args:
        inclination_deg: Orbital inclination(s) in degrees.

    Returns:
        Model angle theta in degrees, in [0, 90].
    """
    xp = array_namespace(inclination_deg)
    beta = xp.asarray(inclination_deg)
    beta = xp.where(beta > 180.0, beta - 180.0, beta)
    beta = xp.where(beta > 90.0, 180.0 - beta, beta)
    return 90.0 - beta


def latitudinal_factor(theta_deg, model="leinert", interp_lon_deg=135.0):
    """Latitudinal brightness factor of the dust disk.

    Three published models (see the EXOSIMS Fundamental Concepts
    documentation for their comparison figure):

    - ``"leinert"``: the Leinert Table 17 latitude profile at a fixed
      helio-ecliptic longitude (default 135 deg, near the local zodi
      minimum), normalized to its in-plane value. EXOSIMS interpolates
      this profile with a cubic spline; this implementation is piecewise
      linear, which agrees at every tabulated latitude and differs by
      under one percent between nodes.
    - ``"lindler2006"``: the TPF planner polynomial (Savransky et al.
      2010, eq. 16), renormalized to 1 at theta = 0:
      ``(2.44 - 0.0403 theta + 0.000269 theta**2) / 2.44``.
    - ``"stark2014"``: Stark et al. (2014), eq. B4:
      ``1.02 - 0.566 sin(theta) - 0.884 sin(theta)**2
      + 0.853 sin(theta)**3`` (note this is 1.02, not 1, at theta = 0).

    Args:
        theta_deg: Latitudinal angle in degrees (for local zodi, the
            absolute ecliptic latitude; for exozodi, use
            :func:`theta_from_inclination`).
        model: ``"leinert"``, ``"lindler2006"``, or ``"stark2014"``.
        interp_lon_deg: Table 17 longitude of the ``"leinert"`` profile;
            static configuration, snapped to the nearest tabulated
            longitude.

    Returns:
        Dimensionless latitudinal factor.
    """
    xp = array_namespace(theta_deg)
    theta = xp.asarray(theta_deg)
    if model == "lindler2006":
        return (2.44 - 0.0403 * theta + 0.000269 * theta**2) / 2.44
    if model == "stark2014":
        s = xp.sin(theta * (np.pi / 180.0))
        return 1.02 - 0.566 * s - 0.884 * s**2 + 0.853 * s**3
    if model == "leinert":
        row = int(np.argmin(np.abs(tables.LON_GRID_DEG - float(interp_lon_deg))))
        profile = tables.TABLE17_FILLED[row]
        profile = profile / profile[0]
        return interp1d_clamped(tables.BETA_GRID_DEG, profile, theta)
    raise ValueError("model must be 'leinert', 'lindler2006', or 'stark2014'")


def blackbody_spectral_radiance(wavelength_um, temperature_k):
    """Planck spectral radiance ``B_lambda`` in W m-2 sr-1 um-1.

    Args:
        wavelength_um: Wavelength(s) in um.
        temperature_k: Blackbody temperature in K.

    Returns:
        Spectral radiance in W m-2 sr-1 um-1.
    """
    xp = array_namespace(wavelength_um, temperature_k)
    _check_wavelength_um(xp, wavelength_um)
    lam_m = xp.asarray(wavelength_um) * 1e-6
    t = xp.asarray(temperature_k)
    exponent = _H_J_S * _C_M_PER_S / (lam_m * _KB_J_PER_K * t)
    radiance_per_m = 2.0 * _H_J_S * _C_M_PER_S**2 / lam_m**5 / xp.expm1(exponent)
    return radiance_per_m * 1e-6


def grey_scatter_intensity(
    wavelength_um,
    star_flux,
    f_star,
    f_thermal,
    t_dust_k=T_DUST_K,
    scatter_cutoff_um=10.0,
):
    """Exozodi spectrum model: grey-scattered starlight plus thermal dust.

    ``I(lam) = f_star * F_star(lam) * [lam <= cutoff]
    + f_thermal * B_lambda(lam, T_dust)``, the model fit to the local
    zodi wavelength dependence in the EXOSIMS Fundamental Concepts
    documentation. The scattered term is truncated beyond the cutoff
    (10 um by default), where scattering no longer contributes.

    Args:
        wavelength_um: Wavelength grid in um.
        star_flux: Stellar spectral flux density sampled on
            ``wavelength_um``, in any per-wavelength units; ``f_star``
            absorbs the normalization.
        f_star: Scattering constant converting stellar flux to surface
            brightness (from :func:`fit_grey_scatter_constants`).
        f_thermal: Thermal constant converting ``B_lambda`` to surface
            brightness (same source).
        t_dust_k: Dust temperature in K.
        scatter_cutoff_um: Wavelength beyond which the scattered term is
            zero.

    Returns:
        Model exozodi specific intensity on ``wavelength_um``, in the
        units set by the calibration of the two constants.
    """
    xp = array_namespace(wavelength_um, star_flux)
    _check_wavelength_um(xp, wavelength_um)
    lam = xp.asarray(wavelength_um)
    scattered = xp.asarray(star_flux) * xp.where(lam <= scatter_cutoff_um, 1.0, 0.0)
    thermal = blackbody_spectral_radiance(lam, t_dust_k)
    return f_star * scattered + f_thermal * thermal


def fit_grey_scatter_constants(
    wavelength_um,
    star_flux,
    target_intensity,
    t_dust_k=T_DUST_K,
    scatter_cutoff_um=10.0,
):
    """Calibrate ``(f_star, f_thermal)`` against a target spectrum.

    The model is linear in the two constants, so the least-squares fit is
    the closed-form solution of the 2x2 normal equations; no iterative
    optimizer is involved. Calibrating with a solar spectrum against the
    Leinert Table 19 local zodi spectrum reproduces the EXOSIMS
    calibration procedure; the resulting constants are then valid for any
    star spectrum supplied in the same units and normalization.

    Args:
        wavelength_um: Wavelength grid in um.
        star_flux: Illuminating-star spectral flux density on the grid
            (the Sun, for the Table 19 calibration).
        target_intensity: Observed dust specific intensity on the grid
            (Table 19 values for the local zodi).
        t_dust_k: Dust temperature in K.
        scatter_cutoff_um: Scattered-term cutoff in um.

    Returns:
        Tuple ``(f_star, f_thermal)``.
    """
    xp = array_namespace(wavelength_um, star_flux, target_intensity)
    _check_wavelength_um(xp, wavelength_um)
    lam = xp.asarray(wavelength_um)
    s = xp.asarray(star_flux) * xp.where(lam <= scatter_cutoff_um, 1.0, 0.0)
    b = blackbody_spectral_radiance(lam, t_dust_k)
    y = xp.asarray(target_intensity)

    ss = xp.sum(s * s)
    sb = xp.sum(s * b)
    bb = xp.sum(b * b)
    sy = xp.sum(s * y)
    by = xp.sum(b * y)
    det = ss * bb - sb * sb
    f_star = (bb * sy - sb * by) / det
    f_thermal = (ss * by - sb * sy) / det
    return f_star, f_thermal


def band_average(wavelength_um, values, throughput):
    """Throughput-weighted bandpass average of a spectral quantity.

    ``<I> = integral(P * I dlam) / integral(P dlam)`` with trapezoidal
    integration; the denominator is the bandpass equivalent width, so
    bandpasses of different shapes are comparable (EXOSIMS Fundamental
    Concepts documentation, band-averaged intensity equation).

    Args:
        wavelength_um: Strictly increasing wavelength grid in um.
        values: Spectral quantity sampled on the grid.
        throughput: Bandpass throughput sampled on the grid.

    Returns:
        The band-averaged value.
    """
    xp = array_namespace(wavelength_um, values, throughput)
    _check_wavelength_um(xp, wavelength_um)
    lam = xp.asarray(wavelength_um)
    weighted = xp.asarray(values) * xp.asarray(throughput)
    numerator = _trapezoid(xp, weighted, lam)
    denominator = _trapezoid(xp, xp.asarray(throughput), lam)
    return numerator / denominator


def _trapezoid(xp, y, x):
    """Trapezoidal integral of y over x (shared-source, differentiable)."""
    dx = x[1:] - x[:-1]
    return xp.sum(0.5 * (y[1:] + y[:-1]) * dx)


def jez0(
    f0v_ph_s_m2,
    mv_star,
    l_star_lsun,
    flambda,
    bandwidth_nm,
    mag_1zodi=MAG_1ZODI_V_ARCSEC2,
    mv_sun=MV_SUN,
):
    """Reference exozodi intensity at the EEID for 1 zodi (EXOSIMS ``calc_JEZ0``).

    ``JEZ0 = F0V * 10**(-0.4 (MV - MV_sun)) * 10**(-0.4 x) * f_lambda
    * bandwidth / L`` in ph s-1 m-2 arcsec-2 when ``F0V`` is in
    ph s-1 m-2 nm-1 and the bandwidth in nm. This is Stark et al. (2014)
    Eq. C4 in the observing band, i.e. the value at the EEID
    (``sqrt(L)`` AU), not at 1 AU. It matches EXOSIMS ``calc_JEZ0``
    numerically. Scale to an epoch with :func:`scale_jez`.

    Args:
        f0v_ph_s_m2: V-band zero-magnitude flux density in
            ph s-1 m-2 nm-1 (the caller's photometric system).
        mv_star: Absolute V magnitude of the star.
        l_star_lsun: Bolometric luminosity in solar units.
        flambda: Band color scale factor (1 for V; see
            :func:`band_average` and :func:`grey_scatter_intensity` for
            constructing it, or a stellar-color ratio for the
            scattered-only flavor).
        bandwidth_nm: Bandpass equivalent width in nm.
        mag_1zodi: V surface brightness of one zodi in mag arcsec-2.
        mv_sun: Absolute V magnitude of the Sun.

    Returns:
        Reference exozodi intensity in ph s-1 m-2 arcsec-2.
    """
    xp = array_namespace(f0v_ph_s_m2, mv_star, l_star_lsun, flambda, bandwidth_nm)
    return (
        xp.asarray(f0v_ph_s_m2)
        * 10.0 ** (-0.4 * (xp.asarray(mv_star) - mv_sun))
        * 10.0 ** (-0.4 * mag_1zodi)
        * xp.asarray(flambda)
        * xp.asarray(bandwidth_nm)
        / xp.asarray(l_star_lsun)
    )


def scale_jez(jez0_value, nzodi, r_au, fbeta, l_star_lsun):
    """Scale the EEID reference intensity to an epoch's geometry.

    ``JEZ = JEZ0 * n_EZ * f(theta) * (EEID / r)**2`` with
    ``EEID**2 = L``, i.e. ``JEZ0 * n_EZ * f(theta) * L / r**2``. The
    ``1/r**2`` illumination falloff is measured from the EEID, where
    :func:`jez0` is evaluated, so the luminosity in ``JEZ0`` cancels and
    the result depends on the star only through its V-band luminosity and
    color.

    EXOSIMS ``SimulatedUniverse.scale_JEZ`` applies ``n_EZ / r**2`` to the
    same ``JEZ0`` without the factor ``L``, which counts the luminosity
    twice; its values equal this function's divided by ``L``.

    Args:
        jez0_value: Reference intensity at the EEID from :func:`jez0`.
        nzodi: Exozodi level in zodis.
        r_au: Planet-star (or evaluation) radius in AU.
        fbeta: Latitudinal factor from :func:`latitudinal_factor`.
        l_star_lsun: Bolometric luminosity in solar units (the same value
            passed to :func:`jez0`).

    Returns:
        Exozodi intensity at the epoch, same units as ``jez0_value``.
    """
    xp = array_namespace(jez0_value, nzodi, r_au, fbeta, l_star_lsun)
    return (
        xp.asarray(jez0_value)
        * xp.asarray(nzodi)
        * xp.asarray(fbeta)
        * xp.asarray(l_star_lsun)
        / xp.asarray(r_au) ** 2
    )
