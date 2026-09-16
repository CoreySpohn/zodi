"""Cross-code benchmark of zodi against the EXOSIMS ZodiacalLight prototype.

Runs EXOSIMS as the reference implementation and reports, for each shared
quantity, the maximum relative difference over a test grid. Differences
have two known, documented sources: EXOSIMS interpolates the Table 19
wavelength dependence with a quadratic spline in log-log space and the
latitudinal profile with a cubic spline, while zodi uses piecewise-linear
interpolation for both (exact agreement at every tabulated point, small
differences between them); and EXOSIMS fills the near-Sun exclusion cells
of Table 17 with 1e6 (a scheduler penalty) where zodi clamps to the
nearest tabulated value, so the comparison grid stays outside the
exclusion zone.

Requires EXOSIMS (pip install EXOSIMS). Run:

    python validate_against_exosims.py
"""

import astropy.units as u
import numpy as np

import zodi
from zodi import exozodi as zexo
from zodi import zodi as zzodi


def main():
    """Run the EXOSIMS comparison and print the per-quantity report."""
    from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight

    zl = ZodiacalLight()
    report = []

    lam_nm = np.linspace(400.0, 2500.0, 43)
    ours = zzodi.color_correction(lam_nm)
    theirs = zl.zodi_color_correction_factor(lam_nm * u.nm)
    report.append(("color correction, power units, 0.4-2.5 um", ours, theirs))

    ours = zzodi.color_correction(lam_nm, photon_units=True)
    theirs = zl.zodi_color_correction_factor(lam_nm * u.nm, photon_units=True)
    report.append(("color correction, photon units, 0.4-2.5 um", ours, theirs))

    lons = np.array([45.0, 60.0, 77.0, 90.0, 111.0, 135.0, 158.0, 180.0])
    lats = np.array([0.0, 7.0, 12.0, 30.0, 45.0, 52.0, 75.0, 90.0])
    lon_g, lat_g = np.meshgrid(lons, lats)
    ours = zzodi.specific_intensity_at_location(lon_g.ravel(), lat_g.ravel())
    theirs = zl.zodi_intensity_at_location(
        lon_g.ravel() * u.deg, lat_g.ravel() * u.deg
    ).to_value(u.W / u.m**2 / u.sr / u.um)
    report.append(("Table 17 intensity, outside exclusion zone", ours, theirs))

    theta = np.linspace(0.0, 90.0, 31)
    for model in ["Lindler2006", "Stark2014", "interp"]:
        ours = zexo.latitudinal_factor(
            theta, model={"interp": "leinert"}.get(model.lower(), model.lower())
        )
        theirs = zl.zodi_latitudinal_correction_factor(theta * u.deg, model=model)
        report.append((f"latitudinal factor, {model}", ours, np.asarray(theirs)))

    inclinations = np.linspace(0.0, 180.0, 37)
    ours = zexo.latitudinal_factor(
        zexo.theta_from_inclination(inclinations), model="leinert"
    )
    theirs = zl.calc_fbeta(inclinations.copy() * u.deg)
    report.append(("fbeta from inclination (interp model)", ours, np.asarray(theirs)))

    mv = np.array([3.5, 4.83, 6.2])
    lum = np.array([2.5, 1.0, 0.4])
    flambda = np.array([0.8, 1.0, 1.3])
    bandwidth = 110.0
    f0v = 1.0e10
    ours = zexo.jez0(f0v, mv, lum, flambda, bandwidth)
    zl.F0V = f0v * u.ph / u.s / u.m**2 / u.nm
    theirs = zl.calc_JEZ0(mv, lum, flambda, bandwidth * u.nm).to_value(
        u.ph / u.s / u.m**2 / u.arcsec**2
    )
    report.append(("calc_JEZ0 magnitude chain", ours, theirs))

    print("quantity                                            max rel diff")
    for name, a, b in report:
        rel = float(np.max(np.abs(np.asarray(a) - b) / np.abs(b)))
        print(f"{name:52s} {rel:.3e}")

    print(
        "\nzodi",
        zodi.__name__,
        "vs EXOSIMS prototype; linear-vs-spline"
        " interpolation differences are documented in docs/conventions.md",
    )


if __name__ == "__main__":
    main()
