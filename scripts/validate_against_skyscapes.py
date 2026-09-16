"""Cross-code benchmark of zodi against skyscapes.background.leinert.

skyscapes carries a JAX implementation of the same Leinert tables with the
same piecewise-linear interpolation conventions, anchored on Table 19 with
a normalized Table 17 position factor. With ``anchor="table19"`` zodi
reproduces it to float64 round-off over the full geometry/wavelength grid.

Requires skyscapes and jax. Run:

    python validate_against_skyscapes.py
"""

import numpy as np

from zodi import zodi as zzodi


def main():
    """Run the skyscapes grid comparison and print worst-case differences."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from skyscapes.background.leinert import (
        leinert_zodi_mag,
        leinert_zodi_spectral_radiance,
    )

    lons = np.array([0.0, 15.0, 45.0, 60.0, 90.0, 112.0, 135.0, 166.0, 180.0])
    lats = np.array([0.0, 5.0, 17.0, 30.0, 45.0, 63.0, 75.0, 90.0])
    lams = np.array([300.0, 500.0, 550.0, 700.0, 1000.0, 1250.0, 2200.0])

    worst_rad = 0.0
    worst_mag = 0.0
    for lam in lams:
        for lon in lons:
            for lat in lats:
                theirs = float(leinert_zodi_spectral_radiance(lam, lat, lon))
                ours = float(zzodi.specific_intensity(lon, lat, lam, anchor="table19"))
                worst_rad = max(worst_rad, abs(ours - theirs) / theirs)
                theirs_mag = float(leinert_zodi_mag(lam, lat, lon))
                ours_mag = float(
                    zzodi.surface_brightness_ab_mag(lon, lat, lam, anchor="table19")
                )
                worst_mag = max(worst_mag, abs(ours_mag - theirs_mag))

    n = len(lams) * len(lons) * len(lats)
    print(f"grid points: {n}")
    print(f"max relative radiance difference: {worst_rad:.3e}")
    print(f"max AB magnitude difference:      {worst_mag:.3e}")


if __name__ == "__main__":
    main()
