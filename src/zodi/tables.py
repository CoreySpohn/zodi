"""Published zodiacal light tables from Leinert et al. (1998).

Data provenance
    Leinert et al. 1998, A&AS 127, 1, "The 1997 reference of diffuse night
    sky brightness":

    - Table 17: zodiacal light brightness as a function of helio-ecliptic
      longitude (rows) and ecliptic latitude (columns), at 500 nm, in units
      of 1e-8 W m-2 sr-1 um-1. Cells inside the near-Sun exclusion zone are
      not tabulated (stored here as ``nan``).
    - Table 19: zodiacal light spectral radiance at the reference geometry
      (helio-ecliptic longitude 90 deg, ecliptic latitude 0 deg) as a
      function of wavelength, in W m-2 sr-1 um-1.

Tables are plain numpy arrays preprocessed at import time; runtime code
lifts them into the caller's array namespace (see ``zodi._interp``).

Established consumer conventions this module makes explicit rather than
hiding: the near-Sun cells are filled with the nearest tabulated value
along each longitude row (so queries clamp instead of returning ``nan``);
codes that instead want near-Sun queries to look catastrophically bright
(a scheduler-penalty convention) can build their own fill from
``TABLE17_RAW``. The reference cell (90 deg, 0 deg) is 259e-8 W m-2 sr-1
um-1, which differs from Table 19 at 0.5 um (2.6e-6) by about 0.4 percent;
see the conventions documentation for how the two anchorings are exposed.
"""

import numpy as np

__all__ = [
    "BETA_GRID_DEG",
    "LON_GRID_DEG",
    "TABLE17_FILLED",
    "TABLE17_RAW",
    "TABLE17_REFERENCE",
    "TABLE17_UNIT_W_M2_SR_UM",
    "TABLE19_RADIANCE_W_M2_SR_UM",
    "TABLE19_WAVELENGTH_UM",
]

LON_GRID_DEG = np.array(
    [0.0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
)
BETA_GRID_DEG = np.array([0.0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90])

# Scale of the TABLE17 values: multiply by this to get W m-2 sr-1 um-1.
TABLE17_UNIT_W_M2_SR_UM = 1e-8

# nan marks the near-Sun exclusion zone (not tabulated in the paper).
_X = np.nan
TABLE17_RAW = np.array(
    [
        [_X, _X, _X, 3140, 1610, 985, 640, 275, 150, 100, 77],
        [_X, _X, _X, 2940, 1540, 945, 625, 271, 150, 100, 77],
        [_X, _X, 4740, 2470, 1370, 865, 590, 264, 148, 100, 77],
        [11500, 6780, 3440, 1860, 1110, 755, 525, 251, 146, 100, 77],
        [6400, 4480, 2410, 1410, 910, 635, 454, 237, 141, 99, 77],
        [3840, 2830, 1730, 1100, 749, 545, 410, 223, 136, 97, 77],
        [2480, 1870, 1220, 845, 615, 467, 365, 207, 131, 95, 77],
        [1650, 1270, 910, 680, 510, 397, 320, 193, 125, 93, 77],
        [1180, 940, 700, 530, 416, 338, 282, 179, 120, 92, 77],
        [910, 730, 555, 442, 356, 292, 250, 166, 116, 90, 77],
        [505, 442, 352, 292, 243, 209, 183, 134, 104, 86, 77],
        [338, 317, 269, 227, 196, 172, 151, 116, 93, 82, 77],
        [259, 251, 225, 193, 166, 147, 132, 104, 86, 79, 77],
        [212, 210, 197, 170, 150, 133, 119, 96, 82, 77, 77],
        [188, 186, 177, 154, 138, 125, 113, 90, 77, 74, 77],
        [179, 178, 166, 147, 134, 122, 110, 90, 77, 73, 77],
        [179, 178, 165, 148, 137, 127, 116, 96, 79, 72, 77],
        [196, 192, 179, 165, 151, 141, 131, 104, 82, 72, 77],
        [230, 212, 195, 178, 163, 148, 134, 105, 83, 72, 77],
    ]
)

TABLE19_WAVELENGTH_UM = np.array(
    [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5, 4.8, 12, 25, 60, 100, 140]
)
TABLE19_RADIANCE_W_M2_SR_UM = np.array(
    [
        2.5e-8,
        5.3e-7,
        2.2e-6,
        2.6e-6,
        2.0e-6,
        1.3e-6,
        1.2e-6,
        8.1e-7,
        1.7e-7,
        5.2e-8,
        1.2e-7,
        7.5e-7,
        3.2e-7,
        1.8e-8,
        3.2e-9,
        6.9e-10,
    ]
)


def _fill_nearest_along_rows(table):
    """Fill nan cells with the nearest valid value along each longitude row.

    Matches the clamp-to-nearest-tabulated-brightness convention: a query
    inside the near-Sun exclusion zone resolves to the closest tabulated
    latitude cell of the same longitude row rather than ``nan``.
    """
    filled = np.array(table, dtype=float)
    for row in filled:
        valid = np.where(np.isfinite(row))[0]
        for j in np.where(~np.isfinite(row))[0]:
            row[j] = row[valid[np.argmin(np.abs(valid - j))]]
    return filled


TABLE17_FILLED = _fill_nearest_along_rows(TABLE17_RAW)

# Brightness at the reference geometry (lon 90 deg, beta 0 deg), the cell
# Table 19 is anchored to.
TABLE17_REFERENCE = float(TABLE17_RAW[np.argmin(np.abs(LON_GRID_DEG - 90.0)), 0])
