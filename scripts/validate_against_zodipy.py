"""Cross-code benchmark of zodi against zodipy (Kelsall 1998 DIRBE model).

zodipy implements an INDEPENDENT interplanetary dust model (Kelsall et
al. 1998, fit to COBE/DIRBE) with an independent codebase, which makes it
a strong external check against unit, orientation, and geometry-convention
errors: the two models describe the same dust cloud and agree to tens of
percent where both are well calibrated, so any large discrepancy indicates
a bug rather than model spread. The comparison runs in the DIRBE
scattered-light bands (1.25 and 2.2 um); the Kelsall model does not extend
to V band, and the backscatter/thermal regions (anti-solar point, 2.2 um)
are where the two models genuinely diverge.

zodipy pins numpy below 2, so this benchmark runs as two processes with a
JSON handoff:

    python validate_against_zodipy.py zodi > /tmp/zodi_rows.json
    uv run --no-project --with zodipy python validate_against_zodipy.py \
        zodipy /tmp/zodi_rows.json

Expected band (measured 2026-07): brightness ratios zodi/Kelsall of
0.95-1.25 at 1.25 um over elongations 60-180 deg and latitudes 0-90 deg,
degrading to ~1.7 at the 2.2 um anti-solar point.
"""

import json
import sys

WAVELENGTHS_UM = [1.25, 2.2]
GEOMETRIES = [  # (helio-ecliptic dlon [deg], ecliptic lat [deg])
    (60.0, 0.0),
    (90.0, 0.0),
    (135.0, 0.0),
    (180.0, 0.0),  # gegenschein
    (90.0, 45.0),
    (90.0, 90.0),  # pole; dlon degenerate there
]
EPOCH_ISO = "2026-03-20T00:00:00"


def emit_zodi():
    """zodi-env side: model brightness on the grid, as JSON on stdout."""
    from zodi import units, zodi

    rows = []
    for wav_um in WAVELENGTHS_UM:
        for dlon, beta in GEOMETRIES:
            intensity = zodi.specific_intensity(dlon, beta, wav_um * 1000.0)
            mjy_sr = units.intensity_to_mjy_per_sr(intensity, wav_um)
            rows.append(
                {
                    "wav_um": wav_um,
                    "dlon": dlon,
                    "beta": beta,
                    "zodi_mjy_sr": float(mjy_sr),
                }
            )
    json.dump(rows, sys.stdout, indent=1)


def eval_zodipy(rows_path):
    """zodipy-env side: Kelsall at matched geometry; print the table."""
    from importlib.metadata import version as pkg_version

    import astropy.units as u
    import zodipy
    from astropy.coordinates import GeocentricMeanEcliptic, SkyCoord, get_sun
    from astropy.time import Time

    with open(rows_path) as f:
        rows = json.load(f)

    epoch = Time(EPOCH_ISO, scale="utc")
    sun_lon = get_sun(epoch).transform_to(GeocentricMeanEcliptic(obstime=epoch)).lon
    print("zodipy", pkg_version("zodipy"), "| epoch", epoch.iso)
    current_wav = None
    for row in rows:
        if row["wav_um"] != current_wav:
            current_wav = row["wav_um"]
            model = zodipy.Model(current_wav * u.um)
            print(f"\n{current_wav} um   dlon  beta |    zodi   |  kelsall  | ratio")
        coord = SkyCoord(
            sun_lon + row["dlon"] * u.deg,
            row["beta"] * u.deg,
            frame=GeocentricMeanEcliptic,
            obstime=epoch,
        )
        kelsall = model.evaluate(coord, obspos="earth").to_value(u.MJy / u.sr)[0]
        ours = row["zodi_mjy_sr"]
        print(
            f"       {row['dlon']:5.0f} {row['beta']:5.0f} | {ours:9.4f}"
            f" | {kelsall:9.4f} | {ours / kelsall:6.2f}"
        )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "zodi":
        emit_zodi()
    elif len(sys.argv) >= 3 and sys.argv[1] == "zodipy":
        eval_zodipy(sys.argv[2])
    else:
        sys.exit("usage: validate_against_zodipy.py zodi | zodipy <rows.json>")
