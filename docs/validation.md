# Validation

Every claim in the conventions page is backed by an executable check.
The unit suite runs in CI on every commit (with and without JAX
installed); the cross-code scripts in `scripts/` run against reference
implementations and are re-run before each release. Numbers below are
from 2026-07 on linux x86-64, float64.

## Gate matrix

| Gate | What it checks | How | Result |
|---|---|---|---|
| Backend parity | numpy = JAX through jit/vmap, grad vs finite differences, for every public function | unit suite (`tests/`) | exact for gather/arithmetic paths; within a few ULPs (rtol 1e-13) where transcendentals / fma fusion apply |
| Table data | tables match the published values | pinned spot values in `tests/test_tables.py` | pass |
| skyscapes reproduction | `anchor="table19"` reproduces `skyscapes.background.leinert` radiance and AB magnitudes | `scripts/validate_against_skyscapes.py`, 504-point grid | max rel radiance diff 2.5e-15; max mag diff 1.1e-14 |
| EXOSIMS conventions | closed-form models and magnitude chain match the EXOSIMS prototype | `scripts/validate_against_exosims.py` | Lindler2006, Stark2014, `calc_JEZ0` chain: exact (0.0). Interpolation-flavor deltas: Table 17 position max 1.35 percent (bilinear vs triangulated); Table 19 color exact at knots, -4.6 percent at 550 nm, max +10.5 percent at 2.5 um (linear vs quadratic spline); latitudinal profile max 2.3 percent (linear vs cubic) |
| Grey-scatter calibration | closed-form fit recovers synthetic constants; blackbody-Sun proxy tracks Table 19 | unit suite | constants recovered to 1e-10; proxy fit within 6 percent over 0.4-0.9 um, worst 0.50 at the 3.5 um crossover (matches the EXOSIMS documentation figure) |
| zodipy cross-model | brightness against the independent Kelsall (1998) implementation at matched Earth-observer geometry | `scripts/validate_against_zodipy.py` (two processes; zodipy pins numpy below 2) | ratios 0.95-1.23 at 1.25 um over elongations 60-180 deg and latitudes 0-90 deg (gate band 0.7-1.4); 1.04-1.67 at 2.2 um, informational |
| Units arbiter | conversions against astropy references and hypothesis round-trips | unit suite | MJy conversion matches astropy 7.2.2 to 1e-12; AB/mag round trips to 1e-10 |

## Running the cross-code scripts

```
python scripts/validate_against_skyscapes.py    # needs skyscapes + jax
python scripts/validate_against_exosims.py      # needs EXOSIMS
python scripts/validate_against_zodipy.py zodi > rows.json
uv run --no-project --with zodipy \
    python scripts/validate_against_zodipy.py zodipy rows.json
```

The zodipy comparison is deliberately model-versus-model: Kelsall and
Leinert are independent fits to different data, so the expected
agreement is a tolerance band, not equality. Its role is to catch
unit, orientation, and geometry-convention errors, which show up as
orders of magnitude rather than tens of percent.
