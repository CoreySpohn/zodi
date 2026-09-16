# zodi

Zodiacal and exozodiacal light brightness conventions for exoplanet imaging,
from a single source that runs on numpy and JAX.

## What it owns

- Local zodiacal light surface brightness: the Leinert et al. (1998)
  Table 17 position dependence and Table 19 wavelength dependence, with the
  interpolation, anchoring, and near-Sun conventions of the production
  codes stated explicitly (`zodi.specific_intensity`, `zodi.zodi_flux_ratio`).
- The exozodi chain of Stark et al. (2014), anchored at the EEID and
  numerically compatible with EXOSIMS `calc_JEZ0`, with the radial scaling
  measured from the EEID (`zodi.exozodi_flux_ratio_v`, `zodi.jez0`,
  `zodi.scale_jez`), the three published latitudinal models
  (`zodi.latitudinal_factor`), and the
  grey-scatterer band correction in both production flavors: the
  stellar-color scaling used by pyEDITH (`zodi.exozodi_flux_ratio_band`)
  and the scattered-plus-thermal spectrum model used by EXOSIMS, with its
  calibration as a closed-form least-squares fit
  (`zodi.fit_grey_scatter_constants`).
- Unit conversions between the dialects in common use: magnitudes per
  square arcsecond, flux ratio per square arcsecond, spectral radiance,
  photon rates, and MJy per steradian (`zodi.units`).

## Documentation and evidence

Documentation is at [zodi.readthedocs.io](https://zodi.readthedocs.io).
`docs/conventions.md` states every model, constant, and known cross-code
difference with sources and measured deltas; `docs/evidence.md` maps
each claim to an executable check and its kind of evidence. Cross-code
benchmark scripts against
EXOSIMS, skyscapes, and zodipy live in `scripts/` and are runnable by
anyone with those packages installed. Current results: exact agreement
with the EXOSIMS closed-form models and magnitude chain, float64
round-off agreement with skyscapes, and 0.95-1.23 brightness ratios
against the independent Kelsall model at 1.25 um.

## Design

One implementation serves both backends through the array API standard.
Functions compute in the namespace of their array inputs (via
`array-api-compat`), so numpy callers get numpy in and numpy out with no JAX
anywhere in their dependency tree, while JAX callers get functions that jit,
vmap, and differentiate natively. Hard dependencies are `numpy` and
`array-api-compat` only; the `[jax]` extra exists so the test suite can run
the JAX side. JAX users should enable float64:

```python
jax.config.update("jax_enable_x64", True)
```

Random sampling is deliberately absent from the library: distributions ship
as quantile functions (inverse CDFs), and callers bring uniforms from their
own generator, whether that is `numpy.random` or `jax.random`.

## Not this library

Structured circumstellar disks (rings, gaps, offsets) and scene rendering
belong to scene simulators; solar-system ephemerides belong to the caller.
For thermal-infrared zodiacal emission modeling, see
[zodipy](https://github.com/Cosmoglobe/zodipy); this library covers the
reflected-light brightness conventions used in exoplanet direct imaging.

## Install

```
pip install zodi
```

The hard dependencies are `numpy` and `array-api-compat`. To run the JAX
side of the test suite, install the extras:

```
pip install "zodi[jax,test]"
```
