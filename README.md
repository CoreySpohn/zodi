# zodi

Zodiacal and exozodiacal light brightness conventions for exoplanet imaging,
from a single source that runs on numpy and JAX.

Status: pre-release scaffold; the public physics API lands with v0.1.

## What it will own

- Local zodiacal light surface brightness (Leinert et al. 1998 tables, plus
  the fixed-V-band variant used by yield codes).
- Exozodi conventions: the one-zodi definition, n-zodi scaling, and radial
  and latitudinal factors.
- The LBTI HOSTS survey n-zodi population distributions, exposed as quantile
  functions so callers supply their own uniform draws.
- Unit conversions between the dialects in common use: magnitudes per square
  arcsecond, flux ratio per square arcsecond, spectral radiance, and photon
  rates.

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

PyPI release pending; for now:

```
pip install git+https://github.com/CoreySpohn/zodi.git
```
