# zodi

Zodiacal and exozodiacal light brightness conventions for exoplanet
imaging, from a single source that runs on numpy and JAX.

The library owns the conventions shared by the exoplanet yield and
exposure-time codes: the Leinert et al. (1998) local zodi tables and
their standard interpolation, the Stark et al. (2014) exozodi chain
including the grey-scatterer band correction, and the conversions
between the surface-brightness dialects in production use. Geometry,
star properties, spectra, and bandpasses are always caller inputs.

Start with the [conventions](conventions.md) page: it states every
model, constant, and known cross-code difference, with sources and
measured deltas. The [validation](validation.md) page maps each claim
to an executable check.

```{toctree}
:maxdepth: 1

conventions
validation
```
