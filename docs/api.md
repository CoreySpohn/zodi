# API reference

The generated API includes `zodi.zodi` (local zodiacal light),
`zodi.exozodi` (scattered and thermal exozodiacal light), `zodi.units`
(surface-brightness conversions), and `zodi.amplitude` (typed amplitudes).
All amplitude names below are also exported directly from `zodi`.

| Record | Definition |
|---|---|
| `AbstractDustAmplitude` | Frozen base with an unchecked, array-capable `level` and static calibration metadata. |
| `SurfaceBrightnessDust` | Stark et al. (2014) surface-brightness amplitude; defaults to solar V, face-on, 22 mag per square arcsecond, and the band luminosity rule. |
| `LocalZodiMultiple` | Multiple of local zodiacal intensity at a fixed pointing. |
| `DensityDust` | Multiplier of the declared `profile_id`; the profile owner supplies any line-of-sight integral. |
| `Passband(name, center_nm, magnitude_system)` | Band metadata; magnitude system is `"vega"` or `"ab"`. |
| `ReferenceStar(band_abs_mag, luminosity_lsun)` | Reference absolute magnitude in the declared band and bolometric luminosity in solar units. |
| `Viewing(inclination_deg, position_angle_deg)` | Calibration viewing angles in degrees; inclination zero is face-on. |

Amplitude constructors take keyword arguments. Each `family` is a fixed class
constant. Shared metadata are `reference_star`, `reference_radius_au` (AU or
`"eeid"`), `passband`, `viewing`, and a nonempty `calibration_id`. Only `level`
is a PyTree child when JAX is installed; metadata compare by value for tracing.

`SurfaceBrightnessDust(level=3)` uses `calibration_id="stark-2014"`,
`reference_radius_au="eeid"`, and `luminosity_rule="band"`. It provides:

```python
flux_ratio(*, host_band_abs_mag, host_luminosity_lsun, r_au)
at_eeid(*, host_band_abs_mag, host_luminosity_lsun)
mag_per_arcsec2_at(*, host_band_abs_mag, host_luminosity_lsun, r_au)
```

Flux ratios are dimensionless per square arcsecond relative to the declared
band's zero-magnitude flux density. Magnitudes use that same system. Host
luminosities are bolometric in solar units. The band rule uses the ratio of
band luminosities; `luminosity_rule="bolometric"` assumes solar colour.
The inverse-square law is evaluated in AU; reference radius and viewing label
the calibration and do not add radius or inclination corrections. Custom bands
require a matching reference-star magnitude and surface-brightness calibration.

`LocalZodiMultiple(level=3, ecliptic_latitude_deg=30, helioecliptic_longitude_deg=90)`
uses `calibration_id="leinert-1998-table17"`. Its
`specific_intensity(wavelength_nm)` returns W m-2 sr-1 um-1 with the existing
Table 17 anchoring. `helioecliptic_longitude_deg` means target longitude minus solar
longitude, matching `dlon_deg`; latitude matches `beta_deg`. The existing
absolute-value symmetry and edge clamping apply.

`DensityDust(level=3, profile_id="example", calibration_id="example-v1")`
has no brightness method. These three definitions have no conversion between
them without a density profile and its line-of-sight integral.
