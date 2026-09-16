# Conventions

This page is the contract of the library: every model, constant, unit
convention, and known cross-code difference, with sources. It is written
for maintainers of existing zodi implementations (EXOSIMS, pyEDITH,
scene simulators) who need to know exactly what a function computes
before trusting it.

## 1. Scope and design rules

The library owns *conventions*, not scenes or catalogs:

- the Leinert et al. (1998) local zodiacal light tables and their
  standard interpolation,
- the Stark et al. (2014) exozodi chain, including the grey-scatterer
  band correction,
- the unit conversions between the dialects in production use.

Geometry (look vectors, helio-ecliptic angles), star properties
(magnitudes, luminosities, spectra), and bandpasses are always caller
inputs. The package never generates random numbers: population
distributions ship as quantile functions and callers supply uniforms.

Every function runs on numpy or JAX arrays from one source
(`zodi._xp`): numpy inputs run pure numpy, JAX inputs trace natively
(jit/vmap/grad), and numpy arrays act as backend-neutral static data
when mixed with JAX inputs. Cross-backend agreement is exact for
arithmetic and gather operations and within a few ULPs (relative
1e-13) where transcendental functions or fused multiply-adds are
involved. JAX users should enable float64.

## 2. Local zodiacal light

### 2.1 Data

- **Table 17** (`zodi.tables.TABLE17_RAW`): brightness versus
  helio-ecliptic longitude difference $\Delta\lambda_\odot$ (19 rows,
  0-180 deg) and ecliptic latitude $\beta$ (11 columns, 0-90 deg), at
  500 nm, in units of $10^{-8}\,\mathrm{W\,m^{-2}\,sr^{-1}\,\mu m^{-1}}$.
  Cells inside the near-Sun exclusion zone are `nan` in the raw table.
- **Table 19** (`zodi.tables.TABLE19_*`): spectral radiance at the
  reference geometry (90 deg, 0 deg) at 16 wavelengths from 0.2 to
  140 um, in $\mathrm{W\,m^{-2}\,sr^{-1}\,\mu m^{-1}}$.

### 2.2 The model

$$
I(\Delta\lambda_\odot, \beta, \lambda)
  = I_{17}(\Delta\lambda_\odot, \beta)\,
    \frac{I_{19}(\lambda)}{I_{19}(500\,\mathrm{nm})}
$$

implemented by `zodi.specific_intensity`. Both angle arguments are
symmetric (absolute values are taken) and clamp to the tabulated
domain.

### 2.3 Conventions with more than one implementation in the wild

**Anchoring.** Table 19 at 0.5 um reads $2.6\times10^{-6}$ while the
Table 17 reference cell reads $259\times10^{-8}$; the tables disagree
by 0.39 percent at their shared anchor point. `anchor="table17"` (the
default, matching EXOSIMS) treats Table 17 as absolute and applies
Table 19 as a ratio; `anchor="table19"` treats Table 19 as absolute
and applies Table 17 as a normalized position factor (matching
implementations that build the absolute scale from the wavelength
table, e.g. skyscapes). The two differ by the constant factor 1.0039.

**Wavelength interpolation.** This library interpolates Table 19
*linearly* in log10-log10 space. EXOSIMS uses a *quadratic spline* in
the same space. The two agree exactly at every tabulated wavelength
and differ between knots; measured against the EXOSIMS prototype over
0.4-2.5 um: -4.6 percent at 550 nm, +2.8 percent at 850 nm, -3 to -6
percent through 1.15-2.05 um, +10.5 percent at 2.5 um (past the last
optical knot at 2.2 um, where the spline's curvature is largest).
Neither choice is more faithful to the published data, which are
discrete; consumers who consider the quadratic spline canonical
should say so and a matching option will be added.

**Position interpolation.** This library is bilinear on the
(longitude, latitude) grid. EXOSIMS triangulates the grid points
(`scipy.interpolate.LinearNDInterpolator`); both are exact at every
tabulated point and differ by up to 1.35 percent between points
(measured over a 64-point off-node grid outside the exclusion zone).

**Near-Sun exclusion cells.** This library fills them with the
nearest tabulated value along the same longitude row, so queries clamp
(`zodi.tables.TABLE17_FILLED`). The EXOSIMS data file fills them with
$10^{6}$, making near-Sun pointings look catastrophically bright (a
scheduler-penalty convention). Both behaviors are legitimate; the raw
table with `nan` markers is exposed for consumers who want their own
policy.

**Power versus photon units.** Intensity *ratios* change value between
power and photon units by exactly $\lambda/\lambda_\mathrm{ref}$. All
ratios here default to power units (the native units of the Leinert
data), with `photon_units=True` applying the conversion, matching the
EXOSIMS convention of converting to photon units as the final step.

### 2.4 The flux-ratio dialect

`zodi.zodi_flux_ratio` returns the surface brightness as the
dimensionless per-arcsecond-squared ratio used by the ETC and yield
codes,

$$
f_Z = \frac{I_\mathrm{photon} / \mathrm{arcsec^2}}{F_0/\Delta\lambda},
$$

with the zero-magnitude flux density $F_0$ supplied by the caller in
their own photometric system (EXOSIMS passes
`mode["F0"]/mode["deltaLam"]`). AB-referenced consumers can instead
use `zodi.surface_brightness_ab_mag` (exact AB zero point,
$10^{23 - 48.6/2.5}$ Jy) and `zodi.units.mag_to_flux_ratio`.

## 3. Exozodiacal light

### 3.1 The magnitude chain

Following Stark et al. (2014, Appendix C): one zodi is the solar
optical depth at 1 AU placed at the target star's Earth-equivalent
instellation distance (EEID), $\sqrt{L_*}$ AU, where the dust receives
the same bolometric insolation as the Solar System's dust at 1 AU. For
a solar twin its V-band surface brightness there is $x = 22$ mag
arcsec$^{-2}$. Equation C4 of Stark et al. (2014) gives the surface
brightness at the EEID,

$$
I_\mathrm{EZ}^V(\mathrm{EEID}) = n_\mathrm{EZ}\, F_0^V\,
  10^{-0.4 (M_V - M_{V,\odot})}\, 10^{-0.4 x}\, \frac{1}{L_*}.
$$

The $1/L_*$ is the $1/r^2$ illumination evaluated at $r^2 = L_*$. For a
radially flat optical depth, the surface brightness at any other radius
follows the $1/r^2$ falloff measured from the EEID,
$I_\mathrm{EZ}^V(r) = I_\mathrm{EZ}^V(\mathrm{EEID})\,(\mathrm{EEID}/r)^2$,
and the luminosity cancels:

$$
I_\mathrm{EZ}(r) = n_\mathrm{EZ}\, F_0^V\,
  10^{-0.4 (M_V - M_{V,\odot})}\, 10^{-0.4 x}\,
  \frac{f_\lambda\, f(\theta)}{r^2}
$$

with $r$ in AU, $L_*$ in solar luminosities, and $M_{V,\odot} = 4.83$.
The stellar V-band luminosity enters through the magnitude term; the
bolometric luminosity only sets where the EEID lies. For a radial
profile $\tau \propto r^{-\gamma}$ the exponent becomes $2 + \gamma$
($\gamma \approx 0.34$ for the zodiacal cloud); this library adopts the
flat profile.

`zodi.exozodi_flux_ratio_v` is the general-radius chain without
$f_\lambda f(\theta)$ and without $F_0$. `zodi.jez0` is the EEID value
in the observing band and matches EXOSIMS `calc_JEZ0` (verified exact
against the prototype). `zodi.scale_jez` carries it to radius $r$:

$$
J_\mathrm{EZ}(r, n_\mathrm{EZ}, \theta)
  = J_\mathrm{EZ,0}\; n_\mathrm{EZ}\, f(\theta)
    \left(\frac{\mathrm{EEID}}{r}\right)^2
  = J_\mathrm{EZ,0}\; \frac{n_\mathrm{EZ}\, f(\theta)\, L_*}{r^2}.
$$

### 3.2 The grey-scatterer band correction

The single physical assumption behind moving the V-band definition to
other bands: the dust scatters greyly, so the scattered part of the
exozodi spectrum is proportional to the spectrum of the *illuminating
star*. Two flavors of the same assumption are in production use, and
both are implemented:

**Stellar-color flavor** (pyEDITH): scale by the star's apparent
color,

$$
f_\lambda = 10^{-0.4 (m_\lambda - m_V)},
$$

i.e. the exozodi spectrum *is* the stellar spectrum, renormalized.
`zodi.exozodi_flux_ratio_band` implements the full pyEDITH expression.

**Scattered-plus-thermal flavor** (EXOSIMS): model the spectrum as

$$
I_\mathrm{EZ}(\lambda) \propto
  f_\mathrm{star} F_*(\lambda)\,[\lambda \le 10\,\mu\mathrm{m}]
  + f_\mathrm{thermal} B_\lambda(\lambda, 261.5\,\mathrm{K}),
$$

where $B_\lambda$ is the Planck function at the Leinert local-zodi
dust temperature, the scattered term is truncated beyond 10 um, and
the two constants are calibrated by fitting the *local* zodi
wavelength dependence (Table 19) with the Sun as $F_*$. Then
$f_\lambda$ is the ratio of throughput-weighted bandpass averages
between the observing band and V
(`zodi.grey_scatter_intensity`, `zodi.band_average`).

Because the model is linear in the two constants, the calibration is a
closed-form least-squares solve
(`zodi.fit_grey_scatter_constants`), so the constants adapt to
whatever units and normalization the caller's star spectrum carries
rather than being copied numbers tied to one code's spectrum handling.
Calibrated with a 5772 K blackbody as a solar proxy, the fit tracks
Table 19 to a few percent through the optical and reproduces the known
worst region near 3.5 um (the crossover between scattered and thermal
components) seen in the EXOSIMS documentation figure.

### 3.3 The latitudinal factor

`zodi.latitudinal_factor` implements the three published models
compared in the EXOSIMS documentation, with their differing
normalizations preserved:

- `"leinert"` (default in EXOSIMS): the Table 17 latitude profile at a
  fixed longitude (135 deg by default), normalized to its in-plane
  value. EXOSIMS interpolates this profile cubically; this library is
  piecewise linear (exact at tabulated latitudes, up to 2.3 percent
  between them, measured).
- `"lindler2006"` (Savransky et al. 2010, eq. 16), renormalized to 1
  at $\theta = 0$: $(2.44 - 0.0403\theta + 0.000269\theta^2)/2.44$.
  Verified exact against EXOSIMS.
- `"stark2014"` (Stark et al. 2014, eq. B4):
  $1.02 - 0.566\sin\theta - 0.884\sin^2\theta + 0.853\sin^3\theta$,
  which is 1.02, not 1, at $\theta = 0$. Verified exact against
  EXOSIMS.

For exozodi the model angle comes from the orbital inclination via
`zodi.theta_from_inclination` ($\theta = 90^\circ - I$ after folding
$I$ into $[0^\circ, 90^\circ]$), matching `EXOSIMS.calc_fbeta`
(verified on a 0-180 deg grid).

## 4. Dialect table

| Consumer | Quantity | Units / convention | zodi equivalent |
|---|---|---|---|
| EXOSIMS | `fZ` | flux ratio arcsec$^{-2}$, mode $F_0$ | `zodi_flux_ratio` |
| EXOSIMS | `JEZ0` | ph s$^{-1}$ m$^{-2}$ arcsec$^{-2}$ | `jez0` + `scale_jez` |
| pyEDITH | exozodi flux | flux ratio arcsec$^{-2}$ via stellar color | `exozodi_flux_ratio_band` |
| jaxedith | `Fzodi`, `Fexozodi` | flux ratio arcsec$^{-2}$, AB-referenced | `mag_to_flux_ratio` of `surface_brightness_ab_mag` |
| skyscapes | spectral radiance / mag | W m$^{-2}$ sr$^{-1}$ um$^{-1}$; AB mag arcsec$^{-2}$ | `specific_intensity(anchor="table19")` |
| zodipy | emission | MJy sr$^{-1}$ | `units.intensity_to_mjy_per_sr` |

## 5. Open cross-code questions for maintainers

Documented divergences found while pinning the conventions. Where a
divergence is an error rather than a convention, the library
implements the corrected form and says so.

1. **Luminosity normalization of the exozodi (resolved).** The EXOSIMS
   Fundamental Concepts documentation starts from Eq. C4, rescales the
   solar term by $(1\ \mathrm{AU}/r)^2$, and keeps C4's $1/L_*$, giving
   $1/(L_* r^2)$. That rescales from 1 AU a value that lives at the
   EEID, so the luminosity is counted twice: at $r = \sqrt{L_*}$ the
   documented form gives $1/L_*^2$ instead of C4's $1/L_*$. In code,
   `calc_JEZ0` (which divides by $L$) is correct as the EEID value, and
   `SimulatedUniverse.scale_JEZ` omits the factor $L_*$; EXOSIMS values
   from `scale_JEZ` equal `zodi.scale_jez` divided by $L_*$. The
   pyEDITH expression carries no $1/L_*$ and no $r$, which equals the
   correct surface brightness at $r = 1$ AU; applied at the EEID it
   lacks C4's $1/L_*$. Whether pyEDITH intends the EEID is for its
   maintainers to confirm.
2. **Table 19 interpolation order** (section 2.3): quadratic spline
   (EXOSIMS) versus log-log linear (this library, skyscapes) is a 4.6
   percent difference at 550 nm.
3. **Near-Sun fill policy** (section 2.3): clamp versus $10^6$
   penalty.
4. **Anchoring** (section 2.3): Table 17 versus Table 19 absolute
   scale, a constant 0.39 percent.

## References

- Leinert, C., et al. 1998, A&AS 127, 1 (Tables 17 and 19)
- Stark, C. C., et al. 2014, ApJ 795, 122 (Appendices B and C)
- Savransky, D., Kasdin, N. J., Cady, E. 2010, PASP 122, 401 (eq. 16)
- Kelsall, T., et al. 1998, ApJ 508, 44 (via zodipy, for validation)
- EXOSIMS documentation, "Fundamental Concepts: Exozodiacal Light",
  and `EXOSIMS.Prototypes.ZodiacalLight` / `TargetList`
- pyEDITH `astrophysical_scene.calc_exozodi_flux`
