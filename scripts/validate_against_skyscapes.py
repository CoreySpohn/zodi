"""Compare the NumPy core to the JAX local-zodi model used by skyscapes.

skyscapes scene containers accept zodi.jax.AbstractLocalZodi directly.
This checks the Table 19 anchor and the conversion to photon radiance
per nm across the geometry/wavelength grid. Requires JAX and Equinox.
"""

import numpy as np

import zodi
from zodi.units import ab_mag_arcsec2_from_intensity, power_to_photon_intensity


def main():
    """Print worst-case core/model differences over the pointing grid."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from zodi.jax import LeinertZodi

    lons = np.array([0.0, 15.0, 45.0, 60.0, 90.0, 112.0, 135.0, 166.0, 180.0])
    lats = np.array([0.0, 5.0, 17.0, 30.0, 45.0, 63.0, 75.0, 90.0])
    lams = np.array([300.0, 500.0, 550.0, 700.0, 1000.0, 1250.0, 2200.0])
    lon, lat, wave = np.meshgrid(lons, lats, lams, indexing="ij")
    power = zodi.specific_intensity(lon, lat, wave, anchor="table19")
    expected = power_to_photon_intensity(power, wave) / 1000.0
    model = LeinertZodi()
    actual = np.asarray(
        jax.jit(model.photon_radiance)(
            jnp.asarray(wave),
            ecliptic_latitude_deg=jnp.asarray(lat),
            helioecliptic_longitude_deg=jnp.asarray(lon),
        )
    )
    model_power = actual * 1000.0 / power_to_photon_intensity(np.ones_like(wave), wave)
    model_mag = ab_mag_arcsec2_from_intensity(model_power, wave)
    core_mag = zodi.surface_brightness_ab_mag(lon, lat, wave, anchor="table19")
    print(f"grid points: {wave.size}")
    print(
        f"max relative radiance difference: {np.max(np.abs(actual / expected - 1)):.3e}"
    )
    print(f"max AB magnitude difference: {np.max(np.abs(model_mag - core_mag)):.3e}")


if __name__ == "__main__":
    main()
