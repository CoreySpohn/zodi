"""Compare the NumPy core to the JAX local-zodi model used by skyscapes.

skyscapes scene containers hold ``skyscapes.AbstractLocalZodi`` models, which
give the Table 19 anchor and the conversion to photon radiance per nm from a
sky frame and a time. This checks them against the core across a grid of
sightlines and wavelengths. Requires skyscapes, orbix and JAX.
"""

import numpy as np

import zodi
from zodi.units import ab_mag_arcsec2_from_intensity, power_to_photon_intensity


def main():
    """Print worst-case core/model differences over the sightline grid."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from orbix.observatory import ecliptic_pointing
    from orbix.system import SkyFrame
    from skyscapes import LeinertZodi

    ras = np.array([0.0, 30.0, 75.0, 120.0, 180.0, 240.0, 300.0, 350.0])
    decs = np.array([-80.0, -45.0, -10.0, 0.0, 20.0, 50.0, 85.0])
    lams = np.array([300.0, 500.0, 550.0, 700.0, 1000.0, 1250.0, 2200.0])
    time_d = 100.0
    model = LeinertZodi()
    worst_radiance = worst_mag = 0.0
    for i, j in np.ndindex(len(ras), len(decs)):
        placed = SkyFrame(10.0, ra_deg=ras[i], dec_deg=decs[j])
        lat, lon = ecliptic_pointing(ras[i], decs[j], placed.epoch, time_d)
        lon, lat = np.asarray(lon), np.asarray(lat)
        power = zodi.specific_intensity(lon, lat, lams, anchor="table19")
        expected = power_to_photon_intensity(power, lams) / 1000.0
        actual = np.asarray(model.photon_radiance(jnp.asarray(lams), placed, time_d))
        model_power = (
            actual * 1000.0 / power_to_photon_intensity(np.ones_like(lams), lams)
        )
        model_mag = ab_mag_arcsec2_from_intensity(model_power, lams)
        core_mag = zodi.surface_brightness_ab_mag(lon, lat, lams, anchor="table19")
        worst_radiance = max(worst_radiance, np.max(np.abs(actual / expected - 1)))
        worst_mag = max(worst_mag, np.max(np.abs(model_mag - core_mag)))
    print(f"grid points: {ras.size * decs.size * lams.size}")
    print(f"max relative radiance difference: {worst_radiance:.3e}")
    print(f"max AB magnitude difference: {worst_mag:.3e}")


if __name__ == "__main__":
    main()
