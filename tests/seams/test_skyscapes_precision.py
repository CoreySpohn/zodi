"""Precision contract at the Skyscapes local-zodi seam."""

import numpy as np
import pytest
import skyscapes
from hwoutils.leaves import canonical
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff.spectral import Band, Quadrature
from orbix.system import SkyFrame


@pytest.mark.float64_only(reason="trace float32 consumers in the x64 lane")
@pytest.mark.parametrize("read", ["point", "node"])
@pytest.mark.parametrize("model_name", ["AYOZodi", "LeinertZodi"])
def test_skyscapes_local_zodi_forward_precision(model_name, read):
    """Fails if the consumer's point or quadrature-node read widens.

    In this band and its adjoining Table 19 cells, the absolute log
    radiance is below six, the absolute log wavelength is below 0.4,
    and the log slope is below two. Per spectrum allow six ordinate
    roundings and five wavelength roundings, amplified by ln(10).
    Allow 80 more roundings for the conversions and sightline geometry.
    """
    model = getattr(skyscapes, model_name)()
    frame = SkyFrame(10.0, ra_deg=100.0, dec_deg=-20.0)
    if read == "point":
        samples = np.array([500.0, 550.0, 600.0, 700.0])

        def radiance(m, wave, frame, time):
            return m.photon_radiance(wave, frame, time)
    else:
        samples = Quadrature.build([Band(470.0, 740.0)], model, order=2)

        def radiance(m, quad, frame, time):
            return m.node_radiance(quad, None, frame, time)

    wide = canonical((model, samples, frame, 40.0), precision="float64")
    spectrum_ops = int(np.ceil(np.log(10.0) * (6 * 6 + 5 * 0.4 * 2)))
    bound = float(ulps(2 * spectrum_ops + 80, np.float32(1)))
    precision_pair(radiance, wide, bound=bound, scale=0)
    assert_precision_closed(radiance, *canonical(wide, precision="float32"))
