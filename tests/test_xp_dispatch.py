"""The dispatch contract: numpy in -> numpy out, jax in -> jax out."""

import subprocess
import sys

import numpy as np
import pytest

from zodi._xp import array_namespace


def test_numpy_arrays_dispatch_to_numpy():
    ns = array_namespace(np.ones(3), np.zeros(2))
    assert "numpy" in ns.__name__


def test_scalars_fall_back_to_numpy():
    ns = array_namespace(1.0, 2)
    assert "numpy" in ns.__name__


def test_no_arguments_falls_back_to_numpy():
    ns = array_namespace()
    assert "numpy" in ns.__name__


def test_mixed_scalars_and_arrays_use_the_array():
    ns = array_namespace(1.0, np.ones(3))
    assert "numpy" in ns.__name__


def test_jax_arrays_dispatch_to_jax():
    jnp = pytest.importorskip("jax.numpy")
    ns = array_namespace(jnp.ones(3))
    assert "jax" in ns.__name__


def test_import_does_not_load_jax_or_equinox():
    """Fails if the default NumPy import loads optional backends; exact check."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import zodi; "
            "assert not any(name.split('.')[0] in {'jax', 'equinox'} "
            "for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_numpy_grey_scattering_keeps_calibration_precision():
    """Fails if the no-JAX scattered term rounds in the stellar flux dtype.

    With scattering active and the thermal scale zero, a float64 product
    of the stored flux and calibration is the reference. The mask is exact.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import numpy as np

sys.modules["jax"] = sys.modules["equinox"] = None
from zodi.exozodi import grey_scatter_intensity

wave = np.array([7.123456789, 8.123456789])
flux = np.array([1, 3.1415926], dtype=np.float32)
calibration = 0.123456789
actual = grey_scatter_intensity(wave, flux, calibration, 0)
assert actual.dtype == wave.dtype
np.testing.assert_array_equal(actual, flux.astype(wave.dtype) * calibration)
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
