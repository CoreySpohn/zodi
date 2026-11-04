"""Shared fixtures: parametrize tests over the installed array backends."""

import importlib.util
import os

import numpy as np
import pytest
from hypothesis import settings

# hwoutils is optional so NumPy-only test installs do not acquire JAX.
pytest_plugins = []
if (
    importlib.util.find_spec("hwoutils") is not None
    and importlib.util.find_spec("hwoutils.testbudget") is not None
):
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

BACKENDS = [pytest.param(np, id="numpy")]

try:  # jax is optional; the suite must pass without it
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    BACKENDS.append(pytest.param(jnp, id="jax"))
except ImportError:
    pass


@pytest.fixture(params=BACKENDS)
def xp(request):
    """Raw array namespace (numpy or jax.numpy) used to build test inputs."""
    return request.param
