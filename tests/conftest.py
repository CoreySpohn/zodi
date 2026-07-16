"""Shared fixtures: parametrize tests over the installed array backends."""

import numpy as np
import pytest

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
