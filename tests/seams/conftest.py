"""Collect the consumer seam when its optional stack is installed."""

from importlib.util import find_spec

collect_ignore = (
    []
    if all(
        find_spec(name) is not None
        for name in ("jax", "skyscapes", "optixstuff", "orbix")
    )
    else ["test_skyscapes_precision.py"]
)
