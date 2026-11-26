"""Clamped interpolation: correctness, clamping, and cross-backend parity."""

import numpy as np
import pytest

from zodi._interp import interp1d_clamped, interp2d_bilinear_clamped

XG = np.linspace(0.0, 10.0, 11)
VG = np.sin(XG) + 2.0
YG = np.linspace(-1.0, 1.0, 5)
TABLE = np.cos(XG)[:, None] * (1.0 + 0.5 * YG)[None, :]

X_QUERIES = np.array([-3.0, 0.0, 0.4, 5.5, 9.7, 10.0, 42.0])  # ends clamp
Y_QUERIES = np.array([-2.0, -1.0, -0.3, 0.25, 0.9, 1.0, 3.0])


def test_interp1d_matches_numpy_reference(xp):
    """At the supplied query, allow a rounded knot and four weighted-sum ops.

    The grid has exact unit steps. The positive ordinates prevent cancellation.
    """
    query = xp.asarray(X_QUERIES)
    got = np.asarray(interp1d_clamped(XG, VG, query))
    expected = np.interp(np.asarray(query, dtype=np.float64), XG, VG)
    np.testing.assert_array_max_ulp(got, expected.astype(got.dtype), maxulp=5)


def test_interp1d_scalar_query_stays_plain_numpy():
    got = interp1d_clamped(XG, VG, 0.4)
    assert type(got).__module__.startswith("numpy")


def test_interp2d_exact_at_nodes(xp):
    """Zero and unit weights reproduce the stored knot exactly at its dtype."""
    xi, yj = np.meshgrid(XG, YG, indexing="ij")
    got = interp2d_bilinear_clamped(
        XG, YG, TABLE, xp.asarray(xi.ravel()), xp.asarray(yj.ravel())
    )
    np.testing.assert_array_equal(np.asarray(got), TABLE.ravel().astype(got.dtype))


def test_interp2d_midpoint_is_cell_average(xp):
    x_mid = 0.5 * (XG[3] + XG[4])
    y_mid = 0.5 * (YG[1] + YG[2])
    got = interp2d_bilinear_clamped(XG, YG, TABLE, xp.asarray(x_mid), xp.asarray(y_mid))
    expected = TABLE[3:5, 1:3].mean()
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-14)


def test_interp2d_clamps_to_edges(xp):
    got = interp2d_bilinear_clamped(
        XG,
        YG,
        TABLE,
        xp.asarray(np.array([-5.0, 99.0])),
        xp.asarray(np.array([0.0, 99.0])),
    )
    expected = np.array([np.interp(0.0, YG, TABLE[0]), TABLE[-1, -1]])
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-14)


def test_backends_agree_to_roundoff():
    """Use the table scale for cancellation between signed corner values.

    Twelve roundings cover cell fractions, two complements, corner products,
    three additions, and narrowing the ordinates at identical query positions.
    """
    jnp = pytest.importorskip("jax.numpy")
    from hwoutils.testing import ulps

    x, y = jnp.asarray(X_QUERIES), jnp.asarray(Y_QUERIES)
    b_np = interp2d_bilinear_clamped(
        XG, YG, TABLE, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    b_jax = interp2d_bilinear_clamped(XG, YG, TABLE, x, y)
    scale = jnp.asarray(np.max(np.abs(TABLE)), dtype=b_jax.dtype)
    np.testing.assert_allclose(b_jax, b_np, rtol=0, atol=float(ulps(12, scale)))
