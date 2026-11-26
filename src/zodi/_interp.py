"""Clamped table interpolation shared by every backend.

Hand-rolled with ``searchsorted`` plus gathers so one implementation serves
numpy and jax (scipy interpolators cannot), stays differentiable, and
reproduces the clamp-at-table-edge convention used by zodiacal brightness
tables: queries outside the grid return the nearest tabulated value rather
than NaN.

Device operands determine the working precision. Host grids and tables are
lifted at that precision, with host grid spacing and offsets computed first.
"""

import numpy as np

from zodi._xp import _working_namespace, array_namespace

__all__ = ["interp1d_clamped", "interp2d_bilinear_clamped"]


def _cell_index(xp, grid, x):
    """Left cell index for each query, clamped to the valid cells."""
    return xp.clip(xp.searchsorted(grid, x, side="right") - 1, 0, grid.shape[0] - 2)


def _cell_fraction(xp, grid, query, dtype):
    """Find the lower knot and fractional position on a clamped axis."""
    query = xp.asarray(query, dtype=dtype)
    if dtype is not None and array_namespace(grid) is array_namespace():
        return _host_cell_fraction(xp, np.asarray(grid), query, dtype)

    axis = xp.asarray(grid, dtype=dtype)
    x = xp.clip(query, axis[0], axis[-1])
    i = _cell_index(xp, axis, x)
    return i, (x - axis[i]) / (axis[i + 1] - axis[i])


def _host_cell_fraction(xp, grid, query, dtype):
    """Stage host widths and offsets while retaining knot order for queries."""
    rounded = grid.astype(dtype)
    offsets = rounded - grid
    # Search with the least representable value at or above each host knot.
    # A query equals or exceeds a host knot exactly when it reaches this value.
    ceiling = rounded.copy()
    np.nextafter(rounded, dtype.type(np.inf), where=offsets < 0, out=ceiling)
    ends = rounded[[0, -1]]
    np.nextafter(ends, dtype.type(-np.inf), where=offsets[[0, -1]] > 0, out=ends)

    axis = xp.asarray(rounded)
    widths = xp.asarray(np.diff(grid), dtype=dtype)
    offsets = xp.asarray(offsets, dtype=dtype)
    x = xp.clip(query, ends[0], ceiling[-1])
    i = _cell_index(xp, xp.asarray(ceiling), query)
    # At the rounded right knot, anchor there to retain its residual and slope.
    at_right = x == axis[i + 1]
    anchor = i + at_right
    fraction = at_right + ((x - axis[anchor]) + offsets[anchor]) / widths[i]
    fraction = xp.where(query < ceiling[0], xp.asarray(0, dtype=dtype), fraction)
    fraction = xp.where(query > ends[-1], xp.asarray(1, dtype=dtype), fraction)
    return i, fraction


def interp1d_clamped(x_grid, values, x):
    """Piecewise-linear interpolation with edge clamping.

    Args:
        x_grid: Strictly increasing sample locations, shape (n,).
        values: Sampled values, shape (n,).
        x: Query locations, any shape.

    Returns:
        Interpolated values with the shape of ``x``; queries beyond the grid
        return the corresponding edge value.
    """
    xp, dtype = _working_namespace(x_grid, values, x)
    vg = xp.asarray(values, dtype=dtype)
    i, t = _cell_fraction(xp, x_grid, x, dtype)
    return (1 - t) * vg[i] + t * vg[i + 1]


def interp2d_bilinear_clamped(x_grid, y_grid, table, x, y):
    """Bilinear interpolation of a 2-D table with edge clamping.

    Args:
        x_grid: Strictly increasing first-axis sample locations, shape (nx,).
        y_grid: Strictly increasing second-axis sample locations, shape (ny,).
        table: Sampled values, shape (nx, ny).
        x: First-axis queries.
        y: Second-axis queries, broadcast-compatible with ``x``.

    Returns:
        Interpolated values; queries beyond either grid clamp to the edge.
    """
    xp, dtype = _working_namespace(x_grid, y_grid, table, x, y)
    tab = xp.asarray(table, dtype=dtype)
    i, tx = _cell_fraction(xp, x_grid, x, dtype)
    j, ty = _cell_fraction(xp, y_grid, y, dtype)
    return (
        (1 - tx) * (1 - ty) * tab[i, j]
        + (1 - tx) * ty * tab[i, j + 1]
        + tx * (1 - ty) * tab[i + 1, j]
        + tx * ty * tab[i + 1, j + 1]
    )
