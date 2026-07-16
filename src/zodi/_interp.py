"""Clamped table interpolation shared by every backend.

Hand-rolled with ``searchsorted`` plus gathers so one implementation serves
numpy and jax (scipy interpolators cannot), stays differentiable, and
reproduces the clamp-at-table-edge convention used by zodiacal brightness
tables: queries outside the grid return the nearest tabulated value rather
than NaN.

Dispatch follows the QUERY arrays: grids and tables are static data and are
lifted into the query namespace with ``asarray``, so numpy tables combined
with jax queries trace cleanly (the tabulated-physics usage pattern).
"""

from zodi._xp import array_namespace

__all__ = ["interp1d_clamped", "interp2d_bilinear_clamped"]


def _cell_index(xp, grid, x):
    """Left cell index for each query, clamped to the valid cells."""
    return xp.clip(xp.searchsorted(grid, x, side="right") - 1, 0, grid.shape[0] - 2)


def interp1d_clamped(x_grid, values, x):
    """Piecewise-linear interpolation with edge clamping.

    Args:
        x_grid: Strictly increasing sample locations, shape (n,).
        values: Sampled values, shape (n,).
        x: Query locations, any shape; dispatch follows this argument.

    Returns:
        Interpolated values with the shape of ``x``; queries beyond the grid
        return the corresponding edge value.
    """
    xp = array_namespace(x)
    xg = xp.asarray(x_grid)
    vg = xp.asarray(values)
    xq = xp.clip(xp.asarray(x), xg[0], xg[-1])
    i = _cell_index(xp, xg, xq)
    t = (xq - xg[i]) / (xg[i + 1] - xg[i])
    return (1 - t) * vg[i] + t * vg[i + 1]


def interp2d_bilinear_clamped(x_grid, y_grid, table, x, y):
    """Bilinear interpolation of a 2-D table with edge clamping.

    Args:
        x_grid: Strictly increasing first-axis sample locations, shape (nx,).
        y_grid: Strictly increasing second-axis sample locations, shape (ny,).
        table: Sampled values, shape (nx, ny).
        x: First-axis queries; dispatch follows the query arguments.
        y: Second-axis queries, broadcast-compatible with ``x``.

    Returns:
        Interpolated values; queries beyond either grid clamp to the edge.
    """
    xp = array_namespace(x, y)
    xg = xp.asarray(x_grid)
    yg = xp.asarray(y_grid)
    tab = xp.asarray(table)
    xq = xp.clip(xp.asarray(x), xg[0], xg[-1])
    yq = xp.clip(xp.asarray(y), yg[0], yg[-1])
    i = _cell_index(xp, xg, xq)
    j = _cell_index(xp, yg, yq)
    tx = (xq - xg[i]) / (xg[i + 1] - xg[i])
    ty = (yq - yg[j]) / (yg[j + 1] - yg[j])
    return (
        (1 - tx) * (1 - ty) * tab[i, j]
        + (1 - tx) * ty * tab[i, j + 1]
        + tx * (1 - ty) * tab[i + 1, j]
        + tx * ty * tab[i + 1, j + 1]
    )
