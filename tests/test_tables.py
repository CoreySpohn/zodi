"""Pinned spot values from the published Leinert (1998) tables."""

import numpy as np

from zodi import tables


def test_grid_shapes_and_monotonicity():
    assert tables.TABLE17_RAW.shape == (19, 11)
    assert tables.LON_GRID_DEG.shape == (19,)
    assert tables.BETA_GRID_DEG.shape == (11,)
    assert np.all(np.diff(tables.LON_GRID_DEG) > 0)
    assert np.all(np.diff(tables.BETA_GRID_DEG) > 0)
    assert tables.TABLE19_WAVELENGTH_UM.shape == (16,)
    assert np.all(np.diff(tables.TABLE19_WAVELENGTH_UM) > 0)


def test_table17_published_spot_values():
    # (lon, beta) -> published value in 1e-8 W m-2 sr-1 um-1
    lon = list(tables.LON_GRID_DEG)
    beta = list(tables.BETA_GRID_DEG)
    assert tables.TABLE17_RAW[lon.index(0), beta.index(15)] == 3140
    assert tables.TABLE17_RAW[lon.index(15), beta.index(0)] == 11500
    assert tables.TABLE17_RAW[lon.index(90), beta.index(0)] == 259
    assert tables.TABLE17_RAW[lon.index(180), beta.index(0)] == 230
    np.testing.assert_array_equal(tables.TABLE17_RAW[:, beta.index(90)], 77)


def test_near_sun_exclusion_cells_are_nan_in_raw():
    assert np.isnan(tables.TABLE17_RAW[0, 0])
    assert np.isnan(tables.TABLE17_RAW[2, 1])
    assert np.isfinite(tables.TABLE17_RAW[3:]).all()


def test_filled_table_clamps_to_nearest_along_row():
    assert np.isfinite(tables.TABLE17_FILLED).all()
    # row lon=0: first valid entry is beta=15 (3140); all nan cells take it
    np.testing.assert_array_equal(tables.TABLE17_FILLED[0, :3], 3140)
    # valid cells are untouched
    finite = np.isfinite(tables.TABLE17_RAW)
    np.testing.assert_array_equal(
        tables.TABLE17_FILLED[finite], tables.TABLE17_RAW[finite]
    )


def test_reference_cell_and_table19_anchor():
    assert tables.TABLE17_REFERENCE == 259.0
    i500 = list(tables.TABLE19_WAVELENGTH_UM).index(0.5)
    assert tables.TABLE19_RADIANCE_W_M2_SR_UM[i500] == 2.6e-6
    # the ~0.4 percent anchoring mismatch documented in the module
    ratio = 2.6e-6 / (259.0 * tables.TABLE17_UNIT_W_M2_SR_UM)
    assert 1.003 < ratio < 1.005
