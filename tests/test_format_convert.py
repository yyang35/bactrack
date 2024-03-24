from bactrack import io, core
import pytest
import numpy as np
import pandas as pd

@pytest.fixture(scope='session')
def compute_result():
    diameter = 30
    radius = diameter // 2
    Y, X = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2)
    mask = dist_from_center <= radius
    single_mask = np.where(mask, 0, 255).astype(np.uint8)

    num_masks = 2
    masks = [single_mask.copy() for _ in range(num_masks)]
    masks_array = np.array(masks)

    hier_arr = core.compute_hierarchy(masks_array, submodel='bact_phase_omni')
    nodes, edges = core.run_tracking(hier_arr, weight_name='overlap_weight', solver_name = "scipy_solver")

    return hier_arr, nodes, edges


def test_hiers_to_df_edge_case(compute_result):
    hier_arr, nodes, edges = compute_result
    df = io.hiers_to_df(hier_arr)
    assert len(df) > 0
    new_hier_arr = io.df_to_hiers(df)
    assert len(new_hier_arr) == len(hier_arr)
    assert new_hier_arr[-1]._index == hier_arr[-1]._index


def test_hiers_to_df_error_handling(compute_result):
    hier_arr, nodes, edges = compute_result
    m, e = io.format_output(hier_arr, nodes, edges)
    assert len(m) > 0
    assert isinstance(e, pd.DataFrame)
