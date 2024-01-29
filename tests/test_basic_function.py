import numpy as np
from bactrack import core
import pytest

@pytest.fixture(scope='session')
def compute_hier_arr():
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
    return hier_arr


def test_segment(compute_hier_arr):
    hier_arr = compute_hier_arr
    assert(len(hier_arr) == 2)


def test_tracking_weight(compute_hier_arr):
    hier_arr = compute_hier_arr
    nodes, edges = core.run_tracking(hier_arr, weight_name='iou_weight', solver_name = "scipy_solver")
    assert(nodes is not None)
    nodes, edges = core.run_tracking(hier_arr, weight_name='overlap_weight', solver_name = "scipy_solver")
    assert(nodes is not None)
    nodes, edges = core.run_tracking(hier_arr, weight_name='distance_weight', solver_name = "scipy_solver")
    assert(nodes is not None)


def test_tracking_solver(compute_hier_arr):
    hier_arr = compute_hier_arr
    nodes, edges = core.run_tracking(hier_arr, weight_name='overlap_weight', solver_name = "scipy_solver")
    assert(nodes is not None)
    nodes, edges = core.run_tracking(hier_arr, weight_name='overlap_weight', solver_name = "mip_solver")
    assert(nodes is not None)


def test_separate_step(compute_hier_arr):
    hier_arr = compute_hier_arr
    from bactrack.tracking import OverlapWeight, IOUWeight, DistanceWeight
    w =  OverlapWeight(hier_arr)
    from bactrack.tracking import MIPSolver
    solver = MIPSolver(w.weight_matrix, hier_arr, coverage=1.0)
    n, e = solver.solve()
    assert(n is not None)