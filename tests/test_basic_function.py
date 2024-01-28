import numpy as np
from bactrack import core

def test_segement():
    # Create a single circle mask
    diameter = 30
    radius = diameter // 2
    Y, X = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2)
    mask = dist_from_center <= radius
    single_mask = np.where(mask, 0, 255).astype(np.uint8)  # Black circle, white background

    # Duplicate the mask
    num_masks = 2
    masks = [single_mask.copy() for _ in range(num_masks)]
    masks_array = np.array(masks)

    hier_arr = core.compute_hierarchy(masks_array, submodel= 'bact_phase_omni')
    assert(len(hier_arr) == 2)

    nodes, edges = core.run_tracking(hier_arr)
    assert(nodes is not None)


