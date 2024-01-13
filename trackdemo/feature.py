import numpy as np
from skimage import filters, measure


def label_hierarchy_array(hier_arr):
    """Labels an array of Hierarchy instances sequentially"""
    total_index = 0
    for hierarchy in hier_arr:
        hierarchy.label_nodes(start_index = total_index)
        total_index = hierarchy._index 


def compute_segementation_metrics(hier_arr):
    """Computer each candidates segementation in hierarchy prepare for weight calculation"""
    for i in range(len(hier_arr)):

        hier = hier_arr[i]
        coords = hier.root.value

        assert len(coords.shape) == 2, "Coordiantes of hierarchy seems get wrong shape"
        n_dim = coords.shape[1]
        assert n_dim in (2,3), "Only can handle 2D/3D cases now"

        for node in hier.all_nodes():
            sub_coords = coords[np.array(node.value)]
            n_dim_coords = [sub_coords[:, n] for n in range(n_dim)]
            centroid = [np.mean(coord) for coord in n_dim_coords]

            node.area = len(sub_coords)
            node.frame = i
            node.centroid = tuple(centroid)
            node.bound = np.vstack((np.min(sub_coords, axis=0), np.max(sub_coords, axis=0))).T
            node.value = sub_coords
            
            mask  = np.zeros(node.shape)
            mask[sub_coords[:, 0], sub_coords[:, 1]] = 1
            labeled_mask, num_features = measure.label(mask, connectivity=1, return_num=True)
            if num_features > 1:
                component_sizes = [np.sum(labeled_mask == label) for label in range(1, num_features + 1)]
                largest_component_label = np.argmax(component_sizes) + 1  # Add 1 to match label indices
                largest_component_mask = (labeled_mask == largest_component_label)
                largest_component_coords = np.column_stack(np.nonzero(largest_component_mask))
                node.value = largest_component_coords

