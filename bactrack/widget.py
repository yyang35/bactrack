
import numpy as np
from bactrack.hierarchy import Node, Hierarchy, _format_hier
"""
self.value = value
self.super = kwargs.get('super', None)
self.subs = []

self.index = kwargs.get('index', None) # index corresponding to linking matrix 

self.shape = kwargs.get('shape', None) # Canvas shape
self.uncertainty = kwargs.get('uncertainty', None)
self.area = kwargs.get('area', None)
self.centroid = kwargs.get('centroid', None)
self.frame = kwargs.get('frame', None)
self.bound = kwargs.get('bound', None)

self.label = kwargs.get('label', None)  # only picked segmentation have label 
self.next = kwargs.get('next', None) # next frame node, only picked sgementation have next

"""

def get_hierarchies_from_masks_folder(masks_folder):
    """
    Given a folder with masks, return the segmentation hierarchy.
    """
    pass

    
def get_hierarchies_from_masks(images):
    """
    Given a list of images, return the segmentation hierarchy.
    """
    hier_arr = []
    for frame in len(images):
        image = images[frame]
        max_label = np.max(image)
        root_node = Node(
            values = np.argwhere(image != 0).tolist(),
            super = -1, 
            shape = image.shape,
        )
        hier = Hierarchy(root_node)
        for i in range(0, max_label+1):
            if np.sum(image == i) == 0:
                continue

            current_segment_node = Node(value = np.argwhere(image == i).tolist())
            root_node.add_sub(current_segment_node)
            current_segment_node.super = root_node
            current_segment_node.label = i
            current_segment_node.frame = frame

        Hierarchy.compute_segmentation_metrics(hier)
        hier_arr.append(hier)
        return hier_arr

        



