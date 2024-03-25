
import numpy as np
from bactrack.hierarchy import Node, Hierarchy
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

def get_segmentation_hierarchy_from_masks_folder(masks_folder):
    """
    Given a folder with masks, return the segmentation hierarchy.
    """
    pass

    
def get_segmentation_hierarchy_from_masks(images):
    """
    Given a list of images, return the segmentation hierarchy.
    """
    for image in images:
        max_label = np.max(image)
        root_node = Node(
            values = 
            super = -1, 


        )

        for i in range(0, max_label+1):

            binary_mask = image == i

            polygon =  extractor.single_cell_mask_to_polygon(binary_mask)
            cells.add(Cell(polygon = polygon, label = i, frame=0))
        



