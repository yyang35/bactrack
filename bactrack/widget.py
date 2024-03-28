
import numpy as np
from bactrack.hierarchy import Node, Hierarchy
from bactrack.io import get_image_files
from PIL import Image


def get_hierarchies_from_masks_folder(masks_folder):
    """
    Given a folder with masks, return the segmentation hierarchy.
    """
    image_files = get_image_files(masks_folder)
    images = []
    for file in image_files:
        image = Image.open(file).convert('L')
        label_mask = np.array(image)
        images.append(label_mask)

    return get_hierarchies_from_masks(images)

    
def get_hierarchies_from_masks(masks):
    """
    Given a list of images, return the segmentation hierarchy.
    """
    hier_arr = []
    for frame in range(len(masks)):
        mask= masks[frame]
        shape = mask.shape
        max_label = np.max(mask)
        root_node = Node(
            value = np.array(np.nonzero(mask)).T.astype(np.int32),
            super = None, # root node has no super, is represented as  -1 in df, and None in code
            shape = shape,
        )
        hier = Hierarchy(root_node)
        for i in range(1, max_label+1):
            if np.sum(mask == i) == 0:
                continue

            current_segment_node = Node(value = np.argwhere(mask == i))
            root_node.add_sub(current_segment_node)
            current_segment_node.super = root_node
            current_segment_node.label = i
            current_segment_node.frame = frame
            current_segment_node.shape = shape

        hier_arr.append(hier)
    
    Hierarchy.label_hierarchy_array(hier_arr)
    Hierarchy.compute_segmentation_metrics(hier_arr)
    return hier_arr

        



