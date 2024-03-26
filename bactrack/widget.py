
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

        



