import numpy as np
import pandas as pd
from PIL import Image
import os
import logging
import glob 
from natsort import natsorted
import fastremap

from .hierarchy import Hierarchy

io_logger = logging.getLogger(__name__)


def load(data, seg_io):

    def is_valid_image_structure(data):
        #can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
        if isinstance(data, list):
            return all(isinstance(item, np.ndarray) and item.ndim in [2, 3] for item in data)
        elif isinstance(data, np.ndarray):
            return data.ndim in [3, 4]
        return False

    if isinstance(data, str) and os.path.isdir(data):
        files = get_image_files(data)
        imgs = [seg_io.imread(f) for f in files]
        return imgs
    elif is_valid_image_structure(data):
        return data  
    elif all(isinstance(item, str) for item in data):
        imgs = [seg_io.imread(f) for f in data]
        return imgs
    else:
        io_logger.warning(f'Invalid input: {data}')
        return None


def format_output(hier_arr, nodes, edges, label_format = "default", overwrite_mask = True):

    if overwrite_mask is True:
        label_format = label_format.lower()
        if label_format == "default":
            _label_default_format(hier_arr, nodes, edges)
        elif label_format == "kevin":
            _label_kevin_format(hier_arr, nodes, edges)
        else:
            raise ValueError(f"Label format {label_format} not found")

    mask_arr = []
    for t in range(len(hier_arr)):
        hier = hier_arr[t]
        mask = np.zeros(hier.root.shape)
        for node in hier.all_nodes():
            if node.label is not None:
                mask[node.value[:,0], node.value[:,1]] = node.label
        mask_arr.append(mask)
    
    data = []

    for (source_index, target_index), _ in edges.items():
    # Assuming hier_arr is indexed the same way as edges
        data.append([source_index, target_index,])

    edge_df = pd.DataFrame(data, columns=['Source Index', 'Target Index'])
    
    return mask_arr, edge_df 

def _label_kevin_format(hier_arr, nodes, edges):
    from queue import Queue

    n_set = set(nodes)
    dict = {}
    for (source_index, target_index), _ in edges.items():
        if source_index in dict:
            dict[source_index].append(target_index)
        else:
            dict[source_index] = [target_index]

    index_nodes_dict = {}
    for hier in hier_arr:
        for node in hier.all_nodes():
            node.label = None
            index_nodes_dict[node.index] = node  

    # Following are BFS on lineage tree
    queue = Queue()
    label = 1
    for node in hier_arr[0].all_nodes():
        if node.index in n_set:
            queue.put(node)
            node.label = label
            label += 1

    frame = 0
    # Do Width First Search on lineage tree here:   
    while queue.empty() == False:
        current_node = queue.get()
        if current_node.frame != frame:
            frame = current_node.frame
            for node in hier_arr[frame].all_nodes():
                if node.index in n_set and node.label is None:
                    queue.put(node)
                    node.label = label
                    label += 1

        target_indices = dict.get(current_node.index)
        if target_indices is None:
            continue
        # if divide, then create new labels for each target cell
        if len(target_indices) > 1:
            for target_index in target_indices:
                target = index_nodes_dict[target_index]
                assert target.label is None, "target cell should not have label"
                queue.put(target)
                target.label = label 
                label += 1

        # if not divide, then same label with source cell
        elif len(target_indices) == 1:
            target = index_nodes_dict[target_indices[0]]
            assert target.label is None, "target cell should not have label"
            queue.put(target)
            target.label = current_node.label
            
    
def _label_default_format(hier_arr, nodes, edges):
     # n_set is a set of all selected IOU's index
    n_set = set(nodes)
    for hier in hier_arr:
        for node in hier.all_nodes():
            node.label = None

    for t in range(len(hier_arr)):
        hier = hier_arr[t]
        label = 1
        for node in hier.all_nodes():
            if node.index in n_set:
                assert node.frame == t, "Segmentation's frame should consist with hierarchy frame"
                node.label = label
                label += 1


def store_mask_arr(mask_arr, basedir):
    # Create directory if it doesn't exist
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # Save mask images
    for idx, mask in enumerate(mask_arr):
        mask = mask.astype(np.uint32)
        # Resize the array to the smallest dtype 
        labels = fastremap.refit(labels)
        mask_image = Image.fromarray(mask)
        mask_image_path = os.path.join(basedir, f'mask_{idx}.png')
        mask_image.save(mask_image_path)


def hiers_to_df(hier_arr):
    df_list = []
    for hier in hier_arr:
        df_list.append(hier.to_df())
    return pd.concat(df_list, ignore_index=True)


def df_to_hiers(df):
    hier_arr = []
    frames = df['frame'].unique()
    df['super'] = df['super'].astype('Int32')
    
    for frame in frames:
        frame_df = df[df['frame'] == frame]
        hier = Hierarchy.read_df(frame_df)
        hier._index = tuple(hier._index)
        hier_arr.append(hier)

    return hier_arr


def get_image_files(folder, extensions = ['png','jpg','jpeg','tif','tiff'], pattern=None):
    """ find all images in a folder and if look_one_level_down all subfolders """

    image_names = []

    for ext in extensions:
        image_names.extend(glob.glob(folder + ('/*.'+ext)))
    
    image_names = natsorted(image_names)
    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names