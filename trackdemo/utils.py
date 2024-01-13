import time
import numpy as np
import pandas as pd
from PIL import Image
import os

from .hierarchy import Hierarchy


def load(basedir, io):
    return read_folder(basedir, io)

def read_folder(basedir, io):
    files = io.get_image_files(basedir)
    imgs = [io.imread(f) for f in files]

    return imgs


def format_output(hier_arr, n, edges):
    n_set = set(n)
    label_assigned = {}
    mask_arr = []

    for t in range(len(hier_arr)):
        hier = hier_arr[t]
        label = 1
        mask = np.zeros(hier.root.shape)
        for node in hier.all_nodes():
            if node.index in n_set:
                assert node.frame == t, "Segementation's frame should consist with hierarchy fram"
                mask[node.value[:,0], node.value[:,1]] = label
                label_assigned[node.index] = {"frame": node.frame, "label" : label} 
                label += 1
        mask_arr.append( mask)

    
    data = []
    for edge in edges:
        source_label = label_assigned.get(edge[0])
        target_label = label_assigned.get(edge[1])
        if source_label and target_label:
            data.append([
                source_label["label"],
                source_label["frame"],
                target_label["label"],
                target_label["frame"]
            ])

    # Create DataFrame
    edge_df = pd.DataFrame(data, columns=['Source Label', 'Source Frame', 'Target Label', 'Target Frame'])
    
    return mask_arr, edge_df 



def store_output(mask_arr, edge_df, basedir):
    # Create directory if it doesn't exist
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # Save mask images
    for idx, mask in enumerate(mask_arr):
        mask_image = Image.fromarray(mask, 'L')
        mask_image_path = os.path.join(basedir, f'mask_{idx}.png')
        mask_image.save(mask_image_path)

    # Save DataFrame to CSV
    edge_df.to_csv(os.path.join(basedir, 'edge_data.csv'), index=False)


def hierarchies_to_df(hier_arr):
    df_list = []
    for hier in hier_arr:
        df_list.append(hier.to_df())
    return pd.concat(df_list, ignore_index=True)


def df_to_hierarchies(df):
    hier_arr = []
    frames = df['Frame'].unique()

    for frame in frames:
        frame_df = df[df['Frame'] == frame]
        hier = Hierarchy.read_df(frame_df)
        hier_arr.append(hierarchy)

    return hier_arr