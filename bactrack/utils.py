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
    mask_arr = []

    for t in range(len(hier_arr)):
        hier = hier_arr[t]
        label = 1
        mask = np.zeros(hier.root.shape)
        for node in hier.all_nodes():
            if node.index in n_set:
                assert node.frame == t, "Segementation's frame should consist with hierarchy fram"
                mask[node.value[:,0], node.value[:,1]] = label
                node.label = label
                label += 1
        mask_arr.append( mask)

    data = []

    for (source_index, target_index), _ in e.items():
    # Assuming hier_arr is indexed the same way as edges
        data.append([source_index, target_index,])

    edge_df = pd.DataFrame(data, columns=['Source Index', 'Target Index'])
    
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
        hier_arr.append(hier)

    return hier_arr
