import os
import logging

import pandas as pd
import numpy as np
from shapely.geometry import Polygon

from bactrack import io
from bactrack import core
from bactrack import gui
from cellpose_omni import io as omni_io
from bactrack.core import ModelEnum
from bactrack.gui.cell import Cell
from bactrack.gui.composer import LinkComposer
import bactrack.gui.extractor  as extractor
import bactrack.gui.visualizer as visualizer


def run_track(path, solver_name, weight_name, submodel, hypermodel, file_extension):
    logging.info(f"Running tracking on {path}")
    logging.info(f"Using solver: {solver_name}, weight: {weight_name}, hypermodel: {hypermodel}, submodel: {submodel}, file_extension: {file_extension}")

    seg_file = path + ".segmentation.pkl"
    if os.path.exists( seg_file):
        hier_arr = pd.read_pickle(seg_file)
    else:
        images = io.load(path, omni_io)
        hier_arr = core.compute_hierarchy(images,hypermodel=hypermodel, submodel=submodel)
        #seems panda could save any object type to pickle, here is backtrack heriarchy
        pd.to_pickle(hier_arr, seg_file) 

    nodes, edges = core.run_tracking(hier_arr, solver_name = solver_name, weight_name = weight_name)
    mask_arr, edge_df = io.format_output(hier_arr, nodes, edges)
    hier_df = io.hiers_to_df(hier_arr)

    merged_df = pd.merge(edge_df, hier_df.add_suffix('_source'), left_on='Source Index', right_on='index_source', how='left')
    merged_df = pd.merge(merged_df, hier_df.add_suffix('_target'), left_on='Target Index', right_on='index_target', how='left')
    n_selected =  hier_df[hier_df['label'].notna()]

    cells = set()
    for index, row in n_selected.iterrows():
        binary_mask = np.zeros(row['shape'], dtype=np.uint8)
        binary_mask[row['value'][:,0], row['value'][:,1]] = 1
        polygon =  extractor.single_cell_mask_to_polygon(binary_mask)
        cells.add(Cell(polygon = polygon, label = row['index'], frame=row['frame']))

    composer =  LinkComposer(cells=cells)
    composer.phase_folder = os.path.join(path , file_extension)
    G = composer.make_new_dircted_graph()

    for index, row in merged_df.iterrows():
        #G.add_edge(Cell(label = row['Source Index'], frame = row['frame_source']), Cell(label = row['Target Index'], frame = row['frame_target']))
        try:
            composer.link(G, Cell(label = row['Source Index'], frame = row['frame_source']), Cell(label = row['Target Index'], frame = row['frame_target'])) 
        except:
            print(f"Cannot link {row['Source Index']} to {row['Target Index']}")

    logging.info(f"Tracking completed: {len(G.nodes)} IOU tracked")
    return composer, G


def open_in_napari(composer, G):
    logging.info("Opening in Napari")
    pass