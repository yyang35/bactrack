import pandas as pd
import numpy as np
import glob
import os
import re
import warnings
import networkx as nx
from typing import Set
import sys
import cv2
from natsort import natsorted

import extractor 
from cell import Cell


class LinkComposer:



    def __init__(self, cells: Set[Cell]):
        self.cells = cells
        self.cells_frame_dict = self.get_cells_frame_dict(cells)
        self.frame_num = len(self.cells_frame_dict)



    # a dictionary of cells, with key as frame index, and value as a set of cell
    def get_cells_frame_dict(self, cells: Set[Cell]) -> dict:
        cells_frame_dict = {}
        for cell in cells:
            if cell.frame not in cells_frame_dict:
                cells_frame_dict[cell.frame] = {cell}
            else:
                cells_frame_dict[cell.frame].add(cell)

        return cells_frame_dict
    


    # provide new graphy base on cell set data 
    def make_new_dircted_graph(self):
        G = nx.DiGraph()
        for cell in self.cells:
            G.add_node(cell)

        return G



    # link two cells by adding edge on graph, protected graph by assert nodes are in graph
    def link(self, G, cell1, cell2):
        assert cell1 in self.cells, "source cell not in cells"
        assert cell2 in self.cells, "target cell not in cells"
        G.add_edge(cell1, cell2)


    
    # extract cells and construct other related info from phase tif, and mask tif #
    @staticmethod
    def read_tif(mask_tif: str, phase_tif = None):
        phase_tif  = mask_tif if phase_tif is None else phase_tif
        mask_dimension, phase_dimension = extractor.get_tiff_info(mask_tif),  extractor.get_tiff_info(phase_tif)
        assert  mask_dimension == phase_dimension, f"phase, mask folder don't have same dimension, phase: {phase_dimension} mask: {mask_dimension}, "
        masks = extractor.read_tiff_sequence(mask_tif)
        cells, error = extractor.get_cells_set_by_mask_dict(masks, force=True)

        composer = LinkComposer(cells)
        composer.error = error
        composer.phase_tif = phase_tif
        composer.mask_tif = mask_tif

        return composer



    # extract cells and construct other related info from phase folder, and mask folder #
    @staticmethod
    def read_folder(mask_folder: str, phase_folder = None):
        phase_folder = mask_folder if phase_folder is None else phase_folder
        mask_dimension, phase_dimension = extractor.get_folder_info(mask_folder) , extractor.get_folder_info(phase_folder)
        assert  mask_dimension == phase_dimension, f"phase, mask folder don't have same dimension, phase: {phase_dimension} mask: {mask_dimension}, "
        masks = extractor.get_mask_dict(mask_folder)
        cells, error  = extractor.get_cells_set_by_mask_dict(masks, force = True)

        composer = LinkComposer(cells)
        composer.error = error
        composer.phase_folder = phase_folder
        composer.mask_folder = mask_folder

        return composer


    # show error mask on extracting cells
    def show_mask_error(self):
        import visualizer
        if hasattr(self, 'mask_tif'):
            masks = extractor.read_tiff_sequence(self.phase_tif)
        elif hasattr(self, 'mask_folder'):
            masks = extractor.get_mask_dict(self.mask_folder)
        else: 
            raise Exception("Composer do't have mask info, use visualizer.plot_error_masks(mask,error) to plot")

        return visualizer.plot_error_masks(mask=masks, error= self.error)
    


    def show_frame_phase(self, G, frame, info = None, **kwargs):
        import visualizer
        this_frame_cells = self.cells_frame_dict[frame]
        connected_cells,_  = visualizer.get_connected_edges_cells(G, this_frame_cells)
        info = visualizer.get_label_info(G, list(connected_cells.union(this_frame_cells)), alphabet_label=True) if info is None else info
        image = self.get_single_frame_phase(frame)
        return visualizer.plot_single_frame_phase(G, info, frame, image, cells_frame_dict=self.cells_frame_dict, **kwargs)
        
    
    def get_single_frame_phase(self, frame):
        import visualizer
        if hasattr(self, 'phase_tif'):
            image = extractor.read_tiff_frame_like_cv2(self.phase_tif, frame)
        elif hasattr(self, 'phase_folder'):
            image = extractor.read_tiff_in_folder(self.phase_folder, frame)
        elif hasattr(self, "mask_tif"): 
            masks = extractor.read_tiff_sequence(self.phase_tif)
            mask_image = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE)
            height, width = mask_image.shape[:2]
            image = np.full((height, width, 3), 128, dtype=np.uint8) 
        elif hasattr(self, 'mask_folder'):
            masks = extractor.get_mask_dict(self.mask_folder)
            mask_image = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE)
            height, width = mask_image.shape[:2]
            image = np.full((height, width, 3), 128, dtype=np.uint8) 
        else:
            raise Exception("no phase")

        return image
        


    #=================== following are read link result file related function =================== #
    
    def get_manual_link_dict(self, excel_path):
        assert excel_path.endswith('.xlsx'), "File must be an Excel file with a .xlsx extension"

        G = self.make_new_dircted_graph()

        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names

        assert self.frame_num == len(sheet_names) + 1, "Linking frames count not same with masks"

        for frame_start in range(len(sheet_names)):
            frame_end = frame_start + 1
            time_sheet = sheet_names[frame_start]
            df = excel_file.parse(time_sheet)
            df = df.astype(str)
            [source, target] = df.columns
            for index, row in df.iterrows():
                df = df.dropna()
                df = df.astype(float).astype(int)
                self.link(G, Cell(frame_start, int(row[source])), Cell(frame_end, int(row[target])))

        return G



    def get_supersegger_file_info_and_tracker_result(self,foldername):
        
        G = self.make_new_dircted_graph()

        npzFiles = glob.glob(foldername)
        npzFiles.sort()
        assert self.frame_num == len(npzFiles), "Linking frames count not same with masks"
        # Don't need last file, there no linking 
        npzFiles.pop()
    
        for frame_start in npzFiles:
            f = npzFiles[frame_start]
            frame_end = frame_start + 1
            data = loadmat(f)
            label = data['regs']['regs_label'][0][0]
            track_result = data['regs']['map'][0][0]['f'][0][0][0]
            
            track_dict = {}
            for i in range(len(track_result)):
                if len(track_result[i][0]) > 0:
                    # Matlab index start at 1, but python start with 0
                    assert(Cell(frame_start, i+1) in G)
                    target_cells = set(track_result[i][0])

                    for target_cell in target_cells:
                        assert(Cell(frame_end, target_cell) in G)
                        G.add_edge(Cell(frame_start, i+1), Cell(frame_end,target_cell))

        return G
    


    def get_trackmate_linking_result(self,spots_filename, edge_filename,  UNIT_CONVERT_COEFF = 1):
        spots = self._match_trackmate_cell_id_to_mask_label(spots_filename,  UNIT_CONVERT_COEFF =  UNIT_CONVERT_COEFF )
        G =  self._abstact_tackmate_assignment_by_edges_file(spots, edge_filename)
        return G
    


    def _match_trackmate_cell_id_to_mask_label(self, spots_filename, UNIT_CONVERT_COEFF = 1):

        # Read top 4 line as header by trackmate dataformat
        spots = pd.read_csv(spots_filename, header=[0, 1, 2, 3])
        # Only use the first row header for convenient
        spots.columns = spots.columns.get_level_values(0)

        # trackmate also have frame start with 0
        assert self.frame_num == np.max(spots["FRAME"])+1, "The number of masks and trackmate frame number is inconsist,  contents of folders don't match up"

        trackmate_frame_index = spots["FRAME"].unique()
        trackmate_frame_index.sort()

        for frame_index in range(len(trackmate_frame_index)):
            frame = trackmate_frame_index[frame_index]
            cells_in_frame = spots[spots["FRAME"] == frame]
            cells_in_mask = self.cells_frame_dict[frame_index]

            for index, row in cells_in_frame.iterrows():
                # Trackmate using timeframe start from 0
                cell_id = row["ID"]
                # Trackmate use different unit of position, check trackmate document / tracks excel 
                trackmate_x = row["POSITION_X"]*UNIT_CONVERT_COEFF
                trackmate_y = row["POSITION_Y"]*UNIT_CONVERT_COEFF

                shorest_distance = sys.maxsize
                matched_cell = None
                total_good_candidates = 0

                for cell in cells_in_mask:
                    distance = (cell.polygon.centroid.x - trackmate_x) ** 2 + (cell.polygon.centroid.y - trackmate_y) ** 2 
                    if distance < 2 : total_good_candidates += 1 
                    if distance < shorest_distance:
                        shorest_distance = distance
                        matched_cell = cell

                if total_good_candidates == 0:
                    warnings.warn(f"Trackmate cell:{cell_id} match back to mask is inaccute, assigned to the nearest cell.")
                if total_good_candidates > 1:
                    warnings.warn(f"Trackmate cell:{cell_id} around by dense cells, multiple candidates, assigned to the nearest cell.")

                spots.loc[index, "cell"] = matched_cell

        return spots
        

    def _abstact_tackmate_assignment_by_edges_file(self, spots_df, edge_filename):
        # Read top 4 line as header by trackmate dataformat
        tracks = pd.read_csv(edge_filename, header=[0, 1, 2, 3])
        # Only use the first row header for convenient
        tracks.columns =  tracks.columns.get_level_values(0)

        G = self.make_new_dircted_graph()

        spots_reduced = spots_df[['ID', 'cell']]
        assert spots_reduced.isna().sum().sum() == 0

        tracks['cell'] = "" # for later suffixes convenient

        tracks = tracks.merge(
            spots_reduced[['ID', 'cell']],
            left_on='SPOT_SOURCE_ID',
            right_on='ID',
            suffixes=('', '_source')
        ).drop('ID', axis=1)

        # Second join on 'SPOT_TARGET_ID'
        tracks = tracks.merge(
            spots_reduced[['ID', 'cell']],
            left_on='SPOT_TARGET_ID',
            right_on='ID',
            suffixes=('', '_target')
        ).drop('ID', axis=1)

        tracks = tracks.drop('cell', axis=1)

        for index, row in tracks.iterrows():
            G.add_edge(row["cell_source"], row["cell_target"])

        return G

