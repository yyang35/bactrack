import glob
import re
import warnings

import cv2
from matplotlib.patches import Polygon as pltPolygon
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from natsort import natsorted
from shapely.geometry import Polygon, LineString
from PIL import Image
import numpy as np

from cell import Cell



# general mask reader which sort file by nature order
# check natsorted for more info 
def get_mask_dict(foldername: str):
    filenames = glob.glob(foldername)
    # Using nature storted name here, for let name like 't1' < 't10' 
    sorted_filenames = natsorted(filenames)
    mask_dict = {}
    for i in range(len(sorted_filenames)):
        filename = sorted_filenames[i]

        img = Image.open(filename)
        img = img.convert('L')
        mask = np.array(img) 

        mask_dict[i] = mask

    return mask_dict


def get_folder_info(foldername: str):
    filenames = glob.glob(foldername)
    assert len(filenames) > 0, f"no image in {foldername}"
    # Using nature storted name here, for let name like 't1' < 't10' 
    frame_count = len(filenames)
    img = Image.open(filenames[0])
    first_frame_shape = np.array(img).shape

    return frame_count, first_frame_shape


def read_tiff_in_folder(phase_folder, frame):
    files = glob.glob(phase_folder)
    sorted_filenames = natsorted(files)
    image = cv2.imread(sorted_filenames[frame], cv2.IMREAD_UNCHANGED)

    # Normalize the image to increase contrast, if possible
    min_val = np.min(image)
    max_val = np.max(image)

    image = image.astype(np.float32)
    normalized_image = (image - min_val) / (max_val - min_val) * 255
    normalized_image = normalized_image.astype(np.uint8)
    rgb_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return rgb_image


def read_tiff_sequence(filename):
    mask_dict = {}
    with Image.open(filename) as img:
        i = 0
        while True:
            # Convert image to grayscale ('L')
            img.seek(i)
            img_converted = img.convert('L')
            mask = np.array(img_converted)
            mask_dict[i] = mask

            i += 1
            try:
                img.seek(i)
            except EOFError:
                # Exit loop when end of file is reached
                break

    return mask_dict



def get_tiff_info(filename):
    with Image.open(filename) as img:
        # Get the shape of the first frame
        img.seek(0)
        first_frame_shape = np.array(img).shape
        
        # Count the total number of frames
        frame_count = 1
        while True:
            try:
                img.seek(frame_count)
                frame_count += 1
            except EOFError:
                break

    return frame_count, first_frame_shape


    
def read_tiff_frame_like_cv2(filename, frame_number):
    try:
        with Image.open(filename) as img:
            img.seek(frame_number)
            frame_np = np.array(img)
    except (EOFError, FileNotFoundError, OSError):
        # If the frame doesn't exist, or file can't be opened, return None
        return None

    # Normalize if the dtype is uint16
    if frame_np.dtype == np.uint16:
        # Normalize to 0-255 range
        frame_np = (frame_np / 256).astype(np.uint8)

    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    return frame_rgb



def get_cells_set_by_mask_dict(mask_dict, force = False):
    cells_set = set()
    frame_keys = sorted(mask_dict)
    error = []
    for frame_index in range(len(frame_keys)):
        mask = mask_dict[frame_keys[frame_index]]
        # start from label = 1, label = 0 is background
        for mask_label in range(1,np.max(mask)+1):
            n_pixels = np.sum(mask == mask_label)
            if n_pixels > 0:
                try:
                    cell_mask = mask == mask_label
                    polygon = single_cell_mask_to_polygon(cell_mask)
                    cells_set.add(Cell(frame = frame_index, label = mask_label, polygon = polygon))
                except AssertionError as e:
                    if force:
                        error.append([frame_index, mask_label])
                        warnings.warn(f"{e}")
                    else:
                        raise
                except Exception as e:
                    # this is for some dirty manually changed mask, some un-noticed little picels might be not earsed
                    error.append([frame_index, mask_label])
                    print(f"Frame:{frame_index}, Mask label:{mask_label}. Pixels number = {n_pixels}. cannot make polygon. {e}")
    
    return cells_set, error



def single_cell_mask_to_polygon(cell_mask):
    cell_mask = (cell_mask * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found")

    # Assuming the largest contour is the exterior
    exterior = contours[0].reshape(-1, 2)
    exterior_polygon = Polygon(exterior)

    holes = []
    for i, contour in enumerate(contours):
        # Check if it's a hole (inside the exterior)
        if hierarchy[0][i][3] == 0:  # If it's a child of the exterior contour
            hole = contour.reshape(-1, 2)
            hole_line = LineString(hole)
            if not hole_line.within(exterior_polygon):
                raise ValueError("Found a contour that is not inside the exterior")
            holes.append(hole)

    polygon = Polygon(shell=exterior, holes=holes)
    return polygon
