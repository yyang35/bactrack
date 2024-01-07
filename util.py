import time
import numpy as np

def load(basedir, io):
    return read_folder(basedir, io)

def read_folder(basedir, io):
    files = io.get_image_files(basedir)
    imgs = [io.imread(f) for f in files]

    return imgs



