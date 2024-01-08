from hierarchy import Node
import numpy as np
import time


""" Cost function functions of tracking/linking algorithm"""

def overlap(nA:Node, nB:Node):
    if _check_overlap(nA.bound, nB.bound) is False:
        return 0
    
    _check_c_set(nA)
    _check_c_set(nB)

    intersection = nA.c_set.intersection(nB.c_set)
    num_common_pixels = len(intersection)

    return num_common_pixels


def IoU(nA:Node, nB:Node):
    intersection = float(overlap(nA,nB))
    union = nA.area + nB.area
    return intersection / union


def distance(nA:Node, nB:Node):
    assert len(nA.centroid) == len(nB.centroid), "Two segementations need in same dimension"

    squared_diffs = [(A - B) ** 2 for A, B in zip(nA.centroid, nB.centroid)]
    distance = sum(squared_diffs)
    return distance


cost_funcs = {
    "overlap": overlap,
    "IoU": IoU,
    "distance": distance
}


def _check_overlap(box1, box2):
    for i in range(len(box1)):
        if box1[i][1] < box2[i][0] or box1[i][0] > box2[i][1]: return False 
    return True  


def _check_c_set(n:Node):
    if n.c_set is None:
        n.c_set = {tuple(coord) for coord in n.value}
