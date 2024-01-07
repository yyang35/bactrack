from hierarchy import Node
import ModelEnum
import numpy as np



""" Cost function zoo of tracking/linking algorithm"""

cost_funcs = {
    "overlap": overlap,
    "IoU": IoU,
    "distance": distance
}


def overlap(n1:Node, n2:Node):
    assert type(n1) == type(n2), "Two segementations need in same dimension"

    if isinstance(n1, list):
        overlap = sum((A.multiply(B)).sum() for A, B in zip(n1, n2))
    else: 
        overlap = (n1.multiply(n2)).sum()
    return overlap


def IoU(n1:Node, n2:Node):
    intersection = float(overlap(n1,n2))
    union = n1.area + n2.area
    return intersection / union


def distance(n1:Node, n2:Node):
    assert len(n1.centroid) == len(n2), "Two segementations need in same dimension"

    squared_diffs = [(coord_A - coord_B) ** 2 for coord_A, coord_B in zip(n1, n2)]
    distance = np.sqrt(sum(squared_diffs))
    return distance
