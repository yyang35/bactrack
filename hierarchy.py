import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from collections import deque

class Node:
    def __init__(self, value, super=None):
        self.value = value
        self.super = super
        self.subs = []

        self.index = None # index corresponding to linking matrix
        self.included = False # whether include in final mask

        self.mask = None
        self.area = None
        self.centriod = None
        self.frame = None

    def add_sub(self, sub):
        """Adds a sub to this node"""
        sub.super = self
        self.subs.append(sub)

    def is_leaf(self):
        """Returns True if the node is a leaf, False otherwise"""
        return len(self.subs) == 0
    
class Hierarchy:
    def __init__(self, root):
        self.root = root
        self.value = None

    def label_nodes(self, start_index = 0):
        """Labels nodes in order"""
        self.value = start_index
        for node in self.root.subs:
            self._label_nodes_recursive(node)

    def _label_nodes_recursive(self, node):
        """Helper method to label nodes recursively"""
        if node:
            node.index = self.index
            self.value += 1
            for sub in node.subs:
                self._label_nodes_recursive(sub)
    
    def find_leaves(self):
        """Returns a list of all leaf nodes"""
        leaves = []
        self._find_leaves_recursive(self.root, leaves)
        return leaves
    
    def _find_leaves_recursive(self, node, leaves):
        """Helper method to find leaves recursively"""
        if node.is_leaf():
            leaves.append(node)
        else:
            for sub in node.subs:
                self._find_leaves_recursive(sub, leaves)

    def all_nodes(self):
        nodes = set()  # Using a set to store nodes

        queue = deque([self.root])
        while queue:
            current_node = queue.popleft()
            if current_node is not self.root:
                nodes.add(current_node)
            queue.extend(current_node.subs)

        nodes.remove(self.root)  # Remove the root node from the set
        return nodes

    @staticmethod
    def label_hierarchy_array(hier_arr):
        """Labels an array of Hierarchy instances sequentially"""
        total_index = 0
        for hierarchy in hier_arr:
            hierarchy.label_nodes(start_index = total_index)
            total_index = hierarchy.index 

        # last assigned index = total_index -1, and index start from 0
        total_num = total_index
        return total_num
    
    @staticmethod
    def compute_segementation_metrics(hier_arr, T = 1):
        """Computer each candidates segementation in hierarchy prepare for weight calculation"""
        for i in range(len(hier_arr)):

            hier = hier_arr[i]

            coords = hier.root.value
            assert len(coords.shape) == 2, "Master coordiantes of hierarchy seems get wrong shape"
            n_dim = coords.shape[2]
            assert n_dim in (2,3), "Only can handle 2D/3D cases now"

            for node in hier:
                node.area = len(sub_coords)
                node.frame = i

                # store mask of each segementation as a parse matrix
                sub_coords = coords[np.array(node.value),:]
                n_dim_coords = [sub_coords[:, n] for n in range(n_dim)]

                if n_dim == 2:
                    data = np.ones(node.area, dtype=int)
                    node.mask = coo_matrix((data, tuple(n_dim_coords)))
                else:
                    data = np.ones(node.area, dtype=int)
                    # storing each 2D slice of the 3D mask in a list
                    slices = []
                    for i in range(len(n_dim_coords[0])):
                        slice_matrix = coo_matrix((data, (n_dim_coords[1][i], n_dim_coords[2][i])))
                        slices.append(slice_matrix)
                    node.mask = slices

                centroid = [np.mean(coord) for coord in n_dim_coords]
                node.centroid = tuple(centroid)







