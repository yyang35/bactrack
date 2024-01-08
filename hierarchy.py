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

        self.shape = None
        self.cost = None

        self.area = None
        self.centroid = None
        self.frame = None
        self.bound = None
        self.c_set = None

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
        self.hyper_index = None

    def label_nodes(self, start_index = 0):
        """Labels nodes in order"""
        self.hyper_index = start_index
        for node in self.root.subs:
            self._label_nodes_recursive(node)

    def _label_nodes_recursive(self, node):
        """Helper method to label nodes recursively"""
        if node:
            node.index = self.hyper_index
            self.hyper_index += 1
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
            total_index = hierarchy.hyper_index 

        # last assigned index = total_index -1, and index start from 0
        total_num = total_index
        return total_num
    
    @staticmethod
    def compute_segementation_metrics(hier_arr):
        """Computer each candidates segementation in hierarchy prepare for weight calculation"""
        for i in range(len(hier_arr)):

            hier = hier_arr[i]
            coords = hier.root.value

            assert len(coords.shape) == 2, "Coordiantes of hierarchy seems get wrong shape"
            n_dim = coords.shape[1]
            assert n_dim in (2,3), "Only can handle 2D/3D cases now"

            for node in hier.all_nodes():
                sub_coords = coords[np.array(node.value)]
                n_dim_coords = [sub_coords[:, n] for n in range(n_dim)]
                centroid = [np.mean(coord) for coord in n_dim_coords]

                node.area = len(sub_coords)
                node.frame = i
                node.centroid = tuple(centroid)
                node.bound = np.vstack((np.min(sub_coords, axis=0), np.max(sub_coords, axis=0))).T
                node.value = sub_coords
