import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from collections import deque
from skimage import filters, measure


class Node:
    def __init__(self, value, super=None):
        self.value = value
        self.super = super
        self.subs = []

        self.index = None # index corresponding to linking matrix
        self.label = None # only picked segementation have label 

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

