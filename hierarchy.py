import pandas as pd

class Node:
    def __init__(self, value, super=None):
        self.value = value
        self.super = super
        self.subs = []

        self.index = None # index corresponding to linking matrix
        self.included = False # whether include in final mask

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

    @staticmethod
    def label_hierarchy_array(hier_arr):
        """Labels an array of Hierarchy instances sequentially"""
        total_index = 0
        for hierarchy in hier_arr:
            hierarchy.label_nodes(start_index = total_index)
            total_index = hierarchy.index 

        # the last index didn't assign to any segementation node, and index start from 0
        # so it's also the total number segementation node across T frame. 
        total_num = total_index
        return total_num