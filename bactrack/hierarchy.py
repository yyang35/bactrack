import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from collections import deque
from skimage import filters, measure


class Node:
    def __init__(self, value, **kwargs):
        self.value = value
        self.super = kwargs.get('super', None)
        self.subs = []

        self.index = kwargs.get('index', None) # index corresponding to linking matrix 
        
        self.shape = kwargs.get('shape', None)
        self.uncertainty = kwargs.get('uncertainty', None)
        self.area = kwargs.get('area', None)
        self.centroid = kwargs.get('centroid', None)
        self.frame = kwargs.get('frame', None)
        self.bound = kwargs.get('bound', None)

        self.label = kwargs.get('label', None)  # only picked segementation have label 
        self.next = kwargs.get('next', None) # next frame node, only picked sgementation have next

 
    def add_sub(self, sub):
        """Adds a sub to this node"""
        sub.super = self
        self.subs.append(sub)

    def is_leaf(self):
        """Returns True if the node is a leaf, False otherwise"""
        return len(self.subs) == 0
    
    def to_dict(self):
        """Converts the Node and its hierarchy into a dictionary"""
        ROOT_PLACEHOLDER = "ROOT"
        node_dict = {attr: getattr(self, attr.lower(), None) for attr in \
                     ['Value', 'Next','Shape', 'Cost', 'Area', 'Centroid', 'Frame', 'Bound','Index', 'Label',]}
        node_dict['Super'] = self.super.index if self.super is not None else ROOT_PLACEHOLDER
        return node_dict


class Hierarchy:
    def __init__(self, root):
        self.root = root
        self._index = None

    def label_nodes(self, start_index = 0):
        """Labels nodes in order"""
        self._index = start_index
        for node in self.root.subs:
            self._label_nodes_recursive(node)

    def _label_nodes_recursive(self, node):
        """Helper method to label nodes recursively"""
        if node:
            node.index = self._index
            self._index += 1
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

    def all_nodes(self, include_root = False):
        nodes = set()  # Using a set to store nodes

        queue = deque([self.root])
        while queue:
            current_node = queue.popleft()
            nodes.add(current_node)
            queue.extend(current_node.subs)
            
        if not include_root:
            nodes.remove(self.root)  # Remove the root node from the set
        return nodes
    
    def to_df(self):
        """Converts the entire hierarchy into a pandas DataFrame."""
        node_dicts = [node.to_dict() for node in self.all_nodes(include_root=True)]
        df = pd.DataFrame(node_dicts)
        df['Index'] = df['Index'].astype(int)
        return df

    @staticmethod
    def read_df(df):
        node_dict = {}
        root_node = None

        # Create all nodes without setting up the hierarchy
        for _, row in df.iterrows():
            kwargs = row.to_dict()
            node = Node(kwargs.pop('Value'), **kwargs)
            node_dict[node.index] = node
            if row['Super'] == 'ROOT':
                root_node = node

        # Set up the hierarchy
        for _, row in df.iterrows():
            node = node_dict[row['Index']]
            if row['Super'] != 'ROOT':
                super_node = node_dict[row['Super']]
                super_node.add_sub(node)

        return Hierarchy(root_node)