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
        
        self.shape = kwargs.get('shape', None) # Canvas shape
        self.uncertainty = kwargs.get('uncertainty', None)
        self.area = kwargs.get('area', None)
        self.centroid = kwargs.get('centroid', None)
        self.frame = kwargs.get('frame', None)
        self.bound = kwargs.get('bound', None)

        self.label = kwargs.get('label', None)  # only picked segmentation have label 
        self.next = kwargs.get('next', None) # next frame node, only picked sgementation have next

 
    def add_sub(self, sub):
        """Adds a sub to this node"""
        sub.super = self
        self.subs.append(sub)

    def all_supers(self):
        n = self
        supers = []
        while n.super is not None and n.super.index != -1 :
            supers.append(n.super.index)
            n = n.super

        return supers

    def is_leaf(self):
        """Returns True if the node is a leaf, False otherwise"""
        return len(self.subs) == 0
    
    def to_dict(self):
        """Converts the Node and its hierarchy into a dictionary"""
        node_dict = {attr: getattr(self, attr.lower(), None) for attr in \
                     ['value', 'next','shape', 'uncertainty', 'area', 'centroid', 'frame', 'bound','index', 'label',]}
        node_dict['super'] = self.super.index if self.super is not None else None
        return node_dict


class Hierarchy:
    def __init__(self, root):
        self.root = root
        self._index = None

    def label_nodes(self, start_index = 0):
        """Labels nodes in order"""
        # be careful with hier._index here, so if it have node
        # [1,2,3,4,5], hier._index = [1,6],
        # _index[-1] is larger than last element in hier, this is defined for convenience of 
        # _index[-1] - _index[0] = total num of hier. 
        index = start_index
        for node in self.all_nodes():
            node.index = index
            index += 1
        self._index = (start_index, index)
        self.root.index = -1

        return self._index 
    
    def all_leaves(self):
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
        import json
        node_dicts = [node.to_dict() for node in self.all_nodes(include_root=True)]
        return pd.DataFrame(node_dicts)

    @staticmethod
    def read_df(df):

        node_dict = {}
        
        # Create all nodes without setting up the hierarchy
        for _, row in df.iterrows():
            kwargs = row.to_dict()
            node = Node(kwargs.pop('value'), **kwargs)
            node_dict[node.index] = node

        root_node = node_dict[-1]

        # Set up the hierarchy
        for _, row in df[df['super'].notna()].iterrows():
            node = node_dict[row['index']]
            super_node = node_dict[row['super']]
            super_node.add_sub(node)

        hier = Hierarchy(root_node)
        hier._index = [np.min(df['index'][df['index'] >= 0]), np.max(df['index']) + 1]

        return hier
    
    @staticmethod
    def label_hierarchy_array(hier_arr):
        """Labels an array of Hierarchy instances sequentially"""
        total_index = 0
        for hierarchy in hier_arr:
            _, total_index = hierarchy.label_nodes(start_index = total_index)

    @staticmethod
    def compute_segmentation_metrics(hier_arr):
        """Computer each candidates segmentation in hierarchy prepare for weight calculation"""
        for i in range(len(hier_arr)):

            hier = hier_arr[i]
            coords = hier.root.value

            assert len(coords.shape) == 2, "Coordiantes of hierarchy seems get wrong shape"
            n_dim = coords.shape[1]
            assert n_dim in (2,3), "Only can handle 2D/3D cases now"

            for node in hier.all_nodes(include_root = True):
                # sub_coords: [[x1,y1], [x2,y2], [x3,y3], ..., [xn,yn]]
                sub_coords =node.value
                # ndim_coords: [[x1, x2, x3, ..., xn], [y1, y2, y3, ..., yn]]
                n_dim_coords = [sub_coords[:, n] for n in range(n_dim)]
                centroid = [np.mean(coord) for coord in n_dim_coords]

                node.area = len(sub_coords)
                node.frame = i
                node.centroid = tuple(centroid)
                node.bound = np.vstack((np.min(sub_coords, axis=0), np.max(sub_coords, axis=0))).T
                
                mask  = np.zeros(node.shape)
                mask[sub_coords[:, 0], sub_coords[:, 1]] = 1
                labeled_mask, num_features = measure.label(mask, connectivity=1, return_num=True)
                
                if num_features > 1: # this mask is not connected, we only keep the largest one
                    component_sizes = [np.sum(labeled_mask == label) for label in range(1, num_features + 1)]
                    largest_component_label = np.argmax(component_sizes) + 1  # Add 1 to match label indices
                    largest_component_mask = (labeled_mask == largest_component_label)
                    largest_component_coords = np.column_stack(np.nonzero(largest_component_mask))
                    node.value = largest_component_coords

