import logging
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree

from .weight import Weight


distance_weight_logger = logging.getLogger(__name__)


class DistanceWeight(Weight):

    def __init__(self, hier_arr, T = 1, k = 5):

        self.hier_arr = hier_arr
        self.seg_N = hier_arr[-1]._index[-1] # last frame, end index

        self.centroids, self.kD_forest = self._make_kd_forest(hier_arr)
        self.k = k

        super().__init__(T) 


    def labels(self, hier_source, hier_traget):
        t_target = hier_traget.root.frame
        kD_tree = self.kD_forest[t_target]

        for source_node in hier_source.all_nodes():
            num_points = len(kD_tree.data)
            k_nearest = min(self.k, num_points) 
            distance, index = kD_tree.query(source_node.centroid, k=k_nearest)
            for i in range(k_nearest):
                coord = tuple(kD_tree.data[index[i]])
                source_index = source_node.index
                target_index = self.centroids[coord]
                self.weight_matrix[source_index, target_index] = distance[i]


    def _make_kd_forest(self, hier_arr):
        centroids = {}
        kD_forest = []
        for hier in hier_arr:
            i_centroids = {}
            for node in hier.all_nodes():
                i_centroids[node.centroid] = node.index
            tree = KDTree(list(i_centroids.keys()))
            kD_forest.append(tree)
            centroids.update(i_centroids)         
                   
        return centroids, kD_forest
        

