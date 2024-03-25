from scipy.sparse import dok_matrix
import logging
import time
import numpy as np


weight_logger = logging.getLogger(__name__)


class Weight:

    def __init__(self, T = 1):
        self.T = T
        self.weight_matrix = dok_matrix((self.seg_N, self.seg_N), dtype=float)
        self.compute_matrix()
        self.mask_penalty = self.compute_mask_penalty()


    def labels(self, hier_source, hier_traget):
        pass 

    def compute_mask_penalty(self):
        mask_penalty = np.zeros(self.seg_N)
        for hier in self.hier_arr:
            for node in hier.all_nodes():
                mask_penalty[node.index] = node.uncertainty

        # Normalize mask_penalty to the range [0, 1]
        min_val = np.min(mask_penalty)
        max_val = np.max(mask_penalty)
        
        # Avoid division by zero in case all values are the same
        if max_val - min_val != 0:
            mask_penalty = (mask_penalty - min_val) / (max_val - min_val)

        return mask_penalty


    def compute_matrix(self):
        """Build the weight matrix which include all matrix connect to """

        weight_logger.info("Weight function start computing weight matrix")
        t_start = time.time()

        # this's a gloabl matrix including all candidates segmentations in all frame
        # its size could be really large, use Scipy sparse matrix to save memory

        frame_num = len(self.hier_arr)

        for i in range( frame_num ):
            hier_source = self.hier_arr[i]
            for j in range(i+1, min(i+1+self.T, len(self.hier_arr))):
                hier_target = self.hier_arr[j]
                self.labels(hier_source, hier_target)

        # Make weights non-negative
        min_weight = np.min(list(self.weight_matrix.values())) if len(self.weight_matrix) > 0 else 0
        if min_weight < 0:
            for key in list(self.weight_matrix.keys()):
                self.weight_matrix[key] -= min_weight

        # Normalize weights to 0-1 range
        max_weight = np.max(list(self.weight_matrix.values())) if len(self.weight_matrix) > 0 else 1
        for key in list(self.weight_matrix.keys()):
            self.weight_matrix[key] /= max_weight

        t_used = time.time() - t_start
        weight_logger.info(f"Weight matrix computed, time consuming:{t_used} sec")

