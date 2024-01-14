from scipy.sparse import dok_matrix
import logging
import time
import numpy as np


weight_logger = logging.getLogger(__name__)


class Weight:

    def __init__(self, hier_arr, T = 1):
        self.hier_arr = hier_arr
        self.seg_N = hier_arr[-1]._index[-1] # last frame, end index
        self.T = T
        self.weight_matrix = dok_matrix((self.seg_N, self.seg_N), dtype=float)
        #self.compute_matrix()


    def labels(self, hier_source, hier_traget):
        pass 

    def compute_matrix(self):
        """Build the weight matrix which include all matrix connect to """

        weight_logger.info("Weight function start computing weight matrix")
        t_start = time.time()

        # this's a gloabl matrix including all candidates segementations in all frame
        # its size could be really large, use Scipy sparse matrix to save memory

        frame_num = len(self.hier_arr)

        for i in range( frame_num ):
            hier_source = self.hier_arr[i]
            for j in range(i+1, min(i+1+self.T, len(self.hier_arr))):
                hier_target = self.hier_arr[j]
                self.labels(hier_source, hier_target)

        # Ensure weight matrix is non-negative, rather then, shift weight matrix up
        min_weight = np.min(list( self.weight_matrix.values())) if len(self.weight_matrix) > 0 else 0
        if min_weight < 0:
            for key in list(self.weight_matrix.keys()):
                self.weight_matrix[key] -= min_weight

        t_used = time.time() - t_start
        weight_logger.info(f"Weight matrix computed, time consuming:{t_used} sec")

