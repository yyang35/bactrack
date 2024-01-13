from scipy.sparse import csr_matrix
import logging
import time
import numpy as np


weight_logger = logging.getLogger(__name__)


class Weight:

    def __init__(self, hier_arr, seg_N,  T, ):
        self.hier_arr = hier_arr
        self.seg_N = hier_arr[-1]._index
        self.T = T

    def labels(self, hier_source, hier_traget):
        pass 

    def compute_matrix(self):
        """Build the weight matrix which include all matrix connect to """

        weight_logger.info("Weight function start computing weight matrix")
        t_start = time.time()

        # this's a gloabl matrix including all candidates segementations in all frame
        # its size could be really large, use Scipy sparse matrix to save memory
        weight_matrix = csr_matrix((self.seg_N, self.seg_N), dtype=float)

        frame_num = len(self.hier_arr)

        for i in range( frame_num ):
            hier_source = self.hier_arr[i]
            for j in range(i+1, min(i+self.T, len(self.hier_arr))):
                hier_target = self.hier_arr[j]
                self.labels(hier_source, hier_target)

        # Ensure weight matrix is non-negative, rather then, shift weight matrix up
        min_weight = np.min(weight_matrix.data)
        if min_weight < 0:
            weight_matrix.data += min_weight


        t_used = time.time() - t_start
        weight_logger.info("Weight matrix computed, time consuming: {t_used}")

        return weight_matrix
