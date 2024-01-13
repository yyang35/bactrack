from scipy.sparse import csr_matrix
import logging
import time
import numpy as np

from .weight import Weight

overlap_weight_logger = logging.getLogger(__name__)


class OverlapWeight(Weight):

    def __init__(self, hier_arr, seg_N,  T, ):
        self.deep_mask = self._make_deep_masks(hier_arr)

    def labels(self, hier_source, hier_traget):
        pass 

    def _make_deep_masks():
        pass


