from scipy.sparse import csr_matrix
import logging
import time
import numpy as np

from .weight import Weight


distance_weight_logger = logging.getLogger(__name__)


class DistanceWeight(Weight):

    def __init__(self, hier_arr, T = 1):
        self.kd_forest = self._make_kd_forest(hier_arr)

    def labels(self, hier_source, hier_traget):
        pass 

    def _make_deep_masks():
        pass


