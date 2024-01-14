from scipy.sparse import csr_matrix
import logging
import time
import numpy as np

from .weight import Weight

overlap_weight_logger = logging.getLogger(__name__)


class OverlapWeight(Weight):

    def __init__(self, hier_arr, T = 1):
        self.masks, self.supers = self._make_masks(hier_arr)
        super().__init__(hier_arr, T) 
        

    def labels(self, hier_source, hier_traget):

        t_target = hier_traget.root.frame
        mask_target = self.masks[t_target]

        for node_source in hier_source.all_nodes():
            # _index = range of index under this hierarchy
            # e.g hier._index = [15, 25] 
            # indicates hier including nodes: 15, 16, .... 23,24, ([15, 25), not include 25)
            crop_mask =  mask_target[node_source.value[:,0],node_source.value[:,1]]
            index, counts = np.unique(crop_mask, return_counts=True)
            stats = dict(zip(index, counts))
            stats.pop(-1, None)

            overlapped_leaves = set(stats.keys()).intersection(self.supers.keys())

            for l in overlapped_leaves:
                for super in self.supers[l]:
                    if super not in stats:
                        stats[super] = stats[l]
                    else:
                        stats[super] += stats[l]

            for target, overlap in stats.items():
                self.weight_matrix[node_source.index, target] = overlap


    def _make_masks(self, hier_arr):
        masks = []
        supers = {}
        for hier in hier_arr:
            shape = hier.root.shape
            mask = -1 * np.ones(shape, dtype=object)
            for node in hier.all_leaves():
                mask[node.value[:, 0], node.value[:, 1]] = node.index
                if (node.super is not None) and (node.super.index != -1):
                    supers[node.index] = node.all_supers()
            masks.append(mask)
        return masks, supers


