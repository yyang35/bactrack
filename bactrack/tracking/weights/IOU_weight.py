from scipy.sparse import csr_matrix
import logging
import time
import numpy as np

from .weight import Weight

IOU_weight_logger = logging.getLogger(__name__)


class IOUWeight(Weight):

    def __init__(self, hier_arr, T = 1):

        self.hier_arr = hier_arr
        self.seg_N = hier_arr[-1]._index[-1] # last frame, end index

        self.masks, self.supers = self._make_leaves_masks(hier_arr)
        self.areas = self._all_areas()
        super().__init__(T) 
        

    def labels(self, hier_source, hier_traget):

        t_target = hier_traget.root.frame
        mask_target = self.masks[t_target]

        for node_source in hier_source.all_nodes():
            crop_mask =  mask_target[node_source.value[:,0],node_source.value[:,1]]
            index, counts = np.unique(crop_mask, return_counts=True)
            stats = dict(zip(index, counts))
            # label = -1 for background, this is not in segmentation. 
            stats.pop(-1, None)

            overlapped_leaves = set(stats.keys()).intersection(self.supers.keys())

            for l in overlapped_leaves:
                for super in self.supers[l]:
                    if super not in stats:
                        stats[super] = stats[l]
                    else:
                        stats[super] += stats[l]

            for target, overlap in stats.items():
                IoU = overlap * 1.0 / (self.areas[target] + node_source.area - overlap) 
                self.weight_matrix[node_source.index, target] = IoU


    def _make_leaves_masks(self, hier_arr):
        masks = []
        supers = {}
        for hier in hier_arr:
            shape = hier.root.shape
            # build a background canvas with value = -1
            mask = -1 * np.ones(shape, dtype=object)
            for node in hier.all_leaves():
                mask[node.value[:, 0], node.value[:, 1]] = node.index
                if (node.super is not None) and (node.super.index != -1):
                    supers[node.index] = node.all_supers()
            masks.append(mask)
        return masks, supers
    
    
    def _all_areas(self):
        areas = np.zeros(self.seg_N)
        for hier in self.hier_arr:
            for node in hier.all_nodes():
                areas[node.index] = node.area
        return areas
