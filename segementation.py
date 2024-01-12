from operator import itemgetter
import numpy as np
from skimage import filters, measure
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from omnipose.utils import torch_GPU, torch_CPU, ARM
import matplotlib.pyplot as plt

import omnipose.core as oc
import cellpose_omni.dynamics as od


from hierarchy import Node, Hierarchy

import time



def get_niter_range(cellprob, ndim, precison = 1):
    """Get the Euler integration range of segementation hierarchy"""
    
    min = 1
    # the niter omnipose used, should be the proper niter
    mid = int(2 * (ndim + 1) * np.mean(cellprob[cellprob > 0]))
    max = int(2 * (ndim + 1) * np.max(cellprob[cellprob > 0]))

    n = precison + 2
    return np.unique(np.concatenate((np.linspace(min, mid, n), np.linspace(mid, max, n)))).astype(int) 


def computer_hierarchy(cellprob,dP):
    """Master method of computer segementation hierarchy"""
    mask_threshold  = 0 
    device = torch_CPU

    iscell = filters.apply_hysteresis_threshold(cellprob, mask_threshold - 1,  mask_threshold) 
    coords = np.array(np.nonzero(iscell)).T.astype(np.int32)
    shape = np.array(dP.shape[1:]).astype(np.int32)
    cell_px = tuple(coords.T)
    niters = get_niter_range(cellprob, 2)

    # normalize DP by rescale
    dP_ = oc.div_rescale(dP, iscell) / 1.0

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    p = np.array(p).astype(np.float32)
    p = p[:,coords[:,0], coords[:,1]]

    # convert formats for step method (especially torch.nn.functional.grid_sample )
    p_torch, dP_torch = _to_torch(p, dP_, device)
    p_norm_torch, dP_norm_torch = _normalize(p_torch, dP_torch, shape) 

    hier = Hierarchy(Node(list(range(coords.shape[0])))) 
    hier.root.shape = cellprob.shape

    # iteration to computer segementation hierarchy
    # every itereation do sub-segementation inside previous segementation
    for t in range(np.max(niters) + 1):
        current_coords = step(p_norm_torch, dP_norm_torch, shape)
        if t in niters:
            hier = put_segement(current_coords, hier, remove_small_masks = True)

 
    hier.root.value = coords
    
    for node in hier.all_nodes(): 
        node.shape = cellprob.shape
        sub_coords = coords[np.array(node.value)]
        mask = np.zeros(cellprob.shape)
        mask[sub_coords[:, 0], sub_coords[:, 1]] = 1
        labeled_mask, num_features = measure.label((cellprob * mask) > 1 , connectivity=1, return_num=True)
        node.cost = 100.0 * num_features / len(node.value)
    

    return hier


def step( pt, dP, shape):
    """Single step of Euler integration of dynamics dP"""
    # calculate the position shift by following flow, func require coordinate in [-1, 1]
    dPt = torch.nn.functional.grid_sample(dP, pt, mode = "nearest", align_corners=False)
    # add shiftted displacement to original location, clamp outsider back to [-1,1]
    # pt(the normalized version coordiante) is update in func, eventhough it never be returned
    for k in range(len(shape)):
        pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] + dPt[:,k,:,:], -1., 1.)

    return _denormalize(pt,shape).squeeze().numpy()


def _normalize(pt, dP, shape):
    """Normalize grid and input to [-1,1]"""
    shape =  np.array(shape)[[1,0]].astype('float')-1
    pt_ = pt.clone()
    dP_ = dP.clone()
    for k in range(len(shape)): 
        dP_[:,k,:,:] *= 2./shape[k]
        pt_[:,:,:,k] /= shape[k]
    pt_ = pt_*2-1 
    return pt_, dP_


def _denormalize(pt, shape):
    """DeNormalize pt coordinates back to original size."""
    shape = np.array(shape)[[1,0]].astype('float')-1
    pt_ = pt.clone()
    pt_ = (pt_+1)*0.5
    # asisgn for each dimension
    for d in range(len(shape)): 
        pt_[:,:,:,d] *= shape[d] 
    return pt_


def _to_torch(p,dP,device):
    """Convert pixcels locations and flow field to required torch format """
    # shape of p: (n_points, 2) to p_torch : (1 1 n_points 2)
    p_torch = torch.from_numpy(p[[1,0]].T).float().to(device).unsqueeze(0).unsqueeze(0) 
    # shape of dP: (2, H, W) to dP_torch: (1, 2, H, W)
    dP_torch = torch.from_numpy(dP[[1,0]]).float().to(device).unsqueeze(0) 
    return p_torch, dP_torch


"""
def put_segement(coords, hier, remove_small_masks = False):
    Put current sub-segementation to the leaves of hierarchy

    MIN_MASK_SZIE = 15

    # do subsegementation under segementation hierachy leaves 
    leaves = hier.find_leaves()
    for leave in leaves:
        sub_indices = leave.value
        sub_coords = coords[sub_indices, :]

        mask = np.zeros((int(np.max(sub_coords[:, 0])) + 1, int(np.max(sub_coords[:, 1])) + 1))
        # Convert each row of sub_coords to a tuple and set the corresponding pixels to 1
        mask[tuple(sub_coords.T.astype(int))] = 1
        labeled_mask, num_features = measure.label(mask , connectivity=1, return_num=True)

        if num_features == 1 and leave != hier.root:
            continue
        
        labels = np.array([labeled_mask[int(x)][int(y)] for x, y in sub_coords])

        # convert small mask to outlier (-1), or rejected masks (-2**63 ). 
        alter_label = -2**63 if remove_small_masks else -1 
        for l in np.unique(labels):
            indices_with_label = np.where(labels == l)[0] 
            if len(indices_with_label) <  MIN_MASK_SZIE:
                labels[indices_with_label] = alter_label

        # after solve small masks, it still have more than 1 sub-segementation detected under current 
        # segementation, or this the root segementation 
        valid_labels = np.unique([label for label in labels if label >= 0])
        if len(valid_labels) > 1 or leave == hier.root: 
            
            # snap outlier to segementation
            labels = snap(sub_coords, labels)
            
            for l in valid_labels:
                indices_with_label = np.where(labels == l)[0] 
                leave.add_sub(Node(itemgetter(*indices_with_label)(sub_indices)))

    return hier
"""



def put_segement(coords, hier, remove_small_masks = False):
    # method to cluster coords: dbscan
    EPS = 2 ** 0.5
    MIN_SAMPLES = 5
    MIN_MASK_SZIE = 15
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES) 

    import time
    t1 = time.time()
    # do subsegementation under segementation hierachy leaves 
    leaves = hier.find_leaves()
    for leave in leaves:
        sub_indices = leave.value
        sub_coods = coords[sub_indices, :]

        db = dbscan.fit(sub_coods)
        labels = db.labels_

        # convert small mask to outlier (-1), or rejected masks (-2**63 ). 
        alter_label = -2**63 if remove_small_masks else -1 
        for l in np.unique(labels):
            indices_with_label = np.where(labels == l)[0] 
            if len(indices_with_label) <  MIN_MASK_SZIE:
                labels[indices_with_label] = alter_label

        # snap outlier to segementation
        labels = snap(sub_coods, labels)

        valid_labels = np.unique([label for label in labels if label >= 0])
        if len(valid_labels) > 1 or leave == hier.root: 
            # more than 1 sub-segementation detected under current segementation 
            for l in valid_labels:
                indices_with_label = np.where(labels == l)[0] 
                leave.add_sub(Node(itemgetter(*indices_with_label)(sub_indices)))

    t2 = time.time()
    print(t2-t1)

    return hier



def cluster(shape, coords):

    EPS = 2 ** 0.5
    MIN_SAMPLES = 5

    cood_num = coords.shape[0]
    labels = np.zeros(cood_num)
    mask = -1 * np.ones(shape)
    for c_index in range(cood_num):
        x, y = map(int, coords[c_index])
        if (x, y) not in mask: 
            mask[x, y] = set()
        mask[x, y].add(c_index)
    
    queue = []
    label = 0
    for c_index in range(cood_num):
        if labels[c_index] == -1 : queue.append(c_index)
        while not queue.empty():
            x, y = map(int, coords[queue.pop()])
            delta = int(1 + EPS)
            neighbors = set().union(*mask[x-delta:x+delta, y-delta:y+delta].flatten())
            close_neighbors = set()
            for coord in neighbors:
                if np.sqrt(( x- coord[0]) ** 2 + (y - coord[1]) ** 2) <= EPS: 
                    close_neighbors.add(coord)

            if close_neighbors > MIN_SAMPLES:
                queue.extend(close_neighbors)

            labels[c_index] = label

        label += 1

    return labels

        

def snap(coords, labels):
    """snapping outliers to nearest cluster"""

    n_samples = len(coords)
    n_neighbors = min(50, n_samples - 1) 

    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors = nearest_neighbors.fit(coords)
    o_inds = np.where(labels == -1)[0]
    if len(o_inds) > 0:
        outliers = [coords[i] for i in o_inds]
        distances, indices = neighbors.kneighbors(outliers)
        ns = labels[indices]
        l = [n[np.where(n != -1)[0][0] if np.any(n != -1) else 0] for n in ns]
        labels[o_inds] = l

    return labels