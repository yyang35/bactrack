import numpy as np
from skimage import filters
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import omnipose.core as oc
import cellpose_omni.dynamics as od

from omnipose.utils import torch_GPU, torch_CPU, ARM
import matplotlib.pyplot as plt



def get_niter_range():
    return [5,15]


def computer_hierarchy(cellprob,dP):

    mask_threshold  = 0 
    device = torch_CPU

    iscell = filters.apply_hysteresis_threshold(cellprob, mask_threshold - 1,  mask_threshold) 
    coords = np.array(np.nonzero(iscell)).T.astype(np.int32)
    shape = np.array(dP.shape[1:]).astype(np.int32)
    cell_px = tuple(coords.T)
    niter = get_niter_range()

    # Normalize DP by rescale
    dP_ = oc.div_rescale(dP, iscell) / 1.0

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    p = np.array(p).astype(np.float32)
    p = p[:,coords[:,0], coords[:,1]]

    p_torch, dP_torch = _to_torch(p, dP_, device)
    p_norm_torch, dP_norm_torch = _normalize(p_torch, dP_torch, shape) 

    for t in range(niter[1]):
        current_coords = step(p_norm_torch, dP_norm_torch, shape)
        mask = make_mask(current_coords, cell_px, shape)


def step( pt, dP, shape):
    #calculate the position shift by following flow
    dPt = torch.nn.functional.grid_sample(dP, pt, mode = "nearest", align_corners=False)
    #add shift on original location, clamp outsider back to [-1,1]
    for k in range(len(shape)):
        pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] + dPt[:,k,:,:], -1., 1.)

    return _denormalize(pt,shape).squeeze().numpy()


def _normalize(pt, dP, shape):
    shape =  np.array(shape)[[1,0]].astype('float')-1
    pt_ = pt.clone()
    dP_ = dP.clone()
    for k in range(len(shape)): 
        dP_[:,k,:,:] *= 2./shape[k]
        pt_[:,:,:,k] /= shape[k]
    pt_ = pt_*2-1 
    return pt_, dP_


def _denormalize(pt, shape):
    shape = np.array(shape)[[1,0]].astype('float')-1
    pt_ = pt.clone()
    pt_ = (pt_+1)*0.5
    for k in range(len(shape)): 
        pt_[:,:,:,k] *= shape[k] 
    return pt_

    
def _to_torch(p,dP,device):
    # p: (n_points, 2) to pt: (1 1 n_points 2)
    p_torch = torch.from_numpy(p[[1,0]].T).float().to(device).unsqueeze(0).unsqueeze(0) 
    # (2, H, W) to (1, 2, H, W)
    dP_torch = torch.from_numpy(dP[[1,0]]).float().to(device).unsqueeze(0) 
    return p_torch, dP_torch


def make_mask(coords, cell_px, shape):

    dbscan = DBSCAN(eps=2**0.5, min_samples=5) 
    db= dbscan.fit(coords)
    labels = db.labels_
    labels = snap(coords, labels)
    mask = np.zeros(shape)
    mask[cell_px] = labels+1

    return mask 


def snap(coords, labels):
    # snapping outliers to nearest cluster 
    nearest_neighbors = NearestNeighbors(n_neighbors=50)
    neighbors = nearest_neighbors.fit(coords)
    o_inds = np.where(labels == -1)[0]
    if len(o_inds) > 1:
        outliers = [coords[i] for i in o_inds]
        distances, indices = neighbors.kneighbors(outliers)
        ns = labels[indices]
        l = [n[np.where(n != -1)[0][0] if np.any(n != -1) else 0] for n in ns]
        labels[o_inds] = l

    return labels