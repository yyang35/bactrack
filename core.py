import logging
import fastremap


from skimage import measure
from skimage import filters
import numpy as np

import omnipose.core as oc
import omnipose.dynamics as od




def compute_masks(dP, dist, bd=None, niter=None, rescale=1.0, mask_threshold=0.0, diam_threshold=12., flow_threshold=0.4,
                  interp=True,  do_3D=False, hole_size=None, 
                  calc_trace=False, verbose=False, use_gpu=False, device=None, nclasses=2,
                    eps=None, hdbscan=False,debug=False):
    """
    Compute masks using dynamics from dP, dist, and boundary outputs.
    Called in cellpose.models(). 
    
    Parameters
    -------------
    dP: float, ND array
        flow field components (2D: 2 x Ly x Lx, 3D: 3 x Lz x Ly x Lx)
    dist: float, ND array
        distance field (Ly x Lx)
    bd: float, ND array
        boundary field
    niter: int32
        number of iterations of dynamics to run
    rescale: float (optional, default None)
        resize factor for each image, if None, set to 1.0   
    mask_threshold: float 
        all pixels with value above threshold kept for masks, decrease to find more and larger masks 
    flow_threshold: float 
        flow error threshold (all cells with errors below threshold are kept) (not used for Cellpose3D)
    interp: bool 
        interpolate during dynamics
    do_3D: bool (optional, default False)
        set to True to run 3D segmentation on 4D image input
    calc_trace: bool 
        calculate pixel traces and return as part of the flow
    verbose: bool 
        turn on additional output to logs for debugging 
    use_gpu: bool
        use GPU of flow_threshold>0 (computes flows from predicted masks on GPU)
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    nclasses:
        number of output classes of the network (Omnipose=3,Cellpose=2)
    eps: float
        internal epsilon parameter for (H)DBSCAN
    hdbscan: 
        use better, but much SLOWER, hdbscan clustering algorithm (experimental)
    flow_factor:
        multiple to increase flow magnitude (used in 3D only, experimental)
    debug:
        option to return list of unique mask labels as a fourth output (for debugging only)

    Returns
    -------------
    mask: int, ND array
        label matrix
    p: float32, ND array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. 
    tr: float32, ND array
        intermediate locations of each pixel during dynamics,
        size [axis x niter x Ly x Lx] or [axis x niter x Lz x Ly x Lx]. 
        For debugging/paper figures, very slow. 
    
    """

    dim = 3 if do_3D else 2
    hole_size = 3 ** (dim // 2) if hole_size is None else hole_size
    labels = None

    # do semantic segementation
    iscell = filters.apply_hysteresis_threshold(dist, mask_threshold - 1,  mask_threshold) 
    if not np.any(iscell):
        logging.info('No cell pixels found.')
        ret = [iscell, np.zeros([2, 1, 1]), [], iscell]
        return (*ret,)
    coords = np.array(np.nonzero(iscell)).astype(np.int32)
    shape = iscell.shape

    # normalize the flow magnitude to rescaled 0-1 divergence. 
    dP_ = oc.div_rescale(dP, iscell) / rescale  

    # do instance segmentation by the ol' Euler-integration + clustering
    p, coords, tr = follow_flows(dP_, dist, coords, niter=niter, interp=interp,
                                    use_gpu=use_gpu, device=device,
                                    calc_trace=calc_trace, verbose=verbose)
    # calculate masks, omnipose.core.get_masks
    masks, _ = od.get_masks(p, bd, dist, iscell, coords, nclasses, diam_threshold=diam_threshold, 
                            verbose=verbose,eps=eps, hdbscan=hdbscan) 
    coords = np.nonzero(labels)
    
    fastremap.renumber(masks, in_place=True)  # convenient to guarantee non-skipped labels

    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    ret = [masks, p, tr, bounds]

    return (*ret,)



def follow_flows(dP, dist, inds, niter=None, interp=True, use_gpu=True,
                 device=None,  calc_trace=False, verbose=False):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------
    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    inds: int, ND array 
        initial indices of pixels for the Euler integration 
    niter: int 
        number of iterations of dynamics to run
    interp: bool 
        interpolate during dynamics 
    use_gpu: bool 
        use GPU to run interpolated dynamics (faster than CPU)   
    omni: bool 
        flag to enable Omnipose suppressed Euler integration etc. 
    calc_trace: bool 
        flag to store and retrun all pixel coordinates during Euler integration (slow)

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics
    inds: int, ND array
        initial indices of pixels for the Euler integration [npixels x ndim]
    tr: float32, ND array
        list of intermediate pixel coordinates for each step of the Euler integration

    """

    assert inds.ndim in (2,3), "initial indices should in 2D/3D coordiante"
    assert inds.ndim == dP.shape, "dP shape should consist with inds(initial indices)"

    d = dP.shape[0]  # dimension is the number of flow components
    shape = np.array(dP.shape[1:]).astype(np.int32)  # shape of masks is the shape of the component field
    niter = np.uint32(niter)
    grid = [np.arange(shape[i]) for i in range(d)]
    p = np.meshgrid(*grid, indexing='ij')

    # not sure why, but I had changed this to float64 at some point... tests showed that map_coordinates expects float32
    # possible issues elsewhere? 
    p = np.array(p).astype(np.float32)
    # added inds for debugging while preserving backwards compatibility 

    cell_px = (Ellipsis,) + tuple(inds)


    if d == 2:
        p, tr = od.steps2D(p, dP.astype(np.float32), inds, niter, suppress=suppress, calc_trace=calc_trace)
    elif d == 3:
        p, tr = od.steps3D(p, dP, inds, niter)

  
    return p, inds, tr

