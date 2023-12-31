from skimage import measure
from skimage import filters
import numpy as np
import omnipose.core as oc


"""
    # for boundary later, also for affinity_seg option
    # steps = utils.get_steps(dim) # perhaps should factor this out of the function 
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)

    #bounds = find_boundaries(masks, mode='inner', connectivity=dim)
    #    # If using default omnipose/cellpose for getting masks, still try to get accurate boundaries 

    if bounds is None:
        if verbose:
            print('Default clustering on, finding boundaries via affinity.')
        affinity_graph, neighbors, neigh_inds, bounds = _get_affinity(steps, masks, dP_pad, dt_pad, p, inds,
                                                                        pad=pad)

        # boundary finder gets rid of some edge pixels, remove these from the mask 
        gone = neigh_inds[3 ** dim // 2, np.sum(affinity_graph, axis=0) == 0]
        # coords = np.argwhere(masks)
        crd = coords.T
        masks[tuple(crd[gone].T)] = 0
        iscell_pad[tuple(crd[gone].T)] = 0
    else:
        # ensure that the boundaries are consistent with mask cleanup
        # only small masks would be deleted here, no changes otherwise to boundaries 
        bounds *= masks > 0
        
"""


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
    iscell = filters.apply_hysteresis_threshold(dist, mask_threshold - 1,  mask_threshold) 

    if not np.any(iscell):
        omnipose_logger.info('No cell pixels found.')
        ret = [iscell, np.zeros([2, 1, 1]), [], iscell]
        return (*ret,)
    
    coords = np.array(np.nonzero(iscell)).astype(np.int32)
    shape = iscell.shape

    dP_ = oc.div_rescale(dP, iscell) / rescale  


    # do the ol' Euler-integration + clustering
    p, coords, tr = follow_flows(dP_, dist, coords, niter=niter, interp=interp,
                                    use_gpu=use_gpu, device=device,
                                    calc_trace=calc_trace, verbose=verbose)
    # calculate masks, omnipose.core.get_masks
    labels, _ = get_masks(p, bd, dist, iscell, coords, nclasses, diam_threshold=diam_threshold, 
                            verbose=verbose,eps=eps, hdbscan=hdbscan) 
    coords = np.nonzero(labels)
        
    
    fastremap.renumber(masks, in_place=True)  # convenient to guarantee non-skipped labels

    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 

    ret = [masks, p, tr, bounds]

    if debug:
        ret += [labels]  # also return the version of labels are prior to filling holes etc.

    if verbose:
        executionTime0 = (time.time() - startTime0)
        omnipose_logger.info('compute_masks() execution time: {:.3g} sec'.format(executionTime0))
        omnipose_logger.info('\texecution time per pixel: {:.6g} sec/px'.format(executionTime0 / np.prod(labels.shape)))
        omnipose_logger.info('\texecution time per cell pixel: {:.6g} sec/px'.format(
            np.nan if not np.count_nonzero(labels) else executionTime0 / np.count_nonzero(labels)))

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
    d = dP.shape[0]  # dimension is the number of flow components
    shape = np.array(dP.shape[1:]).astype(np.int32)  # shape of masks is the shape of the component field

    if verbose:
        omnipose_logger.info('niter is {}'.format(niter))

    niter = np.uint32(niter)

    grid = [np.arange(shape[i]) for i in range(d)]
    p = np.meshgrid(*grid, indexing='ij')
    # not sure why, but I had changed this to float64 at some point... tests showed that map_coordinates expects float32
    # possible issues elsewhere? 
    p = np.array(p).astype(np.float32)
    # added inds for debugging while preserving backwards compatibility 

    if inds.ndim < 2 or inds.shape[0] < d:
        omnipose_logger.warning('WARNING: no mask pixels found')
        return p, inds, None

    cell_px = (Ellipsis,) + tuple(inds)

    if not interp:
        omnipose_logger.warning('not interp')
        if d == 2:
            p, tr = steps2D(p, dP.astype(np.float32), inds, niter,
                            suppress=suppress,  # no momentum term here, suppress toggled by omni upstream
                            calc_trace=calc_trace)
        elif d == 3:
            p, tr = steps3D(p, dP, inds, niter)
        else:
            omnipose_logger.warning('No non-interp code available for non-2D or -3D inputs.')

    else:
        # I am not sure why we still use p[cell_px]... instead of just cell_px. 
        p_interp, tr = steps_interp(p[cell_px], dP, dist, niter, use_gpu=use_gpu,
                                    device=device,
                                    calc_trace=calc_trace,
                                    verbose=verbose)
        p[cell_px] = p_interp
    return p, inds, tr




def steps_interp(p, dP, dist, niter, use_gpu=True, device=None, omni=True, suppress=True,
                 calc_trace=False, calc_bd=False, verbose=False):
    """Euler integration of pixel locations p subject to flow dP for niter steps in N dimensions. 
    
    Parameters
    ----------------
    p: float32, ND array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)
    dP: float32, ND array
        flows [axis x Lz x Ly x Lx]
    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics

    """
    niter = 40
    align_corners = True
    # I think bilinear is actually a problem, as averaging to zero causes stranded pixels 
    # nearest is also faster and does just as well 
    # However, I will keep it as the default for the old omnipose version or cellpose 
    mode = 'nearest' if (omni and not suppress) else 'bilinear'

    d = dP.shape[0]  # number of components = number of dimensions
    shape = dP.shape[1:]  # shape of component array is the shape of the ambient volume
    inds = list(range(d))[::-1]  # grid_sample requires a particular ordering

    print('d', d)
    print('inds_main', inds, p.shape)

    # if verbose:
    #     startTime = time.time()
    #     print('device',device)

    if device is None:
        if use_gpu:
            device = torch_GPU
        else:
            device = torch_CPU
    # for now, looks like grid_sampler_2d is not implemented for mps
    # so it is much faster to just default to CPU instead of allowing for fallback
    # but now that I am realizing that inteprolation is no good... maybe there is a better way 
    # yes, I should use my torchvf code <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if ARM:
        device = torch_CPU
    shape = np.array(shape)[inds] - 1.  # dP is d.Ly.Lx, inds flips this to flipped X-1, Y-1, ...

    # for grid_sample to work, we need im,pt to be (N,C,H,W),(N,H,W,2) or (N,C,D,H,W),(N,D,H,W,3). The 'image' getting interpolated
    # is the flow, which has d=2 channels in 2D and 3 in 3D (d vector components). Output has shape (N,C,H,W) or (N,C,D,H,W)
    pt = torch.tensor(p[inds].T, device=device)

    print('pt_main', pt.shape)

    pt0 = pt.clone()  # save first
    for k in range(d):
        pt = pt.unsqueeze(0)  # get it in the right shape

    print('p_main', p.shape, p.min(), p.max(), pt.shape)
    print("n_iter", niter)

    if isinstance(dP, torch.Tensor):
        flow = dP[inds].to(device).unsqueeze(0)
    else:
        flow = torch.tensor(dP[inds], device=device).unsqueeze(
            0)  # covert flow numpy array to tensor on GPU, add dimension

    print('flow_main', flow.shape)
    # we want to normalize the coordinates between 0 and 1. To do this, 
    # we divide the coordinates by the shape along that dimension. To symmetrize,
    # we then multiply by 2 and subtract 1. I
    # We also need to rescale the flow by the same factor, but no shift of -1. 

    for k in range(d):
        pt[..., k] = 2 * pt[..., k] / shape[k] - 1
        flow[:, k] = 2 * flow[:, k] / shape[k]

    # make an array to track the trajectories 
    if calc_trace:
        trace = torch.clone(pt).detach()
        # trace = torch.zeros((niter,)+pt.shape) # slower to preallocate...

    # init 
    if omni and OMNI_INSTALLED and suppress:
        dPt0 = torch.nn.functional.grid_sample(flow, pt, mode=mode, align_corners=align_corners)

    import matplotlib.pyplot as plt
    # here is where the stepping happens
    for t in range(niter):
        if calc_trace and t > 0:
            trace = torch.cat((trace, pt))
            # trace[t] = pt.detach()
        # align_corners default is False, just added to suppress warning
        dPt = torch.nn.functional.grid_sample(flow, pt, mode=mode,
                                              align_corners=align_corners)  # see how nearest changes things

        if omni and OMNI_INSTALLED and suppress:
            dPt = (dPt * 0.8 + dPt0 * 0.2)   # average with previous flow
            # dPt0 = dPt.clone() # update old flow
            dPt0.copy_(dPt)  # update old flow
            dPt /= step_factor(t)  # suppression factor


        for k in range(d):  # clamp the final pixel locations, <<<< could be done for all at once?
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1., 1.)


    # undo the normalization from before, reverse order of operations
    pt = (pt + 1) * 0.5
    for k in range(d):
        pt[..., k] *= shape[k]

    if calc_trace:
        trace = (trace + 1) * 0.5
        for k in range(d):
            trace[..., k] *= shape[k]

    # pass back to cpu
    if calc_trace:
        tr = trace[..., inds].cpu().numpy().squeeze().T
    else:
        tr = None

    p = pt[..., inds].cpu().numpy().squeeze().T

    empty_cache()  # release memory
    return p, tr
