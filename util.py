
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