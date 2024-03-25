
#segmentation dynamic related 
MIN_MASK_SZIE = 15

#tracking related
MERGE_COST = -0.1
DIVISION_COST = -0.1
APPEAR_COST = -0.1
DISAPPEAR_COST = -0.1

#segmentation predict related 
SEGMENTATION_PARAMS_OMNIPOSE = {
    'rescale': None, # upscale or downscale your images, None = no rescaling 
    'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
    'affinity_seg': False, # new feature, stay tuned...
    'compute_masks': False
}

SEGMENTATION_PARAMS_CELLPOSE = {
    'rescale': None, # upscale or downscale your images, None = no rescaling 
    'compute_masks': False
}

