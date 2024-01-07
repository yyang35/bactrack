from core import ModelEnum

#segementation dynamic related 
MIN_MASK_SZIE = 15

#tracking related
DIVISION_COST = -0.001
APPEAR_COST = -0.001
DISAPPEAR_COST = -0.001

#segementation predict related 
SEGEMENTATION_PARAMS_OMNIPOSE = {
    'rescale': None, # upscale or downscale your images, None = no rescaling 
    'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
    'affinity_seg': False, # new feature, stay tuned...
    'compute_masks': False
}

SEGEMENTATION_PARAMS_CELLPOSE = {
    'rescale': None, # upscale or downscale your images, None = no rescaling 
    'compute_masks': False
}

SEGEMENTATION_PARAMS = {
    ModelEnum.OMNIPOSE: SEGEMENTATION_PARAMS_OMNIPOSE,
    ModelEnum.CELLPOSE: SEGEMENTATION_PARAMS_CELLPOSE
}