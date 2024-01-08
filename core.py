from enum import Enum
import util
import logging
from config import SEGEMENTATION_PARAMS_OMNIPOSE, SEGEMENTATION_PARAMS_CELLPOSE

# To avoid any cyclic import, packages are import locally inside method. 


core_logger = logging.getLogger(__name__)


class ModelEnum(Enum):
    OMNIPOSE = "Omnipose"
    CELLPOSE = "Cellpose"

SEGEMENTATION_PARAMS = {
    ModelEnum.OMNIPOSE: SEGEMENTATION_PARAMS_OMNIPOSE,
    ModelEnum.CELLPOSE: SEGEMENTATION_PARAMS_CELLPOSE
}


def process(basedir, hypermodel: ModelEnum = None, chans = [0,0], submodel = None,):

    hypermodel = ModelEnum.OMNIPOSE if hypermodel is None else hypermodel

    if hypermodel == ModelEnum.OMNIPOSE:
        import omnipose
        from cellpose_omni import io, transforms, models, core
        from omnipose.utils import normalize99
    elif hypermodel == ModelEnum.CELLPOSE:
        import cellpose
        from cellpose import io, transforms, models, core
        from cellpose.transforms import normalize99
    else:
        raise Exception("No support on model {hypermodel}")
    
    if submodel not in models.MODEL_NAMES:
        core_logger.info(
            "Model {submodel} isn't in {hypermodel.value}'s model zoo. Use default model"
        )
        submodel = None
    
    use_GPU = core.use_gpu()
    model = models.CellposeModel(gpu=use_GPU, model_type = submodel)
    
    imags = util.load(basedir, io)
    params = SEGEMENTATION_PARAMS[hypermodel]

    # segementation model predict field (distance field + flow field), but does not compute mask
    params['compute_masks'] = False
    params['channels'] = chans
    _, flows, _ = model.eval(imags, **params)

    core_logger.info("Segementation: predicting fields finish.")

    # base on predicted field, run dynamic integration, computer segementation hierarchy
    hier_arr = []
    for flow in flows:
        hier_arr.append(compute_masks(flow))

    core_logger.info("Segementation hierarchy builded.")

    total_num = run_tracking(hier_arr)
    return hier_arr,total_num


def run_tracking(hier_arr):

    from hierarchy import Hierarchy
    total_num = Hierarchy.label_hierarchy_array(hier_arr)
    Hierarchy.compute_segementation_metrics(hier_arr)

    return total_num




def compute_masks(flow):

    from segementation import computer_hierarchy

    [RGB_dP, dP, cellprob, p, bd, tr, affinity, bounds] = flow
    dP, cellprob = dP.squeeze(), cellprob.squeeze()
    hier = computer_hierarchy(cellprob, dP)

    return hier







