import mip
import config
import logging
import numpy as np
from scipy.sparse import csr_matrix

#from weight import cost_funcs



tracking_logger = logging.getLogger(__name__)


def solve(hier_arr, seg_num, cost_func_name = "overlap", do_filter = False):
    tracking_logger.info("Start building up problem")
    weights = make_weight_matrix(hier_arr, seg_num, cost_func_name = cost_func_name)
    hier_limits = hierarchy_limits(hier_arr)
    mask_costs = mask_cost(hier_arr, seg_num)  if do_filter else None
    not_start,not_end = make_end_exception(hier_arr, seg_num)

    tracking_logger.info("Start solving problem")

    nodes, edges = run_mip_solver(hier_arr, seg_num, weights, hier_limits, not_start, not_end, mask_costs)
    n = np.asarray([i for i, node in enumerate(nodes) if node.x > 0.5], dtype=int)
    e = np.asarray([i for i, edge in enumerate(edges) if edge.x > 0.5], dtype=int)

    return n, e, weights.nonzero() 


def hierarchy_limits(hier_arr):
    limits = set()
    for hier in hier_arr:
        for node in hier.all_nodes():
            for sub in node.subs:
                limits.add((node.index, sub.index))
    return limits

    
def make_weight_matrix(hier_arr, seg_num, T = 2, cost_func_name = "overlap"):
    """Build the weight matrix which include all matrix connect to """
    assert cost_func_name in cost_funcs, "There no such cost function named {cost_func_name}"
    cost = cost_funcs[cost_func_name]
    # this's a gloabl matrix including all candidates segementations in all frame
    # its size could be really large, use Scipy sparse matrix to save memory
    weight_matrix = csr_matrix((seg_num, seg_num), dtype=float)

    for i in range(len(hier_arr)):
        hier_source = hier_arr[i]
        for j in range(i+1, min(i+T, len(hier_arr))):
            hier_target = hier_arr[j]
            for node_source in hier_source.all_nodes():
                for node_target in hier_target.all_nodes():
                    assert node_source.frame < node_target.frame
                    c =  cost(node_source,node_target)
                    if c > 0:
                        weight_matrix[node_source.index,node_target.index] = c
    tracking_logger.info(len(weight_matrix.data))
    min_cost = np.min(weight_matrix.data)
    if min_cost < 0:
          weight_matrix.data += min_cost

    return weight_matrix


def mask_cost(hier_arr, seg_num):
    mask_costs = np.zeros(seg_num)
    for hier in hier_arr:
        for node in hier.all_nodes():
            mask_costs[node.index] = node.cost
    print(mask_costs)
    return mask_costs


def make_end_exception(hier_arr, seg_num):
    frames = np.array([node.frame for hier in hier_arr for node in hier.all_nodes()])
    return frames == 0, frames == np.max(frames)



def run_mip_solver(hier_arr, seg_num, weights, hier_limits, not_start, not_end,  mask_costs):
    model = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GUROBI)

    # set objective
    nodes = model.add_var_tensor((seg_num,), name="node", var_type=mip.BINARY)
    appearances = model.add_var_tensor((seg_num,), name="appear", var_type=mip.BINARY)
    disappearances = model.add_var_tensor((seg_num,), name="disappear", var_type=mip.BINARY )
    divisions = model.add_var_tensor((seg_num,), name="division", var_type=mip.BINARY)
    edges = model.add_var_tensor((len(weights.data),), name="edges", var_type=mip.BINARY)

    model.objective = (
        mip.xsum(divisions * config.DIVISION_COST)
        + mip.xsum( (appearances * not_start) * config.APPEAR_COST)
        + mip.xsum( (disappearances * not_end) * config.DISAPPEAR_COST)
        + mip.xsum(weights.data * edges)
    )
    
    if mask_costs is not None:
        model.objective -= mip.xsum(mask_costs * nodes)

    # set constrain 
    rows, cols = weights.nonzero() 
    for i in range(seg_num):
        target_indices = np.where(rows == i)[0]
        source_indices = np.where(cols == i)[0]
        # single incoming node
        model.add_constr(mip.xsum(edges[source_indices]) + appearances[i] == nodes[i])
        # flow conservation
        model.add_constr(nodes[i] + divisions[i] == mip.xsum(edges[target_indices]) + disappearances[i])
        # check this 
        model.add_constr(nodes[i] >= divisions[i])

    for limit in hier_limits:
        super, sub = limit
        model.add_constr(nodes[super] + nodes[sub] <= 1)


    tracking_logger.info("MIP problem be set up, start solving")
    model.optimize()

    return nodes, edges