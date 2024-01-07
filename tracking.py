import mip
import config
import logging
import numpy as np
from scipy.sparse import csr_matrix
from cost import cost_funcs



tracking_logger = logging.getLogger(__name__)


def solve(hier_arr, seg_num, cost_func_name = "overlap"):
    weights = make_weight_matrix(hier_arr, seg_num, cost_func_name = cost_func_name)
    run_mip_solver(seg_num, weights)
    
    
def make_weight_matrix(hier_arr, seg_num, T = 1, cost_func_name = "overlap"):
    """Build the weight matrix which include all matrix connect to """
    assert cost_func_name in cost_funcs, "There no such cost function named {cost_func_name}"
    cost = cost_funcs[cost_func_name]
    # this's a gloabl matrix including all candidates segementations in all frame
    # its size could be really large, use Scipy sparse matrix to save memory
    weight_matrix = csr_matrix((seg_num, seg_num), dtype=float)

    for i in range(len(hier_arr)):
        hier_source = hier_arr[i]
        for j in range(i+1, i+T):
            hier_target = hier_arr[j]
            for node_source in hier_source.all_nodes():
                for node_target in hier_target.all_nodes():
                    weight_matrix[node_source.index,node_target.index] = cost(node_source,node_target)

    min_cost = weight_matrix.data.min()
    if min_cost < 0:
          weight_matrix.data += min_cost

    return weight_matrix


def run_mip_solver( seg_num, weights):
    model = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)

    # set objective
    nodes = model.add_var_tensor(seg_num, name="node", var_type=mip.BINARY)
    appearances = model.add_var_tensor(seg_num, name="appear", var_type=mip.BINARY)
    disappearances = model.add_var_tensor(seg_num, name="disappear", var_type=mip.BINARY )
    divisions = model.add_var_tensor(seg_num, name="division", var_type=mip.BINARY)

    model.objective = (
        mip.xsum(divisions * config.DIVISION_COST)
        + mip.xsum(appearances * config.APPEAR_COST)
        + mip.xsum(disappearances * config.DISAPPEAR_COST)
    )
    
    edges = model.add_var_tensor(len(weights.data), name="edges", var_type=mip.BINARY)
    objective += mip.xsum(weights.data * edges)

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


    tracking_logger.info("MIP problem be set up, start solving")
    model.optimize()

    return nodes, edges


def add_overlap_constraints(self, sources: ArrayLike, targets: ArrayLike) -> None:
    """Add constraints such that `source` and `target` can't be present in the same solution.

    Parameters
    ----------
    source : ArrayLike
        Source nodes indices.
    target : ArrayLike
        Target nodes indices.
    """
    sources = self._forward_map[np.asarray(sources, dtype=int)]
    targets = self._forward_map[np.asarray(targets, dtype=int)]

    for i in range(len(sources)):
        self._model.add_constr(
            self._nodes[sources[i]] + self._nodes[targets[i]] <= 1
        )

def enforce_node_to_solution(self, indices: ArrayLike) -> None:
    """Constraints given nodes' variables to 1.

    Parameters
    ----------
    indices : ArrayLike
        Nodes indices.
    """
    indices = self._forward_map[np.asarray(indices, dtype=int)]
    for i in indices:
        self._model.add_constr(self._nodes[i] >= 1)

def set_nodes_sum(self, indices: ArrayLike, total_sum: int) -> None:
    """Set indices sum to total_sum as constraint.

    sum_i nodes[i] = total_sum

    Parameters
    ----------
    indices : ArrayLike
        Nodes indices.
    total_sum : int
        Total sum of nodes' variables.
    """
    indices = self._forward_map[np.asarray(indices, dtype=int)]
    self._model.add_constr(mip.xsum([self._nodes[i] for i in indices]) == total_sum)
