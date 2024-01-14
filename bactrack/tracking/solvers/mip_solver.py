import mip
import logging
import numpy as np
import time

from bactrack import config
from .solver import Solver

MIP_solver_logger = logging.getLogger(__name__)

class MIPSolver(Solver):
    def __init__(self, weight_matrix, hier_arr, do_filter = False):
        self.hier_arr = hier_arr
        self.weight_matrix = self.weight_matrix
        self.seg_N = hier_arr[-1]._index[-1]
        self.model =  mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GUROBI)
        self.do_filter = do_filter
        self._build_mip()
        

    def solve(self):
        
        self.model.optimize()

        nodes = self.model.var_by_name('nodes')
        edges = self.model.var_by_name('edges')
        n = np.asarray([i for i, node in enumerate(nodes) if node.x > 0.5], dtype=int)
        e = np.asarray([i for i, edge in enumerate(edges) if edge.x > 0.5], dtype=int)

        return n, e


    def _build_mip(self):
        MIP_solver_logger.info("Setting up MIP problem")
        t1 = time.time()

        self._basic_mip()
        self._add_hierarchy_conflict()
        if self.do_filter: 
            self._add_masks_cost()

        t_used = time.time() - t1
        MIP_solver_logger.info(f"MIP problem set up: time used {t_used} sec")
 

    def _add_hierarchy_conflict(self, hier_arr):
        limits = set()
        nodes = self.model.var_by_name('nodes')
        for hier in hier_arr:
            for node in hier.all_nodes():
                for sub in node.subs:
                    limits.add((node.index, sub.index))
                    self.model.add_constr(nodes[node.index] + nodes[sub.index] <= 1)


    def _add_masks_cost(self):
        nodes = self.model.var_by_name('nodes')
        mask_costs = np.zeros(self.seg_N)
        for hier in self.hier_arr:
            for node in hier.all_nodes():
                mask_costs[node.index] = node.uncertainity
        self.model.objective -= mip.xsum(mask_costs * nodes)


    def _basic_mip(self):
       
        frames = np.array([node.frame for hier in self.hier_arr for node in hier.all_nodes()])
        not_start = frames != 0
        not_end = frames != np.max(frames)

        # set objective
        nodes = self.model.add_var_tensor((self.seg_N,), name="nodes", var_type=mip.BINARY)
        appearances = self.model.add_var_tensor((self.seg_N,), name="appear", var_type=mip.BINARY)
        disappearances = self.model.add_var_tensor((self.seg_N,), name="disappear", var_type=mip.BINARY )
        divisions = self.model.add_var_tensor((self.seg_N,), name="division", var_type=mip.BINARY)
        edges = self.model.add_var_tensor((self.weight_matrix.count_nonzero(),), name="edges", var_type=mip.BINARY)

        self.model.objective = (
            mip.xsum(divisions * config.DIVISION_COST)
            + mip.xsum( (appearances * not_start) * config.APPEAR_COST)
            + mip.xsum( (disappearances * not_end) * config.DISAPPEAR_COST)
            + mip.xsum( self.weight_matrix.data * edges)
        )
        
        # set constrain 
        rows, cols = self.weight_matrix.nonzero() 
        for i in range(self.seg_N):
            target_indices = np.where(rows == i)[0]
            source_indices = np.where(cols == i)[0]
            # single incoming node
            self.model.add_constr(mip.xsum(edges[source_indices]) + appearances[i] == nodes[i])
            # flow conservation
            self.model.add_constr(nodes[i] + divisions[i] == mip.xsum(edges[target_indices]) + disappearances[i])
            # check this 
            self.model.add_constr(nodes[i] >= divisions[i])

