import mip
import logging
import numpy as np
import time
from scipy.sparse import dok_matrix, csr_matrix

from bactrack import config
from .solver import Solver

MIP_solver_logger = logging.getLogger(__name__)

class MIPSolver(Solver):
    def __init__(self, weight_matrix, hier_arr, mask_penalty = None):

        self.hier_arr = hier_arr
        self.weight_matrix = weight_matrix.tocsr()
        self.seg_N = hier_arr[-1]._index[-1]
        self.model =  mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GUROBI)
        self.mask_penalty = mask_penalty
        self.nodes = None
        self.edges = None
        self._build_mip()
        

    def solve(self):
        
        self.model.optimize()
        n = np.asarray([i for i, node in enumerate(self.nodes) if node.x > 0.5], dtype=int)
        e = np.asarray([i for i, edge in enumerate(self.edges) if edge.x > 0.5], dtype=int)
        
        row, col = self.weight_matrix.nonzero()
        e_matrix = dok_matrix(self.weight_matrix.shape)
        for r, c in zip(row[e], col[e]):
            e_matrix[r, c] = 1

        return n, e_matrix


    def _build_mip(self):
        MIP_solver_logger.info("Setting up MIP problem")
        t1 = time.time()

        self.nodes, self.edges = self._basic_mip()
        self._add_hierarchy_conflict()
        if self.mask_penalty is not None:
             self.model.objective -= mip.xsum(self.mask_penalty * self.nodes)

        t_used = time.time() - t1
        MIP_solver_logger.info(f"MIP problem set up: time used {t_used} sec")


    def _basic_mip(self):

        LARGE = 2 ** 32
       
        frames = np.array([node.frame for hier in self.hier_arr for node in hier.all_nodes()])
        not_start = frames != 0
        not_end = frames != np.max(frames)

        # create objects
        nodes = self.model.add_var_tensor((self.seg_N,), name="nodes", var_type=mip.INTEGER)
        edges = self.model.add_var_tensor((self.weight_matrix.count_nonzero(),), name="edges", var_type=mip.BINARY)
        appearances = self.model.add_var_tensor((self.seg_N,), name="appear", var_type=mip.BINARY)
        disappearances = self.model.add_var_tensor((self.seg_N,), name="disappear", var_type=mip.BINARY )
        divisions = self.model.add_var_tensor((self.seg_N,), name="division", var_type=mip.INTEGER)
        merges = self.model.add_var_tensor((self.seg_N,), name="merge", var_type=mip.INTEGER)

        is_nodes = self.model.add_var_tensor((self.seg_N,), name="nodes_binary", var_type=mip.BINARY)
        is_divisions = self.model.add_var_tensor((self.seg_N,), name="division_binary", var_type=mip.BINARY)
        is_merges = self.model.add_var_tensor((self.seg_N,), name="merge_binary", var_type=mip.BINARY)
        is_edges_in = self.model.add_var_tensor((self.seg_N,), name="edges_in_binary", var_type=mip.BINARY)
        is_edges_out = self.model.add_var_tensor((self.seg_N,), name="edges_out_binary", var_type=mip.BINARY)
        
        # set objective
        self.model.objective = (
            mip.xsum( self.weight_matrix.data * edges )
            # Notice all following cost are nagetive 
            + mip.xsum( divisions * config.DIVISION_COST )
            + mip.xsum( merges * config.MERGE_COST )
            + mip.xsum( (appearances * not_start) * config.APPEAR_COST)
            + mip.xsum( (disappearances * not_end) * config.DISAPPEAR_COST)
            + mip.xsum( self.mask_penalty * is_nodes )
        )
        
        # set integer to binary constrain:
        # nothing interestin below: it just makes when Integer N > 0, binary n = 1, and N = 0, n =0
        # by setting N <= M * n (where M is large number) 
        for i in range(self.seg_N):
            self.model.add_constr( nodes[i] <= is_nodes[i] * LARGE )
            self.model.add_constr( divisions[i] <= is_divisions[i] * LARGE )
            self.model.add_constr( merges[i] <= is_merges[i] * LARGE )

        # set flow constrain 
        rows, cols = self.weight_matrix.nonzero() 
        for i in range(self.seg_N):
            target_indices = np.where(rows == i)[0]
            source_indices = np.where(cols == i)[0]
            # single incoming node
            self.model.add_constr(mip.xsum(edges[source_indices]) + appearances[i] == nodes[i] + merges[i])
            # flow conservation
            self.model.add_constr(mip.xsum(edges[target_indices]) + disappearances[i] == nodes[i] + divisions[i])

            # set binary constrain:
            self.model.add_constr( mip.xsum(edges[source_indices]) <= is_edges_in[i] * LARGE)
            self.model.add_constr( mip.xsum(edges[target_indices]) <= is_edges_out[i] * LARGE)

        # set cell event constrain:
        for i in range(self.seg_N):
            self.model.add_constr(is_merges[i] + is_divisions[i] <= 1)
            self.model.add_constr(is_edges_in[i] + appearances[i] <= 1)
            self.model.add_constr(is_edges_out[i] + disappearances[i] <= 1)

        # cell event across cell constain:
        for e in range(self.weight_matrix.count_nonzero()):
            target_index = rows[e]
            source_indiex = cols[e]
            self.model.add_constr(is_divisions[target_index] + is_merges[source_indiex] + edges[e] <= 2)

        return is_nodes, edges


    def _add_hierarchy_conflict(self):
        for hier in self.hier_arr:
            for node in hier.all_nodes():
                for super in node.all_supers():
                    self.model.add_constr(self.nodes[node.index] + self.nodes[super] <= 1)
       