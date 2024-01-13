import mip
import logging

from .solver import Solver

MIP_solver_logger = logging.getLogger(__name__)

class MIPSolver(Solver):
    def __init__(self, weight_matrix, hier_arr):
        self.hier_arr = hier_arr
        self.weight_matrix = self.weight_matrix
        self.model =  mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GUROBI)
        self._build_mip()

    def solve(self):
        pass 

    def _build_mip(self):
        pass
