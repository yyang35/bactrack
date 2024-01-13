import logging

Graph_solver_logger = logging.getLogger(__name__)

class GraphSolver:
    def __init__(self, weight_matrix, hier_arr):
        self.hier_arr = hier_arr
        self.weight_matrix = self.weight_matrix
        self.graph = self._build_graph()

    def solve(self):
        pass 

    def _build_graph(self):
        pass