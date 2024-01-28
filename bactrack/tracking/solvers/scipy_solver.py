import logging

from scipy.optimize import milp, LinearConstraint
import scipy
import numpy as np

from .solver import Solver
import bactrack.config as config


scipy_solver_logger = logging.getLogger(__name__)


class ScipySolver(Solver):
    def __init__(self, weight_matrix, hier_arr, mask_penalty = None):
        self.hier_arr = hier_arr
        self.weight_matrix = weight_matrix.tocsr()
        self.seg_N = hier_arr[-1]._index[-1]
        self.mask_penalty = mask_penalty
        self.c, self.constraints, self.integrality = self._build_mip()


    def solve(self):
        res = milp(c=self.c, constraints=self.constraints, integrality=self.integrality)
        return res.x
        

    def _build_mip(self):

        """
            mip problem:
            objective: cx
            s.t:  A @ x <= b
                    x >= 0
        """

        # node_frames in format [0,0,0,0,1,1,1,1,2,2,2,2,]
        # it's node index to node frame list
        # e.g 5th segementation candidates came from the frame 1, and 4 sege in frame 1
        node_frames = np.array([node.frame for hier in self.hier_arr for node in hier.all_nodes()])
        assert len(node_frames) == self.seg_N, \
            "node's frames info list should have a length of total segementation candidates"

        # set up c in mip problem, also known as objective function coeff

        # appearence objective function coeff, penalty as long as not start frame
        appear_obj = ( node_frames != 0 ) * config.APPEAR_COST
        # disappearence objective function coeff, penalty as long as not end frame
        disappear_obj = (node_frames != np.max(node_frames)) * config.DISAPPEAR_COST
        # dividsion objective function coeff
        division_obj = np.ones(len(node_frames)) * config.DIVISION_COST 
        # edge objective function coeff
        # weight_matrix store edge in: weight_matrix.nonzero() = rows, cols and weight_matrix.data
        # this means: an edge come from rows[i] to cols[i] have an weight data[i]
        edge_obj = self.weight_matrix.data

        # master objective coeff
        c = appear_obj.tolist() + disappear_obj.tolist() \
            + division_obj.tolist() + edge_obj.tolist()

        
        def _index(type, index):
            return ['APPEAR', 'DISAPPEAR', 'DIVISION', 'EDGE'].index(type) * self.seg_N

        rows, cols = self.weight_matrix.nonzero() 
        A, b_lb, b_ub = [], [], []

        # set constrain part, which is the A, b part of 
        for i in range(self.seg_N):
            target_indices = np.where(rows == i)[0]
            source_indices = np.where(cols == i)[0]

            # A_i for flow conservation:
            # sum(in_edge) + appear = sum(out_edge) + disappear - division
            # sum(in_edge) + appear - sum(out_edge) - disappear +  division = 0
            A_i = np.zeros(len(c))
            for target_index in target_indices:
                A_i[_index('EDGE', target_index)] = 1
            for source_index in source_indices:
                A_i[_index('EDGE', source_index)] = -1

            A_i[_index('APPEAR', i)] = 1
            A_i[_index('DISAPPEAR', i)] = -1
            A_i[_index('DIVISION', i)] = 1

            A.append(A_i)
            b_lb.append(0)
            b_ub.append(0)

            # A_ii for capcity constain
            # sum(in_edge) + appear <= 1
            A_ii = np.zeros(len(c))
            for target_index in target_indices:
                A_ii[_index('EDGE', target_index)] = 1

            A_ii[_index('APPEAR', i)] = 1

            A.append(A_i)
            b_lb.append(0)
            b_ub.append(1)

        # set constrain part, set the constrain on hierachy conflict
        for hier in self.hier_arr:
            for node in hier.all_nodes():
                for super in node.all_supers():
                    A_i = np.zeros(len(c))
                    A_i[node.index] = 1
                    A_i[super] = 1
                    
                    A.append(A_i)
                    b_lb.append(0)
                    b_ub.append(1)

        
        constraints = LinearConstraint(A, b_lb, b_ub)
        integrality = np.ones_like(c)
        return c, constraints, integrality

