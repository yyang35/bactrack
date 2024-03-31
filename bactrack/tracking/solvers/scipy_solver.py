import logging

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import dok_matrix, csr_matrix
import scipy
import numpy as np

from .solver import Solver
import bactrack.config as config


scipy_solver_logger = logging.getLogger(__name__)


class ScipySolver(Solver):
    def __init__(self, weight_matrix, hier_arr, mask_penalty = None, coverage = None):
        self.hier_arr = hier_arr
        self.weight_matrix = weight_matrix.tocsr()
        self.rows, self.cols = self.weight_matrix.nonzero() 
        self.data = self.weight_matrix.data
        assert len(self.rows) == len(self.cols) == len(self.data), \
            "Weight matrix should have same length of rows, cols and data"
        self.seg_N = hier_arr[-1]._index[-1]
        self.coverage = 1 if coverage is None else coverage
        self.mask_penalty = np.zeros(self.seg_N) if mask_penalty is None else mask_penalty
        self.c, self.constraints, self.integrality = self._build_mip()


    def solve(self):
        bounds = Bounds(0, 1)
        res = milp(c=self.c, constraints=self.constraints, integrality=self.integrality, bounds=bounds)
        edge_list = res.x[4*self.seg_N:]
        node_list = res.x[:self.seg_N]
        self.res = res

        """
            assert len(edge_list) == self.weight_matrix.count_nonzero(), \
            "Edge binary choosen list should have same length with edge list it self"
        """

        e_matrix = dok_matrix(self.weight_matrix.shape)
        for i in np.where(edge_list > 0.5)[0]:
            #assert node_list[rows[i]] == 1 and node_list[cols[i]] == 1, "Edge should have a start node"
            e_matrix[self.rows[i], self.cols[i]] = 1

        n = np.where(node_list > 0.5 )[0]
        return n, e_matrix
        

    def _build_mip(self):

        """
            mip problem:
            objective: cx
            s.t:  A @ x <= b
                    x >= 0
        """

        # node_frames in format [0,0,0,0,1,1,1,1,2,2,2,2,]
        # it's node index to node frame list
        frames = np.array([node.frame for hier in self.hier_arr for node in hier.all_nodes()])
        not_start = frames != 0
        not_end = frames != np.max(frames)

        assert len(frames) == self.seg_N, \
            "node's frames info list should have a length of total segmentation candidates"
        
        assert len(self.mask_penalty) == self.seg_N, \
            "Mask penalty should have a length of total segmentation candidates"

        # set up c in mip problem, also known as objective function coeff

        # node objective function coeff
        node_obj = -1 * self.mask_penalty
        # appearence objective function coeff, penalty as long as not start frame
        appear_obj = -1 * not_start * config.APPEAR_COST
        # disappearence objective function coeff, penalty as long as not end frame
        disappear_obj = -1 * not_end * config.DISAPPEAR_COST
        # dividsion objective function coeff
        division_obj = -1 * np.ones(self.seg_N) * config.DIVISION_COST 
        # edge objective function coeff
        # weight_matrix store edge in: weight_matrix.nonzero() = rows, cols and weight_matrix.data
        # this means: an edge come from rows[i] to cols[i] have an weight data[i]
        edge_obj = self.data

        # master objective coeff
        c = np.concatenate((node_obj, appear_obj, disappear_obj, division_obj, edge_obj))

        # scipy milp only objective to minimize, so reverse the coeff
        c *= -1

        self.c = c 

        def _index(type, offset):
            return ['NODE','APPEAR', 'DISAPPEAR', 'DIVISION', 'EDGE'].index(type) * self.seg_N + offset
        

        # A @ x <= b 
        # A is a matrix, x is the column vector, b is the column vector

        rows, cols = self.rows, self.cols
        b_lb, b_ub = [], [] # lower bound and upper bound of b
        A = dok_matrix((self.seg_N * 3, len(c)), dtype=np.float32)
        row_index = -1

        # set constrain part, which is the A, b part of 
        for i in range(self.seg_N):
            # the indices below are unique zip edges indices
            target_indices = np.where(rows == i)[0]
            source_indices = np.where(cols == i)[0]

            # Step 1
            # A_i for flow conservation:
            # sum(in_edge) + appear = node
            # equivalent to inequations
            #   sum(in_edge) + appear - node >=  0
            #   sum(in_edge) + appear - node <=  0

            row_index += 1
            for source_index in source_indices:
                # in edge
                A[row_index, _index('EDGE', source_index)] = 1

            A[row_index, _index('APPEAR', i)] = 1
            A[row_index, _index('NODE', i)] = -1

            b_lb.append(0)
            b_ub.append(0)

            
            # Step 2
            # node + division = sum(out_edge) + disappear
            # equivalent to inequations
            #   node + division - sum(out_edge) -  disappear >= 0
            #   node + division - sum(out_edge) -  disappear <= 0

            row_index += 1
            for target_index in target_indices:
                # out edge
                A[row_index, _index('EDGE', target_index)] = -1

            A[row_index, _index('DIVISION', i)] = 1
            A[row_index, _index('NODE', i)] = 1
            A[row_index, _index('DISAPPEAR', i)] = -1

            b_lb.append(0)
            b_ub.append(0)


            # Step 3
            # M * node >= division
            # M * node - division >= 0
            row_index += 1
            A[row_index, _index('DIVISION', i)] = -1
            A[row_index, _index('NODE', i)] = 1

            b_lb.append(0)
            b_ub.append(np.inf)

        print(A.shape)
        print(row_index)

        # Step 4
        # set constrain part, set the constrain on hierachy conflict
        for hier in self.hier_arr:
            for node in hier.all_nodes():
                for super in node.all_supers():
                    row_index += 1
                    A.resize((row_index + 1, A.shape[1]))
                    A[row_index, _index('NODE', node.index)] = 1
                    A[row_index, _index('NODE', super)] = 1
                    b_lb.append(0)
                    b_ub.append(1)


        # Step 4
        # set constrain part, set the constrain on hierachy conflict
        threshold = self.coverage
        row_index += 1
        A.resize((row_index + 1, A.shape[1]))
        for hier in self.hier_arr:
            hier.root.coverage = 1 / len(self.hier_arr)
            self._assign(hier.root)
            for node in hier.all_nodes():
                A[row_index, _index('NODE', node.index)] = node.coverage

        b_lb.append(threshold)
        b_ub.append(np.inf)

        assert row_index == A.shape[0] - 1, \
            "Row index should equal to A shape[0] -1"
        assert (len(b_lb) == A.shape[0]) and (len(b_ub) == A.shape[0]), \
            "Lower bound and upper bound should have same length with A shape[0]"
        
        print(b_lb)
        print(b_ub)

        self.A = A
        self.ub = b_ub

        constraints = LinearConstraint(A, lb = b_lb, ub = b_ub)
        integrality = None
        return c, constraints, integrality

