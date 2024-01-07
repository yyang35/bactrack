import mip
import config
from scipy.sparse import csr_matrix


# in bipar
def solve(hier_arr):
    pass


    


def make_weight_matrix(hier_arr, len):
    """Build the weight matrix which include all matrix connect to """
    # since this's a gloabl matrix including all candidates segementations in all frame
    # its size could be really large, use Scipy sparse matrix to save memory
    empty_csr_matrix = csr_matrix((len, len), dtype=float)
    csr_matrix[i,j] = 





    





def set_object(model):

    appearances = model.add_var_tensor(size, name="appear", var_type=mip.BINARY)
    disappearances = model.add_var_tensor(size, name="disappear", var_type=mip.BINARY )
    divisions = model.add_var_tensor(size, name="division", var_type=mip.BINARY)

    model.objective = (
        mip.xsum(divisions * config.DIVISION_COST)
        + mip.xsum(appearances * config.APPEAR_COST)
        + mip.xsum(disappearances * config.DISAPPEAR_COST)
    )
    
    edges = model.add_var_tensor((len(weights),), name="edges", var_type=mip.BINARY)

    objective += mip.xsum(weights * edges)

    pass
    



def set_constraints(model):

    for i in range(self._nodes.shape[0]):

        try:
            i_sources = edges_targets.get_group(i).index
        except KeyError:
            i_sources = []

        try:
            i_targets = edges_sources.get_group(i).index
        except KeyError:
            i_targets = []

        # single incoming node
        model.add_constr( mip.xsum(self._edges[i_sources]) + self._appearances[i]
            == self._nodes[i]
        )

        # flow conservation
        model.add_constr(
            self._nodes[i] + self._divisions[i]
            == mip.xsum(self._edges[i_targets]) + self._disappearances[i]
        )

        # check this 
        model.add_constr(self._nodes[i] >= self._divisions[i])






    pass