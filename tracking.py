import mip
import config


# in bipar
def solve(hier_arr):
    pass


def add_weight():
    pass
    


def add_penalty(model):

    appearances = model.add_var_tensor(size, name="appear", var_type=mip.BINARY)
    disappearances = model.add_var_tensor(size, name="disappear", var_type=mip.BINARY )
    divisions = model.add_var_tensor(size, name="division", var_type=mip.BINARY)

    model.objective = (
        mip.xsum(divisions * config.DIVISION_COST)
        + mip.xsum(appearances * config.APPEAR_COST)
        + mip.xsum(disappearances * config.DISAPPEAR_COST)
    )
    
    edges = model.add_var_tensor((len(weights),), name="edges", var_type=mip.BINARY)

    objective += mip.xsum(weights * self._edges)

    

def add