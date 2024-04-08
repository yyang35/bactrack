
class Solver:
    def __init__(self, weight_matrix, hier_arr):
        self.hier_arr = hier_arr
        self.weight_matrix = self.weight_matrix
        self.seg_N = hier_arr[-1]._index[-1]


    def solve(self):
        pass 


    def _assign(self, node):
        if node is None:
            return  
        for sub in node.subs:
            sub.coverage = node.coverage / len(node.subs)
            self._assign(sub)
