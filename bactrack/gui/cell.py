from shapely.geometry import Polygon

class Cell:
    def __init__(self, frame:int, label:int, polygon:Polygon = None):
        self.frame = frame
        self.label = label
        self.polygon = polygon

    def __hash__(self):
        return hash((self.frame, self.label))

    def __eq__(self, other):
        return (self.frame, self.label) == (other.frame, other.label)
    
    def __lt__(self, other):
        # First sort by frame index, then by cell label
        return (self.frame, self.label) < (other.frame, other.label)

    def __str__(self):
        return f"Cell: frame_{self.frame}, label_{self.label}"
    
    def __repr__(self):
        return f"{str(self)}, <{hash(self):#x}>"