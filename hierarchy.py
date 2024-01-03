import pandas as pd

class Node:
    def __init__(self, value, super=None):
        self.value = value
        self.super = super
        self.subs = []

    def add_sub(self, sub):
        """Adds a sub to this node"""
        sub.super = self
        self.subs.append(sub)

class Hierarchy:
    def __init__(self, root):
        self.root = root
        self.index = 0
        self.node_dict = {} 

    def to_dataframe(self):
        """Serializes the hierarchy into a DataFrame"""
        rows = []
        self._to_dataframe_recursive(self.root, rows)
        return pd.DataFrame(rows, columns=['Node Index', 'Value', 'Super Index'])

    def _to_dataframe_recursive(self, node, rows, super_index=None):
        """Helper method to convert to DataFrame recursively"""
        node_index = self._get_node_index(node)
        rows.append({'Node Index': node_index, 'Value': str(node.value), 'Super Index': super_index})
        for sub in node.subs:
            self._to_dataframe_recursive(sub, rows, super_index=node_index)

    def _get_node_index(self, node):
        """Assigns a unique index to a node and stores it in node_dict"""
        if node not in self.node_dict:
            self.node_dict[node] = self.index
            self.index += 1
        return self.node_dict[node]

    @staticmethod
    def from_dataframe(df):
        """Deserializes the DataFrame back into a Hierarchy"""
        nodes = {}
        for _, row in df.iterrows():
            node_index = row['Node Index']
            node_value = row['Value']
            node = Node(node_value)
            nodes[node_index] = node
            super_index = row['Super Index']
            if pd.notnull(super_index):
                super_node = nodes.get(super_index)
                if super_node:
                    super_node.add_sub(node)
        # The root node is the one with no super
        root_node = next(node for index, node in nodes.items() if df[df['Node Index'] == index]['Super Index'].isnull().any())
        return Hierarchy(root_node)