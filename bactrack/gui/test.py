import sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt


class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)

    def mousePressEvent(self, event):
        print("mousePress")
        if event.button() == Qt.LeftButton:
            print("get into loop")
            x, y = event.pos().x(), event.pos().y()
            print(x,y)
            node, _ = self.ax.transData.inverted().transform([x, y])
            node = int(round(node))       
            if node in self.graph.nodes:
                self.parent().selected_node = node
                print(f"Selected Node: {self.parent().selected_node}")



class GraphApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.graph = nx.Graph()
        self.create_graph()

        self.central_widget = QWidget()

        self.canvas = MyCanvas(self.central_widget)
        self.canvas.graph = self.graph  # Pass the graph reference to the canvas
        self.update_plot()

        self.delete_button = QPushButton("Delete Node")
        self.delete_button.clicked.connect(self.delete_selected_node)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.canvas)
        layout.addWidget(self.delete_button)

        self.setCentralWidget(self.central_widget)

        self.selected_node = None

    def create_graph(self):
        self.graph.add_nodes_from([1, 2, 3, 4])
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

    def update_plot(self):
        pos = nx.spring_layout(self.graph)
        print(pos)
        nx.draw(self.graph, pos, ax=self.canvas.ax, with_labels=True, node_size=700, node_color='skyblue', font_size=8)

    def delete_selected_node(self):
        if self.selected_node is not None:
            self.graph.remove_node(self.selected_node)
            self.update_plot()
            self.selected_node = None


def main():
    app = QApplication(sys.argv)
    window = GraphApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
