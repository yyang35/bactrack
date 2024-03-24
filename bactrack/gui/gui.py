from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from viz import Viz 
from run import run_track

# from your_viz_module import Viz
# from your_tracking_module import run_track

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtrack")
        self.setGeometry(100, 100, 800, 600)  # Adjust size as needed

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.drag_label = QLabel("Drag Folder Here")
        self.drag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.drag_label)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.viz = None


    def init_viz(self, composer, G):
        # Remove placeholder label if it exists
        if self.drag_label:
            self.drag_label.deleteLater()
            self.drag_label = None

        # Initialize Viz with the existing figure
        if self.viz is None:
            self.viz = Viz(composer, G, self.figure)
        else:
            # If viz already exists, just update it
            self.viz.update_plot()

        self.canvas.show()
        self.canvas.draw_idle()


    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            # You need to define how you obtain `composer` and `G` from the folder_path
            # This might involve calling a function similar to run_track(folder_path)
            # Assuming run_track returns `composer` and `G`
            composer, G = run_track(folder_path)
            self.init_viz(composer, G)
            self.canvas.draw_idle() 

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
