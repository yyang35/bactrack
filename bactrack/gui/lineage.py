import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from mpl_interactions import zoom_factory
from IPython.display import display
import matplotlib


from bactrack.gui.cell_event import CellEvent
from bactrack.gui.visualizer import CELL_EVENT_COLOR
import bactrack.gui.visualizer as visualizer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


class Lineage(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('none')
        self.fig.patch.set_facecolor('none')

        
        super(Lineage, self).__init__(self.fig)

        self.fig.patch.set_alpha(0.0)
        self.ax.patch.set_alpha(0.0)
        
        # Set the Qt widget's palette to transparent
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0, 0))
        self.setPalette(palette)

        # Set the background of the QWidget which contains the canvas to transparent
        self.setAttribute(Qt.WA_TranslucentBackground)


    def show(self, G):
        tag = visualizer.tag_type(G)
        pos = visualizer.get_lineage_pos(G)

        cells_s = {CELL_EVENT_COLOR[CellEvent.BIRTH]: tag[CellEvent.BIRTH], CELL_EVENT_COLOR[CellEvent.DIE]: tag[CellEvent.DIE]}
        edges_s = {CELL_EVENT_COLOR[CellEvent.SPLIT]: tag[CellEvent.SPLIT], CELL_EVENT_COLOR[CellEvent.MERGE]: tag[CellEvent.MERGE]}

        self.ax = visualizer.subplot_lineage(self.ax, 
                                             G, 
                                             pos, 
                                             with_background=True, 
                                             nodes_special=cells_s, 
                                             edges_special=edges_s, 
                                             show_stat = False
                                             )
        #self.ax.set_axis_off()
