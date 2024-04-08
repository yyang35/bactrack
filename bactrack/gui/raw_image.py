import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from mpl_interactions import zoom_factory
from IPython.display import display
import matplotlib

from bactrack.gui.cell_event import CellEvent
from bactrack.gui.visualizer import CELL_EVENT_COLOR
import bactrack.gui.visualizer as visualizer
import bactrack.gui.composer as composer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


class RawImage(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.max_frame = 0
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.axis('off')
        
        super(RawImage, self).__init__(self.fig)

        self.ax.set_facecolor('none')
        self.fig.patch.set_facecolor('none')
        
        # Set the Qt widget's palette to transparent
        """
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0.5, 0.5, 0.5, 1))
        self.setPalette(palette)
        
        """
        # Set the background of the QWidget which contains the canvas to transparent
        #self.setAttribute(Qt.WA_TranslucentBackground)


    def show(self, images):
        self.max_frame = len(images)
        self.images = images
        self.ax.imshow(images[0], cmap='gray')


    def update_plot(self, main_window):
        # Assuming visualizer, composer, and G are defined
        # Create the initial plot
        # Update plot function
        if self.images is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)  
        self.ax.set_facecolor('none')  
        
        self.ax.clear()
        self.ax.imshow(self.images[main_window.frame], cmap='gray')

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        #self.ax.set_title(f"Frame: {self.frame}", weight = 600) 

        self.ax.set_facecolor('none')
        self.fig.patch.set_facecolor('none')
        self.fig.canvas.draw_idle()
        

    def reset_zoom(self):
        """Reset the zoom level to the original xlim and ylim."""

        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.fig.canvas.draw_idle()
