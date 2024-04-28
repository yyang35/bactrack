import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import ipywidgets as widgets
from mpl_interactions import zoom_factory
from IPython.display import display
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

from bactrack.gui.cell_event import CellEvent
from bactrack.gui.visualizer import CELL_EVENT_COLOR
import bactrack.gui.visualizer as visualizer
import bactrack.gui.composer as composer

class ImageEnum(Enum):
    RAW = "raw image"
    LINK = "link result"
    FLOW = "flow field"


class Viz(FigureCanvasQTAgg):

    def __init__(self, main_window, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.max_frame = 0
        self.ax = self.fig.add_subplot(111)
        #self.ax.set_axis_off()
        #self.ax.axis('off')
        self.choice = ImageEnum.RAW
        self.fig.set_facecolor(main_window.bg_color)
        
        super(Viz, self).__init__(self.fig)

    def run(self, composer, G):
        self.composer = composer
        self.G = G
        self.label_info_index = 0
        self.label_style_index = 0

        label_info_1 = visualizer.get_label_info(G)
        label_info_2 = visualizer.get_generation_label_info(G)
        self.labels = [label_info_1, label_info_2]

        label_info = self.labels[self.label_info_index]
        self.label_styles = ["regular", "circled", "empty"]
        self.label_style = self.label_styles[self.label_style_index]
  
        image = self.composer.get_single_frame_phase(frame = 0)
        self.ax = visualizer.subplot_single_frame_phase(ax=self.ax, G=G, image=image, cells_frame_dict=composer.cells_frame_dict, label_style  = self.label_style, frame=0, info=label_info, fontsize=7, representative_point=True)

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0) 

        #self.ax.set_title(f"Frame: {self.frame}", weight = 600) 
        self.ax.set_axis_off()
        self.fig.canvas.draw()

    def show_raw(self, images):
        self.max_frame = len(images)
        self.images = images
        self.ax.imshow(images[0], cmap='gray')

    def update_plot(self, main_window):
        self.update_plot_raw(main_window)
        if self.choice == ImageEnum.RAW:
            self.update_plot_raw(main_window)
        elif self.choice == ImageEnum.LINK:
            self.update_plot_link(main_window)

    def update_plot_link(self, main_window):
        # Assuming visualizer, composer, and G are defined
        # Create the initial plot
        # Update plot function
        label_info = self.labels[main_window.label_index]
        label_style = self.label_styles[main_window.style_index]

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)  
        self.ax.set_facecolor('none')  
        
        self.ax.clear()
        image = self.composer.get_single_frame_phase(main_window.frame)
        self.ax = visualizer.subplot_single_frame_phase(
            ax=self.ax, 
            G=self.G, 
            image=image, 
            cells_frame_dict=self.composer.cells_frame_dict, 
            label_style  = label_style, 
            frame=main_window.frame, 
            info=label_info, 
            fontsize=7, 
            representative_point=True
        )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        disconnect_zoom = zoom_factory(self.ax)
        self.fig.canvas.draw_idle()

    def update_plot_raw(self, main_window):
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
        self.fig.canvas.draw_idle()

    def reset_zoom(self):
        """Reset the zoom level to the original xlim and ylim."""
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.fig.canvas.draw_idle()
