import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from mpl_interactions import zoom_factory
from IPython.display import display
import matplotlib

from cell_event import CellEvent
from visualizer import CELL_EVENT_COLOR
import visualizer
import composer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


class Viz(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.axis('off')
        
        super(Viz, self).__init__(self.fig)

        self.ax.set_facecolor('none')
        self.fig.patch.set_facecolor('none')
        
        # Set the Qt widget's palette to transparent
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0, 0))
        self.setPalette(palette)

        self.ax.callbacks.connect('xlim_changed', self.on_zoom)
        self.ax.callbacks.connect('ylim_changed', self.on_zoom)
        
        # Set the background of the QWidget which contains the canvas to transparent
        self.setAttribute(Qt.WA_TranslucentBackground)


    def run(self, composer, G):
        self.composer = composer
        self.G = G
        self.label_info_index = 0
        self.label_style_index = 0
        self.max_frame = composer.frame_num - 1

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


    def update_plot(self, main_window):
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
        #self.ax.set_title(f"Frame: {self.frame}", weight = 600) 

        self.ax.set_facecolor('none')
        self.fig.patch.set_facecolor('none')

        disconnect_zoom = zoom_factory(self.ax)
        self.fig.canvas.draw_idle()
        

    def reset_zoom(self):
        """Reset the zoom level to the original xlim and ylim."""

        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.fig.canvas.draw_idle()


    def on_zoom(self, event):
        # Example zoom event handler that would need to be connected to the matplotlib event system
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Calculate aspect ratios
        plot_aspect_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
        canvas_aspect_ratio = self.figure.get_figwidth() / self.figure.get_figheight()

        # Adjust limits based on aspect ratio comparison
        if plot_aspect_ratio > canvas_aspect_ratio:
            # Adjust ylim to maintain aspect ratio
            xc = np.mean(xlim)
            xrange = (ylim[1] -ylim[0]) * canvas_aspect_ratio
            # [xc - xrange / 2, xc + xrange / 2]
            self.ax.set_xlim(ylim * canvas_aspect_ratio)
        else:
            yc = np.mean(ylim)
            yrange = (xlim[1] - xlim[0]) / canvas_aspect_ratio
            #self.ax.set_ylim([yc - yrange / 2, yc + yrange / 2])
            self.ax.set_ylim(xlim / canvas_aspect_ratio)


        self.draw_idle()
