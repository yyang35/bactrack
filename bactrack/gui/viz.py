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

class Viz:
    def __init__(self, composer, G):
        self.composer = composer
        self.G = G
        self.label_info_index = 0
        self.label_style_index = 0
        self.frame = 0
        self.max_frame = composer.frame_num - 1

        plt.close('all')
        plt.ioff()
        #matplotlib.use('TkAgg')  # Make sure this backend is compatible with your environment

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [5, 2]})
        self.fig.tight_layout()

        label_info_1 = visualizer.get_label_info(G)
        label_info_2 = visualizer.get_generation_label_info(G)
        self.labels = [label_info_1, label_info_2]

        tag = visualizer.tag_type(G)
        pos = visualizer.get_lineage_pos(G)

        label_info = self.labels[self.label_info_index]
        self.label_styles = ["regular", "circled", "empty"]
        self.label_style = self.label_styles[self.label_style_index]

        cells_s = {CELL_EVENT_COLOR[CellEvent.BIRTH]: tag[CellEvent.BIRTH], CELL_EVENT_COLOR[CellEvent.DIE]: tag[CellEvent.DIE]}
        edges_s = {CELL_EVENT_COLOR[CellEvent.SPLIT]: tag[CellEvent.SPLIT], CELL_EVENT_COLOR[CellEvent.MERGE]: tag[CellEvent.MERGE]}
                
        image = self.composer.get_single_frame_phase(self.frame)
        self.ax1 = visualizer.subplot_single_frame_phase(ax=self.ax1, G=G, image=image, cells_frame_dict=composer.cells_frame_dict, label_style  = self.label_style, frame=self.frame, info=label_info, fontsize=7, figsize=(15,15), representative_point=True)
        self.ax1.set_title(f"Frame: {self.frame}", weight = 600) 

        self.ax2 = visualizer.subplot_lineage(self.ax2, G, pos, with_background=True, nodes_special=cells_s, edges_special=edges_s)
        self.ax2.set_axis_off()
        disconnect_zoom = zoom_factory(self.ax1)

        self.fig.canvas.mpl_connect('key_press_event', lambda event: self.on_key(event))
        plt.show()



    def update_plot(self):
        # Assuming visualizer, composer, and G are defined
        # Create the initial plot
        # Update plot function

        label_info = self.labels[self.label_info_index]
        label_style = self.label_styles[self.label_style_index]

        xlim = self.ax1.get_xlim()
        ylim = self.ax1.get_ylim()
        
        self.ax1.clear()
        image = self.composer.get_single_frame_phase(self.frame)
        self.ax1 = visualizer.subplot_single_frame_phase(
            ax=self.ax1, 
            G=self.G, 
            image=image, 
            cells_frame_dict=self.composer.cells_frame_dict, 
            label_style  = label_style, 
            frame=self.frame, 
            info=label_info, 
            fontsize=7, 
            figsize=(15,15), 
            representative_point=True
        )
        self.ax1.set_xlim(xlim)
        self.ax1.set_ylim(ylim)
        self.ax1.set_title(f"Frame: {self.frame}", weight = 600) 

        disconnect_zoom = zoom_factory(self.ax1)
        self.fig.canvas.draw_idle()


    def on_key(self, event):
        if event.key in ['right', 'down']:
            self.frame = min(self.frame + 1, self.max_frame) 
        elif event.key in ['left', 'up']:
            self.frame = max(self.frame - 1, 0)
        elif event.key == 'c':
            self.label_style_index = (self.label_style_index + 1) % len(self.label_styles)
        elif event.key == 'l':
            self.label_info_index = (self.label_info_index + 1) % len(self.labels)
        else:
            return
        self.update_plot()