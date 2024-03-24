import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
from PIL import Image
import string
import cv2
import os
import matplotlib.patches as patches
import glob
import networkx as nx
import matplotlib.ticker as ticker
from typing import Set
from pathlib import Path
import random
import colorsys
from descartes import PolygonPatch

from cell import Cell
from cell_event import CellEvent, CellType, CellDefine
from composer import LinkComposer


# =========================  color styling constant ================================ #

CELL_EVENT_COLOR = {
    CellEvent.SPLIT:"blue", 
    CellEvent.MERGE:"green", 
    CellEvent.DIE:"red", 
    CellEvent.BIRTH: "purple"
}


CELL_TYPE_COLOR = {
    CellType.REGULAR: "#878787", 
    CellType.SPLIT:"#1500FF", 
    CellType.SPLITED:"#756BE1", 
    CellType.MERGE:"#30FF00", 
    CellType.MERGED:"#8DE279", 
    CellType.DIE:"#FF0000", 
    CellType.BIRTH: "#FF00ED",
    CellType.UNKOWN: "#666666",
}


# ============================ lineage related ====================================== #

# get each node lineage's position, get each node's location
# if input certain set of cells, position will be optimized depending on this specifc set.
def get_lineage_pos(G: nx.Graph, cells: Set[Cell] = None):
    """
    Get the horizontal position of each cell in the lineage graph.

    """
    cells = list(G.nodes()) if cells is None else cells
    cells.sort()

    pos = {}
    left_pos = 0
    for cell in cells:
        if cell not in pos:
            make_pos(G, cell, pos, left_pos, 1)
            left_pos += 1
    return pos


# helper function for get_lineage_pos
# Deep Fist Search to label all horizontal position of each cell
def make_pos(G, node, pos, left_pos, width):
   """
   Recursively assign horizontal positions to each node in the graph.
   """
   pos[node] = (left_pos + width/2, -1 * node.frame)
   children_nodes =  list(G.successors(node))
   children_nodes.sort()
   if len(children_nodes) == 0 : return 
   slice_width = width / len(children_nodes)
   for i in range(len(children_nodes)):
      node = children_nodes[i]
      make_pos(G, node, pos, left_pos + i * slice_width, slice_width)



# for lineage, return a set of special edges and nodes, which used to shown overlap on normal lineage
def tag_type(G, cells = None):
    """
    Tag each cell in the lineage graph with its event type (birth, death, split, merge).
    """
    tag_dict = {CellEvent.DIE: set(), CellEvent.BIRTH: set(), CellEvent.SPLIT: set(), CellEvent.MERGE:set()}     
    cells = set(G.nodes()) if cells is None else cells
    for cell in cells:
        define = CellDefine(G, cell)
        # add special nodes:
        if define.die: tag_dict[CellEvent.DIE].add(cell)
        elif define.birth: tag_dict[CellEvent.BIRTH].add(cell)
        # add special edges:
        if define.merge:
            edges = G.in_edges(cell)
            tag_dict[CellEvent.MERGE].update(set(edges))
        if define.split:
            edges = G.out_edges(cell)
            tag_dict[CellEvent.SPLIT].update(set(edges))
    return tag_dict


# This plot lineage make some default highlight infomation on lineage: include cell events and basically statstic information
def quick_lineage(G, globally = False, figsize = (10,8), **kwargs):
    """
    Plot the lineage graph with special nodes and edges highlighted.
    """
    tag = tag_type(G)

    defines = CellDefine.define_cells(G)
    cells = [define.cell for define in defines if not define.ghost]
    
    pos = get_lineage_pos(G) if globally else get_lineage_pos(G, cells)

    edges = {"grey": G.edges()}
    nodes = {"grey": set(G.nodes()).difference(cells)} if globally else {} 

    nodes.update({CELL_EVENT_COLOR[CellEvent.BIRTH]: tag[CellEvent.BIRTH], CELL_EVENT_COLOR[CellEvent.DIE]: tag[CellEvent.DIE]})
    edges.update({CELL_EVENT_COLOR[CellEvent.SPLIT]: tag[CellEvent.SPLIT], CELL_EVENT_COLOR[CellEvent.MERGE]: tag[CellEvent.MERGE]})

    plot_lineage(G, pos, with_background = False , nodes_special = nodes, edges_special = edges, figsize = figsize,  **kwargs)



# master lineage function
def plot_lineage(G, pos, figsize = (15,12), **kwargs):
    """
    Plot the lineage graph with special nodes and edges highlighted.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax =  subplot_lineage(ax, G, pos, **kwargs)
    plt.show()



# plot lineage on a ax, this be factor out for any use of subplot, 
def subplot_lineage(ax, G, pos, **kwargs):
    """
    Plot the lineage graph with special nodes and edges highlighted.
    """
    node_list = list(G.nodes())
    node_list.sort()

    with_background = kwargs.get('with_background', False)
    nodes_special = kwargs.get('nodes_special', {})
    edges_special = kwargs.get('edges_special', {})
    show_stat = kwargs.get('show_stat', True)
    arrow = kwargs.get('arrow', False)
    
    # draw background
    if with_background: 
        nx.draw(G, pos, node_size = 0,  width=1, edge_color="grey", arrows = arrow, ax=ax)
    # draw special nodes, and edges that need be highlight
    for color, nodes in nodes_special.items():
        nx.draw_networkx_nodes(G, pos, nodelist=list(nodes), node_size=15, node_color=color, ax=ax)
    for color, edges in edges_special.items():
        nx.draw_networkx_edges(G, pos, edgelist=edges , width=1, edge_color=color, arrows= arrow, ax=ax)
    # show statstic infomation 
    if show_stat:
        text = get_graph_stats_text(G)
        ax.text(0.95, 0.95,  text, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')

    # styling below
    ax.set_frame_on(False)
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)

    def format_fn(tick_val, tick_pos):
        return f"frame {int(abs(tick_val))}" if tick_val % 1 == 0 else ""

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_fn))

    limits=plt.axis('on')
    ax.grid(True, linestyle='--', alpha=0.5)

    return ax



# function decide what to show on lineage slice for each frame
# extract all the node appear on this frame, get all edges/nodes it connected to
def get_single_frame_lineage_info(G, frame):
    """
    Get the special nodes and edges to highlight for a specific frame in the lineage graph.
    """
    composer = LinkComposer(G.nodes())

    cells_center = composer.cells_frame_dict[frame]
    cell_s = {"red": cells_center}

    connected_nodes, connected_edges = get_connected_edges_cells(G, cells_center)
    
    edge_s = {"grey": connected_edges}
    cell_s = {"grey": connected_nodes, "blue": cells_center}

    all_cells = connected_nodes.union(cells_center)
    tag = tag_type(G, cells_center)

    edge_s[CELL_EVENT_COLOR[CellEvent.SPLIT]] = tag[CellEvent.SPLIT]
    edge_s[CELL_EVENT_COLOR[CellEvent.MERGE]] = tag[CellEvent.MERGE]

    cell_s[CELL_EVENT_COLOR[CellEvent.BIRTH]] = tag[CellEvent.BIRTH]
    cell_s[CELL_EVENT_COLOR[CellEvent.DIE]] = tag[CellEvent.DIE]

    pos = get_lineage_pos(G, list(connected_nodes.union(cells_center)))

    return  cell_s, edge_s, pos



def get_connected_edges_cells(G, cells):
    """
    Get the set of connected edges and nodes for a set of cells in the lineage graph.
    """

    connected_nodes = set()
    connected_edges = set()

    for cell in cells:
        # Get incoming and outgoing edges for each cell
        in_edges = G.in_edges(cell)
        out_edges = G.out_edges(cell)

        # Update the sets of connected edges and nodes
        connected_edges.update(in_edges)
        connected_edges.update(out_edges)

        # Extract nodes from edges
        for edge in in_edges:
            connected_nodes.add(edge[0])  # Source node of the incoming edge
        for edge in out_edges:
            connected_nodes.add(edge[1])  # Target node of the outgoing edge
    
    return connected_nodes, connected_edges


# give a block of statstic text of a linking graph, shown on lineage
def get_graph_stats_text(G):
    """
    Get a block of text containing statistics about the lineage graph.
    """

    composer = LinkComposer(G.nodes())
    max_frame = composer.frame_num - 1

    merge, split, birth, death, ghost, irregular_death = 0, 0, 0, 0, 0, 0

    for cell in composer.cells:
        define = CellDefine(G, cell)
        merge += define.merge
        split += define.split
        birth += define.birth
        death += define.die
        ghost += define.ghost
        if cell.frame != max_frame and define.die: irregular_death += 1

    coverage_rate = (len(composer.cells) - ghost) / len(composer.cells)
    frame_index = sorted(composer.cells_frame_dict)
    last_frame_cells = composer.cells_frame_dict[frame_index[-1]]
    cell_num = len(last_frame_cells)
    text = f"""
            Max frame: {max_frame}
            Coverage rate: {coverage_rate:.2%}
            Last frame cell num: {cell_num}
            Merge: {merge}, Split: {split}, Birth: {birth}, Death: {death}
            Ghost: {ghost}, Irregular death: {irregular_death}"""
    return text



# ============================== phase/fluorescent video stretch related ================================== #

# for images visualization, label the cell label and it's type 
# this cell label only for reability and represent relative relationship, no strict label be applied
def get_label_info(G):
    """
    Get the label information for each cell in the microscope image.
    """
    cells = list(G.nodes())
    cells.sort()

    max_frame = cells[-1].frame 

    info = {}
    cell_id = 0

    def get_new_label():
        nonlocal cell_id 
        cell_id = cell_id + 1
        return cell_id


    for cell in cells:
        if cell not in info:
            define = CellDefine(G, cell)
            incoming = len(list(G.predecessors(cell)))
            # label birth/death, notice they are not conflict with merge/split
            # label birth/death first, so merge/split have higher priority, since will keep the label of latest asssigned
            if define.birth:
                info[cell] = (get_new_label(), CELL_TYPE_COLOR[CellType.BIRTH])
            elif define.ghost:
                info[cell] = ("Ø", CELL_TYPE_COLOR[CellType.UNKOWN])

            # lable all other events. regular(1 to 1), split, and merge are parallel structure. 
            if incoming > 0:
                mother = list(G.predecessors(cell))[0]
                if define.regular: 
                    info[cell] = (info[mother][0], CELL_TYPE_COLOR[CellType.REGULAR])
                elif define.split:
                    # label itself 
                    info[cell] = (info[mother][0], CELL_TYPE_COLOR[CellType.SPLIT])
                    # label it's outgoing cells 
                    cell_list = list(G.successors(cell))
                    cell_list.sort()
                    for i in range(len(cell_list)):
                        outgoing_cell = cell_list[i]
                        info[outgoing_cell] = (get_new_label(), CELL_TYPE_COLOR[CellType.SPLITED])
                elif define.merge:
                    # label itself
                    info[cell] = (get_new_label(), CELL_TYPE_COLOR[CellType.MERGE])
                    # label it's incoming cells 
                    cell_list = list(G.predecessors(cell))
                    cell_list.sort()
                    for i in range(len(cell_list)):
                        income_cell = cell_list[i] 
                        label = info[income_cell][0]
                        info[income_cell] = (label,CELL_TYPE_COLOR[CellType.MERGED])       
                elif define.die:
                    if cell.frame == max_frame:
                        info[cell] = (info[mother][0], CELL_TYPE_COLOR[CellType.UNKOWN])
                    else:
                        info[cell] = (info[mother][0], CELL_TYPE_COLOR[CellType.DIE])  

    return info



def get_edges_related_label_info(G, edges, color = "#FF7000"):

    """
    Get the label information for each cell in the microscope image connected to certain edges.
    """

    cells = []
    # Iterate over the edge set and add the nodes to cells
    for node1, node2 in edges:
        cells.append(node1)
        cells.append(node2)

    cells = list(cells)
    cells.sort()

    info = {}
    cell_id = 0
    alphabet = string.ascii_letters

    def get_new_label():
        nonlocal cell_id 
        label = alphabet[cell_id % len(alphabet)]
        cell_id = cell_id + 1
        return label

    for cell in cells:
        if cell not in info:
            incoming = list(G.predecessors(cell))

            if len(incoming) == 1 and incoming[0] in info:
                info[cell] = (info[incoming[0]][0], color)
            else:
                info[cell] = (get_new_label(), color)
    
    for node_source, node_target in edges:
        if CellDefine(G, node_source).split:
            # label it's outgoing cells 
            cell_list = list(G.successors(cell))
            cell_list.sort()
            for i in range(len(cell_list)):
                outgoing_cell = cell_list[i]
                label = info[node_source][0] + str(i)
                info[outgoing_cell] = (label, color)

        if CellDefine(G, node_target).merge:
            # label it's incoming cells 
            cell_list = list(G.predecessors(cell))
            cell_list.sort()
            for i in range(len(cell_list)):
                income_cell = cell_list[i] 
                label = info[cell][0]+ str(i) 
                info[income_cell] = (label,color)         
                
    return info



def get_generation_label_info(G):
    """
    Get the 'generation style' label information for each cell in the microscope image.
    """
    cells = list(G.nodes())
    cells.sort()
    info = {}
    cell_id = 0
    random.seed(119)

    alphabet = string.ascii_lowercase

    def get_new_label():
        nonlocal cell_id 
        label = alphabet[cell_id % len(alphabet)]
        cell_id = cell_id + 1
        return label
    
    def label_cell(cell, color, label, index):
        current_label = label + str(index) if index > 0 else label
        info[cell] = (current_label, color)

        outgoing_cells = list(G.successors(cell))
        new_index = index if len(outgoing_cells) == 1 else index + 1
        for outgoing_cell in list(G.successors(cell)):
            label_cell(outgoing_cell, color, label, new_index)

    for cell in cells:
        if cell not in info:
            color = new_color(random.random())
            label_cell(cell, color, get_new_label(), 0)

    return info



def new_color(hue, saturation = 1, brightness = 1):
    """
    Generate a random pure color with full saturation and brightness.
    """
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))



def plot_single_frame_phase(G, info, frame, image, cells_frame_dict= None, save = False, **kwargs):
    """
    Plot the microscope image with the cell labels and types for a single frame.
    """
    # Create a new figure and axis
    figsize = kwargs.get('figsize', None)
    fig, ax = plt.subplots(figsize = figsize)

    cells_frame_dict = LinkComposer(set(G.nodes())).cells_frame_dict if cells_frame_dict is None else cells_frame_dict
    ax = subplot_single_frame_phase(ax, image, cells_frame_dict, info, frame, **kwargs)
    
    plt.axis('off')
    plt.tight_layout()
    if save:
        dir_path = kwargs.get('dir_path', './output')
        output_path = os.path.join(dir_path, f"frame{frame:05d}.png")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)



# Draw polygon and cell id label on each raw image 
def subplot_single_frame_phase(ax, image, cells_frame_dict, info, frame,  **kwargs):
    """
    Plot the microscope image with the cell labels and types for a single frame.
    """
    # Display the image
    ax.imshow(image)

    label_style = kwargs.get('label_style', 'empty')
    representative_point = kwargs.get('representative_point', False)


    for cell in cells_frame_dict[frame]:
        if cell in info:
            color = info[cell][1]
            label = info[cell][0]

            # Half transparent
            facecolor = color + "60"

            # Create a polygon patch
            path = PolygonPatch(cell.polygon.__geo_interface__, edgecolor=color, facecolor=facecolor)

            # Add the polygon patch to the axis
            ax.add_patch(path)

            if representative_point: 
                 # this is not center of cell, but a point that guaranteed in polygon
                centroid_x = cell.polygon.representative_point().x
                centroid_y = cell.polygon.representative_point().y
            else:
                # Add text annotation
                centroid_x = cell.polygon.centroid.x
                centroid_y = cell.polygon.centroid.y

            fontsize = kwargs.get('fontsize', 8)
            if label is not None and label != "":
                if label_style == 'circled':
                    ax.text(centroid_x, centroid_y, str(label), color='black', fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'), fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
                elif label_style == 'regular':
                    ax.text(centroid_x, centroid_y, str(label), color='white', fontweight='bold', fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

    # Optionally, set the axis limits based on the image size
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    ax.set_axis_off()
    ax.set_frame_on(False)

    return ax 

