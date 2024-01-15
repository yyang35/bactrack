import argparse
import os

from .core import compute_hierarchy, run_tracking
from .core import ModelEnum
from . import utils 


def main():
    parser = argparse.ArgumentParser(description="Run Segmentation and Tracking")

    # Define arguments for compute_hierarchy
    parser.add_argument('--basedir', type=str, required=True, help='Base directory for images')
    parser.add_argument('--hypermodel', type=str, default=None, choices=['omnipose', 'cellpose'], help='Hypermodel to use')
    parser.add_argument('--chans', nargs=2, type=int, default=[0, 0], help='Channel configuration')
    parser.add_argument('--submodel', type=str, default=None, help='Submodel to use')

    # Define arguments for run_tracking
    parser.add_argument('--solver_name', type=str, required=True, help='Name of the solver to use')
    parser.add_argument('--weight_name', type=str, required=True, help='Name of the weight to use')

    args = parser.parse_args()

    # Convert hypermodel string to enum if necessary
    if args.hypermodel:
        args.hypermodel = ModelEnum[args.hypermodel.upper()]

    # Run compute_hierarchy
    hier_arr = compute_hierarchy(
        args.basedir,
        args.hypermodel,
        args.chans,
        args.submodel
    )

    # Run run_tracking
    nodes, edges = run_tracking(hier_arr, args.solver_name, args.weight_name)

    # Output ans store result 
    masks, edges_df = utils.format_output(hier_arr, nodes, edges)
    nodes_df = utils.hiers_to_df(hier_arr)

    nodes_df.to_pickle(os.path.join(args.basedir, "cells.pkl"))
    edges_df.to_pickle(os.path.join(args.basedir, "links.pkl"))

    

if __name__ == "__main__":
    main()
