"""Lymph interface script.

Handles calls from the MATLAB 'lymph' library,
returning a polytopal mesh obtained by agglomeration.

Calls to this script should be done using MATLAB `pyrunfile` with the
following signature:

The parameters are passed to `pyrunfile` as keyword arguments.

Parameters
----------
meshpath : str
    Input mesh file path.
aggmodel : {'METIS', 'KMEANS', 'SageBase2D'}
    The agglomeration model to be used. The supported options are:
    'METIS' : Metis graph partitioning algorithm.
    'KMEANS' : Kmeans algorithm.
    'SageBase2D' : Graph Neural Network based strategy.
mode : {'Nref', 'mult_factor'}
    Agglomeration mode.
    'Nref' : bisect the mesh recursively a set number of times.
    'mult_factor' : bisect the mesh until the agglomerated elements are small
    enough.
agg_parameter : int | float
    If `agglomeration_mode` is 'Nref', then this is an integer corresponding
    to the number of refinements; if it is 'mult_factor'then this is a float
    corresponding to the ratio between the the desired agglomerted elements
    diameter and that of the entire mesh. Must be between 0 and 1.
b_tags : Array_like of int, optional.
    The tags of the elements to insert in the boundary.
    By default, insert in the boundary all elements of suitable dimension
    (i.e. all lines in 2D), and assign them tag `1`.
b_tag_name : str, optional
    Field name of the tags to be used in `meshio.Mesh.cell_data` dictionary
    for boundary extraction.

Notes
-----
The input mesh is first loaded using `meshio` and may be in any format
compatible with it.

Regarding the boundary:

See Also
--------
magnet.aggmodels.AgglomerationModel.agglomerate : agglomerate a mesh.
magnet.io.load_mesh : load mesh from file.
magnet.io.get_boundary : extract boundary of the mesh.
"""
import os
import argparse
import numpy as np

from magnet import aggmodels
from magnet.io import load_mesh, save_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--meshpath', type=str,
                        help='Path of the mesh to be agglomerated.')
    parser.add_argument('--aggmodel', default='KMEANS', type=str,
                        help='Agglomeration model to be used')
    parser.add_argument('--mode', default='Nref', type=str,
                        help='Agglomeration mode to be used')
    parser.add_argument('--nref', default=7, type=int,
                        help='Number of refinements in "nref" and \
                            "multilevel" modes.')
    parser.add_argument('--multfactor', default=0.1, type=float,
                        help='Multiplicative factor in "mult_factor" and \
                            "bogo" modes.')
    parser.add_argument('--k', default=128, type=int,
                        help='Number of agglomerated elements in "kway" mode.')
    parser.add_argument('--cthreshold', default=200, type=int,
                        help='Coarsening graph size threshold for multilevel \
                            approach.')
    parser.add_argument('--getboundary', default=False, type=bool,
                        help='Extract the boundary when loading the mesh.')
    parser.add_argument('--btags', nargs="*", default=None, type=int,
                        help='Tags of boundary elements to be loaded \
                            (defaults to all).')
    parser.add_argument('--btagname', default=None, type=str,
                        help='Name of the tag field to consider when \
                            extracting boundary.')
    parser.add_argument('--save', default=None, type=str,
                        help='Output path for mesh saving. If not given, the \
                            mesh is not saved.')
    parser.add_argument('--tolymph', default=False, type=bool,
                        help='Package data for lymph conversion.')

    args = parser.parse_args()

    # get the current directory of the script and construct the path to the
    # 'models' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, 'models')

    # Load mesh.
    M = load_mesh(args.meshpath,
                  get_boundary=args.getboundary,
                  b_tags=args.btags,
                  b_tag_name=args.btagname)
    if args.tolymph and M.dim > 2:
        raise ValueError('The mesh must be 2D for lymph conversion.')

    # initialize agglomeration model.
    match args.aggmodel:
        case 'METIS':
            agg_model = aggmodels.METIS()
        case 'KMEANS':
            agg_model = aggmodels.KMEANS()
        case 'SAGEBase2D':
            agg_model = aggmodels.SageBase2D(64, 32, 3, 2).to(aggmodels.DEVICE)
            agg_model.load_model(os.path.join(models_path,'SAGEBase2D.pt'))
        case 'multiSAGE':
            agg_model = aggmodels.SageBase2D(64, 32, 3, 2).to(aggmodels.DEVICE)
            agg_model.load_model(os.path.join(models_path,'SAGEBase2D.pt'))
            refiner = aggmodels.Reyyy(5, 10).to(aggmodels.DEVICE)
            refiner.load_model(os.parh.join(models_path,'RLrefiner.pt'))
        case _:
            raise ValueError('Unknown agglomeration model: %s' % args.aggmodel)

    # agglomerate the mesh
    match args.mode:
        case 'Nref':
            aggM = agg_model.agglomerate(M, args.mode,
                                         nref=args.nref)
        case 'mult_factor' | 'bogo':
            aggM = agg_model.agglomerate(M, args.mode,
                                         mult_factor=args.multfactor)
        case 'direct_kway':
            aggM = agg_model.agglomerate(M, args.mode, nref=args.k)
        case 'multilevel':
            aggM = agg_model.agglomerate(M, args.mode,
                                         refiner=refiner,
                                         threshold=args.cthreshold,
                                         nref=args.nref)
        case _:
            raise ValueError('Unknown agglomeration mode: %s' % args.mode)

    # Save the mesh to file.
    if args.save is not None:
        save_mesh(aggM, args.save)

    if args.tolymph:
        # pre-conversions to speed up conversion in MATLAB;
        # also, change indices to start from 1.
        aggM._sort_counterclockwise()
        vertices = aggM.Vertices
        connectivity = [np.array(C.Nodes)+1 for C in aggM.Cells]
        # use physical group as element tags.
        physical_groups = aggM.Physical_Groups
        if aggM.Boundary is not None:
            boundary = np.array(aggM.Boundary.Faces)+1
            b_tags = aggM.Boundary.Tags
        else:
            boundary = np.zeros(0)
            b_tags = np.zeros(0)
