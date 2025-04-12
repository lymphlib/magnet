"""Mesh generation module.

Contains functions for generating simple 2D and 3D meshes from scratch.
3D meshes are generated using `gmsh`.
"""

import os
import random
import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, Voronoi

import meshio
import gmsh

from magnet.geometric_utils import (tetrahedron_center, tetrahedron_volume,
                                    shoelace_formula, polygon_centroid)
from magnet._types import adj_ind_type


def delaunay_tria(output_path: str, bounds: tuple[int, int] = (10, 400)):
    """Generate a mesh of random triangles.

    Generates a mesh of the unit square made of random triangles using
    Delaunay triangulation.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    bounds : tuple[float, float], optional
        Bounds of the uniform distribution from which the number of internal
        points of the mesh is sampled (default is (10, 400)).

    Returns
    -------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    centroids : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    areas : np.ndarray of float
        Areas of the cells.

    Notes
    -----
    Internal points are sampled uniformly, while points on the edges are
    equispaced.
    """
    dimension = 2
    # generate the points of the mesh
    n_points = random.randint(bounds[0], bounds[1])
    points_coords = np.array([[random.random() for i in range(dimension)] for j in range(n_points)])
    n_edge_elem = int(n_points**0.5)
    edge1 = np.array([[0, 1/n_edge_elem*i] for i in range(n_edge_elem+1)])
    edge2 = np.array([[1/n_edge_elem*i, 1] for i in range(1, n_edge_elem+1)])
    edge3 = np.array([[1, 1/n_edge_elem*i] for i in range(n_edge_elem+1)])
    edge4 = np.array([[1/n_edge_elem*i, 0] for i in range(1, n_edge_elem+1)])
    points_coords = np.concatenate((points_coords, edge1, edge2, edge3, edge4))

    # compute Delaunay triangulation
    triangulation = Delaunay(points_coords)
    cells = triangulation.simplices  # connectivity data
    num_cells = cells.shape[0]

    adjacency = np.zeros((num_cells, num_cells), dtype=adj_ind_type)
    centroids = np.zeros((num_cells, dimension))
    areas = np.zeros((num_cells, 1))

    # loop over all the simplices of the triangulation
    for i in range(num_cells):
        vertices = points_coords[cells[i]]
        centroids[i] = tetrahedron_center(vertices)
        areas[i] = tetrahedron_volume(vertices)
        for j in triangulation.neighbors[i]:
            if j != -1:
                adjacency[i, j] = 1
    adjacency = sparse.csr_matrix(adjacency, dtype=adj_ind_type)

    cells = [('triangle', cells)]
    _save_mesh(output_path, points_coords, cells)

    return adjacency, centroids, areas


def structured_quads(output_path: str, bounds: tuple[int, int] = (5, 60)):
    """Generate a structured mesh of rectangles.

    Generates a structured mesh of the unit square made of identical
    rectangles. The number of rectangles is selected randomly.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    bounds : tuple[float, float], optional
        Bounds of the uniform distribution from which the numbers of elements
        per edge are sampled (default is (5, 60)).

    Returns
    -------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    centroids : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    areas : np.ndarray of float
        Areas of the cells.
    """
    # n° of elements in x and y direction:
    n_elem_edges_x = random.randint(bounds[0], bounds[1])
    n_elem_edges_y = random.randint(bounds[0], bounds[1])
    hx = 1/n_elem_edges_x
    hy = 1/n_elem_edges_y
    num_cells = int(n_elem_edges_x*n_elem_edges_y)

    areas = np.array([1/num_cells for j in range(num_cells)], dtype=float)
    centroids = np.array([[hx/2+hx*i, hy/2+hy*j]
                          for j in range(n_elem_edges_y)
                          for i in range(n_elem_edges_x)], dtype=float)

    adjacency = np.zeros((num_cells, num_cells), dtype=adj_ind_type)
    for i in range(num_cells):
        neighbours = np.array([i-1, i+1, i-n_elem_edges_x, i+n_elem_edges_x])
        if i % n_elem_edges_x == 0:
            neighbours = np.delete(neighbours, 0)  # remove left neighbour
        if i % n_elem_edges_x == n_elem_edges_x-1:
            neighbours = np.delete(neighbours, 1)  # remove right neighbour
        adjacency[i, neighbours[np.logical_and(neighbours >= 0, neighbours < num_cells)]] = 1
    adjacency = sparse.csr_matrix(adjacency, dtype=adj_ind_type)

    # saving the mesh
    points_coords = np.array([[hx*i, hy*j]
                              for j in range(n_elem_edges_y+1)
                              for i in range(n_elem_edges_x+1)])
    connectivity = np.array([[i+j*(n_elem_edges_x+1),
                              i+j*(n_elem_edges_x+1)+1,
                              i+j*(n_elem_edges_x+1)+1+n_elem_edges_x+1,
                              i+j*(n_elem_edges_x+1)+1+n_elem_edges_x]
                             for j in range(n_elem_edges_y)
                             for i in range(n_elem_edges_x)], dtype=int)
    cells = [('quad', connectivity)]
    _save_mesh(output_path, points_coords, cells)

    return adjacency, centroids, areas


def structured_tria(output_path: str, bounds: tuple[int, int] = (4, 25)):
    """Generate a structured mesh of triangles.

    Generates a structured mesh of the unit square made of identical
    triangles. The number of triangles is selected randomly.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    bounds : tuple[float, float], optional
        Bounds of the uniform distribution from which the number of triangles
        per edge is sampled (default is (4, 25)).

    Returns
    -------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    centroids : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    areas : np.ndarray of float
        Areas of the cells.
    """
    n_elem_edges = random.randint(bounds[0], bounds[1])
    h = 1/n_elem_edges
    num_cells = 2*n_elem_edges**2

    areas = np.array([1/num_cells for i in range(num_cells)])
    centroids = np.zeros((num_cells, 2))
    adjacency = np.zeros((num_cells, num_cells), dtype=adj_ind_type)

    for i in range(n_elem_edges**2):
        # compute the baricenters of the upper and the lower triangles
        centroids[2*i] = [h*(1/3+i % n_elem_edges), h*(2/3+i//n_elem_edges)]
        centroids[2*i+1] = [h*(2/3+i % n_elem_edges), h*(1/3+i//n_elem_edges)]

        neighbours_U = np.array([2*i-1, 2*i+1, 2*i+2*n_elem_edges+1])
        neighbours_L = np.array([2*i, 2*i+2, 2*i-2*n_elem_edges])
        if i % n_elem_edges == 0:
            neighbours_U = np.delete(neighbours_U, 0)  # remove left neighbour
        if i % n_elem_edges == n_elem_edges-1:
            neighbours_L = np.delete(neighbours_L, 1)  # remove right neighbour
        adjacency[2*i, neighbours_U[np.logical_and(neighbours_U >= 0, neighbours_U < num_cells)]] = 1
        adjacency[2*i+1, neighbours_L[np.logical_and(neighbours_L >= 0, neighbours_L < num_cells)]] = 1
    adjacency = sparse.csr_matrix(adjacency, dtype=adj_ind_type)

    # saving the mesh
    points_coords = np.array([[h*i, h*j]
                              for j in range(n_elem_edges+1)
                              for i in range(n_elem_edges+1)])
    connectivity = np.array([triangle_vert_ids
                             for j in range(n_elem_edges)
                             for i in range(n_elem_edges)
                             for triangle_vert_ids in (
                              [(i+j*(n_elem_edges+1)),
                               (i+j*(n_elem_edges+1))+(n_elem_edges+1)+1,
                               (i+j*(n_elem_edges+1))+(n_elem_edges+1)],
                              [(i+j*(n_elem_edges+1)),
                               (i+j*(n_elem_edges+1))+1,
                               (i+j*(n_elem_edges+1))+(n_elem_edges+1)+1]
                              )])
    cells = [('triangle', connectivity)]
    _save_mesh(output_path, points_coords, cells)

    return adjacency, centroids, areas


def voronoi_tess(output_path: str, bounds: tuple[int, int] = (50, 1000)):
    """Generate a Voronoi mesh.

    Generates a Voronoi mesh in the unit square. The number of seed points and
    their positions are selected randomly.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    bounds : tuple[int, int], optional
        Bounds of the uniform distribution from which the number of seeds
        is sampled (default is (50, 1000)).

    Returns
    -------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    centroids : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    areas : np.ndarray of float
        Areas of the cells.

    Notes
    -----
    The mesh is not the entire unit square but is roughly contained in it,
    since it is obtained by removing any cell whose centroid lies outside the
    unit square.
    """
    # generate the points of the mesh
    n_points = random.randint(bounds[0], bounds[1])
    points_coords = np.array([[random.random() for i in range(2)]
                              for j in range(n_points)])

    # compute Voronoi tesselation
    tesselation = Voronoi(points_coords)
    new_vertices = set()
    centroids, areas = [], []
    # regions only corresponding to points
    # keys: node index of the corresponding region
    regions = {i: tesselation.regions[tesselation.point_region[i]]
               for i in range(n_points)}
    for i in range(len(regions)):
        reg_nodes = regions[i]
        v = tesselation.vertices[reg_nodes]
        # extract only the finite regions: if there is a -1 in the vertices,
        # then it isn't finite
        if -1 in reg_nodes:
            del regions[i]
        else:
            # extract in bounds regions
            centroid = polygon_centroid(v)
            if (centroid[0] < 0 or centroid[0] > 1
               or centroid[1] < 0 or centroid[1] > 1):
                del regions[i]
            else:
                new_vertices = new_vertices.union(reg_nodes)
                centroids.append(centroid)
                areas.append(abs(shoelace_formula(v)))

    areas = np.array(areas)
    centroids = np.array(centroids)

    new_nodes = sorted(regions.keys())
    old_to_new = {new_nodes[i]: i for i in range(len(new_nodes))}
    num_cells = len(regions)

    adjacency = np.zeros((num_cells, num_cells), dtype=adj_ind_type)
    for ridge in tesselation.ridge_points:
        if ridge[0] in regions.keys() and ridge[1] in regions.keys():
            adjacency[old_to_new[ridge[0]], old_to_new[ridge[1]]] = 1
            adjacency[old_to_new[ridge[1]], old_to_new[ridge[0]]] = 1
    adjacency = sparse.csr_matrix(adjacency, dtype=adj_ind_type)

    # remove now unnecessary vertices and update indexing
    new_vertices = sorted(new_vertices)
    vertices = tesselation.vertices[new_vertices]
    old_to_new_vert = {new_vertices[i]: i for i in range(len(new_vertices))}

    polygons = []
    for i in new_nodes:
        new_region = [old_to_new_vert[j] for j in regions[i]]
        new_region = np.array(new_region).reshape((1, -1))
        polygons.append(('polygon', new_region))

    _save_mesh(output_path, vertices, polygons)

    return adjacency, centroids, areas


def circular_holes(output_path: str,
                   lc: float | tuple[float, float] = 0.04,
                   N: int | tuple[int, int] = 10,
                   r: float = 0.03):
    """Create a mesh with circular holes.

    Creates a triangular mesh of the unit square  including `N` circular holes
    of radius `r`.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    lc : tuple[float, float], optional
        Triangle size parameter, or bounds of the uniform distribution from
        which it is sampled. Default is 0.04.
    N : int or tuple[int, int], optional
        Number of holes in the mesh, or bounds of the uniform distribuiton
        from which it is sampled. Default is 10.
    r : float, optional
        Radius of the circular holes; default is 0.03.

    Returns
    -------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    centroids : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    areas : np.ndarray of float
        Areas of the cells.

    Notes
    -----
    To avoid intersections between the holes and with the external boundary, a
    minimum distance equal to 10% of `r` is imposed.
    To have a fairly uniform mesh, it is recommended to choose `r` at least 3
    times bigger than `lc`.
    """

    while True:
        try:
            if not gmsh.is_initialized():
                gmsh.initialize()
            gmsh.model.add(output_path)

            # generate parameters
            if isinstance(lc, tuple):
                lc = random.uniform(lc[0], lc[1])
            if isinstance(N, tuple):
                N = random.randint(N[0], N[1])

            # Generate unit square
            vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
            points = [gmsh.model.geo.addPoint(x, y, z, lc) for x, y, z in vertices]
            lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
            lines = [gmsh.model.geo.addLine(points[i], points[j]) for i, j in lines]
            square = gmsh.model.geo.addCurveLoop(lines)

            # generate the circular holes
            centers = np.empty((N, 2))
            circles = np.empty(N)
            i, missed_circles = 0, 0
            while i < N:
                centers[i] = np.array([random.uniform(0+1.1*r, 1-1.1*r),
                                       random.uniform(0+1.1*r, 1-1.1*r)])
                if i == 0 or min([np.linalg.norm(centers[i]-centers[j])
                                  for j in range(0, i)]) > 2.2*r:
                    circles[i], _ = _gmsh_circle(centers[i], r, lc)
                    i += 1
                else:
                    missed_circles += 1
                if missed_circles > N*5:
                    raise TimeoutError('Failed to fit holes into the mesh.')

            gmsh.model.geo.addPlaneSurface([square]+list(circles))
            # Generate the mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            break
        except Exception:
            gmsh.finalize()

    Adjacency, Baricenters, Areas = _get_gmsh_2D_graph_data()
    # Save the mesh
    gmsh.write(output_path)
    gmsh.finalize()

    return Adjacency, Baricenters, Areas


def holed_square(output_path: str, lc: float = 0.15, r: float = 0.25):
    """Create a square mesh with a circular hole.

    Creates a triangular mesh of the unit square with a circular hole of
    radius `r` in its center (`r` must be less than 1).

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    lc : float, optional
        Triangle size parameter. Default is 0.15.
    r : float, optional
        Radius of the circular hole; default is 0.25.

    Returns
    -------
    None
    """
    if r >= 1:
        raise ValueError(
            'Chosen radius is too big: the hole will not fit.')

    if not gmsh.is_initialized():
        gmsh.initialize()
    gmsh.model.add(output_path)

    # Generate unit square
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    points = [gmsh.model.geo.addPoint(x, y, z, lc) for x, y, z in vertices]
    lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
    lines = [gmsh.model.geo.addLine(points[i], points[j]) for i, j in lines]
    square = gmsh.model.geo.addCurveLoop(lines)

    circle_loop, circle_arcs = _gmsh_circle((0.5, 0.5), r, lc)

    for i in range(len(lines)):
        gmsh.model.geo.addPhysicalGroup(1, [lines[i]])
    gmsh.model.geo.addPhysicalGroup(1, circle_arcs)

    area = gmsh.model.geo.addPlaneSurface([square, circle_loop])
    gmsh.model.geo.addPhysicalGroup(2, [area], 1)

    # Generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Save the mesh
    gmsh.write(output_path)

    gmsh.finalize()


def double_circle(output_path: str,
                  r: float = 0.6,
                  center: float = 0.5,
                  lc: float = 0.05):
    """Create a mesh made by 2 intersecting circles.

    Creates a triangular mesh comprising 2 circles of radius `r` and with
    center respectively in (`center`, 0) and (-`center`, 0).
    `r` must be greater than `center`.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    lc : float, optional
        Triangle size parameter: default is 0.05.
    r : float, optional
        Radius of the two circles; default is 0.6.

    Returns
    -------
    None
    """
    if r <= center:
        raise ValueError(
            'Chosen radius is too small: the 2 circles will not intersect.')

    if not gmsh.is_initialized():
        gmsh.initialize()
    gmsh.model.add(output_path)

    # create the centers and arc points
    h_intersection = (r**2 - center**2)**0.5
    c1 = gmsh.model.geo.addPoint(center, 0, 0)
    c2 = gmsh.model.geo.addPoint(-center, 0, 0)
    p_up = gmsh.model.geo.addPoint(0, h_intersection, 0, lc)
    p_down = gmsh.model.geo.addPoint(0, -h_intersection, 0, lc)
    p_right = gmsh.model.geo.addPoint(center+r, 0, 0, lc)
    p_left = gmsh.model.geo.addPoint(-center-r, 0, 0, lc)

    # add circle arcs and their physical groups
    right_arc = [gmsh.model.geo.addCircleArc(p_up, c1, p_right),
                 gmsh.model.geo.addCircleArc(p_right, c1, p_down)]
    gmsh.model.geo.addPhysicalGroup(1, right_arc)

    left_arc = [gmsh.model.geo.addCircleArc(p_down, c2, p_left),
                gmsh.model.geo.addCircleArc(p_left, c2, p_up)]
    gmsh.model.geo.addPhysicalGroup(1, left_arc)

    # add outer loop and surface
    curve_loop = gmsh.model.geo.addCurveLoop(right_arc + left_arc)
    area = gmsh.model.geo.addPlaneSurface([curve_loop])
    gmsh.model.geo.addPhysicalGroup(2, [area], 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(output_path)
    gmsh.finalize()


def circular_inclusions(output_path: str,
                        lc: float | tuple = 0.15,
                        N: int | tuple = 25,
                        r: float = 0.015):
    """Create a mesh with circular inclusions.

    Creates a triangular mesh of the unit square  including `N` circular
    inclusions of radius `r`. The cinclusions have physical group equal to 1,
    while the background has it equal to 0.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    lc : tuple[float, float], optional
        Triangle size parameter, or bounds of the uniform distribution from
        which it is sampled. Default is 0.04.
    N : int or tuple[int, int], optional
        Number of holes in the mesh, or bounds of the uniform distribuiton
        from which it is sampled. Default is 10.
    r : float, optional
        Radius of the circular holes; default is 0.03.

    Returns
    -------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    centroids : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    areas : np.ndarray of float
        Areas of the cells.
    Physical_groups : np.ndarray of int
        Physical groups of the cells.

    Notes
    -----
    To avoid intersections between the inclusions and with the external
    boundary, a minimum distance equal to 10% of `r` is imposed.
    To have a fairly uniform mesh, it is recommended to choose `r` at least 3
    times bigger than `lc`.
    """

    while True:
        try:
            if not gmsh.is_initialized():
                gmsh.initialize()
            gmsh.model.add(output_path)

            # generate parameters
            if isinstance(lc, tuple):
                lc = random.uniform(lc[0], lc[1])
            if isinstance(N, tuple):
                N = random.randint(N[0], N[1])

            # Generate unit square
            vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
            points = [gmsh.model.geo.addPoint(x, y, z, lc) for x, y, z in vertices]
            lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
            lines = [gmsh.model.geo.addLine(points[i], points[j]) for i, j in lines]
            square = gmsh.model.geo.addCurveLoop(lines)

            # generate the circular holes
            centers = np.empty((N, 2))
            circles = np.empty(N)
            inclusions = np.empty(N)
            i, missed_circles = 0, 0
            while i < N:
                centers[i] = np.array([random.uniform(0+1.1*r, 1-1.1*r),
                                       random.uniform(0+1.1*r, 1-1.1*r)])
                if i == 0 or min([np.linalg.norm(centers[i]-centers[j])
                                  for j in range(0, i)]) > 2.2*r:
                    circles[i], _ = _gmsh_circle(centers[i], r, lc)
                    inclusions[i] = gmsh.model.geo.addPlaneSurface([circles[i]])
                    i += 1
                else:
                    missed_circles += 1
                if missed_circles > N*5:
                    raise TimeoutError('Failed to fit holes into the mesh.')

            area = gmsh.model.geo.addPlaneSurface([square]+list(circles))
            gmsh.model.geo.addPhysicalGroup(2, [area], 0)
            gmsh.model.geo.addPhysicalGroup(2, inclusions, 1)
            # Generate the mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            break
        except Exception:
            print('Exception occurred, restarting generation process.')
            gmsh.finalize()

    Adjacency, Baricenters, Areas = _get_gmsh_2D_graph_data()
    Physical_groups = _get_gmsh_physical_groups(element_type=2)  # triangles

    # Save the mesh
    gmsh.write(output_path)
    gmsh.finalize()
    return Adjacency, Baricenters, Areas, Physical_groups


def heterogeneous_square(output_path: str, lc: tuple[float, float] = (0.07, 0.47)):
    """Create a mesh of a heterogeneous square.

    Creates a triangular mesh of the unit square  divided into two parts along
    a line; the two parts have different physical group.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    lc : tuple[float, float], optional
        Triangle size parameter, or bounds of the uniform distribution from
        which it is sampled. Default is (0.07, 0.47).

    Returns
    -------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    centroids : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    areas : np.ndarray of float
        Areas of the cells.
    Physical_groups : np.ndarray of int
        Physical groups of the cells.
    """
    if not gmsh.is_initialized():
        gmsh.initialize()

    # Create a new model
    gmsh.model.add(output_path)

    lc = random.uniform(lc[0], lc[1])
    # Generate unit square
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (random.random(), 0, 0), (random.random(), 1, 0)]
    points = [gmsh.model.geo.addPoint(x, y, z, lc) for x, y, z in vertices]
    lines = [(0, 4), (4, 5), (5, 3), (3, 0),
             (4, 1), (1, 2), (2, 5)]
    lines = [gmsh.model.geo.addLine(points[i], points[j]) for i, j in lines]
    curve_loops = [(1, 2, 3, 4), (5, 6, 7, -2)]
    left = gmsh.model.geo.addCurveLoop(curve_loops[0])
    right = gmsh.model.geo.addCurveLoop(curve_loops[1])
    area1 = gmsh.model.geo.addPlaneSurface([left])
    area2 = gmsh.model.geo.addPlaneSurface([right])
    gmsh.model.geo.addPhysicalGroup(2, [area1], 0)
    gmsh.model.geo.addPhysicalGroup(2, [area2], 1)
    # Generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    Adjacency, Baricenters, Areas = _get_gmsh_2D_graph_data()
    Physical_groups = _get_gmsh_physical_groups(element_type=2)  # triangles

    # Save the mesh
    gmsh.write(output_path)
    gmsh.finalize()
    return Adjacency, Baricenters, Areas, Physical_groups


def dataset_2D(composition: dict[str: int],
               output_path: str,
               dataset_name: str,
               bounds: tuple[int, int],
               custom_kwargs: dict[str, dict] = None,
               extension: str = 'vtk',
               base_name: str = 'mesh',
               seed: int = None
               ) -> None:
    """Generate dataset of 2D meshes.

    Generates dataset of 2D meshes of the unit cube and portions
    of the unit cube. The meshes are saved together with a .npz file
    containing adjacency, centroid and volume data of the meshes. A text file
    containing a summary of the dataset is also saved.

    Parameters
    ----------
    !!!!!!!!!!!!!!!!!!!!!!
    output_path : str
        folder path where the dataset will be saved.
    dataset_name : str
        name of the folder containing the dataset
    extension : str, optional
        Mesh file format extension (default is 'vtk').
    base_name : str, optional
        Root name of the mesh files (default is 'mesh').
    seed : int, optional
        Seed of the Random number generator. If not provided, uses a random
        seed.

    Returns
    -------
    None

    See Also
    --------
    load_dataset : load mesh dataset.
    """

    dataset_folder = output_path+'/'+dataset_name
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    seed = _setup_rng(seed)
    dataset_size = sum(composition.values())

    # initialize log file content
    content = ('Dataset name:\t'+dataset_name+'\n'
               + 'Seed:\t'+str(seed)+'\n'
               + 'Total number of meshes:\t'+str(dataset_size))
    corr = {'structured_quads': structured_quads,
            'structured_tria': structured_tria,
            'delaunay_tria': delaunay_tria,
            'voronoi_tess': voronoi_tess,
            'circular_holes': circular_holes
            }
    custom_kwargs = {'structured_quads': {'bounds': (int(bounds[0]**(0.5)), int(bounds[1]**(0.5)))},
                     'structured_tria': {'bounds': (int((bounds[0]/2)**(0.5)), int((bounds[1]/2)**(0.5)))},
                     'delaunay_tria': {'bounds': (bounds[0]//2, bounds[1]//2)},
                     'voronoi_tess': {'bounds': (bounds[0], bounds[1])},
                     'circular_holes': {'lc': (bounds[1]**(-0.5), bounds[0]**(-0.5)), 'N': (6, 12), 'r': 0.05}
                     }

    adjacencies, coords, areas = {}, {}, {}
    base = 0

    for mesh_type, n_meshes in composition.items():
        generator = corr[mesh_type]
        n_cells = []
        for index in range(base, base + n_meshes):
            adjacencies[index], coords[index], areas[index] = generator(
                dataset_folder+'/'+base_name+str(index)+'.'+extension,
                **custom_kwargs[mesh_type])
            n_cells.append(areas[index].shape[0])
            print('Progress:\t'+str(index/dataset_size*100)+' %', end="\r", flush=True)
            # update log file
        content += ('\n\n-'+str(n_meshes)+' mesh -> '+mesh_type+' \t [from mesh '+str(base)+' to mesh '+str(base+n_meshes)+']\n'
                    + 'minimum number of cells:\t'+str(min(n_cells))+'\t\tmaximum:\t'+str(max(n_cells)))
        base += n_meshes

    # save .npz file
    print('Saving...')
    np.savez(dataset_folder+'/'+dataset_name,
             adjacency=adjacencies, coords=coords, volumes=areas)

    # save log file
    with open(dataset_folder+'/'+dataset_name+'_details.txt', 'w') as f:
        f.write(content)
        f.close()


def dataset_2D_hetero(composition: dict[str: int],
                      output_path: str,
                      dataset_name: str,
                      bounds: tuple[int, int],
                      additional_kwargs: dict[str, dict] = None,
                      extension: str = 'vtk',
                      base_name: str = 'mesh',
                      seed: int = None
                      ) -> None:
    """Generate dataset of 2D meshes.

    Generates dataset of 2D meshes of the unit cube and portions
    of the unit cube. The meshes are saved together with a .npz file
    containing adjacency, centroid and volume data of the meshes. A text file
    containing a summary of the dataset is also saved.

    Parameters
    ----------
    !!!!!!!!!!!!!!!!!!!!!!
    output_path : str
        folder path where the dataset will be saved.
    dataset_name : str
        name of the folder containing the dataset
    extension : str, optional
        Mesh file format extension (default is 'vtk').
    base_name : str, optional
        Root name of the mesh files (default is 'mesh').
    seed : int, optional
        Seed of the Random number generator. If not provided, uses a random
        seed.

    Returns
    -------
    None

    See Also
    --------
    load_dataset : load mesh dataset.
    """

    dataset_folder = output_path+'/'+dataset_name
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    seed = _setup_rng(seed)
    dataset_size = sum(composition.values())

    # initialize log file content
    content = ('Dataset name:\t'+dataset_name+'\n'
               + 'Seed:\t'+str(seed)+'\n'
               + 'Total number of meshes:\t'+str(dataset_size))
    corr = {'circular_inclusions': circular_inclusions,
            'heterogeneous_square': heterogeneous_square
            }
    custom_kwargs = {
        'circular_inclusions': {'lc': (bounds[1]**(-0.5), bounds[0]**(-0.5)),
                                'N': (6, 12),
                                'r': 0.05},
        'heterogeneous_square': {'lc': (bounds[1]**(-0.5), bounds[0]**(-0.5))}
                     }
    for mesh_type, keywords in additional_kwargs.items():
        for key, value in keywords.items():
            custom_kwargs[mesh_type][key] = value

    adjacencies, coords, areas, phys_groups = {}, {}, {}, {}
    base = 0

    for mesh_type, n_meshes in composition.items():
        generator = corr[mesh_type]
        n_cells = []
        for index in range(base, base + n_meshes):
            adjacencies[index], coords[index], areas[index], phys_groups[index] = generator(
                dataset_folder+'/'+base_name+str(index)+'.'+extension,
                **custom_kwargs[mesh_type])
            n_cells.append(areas[index].shape[0])
            print('Progress:\t'+str(round(index/dataset_size*100))+' %', end="\r")
            # update log file
        content += ('\n\n-'+str(n_meshes)+' mesh -> '+mesh_type+' \t [from mesh '+str(base)+' to mesh '+str(base+n_meshes)+']\n'
                    + 'minimum number of cells:\t'+str(min(n_cells))+'\t\tmaximum:\t'+str(max(n_cells)))
        base += n_meshes

    # save .npz file
    print('Saving...')
    np.savez(dataset_folder+'/'+dataset_name,
             adjacency=adjacencies, coords=coords, volumes=areas,
             physical_groups=phys_groups)

    # save log file
    with open(dataset_folder+'/'+dataset_name+'_details.txt', 'w') as f:
        f.write(content)
        f.close()


def _setup_rng(seed: int | None = None):
    """Setup RNG seed.

    If a seed is provided, use that one; otherwise use a random seed.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(8), byteorder="big")
    random.seed(seed)
    return seed


def _save_mesh(output_path: str, points_coords: np.ndarray, cells: list[tuple[str, np.ndarray]]) -> None:
    """Save mesh to file using `meshio`."""
    # Add third coordinate = 0 if vtk format.
    if output_path.endswith('vtk'):
        points_coords = np.pad(points_coords, ((0, 0), (0, 1)), 'constant', constant_values=0)
    mesh = meshio.Mesh(points_coords, cells)
    meshio.write(output_path, mesh)


def generate_cube(output_path: str, bounds: tuple[float, float] = (0.05, 0.20)):
    """Generate a mesh of the unit cube.

    Generates a tetrahedral mesh of the unit cube. The size of the elements is
    selected randomly.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    bounds : tuple[float, float], optional 
        Bounds of the uniform distribution from which the mesh size parameter
        is sampled. The smaller this parameter, the smaller the elements of
        the mesh will be (default is (0.05, 0.20)).

    Returns
    -------
    Adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    Coords : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    Volumes : np.ndarray of float
        Volumes of the cells.
    """

    if not gmsh.is_initialized():
        gmsh.initialize()
    # Create a new model
    gmsh.model.add(output_path)  # the name is not important

    lc1 = random.uniform(bounds[0], bounds[1])

    # Define the coordinates of the vertices
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    # Define the lines
    lines = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    # add surfaces
    surfaces = [(0, 1, 2, 3),       (4, 5, 6, 7),       (0, 9, -4, -8),
                (1, 10, -5, -9),    (2, 11, -6, -10),   (3, 8, -7, -11)]
    volumes = [(0, 1, 2, 3, 4, 5)]
    _assemble_gmsh_mesh(lc1, vertices, lines, surfaces, volumes)

    # Generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    # Save the mesh
    gmsh.write(output_path)

    # get the mesh graph data
    Adjacency, Coords, Volumes = _get_gmsh_graph_data()

    gmsh.finalize()

    return Adjacency, Coords, Volumes


def generate_cube_portion(output_path: str, bounds: tuple[float, float] = (0.03, 0.08)):
    """Generate a mesh of a portion of the unit cube.

    Generates a tetrahedral mesh of a randomly selected portion of the unit
    cube. The size of the elements is also selected randomly.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    bounds : tuple[float, float], optional
        Bounds of the uniform distribution from which the mesh size parameter
        is sampled. The smaller this parameter, the smaller the elements of
        the mesh will be (default is (0.03, 0.08)).

    Returns
    -------
    Adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    Coords : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    Volumes : np.ndarray of float
        Volumes of the cells.

    Notes
    -----
    The random portion of the unit cube is generated by randomly taking a
    vertex in each of the 8 sectors of the cube. Beacuase of this, sometimes
    the geometry is self intersecting and so we start over.
    """
    # sometimes the randomly selected points produce a self intersecting mesh;
    # in that case we start over.
    while True:
        try:
            if not gmsh.is_initialized():
                gmsh.initialize()
            # Create a new model
            gmsh.model.add(output_path)

            # Parameters for the cube
            lc1 = random.uniform(bounds[0], bounds[1])

            # Create the portion of the cube by selecting randomly the
            # vertices in the 8 sectors of the unit cube.
            vertices = [(random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5)),
                        (random.uniform(0.5, 1), random.uniform(0, 0.5), random.uniform(0, 0.5)),
                        (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0, 0.5)),
                        (random.uniform(0, 0.5), random.uniform(0.5, 1), random.uniform(0, 0.5)),
                        (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0.5, 1)),
                        (random.uniform(0.5, 1), random.uniform(0, 0.5), random.uniform(0.5, 1)),
                        (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)),
                        (random.uniform(0, 0.5), random.uniform(0.5, 1), random.uniform(0.5, 1))]

            # Define the lines
            lines = [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
            # add surfaces
            surfaces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 9, -4, -8),
                        (1, 10, -5, -9), (2, 11, -6, -10), (3, 8, -7, -11)]
            volumes = [(0, 1, 2, 3, 4, 5)]
            _assemble_gmsh_mesh(lc1, vertices, lines, surfaces, volumes)

            # Generate the mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(3)

            break  # if successfully generated, exit the loop

        except Exception:  # not clean
            gmsh.finalize()

    # Save the mesh
    gmsh.write(output_path)

    Adjacency, Coords, Volumes = _get_gmsh_graph_data()

    gmsh.finalize()

    return Adjacency, Coords, Volumes


def dataset_3D(n_cubes: int,
                        n_cubes_portions: int,
                        output_path: str,
                        dataset_name: str,
                        extension: str = 'vtk',
                        base_name: str = 'mesh',
                        seed: int = None) -> None:
    """Generate dataset of 3D meshes.

    Generates dataset of 3D tetrahedral meshes of the unit cube and portions
    of the unit cube. The meshes are saved together with a .npz file
    containing adjacency, centroid and volume data of the meshes. A text file
    containing a summary of the dataset is also saved.

    Parameters
    ----------
    n_cubes : int
        Number of meshes of the unit cube to be generated.
    n_cubes_portions : int
        Number of meshes of portions of the unit cube to be generated.
    output_path : str
        folder path where the dataset will be saved.
    dataset_name
        name of the folder containing the dataset
    extension : str, optional
        Mesh file format extension (default is 'vtk').
    base_name : str, optional
        Root name of the mesh files (default is 'mesh').
    seed : int, optional
        Seed of the Random number generator. If not provided, uses a random
        seed.

    Returns
    -------
    None

    See Also
    --------
    magnet.io.load_dataset : load mesh dataset.
    """
    dataset_folder = output_path+'/'+dataset_name
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)

    seed = _setup_rng(seed)

    adjacencies = {}
    coords = {}
    volumes = {}

    num_cells_cubes = np.zeros(n_cubes)
    num_cells_portions = np.zeros(n_cubes_portions)

    for index in range(n_cubes):
        start = time.time()

        Adj, Coor, Vol = generate_cube(dataset_folder+'/'+base_name+str(index)+'.'+extension)
        adjacencies[index] = Adj
        coords[index] = Coor
        volumes[index] = Vol
        num_cells_cubes[index] = Vol.shape[0]

        print('Mesh:', index,
              '\t\tNumber of cells:', Vol.shape[0],
              '\t\tElapsed time:', round(time.time()-start, 2), 's')

    for index in range(n_cubes, n_cubes+n_cubes_portions):
        start = time.time()

        Adj, Coor, Vol = generate_cube_portion(dataset_folder+'/'+base_name+str(index)+'.'+extension)
        adjacencies[index] = Adj
        coords[index] = Coor
        volumes[index] = Vol
        num_cells_portions[index-n_cubes] = Vol.shape[0]

        print('Mesh:', index,
              '\t\tNumber of cells:', Vol.shape[0],
              '\t\tElapsed time:', round(time.time()-start, 2), 's')

    if gmsh.is_initialized():
        gmsh.finalize()

    # save .npz file
    print('Saving...')
    np.savez(dataset_folder+'/'+dataset_name, adjacency=adjacencies,
             coords=coords, volumes=volumes)

    # save log file
    content = ('Dataset name:\t'+dataset_name+'\n'
               + 'Seed:\t'+str(seed)+'\n'
               + 'Total number of meshes:\t'+str(n_cubes+n_cubes_portions)+'\n\n'

               + '-'+str(n_cubes)+' mesh -> unit cubes  [from mesh 0 to mesh '+str(n_cubes-1)+']\n'
               + 'minimum number of tetrahedra:\t'+str(min(num_cells_cubes))+'\t\tmaximum:\t'+str(max(num_cells_cubes))+'\n\n'

               + '-'+str(n_cubes_portions)+' mesh -> portions of the unit cube  [from mesh '+str(n_cubes)+' to mesh '+str(n_cubes+n_cubes_portions-1)+']\n'
               + 'minimum number of tetrahedra:\t'+str(min(num_cells_portions))+'\t\tmaximum:\t'+str(max(num_cells_portions))
               )

    with open(output_path+'/'+dataset_name+'_details.txt', 'w') as f:
        f.write(content)
        f.close()


def generate_Heterogeneous_cube(output_path: str, N_parts: int, bounds: tuple[float, float] = (0.07, 0.47)):
    """Generate a heterogeneous mesh of the unit cube.

    Generates a tetrahedral mesh of the unit cube divided into `N_parts`
    sections with alternating physical group (0 or 1). The interfaces between
    sections are generated randomly, as well as the size of the elements.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.
    N_parts : int
        Number of parts in which to split the cube. The cube will be split
        along the z-axis (must be at least 1).
    bounds : tuple[float, float], optional
        Bounds of the uniform distribution from which the mesh size parameter
        is sampled. The smaller this parameter, the smaller the elements of
        the mesh will be (default is (0.07, 0.47)).

    Returns
    -------
    Adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the generated mesh.
    Coords : np.ndarray of float
        Centroid coordinates of each cell of the generated mesh.
    Volumes : np.ndarray of float
        Volumes of the cells.

    Notes
    -----
    The sections are generated by uniformly sampling `N_parts`-1 points on
    each edge of the cube oriented along the z-axis, and then connecting 4 of
    them at a time in the order they apper from z=0 to z=1 to create an
    interface. The mesh will thus be made of 'layers' along this direction.
    """
    if not gmsh.is_initialized():
        gmsh.initialize()

    # Create a new model
    gmsh.model.add(output_path)

    lc1 = random.uniform(bounds[0], bounds[1])

    # vertices of the model
    interface = [sorted([random.uniform(0, 1) for i in range(N_parts-1)]) for j in range(4)]
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]  # first face
    for n in range(N_parts-1):
        vertices += [(0, 0, interface[0][n]),    (1, 0, interface[1][n]),    (1, 1, interface[2][n]),    (0, 1, interface[3][n])] # interfaces in the middle
    vertices += [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]  # last face

    # define lines
    lines = []
    for j in range(N_parts):
        lines += [(4*j+i, 4*j+(i+1) % 4) for i in range(4)]  # interface loop
        lines += [(4*j+i, 4*j+i+4) for i in range(4)]  # edges connecting 2 interfaces
    lines += [(4*(N_parts) + i, 4*(N_parts) + (i+1) % 4) for i in range(4)]  # last face

    # define surface loops
    surfaces = []
    for j in range(N_parts):
        b = 8*j  # reference edge
        surfaces.append((b, b+1, b+2, b+3))
        surfaces += [(b+i, b+i+5-4*(i//3), -(b+i+8), -(b+i+4)) for i in range(4)]
    b = 8*N_parts
    surfaces.append((b, b+1, b+2, b+3))

    # define the volumes
    volumes = [(v, v+1, v+2, v+3, v+4, v+5) for v in range(0, N_parts*5, 5)]

    volume_tags = _assemble_gmsh_mesh(lc1, vertices, lines, surfaces, volumes)

    # add phisical groups
    for i in range(len(volume_tags)):
        gmsh.model.geo.addPhysicalGroup(3, [volume_tags[i]], i)

    # Generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(output_path)

    Adjacency, Coords, Volumes = _get_gmsh_graph_data()
    Physical_groups = _get_gmsh_physical_groups()
    # modify so that only 0m1 physical groups are present
    Physical_Groups = Physical_Groups % 2

    # Finalize Gmsh
    gmsh.finalize()

    return Adjacency, Coords, Volumes, Physical_groups


def generate_dataset_heterogeneous(n_cubes: dict[int:int],
                                   output_path: str,
                                   dataset_name: str,
                                   extension: str = 'vtk',
                                   base_name: str = 'mesh',
                                   seed: int = None) -> None:
    """Generate dataset of 3D heterogeneous meshes.

    Generates dataset of 3D tetrahedral meshes of the unit cube divided into
    sections with alternating physical group (0 or 1). The meshes are saved
    together with a .npz file containing adjacency, centroid, volume and
    physical group data of the meshes. A text file containing a summary of the
    dataset is also saved.

    Parameters
    ----------
    n_cubes : dict of [int : int]
        dictionary with keys the number of parts of the cube and values equal
        to the number of meshes to be generated with that number of parts.
    output_path : str
        folder path where the dataset will be saved.
    dataset_name : str
        name of the folder containing the dataset
    extension : str, optional
        Meshe file format extension (default is 'vtk').
    base_name :str, optional
        Root name of the mesh files (default is 'mesh').

    Returns
    -------
    None

    See Also
    --------
    load_dataset : load mesh dataset.
    generate_Heterogeneous_cube : heterogeneous mesh of the unit cube.
    """
    dataset_folder = output_path+'/'+dataset_name
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)

    seed = _setup_rng(seed)

    adjacencies = {}
    coords = {}
    volumes = {}
    physical_groups = {}

    # store the number of cells of each mesh
    dataset_size = np.sum(np.array([n for n in n_cubes.values()]))
    num_cells = np.zeros(dataset_size)

    base_index = 0
    for n_parts, n_meshes in n_cubes.items():
        for i in range(base_index, base_index+n_meshes): 
            start = time.time()

            adjacencies[i], coords[i], volumes[i], physical_groups[i] = generate_Heterogeneous_cube(dataset_folder+'/'+base_name+str(i)+'.'+extension, n_parts)

            num_cells[i] = volumes[i].shape[0]

            print('Mesh:', i,
                  '\t\tNumber of cells:', num_cells[i],
                  '\t\tElapsed time:', round(time.time()-start, 2),'s')
        base_index += n_meshes

    if gmsh.is_initialized():
        gmsh.finalize()

    print('Saving...')
    np.savez(dataset_folder+'/'+dataset_name, adjacency=adjacencies,
             coords=coords, volumes=volumes, physical_groups=physical_groups)

    # save log file with dataset details
    content = ('Dataset name:\t'+dataset_name+'\n'
               + 'Seed:\t'+str(seed)+'\n'
               + 'Total number of meshes:\t'+str(dataset_size)+'\n\n'

               + 'minimum number of tetrahedra:\t'+str(min(num_cells))+'\t\tmaximum:\t'+str(max(num_cells))+'\n\n'
               )
    base_index = 0
    for n_parts, n_meshes in n_cubes.items():
        content += '-'+str(n_meshes)+' mesh -> '+str(n_parts)+' portion  [from mesh '+str(base_index)+' to mesh '+str(base_index+n_meshes-1)+']\n'
        base_index += n_meshes

    with open(dataset_folder+'/'+dataset_name+'_details.txt', 'w') as f:
        f.write(content)
        f.close()


def mixed_tets_hexa(output_path: str, lc: float = 0.05):
    """Generate a mixed mesh of tetrahedra and hexahedra.
    
    Generates a mesh of the unit cube, half of tetrahedra and half of
    hexahedra, with pyramids for the transition between the two.
    
    Parameters
    ----------

    """
    if not gmsh.is_initialized():
        gmsh.initialize()
    gmsh.model.add(output_path)

    # unit cube divided in two halves, lower and upper
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    lines = [(0, 1), (1, 2), (2, 3), (3, 0),
             (0, 4), (1, 5), (2, 6), (3, 7),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (4, 8), (5, 9), (6, 10), (7, 11),
             (8, 9), (9, 10), (10, 11), (11, 8)]
    surfaces = [(0, 1, 2, 3), (0, 5, -8, -4), (1, 6, -9, -5), (2, 7, -10, -6),
                (3, 4, -11, -7), (8, 9, 10, 11), (8, 13, -16, -12), (9, 14, -17, -13),
                (10, 15, -18, -14), (11, 12, -19, -15), (16, 17, 18, 19)]
    volumes = [(0, 1, 2, 3, 4, 5), (5, 6, 7, 8, 9, 10)]

    _assemble_gmsh_mesh(lc, vertices, lines, surfaces, volumes)

    gmsh.model.geo.synchronize()

    # create the hexahedral half using transfinite meshing
    NN = int(1/lc)
    for c in [1, 2, 3, 4, 9, 10, 11, 12]:
        gmsh.model.mesh.setTransfiniteCurve(c, NN)
    for c in [5, 6, 7, 8]:
        gmsh.model.mesh.setTransfiniteCurve(c, NN//2)
    for s in range(1, 7):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.mesh.setTransfiniteVolume(1)

    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)

    gmsh.finalize()


def tetrahedra_from_stl(surface_path: str,
                        min_size: float = 0,
                        remesh: bool = False):
    """Create a tetrahedral mesh from a STL file.

    Generates tetrahedral elements starting from the surface triangulation in
    the STL using `gmsh` and then saves the new mesh.

    Parameters
    ----------
    surface_path : str
        STL file path.
    min_size : float, optional
        Minumu size of the tetrahedral elements. Default is 0 (no lower bound).
    remesh : bool, optional
        If True, remesh the STL before generating the mesh (default is False).

    Returns
    -------
    None

    Notes
    -----
    Remeshing by `gmsh` works well when traingles have similar sizes and are
    not very elongated; usually, STL coming from CAD projects are not suitable
    for this.
    """
    gmsh.initialize()
    gmsh.model.add(surface_path)
    gmsh.merge(surface_path)

    # remesh surface
    if remesh:
        gmsh.onelab.set("""[
        {
            "type":"number",
            "name":"Angle",
            "values":[40],
            "min":20,
            "max":120,
            "step":1
        }]""")

        gmsh.model.mesh.classifySurfaces(gmsh.onelab.getNumber('Angle')[0]/180*gmsh.pi, True, True, gmsh.pi)
        gmsh.model.mesh.createGeometry()

    s = gmsh.model.getEntities(2)  # get the ids of all the triangles
    surf = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.add_volume([surf])

    gmsh.model.geo.synchronize()
    # gmsh.option.setNumber('Mesh.Algorithm', 1)
    # gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size)
    gmsh.option.setNumber('Mesh.MeshSizeMin', min_size)

    gmsh.model.mesh.generate(3)

    # save in the same folder as vtk
    gmsh.write(surface_path[:-3]+'vtk')

    gmsh.fltk.run()

    gmsh.finalize()


def _gmsh_circle(c: tuple[float, float], r: float, lc: float):
    """Add 2D circle to the current Gmsh model.

    Adds a 2D circle of center `c` and radius `r` to the current model
    """
    center = gmsh.model.geo.addPoint(c[0], c[1], 0)

    p1 = gmsh.model.geo.addPoint(c[0]+r, c[1], 0, lc)
    p2 = gmsh.model.geo.addPoint(c[0], c[1]+r, 0, lc)
    p3 = gmsh.model.geo.addPoint(c[0]-r, c[1], 0, lc)
    p4 = gmsh.model.geo.addPoint(c[0], c[1]-r, 0, lc)

    a1 = gmsh.model.geo.addCircleArc(p1, center, p2)
    a2 = gmsh.model.geo.addCircleArc(p2, center, p3)
    a3 = gmsh.model.geo.addCircleArc(p3, center, p4)
    a4 = gmsh.model.geo.addCircleArc(p4, center, p1)

    return gmsh.model.geo.addCurveLoop([a1, a2, a3, a4]), [a1, a2, a3, a4]


def _get_gmsh_2D_graph_data():
    """Extract 2D mesh graph data from the model.

    Computes adjacency matrix, centroids and cell volumes of the current Gmsh
    model.

    Parameters
    ----------
    None

    Returns
    -------
    Adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the generated mesh.
    Baricenters : np.ndarray of float
        Centroid coordinates of each cell of the generated mesh.
    Volumes : np.ndarray of float
        Volumes of the cells.
    """
    # get tets and faces
    _, nodeTags = gmsh.model.mesh.getElementsByType(2, -1)
    faces_nodes = gmsh.model.mesh.getElementEdgeNodes(2, -1)
    n_tria = int(len(nodeTags)/3)
    trias = list(range(n_tria))

    # presort face nodes id:
    Faces = [tuple(sorted(faces_nodes[i:i+2])) for i in range(0, len(faces_nodes), 2)]
    # compute face x triangle incidence
    FxT = {Faces[i]: [] for i in range(len(Faces))}
    # loop over all faces
    for i in range(len(Faces)):
        tria = trias[i//3]  # every triangle has 3 faces
        FxT[Faces[i]].append(tria)

    Adjacency = np.zeros((n_tria, n_tria), dtype=adj_ind_type)
    for i in range(len(Faces)):
        tria = trias[i//3]
        for tt in FxT[Faces[i]]:
            if tt != tria:
                Adjacency[tria, tt] = 1

    Adjacency = sparse.csr_matrix(Adjacency, dtype=adj_ind_type)

    Baricenters = np.zeros((n_tria, 2))
    Volumes = np.zeros((n_tria, 1))
    _, Vertices, _ = gmsh.model.mesh.getNodesByElementType(2, -1)  # nodes of all tria
    for i in range(0, len(Vertices), 9):
        cell_vertices = np.array(Vertices[i:i+9]).reshape((3, 3))[:, :2]
        Baricenters[i//9] = tetrahedron_center(cell_vertices)
        Volumes[i//9] = tetrahedron_volume(cell_vertices)

    return Adjacency, Baricenters, Volumes


def _get_gmsh_graph_data():
    """Extract mesh graph data from the model.

    Computes adjacency matrix, centroids and cell volumes of the current Gmsh
    model.

    Parameters
    ----------
    None

    Returns
    -------
    Adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the generated mesh.
    Baricenters : np.ndarray of float
        Centroid coordinates of each cell of the generated mesh.
    Volumes : np.ndarray of float
        Volumes of the cells.
    """
    # get tets and faces
    _, nodeTags = gmsh.model.mesh.getElementsByType(4, -1)
    faces_nodes = gmsh.model.mesh.getElementFaceNodes(4, 3)
    n_tetrahedra = int(len(nodeTags)/4)
    tets = list(range(n_tetrahedra))

    # presort face nodes id:
    Faces = [tuple(sorted(faces_nodes[i:i+3])) for i in range(0, len(faces_nodes), 3)]
    # # compute face x tetrahedron incidence
    FxT = {Faces[i]: [] for i in range(len(Faces))}
    # loop over all faces
    for i in range(len(Faces)):
        tet = tets[i//4]  # every tetrahedron has 4 faces
        FxT[Faces[i]].append(tet)

    Adjacency = np.zeros((n_tetrahedra, n_tetrahedra), dtype=adj_ind_type)
    for i in range(len(Faces)):
        tet = tets[i//4]
        for tt in FxT[Faces[i]]:
            if tt != tet:
                Adjacency[tet, tt] = 1

    Adjacency = sparse.csr_matrix(Adjacency, dtype=adj_ind_type)

    Baricenters = np.zeros((n_tetrahedra, 3))
    Volumes = np.zeros((n_tetrahedra, 1))
    _, Vertices, _ = gmsh.model.mesh.getNodesByElementType(4, -1)  # nodes of all tetrahedra
    for i in range(0, len(Vertices), 12):
        cell_vertices = np.array(Vertices[i:i+12]).reshape((4, 3))
        Baricenters[i//12] = tetrahedron_center(cell_vertices)
        Volumes[i//12] = tetrahedron_volume(cell_vertices)

    return Adjacency, Baricenters, Volumes


def _get_gmsh_physical_groups(element_type=4) -> np.ndarray:
    """Extract physical group data from the model.

    Computes the physical groups vector of the current `gmsh` model. Physical
    group 0 or 1 is assigned to each physical volume of the `gmsh` model.

    Parameters
    ----------
    None

    Returns
    -------
    Physical_Groups : np.ndarray of float
        Array with shape (num_cells, 1) containing the physical group of each
        cell.

    Notes
    -----
    `gmsh` does not allow to have different volumes with same physical groups,
    so to have a non connected part of the mesh with same physical group we
    change it here.
    """
    Physical_Groups = np.zeros((len(gmsh.model.mesh.getElementsByType(element_type, -1)[0]), 1))

    for dim, physical_group in gmsh.model.getPhysicalGroups():
        # return volume tag for physical group.
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, physical_group)
        for entity_tag in entity_tags:
            cell_tags, _ = gmsh.model.mesh.getElementsByType(element_type, entity_tag)
            Physical_Groups[cell_tags-1] = physical_group  # Gmsh indexing starts from 1

    return Physical_Groups


def _assemble_gmsh_mesh(lc: float,
                        vertices: np.ndarray,
                        lines: list[tuple[int, int]],
                        surfaces: list[list[int]],
                        volumes: list[list[int]]
                        ) -> list[int]:
    """Adds all geometrical entities to the current model.

    Parameters
    ----------
    vertices: np.ndarray
        Coordinates of the vertices.
    lines: list[tuple[int, int]]
        Pairs of nodes corresponding to lines.
    surfaces: list[list[int]]
        List of line ids corresponding to surface loops.
    volumes: list[list[int]]
        List of surface ids that correspond to a closed volume.

    Returns
    -------
    list of int
        The volume tags of the generated model.
    """
    # add the points
    points = [gmsh.model.geo.addPoint(x, y, z, lc) for x, y, z in vertices]
    # add the lines
    lines = [gmsh.model.geo.addLine(points[i], points[j]) for i, j in lines]
    # add the curve loops
    surfaces = [[lines[i] if i >= 0 else -lines[-i] for i in surface] for surface in surfaces]
    curve_loops = [gmsh.model.geo.addCurveLoop(loop) for loop in surfaces]
    # Add the surfaces
    Faces = [gmsh.model.geo.addSurfaceFilling([surface_loop]) for surface_loop in curve_loops]
    # create surface loop and add volumes
    volume_tags = []
    for volume in volumes:
        surface_loop = gmsh.model.geo.addSurfaceLoop([Faces[i] for i in volume])
        volume_tags.append(gmsh.model.geo.addVolume([surface_loop]))
    return volume_tags


def view_using_gmsh_fltk(mesh_path: str) -> None:
    """Visualize a mesh using Gmsh GUI.

    Parameters
    ----------
    mesh_path : str
        Mesh file path (format must be `gmsh` compatible).

    Returns
    -------
    None
    """
    gmsh.initialize()
    gmsh.model.add(mesh_path)
    gmsh.merge(mesh_path)
    gmsh.fltk.run()
    gmsh.finalize()
