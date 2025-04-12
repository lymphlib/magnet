"""Classes for handling meshes.

Provides classes for mesh graphs, which are reduced representations of meshes
used for training of GNNs and for agglomeration algorithms, and for
agglomerable meshes.
Provides also a list-like dataset object to hold such meshes.
"""

from typing import Iterable
import time
from collections.abc import Sequence
import copy
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import networkx as nx

from magnet.cell import Cell, Polygon, Polyhedron
from magnet._types import ClassList


class Mesh():
    """Graph data that describes a mesh.

    Each cell of the mesh corresponds to a node of the graph: 2 nodes are
    connected by an edge if the corresponding cells are adjacent in the
    mesh. Each node is also characterized by its node features: coordinates
    and a area/volumes of the respective cell. This information is held in 3
    corresponding attributes (`Adjacency`, `Coords`, `Volumes`).
    Heterogeneity is described by an additional node feature,
    `Physical_Groups`.

    Used as interface for running agglomeration models and training GNNs.
    It does not contain the full data needed to describe the mesh (that is
    reserved to the `AgglomerableMeshHeterogeneous` class).

    Parameters
    ----------
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    coords : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    volumes : np.ndarray of float
        Volumes (or areas, if the mesh is 2D) of the cells.
    physical_groups : np.ndarray of float
        Physical group of each cell. Values should be between 0 and 1.

    Attributes
    ----------
    dim : int
        Spatial dimensions of the mesh (2 or 3).
    num_cells : int
        Number of cells in the mesh.
    Adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix of shape (num_cells, num_cells) describing the mesh.
        Adjacency[i, j]=1 if cells i and j are adjacent, 0 otherwise.
    Coords : np.ndarray of float
        Array with shape (num_cells, dim) containing the centroid coordinates
        of each cell of the mesh.
    Volumes : np.ndarray of float
        Array with shape (num_cells, 1) containing the volumes (or areas, if
        the mesh is 2D) of the cells.
    Physical_Groups : np.ndarray of float, optional
        Array with shape (num_cells,1) containing the physical group of each
        cell. By default, set all physical groups to 0, i.e. the mesh is
        considered homogeneous.

    See Also
    --------
    AggMesh : Agglomerable version.
    """

    def __init__(
            self,
            adjacency: sparse.csr_matrix,
            coords: np.ndarray,
            volumes: np.ndarray,
            physical_groups: np.ndarray | None = None
            ) -> None:

        self.dim = coords.shape[-1]
        self.num_cells = volumes.shape[0]

        self.Adjacency = adjacency
        self.Coords = coords
        self.Volumes = volumes.reshape((-1, 1))

        if physical_groups is None:
            self.Physical_Groups = np.zeros((self.num_cells, 1))
        else:
            self.Physical_Groups = physical_groups.reshape((-1, 1))

    def subgraph(self, node_ids: Iterable[int]) -> 'Mesh':
        """Extract submesh corresponding to given element ids.

        Parameters
        ----------
        node_ids : Iterable of int
            Iterable containing the ids of the elements that will bellong to
            the new submesh.

        Returns
        -------
        Mesh
            The new submesh.
        """
        return Mesh(self.Adjacency[node_ids][:, node_ids],
                    self.Coords[node_ids],
                    self.Volumes[node_ids],
                    self.Physical_Groups[node_ids])

    def visualize_as_graph(self,
                           classes: ClassList | None = None,
                           view_phys_groups: bool = False,
                           palette: Iterable[str] | None = None,
                           edge_color: str = "tab:gray",
                           edge_width: float = 1,
                           node_size: Iterable[float] | None = None,
                           ) -> None:
        """Visualize the graph representing the mesh.

        If a partition of the graph defined by `classes` is provided, color
        each portion differently. Alternatively, using `view_phys_groups=True`,
        each physicallly heterogeneoous part of the mesh is colored
        differently.

        Parameters
        ----------
        classes : ClassList, optional
            Graph partition (e.g. coming from an agglomeration model).
            If provided, color each set differently.
        view_phys_groups : bool, optional
            If `True`, color nodes based on physical groups; in that case,
            `classes` parameter will be ignored. Default id `False`.
        palette : str or Iterable[str], optional
            Single color for graph nodes, or color palette to be used in
            conjunction with `classes` or `view_phys_groups`. By default,
            uses `matplotlib` default color palette.
        node_size : Iterable[float], optional
            Size of each node drawn in the graph. If a single value is
            provided, draw all nodes with that size. By default, node
            dimensions scale with the corresponding element area/volume.
        edge_color : str, optional
            Color of the graph edges. Default is `tab:gray`.
        edge_width : float, optional
            Width of the graph edges.

        Returns
        -------
        None
        """
        # manage displayed colors
        if view_phys_groups:
            classes = self._get_heterogeneus_parts()

        if classes is not None:
            # only in the case of classes the full palette is used
            colors = self._assign_color_classes(classes, palette=palette)
        elif palette is None:
            colors = '#1f78b4'  # default color
        elif isinstance(colors, str | tuple[float, float, float]):
            colors = palette
        else:
            # if palette is passed with multiple items and there are no
            # classes, use first color
            colors = palette[0]

        if node_size is None:
            node_size = self.Volumes/np.max(self.Volumes)*50

        G = nx.from_scipy_sparse_array(self.Adjacency)
        if self.dim == 2:
            dic_pos = {i: self.Coords[i] for i in range(self.num_cells)}
            nx.draw(G, dic_pos, node_size=node_size, node_color=colors,
                    width=edge_width, edge_color=edge_color)
        else:
            # Extract node and edge positions from the layout
            node_xyz = np.array([self.Coords[v] for v in sorted(G)])
            edge_xyz = np.array([(self.Coords[u], self.Coords[v])
                                 for u, v in G.edges()])

            # Create the 3D figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(*node_xyz.T, s=node_size, c=colors)
            # Plot the edges
            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color=edge_color)

            ax.grid(False)
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])

            fig.tight_layout()
            plt.show()

    def _assign_color_classes(self, classes: ClassList, palette=None):
        """Assign a color to each graph portion for visualization.

        Parameters
        ----------
        classes : ClassList
            Groupings of nodes that will share the same color.
        palette : Iterable of str, optional
            Color palette to use. By default, use `matplotlib` default palette.

        Returns
        -------
        list of str
            List containing the color of each node.
        """
        # manage color palette
        if palette is None:
            palette = list(mcolors.TABLEAU_COLORS.values())
        elif isinstance(palette, str):
            palette = [palette]
        colors = ['' for i in range(self.num_cells)]
        for class_index in range(len(classes)):
            for j in classes[class_index]:
                colors[j] = palette[class_index % len(palette)]
        return colors

    def _get_heterogeneus_parts(self, subset=None) -> ClassList:
        """Extract physically heterogeneous parts of the mesh.

        Parameters
        ----------
        subset
            Eventual subset of the mesh to consider (by default consider the
            whole mesh).

        Returns
        -------
        ClassList
            Each array in the list contains the elements corresponding to a
            different physical group.
        """
        parts = {}
        subset = np.arange(self.num_cells) if subset is None else subset
        for cell_index, pg in zip(subset, self.Physical_Groups[subset]):
            if pg[0] not in parts:
                parts[pg[0]] = [cell_index]
            else:
                parts[pg[0]].append(cell_index)
        parts = [np.array(part, dtype=int) for part in parts.values()]
        return parts


class Boundary():
    """Mesh boundary structure.

    Parameters
    ----------
    Faces : list of list of int
        Faces (or edges, in 2D) that form the boundary, each deascribed by the
        Ids of its vertices.
    Tags : np.ndarray of int
        Boundary elements tags.

    Attributes
    ----------
    Faces : list of list of int
        Faces (or edges, in 2D) that form the boundary, each deascribed by the
        Ids of its vertices.
    Tags : np.ndarray of int
        Boundary elements tags. May be used to distinguish different parts of
        the boundary (e.g. for imposing boundary conditions).
    """
    def __init__(self, faces: list, tags: np.ndarray):
        self.Faces = faces
        self.Tags = tags


class AggMesh(Mesh):
    """Agglomerable mesh.

    Parameters
    ----------
    vertices : np.ndarray of float
        Coordinates of the vertices of the mesh.
    cells : list of Cell
        The connectivity data of the cells of the mesh.
    adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix describing the mesh.
    coords : np.ndarray of float
        Centroid coordinates of each cell of the mesh.
    volumes : np.ndarray of float
        Volumes (or areas, if the mesh is 2D) of the cells.
    physical_groups : np.ndarray of float
        Physical group of each cell. Values should be between 0 and 1.
    boundary : Boundary, optional
        Mesh boundary (faces that form the boundary and their tags).
        Default is `None`; it is not needed for agglomeration.

    Attributes
    ----------
    dim : int
        Spatial dimensions of the mesh (2 or 3).
    num_cells : int
        Number of cells in the mesh.
    Vertices : np.ndarray of float
        Array of shape (N, `dim`), where N is the number of total vertices of
        the mesh, containing the coordinates of the vertices.
    Cells : list of Cell
        List of length `num_cells` that contains the connectivity data of the
        cells of the mesh.
    Adjacency : sparse.csr_matrix of np.uint8
        Adjacency matrix of shape (num_cells, num_cells) describing the mesh.
        Adjacency[i, j]=1 if cells i and j are adjacent, 0 otherwise.
    Coords : np.ndarray of float
        Array with shape (num_cells, dim) containing the centroid coordinates
        of each cell of the mesh.
    Volumes : np.ndarray of float
        Array with shape (num_cells, 1) containing the volumes (or areas, if
        the mesh is 2D) of the cells.
    Physical_Groups : np.ndarray of float
        Array with shape (num_cells,1) containing the physical group of each
        cell. Values should be between 0 and 1.
    Boundary : Boundary
        Mesh boundary (faces that form the boundary and their tags).

    Notes
    -----
    This class inherits from `Mesh` for class extension and to have a common
    interface rather than overriding methods.
    """

    def __init__(
            self,
            vertices: np.ndarray = None,
            cells: list[Cell] = None,
            adjacency: sparse.csr_matrix = None,
            coords: np.ndarray = None,
            volumes: np.ndarray = None,
            physical_groups: np.ndarray = None,
            boundary: Boundary = None,
            ) -> None:

        self.Vertices = vertices
        self.Cells = cells

        self.Boundary = boundary

        super().__init__(adjacency=adjacency, coords=coords, volumes=volumes,
                         physical_groups=physical_groups)

    def _agglomeration(self,
                       classes: ClassList,
                       non_agg_part=np.zeros(0)
                       ) -> 'AggMesh':
        """Mesh agglomeration algorithm.

        Create a new mesh starting from this one by agglomerating elements as
        prescribed by `classes`.

        Parameters
        ----------
        classes : ClassList
            A list of arrays, each containing the indices of the elements
            corresponding to one of the agglomerated elements in the new mesh.
        non_agg_part : np.ndarray, optional
            Indices of elements that are not agglomerated.
        Returns
        -------
        AggMesh
            The agglomerated mesh.

        See Also
        --------
        abstractmodels.AgglomerationModel
        """
        # copy data of non agg part in a new mesh
        n_old = len(non_agg_part)
        new_elem_n = len(classes) + n_old
        new_volumes = np.empty((new_elem_n, 1))
        new_centroids = np.empty((new_elem_n, self.dim))
        new_phys_groups = np.empty((new_elem_n, 1))
        if len(non_agg_part) > 0:
            new_nodes = {n for index in non_agg_part for n in self.Cells[index].Nodes}
            new_cells = copy.deepcopy([self.Cells[index] for index in non_agg_part])
            new_volumes[:n_old] = self.Volumes[non_agg_part]
            new_centroids[:n_old] = self.Coords[non_agg_part]
            new_phys_groups[:n_old] = self.Physical_Groups[non_agg_part]
        else:
            new_nodes = set()
            new_cells = []

        for jj, new_element_ids in enumerate(classes, start=n_old):
            # the faces of the new element are the ones that are not shared by
            # two elements in the class, and each face is shared at most by 2
            # neigbouring elements.
            # NOTE: sorting the face nodes probably messes up the code when
            # the faces are not triangles.

            if self.dim == 2:
                Faces = [tuple(sorted(self.Cells[ids].Faces[i]))
                         for ids in new_element_ids
                         for i in range(len(self.Cells[ids].Faces))]

                Face_set = set()
                for face in Faces:
                    if face in Face_set:
                        Face_set.remove(face)
                    else:
                        Face_set.add(face)
            else:
                Face_dict = dict()
                for ids in new_element_ids:
                    for face in self.Cells[ids].Faces:
                        face_key = tuple(sorted(face))
                        if face_key in Face_dict:
                            del Face_dict[face_key]
                        else:
                            Face_dict[face_key] = tuple(face)
                Face_set = Face_dict.values()

            Nodes_set = {x for f in Face_set for x in f}
            new_nodes = new_nodes.union(Nodes_set)
            if self.dim == 2:
                new_cells.append(Polygon(sorted(Nodes_set), faces=list(Face_set)))
            else:
                new_cells.append(Polyhedron(sorted(Nodes_set), faces=list(Face_set)))
            new_volumes[jj] = np.sum(self.Volumes[new_element_ids])
            # the centroid is the weighted average of the original centroids:
            new_centroids[jj] = (self.Volumes[new_element_ids].T @
                                 self.Coords[new_element_ids])/new_volumes[jj]
            # we set the physical group of a new element as the weighted average:
            # new_phys_groups[jj] = (self.Volumes[new_element_ids].T @
            #                        self.Physical_Groups[new_element_ids]
            #                        )/new_volumes[jj]
            # we set the physical group of a new element as the most common one:
            values, counts = np.unique(self.Physical_Groups[new_element_ids], return_counts=True)
            new_phys_groups[jj] = values[counts.argmax()]

        # since some vertices will be removed, we need to update the indexing:
        new_nodes = sorted(new_nodes)
        Verts = self.Vertices[new_nodes]

        old_to_new_index = {new_nodes[i]: i for i in range(len(new_nodes))}
        for C in new_cells:
            C.Nodes = [old_to_new_index[C.Nodes[i]] for i in range(len(C.Nodes))]
            C.Faces = [[old_to_new_index[C.Faces[i][j]] for j in range(len(C.Faces[i]))]
                       for i in range(len(C.Faces))]
            C.MeshVertices = Verts  # was not added earlier
        if self.Boundary is not None:
            new_boundary = Boundary(
               [[old_to_new_index[F[i]] for i in range(len(F))] for F in self.Boundary.Faces],
               self.Boundary.Tags)
        else:
            new_boundary = None

        aggM = AggMesh(Verts, new_cells, coords=new_centroids,
                       volumes=new_volumes,
                       physical_groups=new_phys_groups,
                       boundary=new_boundary)
        # sort polygon nodes counterclockwise in 2D case
        if aggM.dim == 2:
            aggM._sort_counterclockwise()
        return aggM

    def _sort_counterclockwise(self, clockwise: bool = False) -> None:
        """Sort nodes of all polygons counterclockwise.

        Parametrs
        ---------
        clockwise: bool, optional
            If True, sort clockwise instead (default is False).

        Returns
        -------
        None
        """
        if self.dim != 2:
            raise ValueError('The mesh must be 2-dimensional to sort nodes.')
        for cell in self.Cells:
            cell.sort_nodes()
            if cell.is_counterclockwise() is clockwise:
                # NOTE: when loading with meshio, Nodes are numpy array
                # -> reverse() does not work.
                # TODO: make the storing of nodes consistent across the code.
                cell.Nodes.reverse()

    def view(self, figsize=(7, 7), colors=None, palette=None,
             edge_color='black', line_width=None, alpha: float = 0.5,
             view_phys: bool = False, ax: plt.Axes | None = None) -> None:
        """Plot the mesh.

        Parameters
        ----------
        figsize : tuple of [int, int]
            Size of the plot figure (default is (7, 7))
        edge_color : optional
            Color of the cell edges (default is 'black').
        alpha : float, optional
            Transparency value of the cell colors (Default is 0.5).
        view_phys : bool, optional
            If True, assign colors based on physical group of the cell
            (default is false).

        Returns
        -------
        None

        Notes
        -----
        In the 2D case, only edges are plotted and cells appear white. In the
        3D case, agglomerated elements have distinct colors, but due to
        `matplotlib` limitations some polygons in background may appear in the
        foreground instead. For 3D mesh visualization, it is recommended to use
        io.exploded_view.

        See Also
        --------
        magnet.io.exploded_view : visualize the mesh using `vtk` interactor.
        """
        # creating axes
        if ax is None:
            fig = plt.figure(figsize=figsize)
            if self.dim == 2:
                ax = fig.add_subplot()
            else:
                fig.add_subplot(projection='3d')

        # determine coloring
        if colors is None:
            if view_phys:
                parts = self._get_heterogeneus_parts()
                colors = self._assign_color_classes(parts, palette=palette)
            else:
                colors = self._assign_colors(palette=palette)
        elif isinstance(colors, str):
            colors = [colors for _ in range(self.num_cells)]

        if self.dim == 2:
            # NOTE: we are not handling non simply connected shapes. They will
            # look wrong (no holes or only an inner hole and no external
            # boundary).
            for cell_id in range(self.num_cells):
                P = patches.Polygon(self.Vertices[self.Cells[cell_id].Nodes],
                                    facecolor=colors[cell_id],
                                    edgecolor=edge_color,
                                    linewidth=line_width)
                ax.add_patch(P)
            ax.set_axis_off()
            upper_bounds = np.max(self.Vertices, axis=0)
            lower_bounds = np.min(self.Vertices, axis=0)
            center = (upper_bounds+lower_bounds)/2
            radius = 0.5*max(upper_bounds-lower_bounds)
            ax.set_xlim(center[0]-radius, center[0]+radius)
            ax.set_ylim(center[1]-radius, center[1]+radius)

        else:
            for cell_id in range(len(self.Cells)):
                for face in self.Cells[cell_id].Faces:
                    face_xyz = self.Vertices[face]
                    tri = Poly3DCollection([face_xyz])
                    tri.set_color(colors[cell_id])
                    tri.set_alpha(alpha)
                    tri.set_edgecolor(edge_color)
                    ax.add_collection3d(tri)
            # set bounding box to contain the mesh and have equal aspect ratio
            # on the three axis
            upper_bounds = np.max(self.Vertices, axis=0)
            lower_bounds = np.min(self.Vertices, axis=0)
            center = (upper_bounds+lower_bounds)/2
            radius = 0.5*max(upper_bounds-lower_bounds)
            ax.set_xlim3d(center[0]-radius, center[0]+radius)
            ax.set_ylim3d(center[1]-radius, center[1]+radius)
            ax.set_zlim3d(center[2]-radius, center[2]+radius)

        ax.grid(False)
        # plt.show()

    def _assign_colors(self, palette=None):
        """Assign colors to agglomerated elements for plotting."""
        if palette is None:
            palette = list(mcolors.TABLEAU_COLORS.values())
        return [palette[i % len(palette)] for i in range(len(self.Cells))]

    def circle_ratio(self) -> np.ndarray:
        """Compute circle ratios of all mesh elements.

        Computes (approximatively) the ratio of the diameters of the biggest
        inscribed sphere and the smallest circumscribed one for each element
        of the mesh (this value is always between 0 and 1).

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of lenght `num_cells` of the computed circle ratios.
        """
        circumscribed_diams = self.mesh_elements_sizes()
        inscribed_diams = np.array([C.inscribed_diameter(self.Coords[i])
                                    for i, C in enumerate(self.Cells)])
        return inscribed_diams/circumscribed_diams

    def area_perimeter_ratio(self) -> np.ndarray:
        """Compute compactness metric of all mesh elements.

        For 2D meshes, the area to perimeter ratio is computed; for 3D
        elements, sphericity is computed instead.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of lenght `num_cells` of the compactness metrics.
        """
        if self.dim == 2:
            # Area to Perimeter Ratio
            perimeters = np.array([C.perimeter() for C in self.Cells])
            if self.Volumes is not None:
                areas = self.Volumes.reshape(-1)
            else:
                areas = np.array([C.area() for C in self.Cells])

            return 4*np.pi*areas/np.square(perimeters)
        else:
            # Sphericity
            surf_areas = np.array([C.surface_area() for C in self.Cells])
            if self.Volumes is not None:
                volumes = self.Volumes.reshape(-1)
            else:
                volumes = np.array([C.volume() for C in self.Cells])
            return (np.pi**(1/3)*(6*volumes)**(2/3))/surf_areas

    def uniformity_factor(self) -> np.ndarray:
        """Compute uniformity factor of all mesh elements.

        Computes the ratio of the diameters of each element to that of the
        element with the largest diameter of the mesh (this value is always
        between 0 and 1).

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of lenght `num_cells` of the computed uniformity factors.
        """
        h = self.mesh_elements_sizes()
        return h/max(h)

    def volumes_difference(self, to_one: bool = False) -> np.ndarray:
        """Compute volume difference of all mesh elements.

        Computes the relative difference of the volume of each element with
        respect to the mean volume (they are positive values).

        Parameters
        ----------
        to_one: bool, optional
            If `True`, rescale the metric to [0,1] so that VD=1 implies
            that all elements have the same volume. Default is `False`.

        Returns
        -------
        np.ndarray of float
            Array of lenght `num_cells` of the computed volume differences.
        """
        Volume_target = np.mean(self.Volumes)
        VD = np.abs((self.Volumes-Volume_target)/Volume_target)
        if to_one:
            return 1/(1 + VD)
        else:
            return VD

    def mesh_elements_sizes(self) -> np.ndarray:
        """Compute the size of all elements of the mesh.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of lenght `num_cells` of element sizes.
        """
        h_vect = np.array([max(pdist(self.Vertices[self.Cells[i].Nodes]))
                          for i in range(self.num_cells)])
        return h_vect

    def get_quality_metrics(self, boxplot=False) -> np.ndarray:
        """Compute CR, UF, VD together.

        Parameters
        ----------
        boxplot : bool, optional
            If True, make a boxplot of the quality metrics (default is False).

        Returns
        -------
        np.ndarray of float
            Array of shape (`num_cells`, 3), where each column corresponds to a
            different quality metric (CR, UF, VD respectively).
        """
        CR = self.circle_ratio()
        APR = self.area_perimeter_ratio()
        UF = self.uniformity_factor()
        VD = self.volumes_difference(to_one=True)
        quality_metrics = np.column_stack([CR, APR, UF, VD])

        if boxplot:
            _boxplot_qualities(quality_metrics)

        return quality_metrics

    def get_mean_quality_metrics(self) -> np.ndarray:
        """Compute mean CR, APR, UF, VD.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of length 3 containing mean CR, UF, VD.
        """
        return np.mean(self.get_quality_metrics(), axis=0)

    def non_homogeneous_fraction(self) -> float:
        """Compute fraction of elements with discontinuities in physical group.

        Parameters
        ----------
        None

        Return
        ------
        float
            The computed fraction.

        Notes
        -----
        This is intended only for agglomerated meshes where the original
        physical groups where all either 0 or 1.
        """
        non_homogeneous = np.count_nonzero(np.logical_and(
            self.Physical_Groups > 0, self.Physical_Groups < 1))
        return non_homogeneous/self.num_cells


class MeshDataset(Sequence):
    """Dataset of mesh graphs.

    List-like object that contains meshes.

    Parameters
    ----------
    list_of_meshes : list of Mesh
        Meshes that form the dataset.
    name : str, optional
        Name of the dataset (default is None).

    Attributes
    ----------
    meshes : list of Mesh
        The meshes that form the dataset.
    name : str, optional
        Name of the dataset.

    Notes
    -----
    The implementation assumes that the dataset is always homogeneous (i.e. all
    meshes are either 2D or 3D, they are all homogeneous or heterogeneous),
    but there are no checks on this.
    """

    def __init__(self, list_of_meshes: list[Mesh], name: str = None):
        self.meshes = list_of_meshes
        self.name = name

    def __len__(self) -> int:
        """Get the size of the dataset.

        Parameters
        ----------
        none

        Returns
        -------
        int
            The number of meshes in the dataset.
        """
        return len(self.meshes)

    def __getitem__(self, index: int) -> Mesh:
        """Get a mesh from the dataset.

        Parameters
        ----------
        index : int
            Index of the desired mesh in the dataset.

        Returns
        -------
        Mesh
            The desired mesh.
        """
        return self.meshes[index]

    def add_mesh(self, mesh: Mesh) -> None:
        """Add a mesh to the dataset.

        Parameters
        ----------
        mesh : Mesh
            The mesh to be added.

        Returns
        -------
        None
        """
        self.meshes.append(mesh)

    def merge(self, other_dataset: 'MeshDataset') -> None:
        """Merge two datasets.

        Parameters
        ----------
        other_dataset : MeshDataset
            The dataset to be merged with this one.

        Returns
        -------
        None
        """
        self.meshes.extend(other_dataset.meshes)


class AggMeshDataset(MeshDataset):
    """Dataset of agglomerable meshes.

    Parameters
    ----------
    list_of_meshes : list of AggMesh
        Meshes that form the dataset.
    name : str, optional
        Name of the dataset (default is None).

    Attributes
    ----------
    meshes : list of AggMesh
        The meshes that form the dataset.
    name : str, optional
        Name of the dataset.

    Notes
    -----
    The implementation assumes that the dataset is always homogeneous (i.e. all
    meshes are either 2D or 3D, they are all homogeneous or heterogeneous),
    but there are no checks on this.
    """
    def __init__(self, mesh_list: list[AggMesh], name: str = None):
        super().__init__(mesh_list, name)

    def compute_quality_metrics(self, boxplot: bool = True) -> np.ndarray:
        """Compute average quality metrics for all meshes in the dataset.

        Parameters
        ----------
        boxplot : bool, optional
            If True, create a boxplot to of the quality metrics (default is
            True).

        Returns
        -------
        np.ndarray of float
            Array of shape (dataset_size, 3): each column corresponds to a
            mean quality metric (CR, UF, VD).
        """
        n_metrics = 4
        quality_metrics = np.zeros((len(self), n_metrics))

        for i in range(len(self)):
            start = time.time()
            quality_metrics[i] = self[i].get_mean_quality_metrics()
            print('Computed metrics for Mesh: ', str(i),
                  '\t\tNumber of cells:', self[i].num_cells,
                  '\t\tElapsed time:', round(time.time()-start, 2), 's')

        # create boxplot
        if boxplot:
            _boxplot_qualities(quality_metrics)

        return quality_metrics

    def compare_quality(self, models: list, boxplot=False, **kwargs):
        """Compare agglomeration models on the dataset.

        Agglomerates the mesh dataset with each of the given agglomeration
        models, and then compares them by computing quality metrics on the
        agglomerated meshes.

        Parameters:
        models : list of AgglomerationModel
            Models to be compared.
        **kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to
            `AggMesh.agglomerate`.

        Returns:
        list of np.ndarray of float
            The computed quality metrics for each agglomeration model.

        Notes
        -----
        It does no make sense to use Kmeans and Metis on heterogeneous domains,
        since they have no way to process this information.

        See Also
        --------
        AggMesh.agglomerate : agglomerate  the mesh.

        Examples
        --------
        >>> from magnet import aggmodels
        >>> from magnet.generate import dataset_2D
        >>> from magnet.io import load_dataset
        >>> dataset_2D({'random_delaunay': 200}, 'datasets', 'Testdataset')
        >>> dataset = load_dataset('datasets/Testdataset')
        >>> mt = aggmodels.METIS()
        >>> km = aggmodels.KMEANS()
        >>> dataset.compare_quality([mt, km], 'mult_factor', mult_factor=.3)
        """
        quality_metrics = []

        for model in models:
            model_name = model.__class__.__name__
            print('-------------Agglomerating '+model_name+'---------------')
            model_agg = model.agglomerate_dataset(self, **kwargs)
            model_quality = model_agg.compute_quality_metrics(boxplot=False)
            quality_metrics.append(model_quality)
        if boxplot:
            group_labels = ['Circle Ratio', 'Area Perimeter Ratio',
                            'Uniformity Factor', 'Volumes Difference']
            legend_labels = [model.__class__.__name__ for model in models]
            create_grouped_boxplots(quality_metrics,
                                    legend_labels=legend_labels,
                                    group_labels=group_labels)
        return quality_metrics


def _boxplot_qualities(quality_metrics: np.ndarray,
                       ax: plt.Axes | None = None):
    """Create boxplot of quality metrics."""
    # create boxplot
    if ax is None:
        _, ax = plt.subplots()
    labels = ['CR', 'APR', 'UF', 'VD']
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
    bplot = ax.boxplot(quality_metrics, patch_artist=True, labels=labels)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Quality metrics')
    ax.grid(True)


def create_grouped_boxplots(arrays,
                            colors=None,
                            title=None,
                            legend_labels=None,
                            group_labels=None,
                            label_fontsize=12,
                            widths=0.6,
                            boxplot_spacing=1,
                            groups_spacing=1,
                            ylim=1.141,
                            figsize=(18, 8)):
    """Creates grouped boxplots from a list of 2D numpy arrays.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of 2D numpy arrays (all of the same shape).
    legend_labels : list of str, optional
        List of labels for the legend corresponding to each array.
        If `None`, labels will be 'Matrix 1', 'Matrix 2', etc.
    group_labels : list of str, optional
        List of labels for the x-axis groups (columns).
        If `None`, labels will be 'Column 1', 'Column 2', etc.
    colors : list of str, optional
        List of colors corresponding to each array. If `None`, a default
        color palette is used.

    Returns
    -------
    None
    """
    # Number of matrices and columns
    N = len(arrays)
    num_columns = arrays[0].shape[1]

    # set default values for optional colors, group labels and legend labels.
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.values())[:N]
    if legend_labels is None:
        legend_labels = [f'Matrix {i+1}' for i in range(N)]

    if group_labels is None:
        group_labels = [f'Column {col+1}' for col in range(num_columns)]

    # Group data by columns
    data, positions, color_map = [], [], []

    for col in range(num_columns):
        for i, array in enumerate(arrays):
            data.append(array[:, col])
            positions.append(col*(N+1)*groups_spacing + i*boxplot_spacing + 1)
            color_map.append(colors[i])

    # Plot the boxplots
    plt.figure(figsize=figsize)

    box = plt.boxplot(data, patch_artist=True,
                      positions=positions, widths=widths)
    plt.setp(box['medians'], color='black')

    # Color the boxes
    for patch, color in zip(box['boxes'], color_map):
        patch.set_facecolor(color)

    # Set the x-ticks to be at the center of each group of boxplots
    plt.xticks([(col + 1/2) * (N + 1) * groups_spacing
                for col in range(num_columns)],
               group_labels, fontsize=label_fontsize, fontweight='bold')

    # Add legend
    for i in range(N):
        plt.plot([], c=colors[i], label=legend_labels[i])
    plt.legend(fontsize=label_fontsize)
    plt.grid(axis='y')
    # ax.xlabel('Groups')
    plt.ylabel('Values', fontsize=label_fontsize, fontweight='bold')
    plt.ylim((-0.02, ylim))
    if title is not None:
        plt.title(title)
    plt.show()
