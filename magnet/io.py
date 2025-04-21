"""IO module.

Contains functions for reading meshes, mesh datasets and for graph data
extraction (Adjacency matrix and node features computation).

`meshio` is used for reading meshes, so most of the mesh formats supported by
it (should) work fine. However, only triangular, quadrilateral, polygonal,
tetrahedral, hexahedral and pyramidal cells are supported.
"""

import os
import warnings
import numpy as np
from scipy import sparse
import scipy.io

import meshio
import vtk

from magnet.mesh import (Mesh, MeshDataset,
                         AggMesh, AggMeshDataset, Boundary)
from magnet import cell
from magnet._types import adj_ind_type


def load_mesh(mesh_path: str,
              mesh_graph: Mesh | str | None = None,
              dim: int | None = None,
              get_boundary: bool = False,
              **kwargs
              ) -> AggMesh:
    """Load mesh from file.

    Loads mesh data from file; if the mesh graph data is not provided, it is
    computed on the spot.

    Mesh graph can be given by a `Mesh` or `MeshHeterogeneous` object,
    or by a file path to a `.npz` file containing it.

    Parameters
    ----------
    mesh_path : str
        Mesh file path.
    mesh_graph : Mesh | str, optional
        `Mesh` object or path to an `.npz`file that contains the graph data of
        the mesh. If not provided, the graph data (adjacency, centroids,
        volumes) are computed.
    dim : int, optional
        Spatial dimensions of the mesh (2 or 3). By default, inferred from the
        mesh vertices coordinates. Can be provided to force a lower
        dimensional representation (this is useful beacuse some mesh formats
        only allow 3D data and so a third 0 coordinate is added to 2D meshes).
    **kwargs : dict[str, Any]
        Additional keywords arguments to pass to `extract_boundary`.

    Returns
    -------
    AggMesh
        The loaded mesh.
    """
    # load mesh
    M = meshio.read(mesh_path)

    # infer dimension
    if dim is None:
        dim = M.points.shape[-1]
        if dim == 3 and all(M.points[:, -1] == 0):
            dim = 2

    # extract cells, vertices and boundary.
    # vertices are converted to float64 (Meshio uses 'f8')
    vertices = M.points[:, :dim].astype(np.float64)
    cells = _make_cells(M, vertices, dim)
    boundary = extract_boundary(M, dim, **kwargs) if get_boundary else None

    # use given graph data (if provided):
    if mesh_graph is not None:
        if isinstance(mesh_graph, Mesh):
            return AggMesh(vertices, cells, mesh_graph.Adjacency,
                           mesh_graph.Coords, mesh_graph.Volumes,
                           mesh_graph.Physical_Groups, boundary)
        if mesh_graph.endswith('.npz'):
            # load from file
            npz_mesh = np.load(mesh_graph, allow_pickle=True)
            if 'physical_groups' in npz_mesh.files:
                return AggMesh(
                    vertices, cells,
                    npz_mesh['adjacency'].item(),
                    npz_mesh['coords'],
                    npz_mesh['volumes'],
                    npz_mesh['physical_groups'],
                    boundary)
            else:
                return AggMesh(
                    vertices, cells,
                    npz_mesh['adjacency'].item(),
                    npz_mesh['coords'],
                    npz_mesh['volumes'],
                    boundary=boundary)
        else:
            raise ValueError('Invalid mesh graph path.')
    else:
        # Compute the graph representation when not given:
        Adjacency = _compute_adjacency_matrix(cells)
        Centroids, Volumes = _extract_node_features(cells, vertices, dim)
        Physical_Groups = _extract_physical_params(M, dim)
        return AggMesh(vertices, cells, Adjacency, Centroids, Volumes,
                       Physical_Groups, boundary)


def _make_cells(mesh: meshio.Mesh, vertices: np.ndarray, dim: int):
    """Compute faces of the cells of a mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh in question.

    Returns
    -------
    list of Cell
        Cells containing cell nodes and faces.

    Notes
    -----
    Supports a limited number of `meshio` cell types: triangle, quad, polygon,
    tetra, hexahedron, pyramid.
    """
    cells = []

    for cellBlock in mesh.cells:
        match cellBlock.type:
            case 'vertex' | 'line':
                pass  # ignore these elements
            case 'triangle':
                if dim == 2:
                    for C in cellBlock.data:
                        cells.append(cell.Triangle(C, vertices))
            case'quad' | 'polygon':
                if dim == 2:
                    for C in cellBlock.data:
                        cells.append(cell.Polygon(C, vertices))
            case 'tetra':
                for C in cellBlock.data:
                    cells.append(cell.Tetrahedron(C, vertices))
            case 'hexahedron':
                for C in cellBlock.data:
                    cells.append(cell.Hexahedron(C, vertices))
            case 'pyramid':
                for C in cellBlock.data:
                    cells.append(cell.Pyramid(C, vertices))
            case _:
                raise NotImplementedError('Cell type %s not supported.'
                                          % cellBlock.type)

    return cells


def _extract_node_features(cells: list[cell.Cell],
                           vertices: np.ndarray,
                           dim: int
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Compute mesh graph node features.

    Computes the centroid coordinates and volumes of the cells of the mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh in question.
    num_cells :
        Number of cells.
    dim : int
        Spatial dimensions.

    Returns
    -------
    Centroids : np.ndarray of float
        Centroids coordinates of mesh cells.
    Volumes : np.ndarray of float
        Volumes of mesh cells.
    """
    Centroids = np.empty((len(cells), dim))
    Volumes = np.empty((len(cells), 1))
    for i, C in enumerate(cells):
        Volumes[i], Centroids[i] = C.volume_center()

    return Centroids, Volumes


def _compute_adjacency_matrix(cells: list[cell.Cell]) -> sparse.csr_matrix:
    """Compute adjacency matrix of a mesh.

    Parameters
    ----------
    cells : list of Cell
        Connectivity data of the mesh.

    Returns
    -------
    sparse.csr_matrix of np.uint8
        The adjacency matrix of the mesh.
    """

    num_cells = len(cells)
    # compute face x cell incidence
    FxT = {}
    for cell_index in range(num_cells):
        for face in cells[cell_index].Faces:
            f = tuple(sorted(face))
            if f not in FxT:
                FxT[f] = [cell_index]
            else:
                FxT[f].append(cell_index)

    # If the number of cells is low, we can use a full matrix since it's
    # faster to insert elements, but if it's high we must use a sparse matrix
    # to save memory. `lil_matrix` is used for populating it.
    if num_cells > 50000:
        Adjacency = sparse.lil_matrix((num_cells, num_cells),
                                      dtype=adj_ind_type)
    else:
        Adjacency = np.zeros((num_cells, num_cells), dtype=adj_ind_type)

    for i in range(num_cells):
        for face in cells[i].Faces:
            for j in FxT[tuple(sorted(face))]:
                if j != i:  # exlude cell itself
                    Adjacency[i, j] = 1
                    Adjacency[j, i] = 1

    return sparse.csr_matrix(Adjacency, dtype=adj_ind_type)


def _extract_physical_params(mesh: meshio.Mesh,
                             dim: int,
                             tag_name: str | None = None
                             ) -> np.ndarray:
    """Extract physical group of the mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh in question
    dim : int
        Spatial dimensions.
    tag_name : str, optional
        field name of the data to be used in 'mesh.cell_data' dictionary. If
        no name is provided, it is implied that only a single field exists and
        thus used. If multiple fields exist, always provide this argument.

    Returns
    -------
    Physical_Groups: np.ndarray of int
        The physical groups of the elements.
    """
    if not mesh.cell_data:
        return None
    if tag_name is None:
        tag_name = next(iter(mesh.cell_data_dict))
    Physical_Groups = np.empty((0, 1), dtype=int)
    for cellBlock in mesh.cells:
        d = mesh.cell_data_dict[tag_name][cellBlock.type].reshape((-1, 1))
        match cellBlock.type:
            case 'line' | 'vertex':
                pass
            case 'triangle' | 'quad' | 'polygon':
                if dim == 2:
                    Physical_Groups = np.concatenate((Physical_Groups, d))
            case 'tetra' | 'hexahedron' | 'pyramid':
                Physical_Groups = np.concatenate((Physical_Groups, d))
            case _:
                raise NotImplementedError(
                    'Cell type %s not supported.' % cellBlock.type)

    return Physical_Groups


def extract_boundary(mesh: meshio.Mesh,
                     dim: int,
                     b_tags: np.ndarray = None,
                     b_tag_name: str = None
                     ) -> Boundary:
    """Get boundary elements of the mesh.

    Extract boundary faces (or boundary edges, in 2D), and saves them in a
    separate data structure.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh in question
    dim : int
        Spatial dimensions.
    b_tags : Array_like of int, optional.
        The tags of the elements to insert in the boundary.
        By default, insert in the boundary all elements of suitable dimension
        (i.e. all lines in 2D, all triangles, quads, etc. in 3D), and assign
        them tag `1`. If the mesh does not have tag data, the function will
        behave as in the default case.
    b_tag_name : str, optional
        Field name of the tags to be used in 'mesh.cell_data' dictionary. If
        no name is provided, it is implied that only a single field exists and
        thus used. If multiple fields exist, always provide this argument.

    Returns
    -------
    boundary: list of list of int
        Connectivity data of the boundary: every term corresponds to a face
        (or edge, in 2D), described by the ids of its vertices.
    boundary_tags: np.ndarray of int
        The tags of the boundary elements.
    """
    boundary = []
    boundary_tags = np.empty((0, 1))

    # no cell data present: insert all elements with suitable dimension;
    if not mesh.cell_data:
        if b_tags is not None:
            warnings.warn('Tags were provided, but cell data was not found.'
                          + 'Inserting all elements with suitable dimension.')

        for cellBlock in mesh.cells:
            if ((dim == 3 and cellBlock.type in {'triangle', 'quad', 'polygon'})
                    or (dim == 2 and cellBlock.type == 'line')):
                boundary.extend(list(cellBlock.data))

        return Boundary(boundary, np.ones(len(boundary), dtype=int))

    # cell data is present:
    if b_tag_name is None:
        b_tag_name = next(iter(mesh.cell_data_dict))
    # compute mask of the elements to extract:
    if b_tags is None:
        def to_be_extracted(tag: int): return True
    else:
        b_tags = np.array(b_tags)

        def to_be_extracted(tag: int):
            return any(b_tags == tag)

    for cellBlock in mesh.cells:
        block_tags = mesh.cell_data_dict[b_tag_name][cellBlock.type]

        if ((dim == 3 and cellBlock.type in {'triangle', 'quad', 'polygon'})
                or (dim == 2 and cellBlock.type == 'line')):

            mask = [to_be_extracted(block_tags[i]) for i in range(len(block_tags))]
            boundary.extend(list(cellBlock.data[mask]))
            boundary_tags = np.concatenate((boundary_tags,
                                            block_tags[mask].reshape((sum(mask), 1))))

    return Boundary(boundary, boundary_tags)


def load_dataset(dataset_path: str,
                 extension: str = 'vtk',
                 base_name: str = 'mesh',
                 ) -> AggMeshDataset:
    """Load mesh dataset from folder.

    The folder should contain the mesh file plus the .npz file (with the same
    name of the folder) containing the adjacency, centroids and volume data.
    Meshes must be named progressively starting from '`base_name`0.`extension`'
    (e.g. 'mesh0.vtk').

    Parameters
    ----------
    dataset_path : str
        Path of the folder that contains the dataset.
    extension : str, optional
        Mesh format extension of the dataset to be loaded (default is 'vtk')
    base_name : str, optional
        Base name of the meshes (default is 'mesh')

    Returns
    -------
    AgglomerableMeshDataset
        Dataset containing the loaded meshes.

    See Also
    --------
    generate2D.generate_2D_dataset : Generate dataset of 2D meshes.
    generate3D.generate_dataset_3D : Generate dataset of 3D meshes.

    Examples
    --------
    >>> from magnet.generate import dataset_3D
    >>> dataset_3D(5, 5, 'datasets', 'Testdataset')
    >>> dataset = load_dataset('datasets/Testdataset')
    Name:       Testdataset
    Dimension:  10
    """
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    npz_dataset = np.load(dataset_path+'/'+dataset_name+'.npz',
                          allow_pickle=True)
    dataset_size = len(npz_dataset['adjacency'].item())

    mesh_list = []
    adjacency = npz_dataset['adjacency'].item()
    coords = npz_dataset['coords'].item()
    volumes = npz_dataset['volumes'].item()
    if 'physical_groups' in npz_dataset.files:
        physical_groups = npz_dataset['physical_groups'].item()
    else:
        physical_groups = [None for i in range(dataset_size)]

    for index in range(dataset_size):
        M = meshio.read(dataset_path+'/'+base_name+str(index)+'.'+extension)
        dim = coords[index].shape[-1]
        verts = M.points[:, :dim].astype(np.float64)
        cells = _make_cells(M, verts, dim)
        mesh_list.append(AggMesh(vertices=verts, cells=cells,
                                 adjacency=adjacency[index],
                                 coords=coords[index],
                                 volumes=volumes[index],
                                 physical_groups=physical_groups[index]))

    print('Name:\t\t', dataset_name)
    print('Dimension:\t', dataset_size)

    return AggMeshDataset(mesh_list, dataset_name)


def load_graph_dataset(dataset_path: str) -> MeshDataset:
    """Load mesh graph dataset.

    Loads the adjacency, centroids, volumes (and physical groups, if present)
    of a mesh dataset from a `.npz` file.

    This is intended mainly for training GNNs, as the dataset does not contain
    the full connectivity data of the meshes.

    Parameters
    ----------
    dataset_path : str
        Path of the `.npz` file containing the datset, or folder path of the
        same name containing said file.

    Returns
    -------
    MeshDataset
        Loaded dataset.

    See Also
    --------
    load_mat_dataset : Load mesh graph dataset from MATLAB files.
    generate.2D_dataset : Generate dataset of 2D meshes.
    generate.dataset_3D : Generate dataset of 3D meshes.
    """
    # get the name of dataset and load it.
    if dataset_path.endswith('.npz'):
        dataset_name = os.path.basename(dataset_path)[:-4]
        npz_dataset = np.load(dataset_path, allow_pickle=True)
    else:
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        npz_dataset = np.load(dataset_path+'/'+dataset_name+'.npz',
                              allow_pickle=True)
    dataset_size = len(npz_dataset['adjacency'].item())

    adjacency = npz_dataset['adjacency'].item()
    coords = npz_dataset['coords'].item()
    volumes = npz_dataset['volumes'].item()
    if 'physical_groups' in npz_dataset.files:  # there is a physical group
        physical_groups = npz_dataset['physical_groups'].item()
    else:
        physical_groups = [None for i in range(dataset_size)]

    mesh_list = [Mesh(adjacency=adjacency[i],
                      coords=coords[i], volumes=volumes[i],
                      physical_groups=physical_groups[i])
                 for i in range(dataset_size)]

    print('Name:\t\t', dataset_name)
    print('Dimension:\t', dataset_size)

    return MeshDataset(mesh_list, dataset_name)


def load_mat_dataset(dataset_path: str) -> MeshDataset:
    """Load mesh graph dataset from MATLAB files.

    Loads the adjacency, centroids, volumes of a mesh dataset from a `.mat`
    files. The dataset folder should contain 3 `.mat` files named
    'AdjacencyMatrices', 'CoordMatrices', 'AreaVectors' with fields of the
    same name.

    This is intended mainly for training GNNs, as the dataset does not contain
    the full connectivity data of the meshes.

    Parameters
    ----------
    dataset_path : str
        Path of the folder containing file containing the datset, or folder
        path of the same name containing said file.

    Returns
    -------
    MeshDataset
        Loaded dataset.

    See Also
    --------
    load_graph_dataset : Load mesh graph dataset from `.npz` file.
    """
    adjacencies = np.squeeze(scipy.io.loadmat(
        dataset_path+'/AdjacencyMatrices.mat')['AdjacencyMatrices'], 0)
    coords = np.squeeze(scipy.io.loadmat(
        dataset_path+'/CoordMatrices.mat')['CoordMatrices'], 0)
    areas = np.squeeze(scipy.io.loadmat(
        dataset_path+'/AreaVectors.mat')['AreaVectors'], 0)

    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    dataset_size = areas.shape[-1]

    output = [Mesh(
              adjacency=sparse.csr_matrix(adjacencies[i], dtype=adj_ind_type),
              coords=coords[i],
              volumes=areas[i])
              for i in range(dataset_size)]

    print('Name:\t\t', dataset_name)
    print('Dimension:\t', dataset_size)

    return MeshDataset(output, dataset_name)


def create_dataset(dataset_path: str,
                   dataset_size: int,
                   extension: str = 'vtk',
                   base_name: str = 'mesh',
                   **kwargs
                   ) -> None:
    """Create dataset from mesh folder.

    Parameters
    ----------
    dataset_path : str
        Path of the folder that contains the meshes.
    dataset_size : int
        Number of meshes in the folder.
    extension : str, optional
        Mesh format extension (default is 'vtk').
    base_name : str, optional
        Base name of the meshes (default is 'mesh').
    **kwargs : dict[str, Any]
        Additional keyword arguments to pass to 'load_mesh'.

    Returns
    -------
    None

    Notes
    -----
    The type of mesh (heterogeneous or not) is inferred from the type of the
    first mesh.
    """
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    n_cells = np.zeros(dataset_size)
    adjacencies, coords, volumes, physical_groups = {}

    for i in range(dataset_size):
        M = load_mesh(dataset_path+'/'+base_name+str(i)+'.'+extension,
                      **kwargs)

        adjacencies[i] = M.Adjacency
        coords[i] = M.Coords
        volumes[i] = M.Volumes
        physical_groups[i] = M.Physical_Groups
        n_cells[i] = volumes[i].shape[0]

    print('Saving...')
    np.savez(dataset_path+'/'+dataset_name,
             adjacency=adjacencies, coords=coords, volumes=volumes)

    # save log file
    content = ('Dataset name:\t'+dataset_name+'\n'
               + 'Total number of meshes:\t'+str(dataset_size)+'\n'
               + 'minimum number of elements:\t'+str(min(n_cells))
               + '\t\tmaximum:\t'+str(max(n_cells))
               )

    with open(dataset_path+'/'+dataset_name+'_details.txt', 'w') as f:
        f.write(content)
        f.close()


def save_mesh(self: AggMesh, output_path: str) -> None:
    """Save mesh to file.

    Parameters
    ----------
    output_path : str
        File path where the mesh will be saved.

    Returns
    -------
    None

    Notes
    -----
    In the 3D case, `vtkXMLUnstructuredGridWriter` is used to write to
    file, so use the XML Unstructured Grid Reader to read the mesh.
    """
    if self.dim == 2:
        # pad points to 3D with 0 third coordinate
        points = np.pad(self.Vertices, ((0, 0), (0, 1)),
                        'constant', constant_values=0)
        # insert polygonal cells
        polygons = [('polygon', np.array(C.Nodes).reshape((1, -1)))
                    for C in self.Cells]
        CellEntityIds = [self.Physical_Groups[i].reshape((1, 1))
                         for i in range(self.num_cells)]
        # insert cell data corresponding to physical groups
        cell_data = {'CellEntityIds': CellEntityIds}
        # insert boundary edges (if any)
        if self.Boundary is not None:
            boundary_edges = np.array(self.Boundary.Faces)
            polygons.append(('line', boundary_edges))
            cell_data['CellEntityIds'].append(self.Boundary.Tags)

        output_mesh = meshio.Mesh(points=points,
                                  cells=polygons,
                                  cell_data=cell_data)
        meshio.write(output_path, output_mesh)
    else:
        # insert vertices coordinates
        points = vtk.vtkPoints()
        for vertex in self.Vertices:
            points.InsertNextPoint(*vertex)
        # create the grid and assign the points to it
        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        # insert each polyhedral cell
        for C in self.Cells:
            faceId = vtk.vtkIdList()
            faceId.InsertNextId(len(C.Faces))
            for face in C.Faces:
                faceId.InsertNextId(len(face))
                [faceId.InsertNextId(i) for i in face]
            ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, faceId)

        # Create a cell data array
        cellData = vtk.vtkFloatArray()
        cellData.SetNumberOfComponents(1)
        cellData.SetName("CellEntityIds")

        # Assign some data to the cells
        for cell_pg in self.Physical_Groups:
            cellData.InsertNextValue(cell_pg)

        # insert boundary faces (if present)
        if self.Boundary is not None:
            for face in self.Boundary.Faces:
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(len(face))
                for j, n in enumerate(face):
                    polygon.GetPointIds().SetId(j, n)
            for boundary_tag in self.Boundary.Tags:
                cellData.InsertNextValue(boundary_tag)

        ugrid.GetCellData().AddArray(cellData)

        # write out the mesh
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(ugrid)
        writer.SetFileName(output_path)
        writer.Update()


def save_graph(mesh: Mesh, output_path: str) -> None:
    """Save mesh graph data to file.

    Parameters
    ----------
    mesh : Mesh
        Mesh graph to be saved.
    output_path : str
        File path where the mesh will be saved.

    Returns
    -------
    None
    """
    if mesh.Adjacency is None and isinstance(mesh, AggMesh):
        mesh.Adjacency = _compute_adjacency_matrix(mesh.Cells)
    if np.array_equal(mesh.Physical_Groups, np.zeros((mesh.num_cells, 1))):
        np.savez(output_path, adjacency=mesh.Adjacency,
                 coords=mesh.Coords, volumes=mesh.Volumes)
    else:
        np.savez(output_path, adjacency=mesh.Adjacency,
                 coords=mesh.Coords, volumes=mesh.Volumes,
                 physical_groups=mesh.Physical_Groups)


def exploded_view(mesh: AggMesh,
                  scale: float = 1,
                  palette: str | list | None = None,
                  background: str = 'white',
                  edge_color: str | None = None,
                  edge_width: int = 1,
                  orientation: tuple[int, int, int] = (0, 0, 0),
                  figsize: tuple[int, int] = (800, 600),
                  save_image_path: str | None = None,
                  image_scaling: int = 2,
                  title: str = None):
    """Visualize exploded mesh using `vtk` renderer.

    Creates a `vtk` window interactor displaying the mesh, in which each
    element has been translated by the distance of its centroid from the
    center of the mesh bounding box, multiplied by `scale` so that it appears
    exploded. If `scale=0` visualizes the original mesh.
    Additionally, allows to save a png image of the rendered window.

    Parameters
    ----------
    mesh : AggMesh
        Mesh to visualize.
    scale : float, optional
        Factor that describes the 'intensity' of the explosion. If set to 0,
        visualizes the original mesh. default is 1.
    palette : str or list of str, optional
        Color palette for mesh elements, or single color for all of them.
        If `None`, uses a default color palette.
    background : str, optional
        Background color. Default is `'white'`.
    edge_color : str, optional
        Color of the edges of the faces. If `None`, they are not visualized.
    edge_width : int, optional
        Width of the edges of the faces in pixels. Default is 1.
    orientation : tuple[int, int, int], optional
        Initial angles of rotation (pitch, roll, ) of the object, expressed in
        degrees.
    figsize : tuple[int, int], optional
        Size of the renderer window. Default is (800, 600).
    save_image_path: str, optional
        Path where the png image will be saved. If not given, the image will
        not be saved.
    image_scaling: int, optional
        Scale image factor when exporting to png. For example, 2 means that
        the image size is doubled.
    title : str, optional
        Title of the displayed window

    Returns
    -------
    None
    """
    if mesh.dim == 2:
        raise ValueError('Mesh should be 3D.')

    # Define the central point as center of the bounding box
    upper_bounds = np.max(mesh.Vertices, axis=0)
    lower_bounds = np.min(mesh.Vertices, axis=0)
    center = (upper_bounds+lower_bounds)/2

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName('Colors')
    # Assign a random color to the cell

    if isinstance(palette, str):
        color = vtk.vtkNamedColors().GetColor3d(palette)
        # convert to 8-bit
        color = tuple(int(round(c * 255)) for c in color)
        for i in range(mesh.num_cells):
            colors.InsertNextTuple(color)
    else:
        if palette is None:
            palette = [(66, 134, 244),   # Light Blue
                       (244, 67, 54),    # Red
                       (76, 175, 80),    # Green
                       (255, 193, 7),    # Amber
                       (156, 39, 176),   # Purple
                       (0, 188, 212),    # Cyan
                       (233, 30, 99),    # Pink
                       (63, 81, 181),    # Indigo
                       (139, 195, 74),   # Light Green
                       (255, 152, 0),    # Orange
                       (103, 58, 183),   # Deep Purple
                       (3, 169, 244),    # Light Cyan
                       (255, 87, 34),    # Deep Orange
                       (205, 220, 57),   # Lime
                       (0, 150, 136),    # Teal
                       (121, 85, 72),    # Brown
                       (255, 235, 59),   # Yellow
                       (158, 158, 158),  # Grey
                       (96, 125, 139),   # Blue Grey
                       (244, 143, 177),  # Light Pink
                       (129, 212, 250),  # Light Blue
                       (197, 202, 233),  # Light Indigo
                       (100, 221, 23),   # Light Green
                       (255, 111, 0)     # Vivid Orange
                       ]
            for i in range(mesh.num_cells):
                colors.InsertNextTuple(palette[i % len(palette)])

    # Create a new vtkUnstructuredGrid to store the exploded mesh
    exploded_mesh = vtk.vtkUnstructuredGrid()
    exploded_mesh.Allocate(mesh.num_cells)
    points = vtk.vtkPoints()

    # Insert each translated polyhedral cell
    for i, C in enumerate(mesh.Cells):

        # compute translated points
        translation_vector = scale*(mesh.Coords[i] - center)
        new_cell_points = vtk.vtkPoints()
        for point in C.Nodes:
            new_point = mesh.Vertices[point] + translation_vector
            new_cell_points.InsertNextPoint(*new_point)

        # Add the points to the points list
        point_ids = [points.InsertNextPoint(new_cell_points.GetPoint(j))
                     for j in range(new_cell_points.GetNumberOfPoints())]

        # insert new polyhedral cell
        old_to_new = {C.Nodes[i]: point_ids[i] for i in range(len(C.Nodes))}
        faceId = vtk.vtkIdList()
        faceId.InsertNextId(len(C.Faces))
        for face in C.Faces:
            faceId.InsertNextId(len(face))
            [faceId.InsertNextId(old_to_new[i]) for i in face]

        exploded_mesh.InsertNextCell(vtk.VTK_POLYHEDRON, faceId)

    # Set the points and colors for the exploded mesh
    exploded_mesh.SetPoints(points)
    exploded_mesh.GetCellData().SetScalars(colors)

    # Visualize the exploded mesh
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(exploded_mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Set edge visibility and color
    if edge_color is not None:
        actor.GetProperty().SetEdgeVisibility(1)
        actor.GetProperty().SetEdgeColor(
            vtk.vtkNamedColors().GetColor3d(edge_color))
        actor.GetProperty().SetLineWidth(edge_width)

    # Set initial rotation of the object (pitch, yaw, roll in degrees)
    actor.SetOrientation(*orientation)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    renderer.AddActor(actor)
    renderer.SetBackground(vtk.vtkNamedColors().GetColor3d(background))
    render_window.SetSize(*figsize)
    if title is not None:
        render_window.SetWindowName(title)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.Render()

    # Save the rendered image to a file
    if save_image_path is not None:
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.SetScale(image_scaling)
        window_to_image_filter.SetInputBufferTypeToRGBA()
        window_to_image_filter.ReadFrontBufferOff()
        window_to_image_filter.Update()

        image_writer = vtk.vtkPNGWriter()
        image_writer.SetFileName(save_image_path)
        image_writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        image_writer.Write()

    render_window_interactor.Start()
