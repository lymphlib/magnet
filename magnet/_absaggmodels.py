"""Agglomeration models module.

Contains classes for agglomeration models and their abstract base classes
with the definitions of bisection algorithms.

Classes
-------
AgglomerationModel
    Abstract base class for mesh agglomeration models.
GNN
    Abstract base class for Graph Neural Networks for mesh agglomeration.
GNNHeterogeneous
    Abstract base class for Graph Neural Networks for agglomerating
    heterogeneous meshes.
KMEANS
    Kmeans clustering algorithm.
METIS
    Metis graph partitioning algorithm.
SageBase2D
    GNN with 4 Sage layers and 3 linear layers.
SageBase
    GNN with 4 Sage layers and 4 linear layers.
SageRes
    GNN that uses residual connections.
SageBaseHeterogeneous
    GNN with 4 Sage layers and 4 linear layers (for heterogeneous meshes).

Notes
-----
To define new GNN architectures, simply define a new class inheriting from
`GNN` or `GNNHeterogeneous`, defining the `__init__` and `forward` methods,
and ovverriding the `loss_function` if necessary.
"""

from abc import ABC, abstractmethod
import time
import numpy as np

from scipy.sparse.csgraph import connected_components

import torch
from torch_geometric.nn import graclus, avg_pool
from torch_geometric.data import Data
from torch_geometric.utils import one_hot, to_scipy_sparse_matrix
from torch_geometric.utils.mask import index_to_mask, mask_to_index

from .mesh import Mesh, AggMesh, AggMeshDataset
from .geometric_utils import maximum_sq_distance
from ._types import ClassList

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgglomerationModel(ABC):
    """Abstract base class for mesh agglomeration models.

    Methods
    -------
    agglomerate
        Agglomerate a mesh.
    agglomerate_dataset
        Agglomerate a dataset of meshes.
    bisect
        Bisect a mesh.
    bisection_Nref
        Bisect a mesh recursively a set number of times.
    bisection_mult_factor
        Bisect a mesh until the agglomerated elements are small enough.
    """

    def agglomerate(self,
                    mesh: AggMesh,
                    mode: str = 'Nref',
                    nref: int = 7,
                    mult_factor: float = 0.4,
                    **kwargs
                    ) -> AggMesh:
        """Agglomerate a mesh.

        Create a new mesh starting from `mesh` by agglomerating elements
        using this model.

        Parameters
        ----------
        mesh : AggMesh
            Input mesh to be agglomerated.
        mode : {'Nref, 'mult_factor'}, optional
            Agglomeration mode.
            'Nref' : bisect the mesh recursively a set number of times.
            'mult_factor' : bisect the mesh until the agglomerated elements are
            small enough.
            'segregated' : same as 'mult_factor', but bisect heterogeneous parts
            of the mesh independently.
        param : int, optional
            Number of refinements for mode 'Nref', ignored otherwise (default
            is 7).
        mult_factor : float, optional.
            Multiplicative factor for mode 'mult_factor', ignored otherwise
            (default is 0.4).

        Returns
        -------
        AggMesh
            The agglomerated mesh.

        Notes
        -----
        The adjacency matrix of the the agglomerated mesh is not computed
        (since it is expensive and in most cases a mesh will be agglomerated
        only once), so it is equal to `None`.

        See Also
        --------
        bisection_Nref
        bisection_mult_factor
        """
        match mode:
            case 'Nref':
                cl = self.bisection_Nref(mesh, nref)
            case 'mult_factor':
                cl = self.bisection_mult_factor(mesh, mult_factor)
            case 'segregated':
                cl = self.bisection_segregated(mesh, mult_factor)
            case 'multilevel':
                cl = self.multilevel_bisection(mesh, nref=nref, **kwargs)
            case 'direct_kway':
                # if isinstance(self, GNN):
                #     raise ValueError('GNN models have no direct k-way\
                #                      agglomeration available.')
                cl = self.direct_k_way(mesh, k=nref, **kwargs)
            case _:
                raise ValueError('Agglomeration mode %s does not exist.' % mode)

        agg_mesh = mesh._agglomeration(classes=cl)  # may modify

        return agg_mesh

    def agglomerate_dataset(self,
                            dataset: AggMeshDataset,
                            **kwargs
                            ) -> AggMeshDataset:
        """Agglomerate all meshes in a dataset.

        Constructs a new dataset by agglomerating the meshes in a dataset.

        Parameters
        ----------
        dataset : AggMeshDataset
            The dataset to be agglomerated.
        **kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to 'agglomerate'.

        Returns
        -------
        AggMeshDataset
            The agglomerated dataset.
        """
        output = []
        for i in range(len(dataset)):
            start = time.time()
            aggl_mesh = self.agglomerate(dataset[i], **kwargs)
            output.append(aggl_mesh)
            print('Agglomerated Mesh: ', str(i),
                  '\t\tNumber of cells:', dataset[i].num_cells,
                  '\t\tElapsed time:', round(time.time()-start, 2), 's')
        return AggMeshDataset(output, name=dataset.name+'_agglomerated')

    def coarsen(self,
                mesh: AggMesh,
                subset: np.ndarray,
                mode: str = 'Nref',
                nref: int = 7,
                mult_factor: float = 0.4
                ) -> AggMesh:
        """Coarsen a subregion of the mesh.
        """
        match mode:
            case 'Nref':
                agg_parts = self.bisection_Nref(mesh, nref, [subset])
            case 'mult_factor':
                agg_parts = self.bisection_mult_factor(mesh, mult_factor, [subset])
            case 'segregated':
                agg_parts = self.bisection_segregated(mesh, mult_factor, subset)
            case _:
                raise ValueError('Agglomeration mode %s does not exist.' % mode)

        non_agg_part = np.ones(mesh.num_cells, dtype=bool)
        non_agg_part[subset] = False
        return mesh._agglomeration(agg_parts,
                                   np.arange(mesh.num_cells)[non_agg_part])

    def bisect(self, mesh: Mesh) -> ClassList:
        """Bisect the mesh once.

        Parameters
        ----------
        mesh : Mesh
            The mesh to be bisected.

        Returns
        -------
        Classlist
            A list of 2 arrays containing the indices of the elements
            belonging to each of the 2 agglomerated elements.
        """
        return self.bisection_Nref(mesh, 1)

    def bisection_Nref(self, mesh: Mesh, Nref: int,
                       warm_start: ClassList = None) -> ClassList:
        """Bisect the mesh recursively a set number of times.

        The agglomearated mesh will have (at most) 2^`Nref` agglomerated
        elements.

        Parameters
        ----------
        mesh : Mesh
            Mesh to be bisected.
        Nref : int
            Number of times to recursively bisect the mesh.

        Returns
        -------
        Classlist
            A list of arrays, each containing the indices of the elements
            corresponding to one of the agglomerated elements.
        """
        if not isinstance(Nref, int) or Nref <= 0:
            raise ValueError('Nref must be a positive integer.')
        graph = self._get_graph(mesh)
        warm_start = [np.arange(0, mesh.num_cells)] if warm_start is None else warm_start
        result = []
        for part in warm_start:
            result.extend(self._bisection_Nref_recursive(mesh, graph, part, Nref))
        return result

    def _bisection_Nref_recursive(self,
                                  mesh: Mesh,
                                  graph,
                                  subset: np.ndarray,
                                  NREF: int,
                                  ) -> ClassList:
        """Recursive bisection algorithm.

        Parameters
        ----------
        graph : Data
            Graph representing the mesh.
        NREF : int
            Number of refinements.
        dim : int
            Spatial dimesnions (2 or 3).

        Returns
        -------
        ClassList
            A list of arrays, each containing the indices of the elements
            corresponding to one of the agglomerated elements.
        """

        if NREF >= 1 and len(subset) > 1:
            bipartition = self._bisect_subgraph(graph, subset, mesh.dim)
            subgraph_0_classes = self._bisection_Nref_recursive(
                mesh, graph, bipartition[0], NREF-1)
            subgraph_1_classes = self._bisection_Nref_recursive(
                mesh, graph, bipartition[1], NREF-1)
            # # reduce the indices of the subgraphs to those of the original one:
            # subgraph_0_classes = [bipartition[0][subgraph_0_classes[j]]
            #                       for j in range(len(subgraph_0_classes))]
            # subgraph_1_classes = [bipartition[1][subgraph_1_classes[j]]
            #                       for j in range(len(subgraph_1_classes))]
            return subgraph_0_classes + subgraph_1_classes
        else:
            # no need to bisect anymore: check for connectedness
            return self._extract_connected_comps(mesh, subset)

    # def bisection_tsize(self, mesh: Mesh, target_size: float) -> ClassList:

    def bisection_mult_factor(self, mesh: Mesh, mult_factor: float,
                              warm_start: ClassList = None
                              ) -> ClassList:
        """Bisect a mesh until the agglomerated elements are small enough.

        The mesh is bisected until all elements have a diameter that is less
        than the diameter of the entire mesh mutliplied by `mult_factor`.
        The number of agglomearated elements is thus variable.

        Parameters
        ----------
        mesh : Mesh
            Mesh to be bisected.
        mult_factor : float
            ratio between the the desired agglomerted elemnts diameter and
            that of the entire mesh. Must be between 0 and 1.

        Returns
        -------
        Classlist
            A list of arrays, each containing the indices of the elements
            corresponding to one of the agglomerated elements.

        Notes
        -----
        The implementation is iterative rather than recursive because for very
        small `mult_factor` the number of recursive calls grows fast and can
        quickly fill the RAM.
        """
        if mult_factor <= 0 or mult_factor > 1:
            raise ValueError('Multiplicative factor must be between 0 and 1.')

        graph = self._get_graph(mesh)
        target_h_sq = maximum_sq_distance(mesh.Coords) * mult_factor**2
        bisection_classes = [np.arange(0, mesh.num_cells)] if warm_start is None else warm_start
        output = []

        while bisection_classes:
            new_set = []
            for partition in bisection_classes:
                if len(partition) > 1:
                    h = maximum_sq_distance(mesh.Coords[partition])
                    if h > target_h_sq:
                        new_set.extend(self._bisect_subgraph(
                                graph, partition, mesh.dim))
                    else:
                        # check for connectedness:
                        output.extend(self._extract_connected_comps(mesh, partition))
                else:
                    output.append(partition)
            bisection_classes = new_set

        return output

    def bisection_segregated(self, mesh: Mesh, mult_factor: float,
                             subset: np.ndarray = None) -> ClassList:
        """Bisect heterogeneous mesh until elements are small enough.

        Heterogeneous parts of the mesh are bisected separately; the physical
        groups identifying different parts should be integer numbers.
        The mesh is bisected until all elements have a diameter that is less
        than the diameter of the entire mesh mutliplied by `mult_factor`.
        The number of agglomearated elements is thus variable.

        Parameters
        ----------
        mesh : Mesh
            Mesh to be bisected.
        mult_factor : float
            ratio between the the desired agglomerted elements diameter and
            that of the entire mesh. Must be between 0 and 1.

        Returns
        -------
        Classlist
            A list of arrays, each containing the indices of the elements
            corresponding to one of the agglomerated elements.

        See Also
        --------
        bisection_mult_factor
        """
        # extract heterogeneous parts
        parts = mesh._get_heterogeneus_parts(subset)
        connected_parts = []
        for part in parts:
            connected_parts.extend(self._extract_connected_comps(mesh, part))

        return self.bisection_mult_factor(mesh, mult_factor, warm_start=connected_parts)

    def multilevel_bisection(self, mesh: Mesh, refiner=None, threshold=200,
                             nref=7, using_cuda: bool = True):
        if using_cuda:
            graph = self._get_graph(mesh)
            return self._multilevel_recursive_bisection(graph, refiner, threshold, nref)
        else:
            graph = self._get_graph(mesh, device=torch.device('cpu'))
            return self._mlrb_light_cuda(graph, refiner, threshold, nref)

    def _multilevel_recursive_bisection(self, og_graph: Data, refiner, threshold=200, nref=7):
        """Strict multilevel bisection algorithm"""
        # ind = np.arange(og_graph.num_nodes)
        # coarsen graph down to the desired size
        # graph = self._get_graph(mesh)
        if nref >= 1 and og_graph.num_nodes > 1:
            # graph = og_graph.subgraph(torch.tensor(subset, device=DEVICE))
            clusters, edge_indices = [], []
            graph = og_graph
            while graph.num_nodes > threshold:
                cluster = graclus(graph.edge_index.cpu()).to(DEVICE)
                # if len(cluster) < graph.num_nodes:
                #     raise err
                edge_indices.append(graph.edge_index)
                clusters.append(cluster)
                coarser_graph = avg_pool(cluster, graph)
                graph = coarser_graph

            # partition the coarsest graph
            bool_mask = self._bisect_graph(graph)
            one_hot_part = one_hot(bool_mask.to(dtype=int))
            biparted_coarse_graph = Data(x=one_hot_part, edge_index=graph.edge_index)
            # uncoarsen and refine
            refined_cut = self._uncoarsen_and_refine(biparted_coarse_graph, clusters, edge_indices, refiner)
            refined_bool_mask = refined_cut.x[:, 0].to(dtype=torch.bool)
            # print('finished', nref)
            # recursive call
            subgraph_0_classes = self._multilevel_recursive_bisection(og_graph.subgraph(refined_bool_mask), refiner, threshold, nref-1)
            subgraph_1_classes = self._multilevel_recursive_bisection(og_graph.subgraph(~refined_bool_mask), refiner, threshold, nref-1)
            # convert result to indeces and return
            subgraph_0_classes = [mask_to_index(refined_bool_mask).cpu()[cl].numpy() for cl in subgraph_0_classes]
            subgraph_1_classes = [mask_to_index(~refined_bool_mask).cpu()[cl].numpy() for cl in subgraph_1_classes]
            return subgraph_0_classes + subgraph_1_classes
        else:
            # no need to bisect anymore: check for connectedness
            return self._extract_conn_comps(og_graph)

    def _mlrb_light_cuda(self, og_graph: Data, refiner, threshold=200, nref=7):
        """Strict multilevel bisection algorithm, but only bisection and
        refinement are on cuda (coarsening is on cpu)"""
        # ind = np.arange(og_graph.num_nodes)
        # coarsen graph down to the desired size
        # graph = self._get_graph(mesh)
        assert og_graph.x.device == torch.device('cpu')
        assert og_graph.edge_index.device == torch.device('cpu')
        if nref >= 1 and og_graph.num_nodes > 1:
            # graph = og_graph.subgraph(torch.tensor(subset, device=DEVICE))
            clusters, edge_indices = [], []
            graph = og_graph
            while graph.num_nodes > threshold:
                cluster = graclus(graph.edge_index, num_nodes=graph.num_nodes)
                # if len(cluster) < graph.num_nodes:
                #     raise err
                edge_indices.append(graph.edge_index)
                clusters.append(cluster)
                coarser_graph = avg_pool(cluster, graph)
                graph = coarser_graph

            # partition the coarsest graph on CUDA
            graph = graph.to(DEVICE)
            bool_mask = self._bisect_graph(graph)
            print('bisected: ', nref, '\tCoarse graph size: ', graph.num_nodes)
            one_hot_part = one_hot(bool_mask.to(dtype=int))
            biparted_coarse_graph = Data(x=one_hot_part, edge_index=graph.edge_index).cpu()
            # uncoarsen on CPU and refine on CUDA
            refined_cut = self._uncoarsen_and_refine(biparted_coarse_graph, clusters, edge_indices, refiner)
            refined_bool_mask = refined_cut.x[:, 0].to(dtype=torch.bool).cpu()
            # print('finished', nref)
            # recursive call
            subgraph_0_classes = self._mlrb_light_cuda(og_graph.subgraph(refined_bool_mask), refiner, threshold, nref-1)
            subgraph_1_classes = self._mlrb_light_cuda(og_graph.subgraph(~refined_bool_mask), refiner, threshold, nref-1)
            # convert result to indeces and return
            subgraph_0_classes = [mask_to_index(refined_bool_mask).cpu()[cl].numpy() for cl in subgraph_0_classes]
            subgraph_1_classes = [mask_to_index(~refined_bool_mask).cpu()[cl].numpy() for cl in subgraph_1_classes]
            return subgraph_0_classes + subgraph_1_classes
        else:
            # no need to bisect anymore: check for connectedness
            return self._extract_conn_comps(og_graph)

    def _uncoarsen_and_refine(self, biparted_coarse_graph: Data, clusters, edge_indices, refiner, **kwargs):
        while len(clusters) > 0:
            cluster = clusters.pop()
            _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
            biparted_coarse_graph.x = biparted_coarse_graph.x[inverse]
            biparted_coarse_graph.edge_index = edge_indices.pop()
            biparted_coarse_graph = refiner._refine(biparted_coarse_graph, **kwargs)
            assert biparted_coarse_graph.num_nodes > 0
        return biparted_coarse_graph

    def _extract_connected_comps(self, mesh: Mesh, subset: np.ndarray) -> ClassList:
        adj = mesh.Adjacency[subset][:, subset]
        n_connected_comps, comps = connected_components(adj, directed=False)
        if n_connected_comps > 1:
            return [subset[comps == i] for i in range(n_connected_comps)]
        else:
            return [subset]

    def _extract_conn_comps(self, graph: Data):
        adj = to_scipy_sparse_matrix(graph.edge_index)
        n_connected_comps, comps = connected_components(adj, directed=False)
        if n_connected_comps > 1:
            return [np.nonzero(comps == i)[0] for i in range(n_connected_comps)]
        else:
            return [np.arange(graph.num_nodes)]

    @abstractmethod
    def _get_graph(self, mesh: Mesh):
        raise NotImplementedError("Must override method _get_graph")

    @abstractmethod
    def _bisect_subgraph(self, graph, subset: np.ndarray, dim: int) -> ClassList:
        raise NotImplementedError("Must override method _bisect_subgraph")
