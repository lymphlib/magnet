# metis

import numpy as np
import networkx as nx
import metispy as metis

from ..mesh import Mesh
from .._absaggmodels import AgglomerationModel
from .._types import ClassList


class METIS(AgglomerationModel):
    """Metis algorithm for graph partitioning."""

    def _get_graph(self, mesh: Mesh) -> nx.Graph:
        # construct the networkx graph
        G = nx.from_scipy_sparse_array(mesh.Adjacency)
        Vmin = min(mesh.Volumes)
        for i in range(G.number_of_nodes()):    # node weights = volumes
            v = np.int64(mesh.Volumes[i]/Vmin)   # weights must be integers
            G.nodes[i]['volume'] = v[0]
            # G.nodes[i]['Volume_float'] = Volumes[i]
        G.graph['node_weight_attr'] = 'volume'
        return G
    
    def _get_graph_unitary_weights(self, mesh: Mesh) -> nx.Graph:
        G = nx.from_scipy_sparse_array(mesh.Adjacency)
        return G

    def _bisect_subgraph(self, graph: nx.Graph, subset: np.ndarray,
                         dim: int = None) -> ClassList:
        """Call Metis on the mesh graph to bisect it once.

        Parameters
        ----------
        graph : nx.Graph
            The graph to be bisected.
        mesh_coords: np.ndarray
            Coordinates of elements centroids.

        Returns
        -------
        ClassList
            A list of 2 arrays, each containing the indices of the elements
            corresponding to one of the two parts.
        """
        graph = graph.subgraph(subset)
        _, parts = metis.part_graph(graph, 2, contig=True)
        # when extracting a nx.subgraph, the order of the nodes is NOT
        # preserved, so we have to take it into account.
        parts = np.array(parts)
        new_nodes_order = np.array(graph.nodes)
        groups = [new_nodes_order[parts == i] for i in range(2)]

        return groups

    def direct_k_way(self, mesh: Mesh, k: int, volume_weights: bool = True) -> ClassList:
        """Bisect the mesh recursively a set number of times.

        The agglomearated mesh will have (at most) 2^`Nref` agglomerated
        elements.

        Parameters
        ----------
        Mesh : mesh
            Mesh to be bisected.
        Nref : int
            Number of times to recursively bisect the mesh.
        volume_weights : bool, optional
            If true, use volume as node weights, otherwise use unitary weights.

        Returns
        -------
        Classlist
            A list of arrays, each containing the indices of the elements
            corresponding to one of the agglomerated elements.
        """
        if volume_weights:
            G = self._get_graph(mesh)
        else:
            G = self._get_graph_unitary_weights(mesh)

        # Partition with metis; contig ensures connected components.
        _, parts = metis.part_graph(G, k, contig=True)
        parts = np.array(parts)
        groups = [np.nonzero(parts == i)[0].astype(np.int64) for i in range(k)]

        return groups
