# kmeans

import numpy as np
from sklearn.cluster import KMeans

from ..mesh import Mesh
from .._absaggmodels import AgglomerationModel
from .._types import ClassList


class KMEANS(AgglomerationModel):
    """Kmeans algorithm applied to mesh agglomeration."""

    def _get_graph(self, mesh: Mesh):
        graph = mesh.Coords
        return graph

    def _bisect_subgraph(self, graph: np.ndarray, subset: np.ndarray,
                         dim: int = None) -> ClassList:
        """Call Kmeans on the mesh to bisect it once.

        Parameters
        ----------
        coords : np.ndarray of float
            The graph to be bisected.
        dim : int
            Spatial dimensions of the mesh (2 or 3).

        Returns
        -------
        ClassList
            A list of 2 arrays, each containing the indices of the elements
            corresponding to one of the two parts.
        """
        parts = KMeans(2).fit_predict(graph[subset])
        groups = [subset[parts == i] for i in range(2)]
        return groups

    def direct_k_way(self, mesh: Mesh, k: int) -> ClassList:
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
        parts = KMeans(k).fit_predict(mesh.Coords)
        groups = [np.nonzero(parts == i)[0] for i in range(k)]
        # connectedness check
        res = []
        for group in groups:
            res.extend(self._extract_connected_comps(mesh, group))
        return res
