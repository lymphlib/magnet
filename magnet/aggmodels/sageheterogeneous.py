# sageheterogeneous

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import (
    degree,
    from_scipy_sparse_matrix,
    add_self_loops,
    scatter,
)
from torch_geometric.nn.conv import SAGEConv

from magnet.mesh import Mesh
from magnet.aggmodels.gnn import GeometricGNN
from magnet.graph_utils import align_to_x_axis
from magnet._absaggmodels import DEVICE


class GNNHeterogeneous(GeometricGNN):
    """Abstract base class for GNNs for agglomerating heterogeneous meshes."""

    def loss_function(self, y: torch.Tensor, graph: Data, coeff=1) -> torch.Tensor:
        """Loss function used during training.

        Parameters
        ----------
        y : torch.Tensor
            Evaluation output of the Neural Network.
        graph : Data
            Graph on which the GNN was evaluated.

        Returns
        -------
        torch.Tensor
            The value of the loss function.

        Notes
        -----
        The loss function may be overridden in subclasses to customize the GNN.
        See the `losses` module for the actual definitions.
        """
        d = degree(graph.edge_index[0], num_nodes=y.size(0))
        gamma = torch.t(y) @ d
        c = torch.sum(y[graph.edge_index[0], 0] * y[graph.edge_index[1], 1])
        loss1 = torch.sum(torch.div(c, gamma))
        loss2 = (
            torch.sum(y[:, 0] * graph.x[:, -1])
            + torch.sum(y[:, 1] * (1 - graph.x[:, -1]))
        ) / graph.num_nodes
        return (loss1 + coeff * loss2).to(DEVICE)

    def get_sample(
        self,
        mesh: Mesh,
        randomRotate=False,
        selfloop=False,
        device=DEVICE,
    ) -> Data:
        """Returns a graph data structure sample for training.

        Parameters
        ----------
        mesh : Mesh
            Heterogeneous mesh to be sampled.
        randomRotate : bool, optional
            If True, randomly rotate the mesh (default is `False`).
        selfloop : bool, optional
            if True, add 1 on the diagonal of the adjacency matrix, i.e
            self-loops on the graph (default is `False`).

        Returns
        -------
        Data
            graph data representing the mesh.
        """
        coords_sample = torch.tensor(mesh.Coords, dtype=torch.float, device=device)
        volumes_sample = torch.tensor(mesh.Volumes, dtype=torch.float, device=device)
        physical_groups_sample = torch.tensor(
            mesh.Physical_Groups, dtype=torch.float, device=device
        )

        if randomRotate:
            coords_sample = self._randomrotate(coords_sample)
        x = torch.cat([coords_sample, volumes_sample, physical_groups_sample], -1)

        edge_index = from_scipy_sparse_matrix(mesh.Adjacency)[0].to(device)
        if selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=mesh.num_cells)

        data = Data(x=x, edge_index=edge_index)
        return data

    def normalize(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Normalize the data before feeding it to the GNN.

        Parameters
        ----------
        x : torch.Tensor
            The data to be normalized.
        edge_index : torch.Tensor
            Edge index tensor equivalent to the adjacency matrix of the mesh.

        Returns
        -------
        torch.Tensor
            The normalized data.

        Notes
        -----
        Overridden implementation of `GNN._normalize` to also handle the
        physical group, which is averaged across neighbours to avoid
        discontinuities that may hamper the GNN learning.
        """
        coords_sample = x[:, : (x.shape[1] - 2)]
        volumes_sample = x[:, -2].unsqueeze(-1)
        physical_groups_sample = x[:, -1].unsqueeze(-1)

        coords_sample = align_to_x_axis(coords_sample)
        coords_sample = (coords_sample - torch.mean(coords_sample, dim=0)) / torch.std(
            coords_sample, dim=0
        )

        volumes_sample = volumes_sample / torch.max(volumes_sample, 0).values
        physical_groups_sample = self._average_physical_group(
            physical_groups_sample, edge_index
        )

        return torch.cat([coords_sample, volumes_sample, physical_groups_sample], -1)

    def _average_physical_group(
        self, x: torch.Tensor, edge_index: torch.Tensor, flow: str = "source_to_target"
    ) -> torch.Tensor:
        """Compute average physical goup across neighbouring elements.

        If the averaged physical group is identically zero across the graph,
        it is set to 0.5 instead.

        Parameters
        ----------
        x : torch.Tensor
            The data to be normalized.
        edge_index : torch.Tensor
            Edge index tensor equivalent to the adjacency matrix of the mesh.

        Returns
        -------
        torch.Tensor
            Data with last node feature averaged.

        Notes
        -----
        This is a modified implementation of `avg_pool_neighbor_x` from
        `torch_geometric.nn.pool` to average only the last node feature.
        """
        num_nodes = x.shape[0]
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index
        row, col = (row, col) if flow == "source_to_target" else (col, row)
        x = scatter(x[row], col, dim=0, dim_size=num_nodes, reduce="mean")

        if torch.equal(x, torch.zeros(num_nodes, device=x.device)):
            x = 0.5 * torch.ones(num_nodes, device=x.device)

        return x


class SageHeterogeneous(GNNHeterogeneous):
    """GNN with 4 SAGE convolutional layers and 4 linear layers.

    GNN for the agglomeration of heterogeneous meshes.

    The convolutional layers have constant dimension of processed features,
    while the dense linear leayers have rpogressively decrasing dimension.
    The last dense layer is followed by a softmax layer.

    Parameters
    ----------
    hidden_units: int
        Number of hidden units of SAGEConv layers.
    lin_hidden_units: int
        Number of linear hidden units.
    num_features: int
        Number of input features
    out_classes: int, optional
        Number of outputs (default is 2).

    Attributes
    ----------
    conv1, conv2, conv3, conv4 : SAGEConv
        Convolutional SAGE layers.
    lin1, lin2, lin3, lin_last : nn.Linear
        Linear layers.
    act : torch.tanh
        Activation function.
    """

    def __init__(
        self,
        hidden_units: int,
        lin_hidden_units: int,
        num_features: int,
        out_classes: int = 2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units, aggr="mean")
        self.conv2 = SAGEConv(hidden_units, hidden_units, aggr="mean")
        self.conv3 = SAGEConv(hidden_units, hidden_units, aggr="mean")
        self.conv4 = SAGEConv(hidden_units, hidden_units, aggr="mean")
        self.lin1 = nn.Linear(hidden_units, lin_hidden_units)
        self.lin2 = nn.Linear(lin_hidden_units, lin_hidden_units // 2)
        self.lin3 = nn.Linear(lin_hidden_units // 2, lin_hidden_units // 8)
        self.lin_last = nn.Linear(lin_hidden_units // 8, out_classes)
        self.act = torch.tanh

    def forward(self, x, edge_index):
        x = self.normalize(x, edge_index)
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        x = self.act(self.conv4(x, edge_index))
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.lin_last(x)
        x = F.softmax(x, dim=1)
        return x
