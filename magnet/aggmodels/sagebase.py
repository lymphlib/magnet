# sagebase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from magnet.aggmodels.gnn import GeometricGNN


class SageBase2D(GeometricGNN):
    """GNN with 4 SAGE convolutional layers and 3 linear layers.

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
    lin1, lin2, lin_last : nn.Linear
        Linear layers.
    act : torch.tanh
        Activation function.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        lin_hidden_units: int = 32,
        num_features: int = 3,
        out_classes: int = 2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units, aggr="mean")
        self.conv2 = SAGEConv(hidden_units, hidden_units, aggr="mean")
        self.conv3 = SAGEConv(hidden_units, hidden_units, aggr="mean")
        self.conv4 = SAGEConv(hidden_units, hidden_units, aggr="mean")
        self.lin1 = nn.Linear(hidden_units, lin_hidden_units)
        self.lin2 = nn.Linear(lin_hidden_units, lin_hidden_units)
        self.lin_last = nn.Linear(lin_hidden_units, out_classes)
        self.act = torch.tanh

    def forward(self, x, edge_index):
        x = self.normalize(x)
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        x = self.act(self.conv4(x, edge_index))
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin_last(x)
        x = F.softmax(x, dim=1)
        return x


class SageBase(GeometricGNN):
    """GNN with 4 SAGE convolutional layers and 4 linear layers.

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
        x = self.normalize(x)
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


class SageRes(GeometricGNN):
    """GNN that uses residual connections.

    This GNN is formed by 7 `SAGEConv` layers: the first 4 have an increasing
    depth of the hidden representation, while the last 3 have decresing depth.
    The convolutional layers are followed by 3 dense linear layers and a
    softmax layer.

    Attributes
    ----------
    conv1, conv2, conv3, conv4, conv1r, conv2r, conv3r : SAGEConv
        Convolutional SAGE layers.
    lin1, lin2, lin_last : nn.Linear
        Dense linear layers.
    act : torch.tanh
        Activation function.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(3, 8)
        self.conv2 = SAGEConv(8, 16)
        self.conv3 = SAGEConv(16, 32)
        self.conv4 = SAGEConv(32, 64)
        self.conv1r = SAGEConv(64, 32)
        self.conv2r = SAGEConv(32, 16)
        self.conv3r = SAGEConv(16, 8)
        self.lin1 = nn.Linear(8, 32)
        self.lin2 = nn.Linear(32, 16)
        self.lin_last = nn.Linear(16, 2)
        self.act = torch.tanh

    def forward(self, x, edge_index):
        x = self.normalize(x)
        x1 = self.act(self.conv1(x, edge_index))
        x2 = self.act(self.conv2(x1, edge_index))
        x3 = self.act(self.conv3(x2, edge_index))
        x4 = self.act(self.conv4(x3, edge_index))
        x = self.act(self.conv1r(x4, edge_index)) + x3
        x = self.act(self.conv2r(x, edge_index)) + x2
        x = self.act(self.conv3r(x, edge_index)) + x1
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin_last(x)
        x = F.softmax(x, dim=1)
        return x
