"""losses module.

Contains loss function definitions for GNNs.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from scipy.spatial.transform import Rotation as Rot

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def neighbours(graph: Data, node: int) -> torch.Tensor:
    """Get the neighbours of a node.

    Returns the indices of the neighbours of `node` in `graph`.

    Parameters
    ----------
    graph : Data
        Graph to be considered
    node : int
        Node of which we want to get the neighbours.

    Returns
    -------
    torch.Tensor
        Tensor containing the indices of the neighbours.
    """
    return graph.edge_index[1][graph.edge_index[0] == node]


def cut(Y: torch.Tensor, edge_index: torch.Tensor):
    """Compute the cut of a bipartite graph.

    Parameters
    ----------
    Y : torch.Tensor
        Tensor of shape (num_nodes, 2) whose values are a one-hot encoding of
        the bipartition of the graph.
    edge_index: torch.Tensor
        Edge index tensor of the graph.

    Returns
    -------
    torch.Tensor
        Value of the cut.

    Notes
    -----
    If `Y` is a tensor of probabilities, this is the expected cut.
    """
    return torch.sum((Y[edge_index[0], 0] * Y[edge_index[1], 1]))


def volumes(Y: torch.Tensor, edge_index: torch.Tensor):
    """Compute the volumes of a partitioned graph.

    Parameters
    ----------
    Y : torch.Tensor
        Tensor of shape (num_nodes, num_subsets) whose values are a one-hot
        encoding of the partition of the graph.
    edge_index: torch.Tensor
        Edge index tensor of the graph.

    Returns
    -------
    torch.Tensor
        Tensor of length num_subsets containing the volumes of each subgraph.

    Notes
    -----
    If `Y` is a tensor of probabilities, these are the expected volumes.
    """
    degs = degree(edge_index[0], num_nodes=Y.size(0)).to(DEVICE)
    volumes = torch.t(Y.to(DEVICE)) @ degs
    return volumes


def normalized_cut(Y: torch.Tensor, edge_index: torch.Tensor):
    """Compute normalized cut for a bipartition.

    Parameters
    ----------
    Y : torch.Tensor
        Tensor of shape (num_nodes, 2) whose values are a one-hot encoding of
        the bipartition of the graph.
    edge_index: torch.Tensor
        Edge index tensor of the graph.

    Returns
    -------
    torch.Tensor
        The value of the loss function.

    Notes
    -----
    If `Y` is a tensor of probabilities, this is the expected normalized cut.
    """
    c = cut(Y, edge_index)
    vol = volumes(Y, edge_index)
    if vol[0] == 0 or vol[1] == 0:
        return torch.tensor(2, device=DEVICE)
    else:
        return torch.sum(torch.div(c, vol)).to(DEVICE)


def expected_normalized_cut(Y: torch.Tensor, graph: Data) -> torch.Tensor:
    """Compute the expected normalized cut for a bipartition.

    Parameters
    ----------
    Y : torch.Tensor
        Tensor of shape (num_nodes, 2) whose values are the probabilities of
        belonging to one of two sets, assigned by the agglomeration model, to
        each node of the graph.
    graph : Data
        Graph on which the GNN was evaluated.

    Returns
    -------
    torch.Tensor
        The value of the expected normalized cut.

    Notes
    -----
    For a detailed explanation of the definition of the expected normalized
    cut, see [1].
    [1] arXiv:2210.17457v2
    """

    D = degree(graph.edge_index[0], num_nodes=Y.size(0))
    gamma = torch.t(Y) @ D
    E_cut = torch.sum(Y[graph.edge_index[0], 0]*Y[graph.edge_index[1], 1])
    return torch.sum(torch.div(E_cut, gamma)).to(DEVICE)


def ENC_multiple(Y: torch.Tensor, graph: Data) -> torch.Tensor:
    """Compute the expected normalized cut for more than 2 partitions.

    Parameters
    ----------
    Y : torch.Tensor

    graph : Data
        Graph on which the GNN was evaluated.

    Returns
    -------
    torch.Tensor
        The value of the loss function.
    """

    D = degree(graph.edge_index[0], num_nodes=Y.size(0))
    E_volume = torch.t(Y) @ D
    E_cut = torch.sum(Y[graph.edge_index[0]]*(1-Y[graph.edge_index[1]]), dim=0)
    return torch.sum(torch.div(E_cut, E_volume)).to(DEVICE)


def loss_heterogeneous_domains(Y: torch.Tensor, graph: Data) -> torch.Tensor:
    """Compute loss function for heterogeneous domains.

    Parameters
    ----------
    y : torch.Tensor
        Evaluation output of the Neural Network: tensor of shape (num_nodes, 2)
        whose values are the probabilities  of belonging to one of two sets
        assigned by the GNN to node of the graph.
    graph : Data
        Graph on which the GNN was evaluated.

    Returns
    -------
    torch.Tensor
        The value of the loss function.

    Notes
    -----
    This loss function is the sum of the expected normalized cut and a term
    that penalizes the presence of very different physical groups in the same
    set of nodes, with suitable weights.
    """
    loss1 = expected_normalized_cut(Y, graph)
    loss2 = (torch.sum(Y[:, 0]*graph.x[:, -1])
             + torch.sum(Y[:, 1]*(1-graph.x[:, -1])))
    return (1.28*loss1 + 0.00022*loss2).to(DEVICE)


def randomrotate(coords_sample: torch.Tensor) -> torch.Tensor:
    """Randomly rotate the mesh.

    Parameters
    ----------
    coords_sample : torch.Tensor
        Coordinates of the mesh to be rotated.

    Returns
    -------
    torch.Tensor
        The rotated coordinates.

    Notes
    -----
    Rotation is always computed on GPU if available.
    """
    if coords_sample.shape[-1] == 2:  # 2-dimensional mesh
        theta = torch.randn(1)*torch.pi
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]],
                               dtype=torch.float, device=DEVICE)
    else:
        theta = (torch.rand(3)-torch.tensor([0.5, 0.5, 0.5]))*360
        rot_mat = Rot.from_euler('xyz', theta, degrees=True).as_matrix()
        rot_mat = torch.tensor(rot_mat, dtype=torch.float, device=DEVICE)

    coords_sample = torch.matmul(rot_mat, coords_sample.t().to(DEVICE)).t()
    return coords_sample


def align_to_x_axis(coords_sample: torch.Tensor) -> torch.Tensor:
    """Rotate the data so to align it with the x-axis.

    Rotates the mesh coordinates so that the largest direction is aligned
    with the x-axis.

    Parameters
    ----------
    coords_sample : torch.Tensor
        Coordinates of the mesh.

    Returns
    -------
    torch.Tensor
        Ther aligned coordinates.

    Notes
    -----
    Rotation is always computed on GPU if available, so the returned
    tensor is on 'DEVICE'.
    """
    max_coords = (torch.max(coords_sample, 0)).values
    min_coords = (torch.min(coords_sample, 0)).values

    if coords_sample.shape[-1] == 2:  # 2-dimensional mesh
        if ((max_coords[1]-min_coords[1]) > (max_coords[0]-min_coords[0])):
            theta = torch.tensor(torch.pi/2)
            rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]],
                                   dtype=torch.float, device=DEVICE)
            coords_sample = torch.matmul(rot_mat, coords_sample.t().to(DEVICE)).t()
    else:
        if (torch.max(max_coords-min_coords) == max_coords[0]-min_coords[0]):
            rot_mat = Rot.from_euler(seq="z", angles=90, degrees=True).as_matrix()
            rot_mat = torch.tensor(rot_mat, dtype=torch.float, device=DEVICE)
            coords_sample = torch.matmul(rot_mat, coords_sample.t().to(DEVICE)).t()

        if (torch.max(max_coords-min_coords) == max_coords[2]-min_coords[2]):
            rot_mat = Rot.from_euler(seq="x", angles=-90, degrees=True).as_matrix()
            rot_mat = torch.tensor(rot_mat, dtype=torch.float, device=DEVICE)
            coords_sample = torch.matmul(rot_mat, coords_sample.t().to(DEVICE)).t()

    return coords_sample
