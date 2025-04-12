# rL

import math
import numpy as np

from scipy.sparse.csgraph import connected_components
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, AttentionalAggregation
from torch_geometric.data import Data
from torch_geometric.utils import (degree, from_scipy_sparse_matrix,
                                   to_scipy_sparse_matrix)

from ..mesh import Mesh
from .gnn import ReinforceLearnGNN
from ..aggmodels import DEVICE
from ..graph_utils import randomrotate, align_to_x_axis, normalized_cut
from .._types import ClassList


class DRLCoarsePartioner(ReinforceLearnGNN):

    def get_sample(self, mesh: Mesh, **kwargs):

        edge_index = from_scipy_sparse_matrix(mesh.Adjacency)[0].to(DEVICE)
        data = torch.zeros((mesh.num_cells, 2), device=DEVICE)
        graph = Data(x=data, edge_index=edge_index)
        self._setup_one_hot(graph)

        return graph

    def _setup_one_hot(self, graph):
        graph.x[:, 0] = 1
        graph.x[:, 1] = 0
        # first_vertex = np.random.choice(np.argmin(degree(
        #     graph.edge_index[0]).cpu().numpy(), keepdims=True))
        # self.change_vert(graph, first_vertex)

    def compute_episode_length(self, graph: Data) -> int:
        return math.ceil(graph.num_nodes/2 - 1)

    def update_state(self, graph: Data, action: int) -> Data:
        return self.change_vert(graph.clone(), action)

    def reward_function(self, new_state: Data, old_state: Data,
                        action: int, **kwargs):
        return (normalized_cut(old_state.x[:, :2], old_state.edge_index)
                - normalized_cut(new_state.x[:, :2], new_state.edge_index))

    def ac_eval(self, graph: Data, perc: float = 0.01):
        # Evaluation of the DRL model on the coarsest graph
        graph_test = graph.clone()
        self._setup_one_hot(graph_test)
        len_episode = self.compute_episode_length(graph)
        error_bal = math.ceil(graph_test.num_nodes * perc)
        cuts, nodes = [], []
        self.eval()
        # Run the episod
        for i in range(len_episode + error_bal):
            policy = self(graph_test)
            policy = policy.view(-1).detach().cpu().numpy()
            action = np.random.choice(graph_test.num_nodes, p=policy)
            # action = np.argmax(policy)
            graph_test = self.update_state(graph_test, action)
            if i >= int(len_episode - error_bal):
                cuts.append(self.cut(graph_test).cpu())
                nodes.append(action)
        # take the state with best cut in the tolerance window
        if len(cuts) > 0:
            minimum_cut_pos = np.argmin(cuts).reshape((-1,))
            # firts tiebreaker: take the state closest to len_episode, i.e.
            # with most balanced volumes
            if len(minimum_cut_pos) > 1:
                minimum_cut_pos = minimum_cut_pos[np.argmin(
                    np.abs(minimum_cut_pos-(len(cuts)//2)))].reshape((-1,))
                # second tiebreaker: choose randomly
                if len(minimum_cut_pos) > 1:
                    minimum_cut_pos = np.random.choice(minimum_cut_pos)
            else:
                minimum_cut_pos = minimum_cut_pos[0]
            self.change_vert(graph_test, nodes[minimum_cut_pos + 1:])

        return graph_test

    def multi_eval(self, graph, step: int = 1, perc: float = 0.01):
        graph_test = graph.clone()
        self._setup_one_hot(graph_test)
        len_episode = int(graph_test.num_nodes / 2 - 1)
        error_bal = math.ceil(graph_test.num_nodes * perc)
        cuts, nodes = [], []
        self.eval()
        # Run the episode
        added_nodes = 0
        while added_nodes < len_episode + error_bal:
            policy = self(graph_test)
            policy = policy.view(-1).detach().cpu().numpy()
            if added_nodes < step:
                # for the first few ones add only one
                actions = np.random.choice(graph_test.num_nodes, size=(1,),
                                           replace=False, p=policy)
            elif np.count_nonzero(policy) < step:
                # if the number of nodes with positive probability is less
                # than step, take all those nodes.
                actions = np.nonzero(policy)[0]
            else:
                actions = np.random.choice(graph_test.num_nodes, size=(step,),
                                           replace=False, p=policy)
            for action in actions:
                added_nodes += 1
                graph_test = self.update_state(graph_test, action)
                if added_nodes >= int(len_episode - error_bal):
                    cuts.append(self.cut(graph_test).cpu())
                    nodes.append(action)
        # take the state with best cut in the tolerance window
        if len(cuts) > 0:
            minimum_cut_pos = np.argmin(cuts).reshape((-1,))
            # firts tiebreaker: take the state closest to len_episode, i.e.
            # with most balanced volumes
            if len(minimum_cut_pos) > 1:
                minimum_cut_pos = minimum_cut_pos[np.argmin(
                    np.abs(minimum_cut_pos-(len(cuts)//2)))].reshape((-1,))
                # second tiebreaker: choose randomly
                if len(minimum_cut_pos) > 1:
                    minimum_cut_pos = np.random.choice(minimum_cut_pos)
            else:
                minimum_cut_pos = minimum_cut_pos[0]
            self.change_vert(graph_test, nodes[minimum_cut_pos + 1:])

        return graph_test

    def _get_graph(self, mesh: Mesh) -> Data:
        return self.get_sample(mesh)

    def _bisect_subgraph(self, graph: Data, subset: np.ndarray, dim: int) -> ClassList:
        subgraph = graph.subgraph(torch.tensor(subset, device=DEVICE))
        # bisected_graph = self.multi_eval(subgraph, step=5)
        mask = self._bisect_graph(subgraph).cpu()
        return [subset[mask], subset[~mask]]

    def _bisect_graph(self, graph: Data):
        bisected_graph = self.ac_eval(graph)
        return bisected_graph.x[:, 0].to(dtype=torch.bool)


class DRLCPGatti(DRLCoarsePartioner):
    """Deep Reinforcement Learning coarse partitioner by A. Gatti et al.

    Graph Neural network with 4 GAT convolutional layers followed by 2 dense
    layers common to both actor and critic.
    """
    def __init__(self, hid_conv, hid_lin, input_features: int = 2):
        super().__init__()
        self.conv1 = GATConv(input_features, hid_conv[0])
        self.conv2 = GATConv(hid_conv[0], hid_conv[1])
        self.conv3 = GATConv(hid_conv[1], hid_conv[2])
        self.conv4 = GATConv(hid_conv[2], hid_conv[3])

        self.lin1 = nn.Linear(hid_conv[3], hid_lin[0])
        self.lin2 = nn.Linear(hid_lin[0], hid_lin[1])

        # branching of actor and critic
        self.actor1 = nn.Linear(hid_lin[1], hid_lin[2])
        self.actor2 = nn.Linear(hid_lin[2], 1)

        self.GlobAtt = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(
                    hid_lin[1], hid_lin[1]), nn.Tanh(), nn.Linear(
                    hid_lin[1], 1)))
        self.critic1 = nn.Linear(hid_lin[1], hid_lin[2])
        self.critic2 = nn.Linear(hid_lin[2], 1)

        self.act = torch.tanh

    def forward(self, graph):
        x_start, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.act(self.conv1(x_start, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        x = self.act(self.conv4(x, edge_index))

        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))

        x_actor = self.act(self.actor1(x))
        x_actor = self.actor2(x_actor)

        # set probabilities of already flipped vertices to zero
        flipped = x_start[:, 1] == torch.tensor(1, device=DEVICE)
        x_actor.data[flipped] = torch.tensor(-np.Inf)
        x_actor = torch.softmax(x_actor, dim=0)

        if self.training:
            x_critic = self.GlobAtt(x, batch)
            x_critic = self.act(self.critic1(x_critic))
            x_critic = self.critic2(x_critic)
            return x_actor, x_critic
        else:
            return x_actor

    def _setup_one_hot(self, graph):
        graph.x[:, 0] = 1
        graph.x[:, 1] = 0
        first_vertex = np.random.choice(np.argmin(degree(
            graph.edge_index[0]).cpu().numpy(), keepdims=True))
        self.change_vert(graph, first_vertex)


class WeakContigDRLCP(DRLCoarsePartioner):
    def reward_function(self, new_state: Data, old_state: Data, action: int,
                        disc_coeff: float = 0) -> float:
        # add penalty if new node is not connected
        if all(old_state.x[self.neighbours(old_state, action), 0] == 1):
            disconnected_penalty = 1
        else:
            # old connected components
            index = torch.arange(old_state.num_nodes, device=DEVICE)
            mask = old_state.x[:, 0].to(dtype=torch.bool)
            g1 = to_scipy_sparse_matrix(old_state.subgraph(index[mask]).edge_index)
            g2 = to_scipy_sparse_matrix(old_state.subgraph(index[~mask]).edge_index)
            N_comps_A, _ = connected_components(g1)
            N_comps_B, _ = connected_components(g2)
            old_comps = N_comps_A + N_comps_B - 2
            # connected components of new state
            index = torch.arange(new_state.num_nodes, device=DEVICE)
            mask = new_state.x[:, 0].to(dtype=torch.bool)
            g1 = to_scipy_sparse_matrix(new_state.subgraph(index[mask]).edge_index)
            g2 = to_scipy_sparse_matrix(new_state.subgraph(index[~mask]).edge_index)
            N_comps_A, _ = connected_components(g1)
            N_comps_B, _ = connected_components(g2)
            new_comps = N_comps_A + N_comps_B - 2
            disconnected_penalty = new_comps - old_comps
        return (normalized_cut(old_state) - normalized_cut(new_state)
                - disc_coeff * disconnected_penalty)

    def get_sample(self, mesh: Mesh, randomRotate: bool = True):
        edge_index = from_scipy_sparse_matrix(mesh.Adjacency)[0].to(DEVICE)
        coords_sample = torch.tensor(mesh.Coords, dtype=torch.float, device=DEVICE)
        if randomRotate:
            coords_sample = randomrotate(coords_sample)
        data = torch.cat((torch.zeros((mesh.num_cells, 2), device=DEVICE),
                         coords_sample), dim=-1)
        graph = Data(x=data, edge_index=edge_index)
        self._setup_one_hot(graph)
        return graph

    def normalize(self, x):
        coords_sample = x[:, 2:]
        coords_sample = align_to_x_axis(coords_sample)

        coords_sample = (coords_sample-torch.mean(coords_sample, dim=0)
                         )/torch.std(coords_sample, dim=0)

        return torch.cat([x[:, :2], coords_sample], -1)

    def __init__(self, hidden_units: int, lin_hidden_units: int,
                 num_features: int):
        super().__init__()
        # common layers
        self.conv1 = SAGEConv(num_features, hidden_units, aggr='mean')
        self.conv2 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.conv3 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.conv4 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.lin1 = nn.Linear(hidden_units, lin_hidden_units)
        self.lin2 = nn.Linear(lin_hidden_units, lin_hidden_units)
        # actor branch
        self.actor1 = nn.Linear(lin_hidden_units, lin_hidden_units//2)
        self.actor2 = nn.Linear(lin_hidden_units//2, 1)
        # critic branch
        self.GlobAtt = AttentionalAggregation(nn.Sequential(
                nn.Linear(lin_hidden_units, lin_hidden_units),
                nn.Tanh(),
                nn.Linear(lin_hidden_units, 1)))
        self.critic1 = nn.Linear(lin_hidden_units, lin_hidden_units//2)
        self.critic2 = nn.Linear(lin_hidden_units//2, 1)
        self.act = torch.tanh

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        flipped = x[:, 1] == torch.tensor(1, device=DEVICE)

        x = self.normalize(x)
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        x = self.act(self.conv4(x, edge_index))
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        # actor branch
        x_actor = self.act(self.actor1(x))
        x_actor = self.actor2(x_actor)
        # set probabilities of already flipped nodes to zero
        x_actor.data[flipped] = torch.tensor(-np.Inf)
        x_actor = torch.softmax(x_actor, dim=0)
        if self.training:
            # critic branch
            x_critic = self.GlobAtt(x, batch)
            x_critic = self.act(self.critic1(x_critic))
            x_critic = self.critic2(x_critic)
            return x_actor, x_critic
        else:
            return x_actor


class ContigDRLCP(DRLCPGatti):
    """Variant of RL partitioner model with strong connectedness requirement.
    """
    def _unmask(self, graph: Data, indices, feature: int):
        graph.x[indices, feature] = 0
        return graph

    def _setup_one_hot(self, graph):
        graph.x[:, 0] = 1
        graph.x[:, 1] = 0
        graph.x[:, 2] = 1
        first_vertex = np.random.choice(np.argmin(degree(
            graph.edge_index[0]).cpu().numpy(), keepdims=True))
        self.change_vert(graph, first_vertex)
        self._unmask(graph, self.neighbours(graph, first_vertex), 2)

    def get_sample(self, mesh: Mesh):
        edge_index = from_scipy_sparse_matrix(mesh.Adjacency)[0].to(DEVICE)
        data = torch.cat(torch.zeros((mesh.num_cells, 3), device=DEVICE),
                         torch.tensor(mesh.Coords, device=DEVICE))
        graph = Data(x=data, edge_index=edge_index)
        self._setup_one_hot(graph)
        return graph

    def update_state(self, graph: Data, action: int) -> Data:
        new_state = graph.clone()
        self.change_vert(new_state, action)
        self._unmask(new_state, self.neighbours(graph, action), 2)
        return new_state

    def forward(self, graph):
        x_start, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.act(self.conv1(x_start, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        x = self.act(self.conv4(x, edge_index))

        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))

        x_actor = self.act(self.actor1(x))
        x_actor = self.actor2(x_actor)

        # set probabilities of already flipped vertices to zero
        mask = torch.logical_or(x_start[:, 1] == 1, x_start[:, 2] == 1)
        x_actor.data[mask] = torch.tensor(-np.Inf)
        x_actor = torch.softmax(x_actor, dim=0)

        if self.training:
            x_critic = self.GlobAtt(x, batch)
            x_critic = self.act(self.critic1(x_critic))
            x_critic = self.critic2(x_critic)
            return x_actor, x_critic
        else:
            return x_actor
