# refiner

import numpy as np
from scipy.sparse.csgraph import connected_components
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree, to_scipy_sparse_matrix, mask_to_index
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import graclus, avg_pool
from torch_geometric.nn.pool import global_mean_pool


from magnet.mesh import Mesh, MeshDataset
from magnet.aggmodels.gnn import ReinforceLearnGNN
from magnet._absaggmodels import AgglomerationModel, DEVICE
from magnet import graph_utils


class DRLRefiner(ReinforceLearnGNN):

    def A2C_train(
        self,
        training_dataset: MeshDataset,
        time_to_sample: int = 8,
        epochs: int = 1,
        gamma: float = 0.9,
        alpha: float = 0.1,
        partitioner: AgglomerationModel = None,
        lr: float = 0.001,
        optimizer=None,
        **kwargs
    ):

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='max', factor=0.5, patience=100,
        #     threshold=0.0001, threshold_mode='abs')
        training_dataset = self._make_coarse_dataset(
            training_dataset, partitioner=partitioner, **kwargs
        )
        print("Number of coarse graphs:", len(training_dataset))
        # alpha = torch.tensor(alpha, device=DEVICE)
        CumRews = []
        self.train()
        start = time.time()

        for epoch in range(1, epochs + 1):

            shuffled_ids = shuffle(np.arange(len(training_dataset)))
            loss = torch.tensor([0.0], device=DEVICE)

            # Here starts the main loop for training
            for i in range(len(training_dataset)):
                graph, starter, subset_k = self.get_sample(
                    training_dataset[shuffled_ids[i]], partitioner=partitioner, **kwargs
                )
                len_episode = self.compute_episode_length(graph)
                rewards, values, logprobs, actions = [], [], [], []
                cumulative_rew = 0

                # here starts the episode
                for t in range(len_episode):
                    policy, value = self(graph)
                    probabilities = policy.view(-1).detach().cpu().numpy()
                    action = np.random.choice(graph.num_nodes, p=probabilities)
                    actions.append(action)
                    new_state = self.update_state(graph, action, starter.num_edges)
                    new_starter = self.change_vert(starter.clone(), subset_k[action])
                    reward = self.reward_function(
                        new_state, graph, new_starter, starter, action, **kwargs
                    )

                    graph = new_state
                    starter = new_starter
                    # collect data for loss computation
                    rewards.append(reward)
                    values.append(value.detach())
                    logprobs.append(torch.log(policy.view(-1)[action]))
                    cumulative_rew += reward.item()

                    if t % time_to_sample == 0 or t == len_episode - 1:
                        values = torch.stack(values).view(-1)
                        logprobs = torch.stack(logprobs).view(-1)
                        # update loss:
                        # cumulative discounted reward (discount is 'in the future')
                        # CDRew[T]=r_T, CDRew[T-1]=gamma*r_T +r_{T-1},..
                        CDRew = torch.zeros(len(rewards), device=DEVICE)
                        CDRew[-1] = rewards[-1]
                        for j in range(2, len(rewards) + 1):
                            CDRew[-j] = rewards[-j] + gamma * CDRew[-j + 1]

                        advantage = CDRew - values
                        loss += -torch.mean(logprobs * advantage) + alpha * torch.mean(
                            torch.pow(advantage, 2)
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # scheduler.step(loss.item())
                        loss = torch.tensor([0.0], device=DEVICE)
                        rewards, values, logprobs = [], [], []
                        if t == len_episode - 1:
                            CumRews.append(cumulative_rew)
                            print(
                                "episode:",
                                i,
                                "\t\tepisode reward:",
                                round(cumulative_rew, 5),
                                "\t\telapsed time:",
                                round((time.time() - start) / 60, 2),
                                "\t\tlearning rate:",
                                optimizer.param_groups[0]["lr"],
                            )

        plt.figure(figsize=(10, 5))
        plt.plot(CumRews)
        plt.xlabel("Episodes")
        plt.legend("Rewards")

    def _make_coarse_dataset(
        self,
        dataset: MeshDataset,
        N: int,
        partitioner: AgglomerationModel | None = None,
        threshold: int = 200,
        **kwargs
    ):
        result = []
        i = 0
        while len(result) < N and i < len(dataset):
            graph = partitioner.get_sample(dataset[i], randomRotate=True)
            result.append(graph)
            while graph.num_nodes > threshold:
                cluster = graclus(graph.edge_index.cpu()).to(DEVICE)
                coarse_graph = avg_pool(cluster, graph)
                result.append(coarse_graph)
                graph = coarse_graph
            i += 1
        return result

    def get_sample(
        self,
        mesh: Mesh | Data,
        k: int = 3,
        partitioner: AgglomerationModel | None = None,
        **kwargs
    ):
        if isinstance(mesh, Mesh):
            graph = partitioner._get_graph(mesh)
        else:
            graph = mesh

        # coarsen
        cluster = graclus(graph.edge_index.cpu())  # graclus does not work on gpu
        coarse_graph = avg_pool(
            cluster.to(DEVICE),
            Batch(batch=graph.batch, x=graph.x, edge_index=graph.edge_index),
        )
        # partition
        cl = partitioner._bisect_subgraph(
            coarse_graph, np.arange(coarse_graph.num_nodes), 2
        )  # mesh.dim
        one_hot_part = torch.zeros(
            (coarse_graph.num_nodes, 2), dtype=torch.float, device=DEVICE
        )
        one_hot_part[cl[0], 0] = 1
        one_hot_part[cl[1], 1] = 1
        biparted_coarse_graph = Data(x=one_hot_part, edge_index=coarse_graph.edge_index)
        # uncoarsen the bisected graph
        _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
        graph.x = biparted_coarse_graph.x[inverse]
        # setup for refiner
        g_red, subset_k = self.k_hop_graph_cut(graph, k)
        g_red = self._setup_intial_subgraph(graph, g_red, subset_k)
        return g_red, graph, subset_k

    def _setup_intial_subgraph(self, biparted_graph, g_red, subset_k):
        volumes = self.volumes(biparted_graph)
        total_vol = biparted_graph.num_edges
        # volumes = [torch.sum(biparted_graph.x[:, i]) for i in (0, 1)]
        # total_vol = biparted_graph.num_nodes

        # normalized_volumes = torch.tensor(V/np.sum(V),
        #                                   dtype=torch.float, device=DEVICE)
        # va = torch.dot(normalized_volumes.view(-1), biparted_graph.x[:, 0])
        # volumes = (va, 1-va)

        g_red.x = torch.cat(
            (
                g_red.x,
                volumes[0]
                / total_vol
                * torch.ones((g_red.num_nodes, 1), dtype=torch.float, device=DEVICE),
                volumes[1]
                / total_vol
                * torch.ones((g_red.num_nodes, 1), dtype=torch.float, device=DEVICE),
            ),
            dim=-1,
        )
        return g_red

    def k_hop_graph_cut(self, graph: Data, k: int):
        """Exrtact k-hop subgraph around the current cut fo refinement."""
        # extract nodes on the interface of the cut (boundary between the subgraphs)
        # assert graph.num_nodes > 0
        # n1 = torch.sum(graph.x[:, 0])
        # if n1 == 0 or n1 == graph.num_nodes:
        #     raise RuntimeError('All nodes in one partition, not good.')
        nei = graph.x[graph.edge_index[0], 0] != graph.x[graph.edge_index[1], 0]
        neib = graph.edge_index[0][nei]
        # extract k hop subgraph, and set node feature to 1 for those exactly
        # at k hops from the interface
        subset_k, edge_ind_k, _, _ = k_hop_subgraph(
            neib, k, graph.edge_index, relabel_nodes=True
        )
        subset_k_1, _, _, _ = k_hop_subgraph(
            neib, k - 1, graph.edge_index, relabel_nodes=True
        )
        nodes_boundary = set(subset_k.cpu()).difference(subset_k_1.cpu())
        boundary_features = torch.tensor(
            [1 if i in nodes_boundary else 0 for i in subset_k],
            dtype=torch.float,
            device=DEVICE,
        ).reshape((-1, 1))
        features = torch.cat((graph.x[subset_k].to(DEVICE), boundary_features), dim=-1)
        g_red = Data(x=features, edge_index=edge_ind_k)

        # if g_red.num_nodes == 0:
        #     raise RuntimeError('The subgraph around the cut is empty.')

        return g_red, subset_k

    def volumes(self, graph):
        return graph_utils.volumes(graph.x[:, :2], graph.edge_index)

    def cut(self, graph):
        return graph_utils.cut(graph.x[:, :2], graph.edge_index)

    def compute_episode_length(self, graph: Data) -> int:
        return int(self.cut(graph)) + 1

    def update_state(self, graph: Data, action: int, nnz):

        va = graph.x[0, -2].clone()
        vb = graph.x[0, -1].clone()
        dv = degree(graph.edge_index[0], num_nodes=graph.num_nodes)[action] / nnz
        # dv = 1/nnz
        # dv = graph.x[action, -3]
        if graph.x[action, 0] == 1.0:
            va -= dv
            vb += dv
        else:
            va += dv
            vb -= dv
        new_state = graph.clone()
        self.change_vert(new_state, action)
        new_state.x[:, -2] = va
        new_state.x[:, -1] = vb
        return new_state

    def objective(self, graph, starter):
        volumes = self.volumes(starter)
        return self.cut(graph) * (1 / volumes[0] + 1 / volumes[1])

    def reward_function(
        self,
        new_state: Data,
        old_state: Data,
        new_starter: Data,
        old_starter: Data,
        action: int,
        disc_penalty=0,
        cerchiobottismo=0,
        **kwargs
    ):
        """Modified normalized cut to take into account cell volumes instead"""
        # reduction in normalized cut
        old_nc = self.objective(old_state, old_starter)
        new_nc = self.objective(new_state, new_starter)
        reward = old_nc - new_nc

        if (
            old_state.x[0, -2] > old_state.x[0, -1]
            and old_state.x[action, 0] == 1
            or old_state.x[0, -2] < old_state.x[0, -1]
            and old_state.x[action, 0] == 0
        ):
            # if chosen correctly, reward proportional to the imbalance in volumes.
            reward += cerchiobottismo * torch.pow(
                old_state.x[0, -2] - old_state.x[0, -1], 2
            )
        else:
            reward -= cerchiobottismo * torch.pow(
                old_state.x[0, -2] - old_state.x[0, -1], 2
            )

        if disc_penalty > 0:
            index = torch.arange(old_state.num_nodes, device=DEVICE)
            mask = old_state.x[:, 0].to(dtype=torch.bool)
            g1 = to_scipy_sparse_matrix(old_state.subgraph(index[mask]).edge_index)
            g2 = to_scipy_sparse_matrix(old_state.subgraph(index[~mask]).edge_index)
            N_comps_A, _ = connected_components(g1)
            N_comps_B, _ = connected_components(g2)
            old_comps = N_comps_A + N_comps_B
            # connected components of new state
            index = torch.arange(new_state.num_nodes, device=DEVICE)
            mask = new_state.x[:, 0].to(dtype=torch.bool)
            g1 = to_scipy_sparse_matrix(new_state.subgraph(index[mask]).edge_index)
            g2 = to_scipy_sparse_matrix(new_state.subgraph(index[~mask]).edge_index)
            N_comps_A, _ = connected_components(g1)
            N_comps_B, _ = connected_components(g2)
            new_comps = N_comps_A + N_comps_B
            reward -= disc_penalty * (new_comps - old_comps)

        return reward

    def _refine(self, biparted_graph: Data, k: int = 3, vols=None):

        # create subgraph from parted graph
        graph, subset_k = self.k_hop_graph_cut(biparted_graph, k)
        graph = self._setup_intial_subgraph(biparted_graph, graph, subset_k).to(DEVICE)
        if graph.num_nodes == 0:
            # print('Skipped refinement')
            return biparted_graph
        # run the refiner on the subgraph
        len_episode = self.compute_episode_length(biparted_graph)
        actions = []
        peak_reward, peak_time = 0, 0
        old_nc = (
            self.cut(graph)
            * (1 / graph.x[0, -2] + 1 / graph.x[0, -1])
            / biparted_graph.num_edges
        )

        self.eval()
        for i in range(len_episode):
            with torch.no_grad():
                policy = self(graph)
            probs = policy.view(-1).detach().cpu().numpy()
            action = np.argmax(probs)
            new_state = self.update_state(graph, action, biparted_graph.num_edges)
            actions.append(action)
            graph = new_state
            reward = (
                old_nc
                - self.cut(graph)
                * (1 / graph.x[0, -2] + 1 / graph.x[0, -1])
                / biparted_graph.num_edges
            )

            if reward > peak_reward:
                peak_reward = reward
                peak_time = i + 1

            # if the changed node is immediately brought back, end the episode
            if i >= 1 and actions[-1] == actions[-2]:
                break

        # finally, change the original graph to reflect the refinement
        for action in actions[:peak_time]:
            self.change_vert(biparted_graph, subset_k[action])
        #     print(subset_k[action], self.volumes(biparted_graph))
        # print(subset_k[actions])
        # extract the new partition and return it
        biparted_graph = self._manual_clean(biparted_graph)
        biparted_graph = self._flip_all(biparted_graph)

        return biparted_graph

    def _manual_clean(self, bipartite_graph):
        mask1 = bipartite_graph.x[:, 0].to(torch.bool)
        mask2 = bipartite_graph.x[:, 1].to(torch.bool)
        g1 = bipartite_graph.subgraph(mask1)
        g2 = bipartite_graph.subgraph(mask2)
        deg1 = degree(g1.edge_index[0], num_nodes=g1.num_nodes)
        deg2 = degree(g2.edge_index[0], num_nodes=g2.num_nodes)
        solitary_nodes1 = mask_to_index(mask1)[deg1 == 0]
        solitary_nodes2 = mask_to_index(mask2)[deg2 == 0]
        self.change_vert(bipartite_graph, solitary_nodes1)
        self.change_vert(bipartite_graph, solitary_nodes2)
        return bipartite_graph

    def _flip_all(self, bipartite_graph):
        """Flipl the one hot encoding to mitigate asymmetries in the
        refinement process."""
        bipartite_graph.x[:, :2] = 1 - bipartite_graph.x[:, :2]
        return bipartite_graph


class RefGatti(DRLRefiner):
    """Deep Reinforcement Learning partition refiner by A. Gatti et al.

    Graph Neural network with 2 SAGE convolutional layers followed by.
    """

    def __init__(self, n_features: int = 5, units: int = 10):
        super().__init__()

        self.conv_first = SAGEConv(n_features, units)
        self.conv_common = SAGEConv(units, units)

        self.conv_actor = SAGEConv(units, 1)

        self.conv_critic = SAGEConv(units, units)
        self.final_critic = nn.Linear(units, 1)
        self.act = torch.tanh

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        do_not_flip = x[:, 2] != 0.0
        x = self.act(self.conv_first(x, edge_index))
        x = self.act(self.conv_common(x, edge_index))

        x_actor = self.conv_actor(x, edge_index)
        x_actor[do_not_flip] = torch.tensor(-np.Inf)
        # x_actor = torch.log_softmax(x_actor, dim=0)   !!!!
        x_actor = torch.softmax(x_actor, dim=0)

        if not self.training:
            return x_actor

        x_critic = self.conv_critic(x.detach(), edge_index)
        x_critic = self.final_critic(x_critic)
        x_critic = self.act(global_mean_pool(x_critic, batch))
        return x_actor, x_critic


class Reyyy(DRLRefiner):
    """Deep Reinforcement Learning refiner variant.

    Graph Neural network with 2 SAGE convolutional layers followed by.
    """

    def __init__(self, n_features: int = 6, units: int = 10):
        super().__init__()

        self.conv_first = SAGEConv(n_features, units)
        self.conv_common = SAGEConv(units, units)

        self.conv_actor = SAGEConv(units, units)
        self.lin_actor = nn.Linear(units, 1)  # new layer

        self.conv_critic = SAGEConv(units, units)
        self.final_critic = nn.Linear(units, 1)
        self.act = torch.tanh

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        do_not_flip = x[:, 2] != 0.0
        x = self.act(self.conv_first(x, edge_index))
        x = self.act(self.conv_common(x, edge_index))

        x_actor = self.act(self.conv_actor(x, edge_index))
        x_actor = self.lin_actor(x_actor)
        x_actor[do_not_flip] = torch.tensor(-np.inf)
        # x_actor = torch.log_softmax(x_actor, dim=0)   !!!!
        x_actor = torch.softmax(x_actor, dim=0)

        if not self.training:
            return x_actor

        x_critic = self.act(
            self.conv_critic(x.detach(), edge_index)
        )  # !!! i put more act
        x_critic = self.final_critic(x_critic)
        x_critic = self.act(global_mean_pool(x_critic, batch))
        return x_actor, x_critic
