# GNN

from abc import abstractmethod
import torch
from torch_geometric.data import Data

from magnet.mesh import Mesh


class GNN(torch.nn.Module):
    """Abstract base class for Graph Neural networks for mesh agglomeration.
    """

    def load_model(self, model_path: str) -> None:
        """Load model from state dictionary.

        Parameters
        ----------
        model_path : str
            The path of the `.pt` state dictionary file.

        Returns
        -------
        None
        """

        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

    def save_model(self, output_path: str) -> None:
        """Save current model to state dictionary.

        Parameters
        ----------
        output_path : str
            The path where the state dictionary `.pt` file will be saved.

        Returns
        -------
        None
        """
        torch.save(self.state_dict(), output_path)

    def get_number_of_parameters(self) -> int:
        """Get total number of parameters of the GNN.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    @abstractmethod
    def get_sample(self, mesh: Mesh) -> Data:
        """Get graph data from mesh for training."""
        raise NotImplementedError("Method must be overridden")


class ReinforceLearnGNN(GeometricGNN):
    """Reinforcement Learning GNN for graph partitioning."""
    def A2C_train(self,
                  training_dataset: MeshDataset,
                  batch_size: int = 1,
                  epochs: int = 1,
                  gamma: float = 0.9,
                  alpha: float = 0.1,
                  optimizer=None,
                  **kwargs):

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=0.001, weight_decay=1e-5)
        # alpha = torch.tensor(alpha, device=DEVICE)
        CumRews = []
        self.train()
        start = time.time()

        for epoch in range(1, epochs+1):

            shuffled_ids = shuffle(np.arange(len(training_dataset)))
            loss = torch.tensor([0.], device=DEVICE)

            # Here starts the main loop for training
            for i in range(len(training_dataset)):
                graph = self.get_sample(training_dataset[shuffled_ids[i]], **kwargs)
                len_epsiode = self.compute_episode_length(graph)
                rewards, values, logprobs = [], [], []
                cumulative_rew = 0

                # here starts the episode
                for _ in range(len_epsiode):
                    policy, value = self(graph)
                    probabilities = policy.view(-1).detach().cpu().numpy()
                    action = np.random.choice(graph.num_nodes, p=probabilities)
                    new_state = self.update_state(graph, action)
                    reward = self.reward_function(new_state, graph, action, **kwargs)
                    graph = new_state
                    # collect data for loss computation
                    rewards.append(reward)
                    values.append(value.detach())
                    logprobs.append(torch.log(policy.view(-1)[action]))
                    cumulative_rew += reward.item()

                CumRews.append(cumulative_rew)
                values = torch.stack(values).view(-1)
                logprobs = torch.stack(logprobs).view(-1)
                # update loss:
                # cumulative discounted reward (discount is 'in the future')
                # CDRew[T]=r_T, CDRew[T-1]=gamma*r_T +r_{T-1},..
                CDRew = torch.zeros(len_epsiode, device=DEVICE)
                CDRew[-1] = rewards[-1]
                for j in range(2, len_epsiode+1):
                    CDRew[-j] = rewards[-j] + gamma*CDRew[-j+1]

                advantage = CDRew - values
                loss += -torch.mean(logprobs*advantage) + alpha * torch.mean(torch.pow(advantage, 2))

                if i % batch_size == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = torch.tensor([0.], device=DEVICE)
                    print('episode:', i,
                          '\t\tepisode reward:', round(cumulative_rew, 5),
                          '\t\telapsed time:', round((time.time()-start)/60, 2))

        plt.figure(figsize=(10, 5))
        plt.plot(CumRews)
        plt.xlabel('Episodes')
        plt.legend('Rewards')

    @abstractmethod
    def compute_episode_length(self, graph: Data) -> int:
        pass

    @abstractmethod
    def update_state(self, graph: Data, action: int) -> Data:
        pass

    @abstractmethod
    def reward_function(self, new_state: Data, old_state: Data, action: int
                        ) -> torch.Tensor:
        pass

    def change_vert(self, graph: Data, action: int):
        """In place change of vertex to other subgraph."""
        graph.x[action, :2] = 1-graph.x[action, :2]
        return graph
