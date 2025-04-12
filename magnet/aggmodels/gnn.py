# GNN

from abc import abstractmethod
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from ..aggmodels import AgglomerationModel, DEVICE
from ..graph_utils import expected_normalized_cut
from .gnn import GNN, GeometricGNN
from ..mesh import Mesh, MeshDataset
from .._types import ClassList
from ..graph_utils import randomrotate, align_to_x_axis


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
    
class GeometricGNN(GNN, AgglomerationModel):
    def _get_graph(self, mesh: Mesh, device=DEVICE) -> Data:
        return self.get_sample(mesh, device=device)

    def _bisect_subgraph(self, graph: Data, subset: np.ndarray,
                         spatial_dimensions: int) -> ClassList:
        """Call the model on the mesh graph to bisect it once.

        Parameters
        ----------
        graph : Data
            The graph to be bisected.
        dim : int
            Spatial dimensions of the mesh (2 or 3).

        Returns
        -------
        ClassList
            A list of 2 arrays, each containing the indices of the elements
            corresponding to one of the two parts.
        """
        if len(subset) < 2:
            raise RuntimeError('Attempting to bisect graph with less than 2 nodes.')
        else:
            # Run the model
            graph = graph.subgraph(torch.tensor(subset, device=DEVICE))

            # Extract the two clusters
            boolean_out = self._bisect_graph(graph).cpu().numpy()
            classes = [subset[boolean_out],
                       subset[~boolean_out]]

            # In case the model returns only one cluster instead of two, use KMeans.
            if (len(classes[0]) == 0 or len(classes[1]) == 0):
                parts = KMeans(2).fit_predict(graph.x[:, :spatial_dimensions].cpu())
                classes = [subset[parts == 0], subset[parts == 1]]

            return classes

    def _bisect_graph(self, graph: Data) -> torch.tensor:
        self.eval()
        out = self(graph.x, graph.edge_index)
        return out[:, 0] > 0.5

    def get_sample(self,
                   mesh: Mesh,
                   randomRotate: bool = False,
                   selfloop: bool = False,
                   device=DEVICE,
                   ) -> Data:
        """create a graph data structure sample from a mesh.

        This is used for both training and running the GNN.

        Parameters
        ----------
        mesh : Mesh
            Mesh to be sampled.
        randomRotate : bool, optional
            If True, randomly rotate the mesh (default is `False`).
        selfloop : bool, optional
            If True, add 1 on the diagonal of the adjacency matrix, i.e
            self-loops on the graph (default is `False`).

        Returns
        -------
        Data
            Graph data representing the mesh.

        Notes
        -----
        The two tensors `x` and `edge_index` are both on `DEVICE` (cuda, if
        available).
        """
        coords_sample = torch.tensor(mesh.Coords, dtype=torch.float, device=device)
        volumes_sample = torch.tensor(mesh.Volumes, dtype=torch.float, device=device)

        if randomRotate:
            coords_sample = randomrotate(coords_sample)

        x = torch.cat([coords_sample, volumes_sample], -1)

        edge_index = from_scipy_sparse_matrix(mesh.Adjacency)[0].to(device)
        if selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=mesh.num_cells)

        data = Data(x=x, edge_index=edge_index)
        return data

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the data before feeding it to the GNN.

        Parameters
        ----------
        x : torch.Tensor
            The data to be normalized.

        Returns
        -------
        torch.Tensor
            The normalized data.

        Notes
        -----
        Normalization consists in aligning the widest direction of mesh to the
        x axis by rotating it and rescaling the coordinates to have zero mean
        and unit variance.

        The output is returned on the the same `torch` device as the output of
        `get_sample`, i.e. `DEVICE`.
        """
        # using `self.get_sample`, `x` is on the same device as `self`.
        coords_sample = x[:, :(x.shape[1]-1)]
        volumes_sample = x[:, -1].unsqueeze(-1)
        coords_sample = align_to_x_axis(coords_sample)

        coords_sample = (coords_sample-torch.mean(coords_sample, dim=0)
                         )/torch.std(coords_sample, dim=0)
        volumes_sample = volumes_sample/torch.max(volumes_sample, 0).values

        return torch.cat([coords_sample, volumes_sample], -1)

    def train_GNN(self,
                  training_dataset: MeshDataset,
                  validation_dataset: MeshDataset,
                  epochs: int,
                  batch_size: int,
                  learning_rate: float = 1e-4,
                  save_logs: bool = True
                  ) -> None:
        """Train the Graph Neural Network.

        Parameters
        ----------
        training_dataset : MeshDataset
            Dataset of meshes on which to train the GNN.
        validation_dataset : MeshDataset
            Validation dataset to check that no overfitting is occurring.
        epochs : int
            Number of training epochs.
        batch_size : int
            Size of the minibatch to be used.
        learning_rate : float, optional
            Initial learning rate for the scheduler (default is 1e-4).
        save_logs : bool, optional
            If True, save the training and validation loss histories, the
            scheduled learning rate, their plots, plus a short summary
            (default is True).

        Returns
        -------
        None
        """

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20,
            threshold=0.0001, threshold_mode='abs')
        dataset_size = len(training_dataset)

        loss_array = []
        loss_val_array = []
        epoch_loss_array = []
        epoch_loss_val_array = []
        lr_array = []

        start = time.time()

        for epoch in range(1, epochs+1):

            shuffled_ids = shuffle(list(range(dataset_size)))

            self.train()
            loss = torch.tensor([0.], device=DEVICE)

            # Training
            for i in range(dataset_size):
                g = self.get_sample(training_dataset[shuffled_ids[i]],
                                    randomRotate=True, selfloop=True)
                out = self(g.x, g.edge_index)
                loss += self.loss_function(out, g)

                if i % batch_size == 0 or i == dataset_size-1:
                    optimizer.zero_grad()
                    loss_array.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    loss = torch.tensor([0.], device=DEVICE)

            epoch_loss_array.append(sum(loss_array[-batch_size:])/batch_size)
            scheduler.step(loss_array[-1])
            lr_array.append(optimizer.param_groups[0]['lr'])

            # Validation
            self.eval()
            loss_val = torch.tensor([0.], device=DEVICE)
            shuffled_val_ids = shuffle(list(range(len(validation_dataset))))

            for i in range(len(validation_dataset)):
                g_val = self.get_sample(validation_dataset[shuffled_val_ids[i]],
                                        randomRotate=True, selfloop=True)
                out_val = self(g_val.x, g_val.edge_index)
                loss_val += self.loss_function(out_val, g_val)
                if i % batch_size == 0 or i == dataset_size-1:
                    loss_val_array.append(loss_val.item())
                    loss_val = torch.tensor([0.], device=DEVICE)

            epoch_loss_val_array.append(sum(loss_val_array[-batch_size:])/batch_size)

            print('epoch:', epoch,
                  '\t\tloss:', round(epoch_loss_array[-1], 5),
                  '\t\tvalidation loss:', round(loss_val_array[-1], 5),
                  '\t\tlr:', optimizer.param_groups[0]['lr'],
                  '\t\telapsed time:', round((time.time()-start)/60, 2))

        training_time = time.time()-start

        # postprocessing: create log file, save loss arrays and plots
        if save_logs:
            path = 'training/'+datetime.now().strftime('%Y%m%d%H')+'_'+self.__class__.__name__
            if not os.path.isdir(path):
                os.mkdir(path)

            content = (
                'Model:\t'+self.__class__.__name__+'\n'
                + 'Number of epochs:\t'+str(epochs)+'\n'
                + 'Batch size:\t'+str(batch_size)+'\n'
                + 'Initial learning rate:\t'+str(learning_rate)+'\n\n'

                + 'Training dataset:\t'+training_dataset.name+'\t\tSize:\t'+str(len(training_dataset))+'\n'
                + 'Validation dataset:\t'+validation_dataset.name+'\t\tSize:\t'+str(len(validation_dataset))+'\n\n'

                + 'Final training loss:\t'+str(loss_array[-1])+'\n'
                + 'Final validation loss:\t'+str(loss_val_array[-1])+'\n'
                + 'Trainig time:\t'+str(round(training_time/60, 2))+' minutes\n\n'

                + 'Date:\t'+datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
                )
            with open(path+'/training_log.txt', 'w') as f:
                f.write(content)
                f.close()

            # save the loss arrays as .csv files
            np.savetxt(path+"/loss.csv", np.asarray(loss_array), delimiter=",")
            np.savetxt(path+"/epoch_loss.csv", np.asarray(epoch_loss_array), delimiter=",")
            np.savetxt(path+"/loss_val.csv", np.asarray(loss_val_array), delimiter=",")
            np.savetxt(path+"/epoch_loss_val.csv", np.asarray(epoch_loss_val_array), delimiter=",")
            np.savetxt(path+"/lr.csv", np.asarray(lr_array), delimiter=",")

            # plot loss
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_loss_array)
            plt.plot(epoch_loss_val_array)
            plt.xlabel('Epochs')
            plt.legend(['Training loss', 'Validation loss'])
            plt.savefig(path+'/loss_plot.png', dpi=300)

            # # plot lr schedule
            # plt.figure(figsize=(10, 5))
            # plt.plot(lr_array)
            # plt.xlabel('Epochs')
            # plt.ylabel('lr')
            # plt.savefig("outputs/plots/lr"+modelname+".png", dpi=300)

    def loss_function(self, output: torch.Tensor, graph: Data) -> torch.Tensor:
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
        return expected_normalized_cut(output, graph)


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
