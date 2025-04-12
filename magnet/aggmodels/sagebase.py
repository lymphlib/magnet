# sagebase

import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from ..aggmodels import AgglomerationModel, DEVICE
from .gnn import GNN
from ..mesh import Mesh, MeshDataset
from .._types import ClassList
from ..graph_utils import randomrotate, align_to_x_axis


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
        return losses.loss_normalized_cut(output, graph)


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

    def __init__(self, hidden_units: int = 64, lin_hidden_units: int = 32,
                 num_features: int = 3, out_classes: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units, aggr='mean')
        self.conv2 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.conv3 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.conv4 = SAGEConv(hidden_units, hidden_units, aggr='mean')
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

    def __init__(self, hidden_units: int, lin_hidden_units: int, num_features: int, out_classes: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units, aggr='mean')
        self.conv2 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.conv3 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.conv4 = SAGEConv(hidden_units, hidden_units, aggr='mean')
        self.lin1 = nn.Linear(hidden_units, lin_hidden_units)
        self.lin2 = nn.Linear(lin_hidden_units, lin_hidden_units//2)
        self.lin3 = nn.Linear(lin_hidden_units//2, lin_hidden_units//8)
        self.lin_last = nn.Linear(lin_hidden_units//8, out_classes)
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
        x = self.act(self.conv1r(x4, edge_index))+x3
        x = self.act(self.conv2r(x, edge_index))+x2
        x = self.act(self.conv3r(x, edge_index))+x1
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin_last(x)
        x = F.softmax(x, dim=1)
        return x
