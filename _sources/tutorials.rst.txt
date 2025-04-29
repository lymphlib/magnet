===========
User guides
===========

.. _installation:

Installation
============
To use magnet, you will need the following python packages:

* torch
* torch_geometric
* meshio
* vtk
* gmsh
* numpy, scipy
* matplotlib
* networkx
* metis-python
* scikit-learn

They are automatically installed when running

.. code:: console

    $ pip install .

.. warning::

    If you intend to use the `Lymph` interface, to be able to call Python from MATLAB, you will
    need a compatible Python version. See the :doc:`Lymph <lymph_interface/lymph>` section for more details.

To avoid conflicts between different packages, it is suggested to create a new virtual environment:

.. code:: console

    $ python -m venv .myvenv

You will also need to install Metis locally. See `metispy <https://github.com/james77777778/metis_python>`_ and `METIS <https://github.com/KarypisLab/METIS>`_ for more details.

.. note::

    If you are not able to import ``metispy``, try to set the environment variable 'METIS_DLL'
    to the exact path of the python shared library file `metis.dll` in the `__init__.py` of the package.

.. _test cases:

Test cases
================

This package comes with a set of examples extracted from the test cases of the Magnet white paper.
They can be found in the folder ``examples``. They can be easily run in Google Colab as show in ``examples/python/examples.ipynb``.

.. _dataset_creation:

Dataset creation
================
There are 2 types of datasets
In the first case, simply call a generate function specifying the number of each type of mesh to be included in the dataset.
magnet provides 2 ways of creating datasets: using the built-in :py:mod:`~magnet.generate2D` module, or by using the :py:func:`create_dataset` function.

.. code-block:: python

    from magnet.generate2D import generate_2D_dataset
    generate_2D_dataset(200, 200, 200, 200, 'datasets', 'trainig_dataset')

This will create a folder 'datasets/training_dataset' containing all the generated meshes named progressively starting from 'mesh0.vtk',
a summary of the dataset porperties and a `.npz` file containig the mesh graph datas.

If instead you want to use other meshes, you first need to put them in one folder with the same naming scheme as
before (progressively from 'mesh0.vtk'), then call :py:func:`~magnet.io.create_dataset`. This will create the `.npz` file and a summary file, similar to before.

.. code-block:: python

    from magnet.io import create_dataset
    create_dataset('datasets/mydatasetfolder', n_meshes=100)

.. Creating a GNN model
.. ====================

.. You can define a particular GNN architecture by defining a new class inheriting from one of the abstract
.. classes in :py:mod:`magnet.aggmodels`. The GNN class is an extension of torch.nn.Module providing utility methods for the 

.. As for any pytorch NN, you will need to defin the __init__ method that defines the architecture of the network,
.. and a forward method, i.e. a forward pass of the NN.

GNN Training
============

To train a GNN, you first need two datasets: a training dataset and a validation dataset.
First, load the datasets using :py:func:`~magnet.io.load_dataset` or :py:func:`~magnet.io.load_graph_dataset`
after having :ref:`created them <dataset_creation>`.

.. code-block:: python

    from magnet.io import load_graph_dataset
    tr_set = load_graph_dataset('datasets/training_dataset')
    val_set = load_graph_dataset('datasets/validation_dataset')

Then, initialize the GNN, e.g. using one of the predefined models.

.. code-block:: python

    from magnet import aggmodels
    GNNtest = aggmodels.SageBase2D(64, 32, 3, 2).to(aggmodels.DEVICE)

.. note::

    When initializing a GNN, always use `to(DEVICE)`. This is because all operations are carried
    out on GPU (if `cuda` is available) since they are faster.

To start the training, call the :py:meth:`~magnet.aggmodels.GNN.train_GNN` method, specifying the number of epochs,
the batch size and learning rate.

.. code-block:: python

    GNNtest.train_GNN(tr_set, val_set, epochs=300, batch_size=4, learning_rate=1e-5)

During training, log messages will describe the training progress:

When training is completed, by default a plot displaying the training a and validation loss functions
and a log file is saved with a summary of the training.

To save the trained model, call :py:meth:`~magnet.aggmodels.GNN.save_model` to save it as a state dictionary.

.. code-block:: python

    GNNtest.save_model('models/SageBase2D_training_test.pt')

Mesh agglomeration
==================

To agglomerate a single mesh, first load it using :py:func:`~magnet.io.load_mesh`:

.. code-block:: python

    from magnet.io import load_mesh
    mesh = load_mesh('datasets/mesh.vtk')

.. note::

    If you intend to use the agglomearted mesh for numerical solvers, it is important to
    correctly extract the boundary elements and tags of the original mesh. To see how to do it,
    read the detailed documentation of :py:func:`~magnet.io.load_mesh`.

Then, initialize the agglomeration model you intend to use:

.. code-block:: python

    from magnet.io import aggmodels
    kmeans = aggmodels.KMEANS()

To agglomerate the mesh you then have to call the :py:meth:`~magnet.aggmodels.AgglomerationModel.agglomerate` method. 
For example, if we want to agglomearte the mesh by bisecting it recursively 7 times, having a total
of 128 agglomerated elements, you would use:

.. code-block:: python

    agg_mesh = kmeans.agglomerate(mesh, mode='Nref', nref=7)

Since :py:meth:`~magnet.aggmodels.AgglomerationModel.agglomerate` has a few different possible options, please check its
full documentation for further details.

Finally, you can plot the agglomerated mesh using :py:meth:`~magnet.mesh.AgglomerableMesh.view` and save it in `vtk`
format using :py:meth:`~magnet.mesh.AgglomerableMesh.save_mesh`.

.. code-block:: python

    agg_mesh.view()
    agg_mesh.save_mesh('outputs/aggmesh.vtk')

Quality metrics and model comparison
====================================

The :py:class:`~magnet.mesh.AgglomerableMesh` class provides some built-in methods to compute quality metrics
of an agglomerated mesh: this can be useful to evaluate the performance of a model.

To compute the quality metrics, you can call the respective methods (:py:meth:`~magnet.mesh.AgglomerableMesh.Circle_Ratio`,
:py:meth:`~magnet.mesh.AgglomerableMesh.Uniformity_Factor`, :py:meth:`~magnet.mesh.AgglomerableMesh.Volumes_Difference`), or
:py:meth:`~magnet.mesh.AgglomerableMesh.get_quality_metrics` to compute them together.

You can also do this on an entire dataset at the same time by using :py:meth:`~magnet.aggmodels.AgglomerationModel.agglomerate_dataset`
and :py:meth:`~magnet.mesh.AgglomerableMeshDataset.get_quality_metrics`

.. code-block:: python

    agg_dataset = mymodel.agglomerate_dataset(dataset)
    QM = agg_dataset.get_quality_metrics()

magnet provides also a :py:meth:`~magnet.mesh.AgglomerableMeshDataset.compare_quality` to automatically compare the performance of different
models on the same dataset by first agglomerating it and then computing the quality metrics.

.. code-block:: python

    from magnet.io import load_dataset
    from magnet import aggmodels
    km = aggmodels.KMEANS()
    mt = aggmodels.METIS()
    dataset = load_dataset('datasets/test_dataset')
    dataset.compare_quality([km, mt], mode='Nref', nref=5)