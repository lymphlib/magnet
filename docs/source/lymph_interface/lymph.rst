===============
Lymph interface
===============

.. toctree::
    :hidden:
    
    lymphcomm

magnet provides an interface with ``Lymph`` (polytopal discountinuous Galerking for multi-physics).
The :py:mod:`~magnet.lymphcomm.py` module offers a main function (wrapped in ``bin/lymphcomm.py``) that can be called from Matlab to agglomerate a mesh using the package and return the
relevant information to Lymph, while the ``Agglomerate.m`` function is used to call the script and for
conversion of the mesh to the Lymph format proper.

Getting Started
===============

The same steps for magnet installation should be followed with some caveats: to call Python modules in MATLAB®,
you must have a supported version of the reference implementation (CPython) installed on your system.
See the MATLAB® support for more details on supported versions and installation.

After having installed a compatible version, create a virtual environment and install the :ref:`relevant packages <installation>`.
in ``Agglomerate.m`` you will need to specify the path of the Python executable in the virtual environment.
To get the location of the executable, first activate the virtual environment, start Python and run:

.. code:: console
    
    $ python
    >>> import sys 
    >>> sys.executable 

This will return the path of the executable, e.g. 'C:\Users\username\myvenv\Scripts\python' on Windows.
You will need to insert it in ``pyenv`` inside the ``Agglomerate.m`` function.

.. code-block:: matlab
    :emphasize-lines: 2

    pyenv('Version', ... 
          'C:\Users\username\myvenv\Scripts\python', ... 
          'ExecutionMode','OutOfProcess');

You are now ready to agglomerate.

Example usage
=============

.. code-block:: matlab

    AggTestLap  % Definition of the problems data
    SimType = 'laplacian';

    mesh_path = 'datasets/meshtest.vtk';
    output_path = fullfile('lymph\Physics\Laplacian\InputMesh', 'AggMeshTest.mat');

    model = 'KMEANS';
    mode = 'Nref';
    param = 7;

    [~, ~] = Agglomerate(mesh_path, output_path, ...
                         Data, SimType, ...
                         model, mode, param);

After the agglomerated mesh is saved, you can use it in ``RunMainLaplacian.m``.


