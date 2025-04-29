from setuptools import setup
from setuptools import setup, find_packages

setup(
    name="Magnet",
    version="0.1.0",
    author="Magnet Team",
    description="Mesh agglomeration based on GNNs",
    url="https://github.com/lymphlib/magnet",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "metis-python",
        "scipy",
        "scikit-learn",
        "torch",
        "torch_geometric",
        "torch-cluster",
        "matplotlib",
        "networkx",
        "meshio",
        "vtk",
        "gmsh",
    ],
    python_requires=">=3.10",
)
