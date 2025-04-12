"""
Magnet
======
Mesh Agglomeration by Graph Neural Network

Provides
  1. Mesh agglomeration models.
  2. Framework to train and test GNN for mesh agglomeration.
  3. I-O of meshes.
"""

# import sys
import os

# If you have trouble importing metispy, set the path of the metis shared
# library manually if needed, like this:
os.environ['METIS_DLL'] = 'C:/METIS/libmetis/Release/metis.dll'

# to avoid memory leaks on windows due to kmeans:
# if sys.platform == 'win32':
os.environ['OMP_NUM_THREADS'] = '1'

from . import aggmodels
from . import generate
from . import io
