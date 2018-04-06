import sys
import numpy
import scipy
import sklearn
import matplotlib
import seaborn
import dask
import dask.distributed
import dask_searchcv
import tensorflow
import h5py
import graphviz
import pydot
import uproot

print('python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
print('numpy version: {}'.format(numpy.__version__))
print('scipy version: {}'.format(scipy.__version__))
print('scikit-learn version: {}'.format(sklearn.__version__))
print('matplotlib version: {}'.format(matplotlib.__version__))
print('seaborn version: {}'.format(seaborn.__version__))
print('dask version: {}'.format(dask.__version__))
print('dask_searchcv version: {}'.format(dask_searchcv.__version__))
print('tensorflow version: {}'.format(tensorflow.__version__))
print('h5py version: {}'.format(h5py.__version__))
print('graphviz version: {}'.format(graphviz.__version__))
print('pydot version: {}'.format(pydot.__version__))
print('keras version: {}'.format(tensorflow.keras.__version__))
print('uproot version: {}'.format(pydot.__version__))
