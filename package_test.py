import sys
import numpy
import scipy
import sklearn
import root_numpy
import ROOT
import rootpy
import matplotlib
import seaborn
import dask
import dask.distributed
import dask_searchcv

print('python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
print ('numpy version: {}'.format(numpy.__version__))
print ('scipy version: {}'.format(scipy.__version__))
print ('scikit-learn version: {}'.format(sklearn.__version__))
print ('root_numpy version: {}'.format(root_numpy.__version__))
print ('rootpy version: {}'.format(rootpy.__version__))
print ('matplotlib version: {}'.format(matplotlib.__version__))
print ('seaborn version: {}'.format(seaborn.__version__))
print ('dask version: {}'.format(dask.__version__))
print ('dask_searchcv version: {}'.format(dask_searchcv.__version__))
