import os
import tempfile
import types

import keras.models
import numpy as np
from sklearn.externals import joblib

import constants


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

make_keras_picklable()

def make_run_dir(name):
  run_dir = os.path.join(constants.RUNS_PATH, name)
  try:
    os.makedirs(run_dir)
  except OSError as e:
    print(e)
  return run_dir

def most_recent_dir():
  return max([os.path.join(constants.RUNS_PATH, d) for d in os.listdir(constants.RUNS_PATH)], key=os.path.getmtime)

def load_model(run_dir):
  model_path = os.path.join(run_dir, 'model.pkl')
  return joblib.load(model_path)

def save_model(model, run_dir):
  model_path = os.path.join(run_dir, 'model.pkl')
  joblib.dump(model, model_path)

def load_test(run_dir):
  X_test = np.load(os.path.join(run_dir, 'X_test.npy'))
  y_test = np.load(os.path.join(run_dir, 'y_test.npy'))
  return X_test, y_test

def save_test(X_test, y_test, run_dir):
  X_test_path = os.path.join(run_dir, 'X_test.npy')
  y_test_path = os.path.join(run_dir, 'y_test.npy')
  np.save(X_test_path, X_test)
  np.save(y_test_path, y_test)
