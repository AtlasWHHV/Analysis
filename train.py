import argparse
import os
import time
import webbrowser

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import uniform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import constants
import dask_searchcv
import data
import simplekeras
import utils
from dask.distributed import Client
from loguniform import LogUniform


def create_model(model_name, local, hyper, n_iter=10):
  """Return an untrained sklearn model based on the given parameters."""
  if model_name == 'GBRT':
    model = GradientBoostingClassifier()
    param_grid = {'learning_rate': LogUniform(loc=-2, scale=1), 'n_estimators': LogUniform(loc=1, scale=2, discrete=True), 'max_depth': [2, 3, 4], 'subsample': uniform(loc=0.1, scale=0.9)}
  elif model_name == 'NB':
    model = GaussianNB()
    param_grid = {}
    n_iter = 1
  elif model_name == 'NN':
    model = Pipeline([('scale', StandardScaler()), ('nn', MLPClassifier())])
    param_grid = {'nn__alpha': LogUniform(loc=-7, scale=6)}
  elif model_name == 'SK':
    model = Pipeline([('scale', StandardScaler()), ('nn', simplekeras.classifier())])
    param_grid = {'nn__alpha': LogUniform(loc=-7, scale=6)}
  if hyper:
    if model_name != 'SK' and not local:
      model = dask_searchcv.RandomizedSearchCV(model, param_grid, n_iter=n_iter, cache_cv=False)
    else:
      model = sklearn.model_selection.RandomizedSearchCV(model, param_grid, n_iter=n_iter)
  return model

def train(model, X_train, y_train):
  print('[train] Beginning model training...')
  begin = time.time()
  model.fit(X_train, y_train)
  elapsed_time = time.time() - begin
  print('[train] Model training complete. Elapsed time = {} (s)'.format(elapsed_time))
  return model

def main():
  parser = argparse.ArgumentParser(description='Train and save a model.')
  parser.add_argument('--run_dir', default=None, help='The directory in which the testing data and trained model should be saved. It is intended that this directory be subsequently passed to downstream scripts such that all data pertaining to a single "run" (including testing data, a trained model, performance metrics, plots, etc.) is stored in this directory.')
  parser.add_argument('--local', action='store_true', help='Perform all calculations locally.')
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument('--no_hyper', action='store_true', help='Skip hyperparameterization.')
  group.add_argument('--n_iter', default=10, help='Number of possibilities to try in hyperparameter randomized search.')
  parser.add_argument('-m', '--model', choices=constants.MODEL_NAMES, help='Type of machine learning model to use.')
  parser.add_argument('--max_events', default=0, type=int, help='The maximum number of standard quark/gluon events to use in training (max_events = 0 indicates to use all of them)')
  parser.add_argument('--recalculate', action='store_true', help='Recalculate features from root files, instead of using pickled features.')
  args = parser.parse_args()
  X_train, y_train = data.get_features_and_labels(modified=False, recalculate=args.recalculate, max_events=args.max_events)
  X_test, y_test = data.get_features_and_labels(modified=True, recalculate=args.recalculate)
  if args.max_events == 0:
    args.max_events = y_train.size
  if not args.local and args.model in ['NN', 'NB', 'GBRT']:
    client = Client('tev01.phys.washington.edu:8786', timeout=10)
    webbrowser.open('http://tev01.phys.washington.edu:8787')

  model = create_model(args.model, args.local, not args.no_hyper, args.n_iter)
  if not args.run_dir:
    run_dir_name = '{}_{}'.format(args.model, args.max_events)
    if not args.no_hyper:
      run_dir_name += '_{}'.format(args.n_iter)
    args.run_dir = utils.make_run_dir(run_dir_name)
    print('[train] New run directory created at {}'.format(args.run_dir))
  else:
    args.run_dir = os.path.join(constants.RUNS_PATH, args.run_dir)
  utils.save_test(X_test, y_test, args.run_dir)
  model = train(model, X_train, y_train)
  utils.save_model(model, args.run_dir)

if __name__ == '__main__':
  main()
