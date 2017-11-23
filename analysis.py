"""Learns quark/gluon discrimination using one of several scikit-learn models.
"""
from __future__ import division
import argparse
import sys
import os
import datetime
import pickle
import math
import webbrowser

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dask_searchcv import GridSearchCV
from dask.distributed import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import constants

def calculate_jets(dataframe):
  with np.errstate(divide='ignore', invalid='ignore'):
    dataframe['dispersion'] = dataframe.apply(lambda r: np.sqrt(np.sum(np.square(r['trackPt']))) / np.sum(r['trackPt']), axis=1)
    dataframe.loc[~np.isfinite(dataframe['dispersion']), 'dispersion'] = 0.0
  def width(row):
    deltaPhi = np.absolute(row['jetPhi']-row['trackPhi'])
    deltaPhi = np.minimum(deltaPhi, 2*math.pi - deltaPhi)
    deltaEta = row['jetEta']-row['trackEta']
    deltaR = np.sqrt(np.square(deltaPhi) + np.square(deltaEta))
    return np.sum(np.multiply(row['trackPt'], deltaR)) / row['jetPt']
  dataframe['width'] = dataframe.apply(width, axis=1)
  return dataframe.drop(columns=['trackPt', 'trackEta', 'trackPhi', 'trackCharge', 'towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi'])

def get_data(modified=False, recalculate=False, max_events=0):
  if modified:
    pickle_file = os.path.join(constants.DATA_PATH, 'modified_data.pickle')
    quarks_path = constants.MODIFIED_QUARKS_PATH
    gluons_path = constants.MODIFIED_GLUONS_PATH
  else:
    pickle_file = os.path.join(constants.DATA_PATH, 'standard_data.pickle')
    quarks_path = constants.STANDARD_QUARKS_PATH
    gluons_path = constants.STANDARD_GLUONS_PATH
  if recalculate or not os.path.exists(pickle_file):
    # Normally, imports should be at the top of the module. However, root_numpy
    # has undesirable side-effects when it is imported, to such an extent that
    # it breaks argparse when it is imported. As such, it is not imported until
    # absolutely necessary, in an attempt to limit the damage it does.
    from root_numpy import root2array
    df_quarks = calculate_jets(pd.DataFrame(root2array(quarks_path)))
    df_gluons = calculate_jets(pd.DataFrame(root2array(gluons_path)))
    with open(pickle_file, 'wb+') as fh:
      pickle.dump((df_quarks, df_gluons), fh)
  else:
    with open(pickle_file, 'rb') as fh:
      df_quarks, df_gluons = pickle.load(fh)
  df_quark_labels = pd.Series(np.zeros(df_quarks.shape[0], dtype=np.uint8))
  df_gluon_labels = pd.Series(np.ones(df_gluons.shape[0], dtype=np.uint8))
  if max_events > 0:
    X = pd.concat([df_quarks[:max_events], df_gluons[:max_events]])
    y = pd.concat([df_quark_labels[:max_events], df_gluon_labels[:max_events]])
  else:
    X = pd.concat([df_quarks, df_gluons])
    y = pd.concat([df_quark_labels, df_gluon_labels])
  X.reset_index(drop=True, inplace=True)
  y.reset_index(drop=True, inplace=True)
  return X, y

def plot_roc(model, X, y, plot_path, title, color):
  y_score = model.predict_proba(X)
  fpr, tpr, thresholds = roc_curve(y, y_score[:,1])
  roc_auc = auc(fpr,tpr) 
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color=color,
      lw=lw, label='ROC curve (area = {:0.2})'.format(roc_auc))
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(title)
  plt.legend(loc="lower right")
  plt.savefig(plot_path)
  plt.draw()

def train(model, X, y, print_report=False):
  X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, random_state=1)
  model.fit(X_dev, y_dev)
  if print_report:
    eval_prediction = model.predict(X_eval)
    print("Classification report on evaluation set:")
    print()
    print(classification_report(y_eval, eval_prediction, target_names=['quark', 'gluon']))
    print()
    sns.set()
    mat = confusion_matrix(y_eval, eval_prediction)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label');\
  return X_eval, y_eval

def print_grid_performance(model):
  print("Best parameters on development set:")
  print()
  print(model.best_params_)
  print()
  print("Scores for each parameter combination on development set:")
  print()
  for mean, std, params in zip(model.cv_results_['mean_test_score'], model.cv_results_['std_test_score'], model.cv_results_['params']):
    print("{:0.3} (+/-{:0.03}) for {!r}".format(mean, 2 * std, params))
  print()
 
def main():
  parser = argparse.ArgumentParser(description='Analyze quark/gluon separation using a particular machine learning model.')
  parser.add_argument('-m', '--model', choices=['NB', 'NN', 'GBRT'], required=True, help='Type of machine learning model to use.')
  parser.add_argument('--recalculate', action='store_true', help='If set, recalculate data from root files, instead of using pickled data.')
  parser.add_argument('--max_events', default=0, type=int, help='The maximum number of standard quark/gluon events to use in training (max_events <= 0 indicates to use all of them)')
  args = parser.parse_args()
  X_standard, y_standard = get_data(modified=False, recalculate=args.recalculate, max_events=args.max_events)
  X_modified, y_modified = get_data(modified=True, recalculate=args.recalculate)
  if args.model == 'GBRT' or args.model == 'NN':
    grid = True
    client = Client('localhost:8786')
    webbrowser.open('http://localhost:8787')
  else:
    grid = False
  if args.model == 'GBRT':
    param_grid = {'learning_rate': [0.01, 0.1], 'n_estimators': [10, 100, 1000], 'max_depth': [2, 3, 4], 'subsample': [0.1, 0.5, 1.0]}
    model = GridSearchCV(GradientBoostingClassifier(), param_grid, scheduler=client)
  elif args.model == 'NB':
    model = GaussianNB()
  elif args.model == 'NN':
    pipeline = Pipeline([('scale', StandardScaler()), ('nn', MLPClassifier())])
    param_grid = {'nn__alpha': (10.0 ** -np.arange(1, 7)).tolist()}
    model = GridSearchCV(pipeline, param_grid, scheduler=client)
  X_eval, y_eval = train(model, X_standard, y_standard, print_report=True)
  if grid:
    print_grid_performance(model)
  timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
  os.mkdir(timestamp)
  with open(os.path.join(timestamp, 'args.txt'), 'w+') as fh:
    fh.write(str(args))
  standard_plot_path = os.path.join(timestamp, 'standard_qg_roc.png')
  modified_plot_path = os.path.join(timestamp, 'modified_qg_roc.png')
  plot_roc(model, X_eval, y_eval, standard_plot_path, 'Receiver Operating Characteristic (Standard Data)', 'orange')
  plot_roc(model, X_modified, y_modified, modified_plot_path, 'Receiver Operating Characteristic (Modified Data)', 'darkred')
  plt.show()

if __name__ == "__main__":
  main()
