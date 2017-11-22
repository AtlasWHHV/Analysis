"""Learns quark/gluon discrimination using one of several scikit-learn models.
"""
import argparse
import sys
import os
import datetime
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import constants

def get_data(modified=False):
  if modified:
    pickle_file = os.path.join(constants.DATA_PATH, 'modified_data.pickle')
    quarks_path = constants.MODIFIED_QUARKS_PATH
    gluons_path = constants.MODIFIED_GLUONS_PATH
  else:
    pickle_file = os.path.join(constants.DATA_PATH, 'standard_data.pickle')
    quarks_path = constants.STANDARD_QUARKS_PATH
    gluons_path = constants.STANDARD_GLUONS_PATH
  # Try to load the data from a pickle file first, otherwise use root numpy.
  try:
    with open(pickle_file, 'rb') as fh:
      df_quarks, df_gluons = pickle.load(fh)
  except (IOError, OSError) as e:
    # Normally, imports should be at the top of the module. However, root_numpy
    # has undesirable side-effects when it is imported, to such an extent that
    # it breaks argparse when it is imported. As such, it is not imported until
    # absolutely necessary, in an attempt to limit the damage it does.
    from root_numpy import root2array
    df_quarks = pd.DataFrame(root2array(quarks_path, branches=constants.BRANCH_NAMES))
    df_gluons = pd.DataFrame(root2array(gluons_path, branches=constants.BRANCH_NAMES))
    with open(pickle_file, 'wb+') as fh:
      pickle.dump((df_quarks, df_gluons), fh)
  df_quark_labels = pd.Series(np.zeros(df_quarks.shape[0], dtype=np.uint8))
  df_gluon_labels = pd.Series(np.ones(df_gluons.shape[0], dtype=np.uint8))
  X = pd.concat([df_quarks, df_gluons])
  y = pd.concat([df_quark_labels, df_gluon_labels])
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
  args = parser.parse_args()
  X_standard, y_standard = get_data(modified=False)
  X_modified, y_modified = get_data(modified=True)
  if args.model == 'GBRT':
    param_grid = {'learning_rate': [0.1, 0.4, 0.7, 1.0], 'n_estimators': [50, 100, 150], 'max_depth': [2, 3, 4], 'subsample': [0.5, 0.75, 1.0]}
    model = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=3, verbose=50)
    grid = True
  elif args.model == 'NB':
    model = GaussianNB()
    grid = False
  elif args.model == 'NN':
    pipeline = Pipeline([('scale', StandardScaler()), ('nn', MLPClassifier())])
    param_grid = {'nn__alpha': (10.0 ** -np.arange(1, 7)).tolist()}
    model = GridSearchCV(pipeline, param_grid, cv=3, verbose=50)
    grid = True
  X_eval, y_eval = train(model, X_standard, y_standard, print_report=True)
  if grid:
    print_grid_performance(model)
  timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
  os.mkdir(timestamp)
  standard_plot_path = os.path.join(timestamp, 'standard_qg_roc.png')
  modified_plot_path = os.path.join(timestamp, 'modified_qg_roc.png')
  plot_roc(model, X_eval, y_eval, standard_plot_path, 'Receiver Operating Characteristic (Standard Data)', 'orange')
  plot_roc(model, X_modified, y_modified, modified_plot_path, 'Receiver Operating Characteristic (Modified Data)', 'darkred')
  plt.show()

if __name__ == "__main__":
  main()
