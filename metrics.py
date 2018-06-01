import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

import constants
import utils


def plot_roc(title, fname, X_test, y_test, model, c=1, show=False):
  y_score = model.predict_proba(X_test)[:, c]
  fpr, tpr, _ = roc_curve(y_test, y_score)
  AUC = auc(fpr, tpr)
  plt.clf()
  plt.plot(fpr, tpr, lw=2, drawstyle='steps-post', color='blue')
  plt.plot([0,1], [0,1], 'r--')
  plt.text(0.6, 0.2, "AUC = {:.3}".format(AUC), fontsize=17, weight=550)
  plt.xlim([0, 1])
  plt.ylim([0, 1.05])
  plt.xlabel('false positive rate', fontsize=15)
  plt.ylabel('true positive rate', fontsize=15)
  plt.title(title, fontsize=19)
  plt.savefig(fname)
  if show:
    plt.show()

def plot_sic(title, fname, X_test, y_test, model, c=1, show=False):
  y_score = model.predict_proba(X_test)[:, c]
  fpr, tpr, _ = roc_curve(y_test, y_score)
  sic = tpr / np.sqrt(fpr)
  plt.clf()
  plt.plot(tpr, sic, lw=2, drawstyle='steps-post', color='red')
  plt.xlabel('true positive rate', fontsize=15)
  plt.ylabel('tpr/sqrt(fpr)', fontsize=15)
  plt.title(title, fontsize=19)
  plt.savefig(fname)
  if show:
    plt.show()

def fixed_efficiency(X_test, y_test, model, c=1):
  y_score = model.predict_proba(X_test)[:, c]
  fpr, tpr, _ = roc_curve(y_test, y_score)
  return fpr[(np.abs(tpr - 0.5)).argmin()]

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Load a given model and calculate performance metrics (roc, sic, etc.).')
  parser.add_argument('--run_dir', default=None, help='The run directory that should be used (see train.py). If unspecified, the most recent run directory is used.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.most_recent_dir()
    print('[metrics] run_dir not specified, using {}'.format(args.run_dir))
  model = utils.load_model(args.run_dir)
  X_test, y_test = utils.load_test(args.run_dir)
  plot_roc('ROC curve', os.path.join(args.run_dir, 'roc_plot.png'), X_test, y_test, model, show=True)
  plot_sic('SIC', os.path.join(args.run_dir, 'sic_plot.png'), X_test, y_test, model, show=False)
  print('At TPR ~ 0.5, FPR = {}'.format(fixed_efficiency(X_test, y_test, model)))

if __name__ == '__main__':
  main()
