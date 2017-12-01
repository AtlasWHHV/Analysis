""" Makes graphs of IML quark/gluon data. """
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import root_numpy as rp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data

def make_graphs(X, y, img_dir):
  # plot selected branches
  if not os.path.exists(img_dir):
    os.mkdir(img_dir)
  for branch_name in X:
    plt.hist(X.loc[y == 0, branch_name], bins=200, normed=True, label='quarks')
    plt.hist(X.loc[y == 1, branch_name], bins=200, normed=True, label='gluons')
    plt.title(branch_name)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(img_dir, branch_name + '.png'))
    plt.close()

def main():
  X, y = data.get_data()
  make_graphs(X, y, 'images')

if __name__ == '__main__':
  main()
