"""load, store, and process quark/gluon data."""
import os
import math

import pandas as pd
import numpy as np
import pickle

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

