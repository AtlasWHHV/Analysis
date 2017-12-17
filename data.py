"""load, store, and process quark/gluon data."""
import os
import math

from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle

import constants

def update_jets(df_jets, root_path):
  missing_features = []
  for feature in constants.FEATURES:
    if feature not in df_jets:
      missing_features.append(feature)
  if missing_features:
    from root_numpy import root2array
    df_jets_raw = pd.DataFrame(root2array(root_path))
    if 'jetMass' in missing_features:
      df_jets['jetMass'] = df_jets_raw['jetMass']
    if 'ntracks' in missing_features:
      df_jets['ntracks'] = df_jets_raw['ntracks']
    if 'ntowers' in missing_features:
      df_jets['ntowers'] = df_jets_raw['ntowers']
    if 'dispersion' in missing_features:
      print('calculating dispersion')
      with np.errstate(divide='ignore', invalid='ignore'):
        df_jets['dispersion'] = df_jets_raw.apply(lambda r: np.sqrt(np.sum(np.square(r['trackPt']))) / np.sum(r['trackPt']), axis=1)
        df_jets.loc[~np.isfinite(df_jets['dispersion']), 'dispersion'] = 0.0
    if 'width' in missing_features:
      print('calculating width')
      def width(row):
        deltaPhi = np.absolute(row['jetPhi']-row['trackPhi'])
        deltaPhi = np.minimum(deltaPhi, 2*math.pi - deltaPhi)
        deltaEta = row['jetEta']-row['trackEta']
        deltaR = np.sqrt(np.square(deltaPhi) + np.square(deltaEta))
        return np.sum(np.multiply(row['trackPt'], deltaR)) / row['jetPt']
      df_jets['width'] = df_jets_raw.apply(width, axis=1)
    if 'n90' in missing_features:
      print('calculating n90')
      def n90(row):
        if row['towerE'].size == 0:
          return 0
        towerE = np.sort(row['towerE'])[::-1]
        targetE = 0.9 * np.sum(towerE)
        return np.argmax(np.cumsum(towerE)>targetE)
      df_jets['n90'] = df_jets_raw.apply(n90, axis=1)
    if 'EMF' in missing_features:
      print('calculating EMF')
      with np.errstate(divide='ignore', invalid='ignore'):
        df_jets['EMF'] = df_jets_raw.apply(lambda r: np.sum(r['towerEem']) / np.sum(r['towerE']), axis=1)
        df_jets.loc[~np.isfinite(df_jets['EMF']), 'EMF'] = 0.0
    if 'charge' in missing_features:
      print('calculating charge')
      df_jets['charge'] = df_jets_raw.apply(lambda r: np.sum(r['trackCharge']), axis=1)
      return len(missing_features) != 0

def load_jets(recalculate, pickle_path, root_path):
  if recalculate or not os.path.exists(pickle_path):
    df_jets = pd.DataFrame()
  else:
    df_jets = pd.read_pickle(pickle_path)
  if update_jets(df_jets, root_path):
    df_jets.to_pickle(pickle_path)
  return df_jets

def get_features_and_labels(modified=False, recalculate=False, max_events=0):
  if modified:
    quarks_pickle_path = os.path.join(constants.PICKLE_PATH, 'modified_quarks.pkl')
    gluons_pickle_path = os.path.join(constants.PICKLE_PATH, 'modified_gluons.pkl')
    quarks_root_path = constants.MODIFIED_QUARKS_ROOT_PATH
    gluons_root_path = constants.MODIFIED_GLUONS_ROOT_PATH
  else:
    quarks_pickle_path = os.path.join(constants.PICKLE_PATH, 'standard_quarks.pkl')
    gluons_pickle_path = os.path.join(constants.PICKLE_PATH, 'standard_gluons.pkl')
    quarks_root_path = constants.STANDARD_QUARKS_ROOT_PATH
    gluons_root_path = constants.STANDARD_GLUONS_ROOT_PATH
  df_quarks = load_jets(recalculate, quarks_pickle_path, quarks_root_path)
  df_gluons = load_jets(recalculate, gluons_pickle_path, gluons_root_path)
  df_quark_labels = pd.Series(np.zeros(df_quarks.shape[0], dtype=np.uint8))
  df_gluon_labels = pd.Series(np.ones(df_gluons.shape[0], dtype=np.uint8))
  if max_events > 0:
    X = pd.concat([df_quarks[:max_events], df_gluons[:max_events]])
    y = pd.concat([df_quark_labels[:max_events], df_gluon_labels[:max_events]])
  else:
    X = pd.concat([df_quarks, df_gluons])
    y = pd.concat([df_quark_labels, df_gluon_labels])
  X, y = shuffle(X, y)
  X.reset_index(drop=True, inplace=True)
  y.reset_index(drop=True, inplace=True)
  return X, y

def main():
  X_standard, y_standard = get_features_and_labels(modified=False)
  X_mod, y_mod = get_features_and_labels(modified=True)
  print(X_standard.head())
  print(X_mod.head())

if __name__ == '__main__':
  main()
