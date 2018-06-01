import os

BASE_PATH = '/phys/groups/tev/scratch3/users/WHHV/qg_analysis/'
RUNS_PATH = os.path.join(BASE_PATH, 'runs')
PICKLE_PATH = os.path.join(BASE_PATH, 'data')
MODIFIED_QUARKS_ROOT_PATH = '/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/quarks_modified/*'
MODIFIED_GLUONS_ROOT_PATH = '/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/gluons_modified/*'
STANDARD_QUARKS_ROOT_PATH = '/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/quarks_standard/*'
STANDARD_GLUONS_ROOT_PATH = '/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/gluons_standard/*'
FEATURES = ['jetMass', 'ntracks', 'ntowers', 'width', 'dispersion', 'EMF', 'charge', 'n90']
MODEL_NAMES = ['NB', 'NN', 'GBRT', 'SK']
