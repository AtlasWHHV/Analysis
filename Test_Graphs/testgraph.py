# making some test graphs for quarks_modified

import os
import root_numpy as rp
import matplotlib.pyplot as plt
import numpy as np

# load the quark root file and convert it to a numpy structured array
quarks_tree = rp.root2array('/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/quarks_modified/REDUCED_quarks_modified_999.root')

# plot selected branches
img_dir = 'images'
if not os.path.exists(img_dir):
  os.mkdir(img_dir)
branch_names = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'ntracks', 'ntowers']
for branch_name in branch_names:
  plt.hist(quarks_tree[branch_name], bins=200)
  plt.title(branch_name)
  plt.savefig(os.path.join(img_dir, branch_name + '.png'))
  plt.close()
