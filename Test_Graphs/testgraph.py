# Makes graphs of IML quark/gluon data.
import os
import root_numpy as rp
import matplotlib.pyplot as plt
import numpy as np

def make_graphs(root_path, img_dir):
  # load the root file and convert it to a numpy structured array
  branch_names = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'ntracks', 'ntowers']
  tree = rp.root2array(root_path, branches=branch_names)

  # plot selected branches
  if not os.path.exists(img_dir):
    os.mkdir(img_dir)
  for branch_name in branch_names:
    plt.hist(tree[branch_name], bins=200, normed=True)
    plt.title(branch_name)
    plt.savefig(os.path.join(img_dir, branch_name + '.png'))
    plt.close()

def main():
  make_graphs('/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/quarks_modified/*', 'quark_images')
  make_graphs('/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/gluons_modified/*', 'gluon_images')

if __name__ == '__main__':
  main()
