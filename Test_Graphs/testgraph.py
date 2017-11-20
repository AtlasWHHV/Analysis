# making some test graphs for quarks_modified

import root_numpy as rp
import matplotlib.pyplot as plt
import numpy as np

# loading and converting to a numpy array
arr = rp.root2array('/phys/groups/tev/scratch3/users/gwatts/IML_QG_Data/quarks_modified/REDUCED_quarks_modified_999.root')

# jetPt, jetEta, jetPhi, jetMass, ntracks, ntowers, trackPt, trackEta, trackPhi, trackCharge
# towerE, towerEem, tower Ehad, towerEta



# selecting branches
x = [arr['jetPt'],arr['jetEta'],arr['jetPhi'],arr['jetMass'],arr['ntracks'],arr['ntowers']]
title = arr.dtype.names

# plotting

for i in xrange(0, 6):
	f = plt.figure(i + 1)
	fig = f.add_subplot(1,1,1)

	fig.hist(x[i], normed=False, bins=200)

	fig.set_title(title[i])
	fig.set_xlabel('x-axis')
	fig.set_ylabel('y-axis')
	f.savefig(title[i] + '.png')

