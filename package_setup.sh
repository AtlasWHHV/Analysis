#!/bin/bash
source /phys/users/gwatts/bin/CommonScripts/configASetup.sh
lsetup root
easy_install-2.7 --install-dir ~/.local/lib/python2.7/site-packages pip
~/.local/lib/python2.7/site-packages/pip install --upgrade --user numpy scipy pandas scikit-learn root_numpy rootpy matplotlib
python package_test.py
