#!/bin/bash

DASK_SSH="$(pipenv --venv)/lib/python2.7/site-packages/distributed/cli/dask_ssh.py"
pipenv run python $DASK_SSH --scheduler tev01 --hostfile hostfile.txt

