#!/bin/bash

DASK_SSH="$(pipenv --venv)/lib/python3.4/site-packages/distributed/cli/dask_ssh.py"
pipenv run python $DASK_SSH --scheduler tev01 --hostfile hostfile.txt

