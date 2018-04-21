#!/bin/bash

# Check to see if we need to install pipenv.
if [[ !$(type -P "pipenv") ]]; then
  # Make sure that the user base binary directory is in the path.
  USER_BASE_BIN_PATH=$(python -m site --user-base)"/bin"
  if [[ ":$PATH:" != *":$USER_BASE_BIN_PATH:"* ]]; then
    echo "Warning: $USER_BASE_BIN_PATH not in PATH (necessary for use of local pipenv installation)!"
    # Add a command to ~/.bashrc adding the user base binary directory to the path
    # if such a command does not already exist.
    export PATH=$PATH:$USER_BASE_BIN_PATH
    if ! grep -s -q -F "$USER_BASE_BIN_PATH" ~/.bashrc; then
      printf "Do you wish to add:\nif [[ \":\$PATH:\" != *\":\$(python -m site --user-base)/bin:\"* ]]; then\n  export PATH=\$PATH:\$(python -m site --user-base)/bin\nfi\nto your ~/.bashrc for future use (recommended)?"
      select yn in "Yes" "No"; do
        case $yn in
          Yes ) printf "if [[ \":\$PATH:\" != *\":\$(python -m site --user-base)/bin:\"* ]]; then\n  export PATH=\$PATH:\$(python -m site --user-base)/bin\nfi" >> ~/.bashrc; break;;
          No ) break;;
        esac
      done
    fi
  fi

  # Make sure that setuptools is updated before installing pipenv.
  pip install -q --user -U setuptools

  # Install pipenv locally.
  pip install --user -U pipenv
fi

# Install the packages listed in the Pipfile
pipenv install

# Try to use the packages.
pipenv run python package_test.py
echo "NOTE: to use installed packages, prepend 'pipenv run' to your commands, or run 'pipenv shell' to activate the virtualenv."
