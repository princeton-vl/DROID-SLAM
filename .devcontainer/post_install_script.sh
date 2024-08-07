#!/bin/bash
set -eux

echo ">>> Start of post install script <<<"

conda activate droidenv
pip install evo --upgrade --no-binary evo
# Install extensions
#python setup.py install
sudo -E env PATH=${PATH} python setup.py install

echo ">>> End of post install script <<<"
