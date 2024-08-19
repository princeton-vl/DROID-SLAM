#!/bin/bash
set -eux

echo ">>> Start of post install script <<<"

conda activate droidenv
pip install evo --upgrade --no-binary evo
pip install gdown
# Install extensions
#python setup.py install

cd droid-slam
python setup.py install

echo ">>> End of post install script <<<"
