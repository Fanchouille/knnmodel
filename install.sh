#!/usr/bin/env bash

#export PIP_CONFIG_FILE=~/.pip.conf

rm -Rf .conda
rm -Rf .ipynb_checkpoints

conda env create -f environment.yml -p .conda

eval "$(conda shell.bash hook)"
# Create jupyter kernel with same name from environment.yml file
conda activate ${PWD}/.conda
pip list --format=freeze > requirements.txt
