#!/bin/bash

# Setup env
conda activate cs236781-hw
echo "hello from $(python --version) in $(which python)"

# Run some arbitrary python
python run_exe.py