#!/usr/bin/env bash -l
conda create -y -n lkan python==3.10 &&
conda activate lkan &&
conda install -y -q cuda-nvcc &&
pip install -r requirements.txt &&
pip install ./lkancpp &&
pip install .