#!/bin/bash

conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg

conda install -c pytorch faiss-gpu

git clone https://github.com/drorlab/gvp-pytorch
cd gvp-pytorch
pip install .
cd ..

pip install -r /home/users/ziyiw23/COLLAPSE/requirements.txt

