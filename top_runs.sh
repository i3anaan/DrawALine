#!/bin/bash

set -o xtrace

#SMALL
python3 src/main.py --small --pca=37 --similarity=sim_cos lda #81.88%
python3 src/main.py --small --distort=shift --pca=45 svc #82.9%

#LARGE
python3 src/main.py --distort=shift mlp --alpha=0.0041152263 --activation=tanh #98.1+%
