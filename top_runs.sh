#!/bin/bash

set -o xtrace

python3 src/main.py --small --pca=37 --similarity=sim_cos lda
