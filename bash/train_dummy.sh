#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python train.py \
  dset=4-8-dummy \
  experiment=aero_4-8_512_256 \
  epochs=10 \
