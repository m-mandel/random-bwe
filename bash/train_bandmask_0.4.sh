#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python train.py \
  dset=4-8 \
  experiment=aero_4-8_512_256 \
  bandmask=0.2 \
  'experiment.name=aero-bandmask-0.2-nfft-512-hl-256' \


