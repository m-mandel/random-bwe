#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python train.py \
  dset=4-8 \
  experiment=aero_4-8_512_256 \
  revecho=0.5 \
  'experiment.name=aero-revecho-0.5-nfft-512-hl-256' \


