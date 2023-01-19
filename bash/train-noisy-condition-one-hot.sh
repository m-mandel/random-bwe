#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python train.py \
  dset=4-8-noisy \
  experiment=aero_4-8_512_256 \
  experiment.aero.condition_on_cutoff=true \
  'experiment.name=aero-condition-one-hot-nfft-512-hl-256' \

