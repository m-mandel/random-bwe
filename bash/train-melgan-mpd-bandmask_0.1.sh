#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python train.py \
  dset=4-8 \
  experiment=aero_4-8_512_256 \
  'experiment.discriminator_models=[ msd_melgan,mpd ]' \
  'experiment.name=aero-mpd-bandmask-0.1-nfft-512-hl-256' \
  bandmask=0.1 \


