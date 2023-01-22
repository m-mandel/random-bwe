#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python train.py \
  dset=2-8-hard \
  experiment=aero_4-8_512_256 \
  'experiment.discriminator_models=[ msd_melgan,mpd ]' \
  'experiment.name=aero-mpd-nfft-512-hl-256' \

