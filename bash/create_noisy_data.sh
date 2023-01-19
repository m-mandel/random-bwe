#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python /cs/labs/adiyoss/moshemandel/random-bwe/aero/data_prep/create_noisy_data.py \
  --data_dir /cs/labs/adiyoss/moshemandel/random-bwe/LibriSpeech/train-clean-100-8k-wav \
  --out_dir /cs/labs/adiyoss/moshemandel/random-bwe/LibriSpeech/train-clean-100-8k-wav-noisy \

