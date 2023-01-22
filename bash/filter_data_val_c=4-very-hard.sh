#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python /cs/labs/adiyoss/moshemandel/random-bwe/aero/data_prep/filter_data.py \
  --json_dir /cs/labs/adiyoss/moshemandel/random-bwe/aero/src/egs/2-8-very-hard/val \
  --out_dir /cs/labs/adiyoss/moshemandel/random-bwe/LibriSpeech/train-clean-100-filtered-very-hard-4 \
  --cutoff_ratio 4 \

