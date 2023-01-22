#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python /cs/labs/adiyoss/moshemandel/random-bwe/aero/data_prep/filter_data.py \
  --json_dir /cs/labs/adiyoss/moshemandel/random-bwe/aero/src/egs/alpha_0.5-1.0_beta_0.5-1.0_1800_200/tr-const \
  --out_dir /cs/labs/adiyoss/moshemandel/random-bwe/LibriSpeech/train-clean-100-filtered-hard-tr-const \
  --cutoff_ratio 2 \
  --constant_cutoff \

