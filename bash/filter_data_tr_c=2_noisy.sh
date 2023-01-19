#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python /cs/labs/adiyoss/moshemandel/random-bwe/aero/data_prep/filter_data.py \
  --json_dir /cs/labs/adiyoss/moshemandel/random-bwe/aero/src/egs/total_noisy_alpha_0.9-1.0_beta_0.9-1.0_1800_200/tr \
  --out_dir /cs/labs/adiyoss/moshemandel/random-bwe/LibriSpeech/train-clean-100-filtered-tr-noisy \
  --cutoff_ratio 2 \

