#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python /cs/labs/adiyoss/moshemandel/random-bwe/aero/data_prep/create_meta_file.py \
  --data_dir /cs/labs/adiyoss/moshemandel/random-bwe/LibriSpeech/train-clean-100-8k-wav-noisy \
  --target_dir /cs/labs/adiyoss/moshemandel/random-bwe/aero/src/egs \
  --json_filename total_noisy \
