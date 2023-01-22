#!/bin/bash

. /cs/labs/adiyoss/moshemandel/random-bwe/aero/venv/bin/activate

python /cs/labs/adiyoss/moshemandel/random-bwe/aero/data_prep/split_meta_file.py \
  /cs/labs/adiyoss/moshemandel/random-bwe/aero/src/egs/alpha_0.5-1.0_beta_0-10.json \
  1800 \
  200 \
