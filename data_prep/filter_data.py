import argparse
import json
import os
import torchaudio
from pathlib import Path

from data_prep.low_pass_filter import LowPassFilter


def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('--json_dir', type=int, help='path to dir containing json file')
    parser.add_argument('--out_dir', help='directory to write target files')
    parser.add_argument('--cutoff', type=int, help='target sample rate')
    return parser.parse_args()

def main():
    args = parse_args()

    json_dir = args.json_dir
    source_json_path = os.path.join(json_dir, 'hr.json')
    target_json_path = os.path.join(json_dir, 'lr.json')

    low_pass_filter = LowPassFilter(args.cutoff)
    files_meta = json.load(source_json_path)

    for path, length, alpha, beta in files_meta:
        name = Path(path).stem
        out_path = os.path.join(args.out_dir, name)
        in, sr = torchaudio.load(path)
        out = low_pass_filter(in, alpha, beta)
        torchaudio.save(out)





if __name__ == '__main__':
    main()