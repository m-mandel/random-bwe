import sys
from pathlib import Path

path = Path(__file__).parent.parent.resolve()
sys.path.append(str(path))

import os
import argparse
from multiprocessing import Pool

import torchaudio
from data_prep.transforms import AddGaussianNoise

def create_noisy_subdir(data_dir, speaker_dir, data_subdir, out_dir):
    print(f'Adding noise to {data_subdir}')
    transform = AddGaussianNoise()
    out_sub_dir = os.path.join(out_dir, speaker_dir, data_subdir)
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
    for file in os.listdir(os.path.join(data_dir, data_subdir)):
        out_path = os.path.join(out_sub_dir, file)
        in_path = os.path.join(data_dir, data_subdir, file)
        if os.path.isfile(out_path):
            print(f'{out_path} already exists.')
        if not (file.lower().endswith('.wav') or file.lower().endswith('.flac')):
            print(f'{in_path}: invalid file type.')
        else:
            if not file.lower().endswith('.wav'):
                out_path = os.path.splitext(out_path)[0] + '.wav'
            signal, sr = torchaudio.load(in_path)
            signal = transform(signal)
            torchaudio.save(out_path, signal, sr)
            print(f'Successfully saved {in_path} to {out_path}')


def create_noisy_data(data_dir, out_dir):
    with Pool() as p:
        p.starmap(create_noisy_subdir,
                  [(os.path.join(data_dir, speaker_dir), speaker_dir, data_subdir, out_dir) for speaker_dir in os.listdir(data_dir)
                        for data_subdir in os.listdir(os.path.join(data_dir, speaker_dir))])


def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('--data_dir', help='directory containing source files')
    parser.add_argument('--out_dir', help='directory to write target files')
    return parser.parse_args()

"""Usage: python data_prep/create_noisy_data.py --data_dir <path for source data> --out_dir <path for target data>"""
def main():
    args = parse_args()
    print(args)

    create_noisy_data(args.data_dir, args.out_dir)
    print(f'Done adding noise.')


if __name__ == '__main__':
    main()