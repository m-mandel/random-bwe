import sox
import os
import sys
import argparse
import glob
import torchaudio
from collections import namedtuple
import json
from multiprocessing import Process, Manager
import pathlib
import random

SEED = 2036
MIN_ALPHA=0.9
MAX_ALPHA=1.0
MIN_BETA=0.9
MAX_BETA=1.0
FILE_PATTERN = '*.wav'
TOTAL_N_SPEAKERS = 108
TRAIN_N_SPEAKERS = 100
TEST_N_SPEAKERS = 8

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def add_subdir_meta(subdir_path, shared_meta, n_samples_limit, append_random_data):
    if n_samples_limit and len(shared_meta) > n_samples_limit:
        return
    print(f'creating meta for {subdir_path}')
    audio_files = glob.glob(os.path.join(subdir_path, FILE_PATTERN))
    for idx, file in enumerate(audio_files):
        info = get_info(file)

        if append_random_data:
            alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
            beta = random.uniform(MIN_BETA, MAX_BETA)
            data = (file, info.length, alpha, beta)
        else:
            data = (file, info.length)

        shared_meta.append(data)



def create_subdirs_meta(subdirs_paths, n_samples_limit, append_random_data):
    with Manager() as manager:
        shared_meta = manager.list()
        processes = []
        for subdir_path in subdirs_paths:
            p = Process(target=add_subdir_meta, args=(subdir_path, shared_meta, n_samples_limit, append_random_data))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        meta = list(shared_meta)
        meta.sort()
        if n_samples_limit:
            meta = meta[:n_samples_limit]
        return meta


def create_meta(data_dir, n_samples_limit=None, append_random_data=True):
    subdirs_paths = [os.path.join(data_dir, speaker_dir, subdir) for speaker_dir in os.listdir(data_dir)
                            for subdir in os.listdir(os.path.join(data_dir, speaker_dir))]
    subdirs_paths.sort()
    print(f'number of speakers: {len(os.listdir(data_dir))}')
    print(f'total number of subdirs: {len(subdirs_paths)}')
    meta = create_subdirs_meta(subdirs_paths, n_samples_limit, append_random_data)

    print(f'total number of files: {len(meta)}')

    if n_samples_limit:
        assert len(meta) == n_samples_limit

    return meta


def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('--data_dir', help='directory containing source files')
    parser.add_argument('--target_dir', help='output directory for created json files')
    parser.add_argument('--json_filename', help='filename for created json files')
    parser.add_argument('--n_samples_limit', type=int, help='limit number of files')
    return parser.parse_args()


"""
usage: python data_prep/create_meta_file.py <data_dir_path> <target_dir> <json_filename>
"""


def main():
    args = parse_args()
    print(args)

    random.seed(SEED)

    os.makedirs(args.target_dir, exist_ok=True)

    meta = create_meta(args.data_dir, args.n_samples_limit)

    total_data_json_object = json.dumps(meta, indent=4)
    with open(os.path.join(args.target_dir,
                           f'{args.json_filename}_alpha_{MIN_ALPHA}-{MAX_ALPHA}_beta_{MIN_BETA}-{MAX_BETA}.json'),
                                                            "w") as train_out:
        train_out.write(total_data_json_object)

    print(f'Done creating meta for {args.data_dir}.')


if __name__ == '__main__':
    main()