import sys
sys.path.append('/cs/labs/adiyoss/moshemandel/random-bwe/aero')

import argparse
import json
import os
from itertools import groupby

import torchaudio
from pathlib import Path
from multiprocessing import Process
from multiprocessing import Pool

from tqdm import tqdm

from data_prep.low_pass_filter import LowPassFilter

def get_out_path(input_path, out_dir):
    parent_dirs = str(Path(input_path).parent).split(os.sep)[-2:]
    out_path = os.path.join(out_dir, *parent_dirs, Path(input_path).name)
    return out_path

def filter_and_save(input_path, alpha, beta, out_dir, cutoff_ratio):
    low_pass_filter = LowPassFilter(cutoff_ratio)
    out_path = get_out_path(input_path, out_dir)
    input, sr = torchaudio.load(input_path)
    out = low_pass_filter(input, alpha, beta)
    torchaudio.save(out_path, out, sr)
    print(f'Done filtering {Path(input_path).name}')

def filter_and_save_dir(dir_files_meta, out_dir, cutoff_ratio):
    for path, length, alpha, beta in dir_files_meta:
        filter_and_save(path, alpha, beta, os.path.join(out_dir, Path(path).name), cutoff_ratio)

def create_dirs(files_meta_by_dir, out_dir):
    for lst in files_meta_by_dir:
        input_path = Path(lst[0][0]).parent
        parent_dirs = str(input_path).split(os.sep)[-2:]
        out_path = os.path.join(out_dir, *parent_dirs)
        os.makedirs(out_path, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('--json_dir', help='path to dir containing json file')
    parser.add_argument('--out_dir', help='directory to write target files')
    parser.add_argument('--cutoff_ratio', type=int, help='target sample rate')
    return parser.parse_args()

def main():
    args = parse_args()

    json_dir = args.json_dir
    source_json_path = os.path.join(json_dir, 'hr.json')
    target_json_path = os.path.join(json_dir, 'lr.json')


    with open(source_json_path) as f:
        files_meta = json.load(f)
    files_meta_by_dir = [list(v) for i, v in groupby(files_meta, lambda x: Path(x[0]).parent)]
    create_dirs(files_meta_by_dir, args.out_dir)

    # low_pass_filter = LowPassFilter(args.cutoff_ratio)
    # for path, length, alpha, beta in tqdm(files_meta):
    #     name = Path(path).name
    #     parent_dirs = str(Path(path).parent).split(os.sep)[-2:]
    #     out_dir = os.path.join(args.out_dir, *parent_dirs)
    #     out_path = os.path.join(out_dir, name)
    #     input, sr = torchaudio.load(path)
    #     out = low_pass_filter(input, alpha, beta)
    #     torchaudio.save(out_path, out, sr)
    #     print(f'Done filtering {name}')

    #
    #
    print('filtering in parallel...')
    with Pool() as pool:
        pool.starmap_async(filter_and_save_dir,
                                 [(meta, args.out_dir, args.cutoff_ratio) for meta in files_meta_by_dir]).get()


    # processes = []
    # for path, length, alpha, beta in files_meta:
    #     parent_dirs = str(Path(path).parent).split(os.sep)[-2:]
    #     out_dir = os.path.join(args.out_dir, *parent_dirs)
    #     p = Process(target=filter_and_save, args=(path, alpha, beta, os.path.join(out_dir, Path(path).name), args.cutoff_ratio))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()






if __name__ == '__main__':
    main()