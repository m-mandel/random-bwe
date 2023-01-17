

# import torch
# torch.multiprocessing.set_start_method('spawn', force=True)


import sys
from pathlib import Path
path = Path(__file__).parent.parent.resolve()
sys.path.append(str(path))

import argparse
import json
import os
from itertools import groupby

import torchaudio
# import torch.cuda

from torch.multiprocessing import Pool
# from multiprocessing import Process
# from multiprocessing import Pool

from tqdm import tqdm

from datetime import datetime

from data_prep.low_pass_filter import LowPassFilter
from data_prep.create_meta_file import create_meta

def get_out_path(input_path, out_dir):
    parent_dirs = str(Path(input_path).parent).split(os.sep)[-2:]
    out_path = os.path.join(out_dir, *parent_dirs, Path(input_path).name)
    return out_path

def format_time_diff(time_diff):
    minutes, seconds = divmod(int(time_diff.total_seconds()), 60)
    hours, minutes = divmod(minutes, 60)
    time_diff_string = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return time_diff_string

def filter_and_save(input_path, alpha, beta, out_dir, cutoff_ratio):
    low_pass_filter = LowPassFilter(cutoff_ratio)
    out_path = get_out_path(input_path, out_dir)
    if os.path.isfile(out_path):
        print(f'{out_path} already exists.')
        return
    print(f'Start filtering to {out_path}')
    start_time = datetime.now()
    input, sr = torchaudio.load(input_path)
    out = low_pass_filter(input, alpha, beta)
    torchaudio.save(out_path, out, sr)

    time_diff = datetime.now() - start_time
    print(f'Done filtering {Path(input_path).name} (time: {format_time_diff(time_diff)})')

def filter_and_save_dir(dir_files_meta, out_dir, cutoff_ratio):
    for path, length, alpha, beta in dir_files_meta:
        filter_and_save(path, alpha, beta, out_dir, cutoff_ratio)

def create_dirs(files_meta_by_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
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
    parser.add_argument('--constant_cutoff', action='store_true', help='target sample rate')
    return parser.parse_args()

def main():

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     torch.cuda.init()

    total_start_time = datetime.now()
    args = parse_args()

    json_dir = args.json_dir
    source_json_path = os.path.join(json_dir, 'hr.json')
    target_json_path = os.path.join(json_dir, 'lr.json')


    with open(source_json_path) as f:
        files_meta = json.load(f)
    files_meta_by_dir = [list(v) for i, v in groupby(files_meta, lambda x: Path(x[0]).parent)]
    create_dirs(files_meta_by_dir, args.out_dir)


    # # filtering in sequence
    # print('filtering in sequence...')
    # low_pass_filter = LowPassFilter(args.cutoff_ratio)
    # for path, length, alpha, beta in tqdm(files_meta):
    #     name = Path(path).name
    #     parent_dirs = str(Path(path).parent).split(os.sep)[-2:]
    #     out_dir = os.path.join(args.out_dir, *parent_dirs)
    #     out_path = os.path.join(out_dir, name)
    #     if os.path.isfile(out_path):
    #         print(f'{out_path} already exists.')
    #         continue
    #     start_time = datetime.now()
    #     input, sr = torchaudio.load(path)
    #     out = low_pass_filter(input, alpha, beta)
    #     torchaudio.save(out_path, out, sr)
    #     time_diff = datetime.now() - start_time
    #     print(f'Done filtering {name}. (time: {format_time_diff(time_diff)})')


    # filtering in parallel
    print('filtering in parallel...')
    args_list = [(meta[0],
                  meta[2] if not args.constant_cutoff else 0, # alpha
                  meta[3] if not args.constant_cutoff else 0, # beta
                  args.out_dir,
                  args.cutoff_ratio) for meta in files_meta]
    with Pool(processes=20) as pool:
        pool.starmap_async(filter_and_save, args_list).get()

    # processes = []
    # for path, length, alpha, beta in files_meta:
    #     parent_dirs = str(Path(path).parent).split(os.sep)[-2:]
    #     out_dir = os.path.join(args.out_dir, *parent_dirs)
    #     p = Process(target=filter_and_save, args=(path,
    #                                               alpha if not args.constant_cutoff else 0,
    #                                               beta if not args.constant_cutoff else 0,
    #                                               args.out_dir, args.cutoff_ratio))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    out_meta = create_meta(args.out_dir, append_random_data=False)
    out_json_data = json.dumps(out_meta, indent=4)
    with open(target_json_path, "w") as f:
        f.write(out_json_data)



    time_diff = datetime.now() - total_start_time
    print(f'Done. Total time: {format_time_diff(time_diff)}')






if __name__ == '__main__':
    main()