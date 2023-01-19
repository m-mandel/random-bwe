from src.metrics import run_metrics
from src.utils import bold
import multiprocessing as mp
import time
import itertools

import sys
import argparse
import torchaudio
from torch.nn import functional as F
import glob
import os
from pathlib import Path
import logging
import tqdm

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
my_format = logging.Formatter('%(message)s')
ch.setFormatter(my_format)

logger = logging.getLogger(__name__)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

MAX_SAMPLES_THRESHOLD = 481
MAX_RMS_THRESHOLD = 0.1
TEST_SUBDIRS=['p360', 'p361', 'p362', 'p363', 'p364', 'p374', 'p376', 's5']
N_SAMPLES_TO_TRIM=882

VISQOL_PATH = "/cs/labs/adiyoss/moshemandel/visqol-master"

def get_pr_dir_file_paths(pr_dir):
    return sorted(glob.glob(os.path.join(pr_dir, '*_pr.wav')))

def get_hr_dir_file_paths(hr_dir):
    return sorted(glob.glob(os.path.join(hr_dir, '*_hr.wav')))

def get_pr_hr_paths(samples_dir):
    pr_files = get_pr_dir_file_paths(samples_dir)
    hr_files = get_hr_dir_file_paths(samples_dir)
    for pr_file, hr_file in zip(pr_files, hr_files):
        file_prefix = Path(hr_file).stem.split('_')[0]
        assert (file_prefix in pr_file)
    return [(pr_file, hr_file) for pr_file, hr_file in zip(pr_files, hr_files)]

import librosa
def match_signal(signal, ref_len, filename):
    sig_len = signal.shape[-1]
    if sig_len < ref_len:
        logger.info(f'{filename} is too short. length: {sig_len}, ref. len: {ref_len}')
        signal = F.pad(signal, (0, ref_len - sig_len))
    elif sig_len > ref_len:
        logger.info(f'{filename} is too long. length: {sig_len}, ref. len: {ref_len}')
        signal = signal[..., :ref_len]
    return signal


def match_signals(signal, ref, filename, music_mode = False):
    if music_mode and signal.shape[-1] > N_SAMPLES_TO_TRIM and ref.shape[-1] > N_SAMPLES_TO_TRIM:
        signal = signal[:,:-N_SAMPLES_TO_TRIM]
        ref = ref[:,:-N_SAMPLES_TO_TRIM]
    sig_len = signal.shape[-1]
    ref_len = ref.shape[-1]
    if sig_len < ref_len:
        n_trimmed_samples = ref_len - sig_len
        residual_signal = ref[:, sig_len:]
        ref = ref[..., :sig_len]
    elif sig_len > ref_len:
        n_trimmed_samples = sig_len - ref_len
        residual_signal = signal[:, ref_len:]
        signal = signal[..., :ref_len]
    # logger.info(f'trimmed {n_trimmed_samples} samples')
    # logger.info(f'signal shape: {signal.shape}')
    rms = librosa.feature.rms(residual_signal.squeeze(0), frame_length=512, hop_length=256).squeeze(0)
    # logger.info(f'Average RMS: {sum(rms)/len(rms)}')
    # logger.info(f'Max RMS: {max(rms)}')
    if n_trimmed_samples > MAX_SAMPLES_THRESHOLD:
        logger.info(f'{filename} signal/reference diff exceeds threshold. sig len: {sig_len}, ref. len: {ref_len}')
        logger.info(f'trimmed samples: {n_trimmed_samples}')
    if max(rms) > MAX_RMS_THRESHOLD and not music_mode:
        logger.info(f'{filename}: Max RMS exceeds threshold: {max(rms)}')
        logger.info(f'residual shape: {residual_signal.shape}')
        logger.info(f'sig len: {sig_len}, ref. len: {ref_len}')
    return signal, ref


results_dict = {}
total_metrics_dict = {'lsd': [], 'visqol': []}
def log_result(result):
    if result['metrics']['lsd'] != 0:
        total_metrics_dict['lsd'].append(result['metrics']['lsd'])
    if result['metrics']['visqol'] != 0:
        total_metrics_dict['visqol'].append(result['metrics']['visqol'])
    results_dict[result['filename']] = result['metrics']

def apply_async_call(hr_path, pr_path, args):
    filename = Path(hr_path).stem
    # logger.info(f'evaluating {filename}')
    hr, hr_sr = torchaudio.load(hr_path)
    pr, pr_sr = torchaudio.load(pr_path)
    if hr.shape[-1] != pr.shape[-1]:
        pr, hr = match_signals(pr, hr, filename, music_mode=args.music_mode)

    lsd, visqol = run_metrics(hr, pr, args, filename)
    metrics = {'lsd': lsd,'visqol': visqol}
    return {'filename': filename, 'metrics': metrics}


def evaluate_on_dir(args):
    """

    :param args:
                    args.hr_dir
                    args.pr_dir
                    args.experiment.speech_mode
                    args.experiment.hr_sr
                    args.visqol

    :return:
    """
    pool = mp.Pool()
    pr_hr_paths = get_pr_hr_paths(args.samples_dir)
    start_time = time.time()
    logger.info('evaluating...')
    for i, (hr_path, pr_path) in enumerate(pr_hr_paths):
        pool.apply_async(apply_async_call, args=(hr_path, pr_path, args), callback=log_result)

    pool.close()
    pool.join()
    logger.info(f'execution time: {time.time() - start_time}')

    total_count = len(results_dict)
    lsd_count = len(total_metrics_dict['lsd'])
    visqol_count = len(total_metrics_dict['visqol'])
    avg_lsd = sum(total_metrics_dict['lsd'])/lsd_count
    avg_visqol = sum(total_metrics_dict['visqol'])/visqol_count
    logger.info(bold(
        f'{args.experiment.name}, {args.experiment.lr_sr}->{args.experiment.hr_sr}. Test set performance:LSD={avg_lsd} ({lsd_count}/{total_count}), VISQOL={avg_visqol} ({visqol_count}/{total_count}).'))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', type=str)
    parser.add_argument('--name', default='evaluation', type=str)
    parser.add_argument('--music_mode', action='store_true')

    parser.add_argument('--hr_sr', nargs="?", default=8000, type=int)
    parser.add_argument('--lr_sr', nargs="?", default=8000, type=int)
    parser.add_argument('--visqol', action='store_true')

    return parser


def update_args(args):
    d = vars(args)
    experiment = argparse.Namespace()
    experiment.name = args.name
    experiment.hr_sr = args.hr_sr
    experiment.lr_sr = args.lr_sr
    experiment.speech_mode = not args.music_mode
    d['experiment'] = experiment
    d['visqol_path'] = VISQOL_PATH


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    update_args(args)
    print(args)

    evaluate_on_dir(args)
