import argparse
import json
import os
from pathlib import Path


def split_json_file(json_path, n_train_samples, n_test_samples):
    with open(json_path) as f:
        json_file = json.load(f)
    train_data = []
    test_data = []
    train_speakers = set()
    test_speakers = set()
    json_iterator = enumerate(json_file)
    for train_i in range(n_train_samples):
        file_idx, data = next(json_iterator)
        train_data.append(data)
        speaker = data[0].split('/')[-3]
        train_speakers.add(speaker)

    curr_speaker = speaker
    while curr_speaker == speaker:
        file_idx, data = next(json_iterator)
        speaker = data[0].split('/')[-3]

    for test_i in range(n_test_samples):
        if test_i > 0:
            file_idx, data = next(json_iterator)
        test_data.append(data)
        speaker = data[0].split('/')[-3]
        test_speakers.add(speaker)

    print(f'train speakers: {train_speakers}')
    print(f'test speakers: {test_speakers}')

    return train_data, test_data, train_speakers, test_speakers

def parse_args():
    parser = argparse.ArgumentParser(description='Split meta file.')
    parser.add_argument('json_filename_path', help='path to json file')
    parser.add_argument('n_train_samples_limit', type=int, help='limit number of train files')
    parser.add_argument('n_test_samples_limit', type=int, help='limit number of test files')
    return parser.parse_args()


"""
usage: python data_prep/split_meta_file.py <json_filename_path> <n_train_samples_limit> <n_test_samples_limit>
"""
def main():
    args = parse_args()
    print(args)

    json_name = Path(args.json_filename_path).stem
    target_dir = os.path.join(str(Path(args.json_filename_path).parent),
                              f'{json_name}_{args.n_train_samples_limit}_{args.n_test_samples_limit}')

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'tr'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)


    train_meta, test_meta, train_speakers, test_speakers = split_json_file(args.json_filename_path, args.n_train_samples_limit, args.n_test_samples_limit)

    train_json_object = json.dumps(train_meta, indent=4)
    test_json_object = json.dumps(test_meta, indent=4)
    with open(os.path.join(target_dir, 'tr', 'hr.json'), "w") as train_out:
        train_out.write(train_json_object)
    with open(os.path.join(target_dir, 'val', 'hr.json'), "w") as test_out:
        test_out.write(test_json_object)

    with open(os.path.join(target_dir, 'speakers.log'), "a") as speakers_log:
        speakers_log.write(f'train speakers: {train_speakers}\n')
        speakers_log.write(f'test speakers: {test_speakers}')


if __name__ == '__main__':
    main()