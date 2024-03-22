import argparse
import json
import os
import numpy as np

def preprocess_one_dir(in_dir, out_dir, out_filename):
    """
    preprocess one directory of npy files
    """
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    npy_list = os.listdir(in_dir)
    for npy_file in npy_list:
        if not npy_file.endswith('.npy'):
            continue
        npy_path = os.path.join(in_dir, npy_file)
        samples = np.load(npy_path)
        file_infos.append((npy_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

def preprocess_npy(args):
    """
    Preprocess the npy files
    """
    for data_type in ['tr', 'cv', 'tt', '']:
        for dt in ['mix', 's1', 's2', 's3']:
            if not os.path.exists(os.path.join(args.in_dir, data_type, dt)):
                continue
            preprocess_one_dir(os.path.join(args.in_dir, data_type, dt),
                               os.path.join(args.out_dir, data_type),
                               dt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("data preprocessing")
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Directory path of series data including tr, cv and tt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Directory path to put output files')
    args = parser.parse_args()
    preprocess_npy(args)