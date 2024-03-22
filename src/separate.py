import argparse
import os
import torch
import numpy as np
import shutil
import time
from data import EvalDataLoader, EvalDataset
from models import FaSNet_base
from utils import remove_pad


parser = argparse.ArgumentParser('Separate time-series using FaSNet_base')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--mix_dir', type=str, default=None,
                    help='Directory including mixture series files')
parser.add_argument('--mix_json', type=str, default=None,
                    help='Json file including mixture series files')
parser.add_argument('--out_dir', type=str, default='exp/result',
                    help='Directory putting separated series files')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample_len', default=8000, type=int,
                    help='sample_len')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')


def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    
    model = FaSNet_base(enc_dim=256, feature_dim=64, hidden_dim=128, layer=6, segment_size=250, nsplit=3, win_len=2)
    model_state = torch.load(args.model_path)
    model.load_state_dict(model_state)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(args.mix_dir, args.mix_json,
                               batch_size=args.batch_size)
    eval_loader =  EvalDataLoader(eval_dataset, batch_size=1)
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    time_sum = 0
    with torch.no_grad():
        for (i, data) in enumerate(eval_loader):
            # Get batch data
            mixture, mix_lengths, filenames = data
            if len(mixture.shape)>2:
                print(mixture[0,:,0])
                mixture = mixture[:,:,0]
            if args.use_cuda:
                mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()
            # Forward
            t1 = time.time()
            estimate_source = model(mixture)  # [B, C, T]
            t2 = time.time()
            time_sum += t2 - t1
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)
            mixture = remove_pad(mixture, mix_lengths)
            
            # Write result
            for i, filename in enumerate(filenames):
                
                filename = os.path.join(args.out_dir,
                                        os.path.basename(filename).strip('.npy'))
               # mixture[i] = mixture[i]/max(abs(mixture[i]))
                np.save(filename, mixture[i])
                C = flat_estimate[i].shape[0]
                for c in range(C):
                    np.save(filename + '_s{}'.format(c+1), flat_estimate[i][c])
    print('Average time: {}'.format(time_sum / len(eval_loader)))
                


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    separate(args)


