

import argparse

import torch
from data import ArrayDataLoader, ArrayDataset
from solver import Solver
#from amazing import FaSNet_base
from models import FaSNet_base,AE_base,FaSNet_base2,FaSNet_mod1
from utils import device

parser = argparse.ArgumentParser(
    "Time series anomaly detection network")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--train_dir2', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--segment_len', default=32000, type=int,
                    help='Segment length')
parser.add_argument('--cv_maxlen', default=64000, type=int,
                    help='max audio length in cv, to avoid OOM issue.')

# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default=False,
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# else
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--loss', default=0, type=int,
                    help='loss function type')
parser.add_argument('--multi_data', default=0, type=int,
                    help='If use cross dataset to train')
parser.add_argument('--fine_tune', default=0, type=int,
                    help='If to fine tune the model')
                    

def main(args):
    # Construct Solver
    # data
    tr_dataset = ArrayDataset(args.train_dir, args.batch_size,
                              segment_len=args.segment_len)
    cv_dataset = ArrayDataset(args.valid_dir, args.batch_size,  # 1 -> use less GPU memory to do cv
                              segment_len=args.segment_len, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    tr_dataset2 = ArrayDataset(args.train_dir2, args.batch_size,
                              segment_len=args.segment_len)
    tr_loader = ArrayDataLoader(tr_dataset, batch_size=1,
                                shuffle=args.shuffle)
    cv_loader = ArrayDataLoader(cv_dataset, batch_size=1)
    tr_loader2 = ArrayDataLoader(tr_dataset2, batch_size=1,
                                shuffle=args.shuffle)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader,'tr_loader2': tr_loader2}

    # model
    model = FaSNet_base(enc_dim=256, feature_dim=64, hidden_dim=128, layer=6, segment_size=250, nsplit = 3, win_len = 2).to(device)
    if args.fine_tune:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.enc_LN.parameters():
            param.requires_grad = False
        for param in model.separator.parameters():
            param.requires_grad = False
        for param in model.mask_conv1x1.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("done")
    if args.use_cuda:
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
