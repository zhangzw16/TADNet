import argparse
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pot.pot import pot_eval

parser = argparse.ArgumentParser("anomaly detection test")
parser.add_argument('--dimension', type=int, default=1,
                    help='dimension of the data')
parser.add_argument('--data_index', type=str, default=None,
                    help='the index of a certain test')

def smooth_filter(x):
    window_len = 101
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[50:len(y)-50]
    
def score(fname, dimension):
    total_score = 0
    for i in range(dimension):
        filename = fname
        filename = filename + '/signal' + str(i)
        smix = np.load(filename + '.npy')
        s1 = np.load(filename + '_s1.npy')
        s2 = np.load(filename + '_s2.npy')
        s3 = np.load(filename + '_s3.npy')
        ss = s1+s2+s3
        ss2 = s2+s3
        ss = ss - np.mean(ss)
        ss2 = ss2 - np.mean(ss2)
        smix = smix - np.mean(smix)
        score = ss-smix[:ss.shape[0]]
        if(np.sum(np.isnan(smix))<=0) and np.sum(smix==np.min(smix))+np.sum(smix==np.max(smix))<smix.shape[0]:
            total_score = total_score + score ** 2
        else:
            total_score = total_score + score ** 2
    return total_score[1:-1]

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    tr_s = score("test/separatet/" + args.data_index,args.dimension)
    tt_s = score("test/separate/" + args.data_index,args.dimension) 
    label_s = np.load("test/labels/labels" + args.data_index + ".npy") 
    labelsFinal = label_s
    if len(label_s.shape)>1:
      labelsFinal = (np.sum(label_s, axis=1) >= 1) + 0
    labelsFinal = labelsFinal[0:tt_s.shape[0]+2]
    if np.sum(tr_s)==0:
      print("error in tr_s, use tt_s instead")
      tr_s = tt_s/4
    result, pred_label = pot_eval(tr_s[2:-2], tt_s[2:-2], labelsFinal[3:-3])
    TP += result['TP']
    TN += result['TN']
    FN += result['FN']
    FP += result['FP']
    p = result['TP']/(result['TP']+result['FP'])
    r = result['TP']/(result['TP']+result['FN'])
    f1 = 2*p*r/(p+r)
    if not os.path.exists('results'):
        os.makedirs('results')
    f = open(f'results/{args.data_index}.txt','a')
    print(f'TP:{TP},FP:{FP},TN:{TN},FN:{FN},p:{p},r:{r},f1:{f1}')
    print(f'TP:{TP},FP:{FP},TN:{TN},FN:{FN},p:{p},r:{r},f1:{f1}',file=f)
    
    
    
