from const import pk_info
from generate import generator
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser('time series separation data generate')

parser.add_argument('--out_dir', type=str, default='data_out',
                    help='Directory putting separated series files')

def write_series(series, file_dir, num1, num2):

    print('generating ' + 'signal-' + str(num1) + '-' + str(num2))
    # load sequences
    trend_sequence = series[0]
    noise_sequence = series[1]
    season_sequence = series[2]
    full_sequence = series[3]

    # make folders 
    mix_dir = file_dir + '/mix'
    trend_dir = file_dir + '/s3'
    noise_dir = file_dir + '/s1'
    season_dir = file_dir + '/s2'

    if not os.path.exists(trend_dir):
        os.makedirs(trend_dir,exist_ok=True)
    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir,exist_ok=True)
    if not os.path.exists(season_dir):
        os.makedirs(season_dir,exist_ok=True)
    if not os.path.exists(mix_dir):
        os.makedirs(mix_dir,exist_ok=True)

    # write files
    np.save(mix_dir + '/signal-' + str(num1) + '-' + str(num2), full_sequence)
    np.save(trend_dir + '/signal-' + str(num1) + '-' + str(num2), trend_sequence)
    np.save(noise_dir + '/signal-' + str(num1) + '-' + str(num2), noise_sequence)
    np.save(season_dir + '/signal-' + str(num1) + '-' + str(num2), season_sequence)




def main(args):
    length =  len(pk_info)
    out_dir = args.out_dir
    num = 0
    for i in range(length):
        g = generator(pk_info[i])
        round = int(pk_info[i][-1])
        for j in range(round):
            series = g.generate()
            write_series(series, out_dir, i+1, j+1)
            num += 1


    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
