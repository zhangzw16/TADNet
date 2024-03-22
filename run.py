import numpy as np
import os
import argparse
parser = argparse.ArgumentParser("run the pipeline")
parser.add_argument('--file_dir', type=str, default=None,
                    help='directory including training info')
parser.add_argument('--mode', type=str, default=None,
                    help='run mode')
parser.add_argument('--pretrain', type=int, default=None,
                    help='with pretraining or not')
parser.add_argument('--loss', type=str, default=None,
                    help='loss function type')
parser.add_argument('--number', type=str, default=None,
                    help='number of experiment')
parser.add_argument('--exists', type=bool, default=False,
                    help='exists or not')

def trans_train(in_file):
    out = "data_c/dataset/tr/"
    out_v = "data_c/dataset/cv/"
    dir_set = ["mix", "s1", "s2", "s3"]
    for i in dir_set:
        os.makedirs(os.path.join(out,i),exist_ok=True)
        os.makedirs(os.path.join(out_v,i),exist_ok=True)
    data = np.load(in_file)
    data = data.transpose()
    j = 0
    if len(data.shape)==1:
        data = np.expand_dims(data, axis = 0)
    # train_dataset
    while data.shape[1] > 100000:
        data = data[:,100000:]
        for i in range(int(data.shape[0])):
            for k in range(4):
                np.save(out + dir_set[k] + '/signal' + str(i+j*int(data.shape[0])), data[i])
        j = j + 1 
    for i in range(int(data.shape[0])):
        for k in range(4):
            np.save(out + dir_set[k] + '/signal' + str(i+j*int(data.shape[0])), data[i])
    # valid_dataset
    for i in range(int(data.shape[0])):
        for k in range(4):
            np.save(out_v + dir_set[k] + '/signal' + str(i+j*int(data.shape[0])), data[i])

def trans_test(in_file):
    out = "data_c/dataset/tt/"
    dir_set = ["mix", "s1", "s2", "s3"]
    for i in dir_set:
        os.makedirs(os.path.join(out,i),exist_ok=True)
    data = np.load(in_file)
    data = data.transpose()
    if len(data.shape)==1:
        data = np.expand_dims(data, axis = 0)
    for i in range(int(data.shape[0])):
        for k in range(4):
            np.save(out + dir_set[k] + '/signal' + str(i), data[i])


if __name__ == '__main__':
    args = parser.parse_args()
    # train
    if args.mode == "train":
        print("processing data")
        trans_train(args.file_dir)
        trans_test(args.file_dir.replace('train','test'))
        os.system("python src/preprocess.py --in-dir data_c/dataset --out-dir data_/dataset")
        print("training")
        if args.exists:
            os.system(f"python src/train.py --train_dir data_/dataset/tr --valid_dir data_/dataset/cv --segment_len 8000 --cv_maxlen 32000 --use_cuda 1 --epochs 10 --half_lr 1 --early_stop 0 --max_norm 5 --shuffle 1 --batch_size 12 --optimizer adam --lr 5e-4 --momentum 0 --l2 0 --save_folder experiments/{args.number}/ --checkpoint 1 --continue_from 'experiments/{args.number}/temp_best.pth.tar' --print_freq 1000 --loss {args.loss} --jt 0 --train_dir2 data_/data_sp")
        else:
            os.system(f"python src/train.py --train_dir data_/dataset/tr --valid_dir data_/dataset/cv --segment_len 8000 --cv_maxlen 32000 --use_cuda 1 --epochs 15 --half_lr 1 --early_stop 0 --max_norm 5 --shuffle 1 --batch_size 12 --optimizer adam --lr 1e-3 --momentum 0 --l2 0 --save_folder experiments/{args.number}/  --checkpoint 1  --print_freq 1000 --loss {args.loss} --train_dir2 data_/data_sp")
    # test
            
    elif args.mode == "test":
        print("processing data")
        trans_train(args.file_dir.replace('test','train'))
        trans_test(args.file_dir)
        os.system("python src/preprocess.py --in-dir data_c/dataset --out-dir data_/dataset")
        print("testing")
        if args.exists:
            os.system(f"python src/separate.py --model_path experiments/{args.number}/temp_best.pth.tar --mix_json data_/dataset/tt/mix.json --out_dir test/separate/{args.number} --use_cuda 1 --batch_size 12")
            os.system(f"python src/separate.py --model_path experiments/{args.number}/temp_best.pth.tar --mix_json data_/dataset/tr/mix.json --out_dir test/separatet/{args.number}  --use_cuda 1 --batch_size 12")
        labels = np.load(args.file_dir.replace('test','labels'))
        if not os.path.exists('test/labels'):
            os.makedirs('test/labels',exist_ok=True)
        np.save('test/labels/labels'+ args.number , labels)
        os.system("python src/test.py --data_index " + args.number)

    # train_by_synthetic
    elif args.mode == "train_by_synthetic":
        print("processing data")
        os.system("python synthetic/main.py --out_dir data_c/data_sp")
        os.system("python src/preprocess.py --in-dir data_c/data_sp --out-dir data_/data_sp")
        print("training")
        if args.exists:
            os.system(f"python src/train.py --train_dir data_/data_sp --valid_dir data_/data_sp --segment_len 8000 --cv_maxlen 32000 --use_cuda 1 --epochs 10 --half_lr 1 --early_stop 0 --max_norm 5 --shuffle 1 --batch_size 12 --optimizer adam --lr 5e-4 --momentum 0 --l2 0 --save_folder experiments/{args.number}/ --checkpoint 1 --continue_from 'experiments/{args.number}/temp_best.pth.tar' --print_freq 1000 --loss {args.loss} --jt 0 --train_dir2 data_/data_sp")
        else:
            os.system(f"python src/train.py --train_dir data_/data_sp --valid_dir data_/data_sp --segment_len 8000 --cv_maxlen 32000 --use_cuda 1 --epochs 15 --half_lr 1 --early_stop 0 --max_norm 5 --shuffle 1 --batch_size 12 --optimizer adam --lr 1e-3 --momentum 0 --l2 0 --save_folder experiments/{args.number}/  --checkpoint 1  --print_freq 1000 --loss {args.loss} --train_dir2 data_/data_sp")
        
