import json
import math
import os

import numpy as np
import torch
import torch.utils.data as data


class ArrayDataset(data.Dataset):
    def __init__(self, json_dir, batch_size, segment_len=32000, cv_maxlen=64000):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of time series segment, when set to -1, use full time series

        xxx_infos is a list and each item is a tuple (array_file, #samples)
        """
        super(ArrayDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        s3_json = os.path.join(json_dir, 's3.json')
        ss_json = mix_json
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        with open(s3_json, 'r') as f:
            s3_infos = json.load(f)
        with open(ss_json, 'r') as f:
            ss_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        sorted_s3_infos = sort(s3_infos)
        sorted_ss_infos = sort(ss_infos)
        if segment_len >= 0.0:
            # segment length and count dropped utts
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} ) which is short than {} samples".format(
                drop_utt, drop_len, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2, part_s3, part_ss = [], [], [], [], []
                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size:
                            # if num_segments of 1st audio > batch_size, skip it
                            if start == end:
                                part_mix.append(sorted_mix_infos[end])
                                part_s1.append(sorted_s1_infos[end])
                                part_s2.append(sorted_s2_infos[end])
                                part_s3.append(sorted_s3_infos[end])
                                part_ss.append(sorted_ss_infos[end])
                                end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])
                        part_s3.append(sorted_s3_infos[end])
                        part_ss.append(sorted_ss_infos[end])
                    end += 1
                if len(part_mix) > 0:
                    minibatch.append([part_mix, part_s1, part_s2, part_s3, part_ss,
                                    segment_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch
        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long array to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen:
                    start = end
                    continue
                minibatch.append([sorted_mix_infos[start:end],
                                  sorted_s1_infos[start:end],
                                  sorted_s2_infos[start:end],
                                  sorted_s3_infos[start:end],
                                  sorted_ss_infos[start:end],
                                  segment_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class ArrayDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(ArrayDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    # wav --> npy
    mixtures, sources = load_mixtures_and_sources_npy(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    #print('mixtures_pad.shape {}'.format(mixtures_pad.shape))
    #print('ilens {}'.format(ilens))
    return mixtures_pad, ilens, sources_pad

# Eval data part
from preprocess import preprocess_one_dir

class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size):
        """
        Args:
            mix_dir: directory including mixture files
            mix_json: json file including mixture files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix')
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end]])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)
    
class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames

# ------------------------------ utils ------------------------------------

def load_mixtures_and_sources_npy(batch):
    """
    Each info include time series path and length.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, s3_infos, ss_infos, segment_len = batch
    # for each utterance
    for mix_info, s1_info, s2_info, s3_info, ss_info in zip(mix_infos, s1_infos, s2_infos, s3_infos, ss_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        s3_path = s3_info[0]
        ss_path = ss_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1] and s1_info[1] == s3_info[1]
        # read wav file
        mix = np.load(mix_path)
        s1 = np.load(s1_path)
        s2 = np.load(s2_path)
        s3 = np.load(s3_path)
        ss = np.load(ss_path)
        s1 = s1 /(max(abs(mix))+1e-7)
        s2 = s2 /(max(abs(mix))+1e-7)
        s3 = s3 /(max(abs(mix))+1e-7)
        ss = ss /(max(abs(mix))+1e-7)
        mix = mix/(max(abs(mix))+1e-7)
        # merge s1 and s2
        s = np.dstack((s1, s2, s3, ss))[0]  # T x C, C = 3
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segment
            for i in range(0, utt_len - segment_len + 1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    return mixtures, sources
def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos = batch[0]
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read npy file
        mix = np.load(mix_path)
        mix = mix/(max(abs(mix))+1e-7)
        print(mix_path)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad