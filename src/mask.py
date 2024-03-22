import random as rand
import numpy as np
import torch

def mask_array(len0, arr):
    arr_length = len(arr)
    left = rand.randint(0, arr_length - len0)
    right = left + len0
    position = (left, right)
    arr[left:right] = torch.zeros([right - left])
    return arr, position