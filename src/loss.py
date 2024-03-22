import torch
import random
EPS = 1e-8

def cal_loss(source, estimate_source, source_lengths, position, loss_phase, epoch):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    if loss_phase == 0:
        mse = cal_loss_mse(source, estimate_source, source_lengths, position)
    elif loss_phase == 1:
        mse = cal_loss_mask(source, estimate_source, source_lengths, position)
    elif loss_phase == 2:
        mse = cal_loss_combined(source, estimate_source, source_lengths, position)
    elif loss_phase == 3:
        mse = cal_loss_without_noise(source, estimate_source, source_lengths, position)
    elif loss_phase == 4:
        mse = cal_loss_combined_without_noise(source, estimate_source, source_lengths, position)
    else:
        mse = cal_loss_rec(source, estimate_source, source_lengths, position)
    return mse, mse, estimate_source

def cal_loss_combined(source, estimate_source, source_lengths, position):
    # mse_loss + reconstruction_loss
    B, C, T = source.size()
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. MSE_loss
    mse = torch.sum((zero_mean_target[:,:zero_mean_estimate.shape[1],:] - zero_mean_estimate[:,:zero_mean_estimate.shape[1],:])**2) + torch.sum((zero_mean_target[:,-1,:] - torch.sum(zero_mean_estimate[:,:zero_mean_estimate.shape[1],:], dim=1))**2)
    return mse

    
def cal_loss_rec(source, estimate_source, source_lengths, position):
    # reconstruction_loss
    B, C, T = source.size()
    mask = get_mask(source, source_lengths)
    estimate_source *= mask
    
    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    zero_mean_target *= mask
    zero_mean_estimate *= mask


    # Step 2. MSE_loss
    mse = torch.sum((zero_mean_target[:,-1,:] - torch.sum(zero_mean_estimate[:,:zero_mean_estimate.shape[1],:], dim=1))**2)
    return mse
    

    
def cal_loss_mask(source, estimate_source, source_lengths, position):
    # masked_mse_loss
    B, C, T = source.size()
    mask = get_mask(source, source_lengths)
    estimate_source *= mask
    
    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. MSE_loss
    tt = torch.ones(1,8000) * 0.1
    tt[0,position[0][0]:position[0][1]] = 1
    tt = tt.cuda()
    print(zero_mean_target.shape)
    mse = torch.sum(((zero_mean_target[:,-1,:] - torch.sum(zero_mean_estimate[:,:zero_mean_estimate.shape[1],:], dim=1))**2)*tt)
    return mse

    
def cal_loss_without_noise(source, estimate_source, source_lengths, position):
    # 
    B, C, T = source.size()
    mask = get_mask(source, source_lengths)
    estimate_source *= mask
    
    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    zero_mean_target *= mask
    zero_mean_estimate *= mask


    # Step 2. MSE_loss
    mse = torch.sum((zero_mean_target[:,1:3,:] - zero_mean_estimate[:,1:3,:])**2)
    return mse

def cal_loss_combined_without_noise(source, estimate_source, source_lengths, position):
    B, C, T = source.size()
    mask = get_mask(source, source_lengths)
    estimate_source *= mask
    
    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    zero_mean_target *= mask
    zero_mean_estimate *= mask


    # Step 2. MSE_loss
    mse = torch.sum((zero_mean_target[:,1:3,:] - zero_mean_estimate[:,1:3,:])**2) + torch.sum((zero_mean_target[:,-1,:] - torch.sum(zero_mean_estimate[:,:zero_mean_estimate.shape[1],:], dim=1))**2)
    return mse
    


def moving_average(x, l):
    """
    x: torch.tensor
    l: int
    """
    kernel = torch.ones([1,1,l]).cuda() / l
    return torch.conv1d(x, kernel)
    

def cal_loss_mse(source, estimate_source, source_lengths, position):
    B, C, T = source.size()
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    zero_mean_target *= mask
    zero_mean_estimate *= mask
    

    # Step 2. MSE_loss
    mse = torch.sum((zero_mean_target[:,-1,:] - zero_mean_estimate)**2)
    return mse


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask