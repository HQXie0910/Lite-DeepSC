# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:26:21 2020

@author: HQ Xie
"""
"""
This is to add stochastic noise to the channel
"""
import numpy as np
import math
import pickle
import torch.nn as nn

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
        
def SNR_to_noise(snr):
    snr = 10**(snr/10)
    noise_std = 1/np.sqrt(2*snr)
    
    return noise_std

def AddRealNoise(channel_matrix, min_snr, max_snr):
    min_noise = SNR_to_noise(min_snr)
    max_noise = SNR_to_noise(max_snr)
    
    sigma_n = np.random.uniform(0, 0.6, (1,))

    noise_CSI = channel_matrix + np.random.normal(0, sigma_n, channel_matrix.shape) 
    noise_level =  np.power(abs(channel_matrix), 0.5) + sigma_n

    return noise_CSI, noise_level


def create_channel_matrix(K):
    mean = math.sqrt(K/2 * (K + 1))
    std = math.sqrt(1/2 * (K + 1))
    
    H_real = np.random.normal(mean, std, (1,))
    H_imag = np.random.normal(mean, std, (1,))
    
    H = np.array([[H_real, -H_imag], [H_imag, H_real]])
    H = np.squeeze(H, axis = 2)

    return H

def create_dataset(train_num, test_num):
    # create train dataset 
    train_data = []
    for _ in range(train_num):
        # rayleigh channel
        train_data.append(create_channel_matrix(0))
        # rician channel
        # train_data.append(create_channel_matrix(1))
    
    test_data = []
    # create test dataset
    for _ in range(test_num):
        # rayleigh channel
        test_data.append(create_channel_matrix(0))
        # rician channel
        # test_data.append(create_channel_matrix(1))
        
    
    # save data
    file = open('./dataset/train_data.pickle', 'wb')
    pickle.dump(train_data, file)
    file.close()
    
    file = open('./dataset/test_data.pickle', 'wb')
    pickle.dump(test_data, file)
    file.close()

create_dataset(6000, 600)        



