# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import torch
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
import math
import os 

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0.0005
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(map(reversed, vocb_dictionary))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for letter in list_of_indices:
            if letter == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(letter))
        words = ' '.join(words)
        return(words) 
        
def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.cuda(), combined_mask.cuda()

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = math.sqrt(2)*torch.mean(x_square).sqrt()
    x = torch.div(x, power)
    
    return x
     
def train_step(model, src, trg, n_var, pad, opt, criterion, 
               channel, CSI, imperfect, is_csinet, CSINet):
    
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    
    opt.optimizer.zero_grad()
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    enc_output = model.encoder(src, src_mask)
        
    channel_enc_output = model.channel_encoder(enc_output)
    
    channel_enc_output = model.quant_constellation(channel_enc_output)
    
    channel_enc_output = PowerNormalize(channel_enc_output)
    
    if channel == 'AWGN':
        channel_enc_output = channel_enc_output \
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
                
    elif channel == 'Rayleigh':
        
        shape = channel_enc_output.shape
        K =  0
        mean = math.sqrt(K/2*(K + 1))
        std = math.sqrt(1/2*(K + 1))
        H_real = torch.normal(mean, std, size = [1]).cuda()
        H_imag = torch.normal(mean, std, size = [1]).cuda()
        
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).cuda()
        
        channel_enc_output = torch.matmul(channel_enc_output.view(shape[0], -1, 2), H)
        
        
        #model.RicianFadingChannel(channel_enc_output)
        
        channel_enc_output = channel_enc_output\
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
        
        if CSI:
            if imperfect:
                # use imperfect CSI to train the model
                H = H + torch.normal(0, n_var, size = H.shape).cuda()
            channel_enc_output = torch.matmul(channel_enc_output, torch.inverse(H)).view(shape)
        else:
            channel_enc_output = channel_enc_output.view(shape)
            
                
    elif channel == 'Rician':
        shape = channel_enc_output.shape
        K =  1
        mean = math.sqrt(K/2*(K + 1))
        std = math.sqrt(1/2*(K + 1))
        H_real = torch.normal(mean, std, size = [1]).cuda()
        H_imag = torch.normal(mean, std, size = [1]).cuda()
        
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).cuda()
        
        channel_enc_output = torch.matmul(channel_enc_output.view(shape[0], -1, 2), H)
        
        channel_enc_output = channel_enc_output\
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
                
        if CSI:
            if imperfect:
                # use imperfect CSI to train the model
                H = H + torch.normal(0, n_var, size = H.shape).cuda()
                if is_csinet:
                    CSINet.eval()
                    H = H.unsqueeze(0).unsqueeze(0)
                    H = CSINet(H)
                    H = H.squeeze(0).squeeze(0)
            channel_enc_output = torch.matmul(channel_enc_output, torch.inverse(H)).view(shape)
        else:
            channel_enc_output = channel_enc_output.view(shape)

    
    
    channel_dec_output = model.channel_decoder(channel_enc_output)
        
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        
    pred = model.dense(dec_output)

    ntokens = pred.size(-1)

    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    loss.backward()
    opt.step()
    
    return loss.item()

def test_step(model, src, trg, n_var, pad, criterion, channel, 
              CSI, imperfect, is_csinet, CSINet):
    
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    enc_output = model.encoder(src, src_mask)
        
    channel_enc_output = model.channel_encoder(enc_output)
    
    channel_enc_output = model.quant_constellation(channel_enc_output)
    
    channel_enc_output = PowerNormalize(channel_enc_output)
    
    if channel == 'AWGN':
        channel_enc_output = channel_enc_output \
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
                
    elif channel == 'Rayleigh':
        
        shape = channel_enc_output.shape
        K =  0
        mean = math.sqrt(K/2*(K + 1))
        std = math.sqrt(1/2*(K + 1))
        H_real = torch.normal(mean, std, size = [1]).cuda()
        H_imag = torch.normal(mean, std, size = [1]).cuda()
        
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).cuda()
        
        channel_enc_output = torch.matmul(channel_enc_output.view(shape[0], -1, 2), H)

        
        channel_enc_output = channel_enc_output\
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
        
        if CSI:
            if imperfect:
                H = H + torch.normal(0, n_var, size = H.shape).cuda()
            channel_enc_output = torch.matmul(channel_enc_output, torch.inverse(H)).view(shape)
        else:
            channel_enc_output = channel_enc_output.view(shape)
            
                
    elif channel == 'Rician':
        shape = channel_enc_output.shape
        K =  1
        mean = math.sqrt(K/2*(K + 1))
        std = math.sqrt(1/2*(K + 1))
        H_real = torch.normal(mean, std, size = [1]).cuda()
        H_imag = torch.normal(mean, std, size = [1]).cuda()
        
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).cuda()
        
        channel_enc_output = torch.matmul(channel_enc_output.view(shape[0], -1, 2), H)

        
        channel_enc_output = channel_enc_output\
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
        
        if CSI:
            if imperfect:
                H = H + torch.normal(0, n_var, size = H.shape).cuda()
                if is_csinet:
                    CSINet.eval()
                    H = H.unsqueeze(0).unsqueeze(0)
                    H = CSINet(H)
                    H = H.squeeze(0).squeeze(0)
            channel_enc_output = torch.matmul(channel_enc_output, torch.inverse(H)).view(shape)
        else:
            channel_enc_output = channel_enc_output.view(shape)
        

    
    channel_dec_output = model.channel_decoder(channel_enc_output)
        
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        
    pred = model.dense(dec_output)
    

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    
    return loss.item()
    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, 
                  channel, CSI, imperfect, is_csinet, CSINet):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).cuda() #[batch, 1, seq_len]
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    
    a = PowerNormalize(channel_enc_output)
    a = a.cpu().detach().numpy()
    np.save("riginal.npy", a)
    
    channel_enc_output = model.quant_constellation(channel_enc_output)
    
    channel_enc_output = PowerNormalize(channel_enc_output)
    
    b = channel_enc_output.cpu().detach().numpy()
    np.save("dequantize.npy", b)
    
    if channel == 'AWGN':
        channel_enc_output = channel_enc_output \
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
                
    elif channel == 'Rayleigh':
        
        shape = channel_enc_output.shape
        K =  0
        mean = math.sqrt(K/2*(K + 1))
        std = math.sqrt(1/2*(K + 1))
        H_real = torch.normal(mean, std, size = [1]).cuda()
        H_imag = torch.normal(mean, std, size = [1]).cuda()
        
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).cuda()
        
        channel_enc_output = torch.matmul(channel_enc_output.view(shape[0], -1, 2), H)
        
        
        #model.RicianFadingChannel(channel_enc_output)
        
        channel_enc_output = channel_enc_output\
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
        
        if CSI:
            if imperfect:
                # use imperfect CSI to train the model
                H = H + torch.normal(0, n_var, size = H.shape).cuda()
                if is_csinet:
                    CSINet.eval()
                    H = H.unsqueeze(0).unsqueeze(0)
                    H = CSINet(H)
                    H = H.squeeze(0).squeeze(0)
            
            channel_enc_output = torch.matmul(channel_enc_output, torch.inverse(H)).view(shape)
        else:
            channel_enc_output = channel_enc_output.view(shape)
            
                
    elif channel == 'Rician':
        shape = channel_enc_output.shape
        K =  1
        mean = math.sqrt(K/2* (K + 1))
        std = math.sqrt(1/2* (K + 1))
        H_real = torch.normal(mean, std, size = [1]).cuda()
        H_imag = torch.normal(mean, std, size = [1]).cuda()
        
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).cuda()
        
        channel_enc_output = torch.matmul(channel_enc_output.view(shape[0], -1, 2), H)
        
        
        #model.RicianFadingChannel(channel_enc_output)
        
        channel_enc_output = channel_enc_output\
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
        
        if CSI:
            if imperfect:
                # use imperfect CSI to train the model
                H = H + torch.normal(0, n_var, size = H.shape).cuda()
                if is_csinet:
                    CSINet.eval()
                    H = H.unsqueeze(0).unsqueeze(0)
                    H = CSINet(H)
                    H = H.squeeze(0).squeeze(0)
                    
            channel_enc_output = torch.matmul(channel_enc_output, torch.inverse(H)).view(shape)
        else:
            channel_enc_output = channel_enc_output.view(shape)
            
    #channel_enc_output = model.blind_csi(channel_enc_output)
          
    memory = model.channel_decoder(channel_enc_output)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.cuda()

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        
        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)
        
        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs

def SNR_to_noise(snr):
    snr = 10**(snr/10)
    noise_std = 1/np.sqrt(2*snr)
    
    return noise_std


