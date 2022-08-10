
from adnet import ADNet
import time
import torch.nn as nn
import os
import torch
import numpy as np
from torch.autograd import Variable
import pickle
import math
from utlis import *

t1 = time.clock()
save_dir = './model_save/'
    
# Load dataset
with open('./dataset/train_data.pickle', 'rb') as file:
    train_data = pickle.load(file)

with open('./dataset/test_data.pickle', 'rb') as file:
    test_data = pickle.load(file)
    
# Build model
model = ADNet(channels=1)
model.cuda()

#net.apply(weights_init_kaiming)
criterion = nn.MSELoss(size_average=False)
criterion.cuda()

# Optimizer
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
noiseL_B=[0,55] # ingnored when opt.mode=='S'
psnr_list = [] 
batch_size = 60
mode = 'B'
noiseL_S = [SNR_to_noise(3)]
noiseL_B = [SNR_to_noise(0), SNR_to_noise(10)]

for epoch in range(120):
    train_losses = AverageMeter()
    test_losses = AverageMeter()
    
    if epoch <= 30:
        current_lr = lr
    if epoch > 30 and  epoch <=60:
        current_lr  =  lr/10.     
    if epoch > 60  and epoch <=90:
        current_lr = lr/100.
    if epoch >90 and epoch <=120:
        current_lr = lr/1000.
    
    st = time.time()
    # set learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    #print('learning rate %f' % current_lr)
    
    # train
    for i in range(math.ceil(len(train_data)/batch_size)):
        # training step
        model.train()
        
        data = train_data[i*batch_size : (i + 1)*batch_size]
        data = np.array(data) #[batch, H, W]
        img_train = torch.from_numpy(data.copy()).type(torch.FloatTensor).unsqueeze(1)
        #print(img_train.shape)
        if mode == 'S': #know the noise level
            noise = torch.normal(mean=0.0, std = noiseL_S[0], size = img_train.shape)
        if mode == 'B': #unknown the noise level
            noise = torch.zeros(img_train.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            #print(stdN)
            for n in range(noise.size()[0]):
                sizeN = noise[0,:,:,:].size()
                noise[n,:,:,:] = torch.normal(mean=0.0, std=stdN[n], size = sizeN) 
        #print(noise)
        imgn_train = img_train + noise
        #imgn_train = torch.matmul(imgn_train, imgn_train.transpose(-2, -1))
        #print(imgn_train.shape)
        img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda()) 
        noise = Variable(noise.cuda())  
        out_train = model(imgn_train)
        loss =  criterion(out_train, img_train) / (imgn_train.size()[0]*2)
        train_losses.update(loss.item())
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

    # validate
    for i in range(math.ceil(len(test_data)/batch_size)):
        model.eval()
        data = test_data[i*batch_size : (i + 1)*batch_size]
        data = np.array(data) #[batch, H, W]
        img_val = torch.from_numpy(data.copy()).type(torch.FloatTensor).unsqueeze(1).cuda()
        
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0.0, std=noiseL_S[0]).cuda()
        imgn_val = img_val + noise
        #imgn_val = torch.matmul(imgn_val, imgn_val.transpose(-2, -1))
        out_val = model(imgn_val)
        test_loss =  criterion(out_val, imgn_val) / (imgn_val.size()[0]*2)
        test_losses.update(test_loss.item())
    
    if epoch%20 == 0:
        model_name = 'model-B' + '.pth' 
        torch.save(model.state_dict(), os.path.join(save_dir, model_name)) 
    
    print('[{0}]\t' 
          'lr: {lr:.5f}\t'
          'Train Loss: {train_losses.val:.4f} ({train_losses.avg:.4f})\t'
          'Test Loss: {test_losses.val:.4f} ({test_losses.avg:.4f})\t'
          'Time: {time:.3f}'.format(epoch,
                 lr=optimizer.param_groups[-1]['lr'],
                 train_losses = train_losses,
                 test_losses = test_losses,
                 time = time.time() - st))
        
