
import os
import time
import utils
import pickle
import torch
import math
import torch.nn as nn
import numpy as np
from hyperparameter import Hyperparameter as hp
from transformer_constellation import Transformer
from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.data import BucketIterator
from adnet import ADNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint_path = './pre-trained Model'

stored_path = './Pruned Model'
label = '99'
PATH =  stored_path + './traceiver-Rician-imperfect_CSI-4bits-' + label + '.pth.tar'

""" preparing the dataset """      
raw_data = pickle.load(open('./data/train_data.pkl', 'rb'))
train_data, test_data = raw_data

tokenize = lambda x: x.split()
#tokenize = get_tokenizer("basic_english"),
# pad_token = '<pad>'
SRC = Field(sequential=True, use_vocab = True,
            tokenize = tokenize,
            init_token = '<start>', 
            eos_token = '<end>', 
            fix_length = hp.MAX_LENGTH, 
            lower = True,
            batch_first = True)

TRG = Field(sequential=True, use_vocab = True,
            tokenize = tokenize,
            init_token = '<start>', 
            eos_token = '<end>', 
            fix_length = hp.MAX_LENGTH, 
            lower = True,
            batch_first = True)

examples_train = []
examples_test = []
fields = [('src', SRC), ('trg', TRG)]

for sent in train_data:
    examples_train.append(data.Example.fromlist([sent, sent], fields))
    
for sent in test_data:
    examples_test.append(data.Example.fromlist([sent, sent], fields))


train = Dataset(examples = examples_train, fields = fields)
test = Dataset(examples = examples_test, fields = fields)

SRC.build_vocab(train.src, test.src)
TRG.build_vocab(train.trg, test.trg)

train_iterator = BucketIterator(train, batch_size = hp.BATCH_SIZE,
                                device = device, sort_key = lambda x: len(x.src), # field sorted by len
                                sort_within_batch = True, shuffle = True)

test_iterator = BucketIterator(test, batch_size =  hp.BATCH_SIZE,
                               device = device, sort_key = lambda x: len(x.src), # field sorted by len
                               sort_within_batch = True, shuffle = True)

pad_idx = TRG.vocab.stoi["<pad>"]
start_symbol = SRC.vocab.stoi["<start>"]
vocb_dictionary = SRC.vocab.stoi.items()
end_idx = SRC.vocab.stoi["<end>"]

StoT = utils.SeqtoText(vocb_dictionary, end_idx)
bleu_score_1gram = utils.BleuScore(1, 0, 0, 0)
best_prec1 = 0

channel = 'Rician' #'AWGN', 'Rician', 'Rayleigh'
CSI = True
imperfect = True
is_csinet = True

if os.path.exists(checkpoint_path + '/ADNet/' + '/model-B.pth'):
    # load existing model
    model_info = torch.load(checkpoint_path + '/ADNet/' +  '/model-B.pth')
    print('==> loading existing model:', checkpoint_path + '/ADNet/' + '/model-B.pth')
    CSINet = ADNet(channels = 1)
    CSINet.cuda()
    CSINet.load_state_dict(model_info)
else:
    print('No Model')
# set the seed
# torch.manual_seed(10)

def main():
    global best_prec1


    # create model
    model = Transformer(hp.num_layers, len(SRC.vocab), len(TRG.vocab), 
                    len(SRC.vocab), len(TRG.vocab), hp.d_model, hp.num_heads, 
                    hp.dff, hp.dropout_rate).to(device)
    
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(PATH), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction = 'none').to(device)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.001, betas=(0.9, 0.98), eps=1e-9)


    for epoch in range(hp.finetune_epoch):
        
        adjust_learning_rate(optimizer, epoch)

        #####################################################################################################
        num_parameters = get_linear_zero_param(model)
        print('Zero parameters: {}'.format(num_parameters)) 
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))
        #####################################################################################################
        # train for one epoch
        print('************************ Train *****************************')
        train(train_iterator, model, criterion, optimizer, epoch, 0.1,
              channel, CSI, imperfect, is_csinet, CSINet)

        # evaluate on validation set
        print('************************ Test *****************************')
        prec1 = validate(test_iterator, model, criterion, channel, CSI, imperfect, is_csinet, CSINet)

        # remember best prec@1 and save checkpoint
        # 'optimizer' : optimizer.state_dict()
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = max(prec1, best_prec1)
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1}, 
                        stored_path  + './refine/traceiver-Rician-imperfect_CSI-4bits-' + label + '.pth.tar')
            print('Model Saved')
    
        
    return

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def get_linear_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += torch.sum(m.weight.data.eq(0))
    return total

def train(train_loader, model, criterion, optimizer, epoch, n_var,
          channel, CSI, imperfect, is_csinet, CSINet):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        src = batch.src
        trg = batch.trg

        # compute output
        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]
    
    
        src_mask, look_ahead_mask = utils.create_masks(src, trg_inp, pad_idx)
    
        enc_output = model.encoder(src, src_mask)
        
        channel_enc_output = model.channel_encoder(enc_output)
        
        channel_enc_output = model.quant_constellation(channel_enc_output)
    
        channel_enc_output = utils.PowerNormalize(channel_enc_output)
    
        if channel == 'AWGN':
            channel_enc_output = channel_enc_output \
                + torch.normal(0, n_var, size = channel_enc_output.shape).cuda()
                
        elif channel == 'Rayleigh':
        
            shape = channel_enc_output.shape
            K =  0
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
        
        channel_dec_output = model.channel_decoder(channel_enc_output)
        
        dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        
        pred = model.dense(dec_output)
        
        ntokens = pred.size(-1)
        loss = utils.loss_function(pred.contiguous().view(-1, ntokens), 
                                   trg_real.contiguous().view(-1), 
                                   pad_idx, criterion)
        
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Linear):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask) 

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

def validate(test_loader, model, criterion, channel, CSI, imperfect, is_csinet, CSINet):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        word = []
        target_word = []
        
        for i, batch in enumerate(test_loader):
            
            src = batch.src
            trg = batch.trg
            #print(src.shape)

            """" compute the loss value """
            loss = utils.test_step(model, src, trg, 0.1, pad_idx, criterion,
                                   channel, CSI, imperfect, is_csinet, CSINet)
            losses.update(loss)
            
            """ use bleu score as accuracy tools"""
            output = utils.greedy_decode(model, src, 0.1, hp.MAX_LENGTH, 
                                         pad_idx, start_symbol, channel, CSI, imperfect, is_csinet, CSINet)
            
            sentences = output.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, sentences))
            word = word + result_string
            
            target_sent = trg.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, target_sent))
            target_word = target_word + result_string
            
            bleu_score = bleu_score_1gram.compute_blue_score(word, target_word)
            #print(bleu_score)
            bleu_score = np.array(bleu_score)
            score = np.mean(bleu_score)
            accuracy.update(score)

            """ measure elapsed time """
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 5 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
                      .format(i, len(test_loader), batch_time=batch_time, 
                              loss = losses, acc = accuracy))

        print('Prec@1 {acc.avg:.3f}'.format(acc=accuracy))

    return accuracy.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
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

if __name__ == '__main__':
    main()

