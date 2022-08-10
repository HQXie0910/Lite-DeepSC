
import torch
import os
import pickle
import time
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import*
from torchtext import data
from torchtext.data import ReversibleField, Field, Dataset
from torchtext.data import BucketIterator
from torchtext.data.utils import get_tokenizer
from hyperparameter import Hyperparameter as hp
from quant_transformer import QuantTransformer
from adnet import ADNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" preparing the dataset """      
raw_data = pickle.load(open('./data/train_data.pkl', 'rb'))
train_data, test_data = raw_data

tokenize = lambda x: x.split()

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

""" define optimizer and loss function """

transformer = QuantTransformer(hp.num_layers, len(SRC.vocab), len(TRG.vocab), 
                               len(SRC.vocab), len(TRG.vocab), hp.d_model, hp.num_heads, 
                               hp.dff, hp.dropout_rate, a_bits = hp.abits, w_bits = hp.wbits).to(device)

criterion = nn.CrossEntropyLoss(reduction = 'none')
optimizer = torch.optim.Adam(transformer.parameters(), 
                             lr=0, betas=(0.9, 0.98), eps=1e-9)
opt = NoamOpt(hp.d_model, 1, 2000, optimizer)

pad_idx = TRG.vocab.stoi["<pad>"]
src_pad_idx = SRC.vocab.stoi["<pad>"]
vocb_dictionary = SRC.vocab.stoi.items()
end_idx = SRC.vocab.stoi["<end>"]
# print(pad_idx, src_pad_idx)
# print(train_iterator.__dict__.keys())
checkpoint_path = './Pruned Model'
pre_trained_path = './pre-trained Model'
quantized_path = './Quantized Model'
pretrained_model = './traceiver-Rician-imperfect_CSI-4bits-95'
quant = '-a16-b16'
channel = 'Rician' #'AWGN', 'Rician', 'Rayleigh'
CSI = True
imperfect = True
is_csinet = True

if os.path.exists(pre_trained_path + '/ADNet/' + '/model-B.pth'):
    # load existing model
    model_info = torch.load(pre_trained_path + '/ADNet/' +  '/model-B.pth')
    print('==> loading existing model:', pre_trained_path + '/ADNet/' + '/model-B.pth')
    CSINet = ADNet(channels = 1)
    CSINet.cuda()
    CSINet.load_state_dict(model_info)
else:
    print('No Model')
    
record_loss = []

"""################################# stastic quantization ###############################
# this module is used for quantizing the pre-trained model"""
if hp.stastic_quantization:
    PATH = checkpoint_path  + pretrained_model  + '.pth.tar'
    checkpoint = torch.load(PATH)

    pretrained_dict = checkpoint['state_dict']
    transformer_dict = transformer.state_dict()

    new_dict = {k: v for k, v in pretrained_dict.items() if k in transformer_dict.keys()}
    transformer_dict.update(new_dict) # 创建新的 model_state_dict()
    transformer.load_state_dict(transformer_dict)
    
    # calibration 
    for i, batch in enumerate(test_iterator):
        transformer.eval()
        test_loss = test_step(transformer, batch.src, batch.trg, 
                                  0.1, pad_idx, criterion, channel, 
                                  CSI, imperfect, is_csinet, CSINet)
        if i > 5:
            break
        else:
            print("test loss: %f" % test_loss)
    
    
    PATH_QUANT = quantized_path  + pretrained_model + quant + '.pth.tar'
    torch.save({'state_dict': transformer.state_dict(), 
                'test_loss': test_loss, }, PATH_QUANT)
    print("Calibration is finished!")

    
    
"""################################# quantization-aware training ###############################
# this module is used for recover the precision of system
"""
if hp.QAT:
    # load pre-trained model
    PATH = quantized_path  + pretrained_model + quant + '.pth.tar'
    checkpoint = torch.load(PATH)

    pretrained_dict = checkpoint['state_dict']
    transformer_dict = transformer.state_dict()

    new_dict = {k: v for k, v in pretrained_dict.items() if k in transformer_dict.keys()}
    transformer_dict.update(new_dict)
    transformer.load_state_dict(transformer_dict)
    
    for epoch in range(10):
        start = time.time()
        total = 0
        test_total = 0
        ##########################################################################
        transformer.train()
        for i, batch in enumerate(train_iterator):
            loss = train_step(transformer, batch.src, batch.trg, 
                              0.1, pad_idx, opt, criterion, channel, 
                              CSI, imperfect, is_csinet, CSINet)
            total += loss
    
        if (epoch + 1)%5 == 0:
            PATH = checkpoint_path + pretrained_model + quant + '.pth.tar'
            torch.save({'state_dict': transformer.state_dict(), 
                        'best_loss': total/200}, PATH)
            print("Model save: Epoch Step: %d " % epoch)
    
        record_loss.append(total/200)
        
        ##########################################################################
        transformer.eval()
        for i, batch in enumerate(test_iterator):
            test_loss = test_step(transformer, batch.src, batch.trg, 
                                  0.1, pad_idx, criterion, channel, 
                                  CSI, imperfect, is_csinet, CSINet)
            test_total += test_loss
            
        ##########################################################################
        elapsed = time.time() - start
        print("Epoch Step: %d Train Loss: %f Test Loss: %f  Sec: %f" % 
              (epoch, total/200, test_total/20, elapsed))
        
    PATH = quantized_path  + pretrained_model + quant + '.pth.tar'
    torch.save({'state_dict': transformer.state_dict(), 
                'best_loss': total/200}, PATH)
    
    print('Model Trained and Save')
    
if hp.Test:
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    
    PATH = quantized_path  + pretrained_model + quant + '.pth.tar'
    checkpoint = torch.load(PATH)
    transformer.load_state_dict(checkpoint['state_dict'])  
    print('model load!')
    
    StoT = SeqtoText(vocb_dictionary, end_idx)
    score = []
    for epoch in range(10):
        final_word = []
        original_word = []
        
        for snr in hp.SNR:
            word = []
            target_word = []
            noise_std = SNR_to_noise(snr)
        
            for i, batch in enumerate(test_iterator):
                transformer.eval()
                src = batch.src
                target = batch.trg
        
                out = greedy_decode(transformer, src, noise_std, hp.MAX_LENGTH, src_pad_idx, 
                                SRC.vocab.stoi["<start>"], channel, CSI, imperfect, is_csinet, CSINet)
        
                sentences = out.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, sentences))
                word = word + result_string
        
                target_sent = target.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, target_sent))
                target_word = target_word + result_string
            
            final_word.append(word)
            original_word.append(target_word)
        
    
        bleu_score = []
        for sent1, sent2 in zip(original_word, final_word):
            # 1-gram
            bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))

        bleu_score = np.array(bleu_score)
        bleu_score = np.mean(bleu_score, axis=1)
        score.append(bleu_score)
        
    score1 = np.array(score)
    score2 = np.mean(score1, axis = 0)
    

        


