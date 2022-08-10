
"""
This is prune file
only prune the transeiver file
"""
import os
import time
import utils
import pickle
import torch
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
PATH = checkpoint_path  + './traceiver-Rician-imperfect_CSI-4bits' + '.pth.tar'

stored_path = './Pruned Model'
label = '60'
prune_path =  stored_path + './traceiver-Rician-imperfect_CSI-4bits-' + label + '.pth.tar'

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
    
def main():

    model = Transformer(hp.num_layers, len(SRC.vocab), len(TRG.vocab), 
                    len(SRC.vocab), len(TRG.vocab), hp.d_model, hp.num_heads, 
                    hp.dff, hp.dropout_rate).to(device)
    

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(PATH), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    
    criterion = nn.CrossEntropyLoss(reduction = 'none').to(device)

    test_acc0 = validate(test_iterator, model, criterion)
    #########################################################################################################################
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.data.numel()  #返回元素个数

    linear_weights = torch.zeros(total).cuda() 
    index = 0
    # copy weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # return the no. of elements
            size = m.weight.data.numel() 
            # copy the abs of elements to linear weights
            linear_weights[index: (index+size)] = m.weight.data.view(-1).abs().clone()
            index += size   
            
    # sort weights
    y, i = torch.sort(linear_weights)
    thre_index = int(total * hp.percent)
    # got the thre value
    thre = y[thre_index]

    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
            weight_copy = m.weight.data.abs().clone()
            # sparsity mask
            mask = weight_copy.gt(thre).float().cuda()  
            # the no. of elements pruned
            pruned = pruned + mask.numel() - torch.sum(mask)
            # prune the network 
            m.weight.data.mul_(mask)
            
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total params: {}, Pruned params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    ########################################################################################################################
   
    test_acc1 = validate(test_iterator, model, criterion)

    torch.save({'epoch': 0, 
                'state_dict': model.state_dict(), 
                'acc': test_acc1,
                'best_acc': 0.}, prune_path)

    with open( stored_path + './prune-' + label + '.txt', 'w') as f:
        f.write('Before pruning: Test Acc:  %.2f\n' % (test_acc0))
        f.write('Total params: {}, Pruned params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned/total))
        f.write('After Pruning: Test Acc:  %.2f\n' % (test_acc1))

        if zero_flag:
            f.write("There exists a layer with 0 parameters left.")
    return

def validate(test_loader, model, criterion):
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
                                         pad_idx, start_symbol, channel, 
                                         CSI, imperfect, is_csinet, CSINet)
            
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