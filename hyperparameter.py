
class Hyperparameter:
    Test = True
    Train = False
    #Data
    train_path = './dataset/train_data.txt'
    train_save_path = './dataset/train_data.pkl'
    test_path = './dataset/test_data.txt'
    test_save_path = './dataset/test_data.pkl'
    
    processed = True
    test_processed = True
    
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    
    SHUFFLE_SIZE = 2000
    BATCH_SIZE = 64
    EPOCHS = 120
    
    MAX_LENGTH = 30
    MIN_LENGTH = 4
    
    
    d_mutual = 256 # the size of dense of channel encoder
    n = 64  # the size of output of channel encoder [batch_size, length, 2*n]
    d_modu = 256 #the size of modulation
    SNR = [ 6, 12] #the noise of channel
    
    # quantization bits level
    abits = 16
    wbits = 16
    stastic_quantization = True
    QAT = False
    
    # prune
    prune = True
    percent = 0.6
    finetune_epoch = 30


