import pandas as pd
import numpy as np
import random
import time
import datetime
import sys
import os
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

from torch_dataloader import load_data

from logging import getLogger, StreamHandler, FileHandler, Formatter
logger = getLogger(__name__)
formatter = Formatter('%(asctime)s - %(name) - %(Levelname)s - %(message)s')
sh = StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
'''
fh = FileHandler('test.log', 'a+')
fh.setFormatter(formatter)
logger.addHandler(fh)
'''

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#MAX_LEN = 128 
SAVE_DIR = '../torch_models'

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

## Fix random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

## Assign paths for train data and models
def make_path(model_name, data, s, cv_num, mix, mixid, model_to_tune):
    '''
    INPUT
    model_name: give name to the model
    data: which data to use for training
    s: to encode sconj (e.g. 'because') or not
    mix, mixid: MIX split IDs for MIX models
    cv_num: which split to use for train/eval
    model_to_tune: model to finetune ('bert-base-uncased' OR the model specified)
    ---
    OUTPUT
    data_path: path for train data
    save_path: path for saving models
    seed: SEED value to use for training
    '''

    if s == True:
        sconj = 's'
    else: sconj = ''

    if data == 'amr':
        #data_path = '../rsc/amr/TRAIN'+sconj+str(cv_num)+'.csv'
        data_path = '../rsc/amr/TRAIN'+str(cv_num)+'.csv'
        seed = 0

        if model_to_tune == 'bert-base-uncased':
            dir_name = 'BERT-AMR'+sconj
            save_dir = os.path.join(SAVE_DIR, dir_name, model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, str(cv_num)+'.pth')
        else: 
            dir_name = 'WIKI-AMR'+sconj
            save_dir = os.path.join(SAVE_DIR, dir_name, 'WIKI_'+str(model_to_tune.split('/')[1])+'_AMR'+sconj+'_'+model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, str(cv_num)+'.pth')
            model_to_tune = os.path.join(SAVE_DIR, model_to_tune+'.pth')
    
    elif data == 'mix':
        mix_name = str(mix)+'_'+str(mixid)
        #data_path = '../rsc/mix'+sconj+'/'+mix_name+'_'+str(cv_num)+'.csv'
        data_path = '../rsc/mix'+'/'+mix_name+'_'+str(cv_num)+'.csv'
        seed = 0

        if model_to_tune == 'bert-base-uncased':
            dir_name = 'BERT-MIX'+sconj
            save_dir = os.path.join(SAVE_DIR, dir_name, mix_name+'_'+model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir,str(cv_num)+'.pth')
        else: 
            dir_name = 'WIKI-MIX'+sconj
            save_dir = os.path.join(SAVE_DIR, dir_name, 'WIKI_'+str(model_to_tune.split('/')[1])+'_MIX'+sconj+'_'+mix_name+'_'+model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir,str(cv_num)+'.pth')
            model_to_tune = os.path.join(SAVE_DIR, model_to_tune+'.pth')
    
    elif data == 'wiki':
        data_path = '../rsc/wiki/merged'+sconj+'/train_dummy.csv'
        seed_num = cv_num
        seed_list = [0, 1, 42, 1337, 31337]
        seed = seed_list[int(seed_num)]

        dir_name = 'BERT-WIKI'+sconj
        save_dir = os.path.join(SAVE_DIR, dir_name, model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, str(seed_num)+'.pth')
    
    else:
        print('NO DATA Specified') 
        sys.exit()

    return data_path, save_path, seed, model_to_tune

## Training loop
def train_loop(s, epochs, lr, batch_size, data_path, save_path, model_to_tune):
    # based on https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = 4,
        output_attentions = False,
        output_hidden_states = False,
    )
    model.cuda()

    if model_to_tune != 'bert-base-uncased':
        model.load_state_dict(torch.load(model_to_tune))
        print('LOAD: '+model_to_tune)
    print('TRAIN_DATA: '+str(data_path))
    print('MODEL_NAME: '+str(save_path))

    train_dataloader = load_data(data_path, s, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr)
    total_steps = len(train_dataloader) * epochs # training steps = batches * epochs.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    loss_values = []
    for epoch_i in tqdm(range(0, epochs)):
        print("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0    
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            if step % 40 == 0 and not step == 0: # Progress update every 40 batches.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # batch: [0]: input ids, [1]: attention masks, [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)        
            model.zero_grad()                
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)# Forward (evaluate the model on this training batch).
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward() # Backward (calculate the gradients). 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # To prevent the "exploding gradients" problem.
            optimizer.step()        
            scheduler.step() # Update the learning rate.
        avg_train_loss = total_loss / len(train_dataloader)            
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        sys.stdout.flush()
    torch.save(model.state_dict(), save_path)
    print("Training complete!")
    torch.cuda.empty_cache()
    sys.stdout.flush()

def torch_train(model_name, data, s, epochs, lr, batch_size, cv_num, mix=1, mixid=1, model_to_tune='bert-base-uncased'):
    data_path, save_path, seed, model_to_tune = make_path(model_name, data, s, cv_num, mix, mixid, model_to_tune)
    set_seed(seed)
    train_loop(s, epochs, lr, batch_size, data_path, save_path, model_to_tune)

