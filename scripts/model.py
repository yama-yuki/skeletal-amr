import pandas as pd
import numpy as np
from scipy.special import softmax

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

MAX_LEN = 128

def load_data(data_path, s, batch_size=1):

    df = pd.read_csv(data_path, delimiter='\t', header=None, names=['SUBORD', 'MATRIX', 'SCONJ'])
    subord = df['SUBORD'].values
    matrix = df['MATRIX'].values
    sconj = df['SCONJ'].values

    sconj_t = [sconj_type(s.lower()) for s in sconj]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    ids, masks = [], []
    for sentence_s, sentence_m, feature_s in zip(subord, matrix, sconj):

        if s == True:
            tagged_sconj = '[unused0] ' + feature_s.lower() + ' [unused0] '
            sentence_s = tagged_sconj + sentence_s

        encoded_sentences = tokenizer.encode_plus(
            sentence_s,
            sentence_m,
            add_special_tokens = True,
            max_length = MAX_LEN,
            pad_to_max_length = True,
            truncation=True,
            return_attention_mask = True,
            return_tensors = 'pt',
            )
        ids.append(encoded_sentences['input_ids'])
        masks.append(encoded_sentences['attention_mask'])

    ids = torch.cat(ids, dim=0)
    masks = torch.cat(masks, dim=0)
    sconj_t = torch.tensor(sconj_t)
    dataset = TensorDataset(ids, masks, sconj_t)
    dataloader = DataLoader(dataset, batch_size = batch_size)

    return dataloader

## For restricting softmax
def sconj_type(sconj):
    if sconj in {'since','as'}:
        sconj_t = 1
    elif sconj == 'while' or sconj == 'whilst':
        sconj_t = 2
    elif sconj == 'when':
        sconj_t = 3
    else: sconj_t = 0
    return sconj_t

def restrict(s_max, sconj_type):
    new_s_max= s_max
    if int(sconj_type) == 1: ##since,as
        new_s_max[0][1] = 0 #cond=0
        new_s_max[0][2] = 0 #conc=0
    elif  int(sconj_type) == 2: ##while
        new_s_max[0][0] = 0 #cause=0
        new_s_max[0][1] = 0 #cond=0
    elif int(sconj_type) == 3: ##when
        new_s_max[0][0] = 0 #cause=0
        new_s_max[0][2] = 0 #conc=0
    else: return new_s_max
    #print(new_s_max)
    return new_s_max

def predict(dataloader, model, device, o=True):
    predictions = []
    sconj_type_list = []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_sconj = batch
        with torch.no_grad():
        # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        sconj_ids = b_sconj.to('cpu').numpy()

        predictions.append(logits)
        sconj_type_list.append(sconj_ids)
    print('Prediction DONE.')

    y_pred_multi = []
    for i in range(len(predictions)):
        s_max = softmax(predictions[i])
        if o == True:
            s_max = restrict(s_max, int(sconj_type_list[i][0]))
        #print(s_max)
        pred_labels_i = np.argmax(s_max, axis=1).flatten()#predictions[i]
        y_pred_multi.append(pred_labels_i[0])

    return y_pred_multi

