import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

MAX_LEN = 128

def load_data(data_path, s, batch_size=1):

    if data_path.split('/')[3][:4] != 'TEST':
        print('\n'+'Loading: '+str(data_path))
    df = pd.read_csv(data_path, delimiter='\t', header=None, names=['SUBORD', 'MATRIX', 'LABEL', 'SCONJ'])
    subord = df['SUBORD'].values
    matrix = df['MATRIX'].values
    labels = df['LABEL'].values
    labels = [int(i) for i in labels]
    sconj = df['SCONJ'].values
    sconj_t = [sconj_type(s.lower()) for s in sconj]

    if data_path.split('/')[3][:4] != 'TEST':
        print('\n'+'Loading BERT tokenizer...')
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
    labels = torch.tensor(labels)
    sconj_t = torch.tensor(sconj_t)
    dataset = TensorDataset(ids, masks, labels, sconj_t)
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
