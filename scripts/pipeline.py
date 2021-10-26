'''
python pipeline.py --model_name WIKI-AMR/WIKI_3_3e-05_64_AMR_10_3e-05_32 --file_path demo.txt
'''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_name", help="model name", type=str, default='BERT-AMR/10_5e-05_16')
parser.add_argument("-f","--file_path", help="input file path", type=str, default='demo.txt')
args = parser.parse_args()
model_name = args.model_name
file_path = args.file_path

import torch
from transformers import BertForSequenceClassification
from torch_dataloader import load_data

from matcher.amr_matcher import matching
from matcher.pattern_loader import pattern_datum_list, create_list, load_dict

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels = 4)
model.cuda()

def load_model(model_name):

    pass

def load_sents(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        sents = f.readlines()
        sents = [sent.rstrip() for sent in sents]
    return sents

def run_matcher(sents):
    data = '../matcher/pattern_dict'
    datum_list = pattern_datum_list(data)
    id_list, _, _, amr_list = create_list(datum_list)
    #pattern_dict = load_dict(id_list, amr_list)
    print('Loaded Pattern Dictionary'+'\n')
    print('Start Dependency Matching'+'\n')
        
    skele_list, conll_list = [], []

    for sent in sents:
        results, _, conll, _ = matching(sent)

        tmp = []
        if results:
            tmp.append(sent)
            conll_list.append(conll)
            for i,result in enumerate(results):
                matched_id = results[i][0]
                skeleton = amr_list[matched_id+1]
                skele = ''.join(skeleton)
                print(skele + '\n')
                tmp.append(skele)    
        skele_list.append(tmp)
    
    ## skele_list = [[sent, skele1, skele2], [sent, skele1]]]
    return skele_list

def disambiguate(skele_list):
    model.eval()

    final_results = []
    for i,skeles in enumerate(skele_list):
        sent = skeles[0]

    predictions, sconj_type_list = [], []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_sconj = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        sconj_ids = b_sconj.to('cpu').numpy()

        predictions.append(logits)
        sconj_type_list.append(sconj_ids)

        test_path = input_sent+'.csv'
        model_path = os.path.join(model_name,str(1)+'.pth')
        test_dataloader = load_data(test_path, s=False, batch_size=1)
        model.load_state_dict(torch.load(model_path))
        d = torch_eval(test_dataloader, o)





        tmp = []
        for skele in skeles:
            tmp.append(skele)
        final_results.append(tmp)

    return final_results

def write_results(results):
    out_path='out.skele'
    with open(out_path, mode='w', encoding='utf-8') as o:
        for i,result in enumerate(results):
            o.write('#SENT '+str(i)+'\n')
            for r in result:
                o.write('#SKELE'+'\n')
                o.write(r+'\n')
            o.write('\n')

def main():
    sents = load_sents(file_path)
    print('###DEPENDENCY MATCHING PHASE###')
    skele_list = run_matcher(sents)
    print('###SEMANTIC DISAMBIGUATION PHASE###')
    load_model(model_name)
    results = disambiguate(skele_list)
    write_results(results)
    return(results)

if __name__ == '__main__':
    main()
