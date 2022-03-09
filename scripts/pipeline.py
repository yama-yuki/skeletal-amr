'''
python pipeline.py --model_path ../classifier/torch_models/WIKI-AMR/WIKI_3_3e-05_64_AMR_10_3e-05_32/1.pth --file_path demo/in.sents
python pipeline.py --model_path demo/model.pth --file_path demo/in.sents
'''
import os
from posixpath import join
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import pandas as pd

import json
from logging.config import dictConfig
from logging import getLogger
with open('log/logging.json') as f:
    dictConfig(json.load(f))
info_logger = getLogger('info.test')

import torch
from transformers import BertForSequenceClassification

from graph import disamb_skele
from model import load_data, predict
from matcher.amr_matcher import matching
from matcher.pattern_loader import pattern_datum_list, create_list, load_dict

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(model_path))
    return model

def load_sents(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        sents = f.readlines()
        sents = [sent.rstrip() for sent in sents]
    return sents

class DepMatch:
    def __init__(self):
        self.sent = ''
        self.skeles = []
        self.clauses = []
        self.ambs = []
        self.results = []
        self.results_cls = []

def run_matcher(sents):
    info_logger.info('- Loading Pattern Dictionary')
    pd_path = '../matcher/pattern_dict'
    pd_list = pattern_datum_list(pd_path)
    id_list, _, _, amr_list = create_list(pd_list)
    #pattern_dict = load_dict(id_list, amr_list)

    info_logger.info('- Start Dependency Match')
    conll_list = []
    ##result = [doc, [skele1, skele2], [(SUB, MAT, SCONJ), (SUB, MAT, SCONJ)], [F, T]]

    d_list = []
    for i,sent in enumerate(sents):
        results, doc, conll, _ = matching(sent)
        tok_list = [t.text for t in doc]
        d = DepMatch()
        d.sent = doc
        
        done = []

        if results:
            info_logger.debug(results)
            conll_list.append(conll)
            for j,result in enumerate(results):
                matched_id = results[j][0]
                skeleton = amr_list[matched_id+1]
                skele_id = id_list[matched_id]
                info_logger.debug(skele_id)
                sp = skele_id.split(".")
                tree_match_id = result[2][0]

                ## V1 sconj1 V2
                if int(sp[0])==1:
                    sconj_i, matv_i, subv_i = tree_match_id[-1], tree_match_id[0], tree_match_id[2]
                    v_id_pair = (matv_i, subv_i)
                    if v_id_pair not in done:
                        done.append(v_id_pair)
                        
                        ##result = (0, [['went', 'I', 'ate', 'I', 'after']], [[1, 0, 7, 6, 5]])
                        mat_stree = [t for t in doc[matv_i].subtree]
                        sub_stree = [t for t in doc[subv_i].subtree]

                        ##(subord, matrix, sconj)
                        subord = ' '.join([t.text for t in sub_stree][1:])
                        if matv_i < subv_i:
                            matrix = ' '.join([t.text for t in mat_stree][:sconj_i])
                            clause_pair = (subord, matrix, tok_list[sconj_i])
                        else:
                            last_sub_tok = [t.i for t in sub_stree][-1]
                            matrix_tok = [t.text for t in mat_stree][last_sub_tok+1:]
                            if matrix_tok[0] == ',':
                                matrix_tok = matrix_tok[1:]
                            if matrix_tok[-1] == '.':
                                matrix_tok = matrix_tok[:-1]
                            matrix = ' '.join(matrix_tok)
                            clause_pair = (subord, matrix, tok_list[sconj_i])

                        ## if ambiguous
                        if sp[-1] == '*': 
                            d.ambs.append(True)
                        ## if not ambiguous
                        else:
                            d.ambs.append(False)
                            #d.clauses.append(('-','-','-'))
                        d.clauses.append(clause_pair)

                elif int(sp[0])==0:
                    ## V2-ing, V1
                    if int(sp[1])==1:
                        ##result = (56, [['came', 'He', ',', 'eating']], [[1, 0, 7, 11]])

                        sconj_i, matv_i, subv_i = tree_match_id[2], tree_match_id[0], tree_match_id[3]
                        v_id_pair = (matv_i, subv_i)
                        if v_id_pair not in done:
                            done.append(v_id_pair)
                        
                        else:
                            info_logger.debug('already matched')
                            d.ambs.append('done')
                            d.clauses.append(('-','-','-'))

                ## V1 sconj2 V2
                elif int(sp[0])==2:
                    pass
                ## V1 sconj3 V2
                elif int(sp[0])==3:
                    pass
                else:
                    pass

                skele = ''.join(skeleton)
                d.skeles.append(skele)
        d_list.append(d)

    return d_list

def disambiguate(model, d_list):

    ## make input
    amb_cp_list = []
    for i,d in enumerate(d_list):
        for j,amb in enumerate(d.ambs):
            if amb == True:
                info_logger.debug(d.clauses)
                amb_cp_list.append(d.clauses[j])

    disamb_path = 'tmp.csv'
    with open(disamb_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for amb_cp in amb_cp_list:
            writer.writerow(amb_cp)

    ## role classification
    dataloader = load_data(disamb_path, s=False, batch_size=1)
    predictions = predict(dataloader, model, device, o=True)
    
    ## provide resulting skeletons
    cnt=0
    for i,d in enumerate(d_list):
        for j,amb in enumerate(d.ambs):
            if amb == True:
                primitive = d.skeles[j]
                result = disamb_skele(primitive, predictions[cnt])
                d.results.append(result)
                
                cnt+=1
            elif amb == False:
                d.results.append(d.skeles[j])
            else:
                continue

    return d_list

def write_results(d_list, out_path):
    with open(out_path, mode='w', encoding='utf-8') as o:
        for i,d in enumerate(d_list):
            o.write('#SENT '+str(d.sent)+'\n')
            for j,r in enumerate(d.results):
                o.write('#SKELE'+str(j)+'\n')
                o.write(r+'\n')
            o.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="model name", type=str)
    parser.add_argument("-f", "--file_path", help="input file path", type=str, default='in.sents')
    args = parser.parse_args()
    model_path = args.model_path
    file_path = args.file_path

    sents = load_sents(file_path)

    info_logger.info('PHASE1: DEPENDENCY MATCH')
    d_list = run_matcher(sents)

    info_logger.info('PHASE2: SEMANTIC DISAMBIGUATION')
    info_logger.info('- Loading '+str(model_path))
    model = load_model(model_path)
    d_list = disambiguate(model, d_list)
    for d in d_list:
        info_logger.debug(d.results)

    info_logger.info('- Writing Out Results')
    out_path='demo/out.skele'
    write_results(d_list, out_path)

if __name__ == '__main__':
    main()

