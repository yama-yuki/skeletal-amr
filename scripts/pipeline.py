'''
python pipeline.py --model_path ../classifier/torch_models/WIKI-AMR/WIKI_3_3e-05_64_AMR_10_3e-05_32/1.pth --file_path demo/in.sents
python pipeline.py --model_path demo/model.pth --file_path demo/in.sents
'''
import argparse
import csv
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ast import literal_eval
from posixpath import join

from logging.config import dictConfig
from logging import getLogger
with open('log/logging.json') as f:
    dictConfig(json.load(f))
info_logger = getLogger('info.test')

import torch
from transformers import BertForSequenceClassification

from graph import disamb_skele
from model_utils import load_data, predict
from utils import find_matched_pos, make_input, split_clause
from matcher.amr_matcher import matching
from matcher.pattern_loader import pattern_entry_list, create_list, wo_nsubj

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
    '''
    constructor for matched results
    '''
    def __init__(self, doc):
        self.sent = doc
        self.skeles = []
        self.clauses = []
        self.ambs = []
        self.results = []
        self.results_cls = []

def run_matcher(sents):
    '''
    dependency matching with ../matcher/amr_matcher.py
    '''

    ##init matcher
    info_logger.info('- Loading Pattern Dictionary')

    pd_path = '../matcher/pattern_dict'
    pd_list = pattern_entry_list(pd_path)
    id_list, const_list, pattern_list, amr_list, pos_list = create_list(pd_list)
    pattern_dict = [wo_nsubj(literal_eval(pattern)) for pattern in pattern_list]

    positions = find_matched_pos(pos_list)

    info_logger.debug(positions)

    def id2tok(doc):
        result = {token.i: token for token in doc}
        return result
    
    ## run dependency matcher
    info_logger.info('- Start Dependency Match')

    d_list, conll_list = [], []
    for sent in sents:
        results, doc, conll, _ = matching(sent)
        depm = DepMatch(doc)

        info_logger.debug(len(results))
        #info_logger.info(results)

        if results: #result = [doc, [skele1, skele2], [(SUB, MAT, SCONJ), (SUB, MAT, SCONJ)], [F, T]]
            conll_list.append(conll)

            clauses = split_clause(results, positions, doc)
            info_logger.debug(clauses)

            for j,result in enumerate(results):
                matched_id = results[j][0]
                skeleton, skele_id = amr_list[matched_id], id_list[matched_id]
                sp = skele_id.split(".")
                tree_match_id = result[2][0]
                pos_id = positions[matched_id] #[0, 2, [3, 1], []]

                ## logs for debugging
                info_logger.debug(const_list[matched_id])
                info_logger.debug(skeleton)
                info_logger.debug(tree_match_id)
                info_logger.debug(pos_id)

                ## if ambiguous sconj
                if sp[-1] == '*': 
                    info_logger.debug('ambiguous')
                    depm.ambs.append(True)
                    info_logger.debug([tok for tok in depm.sent])
                    tok_dict = id2tok(depm.sent)

                    matv_i, subv_i = tree_match_id[pos_id[0]], tree_match_id[pos_id[1]]
                    matv_tok, subv_tok = tok_dict[matv_i], tok_dict[subv_i]
                    sc_i = [tree_match_id[i] for i in pos_id[2]]
                    sc = [tok_dict[tree_match_id[i]] for i in pos_id[2]]

                    info_logger.debug([matv_tok,subv_tok,sc]) #[believed, seemed, [As]]
                    info_logger.debug([matv_i,subv_i,sc_i]) #[8, 3, [0]]

                    '''
                    legacy
                    if matv_i < subv_i:
                        info_logger.debug('mat < sub')
                        ##result = (0, [['went', 'I', 'ate', 'I', 'after']], [[1, 0, 7, 6, 5]])
                    else:
                        info_logger.debug('sub < mat')
                    '''

                    if len(results) == 1:
                        clause_pair = make_input(doc, matv_i, subv_i, sc_i, sc)
                        depm.clauses.append(clause_pair)
                        info_logger.debug(clause_pair)
                    
                    else:
                        clause_pair = make_input(doc, matv_i, subv_i, sc_i, sc, clauses)
                        depm.clauses.append(clause_pair)
                        info_logger.debug(clause_pair)

                ## if not ambiguous sconj
                else:
                    info_logger.debug('not ambiguous')
                    depm.ambs.append(False)
                    depm.clauses.append(('-','-','-'))
                
                skele = ''.join(skeleton)
                depm.skeles.append(skele)

        d_list.append(depm)

    return d_list

def disambiguate(model, d_list):
    '''
    semantic disambiguation with a fine-tuned model
    '''

    ## make input for classifier
    amb_cp_list = []
    for depm in d_list:
        for i,amb in enumerate(depm.ambs):
            if amb == True:
                info_logger.debug(depm.clauses)
                amb_cp_list.append(depm.clauses[i])

    disamb_path = 'tmp.csv'
    with open(disamb_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for amb_cp in amb_cp_list:
            writer.writerow(amb_cp)

    ## classify roles
    dataloader = load_data(disamb_path, s=False, batch_size=1)
    predictions = predict(dataloader, model, device, o=True)
    
    ## provide resulting skeletons
    cnt=0

    for depm in d_list:
        info_logger.debug(depm.sent)
        info_logger.debug(depm.ambs)
        info_logger.debug(depm.skeles)

        for j,amb in enumerate(depm.ambs):
            if amb == True:
                primitive = depm.skeles[j]
                result = disamb_skele(primitive, predictions[cnt])
                depm.results.append(result)
                
                cnt+=1
            elif amb == False:
                depm.results.append(depm.skeles[j])
            else:
                continue

    os.remove(disamb_path)
    return d_list

def write_results(d_list, out_path):
    with open(out_path, mode='w', encoding='utf-8') as o:
        for depm in d_list:
            o.write('#SENT '+str(depm.sent)+'\n')
            for j,result in enumerate(depm.results):
                o.write('#SKELE'+str(j)+'\n')
                o.write(result.rstrip('\n')+'\n')
            o.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="model name", type=str)
    parser.add_argument("-f", "--file_path", help="input file path", type=str, default='in.sents')
    args = parser.parse_args()
    model_path = args.model_path
    file_path = args.file_path

    #stanza.download("en")
    sents = load_sents(file_path)

    info_logger.info('PHASE1: DEPENDENCY MATCH')
    d_list = run_matcher(sents)

    info_logger.info('PHASE2: SEMANTIC DISAMBIGUATION')
    if d_list:
        info_logger.debug(d_list)
        info_logger.info('- Loading '+str(model_path))
        model = load_model(model_path)
        d_list = disambiguate(model, d_list)
     
        for depm in d_list:
            info_logger.debug(depm.results)

    out_path='demo/out.skele'
    info_logger.info('- Writing Out Results to '+out_path)
    
    write_results(d_list, out_path)

if __name__ == '__main__':
    main()

