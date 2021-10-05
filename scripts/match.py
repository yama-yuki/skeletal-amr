import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_name", help="model name", type=str, default='BERT-AMR_10_5e-05_16')
parser.add_argument("-f","--file_path", help="input file path", type=str, default='demo.txt')
args = parser.parse_args()
model_name = args.model_name
file_path = args.file_path

from matcher.amr_matcher import matching
from matcher.pattern_loader import pattern_datum_list, create_list, load_dict

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
    print('Loaded Pattern Dictionary')
    print('Start Dependency Matching')

    with open(out_path, mode='w', encoding='utf-8') as o:
        
        skele_list, conll_list = [], []
        for sent in sents:
            results, _, conll, _ = matching(sent)

            if results:
                conll_list.append(conll)
                for i,result in enumerate(results):
                    matched_id = results[i][0]
                    skeleton = amr_list[matched_id+1]
                    #print(skeleton)
                    skele = ''.join(skeleton)

                    print(skele)

def main():
    load_model(model_name)
    sents = load_sents(file_path)
    run_matcher(sents)

if __name__ == '__main__':
    main()
