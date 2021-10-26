'''
python pipeline.py --model_name WIKI-AMR/WIKI_3_3e-05_64_AMR_10_3e-05_32 --file_path 
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
    print('Loaded Pattern Dictionary'+'\n')
    print('Start Dependency Matching'+'\n')
        
    skele_list, conll_list = [], []
    for sent in sents:
        results, _, conll, _ = matching(sent)

        tmp = []
        if results:
            conll_list.append(conll)
            for i,result in enumerate(results):
                matched_id = results[i][0]
                skeleton = amr_list[matched_id+1]
                skele = ''.join(skeleton)
                print(skele + '\n')
                tmp.append(skele)
        skele_list.append(tmp)

    return skele_list

def disambiguate(skele_list):
    final_results = []

    
    for i,skeles in enumerate(skele_list):
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
    load_model(model_name)
    sents = load_sents(file_path)
    print('###DEPENDENCY MATCHING PHASE###')
    skele_list = run_matcher(sents)
    print('###SEMANTIC DISAMBIGUATION PHASE###')
    results = disambiguate(skele_list)
    write_results(results)
    return(results)

if __name__ == '__main__':
    main()
