##legacy

import sys
from ast import literal_eval
from pprint import pprint

import spacy
from spacy.matcher import DependencyMatcher
import stanza
from spacy_stanza import StanzaLanguage

snlp = stanza.Pipeline(lang="en", use_gpu=True)
nlp = StanzaLanguage(snlp)

from .pattern_loader import pattern_datum_list, create_list

import os
this_dir, this_filename = os.path.split(__file__)
data =  os.path.join(this_dir, 'copula_patterns')

matched_copula = []
copula_count=0

def pipeline(input_snt):
    text = input_snt
    doc = nlp(text)
    #print(type(doc))
    #print(doc)
    #print(dir(doc))
    #print(type(doc[0]))
    #print(doc[0])
    print('-------------------')
    token_list=[]
    for token in doc:
        token_list.append(token.text)
        print(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.head.i, token.dep_, token.head.text)
        #[0]token_id|[1]token|[2]lemma|[3]token_pos|[4]token_tag|[5]head_id|[6]dep|[7]head_token
    print('-------------------')
    return text, doc, token_list

def tree_match(pattern):
    matcher = DependencyMatcher(nlp.vocab)
    matcher.add('pattern', None, pattern)
    matches = matcher(doc)
    return matches

def match_result(matches,pattern_idx):
    if matches[0][1] != []:
        #print('[(match_id, [token_idx])]: '+str(matches))
        for match_id, token_idx in matches:
            count=0
            for i in token_idx:
                i = sorted(i)
                matched_words = [token_list[j] for j in i]
                print('Matched Words: '+str(matched_words))
                count+=1
        #print('Matched Subtree: '+str(count))
        matched_copula.append(pattern_idx+1)
    #else:
        #print('No Subtree Matched')

def total_result(id_list, const_list, matched_copula):
    mp_id = [id_list[i-1] for i in matched_copula]
    print('Matched Pattern_ID: '+str(mp_id))
    mp = [const_list[i-1] for i in matched_copula]
    print('Matched Patterns: '+str(mp))
    print('-------------------')
    print('Total Patterns: '+str(copula_count))

def pattern_matcher(pattern, pattern_list):
    pattern_dict = literal_eval(pattern)
    pattern_idx = pattern_list.index(pattern)
    #print('Pattern_ID '+str(id_list[pattern_idx])+': '+str(const_list[pattern_idx]))
    #pprint(pattern_dict)
    matches = tree_match(pattern_dict)
    match_result(matches,pattern_idx)

if __name__ == '__main__':
    #input_snt = 'A man is walking a dog.'
    input_snt = input('Sentence: ')
    #print('-------------------')

    text, doc, token_list = pipeline(input_snt)
    datum_list = pattern_datum_list(data)
    id_list, const_list, pattern_list, amr_list = create_list(datum_list)

    for pattern in pattern_list:
        if pattern != '':
            copula_count+=1
            pattern_matcher(pattern, pattern_list)
    
    total_result(id_list, const_list, matched_copula)

