import itertools
import os
import sys
from ast import literal_eval
from pprint import pprint

from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc
from spacy_conll import init_parser, ConllFormatter
from spacy_stanza import StanzaLanguage

import logging
from logging import getLogger, StreamHandler, Formatter
info_logger = getLogger('info.test').getChild('sub')

from .pattern_loader import pattern_entry_list, create_list, wo_nsubj
from .conll_load import conll_list_to_doc, convert


this_dir, this_filename = os.path.split(__file__)
TEMP = os.path.join(this_dir, 'temp.conllu')

nlp = init_parser("stanza",
                  "en",
                  parser_opts={"use_gpu": True, "verbose": False},
                  include_headers=True)

def pipeline(input_snt):

    text = input_snt
    doc = nlp(text)
    token_list=[]
    for token in doc:
        token_list.append(token.text)
        #print(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.head.i, token.dep_, token.head.text)
        #[0]token_id|[1]token|[2]lemma|[3]token_pos|[4]token_tag|[5]head_id|[6]dep|[7]head_token
    return text, doc, token_list

def tree_match(doc, pattern):

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add('pattern_dict', None, pattern)
    matches = matcher(doc)

    return matches

def match_result(token_list, matches, pattern_idx):

    matched_id, matched_token, matched_token_id = [], [], []
    if matches[0][1] != []:
        for match_idx, token_idx in matches:
            count=0
            for i in token_idx:
                matched_words = [token_list[j] for j in i]
                matched_words_id = [j for j in i]
                if matched_words:
                    matched_token.append(matched_words)
                    matched_token_id.append(matched_words_id)
                count+=1
        matched_id.append(pattern_idx)#+1)
        result = (pattern_idx, matched_token, matched_token_id)
        return result

    return

def pattern_matcher(doc, token_list, pattern, pattern_list, mode='copula', subj=False):
    '''
    (2)
    Dependency Matcher
    '''
    pattern_dict = literal_eval(pattern)
    if subj == False:
        pattern_dict = wo_nsubj(pattern_dict)
    pattern_idx = pattern_list.index(pattern)
    if mode == 'copula':
        matches = tree_match(doc, pattern_dict)
        matched_token_idx = copula_result(token_list,matches,pattern_idx)
        return matches, matched_token_idx
    
    elif mode == 'complex':
        matches = tree_match(doc, pattern_dict)
        result = match_result(token_list,matches,pattern_idx)
        return result

    else:
        sys.exit('Need to specify mode for pattern_matcher()')

def resolve_subset_overlap(results):
    '''
    (3)
    Resolve Overlapping Patterns
    Input:
    [(2, [['has', 'theory', 'have', 'people', 'though']], [[17, 15, 7, 6, 3]]),
    (36, [['has', 'theory', 'have', 'people', 'even', 'though']], [[17, 15, 7, 6, 2, 3]])]
    Output:
    [(36, [['has', 'theory', 'have', 'people', 'even', 'though']], [[17, 15, 7, 6, 2, 3]])]
    '''
    
    comb = sorted(list(itertools.combinations(results, 2)))

    overlap = []
    for pair in comb:
        a, b = pair
        a_id, b_id = a[0], b[0] #2
        am_id, bm_id = set(a[2][0]), set(b[2][0]) #{17, 15, 7, 6, 3}
        if (a_id not in overlap) & (b_id not in overlap):
            '''
            if am_id <= bm_id:
                overlap.append(a)
            '''
            if len((am_id&bm_id)) >= 2:
                overlap.append(a)

    for s in overlap:
        if s in results:
            results.remove(s)

    return results

def doc_to_conll(doc):
    ##doc to conll
    conll = [list(map(str, [token.i+1, token.text, token.lemma_, token.pos_, token.tag_, '_',
                token.head.i+1, token.dep_, '_', '_'])) for token in doc]
    
    ##token.head.i -> 0 if token.head is root
    for col in conll:
        if col[0] == col[6]:
            col[6] = '0'

    return conll

def copula_result(token_list, matches, pattern_idx):

    matched = []
    matched_id = []
    if matches[0][1] != []:
        for match_id, token_idx in matches:
            for i,t_i in enumerate(token_idx):
                matched.append(t_i)
                matched_words = [token_list[j] for j in t_i]
                #print('Matched Words: '+str(matched_words))
        matched_id.append(pattern_idx+1)
    #copula_matched.append(matched)
    return matched_id

def cop_conv(doc):
    '''
    (1)
    Find Copula Dependent Structure
    Convert to Copula Head Structure
    Convert stanza parse tree to CONLL
    '''

    token_list = [token.text for token in doc]
    
    pattern_path = os.path.join(this_dir,'copula_patterns')
    entry_list = pattern_entry_list(pattern_path)
    _, _, copula_list, _, _ = create_list(entry_list)
    matches, _ = pattern_matcher(doc, token_list, copula_list[0], copula_list, 'copula')
    #info_logger.info(matches)
    
    conll = doc_to_conll(doc) # [[1, a, a, ...],[]]

    if matches[0][1]:
        conll = convert(conll, matches[0][1])
        doc = conll_list_to_doc(nlp.vocab, conll)

    return doc, token_list, conll

def matching(input_snt):
    '''
    Input:
    I am hungry because I didn't have lunch.
    Output:
    [(12, [['am', 'have', 'because']], [[1, 7, 3]])]
    '''
    _, doc, token_list = pipeline(input_snt)

    ##(1) Convert to Copula Headed CONLL

    matched_id_list = []
    p_count=0

    doc, token_list, conll = cop_conv(doc)

    ##(2) Dependency Match

    results = []
    p_count=0
    pattern_path = os.path.join(this_dir, 'pattern_dict')
    entry_list = pattern_entry_list(pattern_path)
    id_list, const_list, pattern_list, amr_list, _ = create_list(entry_list)
    ## id_list: ['1.1', '1.2', '1.3', ...]
    for pattern in pattern_list:
        if pattern != '':
            p_count+=1
            result = pattern_matcher(doc, token_list, pattern, pattern_list, 'complex')
            ## result: (pattern_idx, matched_token, matched_token_id)
            if result:
                results.append(result)

    ##############################################################################################################
    ## Single Match
    ##[(12, [['am', 'have', 'because']], [[1, 7, 3]])]
    ## Subset Overlap
    ##[(35, [['has', 'have', 'though']], [[17, 7, 3]]), (36, [['has', 'have', 'even', 'though']], [[17, 7, 2, 3]])]
    ## Partial Overlap
    ##[(57, [['been', 'we', ',', 'been'], ['been', 'we', ',', 'measured']], [[50, 47, 46, 8], [50, 47, 46, 38]])]
    ##############################################################################################################

    ##(3) Resolve Subset Overlap
    
    if len(results) > 1:
        results = resolve_subset_overlap(results)

    return results, doc, conll, id_list

