from spacy.matcher import DependencyMatcher
from ast import literal_eval
from pprint import pprint

import stanza
from spacy_stanza import StanzaLanguage
from spacy_conll import init_parser, ConllFormatter

from .pattern_loader import pattern_datum_list, create_list
from .conll_load import convert, conll_to_doc, process, conll_data

nlp = init_parser("stanza",
                  "en",
                  parser_opts={"use_gpu": True, "verbose": False},
                  include_headers=True)

import os
this_dir, this_filename = os.path.split(__file__)
TEMP = os.path.join(this_dir, 'temp.conllu')

copula_matched = []

def init():
    global copula_matched
    copula_matched = []
    return

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
    matcher.add('pattern', None, pattern)
    matches = matcher(doc)
    return matches

def match_result(token_list, matches, pattern_idx):
    matched_id = []
    matched_token = []
    matched_token_id = []
    if matches[0][1] != []:
        for match_idx, token_idx in matches:
            count=0
            for i in token_idx:
                matched_words = [token_list[j] for j in i]
                matched_words_id = [j for j in i]
                if matched_words:
                    matched_token.append(matched_words)
                    matched_token_id.append(matched_words_id)
                    #print('Matched Words: '+str(matched_words))
                count+=1
        matched_id.append(pattern_idx)#+1)
        result = (pattern_idx, matched_token, matched_token_id)
        return result
    return

def copula_result(token_list, matches,pattern_idx):
    matched = []
    matched_id = []
    if matches[0][1] != []:
        for match_id, token_idx in matches:
            count=0
            for i in token_idx:
                matched.append(i)
                #i = sorted(i)
                matched_words = [token_list[j] for j in i]
                #print('Matched Words: '+str(matched_words))
                count+=1
        matched_id.append(pattern_idx+1)
    copula_matched.append(matched)
    return matched_id

def make_copula_head(doc, token_list, copula_matched):
    '''
    (1)
    Find Copula Dependent Structure
    Convert to Copula Head Structure

    Convert stanza parse tree to CONLL
    '''
    change_idx,child_idx = [],[]
    #V1 = None
    for p_idx in range(len(copula_matched)):
        if copula_matched[p_idx] != []:
            for matched in copula_matched[p_idx]:
                C1,B1 = matched[0],matched[1]
                change_idx.append([C1+1,B1+1])
                c1 = [i.i+1 for i in doc[C1].children]
                b1 = [i.i+1 for i in doc[B1].children]
                child_idx.append([c1,b1])
                '''
                if p_idx == 0:
                    C1,B1,V2 = matched[0],matched[1],matched[3]
                    change_idx.append([p_idx,C1+1,B1+1,V2+1])
                    #change_idx.append([])
                    c1 = [i.i+1 for i in doc[C1].children]
                    b1 = [i.i+1 for i in doc[B1].children]
                    child_idx.append([p_idx,c1,b1])
                    #child_idx.append([[],[]])
                elif p_idx == 1:
                    V1,C2,B2 = matched[0],matched[2],matched[3]
                    #change_idx.append([])
                    change_idx.append([p_idx,C2+1,B2+1,V1+1])
                    c2 = [i.i+1 for i in doc[C2].children]
                    b2 = [i.i+1 for i in doc[B2].children]
                    #child_idx.append([[],[]])
                    child_idx.append([p_idx,c2,b2])
                elif p_idx == 2:
                    C1,B1,C2,B2 = matched[0],matched[1],matched[3],matched[4]
                    change_idx.append([p_idx,C1+1,B1+1,C2+1,B2+1])
                    #change_idx.append([C1,B1])
                    #change_idx.append([C2,B2])
                    c1 = [i.i+1 for i in doc[C1].children]
                    b1 = [i.i+1 for i in doc[B1].children]
                    c2 = [i.i+1 for i in doc[C2].children]
                    b2 = [i.i+1 for i in doc[B2].children]
                    child_idx.append([p_idx,c1,b1,c2,b2])
                    #child_idx.append([c2,b2])
                '''

    with open(TEMP,mode='w',encoding='utf-8') as f:
        conll = doc._.conll_str
        f.write(conll)
    
    if change_idx != []:
        conll = convert(TEMP, change_idx, child_idx)
        #print(conll)
        if conll:
            doc = conll_to_doc(nlp.vocab, conll)
            _, doc, token_list = process(doc)
            #for token in doc:
                #print(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.head.i, token.dep_, token.head.text)
    else:
        conll_snt=[]
        with open(TEMP, mode='r', encoding='utf-8') as d:
            lines = d.readlines()
            for line in lines:
                if line[0] != '#' and line[0] != '\n':
                    conll_snt.append(line.rstrip('\n').split('\t'))
        conll = conll_snt
    return doc, token_list, conll

def total_result(id_list, const_list, matched_id, p_count):
    mp_id = [id_list[i-1] for i in matched_id]
    #print('Matched Pattern_ID: '+str(mp_id))
    mp = [const_list[i-1] for i in matched_id]
    if mp != []:
        #print('Matched Patterns: '+str(mp))
        return
    return mp_id, mp

def pattern_matcher(doc, token_list, pattern, pattern_list, mode='copula'):
    '''
    (2)
    Dependency Matcher
    '''
    pattern_dict = literal_eval(pattern)
    pattern_idx = pattern_list.index(pattern)
    if mode == 'copula':
        matches = tree_match(doc, pattern_dict)
        matched_token_idx = copula_result(token_list,matches,pattern_idx)
        return
    else:
        matches = tree_match(doc, pattern_dict)
        result = match_result(token_list,matches,pattern_idx)
        return result

def resolve_subset_overlap(results):
    '''
    (3)
    Resolve Subset Overlapping Patterns
    Input:
    [(2, [['has', 'theory', 'have', 'people', 'though']], [[17, 15, 7, 6, 3]]),
    (36, [['has', 'theory', 'have', 'people', 'even', 'though']], [[17, 15, 7, 6, 2, 3]])]
    Output:
    [(36, [['has', 'theory', 'have', 'people', 'even', 'though']], [[17, 15, 7, 6, 2, 3]])]
    '''
    final = []
    overlap_id = []
    #print(results)
    if len(results) > 1:
        id_set = [set(result[2][0][:4]) for result in results]
        for i,result in enumerate(results):
            left = [i for i in id_set]
            del left[i]
            if set(result[2][0][:4]) in left:
                overlap_id.append(result[0])
            else: 
                final.append(result)

        if overlap_id:
            pos = None
            o = max(overlap_id)
            for y, row in enumerate(results):
                try:
                    pos = (y, row.index(o))
                    break
                except ValueError:
                    pass
            
            if pos:
                res_id = pos[0]
                final.append(results[res_id])
    
    else: final = results
    return final

'''
def matched_sconj(results):
    for result in results:
        if len(result[1][0]) == 5:

            ##sv sconj*1 sv
            ##v1-s1-v2-s2-sconj

            if sconj == result[1][0][4]:
                matched_id = result[2][0]
                sconj_matched.append(sconj)
                snt_matched.append(snt_count)
                temp.append((snt_count, input_snt, sconj, matched_id))#, doc)

        elif len(result[1][0]) == 6:

            ##sv sconj*2 sv
            ##2.x.1: v1-s1-v2-s2-sconj1-sconj2
            ##2.x.2: v1-s1-sconj1-v2-s2-sconj2
                                 
            if sconj in result[1][0]:
                matched_id = result[2][0]
                sconj = '-'.join(result[1][0][4:])
                #print(sconj)
                sconj_matched.append(sconj)
                snt_matched.append(snt_count)
                temp.append((snt_count, input_snt, sconj, matched_id))

        elif len(result[1][0]) == 7:

            ##sv sconj*3 sv
            ##3.x.1: v1-s1-v2-s2-sconj1-sconj2-sconj3
            ##3.x.2: v1-s1-sconj-v2-s2-as1-as2

            if sconj in result[1][0]:
                matched_id = result[2][0]
                sconj = '-'.join(result[1][0][4:])
                #print(sconj)
                sconj_matched.append(sconj)
                snt_matched.append(snt_count)
                temp.append((snt_count, input_snt, sconj, matched_id))
        
        else: continue
    return sconj
'''

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
    data = os.path.join(this_dir, 'copula_patterns')
    datum_list = pattern_datum_list(data)
    _, _, copula_list, _ = create_list(datum_list)
    for pattern in copula_list:
        if pattern != '':
            p_count+=1
            pattern_matcher(doc, token_list, pattern, copula_list)
    doc, token_list, conll = make_copula_head(doc, token_list, copula_matched)

    ##(2) Dependency Match

    results = []
    p_count=0
    data = os.path.join(this_dir, 's_patterns')
    datum_list = pattern_datum_list(data)
    id_list, const_list, pattern_list, amr_list = create_list(datum_list)
    ## id_list: ['1.1', '1.2', '1.3', ...]
    for pattern in pattern_list:
        if pattern != '':
            p_count+=1
            result = pattern_matcher(doc, token_list, pattern, pattern_list, 'complex')
            ## result: (pattern_idx, matched_token, matched_token_id)
            if result:
                results.append(result)
    #total_result(id_list, const_list, matched_id_list, p_count)
    #conll = doc._.conll_str
    init()

    ##############################################################################################################
    ## Single Match
    ##[(12, [['am', 'have', 'because']], [[1, 7, 3]])]
    ## Subset Overlap
    ##[(35, [['has', 'have', 'though']], [[17, 7, 3]]), (36, [['has', 'have', 'even', 'though']], [[17, 7, 2, 3]])]
    ## Partial Overlap
    ##[(57, [['been', 'we', ',', 'been'], ['been', 'we', ',', 'measured']], [[50, 47, 46, 8], [50, 47, 46, 38]])]
    ##############################################################################################################

    ##(3) Resolve Subset Overlap
    
    results = resolve_subset_overlap(results)
    #print(results)
    #print('---')
    
    ##(4) Resolve Partial Overlap

    return results, doc, conll, id_list

