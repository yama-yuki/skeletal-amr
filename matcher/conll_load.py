from pprint import pprint
import os, sys
this_dir, this_filename = os.path.split(__file__)
conll_data = os.path.join(this_dir, 'temp.conllu')

import spacy
from spacy.tokens import Doc
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from spacy_conll import init_parser

#nlp = spacy.load('en_core_web_sm')
#spacy.load('en_core_web_sm')
#conll = [[['1', 'Test', '_', 'NOUN', 'NN', 'Number=Sing', '0', '_', '_', 'start_char=0|end_char=4'], ['2', 'sentence', '_', 'NOUN', 'NN', 'Number=Sing', '1', '_', '_', 'start_char=5|end_char=13'], ['3', '.', '_', 'PUNCT', '.', '_', '2', '_', '_', 'start_char=13|end_char=14']]] # conll is List[List[List]], representing each token / word in each sentence in the document
#dicts = CoNLL.convert_conll(conll) # dicts is List[List[Dict]], representing each token / word in each sentence in the document

nlp = init_parser("stanza",
                  "en",
                  parser_opts={"use_gpu": True, "verbose": False},
                  include_headers=True)

def convert(conll_data, change_idx, child_idx):
    snt,tmp=[],[]
    #print(change_idx)
    #print(child_idx)
    
    conll_snt=[]
    with open(conll_data, mode='r', encoding='utf-8') as d:
        lines = d.readlines()
        for line in lines:
            if line[0] != '#' and line[0] != '\n':
                conll_snt.append(line.rstrip('\n').split('\t'))

    done_be = []
    done_comp = []
    for i in range(len(change_idx)):
        comp = change_idx[i][0]
        be = change_idx[i][1]
        done_comp.append(comp)
        done_be.append(be)

        if child_idx[i][0]:
            for cid in child_idx[i][0]:
                if cid not in done_be and cid not in done_comp:
                    conll_snt[cid-1][6] = str(be)
            for cid in child_idx[i][1]:
                if cid not in done_be and cid not in done_comp:
                    conll_snt[cid-1][6] = str(comp)

        comp_head = conll_snt[comp-1][6]
        comp_dep = conll_snt[comp-1][7]
        be_head = conll_snt[be-1][6] #comp
        be_dep = conll_snt[be-1][7]

        conll_snt[comp-1][6] = str(be)
        conll_snt[comp-1][7] = be_dep
        conll_snt[be-1][6] = str(comp_head)
        conll_snt[be-1][7] = comp_dep

    for i in range(len(change_idx)):
        comp = change_idx[i][0]
        be = change_idx[i][1]

        if int(conll_snt[be-1][6]) in done_comp:
            for j in change_idx:
                if j[0] == int(conll_snt[be-1][6]):
                    conll_snt[be-1][6] = str(j[1])

    #print('--------------------')
    #for i in conll_snt:
        #print(i)
    #print(conll)
    #dicts = CoNLL.convert_conll(conll)
    #pprint(dicts)
    conll = conll_snt
    #print(conll)
    return conll

def process(doc):
    #print(doc)
    text = str(doc)
    doc.is_parsed = True
    doc.is_tagged = True
    token_list=[]
    for token in doc:
        token_list.append(token.text)
        #print(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.head.i, token.dep_, token.head.text)
        #[0]token_id|[1]token|[2]lemma|[3]token_pos|[4]token_tag|[5]head_id|[6]dep|[7]head_token
    return text, doc, token_list

def conll_to_doc(vocab, conll):
    words, spaces, tags, poses, morphs, lemmas = [], [], [], [], [], []
    heads, deps = [], []
    for i in range(len(conll)):
        line = conll[i]
        parts = line
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if "." in id_ or "-" in id_:
            continue
        if "SpaceAfter=No" in misc:
            spaces.append(False)
        else:
            spaces.append(True)

        id_ = int(id_) - 1
        head = (int(head) - 1) if head not in ("0", "_") else id_
        tag = pos if tag == "_" else tag
        morph = morph if morph != "_" else ""
        dep = "ROOT" if dep == "root" else dep

        words.append(word)
        lemmas.append(lemma)
        poses.append(pos)
        tags.append(tag)
        morphs.append(morph)
        heads.append(head)
        deps.append(dep)

    doc = Doc(vocab, words=words, spaces=spaces)
    for i in range(len(doc)):
        doc[i].tag_ = tags[i]
        doc[i].pos_ = poses[i]
        doc[i].dep_ = deps[i]
        doc[i].lemma_ = lemmas[i]
        doc[i].head = doc[heads[i]]
    doc.is_parsed = True
    doc.is_tagged = True

    return doc

'''
def old_conll_to_doc(vocab, conll):
    words, spaces, tags, poses, morphs, lemmas = [], [], [], [], [], []
    heads, deps = [], []
    lines = conll
    token_id = 0

    for i in range(len(lines)):
        line = lines[i]
        parts = line
        original_id, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if "." in original_id or "-" in original_id:
            continue
        if "SpaceAfter=No" in misc:
            spaces.append(False)
        else:
            spaces.append(True)

        head = token_id + (int(head) - int(original_id))
        tag = pos if tag == "_" else tag
        morph = morph if morph != "_" else ""
        dep = "ROOT" if dep == "root" else dep

        words.append(word)
        lemmas.append(lemma)
        poses.append(pos)
        tags.append(tag)
        morphs.append(morph)
        heads.append(head)
        deps.append(dep)
        
        token_id += 1

    doc = Doc(vocab, words=words, spaces=spaces)
    
    for i in range(len(doc)):
        doc[i].tag_ = tags[i]          
        doc[i].pos_ = poses[i]
        doc[i].dep_ = deps[i]
        doc[i].lemma_ = lemmas[i]
        doc[i].head = doc[heads[i]]
    doc.is_parsed = True
    doc.is_tagged = True

    return doc

def old_convert(conll_data, change_idx, child_idx, V1):
    snt,tmp=[],[]
    print(change_idx)
    print(child_idx)

    with open(conll_data, mode='r', encoding='utf-8') as d:
        lines = d.readlines()
        for line in lines:
            if line[0] != '#' and line[0] != '\n':
                l = line.rstrip('\n').split('\t')
                #print(l)
                for i,child in zip(change_idx,child_idx):

                    if i != []: # mat/sub=cop
                        if child[0] != []: #comp has child
                            for c in child[0]:
                                if int(l[0])-1 == c.i:
                                    l[6] = str(i[1]+1) #move child to cop

                        if child[1] != []: #cop has child
                            for c in child[1]:
                                if int(l[0])-1 == c.i:
                                    l[6] = str(i[0]+1) #move child to comp
                
                if change_idx[0] != []: #mat=cop
                    if int(l[0])-1 == change_idx[0][0]: #is comp
                        l[6] = str(change_idx[0][1]+1) #change parent to cop
                        l[7] = 'cop'
                    if int(l[0])-1 == change_idx[0][1]: #is cop
                        l[7] = 'root'

                if change_idx[1] != []: # sub=cop
                    if int(l[0])-1 == change_idx[1][0]: #is comp
                        l[6] = str(change_idx[1][1]+1)
                        temp_dep = l[7]
                        print(temp_dep)
                        l[7] = 'cop'

                if l[0] != '':
                    tmp.append(l)
                    #print(l)

            if line == '\n':
                #print(tmp)
                snt.append(tmp)
                tmp=[]

        #print(tmp)
        try:
            if change_idx[1] != []:
                for l in tmp:
                    if int(l[0])-1 == change_idx[1][1]:
                        if V1 != None:
                            l[6] = str(int(V1)+1)
                        elif change_idx[0] != []:
                            l[6] = str(change_idx[0][1]+1)
                        l[7] = temp_dep
        except:
            return None

        snt.append(tmp)
        tmp=[]

    conll = snt
    #print(conll)
    #dicts = CoNLL.convert_conll(conll)
    #print(dicts)
    #pprint(dicts)
    #doc = Document(dicts)
    #print(doc)

    return conll

def conll_to_doc(conll_dicts):
    snt = []
    words = []

    for word in conll_dicts:
        #print(word)
        words.append(word['text'])
        snt.append(word)

    #print('snt')
    #print(snt)
    #print(words)

    doc = Doc(nlp.vocab, words=words)
    for i,w in enumerate(snt):
        doc[i].pos_ = snt[i]['upos']
        if snt[i]['head'] != 0:
            doc[i].head = doc[snt[i]['head']-1]

    head=[]
    for i,w in enumerate(snt):
        doc[i].pos_ = snt[i]['upos']
        doc[i].tag_ = snt[i]['xpos']
        doc[i].dep_ = snt[i]['deprel']
        doc[i].lemma_ = snt[i]['lemma']
        if snt[i]['head'] != 0:
            doc[i].head = doc[snt[i]['head']-1]
            head.append(snt[i]['head']-1)
        else:
            doc[i].head = doc[i]
            head.append(int(doc[i].head.i))
        #doc[i].pos_ = snt[i]['upos']

    for token in doc:
        print(token.i, token.text, token.lemma_, token.pos_, token.dep_, token.head, token.head.i)
    #print(head)

    return doc
'''

