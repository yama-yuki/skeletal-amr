def total_result(id_list, const_list, matched_id, p_count):
    mp_id = [id_list[i-1] for i in matched_id]
    #print('Matched Pattern_ID: '+str(mp_id))
    mp = [const_list[i-1] for i in matched_id]
    if mp != []:
        #print('Matched Patterns: '+str(mp))
        return
    return mp_id, mp

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



data = os.path.join(this_dir, 'copula_patterns')
datum_list = pattern_datum_list(data)
_, _, copula_list, _ = create_list(datum_list)
for pattern in copula_list:
    if pattern != '':
        p_count+=1
        pattern_matcher(doc, token_list, pattern, copula_list, 'copula')
doc, token_list, conll = make_copula_head(doc, token_list, copula_matched)


def make_copula_head(doc, token_list, copula_matched):

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

    with open(TEMP,mode='w',encoding='utf-8') as f:
        conll = doc._.conll_str
        f.write(conll)
    
    if change_idx != []:
        conll = convert(TEMP, change_idx, child_idx)
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

