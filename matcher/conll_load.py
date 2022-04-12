from spacy.tokens import Doc

def convert(cols, matched_token_idx):
    #temp_path = os.path.join(this_dir,'temp.conllu')

    matched_i = [list(map(lambda x: str(x+1), matched)) for matched in matched_token_idx]
    for matched in matched_i:
        cols = change_head(matched, cols)

    return cols

def change_head(matched, cols):

    new_cols = []
    comp,be = matched[0],matched[1]

    dep_comp = cols[int(comp)-1][7]
    dep_be = cols[int(be)-1][7]
    head_comp = cols[int(comp)-1][6]

    for j,col in enumerate(cols):
        col_i, head_i, dep = str(j+1), col[6], col[7]
        if head_i == comp:
            col[6] = be
        if col_i == be:
            col[6], col[7] = head_comp, dep_comp
        elif col_i == comp:
            col[6], col[7] = be, dep_be
        new_cols.append(col)

    return new_cols

def conll_list_to_doc(vocab, conll):
    ## conll_list for a single sentence
    ## [['12', '31', '31', 'NUM', 'CD', 'NumType=Card', '13', 'nummod', '_', 'start_char=242|end_char=244'], ['13', 'October', 'October', 'PROPN', 'NNP', 'Number=Sing', '10', 'obl', '_', 'start_char=245|end_char=252']]

    words, spaces, tags, poses, morphs, lemmas = [], [], [], [], [], []
    heads, deps = [], []
    for i in range(len(conll)):
        line = conll[i]
        parts = line
        id_, word, lemma, pos, tag, morph, head, dep, _, misc = parts

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
##legacy
def process(doc):

    text = str(doc)
    doc.is_parsed = True
    doc.is_tagged = True
    token_list = [token.text for token in doc]

    return text, doc, token_list
'''