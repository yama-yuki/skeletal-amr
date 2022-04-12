from graph import SkelTree

def find_matched_pos(pos_list):
    '''
    for each entry in a pattern dictionary, return positions of 
    matrix verb, subordinate verb, sub. conjunction (, corr. conjunction)

    input: pos_list
        [['v1', 's1', 'a2', 'v2', 'a1']]
        -> [['v1', 'a2', 'v2', 'a1']]
    output: result
        [[0, 2, [1, 3], []]]
    '''

    result = []
    for pos in pos_list:
        if 's1' in pos:
            pos.remove('s1')
        if 's2' in pos:
            pos.remove('s2')

    for pos in pos_list:
        if 'v1' in pos:
            mat_pos = pos.index('v1')
        else: mat_pos = None
        if 'v2' in pos:
            sub_pos = pos.index('v2')
        else: sub_pos = None
        
        a_pos_list, b_pos_list = [], []
        pos_sp = [list(p) for p in pos]
        a_len, b_len = sum(p[0]=='a' for p in pos_sp), sum(p[0]=='b' for p in pos_sp)
        if a_len > 0:
            for idx in range(a_len):
                idx = str(idx+1)
                a_pos = 'a'+idx
                a_pos_list.append(pos.index(a_pos))
        if b_len > 0:
            for idx in range(b_len):
                idx = str(idx+1)
                b_pos = 'b'+idx
                b_pos_list.append(pos.index(b_pos))

        result.append([mat_pos, sub_pos, a_pos_list, b_pos_list])

    return result

def split_clause(results, positions, doc):
    '''
    input: results
        [(5, [['came', 'was', 'before']], [[1, 5, 3]]), 
        (13, [['eating', 'was', 'as']], [[11, 15, 13]]), 
        (20, [['came', 'eating', 'while']], [[1, 11, 8]])]
    output: clauses
        []
    '''
    clauses = []

    pos_list, v_list = [], []
    for res in results:
        #res = (5, [['came', 'was', 'before']], [[1, 5, 3]])
        const_id = res[0]
        matched_verb = res[1][0]
        matched_id = res[2][0]

        matv_pos, subv_pos = positions[const_id][:2]
        pos_list.append([matched_id[matv_pos], matched_id[subv_pos]]) #[[1,5],[11,15],[1,11]]
        v_list.append([matched_verb[matv_pos], matched_verb[subv_pos]]) #[["came","was"],["eating","was"],["came","eating"]]

    stree = build_skeltree(pos_list, v_list)
    clauses = stree

    return clauses

def make_input(doc, matv_i, subv_i, sc_i, sc, *args):
    '''
    make a pair of clauses for disambiguation

    input:
        doc: input sentence in spacy's doc format
        matv_i/subv_i/sc_i: position id of matrix/subordinate verb or sconj
        sc: sconj

    output: 
        clause_pair: (subord, matrix, sconj)
    '''

    sconj = ' '.join([t.text.lower() for t in sc])

    ##sentence with 2 clauses
    if not args:
        matv_childtree = [t for t in doc[matv_i].subtree]
        subv_childtree = [t for t in doc[subv_i].subtree]
        ##get matrix clause
        mat_clause = [t.text for t in matv_childtree if t not in subv_childtree] #diff
        mat_clause = _process_toks(mat_clause) #punct
        ##get subordinate clause
        sub_clause = [t.text for t in subv_childtree if t.i not in sc_i]
        ##make clause_pair
        matrix, subord = ' '.join(mat_clause), ' '.join(sub_clause)
        clause_pair = (subord, matrix, sconj)

        return clause_pair
    
    ##sentence with 2< clauses
    else:
        stree = args[0]
        matv_childs = stree.tree[matv_i]
        subv_childs = stree.tree[subv_i]
        matv_childtree = [t for t in doc[matv_i].subtree]
        subv_childtree = [t for t in doc[subv_i].subtree]

        ##matv with only one subordinate clause
        if len(matv_childs) == 1:
            ##get matrix clause
            mat_clause = [t.text for t in matv_childtree if t not in subv_childtree]
            mat_clause = _process_toks(mat_clause)

            ##subv with no subordinate clause
            if not subv_childs:
                ##get subordinate clause
                sub_clause = [t.text for t in subv_childtree if t.i not in sc_i]
                sub_clause = _process_toks(sub_clause)
                ##make clause_pair
                matrix, subord = ' '.join(mat_clause), ' '.join(sub_clause)
                clause_pair = (subord, matrix, sconj)

                return clause_pair

            ##subv with subordinate clause
            else:
                ##get subordinate clause
                temp_subv_childtree = []
                for subv_child in subv_childs:
                    temp_subv_childtree.extend([t for t in doc[subv_child].subtree])
                sub_clause = [t.text for t in subv_childtree if (t not in temp_subv_childtree) and (t.i not in sc_i)]
                sub_clause = _process_toks(sub_clause)
                ##make clause_pair
                matrix, subord = ' '.join(mat_clause), ' '.join(sub_clause)
                clause_pair = (subord, matrix, sconj)

                return clause_pair

        ##matv with multiple subordinate clause
        else:
            ##get matrix clause
            temp_matv_childtree = []
            for matv_child in matv_childs:
                temp_matv_childtree.extend([t for t in doc[matv_child].subtree])
            mat_clause = [t.text for t in matv_childtree if t not in temp_matv_childtree]
            mat_clause = _process_toks(mat_clause)

            ##subv with no subordinate clause
            if not subv_childs:
                ##get subordinate clause
                sub_clause = [t.text for t in subv_childtree if t.i not in sc_i]
                sub_clause = _process_toks(sub_clause)
                ##make clause_pair
                matrix, subord = ' '.join(mat_clause), ' '.join(sub_clause)
                clause_pair = (subord, matrix, sconj)

                return clause_pair

            ##subv with subordinate clause
            else:
                ##get subordinate clause
                temp_subv_childtree = []
                for subv_child in subv_childs:
                    temp_subv_childtree.extend([t for t in doc[subv_child].subtree])
                sub_clause = [t.text for t in subv_childtree if t not in temp_subv_childtree]
                sub_clause = _process_toks(sub_clause)
                ##make clause_pair
                matrix, subord = ' '.join(mat_clause), ' '.join(sub_clause)
                clause_pair = (subord, matrix, sconj)

                return clause_pair

def build_skeltree(pos_list, v_list):
    '''
    build skeletree (skeleton structure of dep tree) from match results

    input:
    pos_list = [[1,5],[11,15],[1,11]]
    v_list = [["came","was"],["eating","was"],["came","eating"]]

    output:
    Tree([[1, 5], [11, 15], [1, 11]]): defaultdict(None, {1: [5, 11], 11: [15], 5: [], 15: []})
    '''

    stree = SkelTree(pos_list, v_list)

    return stree

def _process_toks(toks):
    #toks_str = [t.text for t in toks]
    if toks[0] == ',':
        toks = toks[1:]
    if toks[-1] == '.':
        toks = toks[:-1]
    return toks

def _find_heads(mat_v, sub_v):
    head_cands = [v for v in mat_v]
    for i,cand in enumerate(head_cands):
        if cand in sub_v:
            head_cands.remove(cand)
    heads = sorted(list(set(head_cands)))
    return heads

if __name__ == '__main__':
    #pos_list = [['v1', 's1', 'a1', 'v2'], ['v1', 's1', 'a2', 'v2', 'a1'], ['v1', 's1', 'a2', 'v2', 'a1']]
    #find_matched_pos(pos_list)

    #matched_results = [(5, [['came', 'was', 'before']], [[1, 5, 3]]), (13, [['eating', 'was', 'as']], [[11, 15, 13]]), (20, [['came', 'eating', 'while']], [[1, 11, 8]])]
    pos_list = [[1,5],[11,15],[1,11]]
    v_list = [["came","was"],["eating","was"],["came","eating"]]
    s = SkelTree(pos_list, v_list)
    print(s)
    print(s.depth(15))
    nodes = s.nodes
    depth = [s.depth(node) for node in s.nodes]
    print(depth)
    max_d = max(depth)
    dep2node = {}
    for k,v in zip(depth,nodes):
        if k not in dep2node:
            dep2node[k] = [v]
        else:
            dep2node[k].append(v)
    print(dep2node)

    for d in range(max_d+1):
        print(d,dep2node[d])
        childs = dep2node[d]

    #print(build_skeltree(pos_list, v_list))

