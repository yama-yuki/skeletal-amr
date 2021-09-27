import penman
from penman.models.amr import model
from penman.transform import dereify_edges
from penman.surface import alignments, role_alignments

from tqdm import tqdm
import pickle

import argparse
parser = argparse.ArgumentParser(description='assign name for path_list')
parser.add_argument('-m', '--mode', help='test, amr')
parser.add_argument('-r', '--role', help='cause, condition, concession, time')
args = parser.parse_args()
mode = args.mode
role = args.role

count = 0
true = 0
sconj_list = []

pic_path = '/home/cl/yuki-yama/train/code/pickle/'+role+'.binaryfile'

def sconj_pic(pic_path):
    global sconj_list
    with open(pic_path, 'wb') as p:
        pickle.dump(sconj_list, p)
    return

def check_mode(mode):
    amr_data = '/home/cl/yuki-yama/train/data/align_all'
    out_path = '/home/cl/yuki-yama/train/data/align_'+role+'.txt'

    test = 'test'
    test_out = 'test_out'

    if mode == 'test':
        amr_data = test
        out_path = test_out
    elif mode == 'amr':
        return amr_data, out_path
    return amr_data, out_path

def main():
    amr_data, out_path = check_mode(mode)
    amr = penman.load(amr_data, model=model)

    with open(out_path, mode='w', encoding='utf-8') as f:
        for g in tqdm(amr):
            g = dereify_edges(g, model)

            rel = g.triples
            concept = g.instances()
            edge = g.edges()
            pc_var = []
            pc_con = []
            psv = []
            parent = None
            child = None
            
            for e in edge:
                if e[1] == ':'+role:
                    pc_var.append((e[0], e[2]))
                    psv.append(e)

            #print(pc_var)
            #print(concept)
            #print(edge)
            for v in pc_var:
                p_node, c_node = v[0], v[1]
                for c in concept:
                    if c[0] == p_node:
                        parent = c[2]
                    elif c[0] == c_node:
                        child = c[2]
                if parent != None and child != None:
                    pc_con.append((parent, child))
                parent, child = None, None
            state, ele = check_vsv(pc_con)
            state2 = if_true(g, state, f)
            #print(g.metadata['alignments'])
            #print(role_alignments(g))
            if state2 == True:
                find_align(g, psv, ele, f)
    print(count)
    print(true)
    sconj_pic(pic_path)
    return

def find_align(g, psv, ele, f):
    global true
    global sconj_list
    #print(g.metadata['tok'])
    e = psv[ele]
    triple = (e[0], e[1], e[2])
    #print(triple)
    al = role_alignments(g)
    #print(al)

    if triple in al:
        #print('True')
        true += 1
        tok = g.metadata['tok']
        t_list = [t for t in tok.split(' ')]
        idx = str(al[triple])[3:]
        i_list = idx.split(',')
        print(i_list)

        temp = [t_list[int(i)] for i in i_list]
        print(temp)
        sconj_list.append(temp)
        #f.write(tok+'\n')
    return

def check_vsv(pc_con):
    global count
    state = False
    num = ['0','1','2','3','4','5','6','7','8','9',0,1,2,3,4,5,6,7,8,9]
    ele = 100
    for ele in range(len(pc_con)):
        pair = pc_con[ele]
        p = [c.split('-') for c in pair]
        count_len = [len(i) if i[-1][0] in num else 0 for i in p]
        #print(all([i > 1 for i in count]))
        state = all([i > 1 for i in count_len])
        if state == True:
            count += 1
            #print(pair)
            return state, ele
    return state, ele

def if_true(g, state, f):
    if state == True:
        snt = g.metadata['tok']
        #f.write(snt+'\n')
        return True
    return False

if __name__ == '__main__':
    main()