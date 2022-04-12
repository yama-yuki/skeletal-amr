import sys
from pprint import pprint
from ast import literal_eval

def pattern_entry_list(pattern_path):
    ##patternをidごとにentry_listへ
    entry_list = []
    with open(pattern_path, mode='r', encoding='utf-8') as f:
        temp = []
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                temp.append(line)
            else:
                entry_list.append(temp)
                temp = []
        if temp:
            entry_list.append(temp)

    return entry_list

def create_list(entry_list):
    ##個別の情報を各listに分解
    id_list, const_list, pattern_list, amr_list, pos_list = [], [], [], [], [] 

    for entry in entry_list:
        sanity = [False]*5
        pat, amr = False, False
        pattern_temp, amr_temp = '', []

        for i,line in enumerate(entry):
            ##headers
            if i == 0:
                id_list.append(line[7:].rstrip())
                sanity[0] = True
            elif i == 1:
                const_list.append(line[10:].rstrip())
                sanity[1] = True

            ##flags up
            elif line[:7] == '# ::pat':
                pat = True
                sanity[2] = True
            elif line[:7] == '# ::amr':
                amr = True
            
            ##flags down
            elif line[:7] == '# ::snt':
                pat = False
                pattern_list.append(pattern_temp)
                sanity[3] = True
            elif line[:7] == '# ::pos':
                amr = False
                amr_list.append(amr_temp)
                pos = line[8:].rstrip()[1:-1].split(',') ##['v1', 's1', 'p', 'v2']
                pos_list.append(pos)
                sanity[4] = True

            ##merge
            elif pat is True:
                pattern_temp += line.rstrip()
            elif amr is True:
                amr_temp.append(line)
        
        if all(sanity) == False:
            sys.exit('Something is missing in an entry')

    return id_list, const_list, pattern_list, amr_list, pos_list

def load_dict(id_list, amr_list):
        pattern_dict = {key: val for key, val in zip(id_list, amr_list)}
        return pattern_dict

def wo_nsubj(pattern_dict):
    new_dict = []

    for p in pattern_dict:
        if p == {"SPEC":{"NODE_NAME":"v2","NBOR_RELOP":">","NBOR_NAME":"v1"},"PATTERN":{"TAG": {"REGEX": "^V"},"DEP":"advcl"}}:
            p["PATTERN"]["TAG"]["REGEX"] = '^V|^M'
            new_dict.append(p)
        elif p == {"SPEC":{"NODE_NAME":"v1"},"PATTERN":{"TAG": {"REGEX": "^V"}}}:
            p["PATTERN"]["TAG"]["REGEX"] = '^V|^M'
            new_dict.append(p)
        elif p != {'SPEC': {'NODE_NAME': 's1', 'NBOR_RELOP': '>', 'NBOR_NAME': 'v1'}, 'PATTERN': {'DEP': {'REGEX': '^nsubj'}}}\
            and p != {'SPEC': {'NODE_NAME': 's2', 'NBOR_RELOP': '>', 'NBOR_NAME': 'v2'}, 'PATTERN': {'DEP': {'REGEX': '^nsubj'}}}:
            new_dict.append(p)

    return new_dict

if __name__ == '__main__':
    pd_path = 'pattern_dict'
    entry_list = pattern_entry_list(pd_path)
    id_list, const_list, pattern_list, amr_list, pos_list = create_list(entry_list)
    print(pos_list)
    #print(id_list[0], const_list[0], amr_list[0], pos_list[0])

'''
##legacy
def create_list(datum_list):
    id_list, const_list, pattern_list, amr_list = [], [], [], []

    pat = False
    amr = False
    pattern_temp = ''
    amr_temp = []

    for datum in datum_list:
        for d in datum:
            if d[:6] == '# ::id':
                amr = False
                amr_list.append(amr_temp)
                amr_temp = []
                id_list.append(d[7:].rstrip())
            elif d[:7] == '# ::con':
                const_list.append(d[10:].rstrip())
            elif d[:7] == '# ::pat':
                pat = True
            elif d[:7] == '# ::amr':
                pat = False
                pattern_list.append(pattern_temp)
                pattern_temp = ''
                amr = True
            elif d[:7] == '# ::snt':
                pass
            elif pat is True:
                pattern_temp += d.rstrip()
            elif amr is True:
                amr_temp.append(d)

    return id_list, const_list, pattern_list, amr_list
'''