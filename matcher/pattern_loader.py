from pprint import pprint

##(1) patternをidごとにdatum_listへ

def pattern_datum_list(data):
    datum_list = []
    with open(data, mode='r', encoding='utf-8') as f:
        datum_temp = []
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                datum_temp.append(line)
            else:
                datum_list.append(datum_temp)
                datum_temp = []
        datum_list.append(datum_temp)
    return datum_list

##(2) 個別の情報を各listに分解

def create_list(datum_list):
    id_list = []
    const_list = []
    pattern_list = []
    amr_list = []

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

def load_dict(id_list, amr_list):
        pattern_dict = {key: val for key, val in zip(id_list, amr_list)}
        return pattern_dict

if __name__ == '__main__':
    data = 'pattern_dict'
    datum_list = pattern_datum_list(data)
    id_list, const_list, pattern_list, amr_list = create_list(datum_list)
    pprint(amr_list)
