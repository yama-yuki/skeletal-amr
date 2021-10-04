import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matcher.amr_matcher import matching
from matcher.pattern_loader import pattern_datum_list, create_list, load_dict

sentences = [
    "I am hungry because I didn't eat lunch befor I went to sleep.",
    "He went to the park after he finished his work."
]

def main():
    data = '../matcher/s_patterns'

    datum_list = pattern_datum_list(data)
    id_list, _, _, amr_list = create_list(datum_list)
    #pattern_dict = load_dict(id_list, amr_list)
    print('Loaded Pattern Dictionary')

    print(amr_list)
    conll_list=[]
    for sent in sentences:
        results, _, conll, _ = matching(sent)
        print(results)

        if results:
            conll_list.append(conll)
            for i,result in enumerate(results):
                matched_id = results[i][0]
                skeleton = amr_list[matched_id+1]
                print(skeleton)

if __name__ == '__main__':
    main()
