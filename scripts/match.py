from matcher.amr_matcher import matching

sentences = [
    "I am hungry because I didn't eat lunch befor I went to sleep.",
    "He went to the park after he finished his work."
]

conll_list=[]
for sent in sentences:
    results, doc, conll, _ = matching(sent)
    print(results)
    print(conll)

    if results:
        conll_list.append(conll)
        temp = []
        for result in results:
            '''
            I am hungry because I didn't have lunch.
            [(12, [['am', 'have', 'because']], [[1, 7, 3]])]

            (35, [['has', 'have', 'though']], [[17, 7, 3]])
            (36, [['has', 'have', 'even', 'though']], [[17, 7, 2, 3]])
            '''
            pass
            
