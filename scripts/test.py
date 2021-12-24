import sys

import penman
from penman.graph import Graph

g1 = penman.decode("(v1 / V1    :cause|time (v2 / V2))")
g2 = penman.decode("(v1 / V1    :cause|condition|concession (c / consider-01        :ARG1(v2 / V2)))")

g_list = [g1, g2]

def find_amb_label(g_list):

    results = []
    for g in g_list:
        edges = g.edges()
        for i,edge in enumerate(edges):
            role = edge[1][1:]
            sp = role.split('|')
            if len(sp) > 1:
                amb = True
                break
            else: amb = False
        try:
            results.append(sp)

        except: sys.exit("error loading pd")
    
    return results

print(find_amb_label(g_list))