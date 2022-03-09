import sys

import penman
from penman import constant
from penman.graph import Graph

from common import ROLES

#g2 = penman.decode("(v1 / V1    :cause|condition|concession (c / consider-01        :ARG1(v2 / V2)))")

def edit_skele(g, label):
    g1 = penman.decode(g)
    attributes = []
    for src, role, tgt in g1.edges():
        #print(src, role, tgt)
        if len(role[1:].split('|')) > 1:
            role = label
        attributes.append((src, role, tgt))

    g2 = penman.Graph(g1.instances() + attributes)
    g2 = str(penman.encode(g2))
    return g2

def disamb_skele(g, pred):
    label = ROLES[pred]
    result = edit_skele(g, label)
    return result
