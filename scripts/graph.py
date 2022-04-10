import itertools
import sys

from collections import defaultdict

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

class SkelTree:
    '''
    build skeltree (skeleton structure of dep tree) from match results

    input:
    pos_list = [[1,5],[11,15],[1,11]]
    v_list = [["came","was"],["eating","was"],["came","eating"]]

    stree instance:
    Tree([[1, 5], [11, 15], [1, 11]]): defaultdict(None, {1: [5, 11], 11: [15], 5: [], 15: []})
    '''

    def __init__(self, pos_list, v_list):
        self.pos_list = pos_list
        self.v_list = v_list
        self.tree = defaultdict()
        self.__init_tree()
    
    def __init_tree(self):
        self._init_dict(self.pos_list)
        self._id2verb(self.pos_list, self.v_list)
        self._build_tree(self.pos_list)

    def _init_dict(self, pos_list):
        nodes = self._flatten(pos_list)
        nodes = list(set(nodes))
        for node in nodes:
            self.tree[node] = []
    
    def _id2verb(self, pos_list, v_list):
        key = self._flatten(pos_list)
        val = self._flatten(v_list)
        id2word = {}
        for k,v in zip(key,val):
            id2word[k] = v
        return id2word

    def _get_key(self, d, val):
        keys = [k for k, v in d.items() if val in v]
        if keys:
            return keys[0]
        return None
    
    def _build_tree(self, pos_list):
        for pos in pos_list:
            parent, child = pos
            self._add_node(parent, child)

    def _add_node(self, parent, child):
        if parent in self.tree:
            self.tree[parent].append(child)
        else:
            self.tree[parent] = [child]

    def _flatten(self, multi_list):
        return list(itertools.chain.from_iterable(multi_list))
    
    #{1: [5, 11], 11: [15], 5: [], 15: []}
    def depth(self, node):
        key = self._get_key(self.tree, node)
        if key == None:
            return 0
        else:
            return 1 + self.depth(key)
    
    @property
    def nodes(self):
        return sorted(list(set(self._flatten(self.pos_list))))

    def __repr__(self):
        return f"Tree({self.pos_list}): {self.tree}"

class Edge:
    def __init__ (self, parent, child, label=None):
        self.parent = parent
        self.child = child
        self.label = label

    def __repr__(self):
        res = "edge from " + str(self.parent) + " to " + str(self.child)
        return res

class Tree:
    ##list of edges
    def __init__(self, edgelist):
        self.edge = edgelist

    def add_edge(self, ed):
        self.edge.append(ed)

    def new_edge(self, parent, child):
        self.add_edge(Edge(parent,child))

    def __repr__(self):
        res = "parent -> child:"
        for e in self.edge:
            res += "\n" + str(e.parent) + " " + str(e.child)
        return res

