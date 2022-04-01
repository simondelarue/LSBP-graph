"""
Top-k algorithm

Given a graph G, find k - a number of nodes - such that the subgraph composed of the top k nodes (according to a metric) has a maximum of m edges.
parameters:
    - G: a graph in the form of a list of edges (u, v)
    - metric: e.g the in-degree
    - m: maxmimum number of edges in the output subgraph
output:
    - subgraph composed of k nodes and m edges
"""

import pandas as pd
import numpy as np
from collections import defaultdict

import sknetwork as skn
from sknetwork.data import convert_edge_list
from sknetwork.visualization import svg_digraph
from IPython.display import SVG

def get_nodes(serie):
    return int(serie['src']), int(serie['dst'])

if __name__=='__main__':

    TOY = True
    DATA_DIR = 'data'

    print('Load edges...')
    if TOY:
        #filename = 'soc-LiveJournal1_toy_bis' 
        filename = 'topk_toy'
        edges = pd.read_csv(f'{DATA_DIR}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter=',')
    else:
        filename = 'soc-LiveJournal1'
        edges = pd.read_csv(f'{DATA_DIR}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter='\t')


    """
    m = 6
    m_tmp = 0
    min_node_deg = 1e9 # initial value for minimum node degree in top dict
    scnd_min_node_deg = 1e9 # initial value for second minimum node degree in top dict
    in_degrees = {} # key=node, value=in_deg
    top = defaultdict(list) #key=node, value=list of (source) nodes
    #deg_node = defaultdict(list) #key:degree, value=list of nodes

    for idx, e in edges.iterrows():
        s, d = get_nodes(e)
        
        in_degrees[d] = in_degrees.get(d, 0) + 1 # count node in-degrees
        top[d] = top.get(d, []) + [s] # add source node to list
        m_tmp += 1
        print(f'Incoming node: ({d},{in_degrees[d]})')

        # if we add a node in top, with a degree lower than previous nodes, we track this node as the minimal node in the dict
        # if we exceed the max number of nodes in the 'top' dict, this min node will be poped out
        if len(top[d]) < min_node_deg:
            min_node = d
            min_node_deg = len(top[d])
        
        # a optimiser
        # Pour chaque noeud dans topk, on calcule le nombre de voisin et on garde le minimum
        d = sorted(top.items(), key=lambda x: len(x[1]))[0]
        min_node = d[0]
        min_node_deg = len(d[1])

        # Size of 'top' dict exceeds m -> we need to pop out a node
        if m_tmp > m:
            m_tmp -= len(top.get(min_node))
            print(f'POPPED OUT minimal node: {min_node} - {len(top.get(min_node))}')
            top.pop(min_node)
            
        print(f'In-degrees: {in_degrees}')
        print(top)
        print(f'Minimal node: ({min_node}, {min_node_deg})')
        print(f'Temp m {m_tmp}\n')"""

    m = 6
    m_tmp = 0
    min_node_deg = 1e9 # initial value for minimum node degree in top dict
    scnd_min_node_deg = 1e9 # initial value for second minimum node degree in top dict
    in_degrees = {} # key=node, value=in_deg
    top = defaultdict(list) #key=node, value=list of (source) nodes

    for idx, e in edges.iterrows():
        s, d = get_nodes(e)
        
        in_degrees[d] = in_degrees.get(d, 0) + 1 # count node in-degrees
        in_degrees[s] = in_degrees.get(s, 0)
        top[d] = top.get(d, []) + [s] # add destination node and its source
        top[s] = top.get(s, []) + [] # add source node 
        m_tmp += 1
        print(f'Incoming node: ({s}->{d})')

        # if we add a node in top, with a degree lower than previous nodes, we track this node as the minimal node in the dict
        # if we exceed the max number of nodes in the 'top' dict, this min node will be poped out
        if len(top[d]) < min_node_deg:
            min_node = d
            min_node_deg = len(top[d])
        
        # a optimiser
        # Pour chaque noeud dans topk, on calcule le nombre de voisin et on garde le minimum
        dic = sorted(top.items(), key=lambda x: len(x[1]))[0]
        min_node = dic[0]
        min_node_deg = len(dic[1])

        # Size of 'top' dict exceeds m -> we need to pop out a node
        if m_tmp > m:
            m_tmp -= len(top.get(min_node)) + 1 # + 1 refers to the removal of minimal node linked to detination node
            print(f'POPPED OUT minimal node: {min_node} (with indegree={len(top.get(min_node))})')
            top.get(d).pop() # remove minimal node linked to destination node
            top.pop(min_node)
            
            
        print(f'In-degrees: {in_degrees}')
        print(top)
        print(f'Minimal node: ({min_node}, {min_node_deg})')
        print(f'Temp m {m_tmp}')
        print(f'Total in degree: {np.sum([len(v) for v in top.values()])}\n')

    # SHUFFLE DataFrame
    print('================================================')
    edges = edges.sample(frac=1)

    m = 6
    m_tmp = 0
    min_node_deg = 1e9 # initial value for minimum node degree in top dict
    scnd_min_node_deg = 1e9 # initial value for second minimum node degree in top dict
    in_degrees = {} # key=node, value=in_deg
    top = defaultdict(list) #key=node, value=list of (source) nodes

    for idx, e in edges.iterrows():
        s, d = get_nodes(e)
        
        in_degrees[d] = in_degrees.get(d, 0) + 1 # count node in-degrees
        in_degrees[s] = in_degrees.get(s, 0)
        top[d] = top.get(d, []) + [s] # add destination node and its source
        top[s] = top.get(s, []) + [] # add source node 
        m_tmp += 1
        print(f'Incoming node: ({s}->{d})')

        # if we add a node in top, with a degree lower than previous nodes, we track this node as the minimal node in the dict
        # if we exceed the max number of nodes in the 'top' dict, this min node will be poped out
        if len(top[d]) < min_node_deg:
            min_node = d
            min_node_deg = len(top[d])
        
        # a optimiser
        # Pour chaque noeud dans topk, on calcule le nombre de voisin et on garde le minimum
        dic = sorted(top.items(), key=lambda x: len(x[1]))[0]
        min_node = dic[0]
        min_node_deg = len(dic[1])

        # Size of 'top' dict exceeds m -> we need to pop out a node
        if m_tmp > m:
            m_tmp -= len(top.get(min_node)) + 1 # + 1 refers to the removal of minimal node linked to detination node
            print(f'POPPED OUT minimal node: {min_node} (with indegree={len(top.get(min_node))})')
            top.get(d).pop() # remove minimal node linked to destination node
            top.pop(min_node)
            
            
        print(f'In-degrees: {in_degrees}')
        print(top)
        print(f'Minimal node: ({min_node}, {min_node_deg})')
        print(f'Temp m {m_tmp}')
        print(f'Total in degree: {np.sum([len(v) for v in top.values()])}\n')
