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

from collections import defaultdict
from tqdm import tqdm
import os
import random

from LSBP.utils import Bunch
from LSBP.data.load import sniff_delimiter, listdir_fullpath

def topk(filename: str, outdir: str, metric: str = 'indegree') -> Bunch():
    ''' Topk algorithm to filter data in linear time. 
    
    Parameter
    ---------
        filename: str
            Name of data file.
        metric: str (default='indegree')
            Parameter used to filter data. 
    
    Output
    ------
        Graph as a Bunch. '''

    graph = Bunch()

    label2idx, idx2label = {}, {}
    in_deg, out_deg = {}, {}
    idx = 0
    m_tmp = 0
    V_tmp_top = {}
   
    
    if filename.endswith('soc-LiveJournal1.txt'):
        nb_rows = 68993774
    else:
        nb_rows = None

    with open(filename) as f:

        delimiter = sniff_delimiter(filename)
        outfile = os.path.join(outdir, f'{os.path.basename(outdir)}')

        for line in tqdm(f, total=nb_rows):
    
            vals = line.strip('\n').split(delimiter)
            src, dst = vals[0], vals[1]
            if src != '' and src != 'Source':

                if src not in label2idx:
                    label2idx[src] = idx
                    idx2label[idx] = src
                    idx += 1
                if dst not in label2idx:
                    label2idx[dst] = idx
                    idx2label[idx] = dst
                    idx += 1

                out_deg[label2idx.get(src)] = out_deg.get(label2idx.get(src), 0) + 1
                in_deg[label2idx.get(dst)] = in_deg.get(label2idx.get(dst), 0) + 1
                
                # TODO: optimize initialization of every existing nodes
                if label2idx.get(dst) not in out_deg:
                    out_deg[label2idx.get(dst)] = 0
                if label2idx.get(src) not in in_deg:
                    in_deg[label2idx.get(src)] = 0

                # Add nodes
                #print('--------------------------------')
                #rint(f'\n{src}->{dst}')

                #print(f'src label: {label2idx.get(src)} -dst label: {label2idx.get(dst)}')
                #print(in_deg)

                #print(f'In degree of source node: {in_deg.get(label2idx.get(src))}')
                if in_deg.get(label2idx.get(src)) > 0:

                    V_tmp_top[dst] = V_tmp_top.get(dst, 0) + 1
                    V_tmp_top[src] = V_tmp_top.get(src, 1)
                    m_tmp += 1
                #print(f'v_tmp : {V_tmp_top}')
                #print(f'm_tmp : {m_tmp}')

        #nodes = dict(sorted(V_tmp_top.items(), key=lambda x: x[1], reverse=True))
        nodes = {k: v for k, v in V_tmp_top.items() if v > 10}
        #print(f'Nodes: {nodes}')
        print(f'Number of nodes retained in dense subraph: {len(nodes)}')

    # Write to result file
    cpt = 0
    with open(f'{outfile}_topk', 'a') as o:
        with open(filename) as f:
            delimiter = sniff_delimiter(filename)
            outfile = os.path.join(outdir, f'{os.path.basename(outdir)}')
            for line in tqdm(f, total=nb_rows):
                vals = line.strip('\n').split(delimiter)
                src, dst = vals[0], vals[1]
                if nodes.get(src) is not None and nodes.get(dst) is not None:
                    o.write(f'{label2idx.get(src)},{label2idx.get(dst)}\n')
                    cpt += 1
    print(f'Number of edges retained in dense subgraph: {cpt}')
        
    graph.nb_edges = idx # True number of nodes in total graph
    graph.nb_nodes = len(label2idx)
    graph.in_degrees = in_deg
    graph.out_degrees = out_deg
    graph.label2idx = label2idx
    graph.idx2label = idx2label
    graph.files = listdir_fullpath(outdir)  
    graph.outdir = outdir

    return graph

def topk_old(filename: str, outdir: str, metric: str = 'indegree') -> Bunch():
    ''' Topk algorithm to filter data in linear time. 
    
    Parameter
    ---------
        filename: str
            Name of data file.
        metric: str (default='indegree')
            Parameter used to filter data. 
    
    Output
    ------
        Graph as a Bunch. '''

    graph = Bunch()

    label2idx, idx2label = {}, {}
    in_deg, out_deg = {}, {}
    idx = 0
    m_max = 100000
    m_tmp = 0
    V_cache = set()
    V_tmp_top = defaultdict(set)
    V_tmp_top_inv = defaultdict(set)
    V_res = set()
    min_degree = 1
    min_degree_tmp = 1e9
    j = 0
    
    with open(filename) as f:

        delimiter = sniff_delimiter(filename)
        outfile = os.path.join(outdir, f'{os.path.basename(outdir)}')

        for line in tqdm(f, total=689000000):
    
            vals = line.strip('\n').split(delimiter)
            src, dst = vals[0], vals[1]
            if src != '' and src != 'Source':

                if src not in label2idx:
                    label2idx[src] = idx
                    idx2label[idx] = src
                    idx += 1
                if dst not in label2idx:
                    label2idx[dst] = idx
                    idx2label[idx] = dst
                    idx += 1

                out_deg[label2idx.get(src)] = out_deg.get(label2idx.get(src), 0) + 1
                in_deg[label2idx.get(dst)] = in_deg.get(label2idx.get(dst), 0) + 1
                
                # TODO: optimize initialization of every existing nodes
                if dst not in out_deg:
                    out_deg[label2idx.get(dst)] = 0
                if src not in in_deg:
                    in_deg[label2idx.get(src)] = 0

                # Add nodes
                #rint('--------------------------------')
                #print(f'\n{src}->{dst}')
                
                V_cache.add(dst)
                
                if src in V_cache:

                    #V_tmp_top[dst] = V_tmp_top.get(dst, 0) + 1
                    #m_tmp = sum([len(vals) for vals in V_tmp_top.values()])

                    # Dense subgraph history such as dst: {src}
                    V_tmp_top[dst] |= {src}
                    # Dense subgraph history such as src: {dst}
                    V_tmp_top_inv[src] |= {dst}
                    m_tmp += 1

                # Remove nodes if number of edges is higher than m_max
                while m_tmp > m_max:
                    #print(f'{m_tmp} > {m_max}')
                    #print('REMOVE NODES')
                    
                    # sort all nodes and keep node with minimal degree
                    #min_deg = sorted({k: len(v) for k, v in V_tmp_top.items()}.items(), key=lambda x: x[1])[0]
                    
                    # iterates over nodes and retain the first with degree <= threshold
                    # Problem -> order of keys is the same most of the time, wich makes the algo
                    # computation time to find smallest node degree increase with time.
                    # Thus, we transform keys and values in lists, and use index of previously found
                    # smallest values as a starting point of iteration

                    for i, (k, v) in tqdm(enumerate(zip(list(V_tmp_top.keys())[j:], list(V_tmp_top.values())[j:])), leave=True):
                    #for i, (k, v) in enumerate(V_tmp_top.items()):
                        if len(v) <= 2*(min_degree):
                            #print(f'after {i} iterations - {min_degree}')
                            min_node_degree = k
                            j = i
                            break
                        elif len(v) > min_degree and len(v) <= min_degree_tmp:
                            min_degree_tmp = len(v)
                            min_node_degree = k
                    # randomly select a node
                    #min_node_degree = random.choice(list(V_tmp_top.keys()))
                            
                    #print(f'min deg: {min_node_degree}-{min_degree}-{V_tmp_top.get(min_node_degree)}')

                    # iterates overs nodes in dense subgraph and remove links from minimal degree node to other nodes
                    # -> needs to be optimized by storing a dense subgraph src: {dest}, in addition to already stored 
                    # dense subgraph dst: {src}
                    """for k, v in V_tmp_top.items():
                        if min_node_degree in v:
                            V_tmp_top[k].remove(min_node_degree)
                            m_tmp -= 1"""
                    
                    # We retrieve all destinations of node with minimal degree, using inversed dense subgraph history,
                    # then we iterate over this nodes in the dense subgraph history, in order to remove the remaining
                    # links with the minimal degree node.
                    neighbs = V_tmp_top_inv.get(min_node_degree, set())

                    if len(neighbs) > 0:
                        for neighb in tqdm(neighbs, leave=False):
                            if neighb in V_tmp_top:
                                V_tmp_top[neighb].discard(min_node_degree)
                                m_tmp -= 1
                        # remove node with minimal degree from inversed dense subgraph history
                        V_tmp_top_inv.pop(min_node_degree)

                    #V_res.remove(min_deg[0])
                    m_tmp -= len(V_tmp_top.get(min_node_degree))
                    V_tmp_top.pop(min_node_degree)


        # Write to result file
        with open(f'{outfile}_topk', 'a') as o:
            for k, v in V_tmp_top.items():
                for val in v:
                    o.write(f'{label2idx.get(k)},{label2idx.get(val)}\n') 
        
    graph.nb_edges = idx # True number of nodes in total graph
    graph.nb_nodes = len(label2idx)
    graph.in_degrees = in_deg
    graph.out_degrees = out_deg
    graph.label2idx = label2idx
    graph.idx2label = idx2label
    graph.files = listdir_fullpath(outdir)  
    graph.outdir = outdir

    return graph
                
"""
def topk_old(filename: str, metric: str = 'indegree'):
    ''' Topk algorithm to filter data in linear time. 
    
    Parameter
    ---------
        filename: str
            Name of data file.
        metric: str (default='indegree')
            Parameter used to filter data. '''

    graph = Bunch()

    label2idx, idx2label = {}, {}
    in_deg, out_deg = {}, {}
    idx = 0
    m = 0

    # Algo
    m_max = 20
    m_tmp = 0
    in_degrees_cache = defaultdict(list)
    top = set()
    min_node_deg = 1e9 
    min_node = 0
    min_deg = 0
    dense_subgraph = defaultdict(set)

    with open(filename) as f:
        
        delimiter = sniff_delimiter(filename)

        for line in tqdm(f):
            vals = line.strip('\n').split(delimiter)
            src, dst = vals[0], vals[1]
 
            if src != '' and src != 'Source':
                if src not in label2idx:
                    label2idx[src] = idx
                    idx2label[idx] = src
                    idx += 1
                if dst not in label2idx:
                    label2idx[dst] = idx
                    idx2label[idx] = dst
                    idx += 1

                out_deg[label2idx.get(src)] = out_deg.get(label2idx.get(src), 0) + 1
                in_deg[label2idx.get(dst)] = in_deg.get(label2idx.get(dst), 0) + 1
                
                # TODO: optimize initialization of every existing nodes
                if dst not in out_deg:
                    out_deg[label2idx.get(dst)] = 0
                if src not in in_deg:
                    in_deg[label2idx.get(src)] = 0
            
                m += 1

                # Algo
                print(f'\n{src}->{dst}')
                
                in_degrees_cache[dst] = in_degrees_cache.get(dst, []) + [src]
                #in_degrees_cache[src] = in_degrees_cache.get(src, [])
                
                #if len(in_degrees_cache[dst]) < min_node_deg:
                #    min_node = dst
                #    min_node_deg = len(in_degrees_cache[dst])
                intersect = set(in_degrees_cache.get(dst)).intersection(set(in_degrees_cache.keys()))
                if len(intersect) > 0:
                    top.update(intersect)
                    top.add(dst)
                    if len(intersect) > min_deg:
                        dense_subgraph[dst] |= set(in_degrees_cache.get(dst)).intersection(top)
                        m_tmp += 1
                        dense_subgraph[src] |= set(in_degrees_cache.get(src)).intersection(top)
                        m_tmp += len(list(set(in_degrees_cache.get(src)).intersection(top)))
                        
                        dic = sorted(dense_subgraph.items(), key=lambda x: len(x[1]))[0]
                        min_node = dic[0]
                        min_deg = len(dic[1])
                        
                
                if m_tmp >= m_max:

                    m_tmp -= len(dense_subgraph.get(min_node)) + 1 # + 1 refers to the removal of minimal node linked to detination node
                    print(f'POPPED OUT minimal node: {min_node} (with indegree={len(dense_subgraph.get(min_node))})')
                    #dense_subgraph.get(dst).remove(min_node) # remove minimal node linked to destination node
                    in_degrees_cache.pop(min_node)
                    dense_subgraph.pop(min_node)
                    top.remove(min_node)
                    for k, v in dense_subgraph.items():
                        if min_node in v:
                            v.remove(min_node)
                    dic = sorted(dense_subgraph.items(), key=lambda x: len(x[1]))[0]
                    min_node = dic[0]
                    min_deg = len(dic[1])

                
                
                print(f'intersect: {intersect}')
                #print(f'In-degrees cache: {in_degrees_cache}')
                print(f'Minimal node: ({min_node}, {min_deg})')
                print(f'Temp m {m_tmp}')
                print(f'TOP: {top}')
                print(f'Dense subgraph: {dense_subgraph}')
                #print(f'Total in degree: {np.sum([len(v) for v in top.values()])}\n')
                
    
    return 0
"""    

"""
m = 2
min_node_deg = 1e9 # initial value for minimum node degree in top dict
scnd_min_node_deg = 1e9 # initial value for second minimum node degree in top dict
in_degrees = {} # key=node, value=in_deg
top = defaultdict(list) #key=node, value=list of (source) nodes
in_degrees_top = defaultdict(set)
topk = set()

for idx, e in edges.iterrows():
    s, d = get_nodes(e)
    print(f'Node: {s}->{d}')
    
    in_degrees[d] = in_degrees.get(d, 0) + 1 # count node in-degrees
    
    # Keep track of all edges
    in_degrees_top[d] |= set({s})

    if in_degrees[d] >= m:
        topk.add(d)

        if s in topk:
            top[d] = top.get(d, []) + [s] # add destination node and its source
            print(f'HISTORIC: {in_degrees_top.get(s)}')
            top[s] = top.get(s, []) + list(in_degrees_top.get(s).intersection(topk))
            
            # Clear values in historic dict in order to reduce computation time at next iteration
            in_degrees_top[d].remove(s)
            in_degrees_top[s] -= topk

    print(f'In-degrees: {in_degrees}')
    print(f'In-degrees TOP: {in_degrees_top}')
    print(f'RESULT : {top}')
    print(f'TOPK: {topk}\n')

with open('output/topk/topk_subgraph_new.txt', 'w') as o:
    for k, v in top.items():
        for val in v:
            o.write(f'{val},{k}\n')
"""