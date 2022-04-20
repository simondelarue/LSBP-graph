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
from scipy import sparse
import numpy as np
import array

from LSBP.utils import Bunch
from LSBP.data.load import sniff_delimiter, listdir_fullpath


def densest_subgraph(filename: str, outdir: str, e_max: int = 100000, d_min: int = 10, alpha: int = 0.5) -> Bunch():
    ''' Computes a one-pass densest subgraph approximation given a stream of edges.
    
    Parameters
    ----------
        filename: str
            Name of data file.
        outdir: str
            Complete path to output directory.
        e_max: int (default=100000)
            Maximum number of edges allowed in densest subgraph.
        d_min: int (default=10)
            Minimum in-degree bound for densest subgraph.
        alpha: int (default=0.5)
            Threshold applied to e_max. Above this threshold, densest subgraph is pruned according to nodes in-degrees.
    
    Output
    ------
        Densest subgraph approximation as a Bunch object. '''

    graph = Bunch()
    
    # Graph information
    label2idx, idx2label = {}, {}
    in_deg, out_deg = {}, {}
    n, m = 0, 0

    # Densest subgraph information
    size = e_max
    label2idx_densest, idx2label_densest = {}, {}
    n_densest = 0
    in_deg_densest = np.zeros(2*size) # initialize of densest subgraph in-degrees vector
    rows, cols, data = array.array('i'), array.array('i'), array.array('i') # use array.array instead of list for faster value appending

    with open(filename) as f:

        delimiter = sniff_delimiter(filename)

        for line in tqdm(f):
    
            vals = line.strip('\n').split(delimiter)
            src, dst = vals[0], vals[1]
            if src != '' and src != 'Source':

                m += 1

                # Reindex nodes
                if src not in label2idx:
                    label2idx[src] = n
                    idx2label[n] = src
                    n += 1
                if dst not in label2idx:
                    label2idx[dst] = n
                    idx2label[n] = dst
                    n += 1

                # In and out degrees
                out_deg[label2idx.get(src)] = out_deg.get(label2idx.get(src), 0) + 1
                in_deg[label2idx.get(dst)] = in_deg.get(label2idx.get(dst), 0) + 1
                
                if label2idx.get(dst) not in out_deg:
                    out_deg[label2idx.get(dst)] = 0
                if label2idx.get(src) not in in_deg:
                    in_deg[label2idx.get(src)] = 0

                # Densest subgraph approximation: nodes are added if their in-degree (in the main graph) is above a threshold
                if in_deg.get(label2idx.get(dst)) >= d_min and in_deg.get(label2idx.get(src)) >= d_min:
                    
                    # Reindex nodes in densest subgraph
                    if label2idx.get(src) not in label2idx_densest:
                        label2idx_densest[label2idx.get(src)] = n_densest
                        n_densest += 1
                    if label2idx.get(dst) not in label2idx_densest:
                        label2idx_densest[label2idx.get(dst)] = n_densest
                        n_densest += 1
                    
                    # Append values to sparse adjacency matrix
                    rows.append(label2idx_densest.get(label2idx.get(src)))
                    cols.append(label2idx_densest.get(label2idx.get(dst)))
                    data.append(1)

                    # Keep track of in-degrees in densest subgraph
                    in_deg_densest[label2idx_densest.get(label2idx.get(dst))] += 1
                    
                    # Pruning: if maximum number of edges in densest subgraph is reach, remove nodes with smallest in-degree. 
                    if len(rows) >= e_max:
                        
                        # Select nodes with highest in-degree, wrt the maximum number of edges allowed.
                        index = np.argsort(-in_deg_densest) 
                        mask = np.cumsum(in_deg_densest[index]) <= alpha * e_max 
                        selected_nodes_idx = index[mask]

                        # Filter sparse matrix
                        s = max(np.max(rows), np.max(cols)) + 1
                        A_coo = sparse.coo_matrix((data, (rows, cols)), shape=(s, s))
                        A_csr = A_coo.tocsr()[selected_nodes_idx][:, selected_nodes_idx] # convert to CSR for efficient slicing
                        A_coo = A_csr.tocoo()
                        rows, cols, data = array.array('i', A_coo.row), array.array('i', A_coo.col), array.array('i', A_coo.data)

                        in_deg_densest = in_deg_densest[selected_nodes_idx]
                        in_deg_densest = np.pad(in_deg_densest, (0, (2*size)-len(in_deg_densest)), 'constant', constant_values=(0, 0))
                        
                        # Update minimal in-degree required to enter subgraph.
                        d_min = max(d_min, np.min(in_deg_densest[in_deg_densest>0]))
                        
                        # Reindex densest subgraph nodes
                        index = np.array(list(label2idx_densest.keys()))[selected_nodes_idx]
                        label2idx_densest = {i: idx for idx, i in enumerate(index)}
                        n_densest = len(label2idx_densest)

    for k, v in list(label2idx_densest.items()):
        idx2label_densest[v] = k

    # Full graph information
    graph.nb_edges_tot = m
    graph.nb_nodes_tot = n
    graph.in_degrees = in_deg
    graph.out_degrees = out_deg
    graph.label2idx = label2idx
    graph.idx2label = idx2label

    # Densest subgraph information
    graph.nb_nodes = A_coo.shape[0]
    graph.nb_edges = A_coo.nnz
    graph.idx2label_densest = idx2label_densest
    graph.files = listdir_fullpath(outdir)  
    graph.outdir = outdir
    graph.adjacency = A_coo
    
    return graph


def topk_old_bis(filename: str, outdir: str, metric: str = 'indegree') -> Bunch():
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
    V_tmp_top_idx = {}
    res = defaultdict(set)
    res_idx = defaultdict(set)
    nb_edges_tot = 0
    cpt = 0
    res = {}
   
    
    if filename.endswith('soc-LiveJournal1.txt'):
        nb_rows = 68993774
    else:
        nb_rows = None

    outfile = os.path.join(outdir, f'{os.path.basename(outdir)}')

    with open(f'{outfile}_topk', 'a') as o:
        with open(filename) as f:

            delimiter = sniff_delimiter(filename)

            for line in tqdm(f, total=nb_rows):
                nb_edges_tot += 1
        
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
                    #print(f'\n{label2idx.get(src)}->{label2idx.get(dst)}')

                    #print(f'src label: {label2idx.get(src)} -dst label: {label2idx.get(dst)}')
                    #print(in_deg)

                    #print(f'In degree of source node: {in_deg.get(label2idx.get(src))}')
                    if in_deg.get(label2idx.get(src)) > 0:

                        V_tmp_top[dst] = V_tmp_top.get(dst, 0) + 1
                        V_tmp_top[src] = V_tmp_top.get(src, 1)

                        #V_tmp_top_idx[label2idx.get(dst)] = V_tmp_top_idx.get(label2idx.get(dst), 0) + 1
                        #V_tmp_top_idx[label2idx.get(src)] = V_tmp_top_idx.get(label2idx.get(src), 1)
                        m_tmp += 1

                        #if V_tmp_top.get(dst) >= 5 or V_tmp_top.get(src) >= 5:
                        if V_tmp_top.get(dst) >= 3 and V_tmp_top.get(src) >= 3:
                            #res[dst] |= {src}
                            #res_idx[label2idx.get(dst)] |= {label2idx.get(src)}
                            cpt += 1
                            o.write(f'{label2idx.get(src)},{label2idx.get(dst)}\n')
                            if res.get(src) is None:
                                res[src] = 1
                            if res.get(dst) is None:
                                res[dst] = 1
                        
                #print(f'Indeg: {in_deg}')
                #print(f'V_tmp_top_idx: {V_tmp_top_idx}')
                #print(f'res_idx: {res_idx}')

        #nodes = dict(sorted(V_tmp_top.items(), key=lambda x: x[1], reverse=True))
        #nodes = {k: v for k, v in V_tmp_top.items() if v > 10}
        #print(f'Nodes: {nodes}')
        
        #print(f'Number of nodes retained in dense subraph: {len(nodes)}')

    # Write to result file
    """cpt = 0
    with open(f'{outfile}_topk', 'a') as o:
        with open(filename) as f:
            delimiter = sniff_delimiter(filename)
            outfile = os.path.join(outdir, f'{os.path.basename(outdir)}')
            for line in tqdm(f, total=nb_rows):
                vals = line.strip('\n').split(delimiter)
                src, dst = vals[0], vals[1]
                if nodes.get(src) is not None and nodes.get(dst) is not None:
                    o.write(f'{label2idx.get(src)},{label2idx.get(dst)}\n')
                    cpt += 1"""
    """cpt = 0
    num_nodes = set()
    with open(f'{outfile}_topk', 'a') as o:
        for k, vals in res.items():
            if k not in num_nodes:
                num_nodes.add(k)
            for v in vals:
                o.write(f'{label2idx.get(k)},{label2idx.get(v)}\n')
                cpt += 1
                if v not in num_nodes:
                    num_nodes.add(v)"""
    print(f'Number of nodes retained in dense subgraph: {len(res)}/{idx}')
    print(f'Number of edges retained in dense subgraph: {cpt}/{nb_edges_tot}')
    #print(f'Number of nodes retained in dense subgraph: {len(num_nodes)}/{idx}')
        
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