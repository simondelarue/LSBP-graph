import os
import inspect
import numpy as np
import pathlib
import time
import pandas as pd
from sklearn.metrics import ndcg_score
import spacy
import matplotlib.pyplot as plt

from LSBP.ranking import PageRank

from IPython.display import SVG, display
from sknetwork.data import load_netset, convert_edge_list, painters
from sknetwork.ranking import PageRank as skPageRank
from sknetwork.visualization import svg_graph

if __name__=='__main__':
    
    DATA_DIR = 'data'

    #print(inspect.getfile(numpy))

    sid = pathlib.Path(__file__).parent
    print(os.path.expanduser(f'~/{sid}'))
    print(sid)

    #filename = f'{DATA_DIR}/soc-LiveJournal1_toy_bis.txt'
    #filename = f'{DATA_DIR}/painters.txt'
    #filename = f'{DATA_DIR}/soc-LiveJournal1.txt'
    filename = f'{DATA_DIR}/wikivitals.txt'

    method = 'topk'
    # If 'split': What it does under the hood ?
    # - preprocess, i.e divide data into chunks of edges
    # - compute PageRank on each chunk

    s = time.time()
    pr = PageRank()
    scores = pr.fit_transform(filename, use_cache=False, method=method)
    topk_scores_idx = np.argsort(-scores)[:10]
    print(max(scores), min(scores))
    #pr.filter_transform(filename, method='indegree')
    
    print('-------------------------------')
    print(f'Total computation time: {(time.time()-s):.4f}s')
    print('-------------------------------')
    
    print(f'Method: {method}')
    print(f'Number of nodes: {pr.graph.nb_nodes} - Number of edges: {pr.graph.nb_edges}')
    #print(f'Number of edges: {pr.graph.nb_edges}')

    nodes_stream = []
    for idx in topk_scores_idx:
        node = pr.graph.idx2label.get(int(pr.graph.idx2label_densest.get(idx)))
        print(f'Node: {node} - {scores[idx]} - indeg: {pr.graph.in_degrees.get(pr.graph.label2idx.get(node))}')
        nodes_stream.append((node))

        #print(f'idx: {idx} - idx2label_densest: {pr.graph.idx2label_densest.get(idx)}')
        #print(f'idx: {idx} - label2idx: {pr.graph.label2idx.get(pr.graph.idx2label_densest.get(idx))}')
        #print(f'Node: {(pr.graph.idx2label_densest.get(idx))} - {scores[idx]}')
        #nodes_stream.append((pr.graph.idx2label_densest.get(idx)))

    # indexes
    #print(f'label2idx: {pr.graph.label2idx}')
    #print(f'idx2label: {pr.graph.idx2label}')
    

    print(f'Densest subgraph adj; {pr.graph.adjacency.shape}')
    image = svg_graph(pr.graph.adjacency.tocsr(), names=[pr.graph.idx2label.get(idx) for idx in np.arange(pr.graph.adjacency.shape[0])])
    #plt.imshow(SVG(image))
    #raise Exception('end')

    print('\n-------------------------------')
    print('Sknetwork')
    print('-------------------------------')
    print('Preprocessing...')
    start = time.time()
    if 'LiveJournal1' in filename:
        filename = '/Users/simondelarue/Documents/PhD/Research/LSBP-graph/data/soc-LiveJournal1.txt'
        df = pd.read_csv(filename, delimiter='\t', names=['src', 'dst'], skiprows=1)
        edge_list = list(df.itertuples(index=False))
        graph = convert_edge_list(edge_list, directed=True)
    elif 'wikivitals' in filename:
        graph = load_netset('wikivitals')
    elif 'painters' in filename:
        graph = painters(metadata=True)

    print(f'Completed in {(time.time()-start):.4f}s')
    print('PageRank...')
    start = time.time()
    sk_pr = skPageRank()
    scores_pr = sk_pr.fit_transform(graph.adjacency)
    print(f'Completed in {(time.time()-start):.4f}s')
    print(f'Number of nodes: {graph.adjacency.shape[0]} - Number of edges: {graph.adjacency.nnz}')
    
    indegs_sk = graph.adjacency.T.dot(np.ones(graph.adjacency.shape[0]))
    for s, n in zip(scores_pr[np.argsort(-scores_pr)[:10]], np.argsort(-scores_pr)[:10]):
        print(f'Node: {n} - score: {s} - indeg: {int(indegs_sk[n])}')

    sk_top10 = set(np.argsort(-scores_pr)[:10])
    stream_top10 = set([int(i) for i in nodes_stream])
    print(sk_top10)
    print(stream_top10)


    print('-------------------------------')
    print('Metrics')
    print('-------------------------------')
    print('Common ratio: ', len(sk_top10.intersection(stream_top10))/10)

    # Compute accuracy on top-k nodes
    ks = [10, 100, 1000]
    pred_rnks = np.argsort(-scores)
    
    for k in ks:

        # sknetwork
        sk_topk = set(np.argsort(-scores_pr)[:k])

        # stream
        topk_scores_idx = np.argsort(-scores)[:k]
        nodes_stream = []
        for idx in topk_scores_idx:
            node_i = pr.graph.idx2label.get(int(pr.graph.idx2label_densest.get(idx)))
            nodes_stream.append(node_i)
        stream_topk = set([int(i) for i in nodes_stream])
        print(f'Common ratio for {k} nodes: ', len(sk_topk.intersection(stream_topk))/k)

    # Compute NDCG on top-k nodes
    """ks = [10, 100, 1000]
    pred_rnks = np.argsort(-scores)
    for k in ks:
        true_top_idx = np.argsort(-scores_pr)[:k]
        true_rnks = np.asarray([np.arange(0, k)])
        true_top_preds_rnks = np.asarray([[np.where(pred_rnks == i) for i in true_top_idx]])
        NDCG = ndcg_score(true_rnks, true_top_preds_rnks)
        print(f'NDCG score for k={k}: {NDCG:.4f}')"""

    print('-------------------------------')

    """for idx, s in enumerate((scores)):
        print(f'Node: {pr.graph.idx2label.get(idx)} - {s}')
        if idx > 10:
            break"""
        

    """print('===============================')
    res_csr, res_coo = [], []
    for i in range(15):
        s = time.time()
        pr = PageRank(n_iter=10, method=0)
        scores = pr.fit_transform(filename, use_cache=False)
        e = time.time()
        res_csr.append(e-s)

        s_coo = time.time()
        pr_coo = PageRank(n_iter=10, method=1)
        scores = pr_coo.fit_transform(filename, use_cache=False)
        e_coo = time.time()
        res_coo.append(e_coo-s_coo)
    print(f'CSR Completed in {np.mean(res_csr):.4f}s')
    print(f'COO Completed in {np.mean(res_coo):.4f}s')
    print('===============================')"""

    
    raise Exception('END IN main.py')

    print()
    pr2 = PageRank()
    scores2 = pr2.fit_transform(filename)
    print(f'Number of nodes: {pr2.graph.nb_nodes}')
    #print(f'Number of edges: {pr2.graph.nb_edges}')
    
    scores = pr.fit_transform()

    
    print(pr)

    """class FileDataset(Dataset):
        '''Base class for datasets that are stored in a local file.
        Small datasets that are part of the river package inherit from this class.
        '''

        def __init__(self, filename, **desc):
            super().__init__(**desc)
            self.filename = filename

        @property
        def path(self):
            return pathlib.Path(__file__).parent.joinpath(self.filename)

        @property
        def _repr_content(self):
            content = super()._repr_content
            content["Path"] = str(self.path)
            return content"""