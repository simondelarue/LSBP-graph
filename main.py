import os
import inspect
import numpy as np
import pathlib
import time
import pandas as pd
from sklearn.metrics import ndcg_score

from LSBP.ranking import PageRank

from sknetwork.data import convert_edge_list
from sknetwork.ranking import PageRank as skPageRank

if __name__=='__main__':
    
    DATA_DIR = 'data'

    #print(inspect.getfile(numpy))

    sid = pathlib.Path(__file__).parent
    print(os.path.expanduser(f'~/{sid}'))
    print(sid)

    #filename = f'{DATA_DIR}/soc-LiveJournal1_toy_bis.txt'
    #filename = f'{DATA_DIR}/painters.txt'
    filename = f'{DATA_DIR}/soc-LiveJournal1.txt'

    method = 'topk'
    # If 'split': What it does under the hood ?
    # - preprocess, i.e divide data into chunks of edges
    # - compute PageRank on each chunk

    s = time.time()
    pr = PageRank()
    scores = pr.fit_transform(filename, use_cache=True, method=method)
    topk_scores_idx = np.argsort(-scores)[:10]
    #print(max(scores), min(scores))
    #pr.filter_transform(filename, method='indegree')
    
    print('-------------------------------')
    print(f'Total computation time: {(time.time()-s):.4f}s')
    print('-------------------------------')
    
    print(f'Method: {method}')
    print(f'Number of nodes: {pr.graph.nb_nodes} - Number of edges: {pr.graph.nb_edges}')
    #print(f'Number of edges: {pr.graph.nb_edges}')

    nodes_stream = []
    for idx in topk_scores_idx:
        print(f'Node: {pr.graph.idx2label.get(pr.graph.idx2label_densest.get(idx))} - {scores[idx]}')
        nodes_stream.append(pr.graph.idx2label.get(pr.graph.idx2label_densest.get(idx)))

    #raise Exception('end here')

    print('\n-------------------------------')
    print('Sknetwork')
    print('-------------------------------')
    print('Preprocessing...')
    start = time.time()
    filename = '/Users/simondelarue/Documents/PhD/Research/LSBP-graph/data/soc-LiveJournal1.txt'
    df = pd.read_csv(filename, delimiter='\t', names=['src', 'dst'], skiprows=1)
    edge_list = list(df.itertuples(index=False))
    graph = convert_edge_list(edge_list, directed=True)
    print(f'Completed in {(time.time()-start):.4f}s')

    print('PageRank...')
    start = time.time()
    pr = skPageRank()
    scores_pr = pr.fit_transform(graph.adjacency)
    print(f'Completed in {(time.time()-start):.4f}s')
    print(f'Number of nodes: {graph.adjacency.shape[0]} - Number of edges: {graph.adjacency.nnz}')
    
    for s, n in zip(scores_pr[np.argsort(-scores_pr)[:10]], np.argsort(-scores_pr)[:10]):
        print(f'Node: {n} - score: {s}')

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
        sk_topk = set(np.argsort(-scores_pr)[:k])
        topk_scores_idx = np.argsort(-scores)[:k]
        nodes_stream = []
        for idx in topk_scores_idx:
            nodes_stream.append(pr.graph.idx2label.get(pr.graph.idx2label_densest.get(idx)))
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