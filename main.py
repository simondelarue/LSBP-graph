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
    # What it does under the hood:
    # - preprocess, i.e divide data into chunks of edges
    # - compute PageRank on each chunk

    s = time.time()
    pr = PageRank(n_iter=5)
    scores = pr.fit_transform(filename, use_cache=True, method=method)
    topk_scores_idx = np.argsort(-scores)[:10]
    print(max(scores), min(scores))
    #pr.filter_transform(filename, method='indegree')
    
    print('-------------------------------')
    print(f'Completed in {(time.time()-s):.4f}s')
    print('-------------------------------')
    
    print(f'Method: {method}')
    print(f'Number of nodes: {pr.graph.nb_nodes}')
    #print(f'Number of edges: {pr.graph.nb_edges}')

    nodes_stream = []
    for idx in topk_scores_idx:
        print(f'Node: {pr.graph.idx2label.get(idx)} - {scores[idx]}')
        nodes_stream.append(pr.graph.idx2label.get(idx))

    print('-------------------------------')
    print('Sknetwork')
    start = time.time()
    filename = '/Users/simondelarue/Documents/PhD/Research/LSBP-graph/data/soc-LiveJournal1.txt'
    df = pd.read_csv(filename, delimiter='\t', names=['src', 'dst'], skiprows=1)
    edge_list = list(df.itertuples(index=False))
    graph = convert_edge_list(edge_list, directed=True)
    pr = skPageRank()
    scores_pr = pr.fit_transform(graph.adjacency)
    for s, n in zip(scores_pr[np.argsort(-scores_pr)[:10]], np.argsort(-scores_pr)[:10]):
        print(f'Node: {n} - score: {s}')

    sk_top10 = set(np.argsort(-scores_pr)[:10])
    stream_top10 = set([int(i) for i in nodes_stream])
    print(sk_top10)
    print(stream_top10)
    print('Common ratio: ', len(sk_top10.intersection(stream_top10))/10)

    # Compute score on top10
    ks = [10, 100, 1000]
    pred_rnks = np.argsort(-scores)
    for k in ks:
        true_top_idx = np.argsort(-scores_pr)[:k]
        true_rnks = np.asarray([np.arange(0, k)])
        true_top_preds_rnks = np.asarray([[np.where(pred_rnks == i) for i in true_top_idx]])
        NDCG = ndcg_score(true_rnks, true_top_preds_rnks)
        print(f'NDCG score for k={k}: {NDCG:.4f}')


    print(f'Completed in {(time.time()-start):.4f}s')
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