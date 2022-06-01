#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''


import numpy as np
from scipy import sparse
from multiprocessing import Pool
import collections, functools, operator
from itertools import repeat
import os
import time

from LSBP.ranking.base import BaseRanking
from LSBP.data import *
from LSBP.utils import Bunch

class PageRank(BaseRanking):

    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 1e-6):
        super().__init__()
        self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter 
        self.tol = tol
        self.filename = ''
        self.cache = None
        self.graph = Bunch()

        '''self.bipartite = None
        self.adjacency = None
        self.out_degrees = None'''


    def _preprocess(self, filename: str, use_cache: bool = True, method: str = 'split'):
        ''' If cache is empty, preprocesses data in parallel. If preprocessed data is already cached,
            load information. 
            
        Parameters
        ----------
            filename: str
                Name of data file containing list of edges.
            use_cache: bool (default=True)
                If True, use cached data if it exists. '''

        self.filename = filename
        self.cache = Cache(self.filename, use_cache=use_cache)
        
        if self.cache.is_empty:           
            print('========= PREPROCESSING ==========')
            start = time.time()
            prefix = os.path.basename(filename).split('.')[0]
            outdir = os.path.join(get_project_root(), 'preproc_data', prefix)

            # Split data and compute graph information
            if method == 'split':
                self.graph = split(self.filename, outdir)
            elif method == 'topk':
                self.graph = densest_subgraph(self.filename, outdir)

            # Cache information
            self.cache.add(self.graph)
            print(f'Elapsed time: {time.time()-start}')
 
        else:
            # Load information from cache
            print('========= LOADING ==========')
            start = time.time()
            self.graph = self.cache.load()
            print(f'Elapsed time: {time.time()-start}\n')


    def fit(self, filename: str, use_cache: bool = True, method: str = 'split'):
        ''' Fit algorithm to data.
        
        Parameters
        ----------
            filename: str
                Name of data file containing list of edges.
            use_cache: bool (default=True)
                If True, use cached data if it exists. '''

        # Preprocessing
        self._preprocess(filename=filename, use_cache=use_cache, method=method)

        n_row, n_col = self.graph.nb_nodes, self.graph.nb_nodes
        
        # Get seeds
        seeds_row = np.ones(n_row)
        if n_row != n_col:
            seeds_col = 0. * np.ones(n_col)
            seeds = np.hstack((seeds_row, seeds_col))
        else:
            seeds = seeds_row

        if seeds.sum() > 0:
            seeds /= seeds.sum()

        if method == 'split':
            # Adjacency matrices for each batch (COO matrices)
            #iterable = list(zip([os.path.join(self.graph.outdir, f) for f in self.graph.files], repeat(self.graph.label2idx)))
            iterable = list(zip([os.path.join(self.graph.outdir, f) for f in self.graph.files], repeat(self.graph.nb_nodes)))
            with Pool() as p:
                adjacencies = p.starmap(from_edge_list, iterable)

            # Inverse degree matrix (COO matrix)
            out_degrees = np.array(list(dict(sorted(self.graph.out_degrees.items(), key=lambda x: x[0], reverse=False)).values()))
            diag: sparse.coo_matrix = sparse.diags(out_degrees, format='coo')
            diag.data = 1 / diag.data

            v0 = (np.ones(n_col) - self.damping_factor) / n_col
            init_scores = np.ones(n_row) / n_row

            # Compute PageRank
            for i in range(self.n_iter):

                iterable = list(zip(adjacencies, repeat(diag), repeat(self.damping_factor), repeat(init_scores), repeat(v0)))
                with Pool() as p:
                    scores = p.starmap(self._get_pagerank, iterable)
                
                # Batch synchronization
                init_scores = np.sum(scores, axis=0)

            self.scores_ = init_scores

        elif method == 'topk':
            
            #f = os.path.join(self.graph.outdir, self.graph.files[0])
            #adjacency = from_edge_list(f, n_row)
            print('========= PageRank ==========')
            start = time.time()

            # get adjacency directly from topk filter
            adjacency = self.graph.adjacency

            # Inverse degree matrix (COO matrix)
            #out_degrees = np.array(list(dict(sorted(self.graph.out_degrees.items(), key=lambda x: x[0], reverse=False)).values()))
            out_degrees = adjacency.dot(np.ones(adjacency.shape[1]))
            diag: sparse.coo_matrix = sparse.diags(out_degrees, format='coo')
            diag.data = 1 / diag.data
            
            v0 = (np.ones(n_col) - self.damping_factor) / n_col
            init_scores = np.ones(n_row) / n_row

            scores = self._get_pagerank(adjacency, diag, self.damping_factor, init_scores, v0)
            self.scores_ = scores

            print(f'Elapsed time: {time.time()-start}\n')

        return self

    def _get_pagerank(self, adjacency: sparse.coo_matrix, diag: sparse.coo_matrix, damping_factor: float, 
                      init_scores: np.ndarray = None, v0= np.ndarray) -> np.ndarray:
        ''' PageRank solver '''

        W = (damping_factor * diag.dot(adjacency)).T.tocoo() # TODO: reduce time of this operation

        #scores_ = W.dot(init_scores) + v0
        #scores_ /= scores_.sum()

        scores = v0
        for i in range(1):
            scores_ = W.dot(scores) + v0
            scores_ /= scores_.sum()
            if np.linalg.norm(scores - scores_, ord=1) < 1e-6:
                break
            else:
                scores = scores_
        
        return scores_