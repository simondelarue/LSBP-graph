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
from LSBP.data import split_data, listdir_fullpath, count_degree
from LSBP.data import Cache
from LSBP.utils import Bunch
from LSBP.data import from_edge_list


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


    def _preprocess(self, filename: str, use_cache: bool = True):
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

            # Split data into chunks
            self.graph.outdir = split_data(self.filename)

            # Count neighbors (parallel computing)
            self.graph.files = listdir_fullpath(self.graph.outdir)
            
            with Pool() as p:
                tuples = p.map(count_degree, self.graph.files)

            self.graph.nb_edges = np.sum(x[2] for x in tuples)

            in_degrees = dict(functools.reduce(
                                operator.add, 
                                map(collections.Counter, [x[0] for x in tuples])
                                ))
            out_degrees = dict(functools.reduce(
                                operator.add, 
                                map(collections.Counter, [x[1] for x in tuples])
                                ))

            self.graph.nodes = np.array(list(set(in_degrees.keys()).union(set(out_degrees.keys()))))
            self.graph.nb_nodes = len(self.graph.nodes)

            # Reindex
            self.graph.label2idx = {}
            self.graph.idx2label = {}
            self.graph.in_degrees = {}
            self.graph.out_degrees = {}

            for idx, k in enumerate(self.graph.nodes):
                self.graph.label2idx[k] = idx
                self.graph.idx2label[idx] = k
                self.graph.in_degrees[idx] = in_degrees.get(k, 0)
                self.graph.out_degrees[idx] = out_degrees.get(k, 0)

            # Cache information
            self.cache.add(self.graph)
 
        else:
            # Load information from cache
            self.graph = self.cache.load()

    def fit(self, filename: str, use_cache: bool = True):
        ''' Fit algorithm to data.
        
        Parameters
        ----------
            filename: str
                Name of data file containing list of edges.
            use_cache: bool (default=True)
                If True, use cached data if it exists. '''

        # Preprocessing
        self._preprocess(filename=filename, use_cache=use_cache)

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

        # Adjacency matrices for each batch (COO matrices)
        iterable = list(zip([os.path.join(self.graph.outdir, f) for f in self.graph.files], repeat(self.graph.label2idx)))
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

        return self

    def _get_pagerank(self, adjacency: sparse.coo_matrix, diag: sparse.coo_matrix, damping_factor: float, 
                      init_scores: np.ndarray = None, v0= np.ndarray) -> np.ndarray:
        ''' PageRank solver '''

        W = (damping_factor * diag.dot(adjacency)).T.tocoo()

        scores_ = W.dot(init_scores) + v0
        scores_ /= scores_.sum()
        
        return scores_