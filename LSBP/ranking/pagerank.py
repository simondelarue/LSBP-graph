#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

import json
import numpy as np
from scipy import sparse
from scipy.sparse.coo import coo_matrix
from itertools import islice
import os
from multiprocessing import Pool
import collections, functools, operator
from pathlib import Path

from LSBP.ranking.base import BaseRanking
from LSBP.data import get_project_root, split_data, listdir_fullpath, count_degree, dict2json, json2dict
from LSBP.data import Cache
from LSBP.utils import Bunch


class PageRank(BaseRanking):

    def __init__(self, filename: str, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 1e-6):
        super().__init__()
        self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter 
        self.tol = tol
        self.filename = filename
        self.cache = None
        self.graph = Bunch()
        self._preprocess()

        self.bipartite = None
        self.adjacency = None
        self.out_degrees = None


    def _preprocess(self):
        ''' If cache is empty, preprocesses data in parallel. If preprocessed data is already cached,
            load information. '''

        self.cache = Cache(self.filename)
        
        if self.cache.is_empty:

            # Split data into chunks
            self.graph.outdir = split_data(self.filename)

            # Count neighbors (parallel computing)
            files = listdir_fullpath(self.graph.outdir)
            with Pool() as p:
                tuples = p.map(count_degree, files)

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
                self.graph.in_degrees[idx] = in_degrees.get(k)
                self.graph.out_degrees[idx] = out_degrees.get(k)

            # Cache information
            self.cache.add(self.graph)
 
        else:
            # Load information from cache
            self.graph = self.cache.load()

    def fit(self, init_scores: np.ndarray = None):

        # Format
        n_row, n_col = self.graph.nb_nodes, self.graph.nb_nodes
        #self.adjacency = input_matrix

        # Get seeds
        seeds_row = np.ones(n_row)
        if n_row != n_col:
            seeds_col = 0. * np.ones(n_col)
            seeds = np.hstack((seeds_row, seeds_col))
        else:
            seeds = seeds_row

        if seeds.sum() > 0:
            seeds /= seeds.sum()

        # Out degree matrix


        # Get pageRank
        #self.scores_ = self._get_pagerank(self.adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
        #                                tol=self.tol, solver=self.solver, init_scores=init_scores)
        self.scores_ = self.get_pagerank(self.graph.outdir)

        '''if self.bipartite:
            self._split_vars(input_matrix.shape)'''

        return self

    def _get_pagerank(self, filename: Path, seeds: np.ndarray, damping_factor: float, n_iter: int, 
                      tol: float, init_scores: np.ndarray = None, neighbs_cnt=None) -> np.ndarray:
        ''' PageRank solver '''

        