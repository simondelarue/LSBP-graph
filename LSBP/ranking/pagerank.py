#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

import numpy as np
from scipy import sparse
from scipy.sparse.coo import coo_matrix
from itertools import islice
import os
from multiprocessing import Pool
import collections, functools, operator

from LSBP.ranking.base import BaseRanking
from LSBP.data import get_project_root, split_data, listdir_fullpath, count_degree
from LSBP.utils import Bunch


class PageRank(BaseRanking):

    def __init__(self, filename: str, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 1e-6):
        super().__init__()
        self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter 
        self.tol = tol
        self.filename = filename
        self.cache = False
        self.outdir = self._preprocess()

        self.bipartite = None
        self.adjacency = None
        self.out_degrees = None


    def _preprocess(self):
        ''' Split input data into equally-sized batches of edges. '''

        self.graph = Bunch()

        # Split data into chunks
        if not self.cache:
            self.graph.outdir = split_data(self.filename)
            self.cache = True
        else:
            print('Use cached data')

        # Count neighbors (parallel computing)
        files = listdir_fullpath(self.graph.outdir)
        with Pool() as p:
            #in_deg_list, out_deg_list, nb_rows_list = p.map(count_degree, files)
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
        
        nodes = np.array(list(set(in_degrees.keys()).union(set(out_degrees.keys()))))

        # Reindex
        self.graph.label2idx = {}
        self.graph.idx2label = {}
        for idx, k in enumerate(nodes):
            self.graph.label2idx[k] = idx
            self.graph.idx2label[idx] = k
        
        self.graph.nb_nodes = len(nodes)


    def fit(self, init_scores: np.ndarray = None):

        # Format
        n_row, n_col = input_matrix.shape
        """if n_row != n_col:
            self.bipartite = True
            self.adjacency = bipartite2undirected(input_matrix)
        else:"""
        self.adjacency = input_matrix

        # Get seeds
        seeds_row = np.ones(n_row)
        if n_row != n_col:
            seeds_col = 0. * np.ones(n_col)
            seeds = np.hstack((seeds_row, seeds_col))
        else:
            seeds = seeds_row

        if seeds.sum() > 0:
            seeds /= seeds.sum()

        # Get pageRank
        self.scores_ = self._get_pagerank(self.adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                        tol=self.tol, solver=self.solver, init_scores=init_scores)

        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self

    def _get_pagerank(self, adjacency: sparse.coo_matrix, seeds: np.ndarray, damping_factor: float, n_iter: int, 
                      tol: float, solver: str = 'piteration', init_scores: np.ndarray = None, neighbs_cnt=None) -> np.ndarray:
        ''' PageRank solver '''

        n = adjacency.shape[0]
        
        if solver == 'piteration':

            out_degrees = adjacency.dot(np.ones(n)).astype(bool)
            norm = adjacency.dot(np.ones(adjacency.shape[1]))
            diag: sparse.coo_matrix = sparse.diags(norm, format='coo')
            diag.data = 1 / diag.data

            W = (damping_factor * diag.dot(adjacency)).T.tocoo()
            v0 = (np.ones(n) - damping_factor * out_degrees) * seeds
            
            if init_scores is not None:
                scores = init_scores * seeds
            else:
                scores = v0

            for i in range(n_iter):
                scores_ = W.dot(scores) + v0 * scores.sum()
                scores_ /= scores_.sum()
                if np.linalg.norm(scores - scores_, ord=1) < tol:
                    break
                else:
                    scores = scores_

            return scores / scores.sum()

        elif solver == 'normal':
            
            n = len(neighbs_cnt)
            
            if init_scores is not None:
                scores = init_scores
            else:
                scores = np.ones(n) / n

            #out_degrees = adjacency.dot(np.ones(n)).astype(bool)
            #out_degrees_inv = 1 / self.out_degrees
            #norm = adjacency.dot(np.ones(adjacency.shape[1]))
            #diag: sparse.coo_matrix = sparse.diags(norm, format='coo')
            #diag.data = 1 / diag.data
            #print(adjacency.row)
            #print(adjacency.col)
            
            #scores_ = np.zeros(n)
            min_pr = (1 - damping_factor) / n
            scores_ = np.zeros(n)

            for src, dst in zip(adjacency.row, adjacency.col):
                #print(src, dst)
                print(f'({src}, {dst}) - Source degree: {neighbs_cnt.get(src)} - source score: {scores[src]} - up: {damping_factor * scores[src] / neighbs_cnt.get(src)}')
                #scores_[dst] += damping_factor * scores[src] / neighbs_cnt.get(src)
                """try:
                    scores[src]
                except IndexError:
                    print(f'Source: {src} - Dest: {dst} - len scores: {len(scores)}')
                try:
                    damping_factor * scores[src] / neighbs_cnt.get(src) + min_pr
                except TypeError:
                    print(f'Source: {src} - Dest: {dst} - len scores: {len(scores)} - score {scores[src]} - neighb: {neighbs_cnt.get(src)}')
                """
                scores_[dst] += damping_factor * scores[src] / neighbs_cnt.get(src) #+ min_pr
            
            scores = scores_ + scores + min_pr
            #scores = scores_ + scores
            print('SCORES: ', scores)
            scores = scores / scores.sum() # normalization of scores between batches

            return scores

        elif solver == 'normal-mask':
            
            n = len(neighbs_cnt)
            
            if init_scores is not None:
                scores = init_scores
            else:
                scores = np.ones(n) / n
            
            min_pr = (1 - damping_factor) / n
            scores_ = np.zeros(n)

            # using numpy masks
            src_idxs = adjacency.row
            dst_idxs = adjacency.col
            out_degrees_src = np.array([neighbs_cnt.get(src) for src in src_idxs])
            scores_[dst_idxs] += damping_factor * scores[src_idxs] / out_degrees_src + min_pr

            #for src, dst in zip(adjacency.row, adjacency.col):
            #    scores_[dst] += damping_factor * scores[src] / neighbs_cnt.get(src) + min_pr
            
            scores = scores_ + scores
            scores = scores / scores.sum() # normalization of scores between batches

            return scores