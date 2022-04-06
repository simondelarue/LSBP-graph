#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import numpy as np
from scipy import sparse


def from_edge_list(filename: str, reindex: dict) -> sparse.coo_matrix:
    ''' Reads edge list and parse it to scipy COO format.
    
    Parameters
    ----------
        filename: str
            Name of data file containing list of edges separated by commas.
        reindex: dict
            Dictionary of reindexed entries, with labels as keys and indexes labels as values. 
            
    Output
    ------
        Adjacency matrix in sparse.COO format. '''
    
    n = len(reindex.keys())
    rows, cols = [], []
    
    with open(filename) as f:
        for line in f:
            vals = line.strip('\n').split(',')
            if vals[0] != '' and vals[0] != 'Source':
                rows.append(reindex.get(vals[0]))
                cols.append(reindex.get(vals[1]))
    data = np.ones(len(rows))

    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n))