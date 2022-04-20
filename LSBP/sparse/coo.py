#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import array
import numpy as np
from scipy import sparse


class coo_matrix_increm():
    ''' Incremental sparse COO matrix using Python buffer protocol. '''

    def __init__(self, arg1=None):

        if isinstance(arg1, sparse.coo_matrix):
            self.row = array.array('i', arg1.row)
            self.col = array.array('i', arg1.col)
            self.data = array.array('i', arg1.data)

        elif arg1 is None:
            self.row = array.array('i')
            self.col = array.array('i')
            self.data = array.array('i')

    def append(self, i, j, v):
        ''' Fill COO matrix arrays, i.e given a sparse COO matrix A, sets A[i, j] = v. 
        
        Parameters
        ----------
            i: int
                Row indice of the matrix entry.
            j: int
                Column indice of the matrix entry.
            v: int
                Value of the matrix entry. '''

        self.row.append(i)
        self.col.append(j)
        self.data.append(v)

    def tocoo(self) -> sparse.coo_matrix:
        ''' Convert triplet of arrays to sparse.coo_matrix. 
        
        Output
        ------
            Scipy sparse matrix in COOrdinate format. '''

        row = np.frombuffer(self.row, dtype=np.int32)
        col = np.frombuffer(self.col, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=np.int32)

        size = max(np.max(row), np.max(col)) + 1

        return sparse.coo_matrix((data, (row, col)), shape=(size, size))

    def nnz(self) -> int:
        ''' Returns number of non-zero elements in COO matrix. 
        
        Output
        ------
            Number of non-zero elements in COO matrix. '''

        return len(self.row)