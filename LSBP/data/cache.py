#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import os
import shutil
import numpy as np

from LSBP.data.load import get_project_root, json2dict, dict2json
from LSBP.utils import Bunch

class Cache:
    ''' Utility for cache. '''

    def __init__(self, directory: str = None, use_cache: bool = True):
        
        data_dir = os.path.basename(directory).split('.')[0]
        self.directory = os.path.join(get_project_root(), 'preproc_data', data_dir)
        self.is_empty = not(os.path.exists(self.directory))

        if not use_cache:
            self.clear()

        if not self.is_empty:
            print(f'Use cached data in {self.directory}')
        else:
            os.makedirs(self.directory)

    def add(self, graph: Bunch):
        ''' Cache graph information on disk.
        
        Parameters
        ----------
            graph: Bunch
                Graph object '''

        # Save information into cache
        dict2json(graph.in_degrees, os.path.join(self.directory, 'in_degrees.json'))
        dict2json(graph.out_degrees, os.path.join(self.directory, 'out_degrees.json'))
        np.save(os.path.join(self.directory, 'nodes.npy'), graph.nodes)
        dict2json(graph.label2idx, os.path.join(self.directory, 'label2idx.json'))
        dict2json(graph.idx2label, os.path.join(self.directory, 'idx2label.json'))

    def load(self) -> Bunch:
        ''' Load cached information into a Bunch object. 
        
        Output
        ------
            graph as a Bunch class. '''

        graph = Bunch()

        # Load information from cache
        graph.outdir = self.directory
        graph.label2idx = json2dict(os.path.join(self.directory, 'label2idx.json'))
        graph.idx2label = json2dict(os.path.join(self.directory, 'idx2label.json'))
        graph.in_degrees = json2dict(os.path.join(self.directory, 'in_degrees.json'), keys_int=True)
        graph.out_degrees = json2dict(os.path.join(self.directory, 'out_degrees.json'), keys_int=True)
        graph.nodes = np.load(os.path.join(self.directory, 'nodes.npy'))
        graph.nb_nodes = len(graph.nodes)
        graph.files = [f for f in os.listdir(self.directory) if os.path.basename(self.directory) in f]

        return graph

    def clear(self):
        ''' Clear cache by deleting directory. '''

        shutil.rmtree(self.directory)
        self.is_empty = True