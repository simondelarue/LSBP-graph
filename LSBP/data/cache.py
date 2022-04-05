#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import os
import shutil

from LSBP.data.load import get_project_root

class Cache:
    ''' Utility for cache. '''

    def __init__(self, directory: str = None):
        
        data_dir = os.path.basename(directory).split('.')[0]
        self.directory = os.path.join(get_project_root(), 'preproc_data', data_dir)
        self.is_empty = not(os.path.exists(self.directory))

        if not self.is_empty:
            print(f'Use cached data in {self.directory}')
        else:
            os.makedirs(self.directory)

    def clear(self):
        ''' Clear cache by deleting directory. '''

        shutil.rmtree(self.directory)
        self.is_empty = True
