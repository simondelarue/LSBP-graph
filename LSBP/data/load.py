#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import os
from os import makedirs
from pathlib import Path

def get_project_root() -> Path:
    ''' Returns a Path to project root directory. 
    
        Output
        ------
            Path '''

    return Path(__file__).parent.parent.parent


def listdir_fullpath(directory: str) -> list:
    ''' Returns list of complete path for each files in directory.
        
        Parameters:
            directory: str
                Name of directory to list.
                
        Output
        ------
            List of Paths. '''
    return [os.path.join(directory, f) for f in os.listdir(directory)]


def split_data(filename: str, max_nb_lines: int = 0) -> bool:
    ''' Split data file into chunks of equal size. Data file is split in chunks along rows.
    
        Parameters
        ---------
            filename: str
                Filename (with complete path) of input data file. 
            max_nb_lines: int
                Maximum number of line in one chunk. 
                
        Output
        ------
            True when split is correctly performed. '''

    prefix = f"{filename.split('/')[1].split('.')[0]}"
    max_nb_lines = 5 #TODO: infer number of line per batch
    root = get_project_root()
    outdir = f'{root}/preproc_data/{prefix}'
            
    if os.system(f"split -l {max_nb_lines} {filename} {prefix}_") == 0:
        if not os.path.exists(outdir):
            makedirs(outdir)
        os.system(f'mv *{prefix}_* {outdir}')
        return outdir
    else:
        print(f'ERROR: No such file: {filename}')
    