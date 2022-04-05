#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import os
import json
from typing import Tuple
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


def dict2json(data: dict, p: Path):
    ''' Save Python dictionary to Json. 
        
        Parameters
        ----------
            data: dict
                Python dictionary containing data.
            p: Path
                Complete path where data is saved. '''

    with open(p, 'w') as f:
        json.dump(data, f)

def json2dict(filename: Path) -> dict:
    ''' Load Json file to Python dictionary. 
        
        Parameters
        ----------
            filename: Path
                Complete path to Json file. 
                
        Output
        ------
            Python dictionary. '''

    with open(filename, 'r') as f:
        data = json.load(f, object_hook=lambda x: {int(k): v for k, v in x.items()})
    return data

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

    prefix = os.path.basename(filename).split('.')[0]
    max_nb_lines = 5 #TODO: infer number of line per batch
    outdir = os.path.join(get_project_root(), 'preproc_data', prefix)
            
    if os.system(f"split -l {max_nb_lines} {filename} {prefix}_") == 0:
        os.system(f'mv *{prefix}_* {outdir}')
        return outdir
    else:
        raise Exception(f'ERROR: No such file: {filename}')


def count_degree(filename: str) -> Tuple[dict, dict, int]:
    ''' Count degree of nodes in data file and returns result in a dictionary. In addition, also
        returns the number of lines, i.e edges in file.
        
        Parameter
        ---------
            filename: str
                Name of input datafile containing list of edges in the form of tuples u,v.
                
        Output
        ------
            Dictionary of degrees with elements as keys and number of occurences as values. '''

    in_deg = {}
    out_deg = {}
    nb_edges = 0

    with open(filename) as f:
        for line in f:
            vals = line.strip('\n').split(',')
            if vals[0] != '' and vals[0] != 'Source':
                src, dst = vals[0], vals[1]
                out_deg[src] = out_deg.get(src, 0) + 1
                in_deg[dst] = in_deg.get(dst, 0) + 1
                nb_edges += 1

    return (in_deg, out_deg, nb_edges)
    