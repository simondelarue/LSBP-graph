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
from tqdm import tqdm

from LSBP.utils import Bunch

def sniff_delimiter(filename: str) -> str:
    ''' Infer delimiter in data file by parsing its first row.
    
    Parameter
    ---------
        filename: str
            Name of data file.
    
    Output
    ------
        Delimiter as a string. '''

    with open(filename, 'r') as f:
        header = f.readline()
        if header.find(";") !=- 1:
            return ";"
        elif header.find(",") != -1:
            return ","
        elif header.find("\t") != -1:
            return "\t"

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

def json2dict(filename: Path, keys_int: bool = False) -> dict:
    ''' Load Json file to Python dictionary. 
        
        Parameters
        ----------
            filename: Path
                Complete path to Json file. 
            keys_int: bool
                If True, forces keys element to be loaded as integers.
                
        Output
        ------
            Python dictionary. '''

    with open(filename, 'r') as f:
        if keys_int:
            data = json.load(f, object_hook=lambda x: {int(k): v for k, v in x.items()})
        else:
            data = json.load(f)
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

def split(filename: str, outdir: Path, batch_size: int = 5000000) -> Bunch:
    ''' Split data file into chunks of equal size. Data file is split in chunks along rows.
    
    Parameters
    ---------
        filename: str
            Filename (with complete path) of input data file. 
        outdir: Path
            Path to output directory.
        batch_size: int
            Maximum number of line in one chunk. 
    
    Output
    ------
        Graph as a Bunch. '''

    graph = Bunch()

    label2idx, idx2label = {}, {}
    in_deg, out_deg = {}, {}
    idx = 0
    m, m_tmp = 0, 0
    num_batch = 0

    with open(filename) as f:
        
        delimiter = sniff_delimiter(filename)
        outfile = os.path.join(outdir, f'{os.path.basename(outdir)}')
        o = open(f'{outfile}_{num_batch}', 'a')

        for line in tqdm(f):
            vals = line.strip('\n').split(delimiter)
            src, dst = vals[0], vals[1]
               
            if src != '' and src != 'Source':
                if src not in label2idx:
                    label2idx[src] = idx
                    idx2label[idx] = src
                    idx += 1
                if dst not in label2idx:
                    label2idx[dst] = idx
                    idx2label[idx] = dst
                    idx += 1

                out_deg[label2idx.get(src)] = out_deg.get(label2idx.get(src), 0) + 1
                in_deg[label2idx.get(dst)] = in_deg.get(label2idx.get(dst), 0) + 1
                
                # TODO: optimize initialization of every existing nodes
                if dst not in out_deg:
                    out_deg[label2idx.get(dst)] = 0
                if src not in in_deg:
                    in_deg[label2idx.get(src)] = 0
            
                # Write to batch
                o.write(f'{label2idx.get(src)},{label2idx.get(dst)}\n')      
            
                m += 1
                m_tmp += 1
                
                # Change batch when batch_size is reached
                if m_tmp >= batch_size:
                    o.close()
                    num_batch += 1
                    m_tmp = 0
                    o = open(f'{outfile}_{num_batch}', 'a')
    
    graph.nb_edges = m
    graph.nb_nodes = len(label2idx)
    graph.in_degrees = in_deg
    graph.out_degrees = out_deg
    graph.label2idx = label2idx
    graph.idx2label = idx2label
    graph.files = listdir_fullpath(outdir)
    graph.outdir = outdir

    return graph


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
    