#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on April 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import os
from pathlib import Path

def check_empty_dir(dir: Path) -> bool:
    ''' Check if a directory is empty. 
    
    Parameter
    ---------
        dir: Path
            Path to directory.
            
    Output
    ------
        True if directory is empty. '''

    if os.path.exists(dir):
        return len(os.listdir(dir)) == 0
    else:
        return True