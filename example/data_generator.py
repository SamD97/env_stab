#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' A working example of
    the implementation of
    the new file-structure.

    @author: Saumil Shah'''

import os
import numpy as np
import pandas as pd
import argparse as ap

# parameters as command-line args by user
parser = ap.ArgumentParser()
parser.add_argument('N', type=int, help='shape of the array')
args = parser.parse_args()
N = args.N

# inititate the file database
db_h = ['name','shape']
db = pd.DataFrame(columns=db_h)

# generate some data
name = np.random.randint(100000)
db.loc[len(db)] = [name, N]
df = np.random.randint(0,100,N)
np.save('{:0>7d}'.format(name),df)

# make an entry in file database
if not os.path.isfile('db'):
    db.to_csv('db', '\t', index=False)
else:
    db.to_csv('db', '\t', index=False, header=False, mode='a')
print('\n Generator Done!\n')