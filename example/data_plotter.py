#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' A working example of
    the implementation of
    the new file-structure.

    @author: Saumil Shah'''

import os
import tarfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if not os.path.isdir('plots'):
    os.mkdir('plots')

with tarfile.open('data.tar.gz', 'r:gz') as tar:
    members = pd.Series(sorted(tar.getnames()))
    db = pd.read_csv('db','\t',dtype={'name':np.str})
    for eID in range(db.shape[0]):
        entries = members[members.str.contains(db.name[eID])]
        for entry in entries:
            tar.extract(entry)
            df = np.lib.format.open_memmap(entry,'r')
            plt.figure()
            plt.plot(range(df.shape[0]),df)
            plt.tight_layout()            
            plt.savefig('plots/{}.png'.format(entry))
            plt.close()
            os.remove(entry)
print('\n Plotter Done!\n')