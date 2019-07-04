#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Wright-Fisher simulations for
    environmental stability with mutations.
    @author: Saumil Shah'''

import os
import gc
import time
import random
import numpy as np
import pandas as pd
import argparse as ap
from sys import stdout    

# taking simulation parameters as command-line args
prsr = ap.ArgumentParser()
prsr.add_argument('K', type=int, help='individuals to simulate')
prsr.add_argument('G', type=int, help='generations to simulate')
prsr.add_argument('R', type=int, help='replicates to simulate')

args = prsr.parse_args()
ccap = args.K
mgen = args.G
mrep = args.R

# number of dump events and dump stepsize
node = np.divide(ccap*mgen,1e7).clip(1)
step = int(mgen / node)

#rotation angle and operator
t = np.pi * (25/180)
R = np.array([[np.cos(t),-np.sin(t)],
               [np.sin(t),np.cos(t)]])

#dfe parameters
#antagonistic pleiotropy modifier
ap = np.array([-1,1])
mean = np.array([0.85,0.85])
cvar = np.array([[2,1],[1,1.5]])/75
dr = np.array([1,1])

def dump(g):
    dumpnow = 0
    temp = np.divmod(g,step)
    if not temp[1]: dumpnow = 1
    return dumpnow,temp[0]

def pick(size, n):
    return random.sample(range(size), min(n,size))

def plei(n):
    X = np.random.multivariate_normal(ap*mean, cvar, n)
    X = np.dot(R,(ap*X - dr).T) + dr.reshape(2,1)
    return X

if not os.path.isdir('data'):
    os.mkdir('data')

db_h = ['name','ccap','mgen','time']
db = pd.DataFrame(columns=db_h)

t1 = time.perf_counter()
df = np.zeros((1 + 2*ccap,step), np.float16)

for r in range(mrep):
    # init a replicate
    #literally a random number
    rpid = np.random.randint(mrep*100000)
    fn = 'data/{:0>7d}'.format(rpid)
    df = np.zeros((step,2*ccap), np.float16)
    db.loc[len(db)] = [fn[5:],ccap,mgen,time.ctime()]
    popi = np.ones((2,ccap), np.float16)
    
    for g in range(mgen):
        gc.collect()
        
        # reproduce
        noff = np.round(5*np.mean(popi,0)).astype(np.int8)
        popo = np.repeat(popi.flat, np.vstack((noff,noff)).flat)
        popo = popo.reshape(2,-1)
        
        # mutate
        nome = np.random.poisson(popo.shape[1]/1e3) #1e3 is mutation rate
        popo[:, pick(popo.shape[1],nome)] = plei(nome)
        
        # sample
        popi = popo[:, pick(popo.shape[1],ccap)]
        
        df[g%step]=popi.flat

        dumpstate = dump(g+1)
        if dumpstate[0]:
                np.save('{}_{:0>4d}'.format(fn,dumpstate[1]),df)
        
        pbar = int( (g+1)*100 / mgen )
        stdout.write('\r [{:-<50}] gen:{:d}% rep:{:d}%\t'.format('#'*int(pbar/2),
                     pbar, int((r)*100/mrep)))
        stdout.flush()

if not os.path.isfile('data/db'):
    db.to_csv('db', '\t', index=False)
else:
    db.to_csv('db', '\t', index=False, mode='a', header=False)

t2 = time.perf_counter()
stdout.write('\n Done! Time elapsed: {:.2f} hours \n'.format((t2-t1)/3600))
stdout.flush()