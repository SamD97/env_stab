#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' analysis-plotter for data.
    @author: Saumil Shah'''

import os
import gc
import sys
import tarfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = input(' Please enter path to folder: ')
if path == '': path = os.getcwd()
os.chdir(path)

if not os.path.isdir('plots'):
    os.mkdir('plots')
    
bins = np.linspace(0,2,201)
dx = (bins[:-1] + bins[1:])/2

with tarfile.open('data.tar.gz', 'r:gz') as tar:
    members = pd.Series(sorted(tar.getnames()))
    db = pd.read_csv('db','\t',dtype={'name':np.str})
    for s in range(db.shape[0]):
        dg = np.array([]).reshape((0,3))
        eps = members[members.str.contains(db.name[s])]
        
        fig1 = plt.figure(figsize=(12,10))
        fig2 = plt.figure(figsize=(12,10))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax2 = fig2.add_subplot(111, projection='3d')
        l = int(dg.shape[0]/10)
        for ep in eps:
            gc.collect()
            tar.extract(ep)
            df = np.lib.format.open_memmap(ep,'r',np.float16)
            dg = np.concatenate((dg,np.vstack((df[:,0].T,
                                    np.mean(df[:,1:db.ccap[s]+1],1).T,
                                    np.mean(df[:,db.ccap[s]+1:],1).T)).T))
            for k in range(l):
                gc.collect()
                dy1,_ = np.histogram(df[5*k,1:10001],bins=bins)
                dy2,_ = np.histogram(df[5*k,10001:],bins=bins)
                ax1.bar(dx,dy1/sum(dy1),5*k,
                        zdir='y',color='k',alpha=0.7,width=0.01)
                ax2.bar(dx,dy2/sum(dy2),5*k,
                        zdir='y',color='k',alpha=0.7,width=0.01)
            os.remove(ep)
            
        ax1.view_init(20,-70)
        ax2.view_init(20,-110)
        for ax in ax1,ax2:
            ax.sex_zlim((0,1))
            ax.sex_ylim((0,db.mgen[s]))
            ax.set_xlabel('Fitness')
            ax.set_zlabel('Frequency')
            ax.set_ylabel('Generations')
        fig1.savefig('plots/{}_hg0.svg'.format(db.name[s]))
        fig2.savefig('plots/{}_hg1.svg'.format(db.name[s]))
        fig1.close()
        fig2.close()
        
        x = np.arange(db.mgen[s])
        plt.figure()
        plt.ylim((0,2))
        plt.axhline(y=1,color='k',alpha=0.7,linestyle='--')
        plt.plot(x,dg[:,0]*.1 + 1,label='env')
        plt.plot(x,dg[:,1],label='fitness env0')
        plt.plot(x,dg[:,2],label='fitness env1')
        plt.plot(x,np.sqrt(dg[:,1]*dg[:,2]),label='gm fitness')
        plt.xlabel('generations')
        plt.ylabel('average fitness')
        plt.title('Population size = ${}*10^{}$'.format(str(db.ccap[s])[0],
                                                    len(str(db.ccap[s])[1:])))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/{}_avg.svg'.format(db.name[s]))
        plt.close()
        
        sys.stdout.write('\r[{}{}] {}% {}\t'.format('#'*int(((s+1)*50)/db.shape[0]),
                     '-'*int(((db.shape[0]-s-1)*50)/db.shape[0]), int(((s+1)*100)/db.shape[0]), db.name[s]))
        sys.stdout.flush()
        
sys.stdout.write('\n Done! \n')
sys.stdout.flush()