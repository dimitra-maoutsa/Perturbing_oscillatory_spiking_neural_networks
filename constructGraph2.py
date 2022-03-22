# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:39:02 2015

@author: Dimi
"""


import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import *
 
#########################################################################

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False


#################### Construct Graph #############################



#n1 = 8
#n2 = 8
#distF = distance Function
#distM = distance meter
#k = 24
def construct_grid_with_k_connectivity(n1,n2,k,figu = False):
    """Constructs directed grid graph with side lengths n1 and n2 and neighborhood connectivity k"""
    """For plotting the adjacency matrix give fig = true"""
        
    
    def feuclidhorz(u , v):
        return np.sqrt((u[0] - (v[0]-n2))**2+(u[1] - v[1])**2)
        
    def feuclidvert(u , v):
        return np.sqrt((u[0] - (v[0]))**2+(u[1] - (v[1]-n1))**2 ) 
        
    def fperiodeuc(u , v):
        return np.sqrt((u[0] - (v[0]-n2))**2  + (u[1] - (v[1]-n1))**2 )  
        
    def finvperiodic(u,v):
        return fperiodeuc(v,u)
        
    def finvvert(u,v):
        return feuclidvert(v,u)
        
    def finvhorz(u,v):
        return feuclidhorz(v,u)
        
        
    def fperiodeucb(u , v):
        return np.sqrt((u[0]-n2 - (v[0]))**2  + (u[1] - (v[1]-n1))**2 ) 
    
    def fperiodeucc(v, u):
        return np.sqrt((u[0]-n2 - (v[0]))**2  + (u[1] - (v[1]-n1))**2 )   
        
        
        
    def fchhorz(u , v):
        return max(abs(u[0] - (v[0]-n2)), abs(u[1] - v[1]))
        
    def fchvert(u , v):
        return max(abs(u[0] - (v[0])),abs(u[1] - (v[1]-n1)) ) 
        
    def fperiodch(u , v):
        return max(abs(u[0] - (v[0]-n2))  , abs(u[1] - (v[1]-n1)) )  
        
    def finvperiodicch(u,v):
        return fperiodch(v,u)
        
    def finvvertch(u,v):
        return fchvert(v,u)
        
    def finvhorzch(u,v):
        return fchhorz(v,u)
        
        
    def fperiodchb(u , v):
        return max(abs(u[0]-n2 - (v[0])) , abs(u[1] - (v[1]-n1))) 
    
    def fperiodchc(v, u):
        return max(abs(u[0]-n2 - (v[0]))  , abs(u[1] - (v[1]-n1)) )
        
        
        
    def fperiodchd(u , v):
        return max(abs(n2-u[0] - (v[0])) , abs(u[1] - (n1-v[1]))) 
    
    def fperiodche(v, u):
        return max(abs(n2-u[0] - (v[0]))  , abs(u[1] - (n1-v[1])) )
    
    #distF = distance Function
    #distM = distance meter
    for case in switch(k):
        if case(4):
            distF = 'euclidean'
            distM = 1 #.41       
            break
        if case(8):
            distF = 'euclidean'
            distM = 1.5
            break
        if case(12):
            distF = 'euclidean'
            distM = 2
            break
        if case(20):
            distF = 'euclidean'
            distM = 2.3 #2.5
            break
        if case(24):  #check this again
            distF = 'chebyshev'
            distM = 2 #or euclidean 2.9
            break
        if case(36):
            distF = 'euclidean'
            distM = 3.5
            break
        if case(44):
            distF = 'euclidean'
            distM = 3.8
            break        
        if case(28):
            distF = 'euclidean'
            distM = 3
            break
        if case(48):
            distF = 'euclidean'
            distM = 4
            break
    
    x = np.linspace(1,n1,n1)
    y = np.linspace(1,n2,n2)
    X,Y = np.meshgrid(x,y)
    
    
    
    XY = np.vstack((Y.flatten(), X.flatten()))
    
    adj = squareform( (pdist(XY.T, metric = distF))  <= distM ) 
    
    if k!= 24:
    
        adjb = squareform( (pdist(XY.T, metric = feuclidhorz)) <= distM )    
        adjc = squareform( (pdist(XY.T, metric = feuclidvert)) <= distM )   
        
        adjd = squareform( (pdist(XY.T, metric = fperiodeuc)) <= distM )    
        adje = squareform( (pdist(XY.T, metric = finvperiodic)) <= distM )    
        adjf = squareform( (pdist(XY.T, metric = finvvert)) <= distM )
        adjg = squareform( (pdist(XY.T, metric = finvhorz)) <= distM )    
        adjx = squareform( (pdist(XY.T, metric = fperiodeucc)) <= distM )
        adjy = squareform( (pdist(XY.T, metric = fperiodeucb)) <= distM )
        
        Adj = ( adj + adjb +adjc+adjd+adje+adjf+adjg+adjx+adjy >=1)
    if k == 24:
        adjb = squareform( (pdist(XY.T, metric = fchhorz)) <= distM )
    
        adjc = squareform( (pdist(XY.T, metric = fchvert)) <= distM )
        
        
        adjd = squareform( (pdist(XY.T, metric = fperiodch)) <= distM )
        
        adje = squareform( (pdist(XY.T, metric = finvperiodicch)) <= distM )
        
        adjf = squareform( (pdist(XY.T, metric = finvvertch)) <= distM )
        adjg = squareform( (pdist(XY.T, metric = finvhorzch)) <= distM )
        
        adjx = squareform( (pdist(XY.T, metric = fperiodchb)) <= distM )
        adjy = squareform( (pdist(XY.T, metric = fperiodchc)) <= distM )
        
        Adj = ( adj + adjb +adjc+adjd+adje+adjf+adjg+adjx+adjy  >=1)
        

    
    
    
    #Adj = ( adj+adjb >=1 )
    #print adj
    #plt.plot(sum(Adj))
    if figu:
        plt.figure(figsize=(1000,1000))
        
        plt.imshow(Adj,interpolation = 'none', extent = [0,n1*n2 , n1*n2,0] )
        
        plt.xticks(np.arange(n1*n2))
        plt.yticks(np.arange(n1*n2))
        plt.grid(ls = 'solid')
        #plt.colorbar()
    """    #text portion
    min_val = 0
    max_val = n1*n2
    diff = 1
    ind_array = np.arange(min_val, max_val, diff)
    x, y = np.meshgrid(ind_array, ind_array)
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = adj[x_val,y_val]
        plt.text(x_val+0.5, y_val+0.5,  '%.2f' % c, fontsize=8,va='center', ha='center')
    """    
    G = nx.from_numpy_matrix(Adj)
    return (G,Adj)


#plt.savefig('euclidean5x5.png')
#plt.savefig('pylab-grid.pdf') 
