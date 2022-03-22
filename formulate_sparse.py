# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 02:23:14 2015

@author: Dimi
"""
import numpy as np
from solve_l1 import solve_l1


def formulate_sparse(A,b):
    
    """
    A must be a M x N array
    
    b must be a M x 1 array
    
    Output:
    y must be a N x 1
    """
    
    
    m ,n  = A.shape
    if y.shape[0] != n:
        print " The shape of y must be %d x 1" %n
    elif y.shape[1] !=1:
        print " The shape of y must be %d x 1" %n
    elif b.shape[1] != 1:
        print " The shape of b must be %d x 1" %m
    elif b.shape[0] != m:
        print " The shape of b must be %d x 1" %m
    else:   # solve
        U ,S, V = np.linalg.svd(A)   #V is already V.T
        invA = np.linalg.pinv(A)
        r = sum(S < 10**(-8) )  # rank of A
        
        c = np.concatenate( [ np.zeros(r) , np.ones(m-r) ])
        
        result = solve_l1(invA , np.dot(V.T,c))   #because the V returned from svd is V.T
        
        
        return result
        