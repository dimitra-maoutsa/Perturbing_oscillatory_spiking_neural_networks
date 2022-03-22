# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 23:28:05 2015

@author: Dimi
"""

import numpy as np
import matplotlib.pyplot as plt
#import networkx as nx
#from mpl_toolkits.mplot3d import Axes3D
#from copy import copy, deepcopy
#import time
#import matplotlib.animation as animation
#frame=1;
#from scipy.optimize import fsolve
#import scipy as sp
import random
#from matplotlib import animation
#plt.rcParams['animation.ffmpeg_path'] = 'C:\ffmpeg\bin\ffmpeg'
#import matplotlib.animation as animation
import pickle



################### Storing initial conditions ######################################
#################### For random networks question 2 ##############################
"""
N = 1024

sigma = 10**(-2)
delta_1024 = np.zeros((N,3))
#delta_400 = np.zeros((400,3))
for i in range(3):
    delta_1024[:,i] = sigma*(np.random.random(N)) #initial perturbation
    
    







var_dict3 = {"N": N, "delta_1024": delta_1024}
pickle.dump(var_dict3, open("InitialPerturb1024_x3.dat", "wb"))  

"""

###################### For 30x30 network #####################
"""
N = 30*30
sigma = 10**(-3)
delta_30_30 = sigma * (np.random.random(N))
var_dict3 = {"N": N, "delta_30_30": delta_30_30,sigma:"sigma"}
pickle.dump(var_dict3, open("InitialPerturb_30_30_2.dat", "wb"))


N = 60*60
sigma = 10**(-3)
delta_60_60 = sigma * (np.random.random(N))
var_dict3 = {"N": N, "delta_60_60": delta_60_60,sigma:"sigma"}
pickle.dump(var_dict3, open("InitialPerturb_60_60_2.dat", "wb"))
N = 40*40
sigma = 10**(-3)
delta_40_40 = sigma * (np.random.random(N))
var_dict3 = {"N": N, "delta_40_40": delta_40_40,sigma:"sigma"}
pickle.dump(var_dict3, open("InitialPerturb_40_40_2.dat", "wb"))
N = 50*50
sigma = 10**(-3)
delta_50_50 = sigma * (np.random.random(N))
var_dict3 = {"N": N, "delta_50_50": delta_50_50,sigma:"sigma"}
pickle.dump(var_dict3, open("InitialPerturb_50_50_2.dat", "wb"))
"""