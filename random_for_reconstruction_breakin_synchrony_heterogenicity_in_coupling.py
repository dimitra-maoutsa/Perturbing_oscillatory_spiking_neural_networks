# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 03:01:39 2015

@author: Dimi
"""



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from copy import copy, deepcopy
#import time
#import matplotlib.animation as animation
#frame=1;
#from scipy.optimize import fsolve
import scipy as sp
import random
#from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\ffmpeg\bin\ffmpeg'
#import matplotlib.animation as animation
#import pickle
#from scipy.stats import pearsonr
#from operator import itemgetter


############################## Rasterplot ######################################

def raster(event_times_list):
    """
    
    event_times_list : iterable
    a list of event time iterables
    color : string
    color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    color='k'
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
        plt.ylim(.5, len(event_times_list) + .5)
    return ax 

############################## LIF PARAMETERS ###############################################
# input driving current
global I
I = 1.1
Is = [1.1]#, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
global Tif
Tif = np.log(I/(I-1))




################################# LIF FUNCTIONS #############################################   


def Uif(x,I): # x is φ
    
    #return I*(1-np.exp(-(Tif)*x))
    #return 0.5* np.log( 1 + (np.exp(2)-1) * x     )
    return 4*(1 - np.exp( -0.275 * x   )     )

def Uinv(y,I):
    #return -np.log(1-(y/I))* (Tif**(-1))     #/Tif
    #return (np.exp(2*y) - 1)/ (np.exp(2)-1)
    return ( np.log(- y / 4  + 1)   ) / (- 0.275) 


def H(x,ee,I):
    #print Uif(x)
    #print Uif(x)++ee
    return Uinv(Uif(x,I)+ee,I)    
     
     
################################### NETWORK PARAMETERS ###################################
#total number of neurons     
N = 300
#proportion of excitation
pe = 0.1
N_E = int (np.ceil(pe*N))
N_I = N - N_E
############################### Creation of adjacency matrix #####################
#
pr = 0.2 # probability of an edge to be present
G = nx.erdos_renyi_graph(N, pr,seed=None, directed= True)
Adj = nx.adjacency_matrix(G)
########################## Create Pre and Post Arrays for convenience ###########################
kis = G.in_degree()  # indegrees of each neuron
Post = []
for i in range(N):
    Post.append(np.nonzero(Adj[i,:])[0])
    
######################### Create Array with the synaptic weights ########################
#epsiex = 1.5
epsiinh = -14

J = np.zeros((N,N))

for i in range(N):
    for j in xrange(N):
        J[i, j] = Adj[i,j] * epsiinh #*np.random.random()
    
######################### Pulse transmission delays ################################
# For now they are all the same - but in case that I want to have different values
# same transmission delay for all postsynaptic neurons
De = np.ones(N)*0.1 
     
######################## Simulate the network ###################################
#define infinite time
infinite = 9999999999


delta = (np.random.random(N)) #initial perturbation
#initial phases
phases = np.zeros(N)+delta
total_time = 0
#duration of simulation
final_time = 205

spiketimes = []
for i in range(N):
    spiketimes.append([])
    
#to timeleft is al list of times left for reception of a pulse emitted from the 
    #corrensponding entry/neuron of the list Neurotimeleft
timeleft = []
Neurontimeleft = []

######### functions used ################
#returns the minimum value and the indexes of its appearence     
def find_mins(list):
    m = min(list)
    return (m,[ind for ind,a in enumerate(list) if a==m] )    
     
     
def evolve_timeleft(timeleft,dt):
    #print timeleft
    return map(lambda a: a-dt , timeleft ) 
    
def delete_multiple(list_, args): # reverses the list so that the deletion doesnt have effect on the other deletions
    indexes = sorted(list(args), reverse=True)
    for index in indexes:
        del list_[index]
    return list_
    
def pop_multiple(list_, args): # reverses the list so that the deletion doesnt have effect on the other deletions
    indexes = sorted(list(args), reverse=True)
    popped = []
    for index in indexes:
         popped.append(list_.pop(index))
    return popped    #returns the reverse popped
###########################################################################
#import matplotlib.cm as cm
#colors = iter(cm.rainbow(np.linspace(0, 1,N)))
allspikes = (np.zeros(len(Is))).tolist()
for indexI, I in enumerate(Is):
    total_time = 0
    while (total_time < final_time):
    
        #>>>>>>>>>>>> calculate time for next reception <<<<<<<<<<<<<<<
        if timeleft: #if nonempty
            dt1, ind1 = find_mins(timeleft) # returns the tuple with the min 1st argument---> min timeleft
            #indx1 are/is the presyniptic neuron(s) from which the pulse that is ready to be received was emitted
        else:
            dt1, ind1 = infinite, infinite
        #print "Timeleft:"
        print total_time
        #print dt1
        #print ind1    
        #>>>>>>>>>> calculate time for next firing <<<<<<<<<<<<
        max_phase = np.amax(phases)
        ind2 = np.argwhere(phases == np.amax(phases))
        dt2 = 1 - max_phase     #time until next firing
        
        #>>>>>>>>>>>>>>>>>>> Track phases and time before updating them for plottong
        #plt.figure(1),plt.plot([total_time]*N,phases,color = colors)
        #>>>>>>>>>>>>>>> If the next event is a neuron firing <<<<<<<<<<<<
        if (dt2 < dt1):
            print "FIRE!!!"
            #evolve the time
            total_time = total_time + dt2
            #evolve phases 
            phases = phases + dt2
            #reduce times in timeleft
            timeleft = evolve_timeleft(timeleft, dt2)
            #reset neuron(s) that just fired 
            phases[ ind2 ] = 0        
            #add the timeleft for the spike reception and the indexes of the neurons that emitted the spike
            for i in range(len(ind2)):
                timeleft.append(De[ind2[i][0]])
                Neurontimeleft.append(ind2[i][0])
                #record the spike time for the neurons that emitted spike
                spiketimes[ind2[i]].append(total_time)
            
        #>>>>>>>>>>>>> If the next event is a spike reception <<<<<<<<<<        
        elif (dt1 <= dt2 ):
            print "...receiving..."
            #evolve the time
            total_time = total_time + dt1
            #advance the phases of all neurons for dt1
            phases = phases + dt1
            #remove corrensponding entries from timeleft and Neurontimeleft
            timeleft = delete_multiple(timeleft, ind1 )
            emitters = pop_multiple(Neurontimeleft, ind1) #indices of neurons that emitted the received spike(s)
            #reduce times in timeleft
            timeleft = evolve_timeleft(timeleft, dt1)
            #advance the faces of the neurons that are receiving the spike(s)
            for ll,val in enumerate(emitters):
                #phases[Post[val][:]] = H(phases[Post[val][:]],J[val, Post[val][:]])
                for lb,pns in enumerate(Post[val]):
                    if phases[pns] < 1:                
                        phases[pns] = H(phases[pns],J[val,pns],I)
                    
            #check whether a neuron has reached threshold
            indcrossed = np.argwhere(phases >=1)
            for la,i in enumerate(indcrossed):
                #reset phase            
                phases[i] = 0
                #add the delay to the timeleft and neuron ID to the Neurontimeleft
                timeleft.append(De[i][0])
                Neurontimeleft.append(i)
                #record spike time for these neuron
                spiketimes[i].append(total_time)
            
        else:
            print(dt1)
            print(dt2)
            break
         
    allspikes[indexI] = spiketimes
################ Plot Rasterplot #####################

spikes = []
for i in range(1,N):
    spikes.append(np.array(spiketimes[i]))




fig = plt.figure()
ax = raster(spikes)
plt.title('Raster plot')
plt.xlabel('time')
plt.ylabel('neuron')
fig.show() 

############### Compute ISIs #######################

ISI = map(np.diff , spikes)

#calculate CV



for i in xrange(len(ISI)):
    indmean = np.mean(ISI[i])
    indstd = np.std(ISI[i])
    print indstd / indmean

import pickle

################# SAVE ###########################
#var_dict3 = {"N": N, "I": I,"N_E":N_E, "N_I":N_I, "Jex": Jex, "Jin":Jin, "J":J, "De":De,"spiketimes":spiketimes,"ISI":ISI}
#pickle.dump(var_dict3, open("Simul_for_reco_N20.dat", "wb"))  


############### LOAD #################################
#d = pickle.load( open( "Simul_for_reco_N10.dat", "rb" ) )
#spiketimes = d["spiketimes"]
#ISI = d["ISI"]
#J = d["J"]
#De = d["De"]
for i in range(1,N):
    plt.figure(2),plt.hist(ISI[i], 70, histtype="stepfilled", alpha=.7)

spikes = (np.zeros(N)).tolist()
for ii in range(N):    
    spikes[ii]= np.array([i for i in spiketimes[ii] if i < 25]).flatten()

binsize = 0.001    
binedspikes = np.zeros((N,25/binsize))
binedtime = [ i*binsize for i in range(int(25/binsize)+1) ]
binedtime2 = [ -i*binsize for i in range(int(25/binsize)) ]
tim = []
tim.extend(binedtime2)
tim.extend(binedtime[1:-1])
for i in range(N):
    for j,val in enumerate(spikes[i]):
        pos = np.floor(val/binsize)
        #print pos
        binedspikes[i,pos] = 1

