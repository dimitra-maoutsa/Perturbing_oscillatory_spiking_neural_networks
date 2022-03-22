# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 04:46:28 2015

@author: Dimi
"""
####### BALANCED 


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
    Creates a raster plot
    Parameters
    ----------
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
Is = [1.2]#, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
global Tif
Tif = np.log(I/(I-1))




################################# LIF FUNCTIONS #############################################   


def Uif(x,I): # x is φ
    
    return I*(1-np.exp(-(Tif)*x))

def Uinv(y,I):
    print(1-(y/I))
    a = -np.log(1-(y/I))
    return  a * (Tif**(-1))     #/Tif


def H(x,ee,I):
    #print Uif(x,I)
    #print Uinv(Uif(x,I)+ee,I)
    return Uinv(Uif(x,I)+ee,I)    
    #return ee*(1-np.cos(x))*np.exp(3*np.cos(x-np.pi/3.)-1) 
     
################################### NETWORK PARAMETERS ###################################
#total number of neurons     
N = 30
#proportion of excitation
beta = 0.8
#number of excitatory units
N_E = int( np.floor(beta * N) )
#number of inhibitory neurons
N_I = N - N_E

# ratio of inhibition strength over excitation 
g = 4.5
# connection strength from excitation 
Jex = 6#0.05
#connection strength from inhibitory
Jin = - g* Jex

############################### Creation of adjacency matrix #####################
#
p = 0.1
#in degree From excitatory
kex1 = int(p * N_E)
#in degree from inhibitory
kin1 = int(p * N_I)
#adjacency matrix
Adj = np.zeros((N,N))   

for i in range(N):
    kex = kex1#random.sample(range(kex1-4,kex1+4),1)
    #print kex
    randoex2 =random.sample(range(0,N_E), kex) 
    Adj[randoex2,i] = 1
    #print i
    #print "ex"
    while Adj[i,i] == 1:
        Adj[i,i] = 0
        ra = random.sample(range(0,N_E),1)
        while (Adj[ra,i] == 1) or (ra == i):
            ra = random.sample(range(0,N_E),1)
        Adj[ra,i] = 1
    kin = kin1#random.sample(range(kin1-1,kin1+1),1)    
    randoex1 =random.sample(range(N_E,N), kin) 
    Adj[randoex1,i] = 1
    #print i
    #print "inh"
    while Adj[i,i] == 1:
        Adj[i,i] = 0
        ra = random.sample(range(N_E,N), 1) 
        while (Adj[ra,i] == 1) or (ra == i):
            ra = random.sample(range(N_E,N), 1) 
        Adj[ra,i] = 1


########################## Create Pre and Post Arrays for convenience ###########################

Post = []
for i in range(N):
    Post.append(np.nonzero(Adj[i,:])[0])
    
######################### Create Array with the synaptic weights ########################
J = np.zeros((N,N))
J[0:N_E, :] = Adj[0:N_E,:] * Jex      #### <<<<<<< edw xanw 1
J[N_E:,:] = Adj[N_E:,:] *Jin     
######################### Pulse transmission delays ################################
# For now they are all the same - but in case that I want to have different values
# same transmission delay for all postsynaptic neurons
De = np.ones(N)*0.5 
     
######################## Simulate the network ###################################
#define infinite time
infinite = 9999999999


delta = (np.random.uniform(0,0.9,N)) #initial perturbation
#initial phases
phases = np.zeros(N)+delta
total_time = 0
#duration of simulation
final_time = 2500

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
    return list(map(lambda a: a-dt , timeleft ) )
    
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
        print( total_time)
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
            print( "FIRE!!!")
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
                spiketimes[int(ind2[i])].append(total_time)
            
        #>>>>>>>>>>>>> If the next event is a spike reception <<<<<<<<<<        
        elif (dt1 <= dt2 ):
            print( "...receiving...")
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
                        print(',,,')
                        print(phases[pns])
                        phases[pns] = H(phases[pns],J[val,pns],I)
                        print(phases[pns])
                    
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
            print('...')
            print(dt1)
            print(dt2)
            break
         
    allspikes[indexI] = spiketimes
################ Plot Rasterplot #####################

spikes = []
for i in range(N):
    spikes.append(np.array(spiketimes[i]))




fig = plt.figure()
ax = raster(spikes)
plt.title('Raster plot')
plt.xlabel('time')
plt.ylabel('neuron')
fig.show() 

############### Compute ISIs #######################

ISI = list(map(np.diff , spikes))

#calculate CV



for i in range(len(ISI)):
    indmean = np.mean(ISI[i])
    indstd = np.std(ISI[i])
    #print indstd / indmean

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
for i in range(14):
    plt.figure(2),plt.hist(ISI[i], 70, histtype="stepfilled", alpha=.7)

spikes = (np.zeros(N)).tolist()
for ii in range(N):    
    spikes[ii]= np.array([i for i in spiketimes[ii] if i < 25]).flatten()
"""
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

"""
"""

################ TrainSet ###############
otherspikes = []
fromneuron = []
for i in range(N):
    otherspikes.append([])
    fromneuron.append([])
    
    
    

for i in range(1):
    #construct train set for neuron i
    pointers = np.zeros(N)
    for indsp in range(len(spikes[i])-1):
        starttime = spikes[i][indsp]
        endtime = spikes[i][indsp+1]
        temp = []
        tempneu = []
        #print "here"
        for j in range(N):
            if (j != i):
                flag = 0
                while (flag == 0) and((spikes[j][pointers[j]])< endtime) and ((spikes[j][pointers[j]])> starttime) :
                    temp.append(spikes[j][pointers[j]])
                    tempneu.append(j)
                    #print "there"
                    pointers[j] = pointers[j] + 1
                    #print pointers[j]
                    if pointers[j] == len(spikes[j]):
                        flag = 1
        otherspikes[i].extend([temp])
        fromneuron[i].extend([tempneu])
                
                
                
                
                
                
                
                
        
                
from mpl_toolkits.mplot3d import Axes3D                
                
                
                
                
            
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
    
    
    
 """   
"""    
#################### REC #################################
import bisect
gammai = 0.0316
def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError
    
def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect_right(a, x)
    if i:
        return i-1
    raise ValueError
    

for i in range(1): #N
    thita = np.zeros(len(Is),N)
    bhta = np.zeros(len(Is))
    #bhta = 
    for iiI, Ii in enumerate(Is):
        for ind,sptime in enumerate(allspikes[iiI][i][1:100]):
            spprin = sptime
            spmeta = spiketimes[i,ind+1]
            flags = np.zeros(N)
            jprinmeta = np.zeros((N,2))
            for j in range(N):#check an tairiazoun ta delays                
                if j!=i:
                    jtimel = find_le(allspikes[iiI][j])
                    jtimer = find_ge(allspikes[iiI][j])
                    if (jtimel != jtimer):                        
                        for innerj in range(jtimel,jtimer+1):
                            if allspikes[iiI][j][innerj]+De[j] == spmeta:
                                flags[j] ==1
                    elif jtimel == jtimer:                        
                        if allspikes[iiI][j][jtimel]+De[j] == spmeta:
                            flags[j] = 1
                    if flags[j] == 0:
                        jprinmeta[j][0] = jtimel #index ---> gia time allspikes[iiI][j][jtimel]
                        jprinmeta[j][1] = jtimer
            if sum(flags) == 0:
                #valid ISI
                for j in range(N):
                    if j!=i:
                        if (jprinmeta[j][0] != jprinmeta[j][1]):
                            for indj in range(jprinmeta[j][0],jprinmeta[j][1]+1)                            
                                thita[iiI,j] = thita[iiI,j] + np.exp(-gammai*(spprin-allspikes[iiI][j][indj]-De[j]))
                    
                        
                            
                    
        
"""    
   
    
    
    
    
    
    
