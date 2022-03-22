# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:54:27 2015

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
#plt.rcParams['animation.ffmpeg_path'] = 'C:\ffmpeg\bin\ffmpeg'
#import matplotlib.animation as animation
import pickle
from scipy.stats import pearsonr

import constructGraph2



######################### Create pos dictionary for drawing ####################################
def create_positions_dictionary(G,n1,n2):
    pos1 = {}
    xx = np.linspace(0,10*n2,num = n2)
    yy = np.linspace(0,10*n1,num = n1) #to be used later for the rectangular case
    x,y = np.meshgrid(xx,yy)
    for nod in G.nodes():
        pos1[nod] = [ x[nod[0],nod[1]], y[nod[0],nod[1]] ]
    return pos1
    
def draw_grid_graph(G,n1,n2):
    pos1 = create_positions_dictionary(G,n1,n2)
    # nodes
    nx.draw_networkx_nodes(G,pos1, nodelist=G.nodes(), node_color='r',
                       node_size=500, alpha=0.8)


    # edges  #TO DO: Draw curved edges
    nx.draw_networkx_edges(G,pos1,width=1.0,alpha=0.5)
    #nx.draw_networkx_edges(D,pos1,edgelist=G.edges(),width=8,alpha=0.5,edge_color='r')


    # some math labels
    labels={}
    labels[(0,0)]=r'$0,0$'
    labels[(1,1)]=r'$1,1$'
    #labels[(0,2)]=r'$0,2$'
    labels[(0,1)]=r'$0,1$'
    #labels[(5,1)]=r'$5,1$'
    """labels[3]=r'$d$'
    labels[4]=r'$\alpha$'
    labels[5]=r'$\beta$'
    labels[6]=r'$\gamma$'
    labels[7]=r'$\delta$'"""
    nx.draw_networkx_labels(G,pos1,labels,font_size=16)

    #plt.axis('off')
    #plt.savefig("labels_and_colors.png") # save as png
    #plt.show() # display

############################# CREATE ADDITIONAL CONNECTIVITY ########################################

def create_additional_connectivity(G,k,n1,n2):
    if k >=8:
        for n1x,n1y in G.nodes():
            p1 = ((n1x+1) % n1,(n1y+1)%n2)
            
            p2 = ((n1x-1) % n1,(n1y+1)%n2)
            p3 = ((n1x-1) % n1 , (n1y+1)%n2)
            p4 = ((n1x-1) % n1 , (n1y-1)%n2)
            G.add_edge(p1,(n1x,n1y))
            G.add_edge((n1x,n1y),p2)
            G.add_edge((n1x,n1y),p3)
            G.add_edge((n1x,n1y),p4)
    if k ==12:
        for n1x,n1y in G.nodes():
            p1 = ((n1x+2)%n1,n1y)            
            p2 = ((n1x-2)%n1,n1y)
            p3 = (n1x , (n1y+2)%n2)
            p4 = (n1x , (n1y-2)%n2)
            G.add_edge(p1,(n1x,n1y))
            G.add_edge((n1x,n1y),p2)
            G.add_edge((n1x,n1y),p3)
            G.add_edge((n1x,n1y),p4)
            
    #print(np.array((nx.adjacency_matrix(G,sorted(G.nodes()))).todense()))
        
        
    return G    

################################## GRAPH CONSTRUCTION ############################################

def create_directed_grid_graph(n1,n2,q,k):
    # q is rewiring probability
    # k: in degree / grid connectivity
    if k == 4:
        GD=nx.grid_2d_graph(n1,n2, periodic=True)
        #plt.figure(30),plt.imshow(nx.to_numpy_matrix(GD),interpolation='nearest')
        GD = GD.to_directed()
        #plt.figure(31),plt.imshow(nx.to_numpy_matrix(GD),interpolation='nearest')
    elif k > 4:
        
        GD,Adj = constructGraph2.construct_grid_with_k_connectivity(n1,n2,k)
        GD = GD.to_directed()
    #draw_grid_graph(G,n1,n2)    
    #G = nx.cycle_graph(10) 
    
    if (q>0):
        #Rewire starting point of each edge with prob q    
        nodes = GD.nodes()        
        for nod in nodes:
            for neighbor in GD.predecessors(nod):
                if random.random() < q:
                    new_neighb = random.choice(nodes)
                    #print new_neighb
                    # avoid self loops or dublicated edges
                    while (new_neighb == nod) or GD.has_edge(new_neighb,nod):
                        new_neighb = random.choice(nodes)
                    GD.remove_edge(neighbor,nod)
                    GD.add_edge(new_neighb,nod)        
          
       
    Pre = []
    Post = []
    N = n1*n2
    for i in range(N):
        Post.append([])
        Pre.append([])
    #print GD.nodes()
    if k==4:
        for i,nod in enumerate(sorted(GD.nodes())):
            
            Post[i] = map( lambda (a,b): n2*a+b , GD.successors(nod))
            #print Post[i]
            Pre[i] = map( lambda (a,b): n2*a+b , GD.predecessors(nod))
            #Post[i] = GD.successors(nod)
            #Pre[i] = GD.predecessors(nod)
    else:
        if q==0:
            AdjT = Adj.T
            for i in range(N):
                Post[i] = np.argwhere(Adj[i] == 1).flatten()
                Pre[i] = np.argwhere(AdjT[i] == 1).flatten()
        else:
            for i,nod in enumerate(sorted(GD.nodes())):
                Post[i] = GD.successors(nod)
                Pre[i] = GD.predecessors(nod)
            
    return (GD, Pre, Post)
    
    
"""

(2,0)-----(2,1)------(2,2)-----(2,3)
  |         |          |         |
(1,0)-----(1,1)------(1,2)-----(1,3)
  |         |          |         |
(0,0)-----(0,1) -----(0,2)-----(0,3)

               ||
               \/

8 -------9 ---------10 ------11     #This is the order of nodes in the Pre and Post Lists
|        |          |         |
4 -------5 -------- 6 ------- 7
|        |          |         |
0 -------1 -------- 2 --------3
"""        
    


################################# LIF FUNCTIONS #############################################
    


def Uif(x): # x is φ
    
    return I*(1-np.exp(-(Tif)*x))

def Uinv(y):
    return -np.log(1-(y/I))/Tif



def H(x,ee):
    return Uinv(Uif(x)+ee)    
     
    
    
def find_slope(data):
    xi = np.arange(0,len(data))
    Ar = np.array([xi , np.ones(len(data))])
    w = np.linalg.lstsq(Ar.T, data)[0]
    line = w[0]*xi+w[1]
    plt.plot(xi,line,'r-',xi,data,'o')    
    return -1/w[0]    


##################################### LINEARIZED SYSTEM ITERATION ###################################

def system_iteration(total_ee,Pre,anim):
    ####### System Iteration #######
    
    global L
    global Adj
    global k
    ims = []
    #iter_time = 0
    perturb = deepcopy(delta)
    perturb.shape = (N,1) #ensure thet the vector is a column
    perturb = perturb - np.min(perturb)
    perturb_history = [] # used for the calculation of the distance to synchrony
    perturb_history.append(perturb)
    A0 =  I*np.exp(-taf*Tif)/(-total_ee+I*np.exp(-taf*Tif)) #######-  
    A = np.zeros((N,N))
    np.fill_diagonal(A, A0)
   # B = ((A0)*L+Adj )*(k**(-1))
    
    for x in range(N):
        A[x][Pre[x]] = (1-A0)*(k**(-1))
    
    count = 0  
    
    #figure(2),imshow(B)
    #figure(3),imshow(A)
    
    while ((perturb-np.amin(perturb))>sigma*0.0001).any():
        if anim:
            ph = copy(perturb)
            ph.shape = (n1,-1)
            im = plt.imshow(ph, cmap = 'hot') #,interpolation='nearest')
            ims.append([im])
        perturb = np.float64(np.dot(A, perturb))
        perturb_history.append(perturb)
        count += 1
        #print(perturb)
        #print("<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>")
        print(count) # counts the # of periods
        
    ###### Comment this out if you dont want to calculate timescale / distance etc
    #perturb_history[0] has length count+1 --> 
    #delta_inf = perturb        #perturb_history[0][-1] #last perturbatiotion
    ##################################################################################
    #### if only timescale needed calculate delta_sharp for 1st and nth period only###
    ##################################################################################
    delta_inf = np.mean(perturb -np.min(perturb))     #perturb_history[0][-1] #last perturbatiotion    
    delta_n = map(lambda a: a - np.min(a) ,perturb_history)
    #delta_n = map(lambda b: map(lambda a: min(a-min(b),1-a),b),phases_history) # δ(n)
    delta_sharp = map(lambda a: a - delta_inf ,delta_n)   #δ' = δ(n) - δinf [all vectors]
    max_delta_0 = max(abs(delta_sharp[1])) #max initial perturbation
    max_delta_n = map(lambda b: max(abs(b))  , delta_sharp)
    synch_distance = max_delta_n / max_delta_0
    
        
    slope, intercept = np.polyfit(np.arange(1,len(synch_distance)+1), np.log(synch_distance), 1)
    timescale =  -1/slope
    #vals,vecs = np.linalg.eig(A)
    #idx = (vals.real).argsort()[::-1]   
    #svals = vals[idx]
    #timescaleeig = -1/np.log(abs(svals[1]))
    print "<<<<<<<>>>>>>>>>>"
    #print timescale
    return(timescale)
    
       
        
    
############################# PLOT SURFACE OF VECTOR ############################################    

def plot_surface_of(vec):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x  = np.linspace(0, n1, n2)
    y  = np.linspace(0, n2, n1)
    X, Y = np.meshgrid(x, y)
    nvec = deepcopy(vec)
    nvec.shape = (n1,-1)
    Z = nvec

    ax.plot_surface(X, Y, Z,cmap= 'YlOrRd')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()    
######### functions used ################
    

    
#returns the minimum value and the indexes of its appearence     
def find_mins(list):
    m = min(list)
    return (m,[ind for ind,a in enumerate(list) if a==m] )    
     
     
def evolve_timeleft(timeleft,dt):
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
################################ NETWORK SIMULATION ##########################################
def simulate_network(total_ee,Pre):

    phases_history = []
    perturb_history=[]
    infinite = 9999
    ims = []
    phases = 0.5*np.ones(N)+delta
    #print phases
    initial_phases = phases
    perturb_history.append(phases - np.min(phases))
    ph = copy(phases)
    ph.shape = (n1,-1)
    sim_time = np.zeros(total_ees.size)
    for ii,total_ee in enumerate(total_ees):
        total_time = 0
        phases = initial_phases
        ee = total_ee / k # εij  
        print(ii)
        timeleft = []  #time left until spike of i is received from Post(i)
        Neurontimeleft = [] # neuron from wich the spike was emitted
        
        s=0  #counter for the periods        
        countspikes = 0
        
        while (abs(phases-[phases[0]]*N)>sigma*0.0001).any():       
        #for metrw in range(25000):      
            #print("Timeleft:")
            #print timeleft
            #>>>>>>>>>>>>>>>>>>>> Calculate next reception <<<<<<<<<<<<<<<<<<<<<<<<<
            if timeleft: #if nonempty
                dt1, ind1 = find_mins(timeleft) # returns the tuple with the min 1st argument---> min timeleft
                #indx1 are/is the presyniptic neuron(s) from which the pulse that is ready to be received was emitted
            else:
                dt1, ind1 = infinite, infinite
            #print dt1
            #print ind1
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Calculate next firing <<<<<<<<<<<<<<<<
            max_phase = np.amax(phases)
            ind2 = np.argwhere(phases == np.amax(phases))
            dt2 = 1 - max_phase     #time until next firing
                 
                
             #>>>>>>>>>>>>>>> If the next event is a neuron firing <<<<<<<<<<<<
            if (dt2 < dt1):
                #print "FIRE!!!"
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
                    timeleft.append(taf)
                    Neurontimeleft.append(ind2[i][0])
                    #record the spike time for the neurons that emitted spike
                    #spiketimes[ind2[i]].append(total_time)
                
                
            #>>>>>>>>>>>>> If the next event is a spike reception <<<<<<<<<<        
            elif (dt1 <= dt2 ):
                #print "...receiving..."
                #evolve the time
                total_time = total_time + dt1
                #advance the phases of all neurons for dt1
                phases = phases + dt1
                #remove corrensponding entries from timeleft and Neurontimeleft
                timeleft = delete_multiple(timeleft, ind1 )
                emitters = pop_multiple(Neurontimeleft, ind1) #indices of neurons that emitted the received spike(s)
                countspikes = countspikes + len(emitters)
                #reduce times in timeleft
                timeleft = evolve_timeleft(timeleft, dt1)
                #advance the faces of the neurons that are receiving the spike(s)
                for ll,val in enumerate(emitters):
                    phases[Post[val][:]] = H(phases[Post[val][:]],ee)
                    #for lb,pns in enumerate(Post[val]):
                                   
                     #   phases[pns] = H(phases[pns],ee)
                        
                #check whether a neuron has reached threshold
                indcrossed = np.argwhere(phases >=1)
                for la,i in enumerate(indcrossed):
                    #reset phase            
                    phases[i] = 0
                    #add the delay to the timeleft and neuron ID to the Neurontimeleft
                    timeleft.append(taf)
                    Neurontimeleft.append(i)
                    #record spike time for these neuron
                    #spiketimes[i].append(total_time)
                
            else:
                print(dt1)
                print(dt2)
                break
             
                        
                                     
            if (countspikes == N): #print(total_time)
            #if (phases[0]==0):
            #    ax.plot([s]*N,phases,'r.',markersize=0.9)
                #ph = copy(phases)
                #ph.shape = (n1,-1)
                #im = plt.imshow(ph)
                #ims.append([im])
                s += 1
                countspikes = 0
                
                pert = phases-np.min(phases)
                pert = pert - np.min(pert)
                perturb_history.append(pert)
                #print(pert)
                print(s)
             
              
        sim_time[ii] = s
        delta_sinf = np.mean(pert)  
        delta_n = perturb_history# δ(n)
        #delta_n = map(lambda b: map(lambda a: min(a-min(b),1-a),b),phases_history) # δ(n)
        delta_sharp = map(lambda a: a - delta_sinf ,delta_n)   #δ' = δ(n) - δinf [all vectors]
        max_delta_0 = max(abs(delta_sharp[1])) #max initial perturbation
        max_delta_n = map(lambda b: max(abs(b))  , delta_sharp)
        synch_distance = max_delta_n / max_delta_0        
        
        slope, intercept = np.polyfit(np.arange(1,len(synch_distance)+1), np.log(synch_distance), 1)
        timescales =  -1/slope
        
        return (timescales)        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def calculate_clustering_coef(Adj,Ni,G):
    coef = 0
    AA3 = np.linalg.matrix_power((Adj + Adj.T),3)
    A2 = np.linalg.matrix_power(Adj,2)
    nod = G.nodes()
    #print AA3
    for i,nd in enumerate(nod):
        #print i
        kout = G.out_degree(nd)
        
        kin = G.in_degree(nd)
        ktot = kin+kout
        coef = coef + AA3[i,i] /(2*(ktot*(ktot-1)-2*A2[i,i]))
    #print coef
    return coef/Ni
    
    
    
    
#########################################################################################################
    

####### GRAPH CONSTRUCTION
#G = nx.cycle_graph(6)  
#GG=nx.cartesian_product(G,G)    
    
#m = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #set of exponents for the rewiring prob 2^(-m)
#qs =map(lambda a: 2**(-a),m)#rewiring pobability
#qs.append(0)
global n1
#n1 = 60 #dimensions of the grid
#n2 = 60
global N
#N = n1*n2 #total # of nodes
global k
#ks = [4,8,12,20,24]
##### Parameters of LIF

global I
I = 1.1
global Tif
Tif = np.log(I/(I-1))

global taf 
taf = 0.05 # τ delay 
global sigma
sigma = 10**(-3)
global delta   
#d= pickle.load( open( "InitialPerturb_60_60_2.dat", "rb" ) )
#delta = d["delta_60_60"]

global total_ees


total_ees = -np.array([0.8]) 

ks = [4,8,12,20,24,28,36,44,48]
m = [0,1,2,4,8,16]
qs = map(lambda a: 2**(-a),m)#rewiring pobability
qs.append(0)
ns = [30,35,40,45,50,55,60]


size_k = len(ks)
size_q = len(qs)
size_n = len(ns)
iter_time1 = np.zeros((size_k,size_q,size_n))
links = np.zeros((size_k,size_q,size_n))
paths = np.zeros((size_k,size_q,size_n))
radia = np.zeros((size_k,size_q,size_n))
diameters = np.zeros((size_k,size_q,size_n))
ass = np.zeros((size_k,size_q,size_n))
ccf = np.zeros((size_k,size_q,size_n))
iter_time2 = np.zeros((size_k,size_q,size_n))
sim_time2 = np.zeros((size_k,size_q,size_n))
sim_time1 = np.zeros((size_k,size_q,size_n))
for kkk,k in enumerate(ks):   
    for qqq,q in enumerate(qs):
        for nnn,n in enumerate(ns):
            n1 = n
            n2 = n
            N = n1* n2
            G, Pre, Post = create_directed_grid_graph(n1,n2, q,k) #q: prob. of rewiring an edge, k: in degree/ connectivity of the grid
            Adj =  nx.adjacency_matrix(G).toarray().astype(int)           
            delta  = sigma*(np.random.random(N)) 
            iter_time1[kkk,qqq,nnn] = system_iteration(total_ees[0],Pre, anim= False)
            sim_time1[kkk,qqq,nnn] = simulate_network(total_ees[0],Pre)
        
            delta  = sigma*(np.random.random(N)) 
            iter_time2[kkk,qqq,nnn] = system_iteration(total_ees[0],Pre, anim= False)
            sim_time2[kkk,qqq,nnn] = simulate_network(total_ees[0],Pre)
            links[kkk,qqq,nnn] = N*k
            try:
                diameters[kkk,qqq,nnn]=nx.diameter(G)
                radia[kkk,qqq,nnn] = nx.radius(G)
            except nx.exception.NetworkXError:
                diameters[kkk,qqq,nnn]=999999
                radia[kkk,qqq,nnn] = 999999                
            ass[kkk,qqq,nnn] = nx.degree_assortativity_coefficient(G,'out','out')
            ccf[kkk,qqq,nnn] = calculate_clustering_coef(Adj,N,G)
            paths[kkk,qqq,nnn] = nx.average_shortest_path_length(G)
            
            

#nx.degree_assortativity_coefficient(G,'out','out')
        
var_dict3 = {"iter_time1":iter_time1,"iter_time2":iter_time2,"sim_time1":sim_time1,"sim_time2":sim_time2,"ccf":ccf,"ass":ass,"links":links, "radia":radia, "diameters":diameters,"paths":paths,"ks":ks,"qs":qs,"ns":ns ,"kkk_qqq_nnn":0}
#pickle.dump(var_dict3, open("LinksG.dat", "wb"))  







d = pickle.load( open( "Links3.dat", "rb" ) )
iter_time2 = d["iter_time2"]
iter_time1 = d["iter_time1"]
d = pickle.load( open( "Sim30_30_1.dat", "rb" ) )
paths2 = d["paths"]
iter_time3 = d["iter_time"]
#iter_time4 = d["iter_time1"]
cc2 = d["ccf"]



"""

ites= iter_time.flatten()
pathsflat = paths.flatten()
diamflat = diameters.flatten()
radiaflat = radia.flatten()
linksflat = links.flatten()


####### Imshow time gia  k vs N gia different qs #######

#des untitled6


######### Plot time vs N gia varius connectivities k

plt.rcParams.update({'legend.labelspacing':0.25,'legend.fontsize': 12,'legend.linewidth': 5})


ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
for i in range(len(ks)):
    
    plt.plot(iter_time[i,-1,:],'k.')
    plt.plot(iter_time[i,-1,:], label = '$k = %d$' %ks[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('q = 0')
plt.xlabel('n1xn2')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')

for i in range(len(ks)):
    
    plt.figure(2),plt.plot(diameters[i,-1,:],'k.')
    plt.figure(2),plt.plot(diameters[i,-1,:], label = '$k = %d$' %ks[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('q = 0')
plt.xlabel('n1xn2')
plt.ylabel(r'diameter')

for i in range(len(ks)):
    
    plt.figure(2),plt.plot(radia[i,-1,:],'k.')
    plt.figure(2),plt.plot(radia[i,-1,:], label = '$k = %d$' %ks[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('q = 0')
plt.xlabel('n1xn2')
plt.ylabel(r'radius')

###################################Synch vs N gia ola ta q gia kapoio k

ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
for i in range(len(qs)):
    
    plt.plot(iter_time[2,i,:],'k.')
    plt.plot(iter_time[2,i,:], label = '$q = %f$' %qs[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('k = 12')
plt.xlabel('n1xn2')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')


ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
for i in range(len(qs)):
    
    plt.figure(2),plt.plot(diameters[2,i,:],'k.')
    plt.figure(2),plt.plot(diameters[2,i,:], label = '$q = %f$' %qs[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('k = 12')
plt.xlabel('n1xn2')
plt.ylabel(r'diameter')

for i in range(len(ks)):
    
    plt.figure(3),plt.plot(radia[2,i,:],'k.')
    plt.figure(3),plt.plot(radia[2,i,:], label = '$q = %f$' %qs[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('k = 20')
plt.xlabel('n1xn2')
plt.ylabel(r'radius')



###################################Synch vs k gia ola ta q gia kapoio N

ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
for i in range(len(qs)):
    
    plt.plot(iter_time[:,i,2],'k.')
    plt.plot(iter_time[:,i,2], label = '$q = %f$' %qs[i])
plt.xticks([0,1,2,3,4,5,6,7,8], ['4','8','12','20','24','28','36','44','48'])
plt.legend(numpoints = 1,loc=4)    
plt.title('n = 40')
plt.xlabel('in-degree k')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')


ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
for i in range(len(qs)):
    
    plt.figure(2),plt.plot(diameters[1:-1,i,2],'k.')
    plt.figure(2),plt.plot(diameters[1:-1,i,2], label = '$q = %f$' %qs[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('n = 40')
plt.xlabel('k')
plt.ylabel(r'diameter')





for i in range(len(qs)):
    
    plt.figure(3),plt.plot((paths[2,i,:]),'k.')
    plt.figure(3),plt.plot((paths[2,i,:]), label = '$q = %f$' %qs[i])

plt.legend(numpoints = 1,loc=4)    
plt.title('n = 40')
plt.xlabel('k')
plt.ylabel(r'average shortest path length')


for i in range(len(qs)):
    plt.figure(4),plt.plot((paths[:,i,1]),(iter_time[:,i,1]),'.')
    #plt.figure(4),plt.plot(qs[i],iter_time[-3,i,0]/(paths[-3,i,-1]),'.')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')
plt.xlabel(r'average shortest path length')
plt.title('k = 48')


pathlastq = paths[:,-3:-1,:]
pathlastq = pathlastq.flatten()
timelastq = iter_time[:,-3:-1,:]
timelastq = timelastq.flatten()
##########!!!!!!!!!!!!!!!!!!!!!!!!!!!! t / length vs q gia ola ta N gia kapoio k

for i in range(len(ns)):
    
    plt.figure(1),plt.plot((iter_time[-1,:,i]/(paths[-1,i,i])),'-')
plt.ylabel(r'synchr. timescale $\tau_{synch} $ / length')
plt.xlabel(r'q')
plt.title('k = ..')





######Paths kai times flatten according to q ############################
pathsq = np.zeros((len(qs),(paths[:,0,:].flatten()).shape[0]))
iterq = np.zeros((len(qs),(paths[:,0,:].flatten()).shape[0]))
for i in range(len(qs)):
    pathsq[i] = paths[:,i,:].flatten()
    iterq[i] = iter_time[:,i,:].flatten()
    plt.plot(pathsq[i],iterq[i],'.')
    
ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
    
for i in range(len(qs)):
    if i == len(qs)-1:
        plt.plot(pathsq[i],iterq[i],'.', label=" $q = 0$")
    else:
        plt.plot(pathsq[i],iterq[i],'.', label=" $q = 2^{-%d}$" % m[i])

plt.xlabel('average shortest path length $l$ ')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')
plt.legend(numpoints = 1,loc=4)
#################################################################################
#Plot diameters

diamsq = np.zeros((len(qs),(diameters[:,0,:].flatten()).shape[0]))
iterq = np.zeros((len(qs),(diameters[:,0,:].flatten()).shape[0]))
for i in range(len(qs)):
    diamsq[i] = diameters[:,i,:].flatten()
    iterq[i] = iter_time[:,i,:].flatten()
    plt.plot(diamsq[i],iterq[i],'.')
    
ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
    
for i in range(len(qs)):
    if i == len(qs)-1:
        plt.plot(diamsq[i],iterq[i],'.', label=" $q = 0$")
    else:
        plt.plot(diamsq[i],iterq[i],'.', label=" $q = 2^{-%d}$" % m[i])

plt.xlabel('diameter $diam$ ')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')
plt.legend(numpoints = 1,loc=4)


#### Difference diameter kai radius
diamsfla = np.zeros(((diameters[:,:,:].flatten()).shape[0]))

radiafla = np.zeros(((diameters[:,:,:].flatten()).shape[0]))
di = np.zeros(((diameters[:,:,:].flatten()).shape[0]))
diamsfla = diameters.flatten()
radiafla = radia.flatten()
for i in range(len(diamsfla)):
    if diamsfla[i] == 999999:
        di[i] = 9999
    else:
        di[i] = diamsfla[i] - radiafla[i]
        
timesfla = iter_time.flatten()

plt.plot(di,timesfla,'.')
        


################Diameter - Radius gia kathe q

diamsq = np.zeros((len(qs),(diameters[:,0,:].flatten()).shape[0]))
radq = np.zeros((len(qs),(diameters[:,0,:].flatten()).shape[0]))
diq = np.zeros((len(qs),(diameters[:,0,:].flatten()).shape[0]))
iterq = np.zeros((len(qs),(diameters[:,0,:].flatten()).shape[0]))
for i in range(len(qs)):
    diamsq[i] = diameters[:,i,:].flatten()
    radq[i] = radia[:,i,:].flatten()
    iterq[i] = iter_time[:,i,:].flatten()
    for j in range(len(diamsq[i])):
        if diamsq[i][j] == 999999:
            diq[i][j] = 9999
        else:
            diq[i][j] = diamsq[i][j] - radq[i][j]
    
ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
    
for i in range(len(qs)):
    if i == len(qs)-1:        
        plt.plot(diq[i],iterq[i],'.', label=" $q = 0$")
    else:
        plt.plot(diq[i],iterq[i],'.', label=" $q = 2^{-%d}$" % m[i])

plt.xlabel(' $diameter - radius$ ')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')
plt.legend(numpoints = 1,loc=4)



################Diameter - Radius gia kathe q

diamsn = np.zeros((len(ks),(diameters[0,:,:].flatten()).shape[0]))
radn = np.zeros((len(ks),(diameters[0,:,:].flatten()).shape[0]))
din = np.zeros((len(ks),(diameters[0,:,:].flatten()).shape[0]))
itern = np.zeros((len(ks),(diameters[0,:,:].flatten()).shape[0]))
for i in range(len(ns)):
    diamsn[i] = diameters[i,:,:].flatten()
    radn[i] = radia[i,:,:].flatten()
    itern[i] = iter_time[i,:,:].flatten()
    for j in range(len(diamsn[i])):
        if diamsn[i][j] == 999999:
            din[i][j] = 9999
        else:
            din[i][j] = diamsn[i][j] - radn[i][j]
    
ax = plt.subplot()    
ax.set_yscale("log", nonposy='clip') 
    
for i in range(len(ks)):
    
    plt.plot(diamsn[i],itern[i],'.', label=" $k = %d$" % ks[i])

plt.xlabel(' $diameter - radius$ ')
plt.ylabel(r'synchr. timescale $\tau_{synch} $')
plt.legend(numpoints = 1,loc=4)





        
#var_dict3 = {"N": N, "Gs": Gs,"Pres":Pres, "Posts":Posts, "iter_time":iter_time,"eig_time":eig_time, "iter_perturb":iter_perturb, "A":A, "A0":A0,"sim_time":sim_time,"sim_perturb":sim_perturb,"ks":ks,"qs":qs,"iter_history":iter_history ,"iter_dist":iter_dist,"sim_history":sim_history,"sim_dist":sim_dist}
#pickle.dump(var_dict3, open("Links.dat", "wb"))  

"""