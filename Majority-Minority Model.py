#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.ma as ma
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random
import scipy as sp
import random
from matplotlib import animation, rc
from IPython.display import HTML


# In[2]:


## NOT USED IN LATER CODE ##
# Create a sparse martix, where all nodes have the same number of connections 
# (0 and 1 indicate absence/presence of a connection)
def watts_stroganov_network_anna(n, k):

    # create a list with the "nodes"
    list_nodes = list(range(n))
    list_edges = [] # a list where we will sotre the edges (tuples with node_in, node_out)

    # dictionary to store the number of connections of each node so we can remove them when it reaches the desired value
    number_connections = {key:0 for key in list_nodes} 

    while len(list_nodes) > 1:

        # randomly choose 2 nodes which are different and which are not already connected
        node1 = random.choice(list_nodes)
        node2 = random.choice(list_nodes)
        # this while loop makes sure that the nodes are different and the edge (node1, node2) does not already exists
        while node2 == node1 or (node1,node2) in list_edges or (node2,node1) in list_edges: 
            node2 = random.choice(list_nodes)

        list_edges.append((node1,node2)) # append the edge to the list of edges

        # increase the counting of the node connection
        number_connections[node1] += 1
        number_connections[node2] += 1

        # if the node has already k connections, then we remove it from the list_nodes
        if number_connections[node1] == k:
            list_nodes.remove(node1)
        if number_connections[node2] == k:
            list_nodes.remove(node2)
            
    G2 = nx.Graph()
    G2.add_nodes_from(range(n))
    G2.add_edges_from(list_edges)
    return G2


# In[3]:


#Calculate the running mean
N = 8

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# In[4]:


#Find the last x and y values of a run 
def values(steps, totals):
    if steps == max_steps:
        x_values = np.arange(steps+2)
        y_values = np.append(totals, totals[steps])
    else:
        x_values = np.arange(steps+3)
        y_values = np.append(totals, totals[steps+1])
    return x_values, y_values 


# In[5]:


#Synchronous vote update

def sync_votes_update(n, k, p_maj, max_steps, opinions, random_initial_votes, b, ar_G2, totals):
    stuck = 0
    steps = 0
    new_vote = np.zeros(n)
    count_votes = np.count_nonzero(random_initial_votes, axis=0)
    pos_end_state = np.ones(n)
    neg_end_state = np.ones(n)*(-1)
    
    #update votes until max steps or an agreement is reached
    while steps<max_steps:
        total_opinions = np.sum(random_initial_votes, axis=0)
        
        for i in range(n):
            initial_vote = opinions[i]
            new_info = total_opinions[i]
            n_votes = count_votes[i]

            #no change if there is a tie or change the sign acccordingly if all votes are the same
            if new_info==0:
                new_vote[i] = initial_vote
            if abs(new_info) == n_votes:
                new_vote[i] = np.sign(new_info)

            #otherwise check the majority vote and change opinions accordingly
            if (np.sign(new_info) < 0 and new_info > (-n_votes)):
                new_vote[i] = np.random.choice([-1,1], p=[p_maj, (1-p_maj)])
            if (np.sign(new_info) > 0 and new_info < n_votes):
                new_vote[i] = np.random.choice([-1,1], p=[(1-p_maj), p_maj])
                            
        opinions = new_vote
        total = np.sum(opinions)
        totals.append(total)
        
# UNCOMMENT FOR ANIMATION
# ----------------------------------------------------------------------------------------------------------        
#         nodesVotePlusOne  = []
#         nodesVoteMinusOne = []

#         for i in range(len(opinions)):
#             if (opinions[i] > 0):
#                 nodesVotePlusOne.append(i)
#             else:
#                 nodesVoteMinusOne.append(i)

#         plt.rcParams['figure.figsize'] = [5,5]
#         fig = plt.figure()
#         nx.draw_networkx_nodes(G2, pos=nx.circular_layout(G2), nodelist=nodesVotePlusOne, node_color='cornflowerblue')
#         nx.draw_networkx_nodes(G2, pos=nx.circular_layout(G2), nodelist=nodesVoteMinusOne, node_color='tomato')
#         nx.draw_networkx_edges(G2, pos=nx.circular_layout(G2), alpha=0.3, width=1)
#         plt.title('time step t =' + str(steps),fontsize=18)
#         plt.savefig("zplot_"+str(steps).zfill(5)+".png")
#         plt.close(fig)  
# ----------------------------------------------------------------------------------------------------------        

        #check the rolling mean of N iterations and see if the values change
        if steps>N:
            means = running_mean(totals, N)
            last_means = means[-(N+1):-1]
            stuck = np.all(last_means == last_means[0])
            if stuck == True:
                stuck=1
                break
        
        #conditions to stop running are if all nodes are 1 or -1, otherwise continue
        if np.all(opinions==pos_end_state) or np.all(opinions==neg_end_state):
            b = np.array([new_vote])
            random_initial_votes = ar_G2*b.T
            break
        else:
            b = np.array([new_vote])
            random_initial_votes = ar_G2*b.T
            steps +=1  
       
    return steps, random_initial_votes, totals, stuck


# In[6]:


#Run a single simulation
def one_run(np_G2, n, k, p_maj, max_steps):
    #randomly assign initial votes
    opinions = np.random.choice([-1,1], n, p=[0.5, 0.5])
    b = np.array([opinions])
    ar_G2=np.asarray(np_G2)
    random_initial_votes = ar_G2*b.T
    totals = []
    totals.append(np.sum(opinions))

    #update the votes synchronously until consensus or max_steps is reached 
    steps, _, totals, _ = sync_votes_update(n, k, p_maj, max_steps, opinions, random_initial_votes, b, ar_G2, totals)

    return steps, totals


# In[7]:


n=4
k=3
max_steps=20
pos_end_state = np.ones(n)
neg_end_state = np.ones(n)*(-1)
p_maj=0.7

#Create a sparse watts-strogatz martix
G2 = nx.watts_strogatz_graph(n, k, p=0.2)
np_G2 = nx.to_numpy_matrix(G2)


steps, totals = one_run(np_G2, n, k, p_maj, max_steps)

#plot two end states and step-wise changes in the total votes
pos_consensus = np.sum(pos_end_state);
neg_consensus = np.sum(neg_end_state);
normalized_totals = [x / n for x in totals]
plt.plot(normalized_totals);

plt.hlines(1, 0, max_steps+1);
plt.hlines(-1, 0, max_steps+1);
plt.xlabel('Time steps');
plt.ylabel('Normalized total vote');
plt.title('Dynamics of the total vote change');


# Animations

# In[8]:


# #Animate the total opinion change
# plt.rcParams["animation.html"] = "jshtml"
# x_values, y_values = values(steps, normalized_totals)
# fig = plt.figure()
# line, = plt.plot([], [], lw=2)

# def init():
#     line.set_data([], [])
#     return (line,)

# def animate(i):
#     line.set_data(x_values[:i], y_values[:i]) 
#     return (line,)

# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x_values+2), interval=200, blit=True) 
# HTML(anim.to_html5_video())


# In[9]:


# from PIL import Image
# import glob 

# # Animation with the network
# # Create the frames
# frames = []
# imgs = glob.glob("zplot_000*.png")

# for n, i in enumerate(imgs):
#     new_frame = Image.open(i)
#     frames.append(new_frame)
    
    
# # Save into a GIF file that loops forever
# frames[0].save('png_to_gif.gif', format='GIF',
#                append_images=frames[1:],
#                save_all=True,
#                duration=300, loop=0)


# In[10]:


# # Add path
# from IPython.display import Image
# Image(filename=".../png_to_gif.gif")


# In[11]:


#asynchronous vote change - once for all nodes
def async_one_vote_update(n, p_maj, opinions, total_opinions, count_votes, ar_G2):

    #place the nodes in random order
    random_nodes = np.random.choice(n, n, replace=False)
    b = np.zeros(n)
    step = 0
    
    for i in range(n):
        
        #pick a node and its values - initial vote, recieved information and total number of nodes
        node = random_nodes[i]
        initial_vote = opinions[node]
        new_info = total_opinions[node]
        n_votes = count_votes[node]
        
        #update the node's vote accordingly
        if new_info==0:
            new_vote = initial_vote
        if abs(new_info) == n_votes:
            new_vote = np.sign(new_info)

        #otherwise check the majority vote and change opinions accordingly
        if (np.sign(new_info) < 0 and new_info > (-n_votes)):
            new_vote = np.random.choice([-1,1], p=[p_maj, (1-p_maj)])
        if (np.sign(new_info) > 0 and new_info < n_votes):
            new_vote = np.random.choice([-1,1], p=[(1-p_maj), p_maj])

        #if the opinion stays the same, move to the next node, otherwise update the vote
        if opinions[node] == new_vote:
            step += 1
        else:
            opinions[node] = np.sign(new_vote)
            step += 1

        #update the matrix and total information recieved by all nodes
        b = np.array([opinions])
        random_initial_votes = ar_G2*b.T
        total_opinions = np.sum(random_initial_votes, axis=0)
                    
    return step, opinions, random_initial_votes


# In[12]:


#Asynchronous vote update

def async_votes_update(n, k, p_maj, max_steps, opinions, random_initial_votes, b, ar_G2, totals):
    steps = 0
    stuck = 0
    new_vote = np.zeros(n)
    count_votes = np.count_nonzero(random_initial_votes, axis=0)
    
    #define end states
    pos_end_state = np.ones(n)
    neg_end_state = np.ones(n)*(-1)
    
    #update votes until max steps or an agreement is reached
    while steps<max_steps:
        total_opinions = np.sum(random_initial_votes, axis=0)

        _, opinions, random_initial_votes = async_one_vote_update(n, p_maj,
                                                                  opinions, total_opinions, count_votes, ar_G2)

        new_vote = opinions
        total = np.sum(opinions)
        totals.append(total)
        
        #check the rolling mean of N iterations and see if the values change
        if steps>N:
            means = running_mean(totals, N)
            last_means = means[-(N+1):-1]
            stuck = np.all(last_means == last_means[0])
            if stuck == True:
                stuck=1
                break   
    
        #conditions to stop running are if all nodes are 1 or -1, otherwise continue
        if np.all(opinions==pos_end_state) or np.all(opinions==neg_end_state):
            b = np.array([new_vote])
            random_initial_votes = ar_G2*b.T
            break
        else:
            b = np.array([new_vote])
            random_initial_votes = ar_G2*b.T
            steps +=1

    return steps, random_initial_votes, totals, stuck


# In[13]:


#Collect information over multiple runs and, if necessary, plot and/or print proportions of end states

def model_plots(func, epsilon, n_nodes, k, p_rewire, runs, max_steps, verbose = True, plots = True):
    
    if plots:
        fig, ax = plt.subplots(len(n_nodes),len(epsilon), figsize=(20, 15));
    
    data = {'n_nodes':[], 'k':[], 'p_rewire':[], 'p_maj':[], 'consensus':[], 'stuck_total':[]}
    cons_distr = {'n_nodes':[], 'k':[], 'p_rewire':[], 'p_maj':[],'consensus_val':[], 'consensus_step':[]}
    stuck_distr = {'n_nodes':[], 'k':[], 'p_rewire':[], 'p_maj':[],'stuck_val':[], 'stuck_step':[]}

    for z, no in enumerate(n_nodes):
        G2 = nx.watts_strogatz_graph(no, k, p=p_rewire)
        np_G2 = nx.to_numpy_matrix(G2)
        for j, p in enumerate(p_maj):
            consensus = 0
            stuck_total = 0
            data['n_nodes'].append(no)
            data['k'].append(k)
            data['p_rewire'].append(p_rewire)
            data['p_maj'].append(p)
            
            for i in range(runs):
                opinions = np.random.choice([-1,1], no, p=[0.5, 0.5])
                b = np.array([opinions])
                ar_G2=np.asarray(np_G2)
                random_initial_votes = ar_G2*b.T
                total_opinions = np.sum(random_initial_votes, axis=0)
                count_votes = np.count_nonzero(random_initial_votes, axis=0)
                totals = []
                totals.append(np.sum(opinions))
                steps, random_initial_votes, totals, stuck = func(no, k, p, max_steps,
                                                                opinions, random_initial_votes, 
                                                                  b, ar_G2, totals)
                normalized_totals = [x / no for x in totals]
                x_values, y_values = values(steps, normalized_totals);
                
                if plots:
                    ax[z][j].set_title("End-states, p_maj " + str(p) + " & n " + str(no))
                    ax[z][j].plot(x_values[-2], y_values[-1], 'bo');
                    
                if stuck==1:
                    stuck_total += stuck
                    stuck_distr['n_nodes'].append(no)
                    stuck_distr['k'].append(k)
                    stuck_distr['p_rewire'].append(p_rewire)
                    stuck_distr['p_maj'].append(p)
                    stuck_distr['stuck_val'].append(y_values[-1])
                    stuck_distr['stuck_step'].append(x_values[-2])  
                if abs(y_values[-1])==1:
                    consensus +=1
                    cons_distr['n_nodes'].append(no)
                    cons_distr['k'].append(k)
                    cons_distr['p_rewire'].append(p_rewire)
                    cons_distr['p_maj'].append(p)
                    cons_distr['consensus_val'].append(y_values[-1])
                    cons_distr['consensus_step'].append(x_values[-2]) 

                i += 1
            data['consensus'].append((consensus/runs))
            data['stuck_total'].append((stuck_total/runs))

            if verbose:
                print('Consensuses reached: p_maj '+str(p)+', n '+str(no)+' is '+str(consensus/runs)+', systems stuck:' +str(stuck_total/runs))
                
            if plots:
                ax[z][j].hlines(1, 0, max_steps+1);
                ax[z][j].hlines(-1, 0, max_steps+1);

        if plots:
            plt.tight_layout()     
    
    return data, stuck_distr, cons_distr


# In[14]:


epsilon = [0.5, 0.3,0.2, 0.1, 0] 
p_maj = [(1-e) for e in epsilon]

n_nodes = [10,25,50]
k = 5
p_rewire = 1

max_steps = 20
runs = 100

data, stuck_distr, cons_distr = model_plots(async_votes_update, epsilon, n_nodes, k, p_rewire, runs, max_steps, verbose=False)


# In[16]:


#Collect data in tables for synchronous and/or aynchronous updates

def many_runs_collect(epsilon, n_nodes, ks, p_rewires, runs, max_steps, sync_votes = True, async_votes = True):
    sync_results = []
    sync_stuck_results = []
    sync_cons_results = []
    async_results = []
    async_stuck_results = []
    async_cons_results = []
    for p_re in p_rewires:
        for k in ks:
            
            if sync_votes:
                sync_data, sync_stuck_distr, sync_cons_distr = model_plots(sync_votes_update, epsilon, n_nodes, k, p_re, runs, max_steps, verbose = False, plots=False)
                sync_results.append(sync_data)
                sync_stuck_results.append(sync_stuck_distr)
                sync_cons_results.append(sync_cons_distr)
                
                total_stuck = pd.concat([pd.DataFrame(table1) for table1 in sync_stuck_results])
                total_cons = pd.concat([pd.DataFrame(table2) for table2 in sync_cons_results])
                total = pd.concat([pd.DataFrame(table3) for table3 in sync_results])
                y = total.groupby(['n_nodes','k','p_rewire','p_maj'], as_index=False).mean()
                y_stuck = total_stuck.groupby(['n_nodes','k','p_rewire','p_maj'], as_index=False).mean()
                y_cons = total_cons.groupby(['n_nodes','k','p_rewire','p_maj'], as_index=False).mean()

                
            if async_votes:
                async_data, async_stuck_distr, async_cons_distr = model_plots(async_votes_update, epsilon, n_nodes, k, p_re, runs, max_steps, verbose = False, plots=False)

                async_results.append(async_data)
                async_stuck_results.append(async_stuck_distr)
                async_cons_results.append(async_cons_distr)  
                
                totala = pd.concat([pd.DataFrame(table1) for table1 in async_results])
                total_astuck = pd.concat([pd.DataFrame(table2) for table2 in async_stuck_results])
                total_acons = pd.concat([pd.DataFrame(table3) for table3 in async_cons_results])
                ya = totala.groupby(['n_nodes','k','p_rewire','p_maj'], as_index=False).mean()
                y_astuck = total_astuck.groupby(['n_nodes','k','p_rewire','p_maj'], as_index=False).mean()
                y_acons = total_acons.groupby(['n_nodes','k','p_rewire','p_maj'], as_index=False).mean()
            
    return y, y_stuck, y_cons, ya, y_astuck, y_acons


# In[17]:


epsilon = [0.5, 0.3, 0.2, 0.1, 0] 
p_maj = [(1-e) for e in epsilon]

n_nodes = [10,25,50]
ks = [5, 8, 10]
p_rewires = [0, 0.2, 1]

max_steps = 60
runs = 50

y, y_stuck, y_cons, ya, y_astuck, y_acons = many_runs_collect(epsilon, n_nodes, ks, p_rewires, runs, max_steps, sync_votes = True, async_votes = True)


# In[20]:


df2 = ya.groupby(['p_rewire','p_maj'])['stuck_total'].mean();
df3 = y.groupby(['p_rewire','p_maj'])['stuck_total'].mean();
a = df2.unstack(level=0)
b = df3.unstack(level=0)
plt.plot(a, marker = 'o');
plt.plot(b, linestyle = ':', marker = 'o');
plt.xlabel('P_maj values');
plt.ylabel('Number of stalemates');
plt.title('Average number of stalemates per rewiring probability');


# In[21]:


df2 = y_cons.groupby(['n_nodes','p_maj'])['consensus_step'].mean();
df3 = y_acons.groupby(['n_nodes','p_maj'])['consensus_step'].mean();
a = df2.unstack(level=0)
b = df3.unstack(level=0)
plt.plot(a, marker = 'o');
plt.plot(b, linestyle = ':', marker = 'o');
plt.xlabel('P_maj values');
plt.ylabel('Number of steps');
plt.title('Average number of steps till consensus per node value');


# In[ ]:

