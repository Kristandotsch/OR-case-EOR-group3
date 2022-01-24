#!/usr/bin/env python
# coding: utf-8

# ## Data Manipulation

# In[2]:


import csv
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

# read .dat to a list of lists
datContent = [i.strip().split() for i in open("krackad.dat").readlines()]

# Create 21 x 21 x 21 array
flat_data = np.array(datContent[26:])
data = np.reshape(flat_data, (21, 21, 21)).astype(int)

data.shape


# ## Aggregation Methods

# ![image.png](attachment:image.png)

# In[3]:


def consensus_structure(array):    
    return array.mean(axis=0).round()

def locally_aggregated_structures(array):
    las_array = np.zeros((21,21))

    for p1 in range(21):
        for p2 in range(21):
            if array[p1, p1, p2] == 1 and array[p2, p1, p2] == 1:
                las_array[p1, p2] = 1         
    return las_array


# In[4]:


def plot_network(array):
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 22))

    edges = np.argwhere(array>0) + 1
    G.add_edges_from(edges)
    return G


# ## Network Plots

# In[5]:


plt.rcParams["figure.figsize"] = (12,8)


# ### Consensus Structure (CS)

# In[6]:


data_cons = consensus_structure(data)

G_cs = plot_network(data_cons)

nx.draw_shell(G_cs, with_labels=True, font_weight='bold')


# In[7]:


df_cs = pd.DataFrame([], index=range(1, 22))
df_cs['CS Out Degree'] = pd.Series(dict(G_cs.out_degree()))
df_cs['CS In Degree'] = pd.Series(dict(G_cs.in_degree()))

df_cs


# In[8]:


## add betweenness score
G = nx.from_numpy_matrix(data_cons)
btw_cons = nx.betweenness_centrality(G)

data_values = btw_cons.values()
df_cs['CS betweenness'] = data_values


# In[9]:


df_cs


# ### Locally Aggregated Structure (LAS) 

# In[10]:


data_las = locally_aggregated_structures(data)
G_las = plot_network(data_las)

nx.draw_shell(G_las, with_labels=True, font_weight='bold')


# In[11]:


df_las = pd.DataFrame([], index=range(1, 22))
df_las['LAS Out Degree'] = pd.Series(dict(G_las.out_degree()))
df_las['LAS In Degree'] = pd.Series(dict(G_las.in_degree()))

df_las


# In[12]:


## add betweenness score
G = nx.from_numpy_matrix(data_las)
btw_las = nx.betweenness_centrality(G)

data_values = btw_las.values()
df_las['LAS betweenness'] = data_values


# In[13]:


df_las


# ### Summary of Node Degrees 

# In[14]:


network_degrees = df_cs.merge(df_las,
                              left_index=True, 
                              right_index=True)
network_degrees


# # Question 3

# The network of a company can be affected by employees moving on and leaving the company. Identify the 
# employee whose leaving the company would have the greatest impact on the social structure of the 
# network.
# - employees who have a high centrality score (indegree, outdegree, betweenness)
# - in terms of the matrix: removing which person and their links changes the matrix the most 
# - what is typcial for an employee who has the greatest impact on the social structure

# In[83]:


## employees with highest indegree
# CS
network_degrees['CS In Degree'].sort_values(ascending=False)[:5]
odc = nx.out_degree_centrality(G_las)
print(odc)
clc = nx.closeness_centrality(G_las)
print(clc)
pr = nx.pagerank(G_las, alpha = 0.8)
pr = dict(sorted(pr.items(), key=lambda item: item[1], reverse=True))
print(pr)


# In[74]:


## employees with highest betweenness
# CS
network_degrees['CS betweenness'].sort_values(ascending=False)[:5]


# In[17]:


## employees with highest indegree
# LAS
network_degrees['LAS In Degree'].sort_values(ascending=False)[:5]


# In[18]:


## employees with highest betweenness
# LAS
network_degrees['LAS betweenness'].sort_values(ascending=False)[:5]


# ### Change of network when removing an employee

# In[19]:


def create_network_without_x(employee, matrix):
    # delete xth row from matrix
    new_matrix = np.delete(matrix, (employee-1), axis=0)
    # delete xth column from matrix
    new_matrix = np.delete(new_matrix, (employee-1), axis=1)
    # compute centrality on new network
    G = plot_network(new_matrix)
    index = [x for x in range(1,22) if x != employee]
    df = pd.DataFrame([], index=index)
    df['New Outdegree'] = pd.Series(dict(G.out_degree()))
    df['New Indegree'] = pd.Series(dict(G.in_degree()))
    
    return df


# In[20]:


# based on correlation
def corr_old_new_network(matrix_curr, df_curr_indegree, df_curr_outdegree):
    df_corr = pd.DataFrame(columns = ['Indegree corr','Outdegree corr'], index=range(1, 22))
    for i in range(1,22):
        df_curr_indegree_no_x = df_curr_indegree.drop(i)
        df_curr_outdegree_no_x = df_curr_outdegree.drop(i)
        df_new = create_network_without_x(i,matrix_curr)
        df_corr.at[i, 'Indegree corr'] = df_curr_indegree_no_x.corr(df_new['New Indegree'])
        df_corr.at[i, 'Outdegree corr'] = df_curr_outdegree_no_x.corr(df_new['New Outdegree'])
        
    return df_corr


# In[21]:


# check impact of removing each employee on CS network
df_change_corr = corr_old_new_network(data_cons, df_cs['CS In Degree'], df_cs['CS Out Degree'])
df_change_corr


# In[88]:


def plot_network_weighted(array):
    arr_ib = np.transpose(array)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 22))
    tup_l = []
    
    for i in range(1,22):
        wx = 0
        if sum(arr_ib[i-1]>0):
            wx = 1/sum(arr_ib[i-1])
        
        edgeC = np.argwhere(arr_ib[i-1]>0) + 1
        
        
        if len(edgeC) > 0:
            for k in range(len(edgeC)):
                wx = float(wx)
                tup = (edgeC[k][0], i, wx)
                tup_l.append(tup)
                             
                
        
    G.add_weighted_edges_from(tup_l)
    nx.draw_shell(G, with_labels=True, font_weight='bold')
    return G
                
newNW = plot_network_weighted(data_las)
newNW.degree(weight='weight')


# In[64]:


color_map = []
for node in newNW:
    if node < 10:
        color_map.append('red')
    else: 
        color_map.append('pink')      
nx.draw(newNW, node_color=color_map, with_labels=True)
plt.show()


# In[30]:


## most corr change for indegree
df_change_corr['Indegree corr'].sort_values(ascending=True)[:5]


# In[31]:


## most corr change for outdegree
df_change_corr['Outdegree corr'].sort_values(ascending=True)[:5]


# In[25]:


def avg_relative_change(series1, series2):
    change_rates = []
    for i in range(len(series1)):
        if series1.iloc[i] == 0:
            continue
        else:
            change_rate = abs(series2.iloc[i]-series1.iloc[i])/series1.iloc[i]
            change_rates.append(change_rate)
    return sum(change_rates)/len(change_rates)


# In[26]:


# based on average change rate
def change_old_new_network(matrix_curr, df_curr_indegree, df_curr_outdegree):
    df_change = pd.DataFrame(columns = ['Indegree change','Outdegree change'], index=range(1, 22))
    for i in range(1,22):
        df_curr_indegree_no_x = df_curr_indegree.drop(i)
        df_curr_outdegree_no_x = df_curr_outdegree.drop(i)
        df_new = create_network_without_x(i,matrix_curr)
        df_change.at[i, 'Indegree change'] = avg_relative_change(df_curr_indegree_no_x,df_new['New Indegree'])
        df_change.at[i, 'Outdegree change'] = avg_relative_change(df_curr_outdegree_no_x,df_new['New Outdegree'])
        
    return df_change


# In[27]:


df_change_rate = change_old_new_network(data_cons, df_cs['CS In Degree'], df_cs['CS Out Degree'])
df_change_rate


# In[28]:


## most change for indegree
df_change_rate['Indegree change'].sort_values(ascending=False)[:5]


# In[29]:


## most change for outdegree
df_change_rate['Outdegree change'].sort_values(ascending=False)[:5]


# In[30]:


## least change for indegree
df_change_rate['Indegree change'].sort_values(ascending=True)[:5]


# In[31]:


## least change for outdegree
df_change_rate['Outdegree change'].sort_values(ascending=True)[:5]


# In[32]:


## plot network without 1
new_matrix = np.delete(data_cons, (0), axis=0)
new_matrix = np.delete(new_matrix, (0), axis=1)
G1 = plot_network(new_matrix)

nx.draw_shell(G1, with_labels=False, font_weight='bold')


# In[33]:


## plot network without 4
new_matrix = np.delete(data_cons, (3), axis=0)
new_matrix = np.delete(new_matrix, (3), axis=1)
G2 = plot_network(new_matrix)

nx.draw_shell(G2, with_labels=False, font_weight='bold')


# In[34]:


## contrast: plot network without 19
new_matrix = np.delete(data_cons, (18), axis=0)
new_matrix = np.delete(new_matrix, (18), axis=1)
G3 = plot_network(new_matrix)

nx.draw_shell(G3, with_labels=False, font_weight='bold')


# ## Question 4

# The social network of a company is perceived differently by different individuals. Are there individuals 
# whose view of the world is so different from that of the majority that this causes concern? 
# - correlations between own slice and two aggregations 
# - percent confirmed indegree/outdegree

# In[100]:


def create_df_kth_slices(kth_matrix):
    G = plot_network(kth_matrix)
    df = pd.DataFrame([], index=range(1, 22))
    df['K Out Degree'] = pd.Series(dict(G.out_degree()))
    df['K In Degree'] = pd.Series(dict(G.in_degree()))
    
    G = nx.from_numpy_matrix(kth_matrix)
    btw_cons = nx.betweenness_centrality(G)
    data_values = btw_cons.values()
    df['K betweenness'] = data_values
    
    return df


# In[102]:


df_correlations = pd.DataFrame(columns = ['Indegree LAS','Outdegree LAS','Betweenness LAS',
                                          'Indegree CS','Outdegree CS','Betweenness CS'], index=range(1, 22))
for i in range(1, 22):
    df_k = create_df_kth_slices(data[i-1])
    df_correlations.at[i, 'Indegree LAS'] = df_k['K In Degree'].corr(df_las['LAS In Degree'])
    df_correlations.at[i, 'Outdegree LAS'] = df_k['K Out Degree'].corr(df_las['LAS Out Degree'])
    df_correlations.at[i, 'Betweenness LAS'] = df_k['K betweenness'].corr(df_las['LAS betweenness'])
    df_correlations.at[i, 'Indegree CS'] = df_k['K In Degree'].corr(df_cs['CS In Degree'])
    df_correlations.at[i, 'Outdegree CS'] = df_k['K Out Degree'].corr(df_cs['CS Out Degree'])
    df_correlations.at[i, 'Betweenness CS'] = df_k['K betweenness'].corr(df_cs['CS betweenness'])


# In[103]:


df_correlations


# In[ ]:




