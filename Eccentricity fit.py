#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libraries and functions
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy.integrate as integrate
import scipy.special as special
from sklearn.metrics import r2_score
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#import the dataset
data = pd.read_csv('CauModCleanData.dat', quotechar='"', dtype='float32', 
                   skipinitialspace=True, delimiter="\t").to_numpy()

#transform the dataset into dataframe (for easier calling)
dataset = pd.DataFrame({
    'Participant':data[:,0],
    'block':data[:,1], 
    'trial':data[:,2], 
    'overlap':data[:,3], 
    'eccentricity':data[:,4], 
    'responce':data[:,5]
})
print(dataset)


# In[3]:


#Sort the dataset by ascending overlap and eccentricity values 
y = pd.DataFrame.sort_values(dataset, by=['overlap', 'eccentricity'], ascending=[True, True])
y_pp = y #saved it again for per participant analysis
z = pd.DataFrame(y).to_numpy()

#this gives us the mean responce values grouped by eccentricity and overlap 
mean_response = y.groupby(by=['eccentricity','overlap']).mean()['responce']
overlap_values = np.unique(mean_response.index.get_level_values(1))
eccentricity_values = np.unique(mean_response.index.get_level_values(0))


# In[4]:


plt.figure();
mean_response_values = []
for i in eccentricity_values:
    mean_response_values.append(mean_response[i])
    #plot the mean responce values per overlap for each eccentricity
    plt.scatter(mean_response[:][i].index,mean_response[:][i], label=i)
plt.title('Proportion of causal reports per ovrelap for each eccentricity');
plt.xlabel('Overlap');
plt.ylabel('Proportion of causal responses');
plt.legend();


# In[5]:


#just played aroung to see how to get to required points
new_y = pd.DataFrame.sort_values(dataset, by=['eccentricity', 'overlap'], ascending=[True, True]).to_numpy()
interval = np.squeeze(np.asarray(np.where(new_y[:,4]==0)))
response = new_y[interval[0]:interval[len(interval)-1], 5]
norm_x = (interval[:-1]-np.min(interval[:-1]))/(np.max(interval[:-1])-np.min(interval[:-1]))


# In[6]:


#A finction to fit a sigmiod regression
def sigmoid_fit(ecc):
    interval = np.squeeze(np.asarray(np.where(new_y[:,4]==ecc))) #for interval of each eccentricity
    response = new_y[interval[0]:interval[len(interval)-1], 5]  #find responses
    norm_x = (interval[:-1]-np.min(interval[:-1]))/(np.max(interval[:-1])-np.min(interval[:-1])) #normalize the values
    clf = LogisticRegression(random_state=0,solver='lbfgs').fit(norm_x.reshape(-1,1), response)
    sigmoid = clf.predict_proba(norm_x.reshape(-1,1))[:,1]
    coef = clf.coef_ #save coefficients
    intercept = clf.intercept_ #and intervals
    return sigmoid, coef, intercept


# In[7]:


sig = np.zeros((len(eccentricity_values), len(response)))
coefs = []
intercepts = []

#plot the mean responce values and the fitted sigmoid
for i in eccentricity_values:
    sig, co, inter = sigmoid_fit(i)
    coefs.append(co)
    intercepts.append(inter)
    plt.plot(norm_x, sig)
    plt.scatter(mean_response[i].index,mean_response[i], label=i)
    
plt.title('Logistic regression of proportion of causal reports');
plt.xlabel('Overlap');
plt.ylabel('Proportion of causal responses');
plt.legend();


# In[8]:


#for each eccentricity find the value of the sigmoid at each overlap values
def response_per_overlap(y, ecc_value):
    ecc = y[y['eccentricity'] == ecc_value] #replace with eccentricity value
    overlap_responses = dict()
    for overlap in overlap_values:
        overlap_responses[overlap] = ecc.sort_values(by='overlap')[ecc['overlap']==overlap]['responce'].values
    return overlap_responses


# In[9]:


eccentricities = y['eccentricity'].unique()
overlap_responses = dict()
probs = []

#plot the sigmoids with points per overlap (not smooth)
for i,ecc in enumerate(eccentricities):
    overlap_responses[ecc] = response_per_overlap(y,ecc)
    x=np.array([])
    y1=np.array([])
    for key in overlap_responses[ecc]:
        x = np.append(x,np.ones_like((overlap_responses[ecc][key]))*key)
        y1 = np.append(y1,overlap_responses[ecc][key])
    clf = LogisticRegression(random_state=0,solver='lbfgs').fit(x.reshape(-1,1), y1)
    prob = clf.predict_proba(x.reshape(-1,1))[:,1]
    probs.append(prob)
    plt.plot(x,prob, label=ecc)
    
plt.title('Logistic regression per overlap');
plt.xlabel('Overlap');
plt.ylabel('Proportion of causal responses');
plt.legend();


# In[10]:


#define a new dataset for a per perticipant analysis
mean_response_pp = y_pp.groupby(by=['Participant','eccentricity','overlap']).mean()['responce']

#update new_y table (same as in the first part)
new_y = pd.DataFrame.sort_values(dataset, by=['Participant','eccentricity', 'overlap'], 
                                 ascending=[True, True, True]).to_numpy()

participants_number = np.unique(mean_response_pp.index.get_level_values(0))


# In[11]:


#Finction to fit a logistic regression per participant
def sigmoid_part(part, ecc):
    interval = np.squeeze(np.asarray(np.where((new_y[:,4]==ecc) & (new_y[:, 0]==part))))
    response = new_y[interval[0]:interval[len(interval)-1], 5]
    norm_x = (interval[:-1]-np.min(interval[:-1]))/(np.max(interval[:-1])-np.min(interval[:-1]))
    clf = LogisticRegression(random_state=0,solver='lbfgs').fit(norm_x.reshape(-1,1), response)
    sigmoid = clf.predict_proba(norm_x.reshape(-1,1))[:,1]
    coef = clf.coef_
    intercept = clf.intercept_            
    return sigmoid, coef, intercept, norm_x


# In[12]:


#Finction to fit a polynomial regression per participant
def polynomial_part(part, ecc):
    interval = np.squeeze(np.asarray(np.where((new_y[:,4]==ecc) & (new_y[:, 0]==part))))
    response = new_y[interval[0]:interval[len(interval)-1], 5]
    norm_x = (interval[:-1]-np.min(interval[:-1]))/(np.max(interval[:-1])-np.min(interval[:-1]))   
    proc = PolynomialFeatures(2) #fit for order 3 polynomial 
    norm_x_poly = proc.fit_transform(norm_x.reshape((-1,1)))
    reg = LinearRegression().fit(norm_x_poly, response)
    poly = reg.predict(norm_x_poly)
    coef = clf.coef_
    intercept = clf.intercept_
    return poly, coef, intercept, norm_x_poly


# In[13]:


#Functions to display uneven number of subplots
def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        return math.ceil(k/3), 3


def generate_subplots(k, row_wise=False):
    nrow, ncol = choose_subplot_dimensions(k)
    figure, axes = plt.subplots(nrow, ncol,
                                sharex=True,
                                sharey=False, 
                                figsize=(15, 25))

    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        axes = axes.flatten(order=('C' if row_wise else 'F'))

        # Delete any unused axes from the figure, so that they don't show
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)

        axes = axes[:k]
        return figure, axes


# In[14]:


def plot_model(model):
    coefs = []
    intercepts = []
    score_sig = {}
    figure, axes = generate_subplots(len(participants_number), row_wise=True)
    
    for j, ax in zip(participants_number, axes): #per participant
        score_sig[j] = []
        for i in eccentricity_values: #per eccentricity
            sig, co, inter, norm_x = sigmoid_part(j, i) #fit a sigmoid
            f = interpolate.interp1d(norm_x,sig)
            y_pred = f(mean_response_pp[j][i].index)
            score_sig[j].append(r2_score(mean_response_pp[j][i],y_pred)) #find r_squared value for each overlap
            coefs.append(co[0,0]) 
            intercepts.append(inter[0])
            ax.plot(norm_x, sig, label=f'i={i}, r2={score_sig[j][-1]:.2f}')
            ax.scatter(mean_response_pp[j][i].index,mean_response_pp[j][i])
        ax.set_title("Causal responces per overlap by participant "+str(int(j)))
        ax.set_xlabel('Overlap');
        ax.set_xlabel('Proportion of causal responses');
        ax.legend()

    plt.tight_layout()
    
    return score_sig 


# In[15]:


score_sig = plot_model(sigmoid_part)


# In[16]:


def plot_model2(polynomial_part):
    coefs = []
    intercepts = []
    score = {}
    figure, axes = generate_subplots(len(participants_number), row_wise=True)


    #this part can be made into a finction to look into more models
    for j, ax in zip(participants_number, axes): #for each participant
        score[j] = []
        for i in eccentricity_values: #per eccentricity     
            poly, coef, intercept, norm_x_poly = polynomial_part(j, i) #calculate polynomial features
            f = interpolate.interp1d(norm_x_poly[:,1],poly)
            y_pred = f(mean_response_pp[j][i].index)
            score[j].append(r2_score(mean_response_pp[j][i],y_pred)) #find r squared values
            coefs.append(coef)
            intercepts.append(intercept)
            ax.plot(norm_x_poly[:,1], poly, label=f'i={i}, r2={score[j][-1]:.2f}')
            ax.scatter(mean_response_pp[j][i].index,mean_response_pp[j][i])
        ax.set_title("Causal responces per overlap by participant "+str(int(j)))
        ax.set_xlabel('Overlap');
        ax.set_xlabel('Proportion of causal responses');
        ax.legend()

    plt.tight_layout()
    
    return score


# In[17]:


score = plot_model2(polynomial_part)


# In[19]:


#create a table with all r_squared values
df_score = pd.DataFrame(score)
df_score.index = [0,4,8,12]

#create a table with all r_squared values
df_score_sig = pd.DataFrame(score_sig)
df_score_sig.index = [0,4,8,12]


# In[20]:


#compare the r squared values per participant and eccentricity 
df_score.T.plot();
plt.title('Polynomial fit');
plt.xlabel('Participants');
plt.ylabel('R^2 values');
plt.ylim([0,1]);

df_score_sig.T.plot();
plt.ylim([0,1]);
plt.title('Logistic regression');
plt.xlabel('Participants');
plt.ylabel('R^2 values');


# In[ ]:




