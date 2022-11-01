#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[3]:


df = pd.read_csv('cancer.csv')
df.head()


# In[4]:


df_new = df.drop(columns=['id','diagnosis', 'Unnamed: 32'])
df_new.head()


# In[5]:


num_samples = df.shape[0]
num_features = 30
preX = df_new.to_numpy().reshape((num_samples,num_features))
X = np.hstack((np.ones((num_samples,1)), preX))
print(X[:5])
y = df['diagnosis'].replace(['M','B'],[1,0]).to_numpy().reshape((num_samples,1))
print(y[:5])
plt.scatter(X[:,1],y)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[7]:


# gradient descent
n_iterations = 50000
learning_rate = 0.00001
gradients = []
iterations = []
params = np.ones((num_features+1,1))
# print(params)

def linear_predict(X,params):
    return X.dot(params)

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def predict(X,params):
    return sigmoid(linear_predict(X,params))
    
def prob_to_binary(X):
    return np.vectorize(lambda x: (int)(x>=0.5))(X)

for i in range(n_iterations):

    # IMP remember formula
    y_pred = predict(X_train,params) 
    iter_grads = (2/num_samples) * X_train.T.dot(y_pred - y_train)
    params = params - learning_rate * iter_grads

    # for plotting
    iterations.append(i)
    gradients.append(iter_grads[2,0])

plt.scatter(iterations, gradients)
plt.show()
print(params)


# In[8]:


y_pred = prob_to_binary(predict(X_test,params))
plt.scatter(X_test[:,1],y_pred)
plt.show()


# In[9]:


from sklearn import metrics
rep = metrics.classification_report(y_test,y_pred)
print(rep)
print(metrics.accuracy_score(y_test, y_pred)*100)

