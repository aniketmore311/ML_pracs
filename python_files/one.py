#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import math


# In[5]:


df = pd.read_csv("tvmarketing.csv")
df.head()


# In[6]:


num_samples = df.shape[0]
num_features = df.shape[1] - 1
TV = df['TV'].to_numpy().reshape((num_samples,1))
ones = np.ones((num_samples,1))
X = np.hstack((ones,TV))
X[:5]


# In[7]:


y = df['Sales'].to_numpy().reshape((num_samples,1))
y[:5]


# In[8]:


plt.scatter(TV,y)
plt.show()


# In[9]:


features = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
features


# In[10]:


y_pred = X.dot(features)
y_pred[:5]


# In[11]:


plt.scatter(TV,y)
plt.plot(TV,y_pred, c="red")
plt.show()


# In[15]:


print("r2Score:",r2_score(y,y_pred))
print("rmse:", math.sqrt(mean_squared_error(y,y_pred)))

