#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import math


# In[27]:


df = pd.read_csv('Real estate.csv')
df.head()


# In[28]:


print(df.columns)


# In[29]:


y_label = 'Y house price of unit area'
x1_label = 'X2 house age'; 
x2_label ='X3 distance to the nearest MRT station' ; 
x3_label = 'X4 number of convenience stores'; 

plt.scatter(df[x1_label], df[y_label], c="red")
plt.show()
plt.scatter(df[x2_label], df[y_label], c="green")
plt.show()
plt.scatter(df[x3_label], df[y_label], c="blue")
plt.show()


# In[30]:


num_samples = df.shape[0]
num_features = 3

X1 = df[x1_label].to_numpy().reshape((num_samples,1))
X2 = df[x2_label].to_numpy().reshape((num_samples,1))
X3 = df[x3_label].to_numpy().reshape((num_samples,1))

X = np.hstack((np.ones((num_samples,1)),X1,X2,X3))
X[:5]


# In[31]:


y = df[y_label].to_numpy().reshape((num_samples,1))
y[:5]


# In[32]:


parameters = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
parameters


# In[33]:


y_pred = X.dot(parameters)
y_pred[:5]


# In[34]:


print("r2Score:",r2_score(y,y_pred))
print("rmse:", math.sqrt(mean_squared_error(y,y_pred)))

