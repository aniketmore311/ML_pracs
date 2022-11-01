#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# In[15]:


df = pd.read_csv('cancer.csv')
df.head()


# In[16]:


print(df.columns)


# In[17]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[18]:


df_new = df.drop(columns=['id','diagnosis','Unnamed: 32'])
df_new.head()


# In[19]:


df.corr()['diagnosis'].sort_values()


# In[20]:


X = df[['concave points_worst','perimeter_worst','concave points_mean','radius_worst','perimeter_mean']]
y = df['diagnosis']
print(X[:5])
print(y[:5])


# In[24]:


pca = PCA(n_components=2)
pca.fit(X)
X_2Dim = pca.transform(X)
print("eigen vectors\n",pca.components_)
print("eigen values\n",pca.explained_variance_)
print("transformed data\n")
print(X_2Dim[:5])


# In[28]:


def getAccuracy(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return accuracy_score(y_test,y_pred)


# In[29]:


print('original accuracy: ')
print(getAccuracy(X,y))
print("2d accuracy")
print(getAccuracy(X_2Dim,y))

