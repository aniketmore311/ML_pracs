#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


# In[2]:


df = pd.read_csv('cancer.csv')
df.head()


# In[3]:


print(df.columns)


# In[4]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[5]:


df.corr()['diagnosis'].sort_values()


# In[6]:


X = df[['concave points_worst','perimeter_worst','concave points_mean','radius_worst','perimeter_mean']]
y = df['diagnosis']
print(X[:5])
print(y[:5])


# In[7]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)


# In[8]:


classifier = SVC(kernel='linear')
classifier.fit(X,y)


# In[9]:


y_pred = classifier.predict(X_test)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

