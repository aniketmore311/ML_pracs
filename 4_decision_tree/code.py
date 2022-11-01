#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# In[49]:


df = pd.read_csv('cancer.csv')
df.head()


# In[50]:


print(df.columns)


# In[51]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[52]:


df_new = df.drop(columns=['id','diagnosis','Unnamed: 32'])
df_new.head()


# In[53]:


df.corr()['diagnosis'].sort_values()


# In[55]:


X = df[['concave points_worst','perimeter_worst','concave points_mean','radius_worst','perimeter_mean']]
y = df['diagnosis']
print(X[:5])
print(y[:5])


# In[56]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)


# In[57]:


dtree = DecisionTreeClassifier(random_state=12)
dtree.fit(X,y)


# In[58]:


plt.figure(figsize=(30,30))
plot_tree(dtree, fontsize=15)
plt.show()


# In[59]:


y_pred = dtree.predict(X_test)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

