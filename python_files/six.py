#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from vecstack import stacking
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# In[5]:


df = pd.read_csv('cancer.csv')
df.head()


# In[6]:


print(df.columns)


# In[7]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[8]:


df.corr()['diagnosis'].sort_values()


# In[9]:


X = df[['concave points_worst','perimeter_worst','concave points_mean','radius_worst','perimeter_mean']]
y = df['diagnosis']
print(X[:5])
print(y[:5])


# In[10]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)


# In[19]:


# stacking 

all_models = [LogisticRegression(), SVC(kernel='linear'), DecisionTreeClassifier()]

s_train, s_test = stacking(all_models, X_test=X_test,X_train=X_train, y_train=y_train, regression=True, random_state=None)

final_model = DecisionTreeClassifier()
final_model.fit(s_train,y_train)
final_pred = final_model.predict(s_test)
print(accuracy_score(y_test, final_pred))


# In[22]:


# baggin (random Forest)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))


# In[23]:


# boosting
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)
print(accuracy_score(y_test, y_pred_xgb))

