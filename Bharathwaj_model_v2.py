#!/usr/bin/env python
# coding: utf-8

# Determine the accuracy of the Data present in CSV file regarding Brest Cancer using KNN 

# In[1]:


'''Importing and data cleaning same like model_v1'''
import pandas as pd

dataset=pd.read_csv("data.csv")
dataset


# In[2]:


dataset.shape
datasetLength=len(dataset)


# In[3]:


for index in range(datasetLength):
    if dataset["diagnosis"].iloc[index]=='M':
        dataset["diagnosis"].iloc[index]=1      #Changing Malignant to 1
    if dataset["diagnosis"].iloc[index]=='B':
        dataset["diagnosis"].iloc[index]=2      #Changing Benign to 2     


# In[4]:


import numpy as np
dataset=dataset.replace(0,np.NaN)


# In[5]:


import missingno as msno
msno.bar(dataset)


# In[6]:


concavityMean=dataset["concavity_mean"].mean()
dataset["concavity_mean"].fillna(concavityMean, inplace = True)
concavePointsMean=dataset["concave points_mean"].mean()
dataset["concave points_mean"].fillna(concavePointsMean, inplace = True)
concavitySe=dataset["concavity_se"].mean()
dataset["concavity_se"].fillna(concavitySe, inplace = True)
concavePointsSe=dataset["concave points_se"].mean()
dataset["concave points_se"].fillna(concavePointsSe, inplace = True)
concavityWorst=dataset["concavity_worst"].mean()
dataset["concavity_worst"].fillna(concavityWorst, inplace = True)
concavePointsWorst=dataset["concave points_worst"].mean()
dataset["concave points_worst"].fillna(concavePointsWorst, inplace = True)

msno.bar(dataset)


# In[7]:


# Import K-Nearest Neighbour from Ski-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = dataset.drop(["id","diagnosis"],axis = 1)
y = dataset["diagnosis"]
y = y.astype('int')


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)


# In[9]:


model=KNeighborsClassifier(n_neighbors=4)
model.fit(X_train,y_train)


# In[10]:


y_Prediction=model.predict(X_test)
accuracy_score(y_test,y_Prediction)

