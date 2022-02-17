#!/usr/bin/env python
# coding: utf-8

# Determine the accuracy of the Data present in CSV file regarding Brest Cancer using Decision Tree Classifier

# In[1]:


'''Import Csv file'''
import pandas as pd

dataset=pd.read_csv("data.csv")
dataset


# In[2]:


dataset.shape
datasetLength=len(dataset)


# In[3]:


'''Data Cleaning process'''
for index in range(datasetLength):
    if dataset["diagnosis"].iloc[index]=='M':
        dataset["diagnosis"].iloc[index]=1      #Changing Malignant to 1
    if dataset["diagnosis"].iloc[index]=='B':
        dataset["diagnosis"].iloc[index]=2      #Changing Benign to 2     


# In[4]:


import numpy as np                       
dataset=dataset.replace(0,np.NaN)           #Replace with Not available


# In[5]:


import missingno as msno
msno.bar(dataset)                        #Verify whether there are any missing number


# In[6]:


#Based on the above bar graph, we can see some columns having values missing. Those are filled using fillna as shwon below

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


'''Import Decision Tree from ski Learn'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Diagnosis is taken as Output (y)  
X = dataset.drop(["id","diagnosis"],axis = 1)        
y = dataset["diagnosis"]
y = y.astype('int')


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)   #Split Dataset. 80% Training and 20% Testing


# In[9]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)        #Fit model for training using Decision Tree 


# In[10]:


y_Prediction=model.predict(X_test)      #Predict the testing data
accuracy_score(y_test,y_Prediction)     #Check for accuracy of the data

