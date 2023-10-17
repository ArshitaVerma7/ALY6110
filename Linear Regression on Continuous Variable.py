#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("df.csv")


# In[4]:


data


# In[5]:


print(data.head(10))
data.tail()


# In[6]:


import os


# In[7]:


plt.scatter(data.age,data.lstat)


# In[8]:


get_ipython().system('pip install -U scikit-learn')


# In[27]:


sns.heatmap(data.corr())


# In[10]:


from sklearn import linear_model
import sklearn


# In[11]:


## Building linear  regression  model

reg = linear_model.LinearRegression()
reg.fit(data[["age"]], data.lstat)


# In[32]:


## Predictions

df = pd.DataFrame([[10],[20]], columns = ["age"])
reg.predict(data[["age"]])


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


# split into train test sets
train, test = train_test_split(data, test_size=0.30, random_state=42)


# In[15]:


## k fold cross validation

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train, test in kf.split(data):
    print("%s %s" % (train, test))


# In[16]:


from sklearn.model_selection import cross_val_score
import statistics
import math 


# In[31]:


##Calculating MAE

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = reg
scores = cross_val_score(model, data[["age"]], data.lstat, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
print(scores)
statistics.mean(abs(scores))


# In[18]:


##Calculating MSE

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = reg
scores = cross_val_score(model, data[["age"]], data.lstat, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)
math.sqrt(statistics.mean(abs(scores)))


# In[29]:


##converting continuous type of variable to categorical type.

from sklearn import metrics
data['Label'] = pd.cut(x=data['age'], bins=[0,  63, 100],
                     labels=['Child',
                             'Elderly'])
print(data)
print("Categories: ")
print(data['Label'].value_counts())

