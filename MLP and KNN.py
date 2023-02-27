#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
import statsmodels.api as sm
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 10)

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("letters.csv")
df.head()


# In[3]:


df.shape


# In[33]:


df.dtypes


# In[34]:


df.isnull().sum()


# In[35]:


df.duplicated().sum()


# In[36]:


display(df.drop_duplicates())


# In[61]:


x = df.drop(['label'], axis = 1)
y= df['label']


# In[62]:


#normalization

x.div(255)


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[64]:


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df, dummies],axis=1)
    return df
data = create_dummies(df,"label")


# In[65]:


data


# In[66]:


unscaled_feature = x_train
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[67]:


x_train_array = sc.fit_transform(x_train.values)
x_train = pd.DataFrame(x_train_array, index = x_train.index, columns = x_train.columns)
x_test_array = sc.transform(x_test.values)
x_test = pd.DataFrame(x_test_array, index = x_test.index, columns = x_test.columns)


# In[68]:


mlp = MLPClassifier(100, solver= 'sgd', learning_rate_init=0.01, max_iter = 10000)
mlp.fit(x_train, y_train)
mlp.score(x_test, y_test)


# In[69]:


y_pred = mlp.predict(x_test)
y_t_pred = mlp.predict(x_train)


# In[70]:


train_acc = []
test_acc = []
train_acc.append(metrics.accuracy_score(y_train, y_t_pred))
test_acc.append(metrics.accuracy_score(y_test, y_pred))
print(train_acc)
test_acc


# In[71]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(100, solver= 'sgd', learning_rate_init=0.01, max_iter = 15000)
mlp.fit(x_train, y_train)
mlp.score(x_test, y_test)


# In[72]:


train_acc = []
test_acc = []
train_acc.append(metrics.accuracy_score(y_train, y_t_pred))
test_acc.append(metrics.accuracy_score(y_test, y_pred))
print(train_acc)
test_acc


# In[73]:


from sklearn import metrics


# In[74]:


k_range = list(range(1,11))
train_acc = []
test_acc = []
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    y_t_pred = knn.predict(x_train)
    train_acc.append(metrics.accuracy_score(y_train, y_t_pred))
    test_acc.append(metrics.accuracy_score(y_test, y_pred))
print(train_acc)
print(max(test_acc))


# In[75]:


print(max(train_acc))

