#!/usr/bin/env python
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
#Hide warnings
import warnings
warnings.filterwarnings("ignore")


# In[41]:


df1 = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx", sheet_name = ['Data'])
print(df1)


# In[42]:


df = df.get('Data')
df


# In[43]:


df.shape


# In[44]:


df.describe


# In[45]:


df.duplicated().sum()


# In[46]:


df.isnull().sum()


# In[47]:


df.dtypes


# In[53]:


x = df.drop(['Personal Loan'], axis=1)
x.columns


# In[54]:


y = df['Personal Loan']


# In[55]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[56]:


model = sm.Logit(y_train, x_train).fit()
print(model.summary())


# In[66]:


model = LogisticRegression(solver="liblinear", random_state = 0).fit(x_train, y_train)
model.score(x_train, y_train)


# In[67]:


print('intercept ', model.intercept_[0])
print('classes', model.classes_)
pd.DataFrame({'coeff': model.coef_[0]}, 
             index=x.columns)

