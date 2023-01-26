#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
#Hide warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics


# In[2]:


df= pd.read_csv("car.csv")
df


# In[3]:


df.shape


# In[4]:


print(df.describe)


# In[5]:


df.columns


# In[6]:


df.duplicated().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


from sklearn.preprocessing import OrdinalEncoder

categories = ["Horsepower"]
enc_oe = OrdinalEncoder()
enc_oe.fit(df[categories])

df[categories] = enc_oe.transform(df[categories])


# In[10]:


df


# In[11]:


x= df[["Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","US Made"]]
x


# In[12]:


y = df["MPG"]


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[14]:


y2 = sm.add_constant(y_train)
est = sm.OLS(y_train,x_train)
est2 = est.fit()
print(est2.summary())


# In[15]:


output = pd.DataFrame(est2.params).reset_index()
output.columns = ["features","coefficients"]
output["abs_coeff"] = abs(output['coefficients'])
output.sort_values(by='abs_coeff',ascending=False).head(15)


# In[16]:


ypred = est2.predict(x_test)
ypred


# In[17]:


def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


# In[18]:


MAPE(y_test, ypred)


# In[19]:


Acc = 100-MAPE(y_test, ypred)
Acc


# In[20]:


from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold


# In[21]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
cv


# In[22]:


model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
model


# In[23]:


model.fit(x, y)


# In[24]:


ypred = model.predict(x_test)
ypred


# In[25]:


def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


# In[26]:


MAPE(y_test, ypred)


# In[27]:


Acc = 100-MAPE(y_test, ypred)
Acc


# In[28]:


yactual = model.predict(x_train)
yactual


# In[29]:


MAPE(y_train, yactual)


# In[30]:


Acc = 100-MAPE(y_test, ypred)
Acc


# In[31]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score


# In[32]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[33]:


model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')


# In[34]:


model.fit(x, y)


# In[35]:


print(model.alpha_)


# In[36]:


ypred = model.predict(x_test)
ypred


# In[37]:


def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


# In[38]:


MAPE(y_test, ypred)


# In[39]:


Acc = 100-MAPE(y_test, ypred)
Acc


# In[40]:


yactual = model.predict(x_train)
yactual


# In[41]:


MAPE(y_train, yactual)


# In[42]:


Acc = 100-MAPE(y_test, ypred)
Acc

