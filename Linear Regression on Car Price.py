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


df= pd.read_csv("CarPrice.csv")
df


# In[3]:


df.describe()


# In[4]:


df.duplicated().sum()


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


df.carbody.unique()


# In[8]:


df.shape


# In[9]:


cor = df.corr()
cor


# In[10]:


corplot = sns.heatmap(cor)


# In[11]:


df['CarBrand'] = df['CarName'].apply(lambda x: x.split(' ')[0])


# In[12]:


x= df.drop(['car_ID','symboling','carheight','stroke','compressionratio','peakrpm','CarName','price'], axis=1) 
y= df['price']
x.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder,MinMaxScaler, OneHotEncoder
def OneHotEncoding(df1, enc, categories):  
  transformed = pd.DataFrame(enc.transform(df1[categories]).toarray(), columns=enc.get_feature_names_out(categories))
  return pd.concat([df1.reset_index(drop=True), transformed], axis=1).drop(categories, axis=1)

categories = ["CarBrand", "fueltype", "aspiration", "doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"]
enc_ohe = OneHotEncoder()
enc_ohe.fit(x[categories])

x = OneHotEncoding(x, enc_ohe, categories)


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[15]:


y2 = sm.add_constant(y_train)
est = sm.OLS(y_train,x_train)
est2 = est.fit()
print(est2.summary())


# In[16]:


output = pd.DataFrame(est2.params).reset_index()
output.columns = ["features","coefficients"]
output["abs_coeff"] = abs(output['coefficients'])
output.sort_values(by='abs_coeff',ascending=False).head(15)


# In[17]:


y_pred = est2.predict(x_test)


# In[18]:


def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


# In[19]:


MAPE(y_test, y_pred)


# In[20]:


Acc = 100-MAPE(y_test, y_pred)
Acc

