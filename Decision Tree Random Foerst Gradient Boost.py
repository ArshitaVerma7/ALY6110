#!/usr/bin/env python
# coding: utf-8

# In[25]:


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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_rows', 10)

import warnings
warnings.filterwarnings("ignore")


# In[27]:


df= pd.read_csv("Nashville_housing_data.csv")
df.head()


# In[3]:


df.dtypes


# In[4]:


df.shape


# In[28]:


df.columns


# In[6]:


object_cols = df.select_dtypes(include='object').columns
print("\nObject Columns:\n",object_cols)


# In[48]:


df['Property City'].ffill()


# In[49]:


print(df['Property City'].isnull().sum())
df['Property City']=df['Property City'].ffill()
print(df['Property City'].isnull().sum())


# In[8]:


df.duplicated().sum()


# In[50]:


#fill the na value with median and mean
df['Half Bath']=df['Half Bath'].fillna(df['Half Bath'].median(),axis=0)
df['Full Bath']=df['Full Bath'].fillna(df['Full Bath'].median(),axis=0)
df['Bedrooms']=df['Bedrooms'].fillna(df['Bedrooms'].median(),axis=0)
df['Foundation Type']=df['Foundation Type'].fillna(df['Foundation Type'].mode(),axis=0)
df['Property City']=df['Property City'].ffill()
df['Finished Area']=df['Finished Area'].fillna(df['Finished Area'].mean(),axis=0)
df['Building Value']=df['Building Value'].fillna(df['Building Value'].mean(),axis=0)
df.isnull().sum()


# In[37]:


df["Sale Date"]= pd.to_datetime(df["Sale Date"])
df['Year'] = df['Sale Date'].dt.year 


# In[62]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
label=le.fit_transform(df['Foundation Type'])
label
df.drop('Foundation Type',axis=1)
df['Foundation Type']=label

label1=le.fit_transform(df['Sale Price Compared To Value'])
df.drop('Sale Price Compared To Value',axis=1)
df['Sale Price Compared To Value']=label1

label2=le.fit_transform(df['Land Use'])
df.drop('Land Use',axis=1)
df['Land Use']=label2

label3=le.fit_transform(df['Sold As Vacant'])
df.drop('Sold As Vacant',axis=1)
df['Sold As Vacant']=label3

label4=le.fit_transform(df['Multiple Parcels Involved in Sale'])
df.drop('Multiple Parcels Involved in Sale',axis=1)
df['Multiple Parcels Involved in Sale']=label4

label5=le.fit_transform(df['Exterior Wall'])
df.drop('Exterior Wall',axis=1)
df['Exterior Wall']=label5

label6=le.fit_transform(df['Grade'])
df.drop('Grade',axis=1)
df['Grade']=label6

df.head()


# In[63]:


y = df[['Sale Price Compared To Value']]
x = df.drop(['Sale Date',"Sale Price Compared To Value",'Unnamed: 0','Parcel ID','Property Address','Legal Reference','State', "Suite/ Condo   #"], axis =1)


# In[64]:


x.isnull().sum()


# In[65]:


x.select_dtypes(include='object').columns


# In[66]:


from sklearn.preprocessing import OneHotEncoder
def OneHotEncoding(df, enc, categories):  
  transformed = pd.DataFrame(enc.transform(df[categories]).toarray(), columns=enc.get_feature_names_out(categories))
  return pd.concat([df.reset_index(drop=True), transformed], axis=1).drop(categories, axis=1)

categories = ["Property City","City","Tax District","Land Use"]
enc_ohe = OneHotEncoder()
enc_ohe.fit(x[categories])

x = OneHotEncoding(x, enc_ohe, categories)


# In[67]:


x.dtypes


# In[68]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)


# In[69]:


treemodel = DecisionTreeClassifier(max_depth = 3)
treemodel.fit(x_train, y_train)


# In[70]:


plt.figure(figsize=(25,10))
plot = tree.plot_tree(treemodel, feature_names = x.columns.values.tolist(), class_names = ['0','1'], filled = True, fontsize = 14)


# In[71]:


model = treemodel.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("accuracy", metrics.accuracy_score(y_test, y_pred))


# In[81]:


from sklearn.linear_model import LogisticRegression


# In[82]:


logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(x_train, y_train)


# In[83]:


y_pred = logreg.predict(x_test)
y_pred


# In[91]:


print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[76]:


rfclass = RandomForestClassifier(n_estimators= 20, random_state = 0)
rfclass.fit(x_train, y_train)
rfypred = rfclass.predict(x_test)


# In[77]:


rfclass.score(x_test,y_test)


# In[78]:


gbclass = GradientBoostingClassifier(n_estimators = 20, random_state = None)
gbclass.fit(x_train, y_train)
gbypred = gbclass.predict(x_test)


# In[79]:


gbclass.score(x_test, y_test)

