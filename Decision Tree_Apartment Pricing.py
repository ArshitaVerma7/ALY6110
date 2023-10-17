#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
pd.set_option('display.max_rows', 10)

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('Apartment_Building_Evaluation.csv')


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[6]:


object_cols = df.select_dtypes(include='object').columns
print("\nObject Columns:\n",object_cols)


# In[7]:


print(df.isnull().sum().sort_values(ascending=False))
pd.set_option('display.max_rows', None)


# In[8]:


df.duplicated().sum()


# In[9]:


#fill the na value with median and mean
df['OTHER_FACILITIES']=df['OTHER_FACILITIES'].fillna(df['OTHER_FACILITIES'].mean(),axis=0)
df['STORAGE_AREAS_LOCKERS']=df['STORAGE_AREAS_LOCKERS'].fillna(df['STORAGE_AREAS_LOCKERS'].mean(),axis=0)
df['GARBAGE_CHUTE_ROOMS']=df['GARBAGE_CHUTE_ROOMS'].fillna(df['GARBAGE_CHUTE_ROOMS'].mean(),axis=0)
df['ELEVATORS']=df['ELEVATORS'].fillna(df['ELEVATORS'].mean(),axis=0)
df['BALCONY_GUARDS']=df['BALCONY_GUARDS'].fillna(df['BALCONY_GUARDS'].mean(),axis=0)
df['YEAR_EVALUATED']=df['YEAR_EVALUATED'].fillna(df['YEAR_EVALUATED'].mean(),axis=0)
df['PARKING_AREA']=df['PARKING_AREA'].fillna(df['PARKING_AREA'].mean(),axis=0)
df['LAUNDRY_ROOMS']=df['LAUNDRY_ROOMS'].fillna(df['LAUNDRY_ROOMS'].mean(),axis=0)
df['YEAR_REGISTERED']=df['YEAR_REGISTERED'].fillna(df['YEAR_REGISTERED'].mean(),axis=0)
df['LONGITUDE']=df['LONGITUDE'].fillna(df['LONGITUDE'].mean(),axis=0)
df['LATITUDE']=df['LATITUDE'].fillna(df['LATITUDE'].mean(),axis=0)
df['Y']=df['Y'].fillna(df['Y'].mean(),axis=0)
df['X']=df['X'].fillna(df['X'].mean(),axis=0)
df['YEAR_BUILT']=df['YEAR_BUILT'].fillna(df['YEAR_BUILT'].median(),axis=0)
df['GRAFFITI']=df['GRAFFITI'].fillna(df['GRAFFITI'].mean(),axis=0)
df['EXTERIOR_GROUNDS']=df['EXTERIOR_GROUNDS'].fillna(df['EXTERIOR_GROUNDS'].mean(),axis=0)
df['GARBAGE_BIN_STORAGE_AREA']=df['GARBAGE_BIN_STORAGE_AREA'].fillna(df['GARBAGE_BIN_STORAGE_AREA'].mean(),axis=0)
df['EXTERIOR_CLADDING']=df['EXTERIOR_CLADDING'].fillna(df['EXTERIOR_CLADDING'].mean(),axis=0)
df['SECURITY']=df['SECURITY'].fillna(df['SECURITY'].mean(),axis=0)
df['EXTERIOR_WALKWAYS']=df['EXTERIOR_WALKWAYS'].fillna(df['EXTERIOR_WALKWAYS'].mean(),axis=0)
df['WATER_PEN_EXT_BLDG_ELEMENTS']=df['WATER_PEN_EXT_BLDG_ELEMENTS'].fillna(df['WATER_PEN_EXT_BLDG_ELEMENTS'].mean(),axis=0)
df['INTERNAL_GUARDS_HANDRAILS']=df['INTERNAL_GUARDS_HANDRAILS'].fillna(df['INTERNAL_GUARDS_HANDRAILS'].mean(),axis=0)
df['STAIRWELLS']=df['STAIRWELLS'].fillna(df['STAIRWELLS'].mean(),axis=0)
df['INTERIOR_WALL_CEILING_FLOOR']=df['INTERIOR_WALL_CEILING_FLOOR'].fillna(df['INTERIOR_WALL_CEILING_FLOOR'].mean(),axis=0)
df['ENTRANCE_LOBBY']=df['ENTRANCE_LOBBY'].fillna(df['ENTRANCE_LOBBY'].mean(),axis=0)
df['ENTRANCE_DOORS_WINDOWS']=df['ENTRANCE_DOORS_WINDOWS'].fillna(df['ENTRANCE_DOORS_WINDOWS'].mean(),axis=0)
df['INTERIOR_LIGHTING_LEVELS']=df['INTERIOR_LIGHTING_LEVELS'].fillna(df['INTERIOR_LIGHTING_LEVELS'].mean(),axis=0)
df.isnull().sum().sort_values(ascending=False)


# In[10]:


df["PROPERTY_TYPE"].value_counts()


# In[11]:


"RESULTs_OF_SCORE" in df.columns


# In[12]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
label=le.fit_transform(df['PROPERTY_TYPE'])
label
df.drop('PROPERTY_TYPE',inplace=True, axis=1)
df['PROPERTY_TYPE']=label

le=LabelEncoder()
label=le.fit_transform(df['WARDNAME'])
label
df.drop('WARDNAME',inplace= True, axis=1)
df['WARDNAME']=label


# In[17]:


df.dtypes


# In[13]:


object_cols = df.select_dtypes(include='object').columns
print("\nObject Columns:\n",object_cols)


# In[14]:


print(df.head)
pd.set_option('display.max_rows', 10)


# In[16]:


y = df[["RESULTS_OF_SCORE"]]


# In[17]:


x = df.drop(['RESULTS_OF_SCORE','_id','SITE_ADDRESS','EVALUATION_COMPLETED_ON','GRID'], axis =1)


# In[18]:


from sklearn.preprocessing import OneHotEncoder
def OneHotEncoding(y, enc, categories):  
  transformed = pd.DataFrame(enc.transform(y[categories]).toarray(), columns=enc.get_feature_names_out(categories))
  return pd.concat([y.reset_index(drop=True), transformed], axis=1).drop(categories, axis=1)

categories = ["RESULTS_OF_SCORE"]
enc_ohe = OneHotEncoder()
enc_ohe.fit(y[categories])

y = OneHotEncoding(y, enc_ohe, categories)


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)


# In[20]:


treemodel = DecisionTreeClassifier(max_depth = 3)
treemodel.fit(x_train, y_train)


# In[21]:


from sklearn.model_selection import GridSearchCV


# In[22]:


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[23]:


grid_search = GridSearchCV(estimator=treemodel, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "recall")


# In[24]:


grid_search.fit(x_train, y_train)


# In[25]:


plt.figure(figsize=(25,10))
plot = tree.plot_tree(treemodel, feature_names = x.columns.values.tolist(), class_names = ['0','1'], filled = True, fontsize = 14)


# In[26]:


model = treemodel.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("accuracy", metrics.accuracy_score(y_test, y_pred))


# In[27]:


dt_best = grid_search.best_estimator_
dt_best


# In[28]:


y_pred = grid_search.predict(x_test)
print("accuracy", metrics.accuracy_score(y_test, y_pred))

