#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#Hide warnings
import warnings
warnings.filterwarnings("ignore")


# In[4]:


df = pd.read_csv('adult-all.csv', header=None, names =["Age","Workclass","fnlwgt","Education","education-num","MaritalStatus","Occupation","Relationship","Race","Sex","CapitalGain","CapitalLoss","Hrs/week","NativeCountry","Salary"])


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.describe


# In[8]:


df.drop(['fnlwgt','NativeCountry','Relationship'], axis=1)


# In[9]:


df.replace('?',np.nan,inplace=True)
df.isnull().sum()


# In[10]:


df.describe().T


# In[11]:


df["Workclass"] = df["Workclass"].astype("category")
df["Education"] = df["Education"].astype("category")
df["MaritalStatus"] = df["MaritalStatus"].astype("category")
df["Relationship"] = df["Relationship"].astype("category")
df["Race"] = df["Race"].astype("category")
df["Sex"] = df["Sex"].astype("category")
df["NativeCountry"] = df["NativeCountry"].astype("category")
df["Salary"] = df["Salary"].astype("category")
df["Occupation"] = df["Occupation"].astype("category")


# In[12]:


df.dtypes


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop_duplicates()


# In[15]:


# Taking a look at the target (income) 
print(f"Ratio above 50k : {(df['Salary'] == '>50K').astype('int').sum() / df.shape[0] * 100 :.2f}%")


# In[17]:


num_feat = df.select_dtypes(include=['int64']).columns
num_feat


# In[18]:


sns.countplot(df.Salary)


# In[19]:


x = df[['Age','Workclass','Education','education-num','MaritalStatus','Occupation','Race','Sex','CapitalGain','CapitalLoss','Hrs/week']]
y=df[['Salary']]


# In[20]:


def OneHotEncoding(df, enc, categories):  
  transformed = pd.DataFrame(enc.transform(df[categories]).toarray(), columns=enc.get_feature_names_out(categories))
  return pd.concat([df.reset_index(drop=True), transformed], axis=1).drop(categories, axis=1)

categories = ["Workclass", "MaritalStatus", "Occupation", "Race"]
enc_ohe = OneHotEncoder()
enc_ohe.fit(x[categories])

x = OneHotEncoding(x, enc_ohe, categories)


# In[21]:


from sklearn.preprocessing import OrdinalEncoder

categories = ['Sex', 'Education']
enc_oe = OrdinalEncoder()
enc_oe.fit(x[categories])

x[categories] = enc_oe.transform(x[categories])


# In[22]:


x.dtypes


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[24]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[25]:


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
print(test_acc)


# In[26]:


max(test_acc)


# In[27]:


# Feature Scaling to avoid optimization and calculation error
cat_cols = x.columns[x.dtypes == 'object']
num_cols = x.columns[(x.dtypes == 'float64') | (x.dtypes == 'int64')]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train[num_cols])
x_train[num_cols] = scaler.transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

x_train.head()


# In[28]:


from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results 
def print_grid_search_metrics(gs):
    print ("Best score: " + str(gs.best_score_))
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(best_parameters.keys()):
        print(param_name + ':' + str(best_parameters[param_name]))


# In[30]:


x_test.head()


# In[31]:


parameters = {
    'n_neighbors':[1,2,3,4,5,6,7,8,9]
}
Grid_KNN = GridSearchCV(KNeighborsClassifier(),parameters, cv=9)
Grid_KNN.fit(x_train, y_train)


# In[32]:


Grid_KNN.best_params_


# In[33]:


print_grid_search_metrics(Grid_KNN)


# In[34]:


best_KNN_model = Grid_KNN.best_estimator_
best_KNN_model.predict(x_test)
best_KNN_model.score(x_test, y_test)

