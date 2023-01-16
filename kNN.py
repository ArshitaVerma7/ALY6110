#!/usr/bin/env python
# coding: utf-8

# In[133]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# In[134]:


df = pd.read_csv('Iris.csv')
df.head()


# In[200]:


rows = len(df.axes[0])
print(rows)
cols = len(df.axes[1])
print(cols)


# In[135]:


pd.to_numeric(df.Id)
pd.to_numeric(df.SepalLengthCm)
pd.to_numeric(df.SepalWidthCm)
pd.to_numeric(df.PetalLengthCm)
pd.to_numeric(df.PetalWidthCm)


# In[136]:


df.columns


# In[137]:


X= df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y= df[['Species']]


# In[158]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[139]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[140]:


ax = sns.heatmap(df.corr(), annot=True)


# In[159]:


k_range = list(range(1,50))
train_acc = []
test_acc = []
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    y_t_pred = knn.predict(x_train)
    train_acc.append(metrics.accuracy_score(y_train, y_t_pred))
    test_acc.append(metrics.accuracy_score(y_test, y_pred))
print(train_acc)
test_acc


# In[165]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[160]:


df_test = x_test.copy()
df_test['actual'] = y_test
df_test['prediction'] = knn.predict(x_test)


# In[170]:


y_test_pred =  knn.predict(x_test)


# In[171]:


confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)


# In[172]:


confusion_matrix


# In[175]:


cm_df =  pd.DataFrame(confusion_matrix,
                     index = ['SETOSA','VERSICOLr','VIRGINICA'],
                     columns = ['SETOSA','VERSICOLr','VIRGINICA'])


# In[176]:


plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[180]:


y_test.value_counts()


# In[191]:


plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
fg = ax.scatter3D(df['SepalLengthCm'], df['SepalWidthCm'], df['PetalWidthCm'],  c='g')
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('SepalWidthCm')
ax.set_zlabel('PetalWidthCm')
plt.colorbar(fg)

