#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


Train = pd.read_csv("Train.csv")
Train_Beneficiarydata = pd.read_csv("Train_Beneficiarydata.csv")
Train_Inpatientdata = pd.read_csv("Train_Inpatientdata.csv")
Train_Outpatientdata = pd.read_csv("Train_Outpatientdata.csv")


# In[3]:


Train_Allpatientdata=pd.merge(Train_Outpatientdata,Train_Inpatientdata,
                              left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'],
                              right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode']
                              ,how='outer')


# In[4]:


Train_Allpatientdata=pd.merge(Train_Allpatientdata,Train_Beneficiarydata,on="BeneID")


# In[5]:


Train_Allpatientdata=pd.merge(Train_Allpatientdata,Train,on="Provider")


# In[6]:


Train_Allpatientdata.shape


# In[7]:


Train_Allpatientdata.head(5)


# In[8]:


Train_Allpatientdata.info()


# In[9]:


a=(Train_Allpatientdata["AttendingPhysician"]==Train_Allpatientdata["OperatingPhysician"])
b=(Train_Allpatientdata["OperatingPhysician"]==Train_Allpatientdata["OtherPhysician"])
c=(Train_Allpatientdata["AttendingPhysician"]==Train_Allpatientdata["OtherPhysician"])

print(a.sum())
print(b.sum())
print(c.sum())
print( (a+b).sum() )


# In[10]:


def physician_same(row):
    atten_oper=row["AttendingPhysician"]==row["OperatingPhysician"]
    oper_other=row["OperatingPhysician"]==row["OtherPhysician"]
    atten_other=row["AttendingPhysician"]==row["OtherPhysician"]
    if atten_oper==True and oper_other==True:# atten = oper = other
        return 0
    elif atten_oper==True and oper_other==False:# atten = oper != other
        return 1
    elif atten_oper==False and oper_other==True:# atten != oper = other
        return 2
    else:# atten != oper != other
        return 3
    
phy_same=Train_Allpatientdata.apply(physician_same,axis=1)


# In[11]:


Train_Allpatientdata["phy_same"]=phy_same


# In[12]:


def physician_count(row,list_count):
    count=0
    for col in list_count:
        if pd.isnull(row[col]):
            continue
        else:
            count+=1
    return count
    
list_count=["AttendingPhysician","OperatingPhysician","OtherPhysician"]
phy_count=Train_Allpatientdata.apply(physician_count,axis=1,args=(list_count,))


# In[13]:


Train_Allpatientdata["phy_count"]=phy_count


# In[14]:


Train_Allpatientdata.head(4)


# In[15]:


startdate= pd.to_datetime( Train_Allpatientdata["ClaimStartDt"] )
enddate= pd.to_datetime( Train_Allpatientdata["ClaimEndDt"] )

period = ( enddate - startdate).dt.days
Train_Allpatientdata["period"] = period


# In[16]:


Train_Allpatientdata.info()


# In[17]:


copy1=Train_Allpatientdata.copy()


# In[18]:


cronic_cols_names=copy1.columns[ copy1.columns.str.startswith("ChronicCond") ]
cronic_cols=copy1[   cronic_cols_names   ]
cronic=cronic_cols.replace({2:0})
copy1[   cronic_cols_names   ]=cronic


# In[19]:


copy1["PotentialFraud"]=copy1["PotentialFraud"].replace({"Yes":1,"No":0})


# In[20]:


copy1["Gender"]=copy1["Gender"].replace({2:0})


# In[21]:


startadmt= pd.to_datetime( copy1["AdmissionDt"] )
enddatadmt= pd.to_datetime( copy1["DischargeDt"] )

periodadmt = ( enddatadmt - startadmt).dt.days
copy1["periodadmt"] = periodadmt
copy1["periodadmt"]=copy1["periodadmt"].fillna(0)


# In[22]:


copy1["RenalDiseaseIndicator"]=copy1["RenalDiseaseIndicator"].replace({"Y":1})


# In[23]:


birthdate=pd.to_datetime(copy1["DOB"])
enddate=pd.to_datetime(copy1["DOD"])

# cheack whether the patient dead or alife
def alife_function(value):
    if value==True:
        return 1
    else:
        return 0
alife = pd.isna(enddate).apply(alife_function)


# get the age of patient
max_date=enddate.dropna().max()
enddate[pd.isna(enddate)]=max_date
period=(((enddate-birthdate).dt.days/356).astype(int))

copy1["age"]=period
copy1["alife"]=alife


# In[24]:


copy1.head(2)


# In[25]:


copy1.info()


# In[26]:


def groupby(df,by,vars_to_group,methods,col_ident,as_index=True,agg=False):
    if agg:
        grouped=df.groupby(by=by,as_index=as_index)[vars_to_group].agg(methods)
        cols=['_'.join(col) for col in grouped.columns.values]
        cols=[col_ident+"_"+col for col in cols]
        grouped.columns=cols
        return grouped
    
    else:
        concat=df.groupby(by=by,as_index=as_index)[vars_to_group].transform(methods[0])
        cols=[ col_ident+"_"+col+"_"+methods[0] for col in concat.columns ]
        concat.columns=cols
        
        for method in methods[1:]:
            grouped=df.groupby(by=by,as_index=as_index)[vars_to_group].transform(method)
            cols=[col_ident+"_"+col+"_"+method for col in grouped.columns]
            grouped.columns=cols
            concat=pd.concat([concat,grouped],axis=1)
        
        return concat


# In[27]:


money_cols=["InscClaimAmtReimbursed","DeductibleAmtPaid","NoOfMonths_PartACov","NoOfMonths_PartBCov",
           "IPAnnualReimbursementAmt","IPAnnualDeductibleAmt","OPAnnualReimbursementAmt","OPAnnualDeductibleAmt"]


# In[28]:


provider_money=groupby(copy1,["Provider"],money_cols,["mean","std"],"provider",
                       True,False)


# In[29]:


banel_money=groupby(copy1,["BeneID"],money_cols,["mean","std"],"banel",
                       True,False)


# In[30]:


diag1_money=groupby(copy1,["ClmDiagnosisCode_1"],money_cols,["mean","std"],"diag1",
                       True,False)


# In[31]:


selected_cols_names=["phy_same","phy_count","period","periodadmt","age","alife","Provider","PotentialFraud"]
selected_cols=copy1[selected_cols_names]


# In[32]:


data=pd.concat([selected_cols,provider_money,banel_money,diag1_money],axis=1)


# In[33]:


grouped=data.groupby(by=["Provider","PotentialFraud"]).agg("mean").reset_index()
grouped


# In[34]:


grouped=grouped.fillna(0)


# In[35]:


features=grouped.iloc[:,2:]
labels=grouped.iloc[:,1]


# In[38]:


pip install imblearn


# In[39]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
features, labels = oversample.fit_resample(features, labels)


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
featuresstand=scaler.fit_transform(features)


# In[41]:


from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

ff=compute_class_weight(class_weight="balanced",classes=np.unique(labels),y=labels)
cw=dict(zip(np.unique(labels),ff))

featuress,labelss=shuffle(featuresstand,labels)
xtrain,xtest,ytrain,ytest = train_test_split(featuress,labelss,test_size=0.1)


# In[42]:


xtrain=xtrain.astype(np.float32)
xtest=xtest.astype(np.float32)
ytrain=ytrain.astype(np.float32).to_numpy()
ytest=ytest.astype(np.float32).to_numpy()


ytrain=ytrain.reshape(ytrain.shape+(1,))
ytest=ytest.reshape(ytest.shape+(1,))

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[43]:


import tensorflow as tf


# In[44]:


inpt=tf.keras.layers.Input((xtrain.shape[1]))
d1=tf.keras.layers.Dense(256, activation='relu')(inpt)
d1=tf.keras.layers.Dense(128, activation='relu')(d1)

d2=tf.keras.layers.Dense(1,activation="sigmoid")(d1)

nural_network=tf.keras.Model(inputs=inpt,outputs=d2)


# In[45]:


nural_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='binary_crossentropy'
                    ,metrics=["accuracy"])

history_combined=nural_network.fit(xtrain,ytrain,validation_data=(xtest,ytest),batch_size=128,epochs=300,
                                  class_weight=cw)

