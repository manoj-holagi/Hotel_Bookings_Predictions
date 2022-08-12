#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Importing Training Dataset

# In[2]:


data  = pd.read_csv("A:\\Data Sets\\train_data_evaluation_part_2.csv")
data.head()


# # DATA UNDERSTANDING & CLEANING

# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.describe(include='object')


# In[6]:


data.info()


# In[7]:


data.columns


# In[8]:


data_cat = data.select_dtypes(include = 'object')
data_cat = data_cat.columns.tolist()
print('Categorical :', data_cat, '\n')

data_num = data.select_dtypes(exclude= 'object')
data_num = data_num.columns.tolist()
print('Numerical :', data_num)


# In[9]:


for i in data_cat:
  print(f'{i}',data[i].unique(), '\n')


# In[10]:


for i in data_num:
  print(f'{i}',data[i].unique(), '\n')


# In[11]:


# dropping the duplicated rows
print('Data shape :', data.shape)

data.drop_duplicates(inplace = True)
print('Data shape after dropping duplicates :', data.shape)


# In[12]:


#checking null nalues

data.isnull().sum()


# In[13]:


#calculating null percentage for age column

(data.Age.isnull().sum()/len(data))*100

#There is only 4.5% of null values, so fill with median value of age column


# In[14]:


#filling null values
data["Age"].fillna(round(data["Age"].mean(),1), inplace = True)


# In[15]:


#checking again null nalues

data.isnull().sum()


# In[16]:


#considering all the values as 1 which are greater than 1,
data['BookingsCheckedIn'] = [1 if i>1 else i for i in data['BookingsCheckedIn'] ]


# In[17]:



for col in data.select_dtypes(include='object'):
    #if data[col].nunique() <=4:
    display(pd.crosstab(data['BookingsCheckedIn'], data[col], normalize='index'))


# In[18]:


# dropping columns which are not required 
data = data.drop(columns = ['Unnamed: 0'])
data = data.drop(columns = ['ID'])
data = data.drop(columns = ['Nationality'])


# # Exploratory Data Analysis

# In[19]:


data['DistributionChannel'].value_counts()


# In[20]:


plt.figure(figsize = (12, 5))
sns.countplot(x = 'DistributionChannel', data=data,
                order = data['DistributionChannel'].value_counts(ascending = True).index)
plt.xlabel('Type of DistributionChannel')
plt.title(f'Distribution of type of DistributionChannel')
sns.despine()


# In[21]:


data['MarketSegment'].value_counts()


# In[22]:


plt.figure(figsize = (12, 5))
sns.countplot(x = 'MarketSegment', data=data,
                order = data['DistributionChannel'].value_counts(ascending = True).index)
plt.xlabel('Type of MarketSegment')
plt.title(f'Distribution of type of MarketSegment')
sns.despine()


# In[23]:


data.corr()


# In[24]:


plt.figure(figsize = (25, 13))
dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
  
# displaying heatmap
plt.show()


# In[25]:


data.hist(figsize=(20,25))
plt.show()


# In[26]:


for col in data.select_dtypes(include='object'):
    #if data[col].nunique() <=6:
    g = sns.catplot(x = col, kind='count', col = 'BookingsCheckedIn', data=data, sharey=False)
    g.set_xticklabels(rotation=60)


# In[27]:


###########################################


# #### Plotting box plots to see outliers

# In[28]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["Age"])


# In[29]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["DaysSinceCreation"])


# In[30]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["AverageLeadTime"])


# In[31]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["LodgingRevenue"])


# In[32]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["OtherRevenue"])


# #### From the above boxplots we can find the outliers

# # Handling Outliers

# In[33]:


# From the above graphs we found outliers in Age,OtherRevenue, LodgingRevenue and AverageLeadTime columns 
#other than Age column all the columns are affecting to dataset so, we will remove them from dataset


# In[34]:


#AverageLeadTime
Q3 = 104
Q1 = 0
IQR = Q3-Q1
max = Q3 + 1.5*IQR
min = Q1 - 1.5*IQR
data = data[(data["AverageLeadTime"] < max) & (data["AverageLeadTime"] > min)]


# In[35]:


#LodgingRevenue
Q3 = 405
Q1 = 0
IQR = Q3-Q1
max = Q3 + 1.5*IQR
min = Q1 - 1.5*IQR
data = data[(data["LodgingRevenue"] < max) & (data["LodgingRevenue"] > min)]


# In[36]:


#OtherRevenue
Q3 =77
Q1 = 0
IQR = Q3-Q1
max = Q3 + 1.5*IQR
min = Q1 - 1.5*IQR
data = data[(data["OtherRevenue"] < max) & (data["OtherRevenue"] > min)]


# ### Boxplots after removing outliers

# In[37]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["AverageLeadTime"])


# In[38]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["LodgingRevenue"])


# In[39]:


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=data["OtherRevenue"])


# # Encoding (one Hot Encoding)

# In[40]:


data_cat = data.select_dtypes(include = 'object')
data_cat = data_cat.columns.tolist()
print('Categorical :', data_cat, '\n')


# In[41]:


dummies1 = pd.get_dummies(data["DistributionChannel"])
data= pd.concat([data,dummies1], axis=1)
data = data.drop('DistributionChannel',axis=1)


# In[42]:


dummies1 = pd.get_dummies(data["MarketSegment"])
data= pd.concat([data,dummies1], axis=1)
data = data.drop('MarketSegment',axis=1)


# In[43]:


#after encoding
data.head()


# In[44]:


data.shape


# In[137]:


plt.figure(figsize = (8, 5))
sns.countplot(x = 'BookingsCheckedIn', data=data,
                order = data['BookingsCheckedIn'].value_counts(ascending = True).index)
plt.xlabel('Type of MarketSegment')
plt.title(f'Distribution of type of BookingsCheckedIn')
sns.despine()


# In[46]:


data['BookingsCheckedIn'].value_counts()


# #### From the above, we conclude that the data is imbalaced data set, so we have balace it and feed to Ml models

# # Balancing the DATASET

# In[48]:


get_ipython().system('pip install imblearn ')


# In[49]:


data.columns


# In[50]:


data.shape


# In[51]:


X = data[['Age','DaysSinceCreation','AverageLeadTime','LodgingRevenue','OtherRevenue', 'BookingsCanceled','BookingsNoShowed','PersonsNights', 'RoomNights', 'DaysSinceLastStay','DaysSinceFirstStay','SRHighFloor','SRLowFloor','SRAccessibleRoom','SRMediumFloor','SRBathtub','SRShower','SRCrib','SRKingSizeBed','SRTwinBed','SRNearElevator','SRAwayFromElevator','SRNoAlcoholInMiniBar','SRQuietRoom','Corporate','Direct','Electronic Distribution','Travel Agent/Operator','Aviation','Complementary','Corporate','Direct','Groups','Other','Travel Agent/Operator']].values
y = data[["BookingsCheckedIn"]].values


# In[52]:


# train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[116]:


# scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train, y_train) # find mean and standard deviation

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


# # Synthetic Minority - Oversampling

# In[115]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X_train, y_train)


# In[55]:


X_sm.shape, y_sm.shape


# ## Oversampling followed by Undersampling

# In[56]:


from imblearn.combine import SMOTEENN
smt = SMOTEENN()
X_sn, y_sn = smt.fit_resample(X_train, y_train)


# In[57]:


X_sn.shape, y_sn.shape


# In[58]:


X_sn = pd.DataFrame(X_sn)
y_sn = pd.DataFrame(y_sn)


# # ML Models

# ## Logistic Regression

# ### SMOTE

# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

def LR(X, y):
  logistic = LogisticRegression()
  logistic.fit(X, y)

  y_pred = logistic.predict(X_test)
  print('Accuracy :', logistic.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))

  

LR(X_sm, y_sm)


# ### SMOTEENN

# In[60]:


LR(X_sn, y_sn)


# ## KNN Classifier

# ### SMOTE

# In[61]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

K = []

for i in [1,2,3,4,5,6,7,8,9,10,15,20,30]: # looping to find best K
    knn = KNeighborsClassifier(i) #initialising the model
    knn.fit(X_sm,y_sm) # training the model

    K.append({'K Value' : i, 'Cross_val_Score': np.mean(cross_val_score(knn, X_sm, y_sm, cv = 5)).round(5)})
    score = pd.DataFrame(K, columns = ['K Value', 'Cross_val_Score'])

score.sort_values(by = 'Cross_val_Score', ascending = False).head()


# In[62]:


def KNN(K, w, p):
  knn = KNeighborsClassifier(n_neighbors = K, weights = w, p = p)
  knn.fit(X_sm,y_sm)

  y_pred = knn.predict(X_test)

  print(w.upper(), f'at p = {p}:')
  print('Accuracy :', knn.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


KNN(1, 'distance', 1)


# In[63]:


KNN(1, 'distance', 2)


# In[64]:


KNN(1, 'uniform', 2)


# ### SMOTEENN

# In[65]:


K = []

for i in [1,2,3,4,5,6,7,8,9,10,15,20,30]: # looping to find best K
    knn = KNeighborsClassifier(i) #initialising the model
    knn.fit(X_sn,y_sn) # training the model

    K.append({'K Value' : i, 'Cross_val_Score': np.mean(cross_val_score(knn, X_sm, y_sm, cv = 5)).round(5)})
    score = pd.DataFrame(K, columns = ['K Value', 'Cross_val_Score'])

score.sort_values(by = 'Cross_val_Score', ascending = False).head()


# In[66]:


def KNN(K,w,p):
  knn = KNeighborsClassifier(n_neighbors=K, weights=w, p=p)
  knn.fit(X_sm,y_sm)

  y_pred = knn.predict(X_test)

  print(w.upper(), f'at p = {p}:')
  print('Accuracy :', knn.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))

  

KNN(1, 'distance', 1)


# In[67]:


KNN(1,'distance', 2)


# In[68]:


KNN(1, 'uniform', 1)


# ## Decision Tree Classifier

# ### SMOTE

# In[69]:


from sklearn.tree import DecisionTreeClassifier

DT = []

for depth in [1,2,3,4,5,6,7,8,9,10,20]:
  dt = DecisionTreeClassifier(max_depth=depth)
  dt.fit(X_sm, y_sm)
  valAccuracy = cross_val_score(dt, X_sm, y_sm, cv=10)
  DT.append({'Depth' : depth, 'Cross_val_Score': np.mean(valAccuracy).round(5)})
  score = pd.DataFrame(DT, columns = ['Depth', 'Cross_val_Score'])

score.sort_values(by = 'Cross_val_Score', ascending = False).head()


# In[70]:


def DT(i):
  dt = DecisionTreeClassifier(max_depth=i)
  dt.fit(X_sm,y_sm)

  y_pred = dt.predict(X_test)

  print('Accuracy :', dt.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))

  

DT(2)


# ### SMOTEENN

# In[71]:


from sklearn.tree import DecisionTreeClassifier

DT = []

for depth in [1,2,3,4,5,6,7,8,9,10,20]:
  dt = DecisionTreeClassifier(max_depth=depth)
  dt.fit(X_sn, y_sn)
  valAccuracy = cross_val_score(dt, X_sm, y_sm, cv=10)
  DT.append({'Depth' : depth, 'Cross_val_Score': np.mean(valAccuracy).round(5)})
  score = pd.DataFrame(DT, columns = ['Depth', 'Cross_val_Score'])

score.sort_values(by = 'Cross_val_Score', ascending = False).head()


# In[72]:


def DT(i):
  dt = DecisionTreeClassifier(max_depth=i)
  dt.fit(X_sm,y_sm)

  y_pred = dt.predict(X_test)

  print('Accuracy :', dt.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


DT(3)


# ## Random Forest Classifier

# ### SMOTE

# In[73]:


from sklearn.ensemble import RandomForestClassifier

depth = int(np.log(10)/np.log(2)) # log2(number of features)

rf = RandomForestClassifier(max_depth= depth, max_features = 'sqrt')
rf.fit(X_sm, y_sm)

y_pred = rf.predict(X_test)

print('Accuracy :', dt.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# ### SMOTEENN

# In[75]:


depth = int(np.log(10)/np.log(2)) # log2(number of features)

rf = RandomForestClassifier(max_depth= depth, max_features = 'sqrt')
rf.fit(X_sn, y_sn)

y_pred = rf.predict(X_test)

print('Accuracy :', dt.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# ## XG BOOST

# In[76]:


pip install xgboost


# ### SMOTE

# In[77]:


import xgboost as xgb

XG = []

for lr in [0.01,0.05,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.5,0.7,1]:
  XGB = xgb.XGBClassifier(learning_rate = lr, n_estimators=100, verbosity = 0) # initialise the model
  XGB.fit(X_sm,y_sm) #train the model
  print("Learning rate : ", lr," Cross-Val score : ", np.mean(cross_val_score(XGB, X_sm, y_sm, cv=10)).round(4))


# In[78]:


XGB = xgb.XGBClassifier(learning_rate = 1, n_estimators=100)
XGB.fit(X_sm,y_sm) #train the model

X_test = pd.DataFrame(X_test)
y_pred = XGB.predict(X_test)

print('Accuracy :', XGB.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# ### SMOTEENN

# In[79]:


for lr in [0.01,0.05,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.5,0.7,1]:
  XGB = xgb.XGBClassifier(learning_rate = lr, n_estimators=100, verbosity = 0) # initialise the model
  XGB.fit(X_sn,y_sn) #train the model
  print("Learning rate : ", lr," Cross-Val score : ", np.mean(cross_val_score(XGB, X_sn, y_sn, cv=10)).round(4))


# In[80]:


XGB = xgb.XGBClassifier(learning_rate = 1, n_estimators=100)
XGB.fit(X_sm,y_sm) #train the model

X_test = pd.DataFrame(X_test)
y_pred = XGB.predict(X_test)

print('Accuracy :', XGB.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# #### We have fed our training dataset to all the supervised ML algorithms, except KNN Classifier all the other models performed well
# 

# # ######################################################################

# # Importing Test Dataset

# In[119]:


df  = pd.read_csv("A:\\Data Sets\\test_data_evaluation_part2.csv")
df.head()


# # DATA CLEANING

# In[120]:


df.shape


# In[121]:


df.info()


# In[122]:


df.columns


# In[123]:


df.isnull().sum()


# In[124]:


#calculating null percentage for age column

df.Age.isnull().sum()/len(df)*100

#There is only 4.5% of null values, so fill with median value of age column


# In[125]:


#filling null values
df["Age"].fillna(round(df["Age"].mean(),1), inplace = True)


# In[126]:


# dropping the duplicated rows
print('Data shape :', df.shape)

df.drop_duplicates(inplace = True)
print('Data shape after dropping duplicates :', df.shape)


# In[127]:


#considering all the values as 1 which are greater than 1,
df['BookingsCheckedIn'] = [1 if i>1 else i for i in df['BookingsCheckedIn'] ]


# In[130]:


# dropping columns which are not required 
df = df.drop(columns = ['Unnamed: 0'])
df = df.drop(columns = ['ID'])
df = df.drop(columns = ['Nationality'])


# # Encoding (one Hot Encoding)

# In[131]:


dummies1 = pd.get_dummies(df["DistributionChannel"])
df= pd.concat([df,dummies1], axis=1)
df = df.drop('DistributionChannel',axis=1)


# In[134]:


dummies1 = pd.get_dummies(df["MarketSegment"])
df = pd.concat([df,dummies1], axis=1)
df = df.drop('MarketSegment',axis=1)


# In[ ]:


#################################################################################################################################


# In[138]:


plt.figure(figsize = (8, 5))
sns.countplot(x = 'BookingsCheckedIn', data=df,
                order = df['BookingsCheckedIn'].value_counts(ascending = True).index)
plt.xlabel('Type of BookingsCheckedIn')
plt.title(f'Distribution of type of BookingsCheckedIn')
sns.despine()


# #### The distribution of 1 and 0 are almost equal so there is no need to balance dataset

# # ML Models

# In[139]:


X = df[['Age','DaysSinceCreation','AverageLeadTime','LodgingRevenue','OtherRevenue', 'BookingsCanceled','BookingsNoShowed','PersonsNights', 'RoomNights', 'DaysSinceLastStay','DaysSinceFirstStay','SRHighFloor','SRLowFloor','SRAccessibleRoom','SRMediumFloor','SRBathtub','SRShower','SRCrib','SRKingSizeBed','SRTwinBed','SRNearElevator','SRAwayFromElevator','SRNoAlcoholInMiniBar','SRQuietRoom','Corporate','Direct','Electronic Distribution','Travel Agent/Operator','Aviation','Complementary','Corporate','Direct','Other','Travel Agent/Operator']].values
y = df[["BookingsCheckedIn"]].values


# In[140]:


# train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[141]:


# scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train, y_train) # find mean and standard deviation

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


# ## Logistic Regression

# In[142]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score


logistic = LogisticRegression()
logistic.fit(X_train,y_train)

y_pred = logistic.predict(X_test)
print('Accuracy :', logistic.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# ## KNN Classification

# In[147]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


K = []

for i in [1,2,3,4,5,6,7,8,9,10,15,20,30]: # looping to find best K
    knn = KNeighborsClassifier(i) #initialising the model
    knn.fit(X_train,y_train) # training the model

    K.append({'K Value' : i, 'Cross_val_Score': np.mean(cross_val_score(knn, X_sm, y_sm, cv = 5)).round(5)})
    score = pd.DataFrame(K, columns = ['K Value', 'Cross_val_Score'])

score.sort_values(by = 'Cross_val_Score', ascending = False).head()


# In[148]:


knn = KNeighborsClassifier(1)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)


print('Accuracy :', knn.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# ## Decison Tree Classifier

# In[149]:


from sklearn.tree import DecisionTreeClassifier

DT = []

for depth in [1,2,3,4,5,6,7,8,9,10,20]:
  dt = DecisionTreeClassifier(max_depth=depth)
  dt.fit(X_train,y_train)
  valAccuracy = cross_val_score(dt, X_sm, y_sm, cv=10)
  DT.append({'Depth' : depth, 'Cross_val_Score': np.mean(valAccuracy).round(5)})
  score = pd.DataFrame(DT, columns = ['Depth', 'Cross_val_Score'])

score.sort_values(by = 'Cross_val_Score', ascending = False).head()


# In[152]:


dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

print('Accuracy :', dt.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# ## Random Forest Classifier

# In[154]:


from sklearn.ensemble import RandomForestClassifier

depth = int(np.log(10)/np.log(2)) # log2(number of features)

rf = RandomForestClassifier(max_depth= depth, max_features = 'sqrt')
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

print('Accuracy :', dt.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# ## XG BOOST

# In[155]:


import xgboost as xgb

XG = []

for lr in [0.01,0.05,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.5,0.7,1]:
  XGB = xgb.XGBClassifier(learning_rate = lr, n_estimators=100, verbosity = 0) # initialise the model
  XGB.fit(X_train,y_train) #train the model
  print("Learning rate : ", lr," Cross-Val score : ", np.mean(cross_val_score(XGB, X_sm, y_sm, cv=10)).round(4))


# In[156]:


XGB = xgb.XGBClassifier(learning_rate = 1, n_estimators=100)
XGB.fit(X_train,y_train) #train the model

X_test = pd.DataFrame(X_test)
y_pred = XGB.predict(X_test)

print('Accuracy :', XGB.score(X_test,y_test).round(3),
        '\nf1-score :', f1_score(y_test, y_pred).round(3),
        '\nAUROC :', roc_auc_score(y_test, y_pred).round(3))


# In[ ]:




