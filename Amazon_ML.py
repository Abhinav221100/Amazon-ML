#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


# # Pre-processing

# In[2]:


train = pd.read_csv("/home/abhi2018/Amazon ML/data/dataset/train.csv",escapechar = "\\",quoting = 3)
test = pd.read_csv("/home/abhi2018/Amazon ML/data/dataset/test.csv",escapechar = "\\",quoting = 3)


# In[3]:


print(train.shape)
train.head()


# In[4]:


train.nunique()


# In[5]:


print(test.shape)
test.head()


# ## Removing null values

# In[6]:


train_set = train.dropna()


# In[7]:


train_set.nunique()


# In[8]:


print(train_set.shape)
train_set.head()


# In[7]:


test_set = test.dropna()


# In[8]:


print(test_set.shape)
test_set.head()


# ## Train set

# In[9]:


#x_train = train_set.drop(columns=['BROWSE_NODE_ID'])
x= train_set['TITLE']
y = train_set['BROWSE_NODE_ID']


# In[10]:


x.shape


# In[11]:


x_train,y_train = x[0:5000],y[0:5000]
x_test,y_test = x[5000:10000],y[5000:10000]


# In[12]:


x_train.head()


# In[13]:


y_train.head()


# ## Feature Extraction

# In[14]:


cv = CountVectorizer()
features = cv.fit_transform(x_train)


# In[15]:


features.shape


# In[ ]:


count = 0
for i in x_train:
    print(i)
    count+=1
    if(count == 10):
        break


# # Model

# In[16]:


model = svm.SVC()


# In[17]:


model.fit(features, y_train)


# In[18]:


features_test = cv.transform(x_test)


# In[19]:


print("Accuracy of the model is :",model.score(features_test,y_test))


# In[ ]:




