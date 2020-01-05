#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_excel('RegressionDataset.xlsx', 'Sheet1')
# data = df[df.sqft_living < 6000]
df.head()


# In[3]:


# Data cleansing
df.isnull().sum()


# In[4]:


# get_ipython().run_line_magic('matplotlib', 'inline')
d = df.filter(['sqft_living', 'price'])
plt.xlabel('living area (sq.ft)')
plt.ylabel('price (k)')
plt.scatter(d.sqft_living/100, d.price/1000, color='green', marker='+')


# In[5]:


# check for any correlations between variables
corr = df.corr()
sns.heatmap(corr)
# sqft_living, grade, sqft_above and sqft_living15 have a  high influence on price
# bathrooms, view, sqft_basement have weaker influence on price


# In[6]:


reg = linear_model.LinearRegression()


# In[7]:


reg.fit(d[['sqft_living']], d.price)


# In[8]:


print(reg.coef_)


# In[9]:


print(reg.intercept_)


# In[10]:


reg.predict([[1020]])


# In[11]:


reg.predict([[1180]])


# In[12]:


reg.predict([[1600]])


# In[ ]:




