#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from scipy.interpolate import *
from matplotlib.pyplot import *
from sklearn.linear_model import LinearRegression
import pandas as pd
import sklearn
pd.set_option('display.max_columns', 999)
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_excel('RegressionDataset.xlsx', 'Sheet1')


# In[3]:


# df = df.head(1000)
# df = df.reset_index()


# In[4]:


X = df.sqft_living


# In[5]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, df.price, test_size=0.2, random_state = 5)


# In[6]:


print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)


# In[7]:


x_train_new= X_train.values.reshape(-1,1)
y_train_new= Y_train.values.reshape(-1,1)
x_test_new= X_test.values.reshape(-1,1)
y_test_new= Y_test.values.reshape(-1,1)


# In[8]:


linear_model = LinearRegression()


# In[9]:


linear_model.fit(x_train_new,y_train_new)


# In[10]:


print ('Estimated intercept coefficient:', linear_model.intercept_)


# In[11]:


pred = linear_model.predict(x_test_new)


# In[ ]:





# In[12]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - y_test_new)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - y_test_new) ** 2))
print("R2-score: %.2f" % r2_score(pred , y_test_new) )


# In[13]:


X = df.drop('price', axis = 1)
X = df.drop('date', axis = 1)


# In[14]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
X, df.price, test_size=0.2, random_state = 5)
# This creates a LinearRegression object
linear_model = LinearRegression()


# In[15]:


# X_train = X_train.apply(pd.to_numeric, errors='coerce')
# Y_train = Y_train.apply(pd.to_numeric, errors='coerce')
linear_model.fit(X_train,Y_train)


# In[16]:


pred = linear_model.predict(X_test)


# In[ ]:





# In[17]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - Y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - Y_test) ** 2))
print("R2-score: %.2f" % r2_score(pred , Y_test) )

