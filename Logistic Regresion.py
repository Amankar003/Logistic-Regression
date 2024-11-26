#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Importing Data

# In[4]:


df = pd.read_csv("Social_Network_Ads (1).csv")


# ### Performing EDA

# In[21]:


df.columns


# In[5]:


df.shape


# In[6]:


df.sample(5)


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# In[24]:


x = df.iloc[:,[2,3]].values


# In[25]:


y = df.iloc[:,4].values


# In[26]:


x


# In[27]:


y


# ### Train Test Split

# In[11]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state = 0)


# In[30]:


x_train


# In[31]:


x_test


# In[32]:


y_train


# In[33]:


y_test


# ### Apply Feature Scaling

# In[35]:


from sklearn.preprocessing import StandardScaler


# In[36]:


sc = StandardScaler()


# In[40]:


x_train = sc.fit_transform(x_train)


# In[41]:


x_test = sc.fit_transform(x_test)


# In[42]:


x_train


# In[43]:


x_test


# ### Making Logistic Regression Model

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[44]:


log_rg = LogisticRegression(random_state = 0)
log_rg.fit(x_train, y_train)

y_pred = log_rg.predict(x_test)


# In[45]:


y_pred


# In[38]:


y_test


# In[47]:


df.columns


# In[52]:


x_test[0]


# In[53]:


x_test[:,0]


# ### Visualize the predicted and tested Data

# In[55]:


plt.scatter(x_test[:,0], y_test, c = y_pred)


# ### Calculating accuracy and confusion matrix

# In[56]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[58]:


print("Accuracy : ", accuracy_score(y_test, y_pred))


# In[59]:


cf = confusion_matrix(y_test, y_pred)


# In[60]:


cf


# In[ ]:




