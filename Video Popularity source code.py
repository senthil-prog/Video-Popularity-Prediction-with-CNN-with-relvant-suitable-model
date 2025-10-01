#!/usr/bin/env python
# coding: utf-8

#installing dependencies
import pandas as pd
import numpy as np
from numpy import savetxt

print("Loading training data...")
#importing training file
data = pd.read_csv("c:/Users/srija/Downloads/Video-Popularity-Prediction-main/Video-Popularity-Prediction-main/Testing and training files/train_meta_df.csv")

print("Data loaded successfully!")
print(f"Data shape: {data.shape}")
print("Columns in dataset:", data.columns.tolist())

#checking for null data issues
print("\nChecking for null values:")
print(data.isnull().sum().sum(), "total null values found")


# In[10]:


#importing LinearRegression model from sklearn library
from sklearn.linear_model import LinearRegression


# In[11]:

print("Processing data and removing outliers...")
min_threshold, max_threshold = data.average_d.quantile([0.01,0.99])
minv_threshold, maxv_threshold = data.Average.quantile([0.01,0.99])


# In[12]:

#removing outliers and unnecessary data to enhance model
data2 = data[(data.Average > minv_threshold) & (data.Average < maxv_threshold) & (data.average_d > min_threshold) & (data.average_d < max_threshold) ]
print(f"Data shape after outlier removal: {data2.shape}")


# In[13]:

print("\nPreparing training and test data...")
#removing features which are not required
X_train = data2.drop(['views','ad_blocked','partner_active','dayofweek','comp_id','average_t'], axis=1)
#uploading testing data set to get predictions
test = pd.read_csv("c:/Users/srija/Downloads/Video-Popularity-Prediction-main/Video-Popularity-Prediction-main/Testing and training files/public_meta_df.csv")
X_test = test.drop(columns=['ad_blocked','partner_active','dayofweek','comp_id','average_t'])


# In[14]:

Y_train = data2['views']


# In[15]:

print("Training the model...")
#calling the model
regr = LinearRegression()


# In[16]:

#training the model
regr.fit(X_train,Y_train)


# In[17]:

print("Making predictions...")
#making predictions
pred = regr.predict(X_test).astype(int)


# In[19]:

print("Post-processing predictions...")
#removing negative values and setting minimum number of views based on observation and 25% percentile of views in training dataset
for x in range(986):
    if pred[x-1] > -1:
        if pred[x-1] < 70:
            pred[x-1] = 70
    else:
        pred[x-1] = -pred[x-1]
        if (pred[x-1]) < 70:
            pred[x-1] = 70
            


# In[20]:

#final predictions
print("Predictions generated successfully!")
print("Sample predictions (first 10):")
print(pred[:10])


# In[34]:

print("\nSaving results to solution.csv...")
#saving to the output file in .csv format
#after this we convert it to the asked to solution format
dataset = pd.DataFrame({'comp_id':test.comp_id, 'views':pred})
dataset.to_csv('solution.csv', index = False)
print(f"Total predictions: {len(pred)}")
print("Solution saved to solution.csv")




