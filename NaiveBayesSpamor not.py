#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Naive Bayes Classifier
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("always")


# In[ ]:


#prepare data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
raw_data = urllib.request.urlopen(url)
#prepare our dataset,and seperated by ','
dataset = np.loadtxt(raw_data, delimiter= ',')
print(dataset[0])


# In[ ]:


#prepare X AND Y
X = dataset[:,0:48]
Y = dataset[:,-1]


# In[ ]:


#make the train test splits
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size =0.33,random_state =17)


# In[ ]:


#our data contain the frequency count of each word ,in the form of contineous value,so the Multinomial model is the best choice
#but first we check it on bernoulNB model,to convert the value into binary by binarize = True
#lets start
BerNB = BernoulliNB(binarize= True)
#fit the model
br=BerNB.fit(x_train,y_train)
y_expect = y_test
#make prediction
y_pred =br.predict(x_test)
#print the accuarcy
print(accuracy_score(y_expect,y_pred)*100)


# In[ ]:


#now multinomial model,because our features contain the contineous values
MulNB = MultinomialNB()
mb = MulNB.fit(x_train,y_train)
#make prediction
y_pred = mb.predict(x_test)
#accuracy score
print(accuracy_score(y_expect,y_pred)*100)
#so the multinomial produce the best results for us


# In[ ]:


#now check for the gaussian model because features are normally distributed
GauNB =GaussianNB()
gb = GauNB.fit(x_train,y_train)
#make prediction

y_pred = gb.predict(x_test)
#check the accuracy score
print(accuracy_score(y_expect,y_pred)*100)


# In[ ]:


#we can improve our binary model if we put the value binarize =0.1
BerNB = BernoulliNB(binarize= 0.1)
#fit the model
br=BerNB.fit(x_train,y_train)
y_expect = y_test
#make prediction
y_pred =br.predict(x_test)
#print the accuarcy
print(accuracy_score(y_expect,y_pred)*100)
#so if binariz =0.1 it will produce the good result then multinomial


# In[ ]:




