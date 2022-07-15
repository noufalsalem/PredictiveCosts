#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_excel('data.xlsx')


# In[4]:


df = df.dropna()


# In[6]:


#df.columns


# In[21]:


flag = True
while flag:
    mode = input('Please Select the mode : \n\nEnter 1 to forcast a single column \nEnter 2 to forcast all the columns:    ')
    if (int(mode) != 1 and int(mode) != 2):
        print(int(mode))
        print('\nError!! You can enter either 1 or 2 only')
    else:
        flag = False


# In[22]:


def show_columns(columns):
    index = 1
    for col in columns:
        print('{:>40}'.format(col) + '          ' + str(index))
        index+=1


# In[23]:


def create_dataframe(col):
    return df[col]


# In[24]:


def get_result(series):
    # split into train and test sets
    train = series.values
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    return yhat


# In[36]:


def get_complete_result():
    data = df.drop(['Month'], axis=1)
    data.index = df.Month

    model = VAR(endog=data)
    model_fit = model.fit()

    # make prediction on validation
    prediction = model_fit.forecast(model_fit.y, steps=1)
    result = pd.Series(prediction[0] , index = data.columns)
    return result


# In[37]:


if mode == '1':
    show_columns(df.columns)
    col = input('Enter the coumn number which you want to predict : ')
    column = df.columns[int(col) - 1]
    new_df = create_dataframe(column)
    forcast = get_result(new_df)
    result = pd.Series(forcast , index = [column] )
    print("\n" + 'Forcast for the next month\n')
    print(result)

else:
    result = get_complete_result()
    print(result)

