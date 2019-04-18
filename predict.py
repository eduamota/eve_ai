# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:41:59 2019

@author: edmota
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import os
import pytz
#%%

os.chdir("C:\\Users\\edmota\\Documents\\Projects\\Eve")

#%%
with open('data.pickle', 'rb') as f: 
    data = pickle.load(f)

with open('data_all_info.pickle', 'rb') as f: 
    data_all = pickle.load(f)

#%%

data_dt = pd.DataFrame(data)

data_all_dt = pd.DataFrame(data_all)

#%%

data = pd.concat([data_dt, data_all_dt])

#%%

data = data.drop('acceptsemails', axis=1)


#%%
data_modified = data.copy()


#%%
data_modified['contact_date'] = data_modified['contact_date'].astype(str).str[:-6]

#%%

data_modified['contact_date'] = pd.to_datetime(data_modified['contact_date'])

#%%

data_modified.dtypes

#%%

data_modified['contacted_age'] = data_modified['contact_date'] - data_modified['last_load']
data_modified = data_modified.drop('contact_date', axis=1)

#%%

data_modified['created_age'] = data_modified['last_load'] - data_modified['created_date']
data_modified = data_modified.drop('created_date', axis=1)

#%%

data_modified['age'] = data_modified['last_load'] - data_modified['dateofbirth']
data_modified = data_modified.drop('dateofbirth', axis=1)

#%%

data_modified['lastdormancy_age'] = data_modified['last_load'] - data_modified['last_dormancy']
data_modified = data_modified.drop('last_dormancy', axis=1)

#%%

#data_modified['last_login_age'] = data_modified['last_load'] - data_modified['last_login_time']
data_modified = data_modified.drop('last_login_time', axis=1)
data_modified = data_modified.drop('last_login_age', axis=1)

#%%

data_modified['lastupdated_age'] = data_modified['last_load'] - data_modified['lastupdated']
data_modified = data_modified.drop('lastupdated', axis=1)

#%%

data_modified['status_age'] = data_modified['last_load'] - data_modified['status_date']
data_modified = data_modified.drop('status_date', axis=1)

#%%

data_modified['status_age'] = data_modified[:5]['status_age'] / np.timedelta64(1, 's')

#%%

data_modified['lastupdated_age'] = data_modified[:5]['lastupdated_age'] / np.timedelta64(1, 's')

#%%

data_modified['lastdormancy_age'] = data_modified[:5]['lastdormancy_age'] / np.timedelta64(1, 's')

#%%

data_modified['age'] = data_modified[:5]['age'] / np.timedelta64(1, 's')

#%%

data_modified['created_age'] = data_modified[:5]['created_age'] / np.timedelta64(1, 's')

#%%

data_modified['contacted_age'] = data_modified[:5]['contacted_age'] / np.timedelta64(1, 's')

#%%

data_final = data_modified.drop('last_load', axis=1)

#%%

var_names = data_final.columns.tolist()

#%%

categs = ['accountType','balance_currency','bank_account','card_account','cellNumber','city','country','domain','dormancy','gender','invalidSin','kyc_account','kyc_level','language', 'last_load_amount','phoneNumber','postcode','status','street']

#%%

quantit = [i for i in var_names if i not in categs]

#%%

accountType = pd.get_dummies(data_final['accountType'])
balance_currency = pd.get_dummies(data_final['balance_currency'])
bank_account = pd.get_dummies(data_final['bank_account'])
card_account = pd.get_dummies(data_final['card_account'])
housing = pd.get_dummies(data_final['housing'])
loan = pd.get_dummies(data_final['loan'])
contact = pd.get_dummies(data_final['contact'])
month = pd.get_dummies(data_final['month'])
day = pd.get_dummies(data_final['day_of_week'])
duration = pd.get_dummies(data_final['duration'])
poutcome = pd.get_dummies(data_final['poutcome'])