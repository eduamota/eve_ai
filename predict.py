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

data_modified = data_modified.drop_duplicates()


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
#data_modified = data_modified.drop('last_login_age', axis=1)

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
data_final.fillna(0, inplace=True)

#%%

data_final['cellNumber'] = data_final['cellNumber'].apply(lambda x: 'Valid' if x and len(x) > 5 else "Invalid")

#%%

data_final['phoneNumber'] = data_final['phoneNumber'].apply(lambda x: 'Valid' if x and len(x) > 5 else "Invalid")


#%%

accountType = pd.get_dummies(data_final['accountType'])
balance_currency = pd.get_dummies(data_final['balance_currency'])
bank_account = pd.get_dummies(data_final['bank_account'])
card_account = pd.get_dummies(data_final['card_account'])
cellNumber = pd.get_dummies(data_final['cellNumber'])
city = pd.get_dummies(data_final['city'])
country = pd.get_dummies(data_final['country'])
domain = pd.get_dummies(data_final['domain'])
dormancy = pd.get_dummies(data_final['dormancy'])
gender = pd.get_dummies(data_final['gender'])
invalidSin = pd.get_dummies(data_final['invalidSin'])
kyc_account = pd.get_dummies(data_final['kyc_account'])
kyc_level = pd.get_dummies(data_final['kyc_level'])
language = pd.get_dummies(data_final['language'])
last_load_amount = pd.get_dummies(data_final['last_load_amount'])
phoneNumber = pd.get_dummies(data_final['phoneNumber'])
postcode = pd.get_dummies(data_final['postcode'])
status = pd.get_dummies(data_final['status'])
street = pd.get_dummies(data_final['street'])

#%%

data_final['contacted'] = data_final['contacted_age'].apply(lambda x: 1 if x != 'None'and x>0 else 0)


#%%

data_final['contacted'].unique()
#%%

data_final = data_final.drop('contacted_age', axis=1)
#%%

var_names = data_final.columns.tolist()
categs = ['accountType','balance_currency','bank_account','card_account','cellNumber','city','country','domain','dormancy','gender','invalidSin','kyc_account','kyc_level','language','phoneNumber','postcode','status','street']

#%%

quantit = [i for i in var_names if i not in categs]


#%%

label = data_final['contacted']
df_numerical = data_final[quantit]
df_names = df_numerical .keys().tolist()

#%%

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df_numerical)
df_temp = pd.DataFrame(x_scaled)
df_temp.columns = df_names

#%%

normalized_df = pd.concat([df_temp,
                      accountType,
balance_currency,
bank_account,
card_account,
cellNumber,
city,
country,
domain,
dormancy,
gender,
invalidSin,
kyc_account,
kyc_level,
language,
phoneNumber,
postcode,
status,
street,
                      label], axis=1)
    
#%%
    
normalized_df.to_csv('accounts_normalized.csv', index = False)

#%%

import tensorflow as tf
import pandas as pd
import numpy as np
import os
#from sklearn.cross_validation import train_test_split # for random split of train/test
#%%

FILE_PATH = 'accounts_normalized.csv'         # Path to .csv dataset 
raw_data = pd.read_csv(FILE_PATH)        # Open raw .csv
print("Raw data loaded successfully...\n")

#%%

raw_data = normalized_df

#%%

type(raw_data.contacted)
#%%

Y_LABEL = 'contacted'    # Name of the variable to be predicted
KEYS = [i for i in raw_data.keys().tolist() if i != Y_LABEL]# Name of predictors
N_INSTANCES = raw_data.shape[0]       # Number of instances
N_INPUT = raw_data.shape[1] - 1            # Input size
N_CLASSES = raw_data.contacted.unique().shape[0] # Number of classes
TEST_SIZE = 0.25         # Test set size (% of dataset)
TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))  # Train size

#%%

print("Variables loaded successfully...\n")
print("Number of predictors \t%s" %(N_INPUT))
print("Number of classes \t%s" %(N_CLASSES))
print("Number of instances \t%s" %(N_INSTANCES))
print("\n")

#%%

LEARNING_RATE = 0.001     # learning rate
TRAINING_EPOCHS = 1000     # number of training epoch for the forward pass
BATCH_SIZE = 100     # batch size to be used during training
DISPLAY_STEP = 20   # print the error etc. at each 20 step
HIDDEN_SIZE = 256   # number of neurons in each hidden layer
# We use tanh as the activation function, but you can try using ReLU as well
ACTIVATION_FUNCTION_OUT = tf.nn.tanh
STDDEV = 0.1        # Standard Deviations
RANDOM_STATE = 100

#%%

data = raw_data[KEYS].get_values()       # X data
labels = raw_data[Y_LABEL].get_values()  # y data

#%%

labels_ = np.zeros((N_INSTANCES, N_CLASSES))
labels_[np.arange(N_INSTANCES), labels] = 1

#%%

