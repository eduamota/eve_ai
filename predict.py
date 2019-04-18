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
categs = ['accountType','balance_currency','bank_account','card_account','cellNumber','city','country','domain','dormancy','gender','invalidSin','kyc_account','kyc_level','language','phoneNumber','postcode','status','street', 'contacted']

#%%

quantit = [i for i in var_names if i not in categs]


#%%

label = data_final['contacted']
df_numerical = data_final[quantit]
df_names = df_numerical.keys().tolist()

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
normalized_df.fillna(0, inplace=True)
normalized_df.to_csv('accounts_normalized.csv', index = False)

#%%
label.unique()

#%%

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split # for random split of train/test
#%%

FILE_PATH = 'accounts_normalized.csv'         # Path to .csv dataset 
raw_data = pd.read_csv(FILE_PATH)        # Open raw .csv
print("Raw data loaded successfully...\n")

#%%

raw_data['contacted'] = raw_data['contacted'].astype(int)

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

data_train, data_test, labels_train, labels_test = train_test_split(data,labels_,test_size = TEST_SIZE,random_state = RANDOM_STATE)
print("Data loaded and splitted successfully...\n")

#%%

n_input = N_INPUT                   # input n labels 
n_hidden_1 = HIDDEN_SIZE            # 1st layer 
n_hidden_2 = HIDDEN_SIZE            # 2nd layer
n_hidden_3 = HIDDEN_SIZE            # 3rd layer 
n_hidden_4 = HIDDEN_SIZE            # 4th layer 
n_classes = N_CLASSES               # output m classes 

#%%

    # input shape is None * number of input
X = tf.placeholder(tf.float32, [None, n_input])

#%%

# label shape is None * number of classes
y = tf.placeholder(tf.float32, [None, n_classes])

#%%

dropout_keep_prob = tf.placeholder(tf.float32)

#%%

def DeepMLPClassifier(_X, _weights, _biases, dropout_keep_prob):
    layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1'])), dropout_keep_prob)
    layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2'])), dropout_keep_prob)
    layer3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer2, _weights['w3']), _biases['b3'])), dropout_keep_prob)
    layer4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer3, _weights['w4']), _biases['b4'])), dropout_keep_prob)
    out = ACTIVATION_FUNCTION_OUT(tf.add(tf.matmul(layer4, _weights['out']), _biases['out']))
    return out

#%%

weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=STDDEV)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=STDDEV)),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=STDDEV)),
    'w4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=STDDEV)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes],stddev=STDDEV)),   
}
biases = { 
    'b1': tf.Variable(tf.random_normal([n_hidden_1])), 
    'b2': tf.Variable(tf.random_normal([n_hidden_2])), 
    'b3': tf.Variable(tf.random_normal([n_hidden_3])), 
    'b4': tf.Variable(tf.random_normal([n_hidden_4])), 
    'out': tf.Variable(tf.random_normal([n_classes])) 
}

#%%

pred = DeepMLPClassifier(X, weights, biases, dropout_keep_prob)

#%%

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

# Optimization op (backprop)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

#%%

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Deep MLP networks has been built successfully...")
print("Starting training...")

#%%

init_op = tf.global_variables_initializer() 

#%%

sess = tf.Session()
sess.run(init_op)

#%%

for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.0
    total_batch = int(data_train.shape[0] / BATCH_SIZE)
    # Loop over all batches
    for i in range(total_batch):
        randidx = np.random.randint(int(TRAIN_SIZE), size = BATCH_SIZE)
        batch_xs = data_train[randidx, :]
        batch_ys = labels_train[randidx, :]
        # Fit using batched data
        sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 0.9})
        # Calculate average cost
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})/total_batch
    # Display progress
    if epoch % DISPLAY_STEP == 0:
        print("Epoch: %3d/%3d cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})
        print("Training accuracy: %.3f" % (train_acc))
print("Your MLP model has been trained successfully.")

#%%

# Plot loss over time
plt.subplot(221)
plt.plot(i_data, cost_list, 'k--', label='Training loss', linewidth=1.0)
plt.title('Cross entropy loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Cross entropy loss')
plt.legend(loc='upper right')
plt.grid(True)

#%%
                   
# Plot train and test accuracy
plt.subplot(222)
plt.plot(i_data, acc_list, 'r--', label='Accuracy on the training set', linewidth=1.0)
plt.title('Accuracy on the training set')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()