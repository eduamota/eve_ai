# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:03:20 2019

@author: aquic
"""

#%%

import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.python.framework import ops
import warnings

#%%
# stop printing the warning produced by TensorFlow
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

#%%
# create the TensorFlow session for the graph
sess = tf.Session()

#%%
# set the RNN parameters
epochs = 300
batch_size = 250
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0001
dropout_keep_prob = tf.placeholder(tf.float32)

#%%
# Let's manually download the dataset and store it in a text_data.txt file in the temp directory
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
#%%
# directly download the dataset in zipped format
if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    text_data = file.decode()
    text_data = text_data.encode('ascii',errors='ignore')
    text_data = text_data.decode().split('\n')
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]
    
#%%
# Let's split the words that have a word length of at least 2
text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

#%%
# create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)
    
#%%
# call the preceding method to clean the text
text_data_train = [clean_text(x) for x in text_data_train]

#%%
# creating word embedding – changing text into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))    

#%%
# let's shuffle to make the dataset balance
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

#%%
# split the data into a training and testing set
ix_cutoff = int(len(y_shuffled)*0.75)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary size: {:d}".format(vocab_size))
print("Training set size: {:d}".format(len(y_train)))
print("Test set size: {:d}".format(len(y_test)))

#%%
# let's create placeholders for our TensorFlow graph
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

#%%
# Let's create the embedding
embedding_mat = tf.get_variable("embedding_mat", shape=[vocab_size, embedding_size], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

#%%
# time to construct our RNN. The following code defines the RNN cell
cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

#%%
# Let's define the way to get the output from our RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

#%%
# we define the weights and the biases for the RNN
weight = bias = tf.get_variable("weight", shape=[rnn_size, 2], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)
bias = tf.get_variable("bias", shape=[2], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)

#%%
# It uses both the weight and the bias from the preceding code
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

#%%
# we define the losses for each prediction so that later on, they can contribute to the loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)

#%%
# define the loss function
loss = tf.reduce_mean(losses)

#%%
# Now define the accuracy of each prediction
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

#%%
# create the training_op with RMSPropOptimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

#%%
# let's initialize all the variables using the global_variables_initializer() method
init_op = tf.global_variables_initializer()
sess.run(init_op)

#%%
# we can create some empty lists to keep track of the training loss, testing loss, training accuracy, and the testing accuracy in each epoch
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

#%%

shuffled_ix = np.random.permutation(np.arange(len(x_train)))
x_train = x_train[shuffled_ix]
y_train = y_train[shuffled_ix]
num_batches = int(len(x_train)/batch_size) + 1
for epoch in range(epochs):
    for i in range(num_batches):
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_acc = sess.run([loss,accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('\nEpoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
    print('Overall accuracy on test set (%): {}'.format(np.mean(temp_test_acc)*100.0))

#%%

epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('RNN training/test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

#%%

plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

#%%

saver = tf.train.Saver()

#%%
save_path = saver.save(sess, 'model/spam.ckpt')
print('Model saved to {}'.format(save_path))

#%%

tf.get_variable_scope().reuse_variables()
saver.restore(sess, 'model/spam.ckpt')

#%%

text = '''
Beloved in Christ.
 
Greetings my dear, I am writing this mail to you with heavy tears in my eyes and great sorrow in my heart. As I informed you earlier, I am (Mrs.)Sandra Sharon Bricks from American and a widow to late Mr.Bricks, I am 63 years old, suffering from long time Cancer of the breast. From all indications my condition is really deteriorating and it's quite obvious that I won't live more than 2 months according to my doctors.
 
I have some funds I inherited from my late loving husband Mr.Bricks a huge sum , deposited in a Finance Company . I need a very honest,reliable and God fearing person that can use these funds for Charity work, helping the Less Privileges, and 40% of this money will be for your time and expenses, while 60% goes to charities.
Please let me know if I can TRUST YOU ON THIS to carry out this favor for me. I look forward to your prompt reply for more details .
 
Yours sincerely
Mrs.Sandra Bricks
9100 K Street, NW
Suite 101,Washington, DC 69110, USA
'''

#%%
# call the preceding method to clean the text
new_data = [clean_text(text),]

#%%
new_processed = np.array(list(vocab_processor.fit_transform(new_data)))    

#%%
new_dict = {x_data: new_processed, dropout_keep_prob:1.0}
result = sess.run(logits_out, feed_dict=new_dict)

    #%%

if result[0][0] > result[0][1]:
    print("Spam")
else:
    print("Valid")

#%%
sess.close()