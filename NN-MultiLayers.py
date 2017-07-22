import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import requests
from tensorflow.python.framework import ops

# name of data file
birth_weight_file = 'birth_weight.csv'

# download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = birth_data[0].split('\t')
    birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
    with open(birth_weight_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(birth_data)
        f.close()

# read birth weight data into memory
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = ['LOW','AGE'	,'LWT','RACE','SMOKE','PTL','HT','UI','BWT']
    for row in csv_reader:
        birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]


# Extract y-target (birth weight)
y_vals = np.array([x[8] for x in birth_data])

# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])


# reset the graph for new run
ops.reset_default_graph()

# Create graph session
sess = tf.Session()

# set batch size for training
batch_size = 100

# make results reproducible
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize(x_vals_train))
x_vals_test = np.nan_to_num(normalize(x_vals_test))


def init_weight(shape, stddev):
    return tf.Variable(tf.truncated_normal(shape,stddev))


def init_bias(shape):
    return tf.Variable(tf.random_normal(shape))


def full_connected(hidden_layer, weight, bias):
    return tf.nn.relu(tf.add(tf.matmul(hidden_layer, weight), bias))


x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

layer1_weight = init_weight([7,25], 10.0)
layer1_bias = init_bias([25])
layer1 = full_connected(x_data, layer1_weight, layer1_bias)

layer2_weight = init_weight([25,10], 10.0)
layer2_bias = init_bias([10])
layer2 = full_connected(layer1, layer2_weight, layer2_bias)

layer3_weight = init_weight([10,3], 3.0)
layer3_bias = init_bias([3])
layer3 = full_connected(layer2, layer3_weight, layer3_bias)

layer4_weight = init_weight([3,1], 1.0)
layer4_bias = init_bias([1])
layer4 = full_connected(layer3, layer4_weight, layer4_bias)

loss = tf.reduce_mean(tf.abs(y_target - layer4))

train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

sess.run(tf.global_variables_initializer())


loss_vec = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={
        x_data: rand_x, y_target: rand_y
    })

    temp_loss = sess.run(loss, feed_dict={
        x_data: rand_x, y_target: rand_y
    })
    loss_vec.append(temp_loss)

    temp_test_loss = sess.run(loss, feed_dict={
        x_data: x_vals_test, y_target: np.transpose([y_vals_test])
    })
    test_loss.append(temp_test_loss)

    if (i + 1) % 25 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

