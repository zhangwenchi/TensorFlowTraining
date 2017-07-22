import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import normalize
import os
import csv
import requests
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('birth_weight.csv'):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = [x for x in birth_data[0].split() if len(x) > 1]
    birth_data = [[float(x) for x in y.split() if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
    with open('birth_weight.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(birth_data)
        f.close()

birth_data = []
with open('birth_weight.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]
x_vals = np.array([x[2:9] for x in birth_data])
y_vals = np.array([x[0] for x in birth_data])

# set for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

train_data = x_vals[train_indices]
train_label = y_vals[train_indices]
test_data = x_vals[test_indices]
test_label = y_vals[test_indices]

# important for normalize
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

train_data = np.nan_to_num(normalize_cols(train_data))
test_data = np.nan_to_num(normalize_cols(test_data))

sess = tf.Session()

batch_size = 25

x_data = tf.placeholder(tf.float32, [None, 7])
y_target = tf.placeholder(tf.float32, [None, 1])

weights = tf.Variable(tf.random_normal([7, 1]))
bias = tf.Variable(tf.random_normal([1, 1]))

outputs = tf.add(tf.matmul(x_data, weights), bias)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=outputs))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

prediction = tf.round(tf.sigmoid(outputs))
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_target, prediction), tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
train_acc = []
test_acc = []
for i in range(10000):
    rand_indices = np.random.choice(len(train_data), batch_size)
    rand_x = train_data[rand_indices]
    rand_y = np.transpose([train_label[rand_indices]])

    sess.run(train_step, feed_dict={
        x_data: rand_x, y_target: rand_y
    })

    loss_vec.append(sess.run(loss, feed_dict={
        x_data: rand_x, y_target: rand_y
    }))
    train_acc.append(sess.run(accuracy, feed_dict={
        x_data: train_data, y_target: np.transpose([train_label])
    }))
    test_acc.append(sess.run(accuracy, feed_dict={
        x_data: test_data, y_target: np.transpose([test_label])
    }))

plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

