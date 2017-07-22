import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import requests


seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

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
y_vals = np.array([x[0] for x in birth_data])

# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

batch_size = 50

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

train_data = x_vals[train_indices]
train_labels = y_vals[train_indices]
test_data = x_vals[test_indices]
test_labels = y_vals[test_indices]


def normalize(x):
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min) / (x_max - x_min)

train_data = np.nan_to_num(normalize(train_data))
test_data = np.nan_to_num(normalize(test_data))

sess = tf.Session()

x_data = tf.placeholder(tf.float32, [None, 7])
y_target = tf.placeholder(tf.float32, [None, 1])


def init_variable(x):
    return tf.Variable(tf.random_normal(shape=x, stddev=1.0))


def get_output(layer, weight, bias, activation=True):
    out = tf.add(tf.matmul(layer, weight), bias)
    if activation:
        return tf.nn.sigmoid(out)
    else:
        return out

weight1 = init_variable([7, 14])
bias1 = init_variable([14])
hidden_layer1 = get_output(x_data, weight1, bias1)

weight2 = init_variable([14, 5])
bias2 = init_variable([5])
hidden_layer2 = get_output(hidden_layer1, weight2, bias2)

weight3 = init_variable([5, 1])
bias3 = init_variable([1])
output = get_output(hidden_layer2, weight3, bias3)

loss = tf.reduce_mean(tf.abs(y_target - output))

train_step = tf.train.GradientDescentOptimizer(0.009).minimize(loss)

prediction = tf.round(output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))


sess.run(tf.global_variables_initializer())

loss_vec = []
train_acc = []
test_acc = []
for i in range(1500):
    rand_index = np.random.choice(len(train_labels), batch_size)
    rand_x = train_data[rand_index]
    rand_y = np.transpose([train_labels[rand_index]])

    sess.run(train_step, feed_dict={
        x_data: rand_x, y_target: rand_y
    })

    temp_loss = sess.run(loss, feed_dict={
        x_data: rand_x, y_target: rand_y
    })
    loss_vec.append(temp_loss)

    train_acc.append(sess.run(accuracy, feed_dict={
        x_data: train_data, y_target: np.transpose([train_labels])
    }))

    test_acc.append(sess.run(accuracy, feed_dict={
        x_data: test_data, y_target: np.transpose([test_labels])
    }))

    if i % 50 == 0:
        print("Loss is :", temp_loss)

plt.plot(loss_vec, 'k-')
plt.title('loss vector')
plt.xlabel('generator')
plt.ylabel('value')
plt.show()

plt.plot(train_acc, 'k-', label='train_accuracy')
plt.plot(test_acc, 'r--', label='test_accuracy')
plt.title('Accuracy')
plt.xlabel('generation')
plt.ylabel('value')
plt.legend(loc='lower right')
plt.show()

