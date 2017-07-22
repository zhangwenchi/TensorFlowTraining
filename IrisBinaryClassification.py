# just classify is Iris setosa or not

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf

sess = tf.Session()

iris = datasets.load_iris()
binary_target = np.array([1. if x == 0 else 0. for
                          x in iris.target])
iris_feature = np.array(iris.data)

batch_size = 20

x_data = tf.placeholder(tf.float32, shape=[None, 4])
y_target = tf.placeholder(tf.float32, shape=[None, 1])
A = tf.Variable(tf.truncated_normal(shape=[4, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
my_output = tf.add(tf.matmul(x_data, A), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

batch = []

for i in range(1000):
    # last 20 to be the test data
    rand_index = np.random.choice(len(iris_feature) - 20, size=batch_size)
    rand_x = iris_feature[rand_index]
    rand_y = [[y] for y in binary_target[rand_index]]

    sess.run(train_step, feed_dict={
        x_data: rand_x, y_target: rand_y
    })
    if (i + 1) % 100 == 0:
        # print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        # print('Step #' + str(i + 1) + ' b = ' + str(sess.run(b)))
        rand_test_index = np.random.choice(len(iris_feature))
        temp_loss = sess.run(loss, feed_dict={
            x_data: rand_x, y_target: rand_y})
        print('Step #' + str(i + 1) + ' Loss = ' + str(temp_loss))

error = 0.0

for i in range(len(iris_feature)):
    sum = sess.run(b)[0][0]
    for j in range(4):
        sum += sess.run(A)[j][0] * iris_feature[i][j]

    if (sum > 0) and binary_target[i] == 0.:
        error += 10
    elif (sum < 0) and binary_target[i] == 1.:
        error += 1

print('error rate is: %f' % (error / len(iris_feature)))
