import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


def gen_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def gen_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = gen_weight([5, 5, 1, 16])
b_conv1 = gen_bias([16])
h_conv1 = tf.nn.relu(tf.add(conv_2d(x_image, W_conv1), b_conv1))
h_pool1 = max_pool(h_conv1)

W_conv2 = gen_weight([5, 5, 16, 32])
b_conv2 = gen_bias([32])
h_conv2 = tf.nn.relu(tf.add(conv_2d(h_pool1, W_conv2), b_conv2))
h_pool2 = max_pool(h_conv2)

W_conv3 = gen_weight([5, 5, 32, 64])
b_conv3 = gen_bias([64])
h_conv3 = tf.nn.relu(tf.add(conv_2d(h_pool2, W_conv3), b_conv3))
h_pool3 = max_pool(h_conv3)

W_fc1 = gen_weight([4*4*64, 1024])  # this should change if add layers, based on the max_pool & conv
b_fc1 = gen_bias([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])  # this should change if add layers, based on the max_pool & conv
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, W_fc1), b_fc1))

keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = gen_weight([1024, 10])
b_fc2 = gen_bias([10])

y_conv = tf.nn.softmax(tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2))

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

test_vec = []
train_vec = []
for i in range(500):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 0.5
    })
    if (i + 1) % 50 == 0:
        temp = sess.run(accuracy, feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0
        })
        train_vec.append(temp)
        print('step %d: Train Accuracy is :%f' % (i+1, temp))
        test_batch = mnist.test.next_batch(500)
        temp = sess.run(accuracy, feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0
        })
        test_vec.append(temp)
        print('step %d: Test Accuracy is :%f' % (i+1, temp))

plt.title('Accuracy')
plt.xlabel('generation')
plt.ylabel('accuracy rate')
plt.plot(train_vec, 'k-', label='train')
plt.plot(test_vec, 'r--', label='test')
plt.legend()
plt.show()

# two conv one full connection
# 32 -> 64 -> 1024, AdamOptimizer(1e-4), train_step: 500,
# step 500: Train Accuracy is :0.900000
# step 500: Test Accuracy is :0.938000

# 3 conv one full connection
# 32 -> 64 -> 128 -> 1024, AdamOptimizer(1e-2), train_step: 500,
# step 500: Train Accuracy is :0.100000
# step 500: Test Accuracy is :0.100000
# The vanishing gradient problem, if more layers, they cannot learn more
# from the above layer, so they are just the random initial value
# and it's slow

# one conv one full connection
# 32 -> 1024, AdamOptimizer(1e-4), train_step: 500,
# step 500: Train Accuracy is :0.920000
# step 500: Test Accuracy is :0.916000
# fast and pretty good

# 3 conv one full connection
# 16 -> 32 -> 64 -> 1024, AdamOptimizer(1e-4), train_step: 500,
# step 500: Train Accuracy is :0.940000
# step 500: Test Accuracy is :0.934000
# I cannot believe that this is so good,
# so deeper is useful, but should change the amount of conv
