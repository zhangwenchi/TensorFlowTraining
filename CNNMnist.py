import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float', [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def con_2x2(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

w_con1 = weight_variable([5, 5, 1, 32])
b_con1 = bias_variable([32])
h_con1 = tf.nn.relu(con_2x2(x_image, w_con1) + b_con1)
h_pool1 = max_pool(h_con1)

w_con2 = weight_variable([5, 5, 32, 64])
b_con2 = bias_variable([64])
h_con2 = tf.nn.relu(con_2x2(h_pool1, w_con2) + b_con2)
h_pool2 = max_pool(h_con2)

w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
drop_prob = tf.placeholder('float')
drop = tf.nn.dropout(h_fc1, drop_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_con = tf.nn.softmax(tf.matmul(drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_con))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y_con, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        print('test %d is %g' % (i,
            accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], drop_prob: 1.0
            })
        ))
    train_step.run(feed_dict={
        x: batch[0], y_: batch[1], drop_prob: 0.5
    })

batch = mnist.test.next_batch(1000)
print('test accuracy is %g.' % (accuracy.eval(feed_dict={
    x: batch[0], y_: batch[1], drop_prob: 1.0
})))
