import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

train_indices = np.random.choice(
    len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(
    list(set(range(len(x_vals))) - set(train_indices)))
train_data = x_vals[train_indices]
train_label = y_vals[train_indices]
test_data = x_vals[test_indices]
test_label = y_vals[test_indices]

batch_size = 50

x_data = tf.placeholder(tf.float32, [None, 1])
y_target = tf.placeholder(tf.float32, [None, 1])

weight = tf.Variable(tf.truncated_normal([1, 1]))
bias = tf.Variable(tf.truncated_normal([1, 1]))

output = tf.add(tf.matmul(x_data, weight), bias)

epsilon = tf.constant([0.5])
loss = tf.reduce_mean(
    tf.maximum(0., tf.subtract(tf.abs(tf.subtract(output, y_target)), epsilon))
)


prediction = tf.sign(output)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(prediction, y_target), tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
for i in range(10000):
    rand = np.random.choice(len(train_data), batch_size)
    rand_x = np.transpose([train_data[rand]])
    rand_y = np.transpose([train_label[rand]])
    sess.run(train_step, feed_dict={
        x_data: rand_x, y_target: rand_y
    })

    loss_vec.append(sess.run(loss, feed_dict={
        x_data: rand_x, y_target: rand_y
    }))

    if (i + 1) % 100 == 0:
        print('Step #' + str(i + 1) + ' A = ' +
              str(sess.run(weight)) + 'b = ' + str(sess.run(bias)))

[[slope]] = sess.run(weight)
[[y_intercept]] = sess.run(bias)
[width] = sess.run(epsilon)

best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
    best_fit_upper.append(slope * i + y_intercept + width)
    best_fit_lower.append(slope * i + y_intercept - width)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()