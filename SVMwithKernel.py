# max(sumi(bi)) - 1/2 * sumi(sumj(yibi(xiyj)yjbj)), where
# sumi(biyi) = 0, 0<= bi <= 1/2ny

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()
# dist=[5.]
# x_data = [[1.,2.]]
# print(sess.run(tf.add(tf.subtract(
#     dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))),
#     tf.transpose(dist))))
(x_vals, y_vals) = datasets.make_circles(
    n_samples=500, factor=0.5, noise=0.1)
y_vals = np.array([1 if y==1 else -1 for y in y_vals])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

batch_size = 500
x_data = tf.placeholder(tf.float32, [None, 2])
y_target = tf.placeholder(tf.float32, [None, 1])
prediction_grid = tf.placeholder(tf.float32, [None, 2])
b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

# rbf
# exp(-gamma * ||x1-x2||^2)
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)  # 1 means add every row ans is 5 if x_data is [1, 2]
dist = tf.reshape(dist, [-1, 1])

# I think here I am right and the book is wrong, mine is RBF actually
sq_dist = tf.subtract(tf.multiply(2., dist), tf.matmul(x_data, tf.transpose(x_data)))
# sq_dist = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))

my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dist)))
# linear
# my_kernel = tf.matmul(x_data, tf.transpose(x_data))

# dual part
# max(sum(ai) - 1/2 sum(aiajyiyj <xi,xj>)), where sum(aiyi) = 0
# let the negative to be minimal == max
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
loss = tf.negative(tf.subtract(first_term, tf.multiply(1/2, second_term)))


rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(
    2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))


prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
batch_accuracy = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={
        x_data: rand_x, y_target: rand_y
    })

    temp_loss = sess.run(loss, feed_dict={
        x_data: rand_x, y_target: rand_y
    })
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={
        x_data: rand_x, y_target: rand_y, prediction_grid: rand_x
    })
    batch_accuracy.append(acc_temp)

    if (i + 1) % 100 == 0:
        print('Step:' + str(i+1))
        print('Loss = ' + str(temp_loss))

x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
[grid_predictions] = sess.run(prediction, feed_dict={
    x_data: rand_x, y_target: rand_y, prediction_grid: grid_points
})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('Gaussian SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()












