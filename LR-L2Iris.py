import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

learning_rate = 0.05
batch_size = 25
x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

output = tf.add(tf.matmul(x_data, A), b)
loss = tf.reduce_mean(tf.square(y_target - output))
# loss = tf.reduce_mean(tf.abs(y_target - output))
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)


loss_vec = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={
        x_data: rand_x, y_target: rand_y
    })
    temp_loss = sess.run(loss, feed_dict={
        x_data: rand_x, y_target: rand_y
    })
    loss_vec.append(temp_loss)
    if (i + 1) % 100 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A))
              + 'b = ' + str(sess.run(b)))
        print('Loss = ''' + str(temp_loss))

[slope] = sess.run(A)
[y_intercept] = sess.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line',linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()



