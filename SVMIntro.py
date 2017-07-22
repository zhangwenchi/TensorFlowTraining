import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

train_indices = np.random.choice(
    len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(
    list(set(range(len(x_vals))) - set(train_indices)))
train_data = x_vals[train_indices]
train_label = y_vals[train_indices]
test_data = x_vals[test_indices]
test_label = y_vals[test_indices]

# for this is SVM
batch_size = 100

x_data = tf.placeholder(tf.float32, [None, 2])
y_target = tf.placeholder(tf.float32, [None, 1])

weight = tf.Variable(tf.truncated_normal([2, 1]))
bias = tf.Variable(tf.truncated_normal([1, 1]))

output = tf.subtract(tf.matmul(x_data, weight), bias)

l2_norm = tf.reduce_sum(tf.square(weight))

# margin parameter
alpha = tf.constant([0.1])

# not prediction, just how far from the correct answer
classification = tf.reduce_mean(
    tf.maximum(0., tf.subtract(1., tf.multiply(output, y_target))))
loss = tf.add(classification, tf.multiply(alpha, l2_norm))

prediction = tf.sign(output)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(prediction, y_target), tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
train_acc = []
test_acc = []
for i in range(10000):
    rand = np.random.choice(len(train_data), batch_size)
    rand_x = train_data[rand]
    rand_y = np.transpose([train_label[rand]])
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

    if (i + 1) % 100 == 0:
        print('Step #' + str(i + 1) + ' A = ' +
              str(sess.run(weight)) + 'b = ' + str(sess.run(bias)))

[[a1], [a2]] = sess.run(weight)
[[b]] = sess.run(bias)
slope = -a2 / a1
y_intercept = b / a1

x1_vals = [d[1] for d in x_vals]

best_fit = []
for i in x1_vals:
    best_fit.append(slope * i + y_intercept)

setosa_x = [d[1] for
            i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for
            i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1]
                for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0]
                for i, d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setosa_x, setosa_y, 'o', label='I.setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.plot(train_acc, 'k-', label='Training Accuracy')
plt.plot(test_acc, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()




