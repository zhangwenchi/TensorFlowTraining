import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

summary_writer = tf.summary.FileWriter('tensorboard', tf.get_default_graph())

if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
print('Running a slowed down linear regression. '
      'Run the command: $tensorboard --logdir="tensorboard"  '
      ' Then navigate to http://127.0.0.0:6006')

time.sleep(3)

batch_size = 50
generations = 100

x_data = np.arange(1000) / 10.
true_slope = 2.
y_data = x_data * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)

train_ix = np.random.choice(len(x_data), size=int(len(x_data) * 0.9), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)
x_data_train, y_data_train = x_data[train_ix], y_data[train_ix]
x_data_test, y_data_test = x_data[test_ix], y_data[test_ix]

x_graph_input = tf.placeholder(tf.float32, [None])
y_graph_input = tf.placeholder(tf.float32, [None])

m = tf.Variable(tf.random_normal([1], dtype=tf.float32), name='Slope')

output = tf.multiply(m, x_graph_input, name='Batch_Multiplication')

residuals = output - y_graph_input
l1_loss = tf.reduce_mean(tf.abs(residuals), name='L1_Loss')

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(l1_loss)

with tf.name_scope('Slope_Estimate'):
    tf.summary.scalar('Slope_Estimate', tf.squeeze(m))

with tf.name_scope('Loss_and_Residuals'):
    tf.summary.histogram('Histogram_Errors', l1_loss)
    tf.summary.histogram('Histogram_Residuals', residuals)

summary_op = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

for i in range(generations):
    batch_indices = np.random.choice(len(x_data_train), size=batch_size)
    x_batch = x_data_train[batch_indices]
    y_batch = y_data_train[batch_indices]
    _, train_loss, summary = sess.run([train_step, l1_loss, summary_op],
                                      feed_dict={x_graph_input: x_batch,
                                                 y_graph_input: y_batch})

    test_loss, test_resids = sess.run([l1_loss, residuals], feed_dict={x_graph_input: x_data_test,
                                                                       y_graph_input: y_data_test})

    if (i + 1) % 10 == 0:
        print('Generation {} of {}. Train Loss: {:.3}, Test Loss: {:.3}.'.format(i + 1, generations, train_loss,
                                                                                 test_loss))

    log_writer = tf.summary.FileWriter('tensorboard')
    log_writer.add_summary(summary, i)
    time.sleep(0.5)


# Create a function to save a protobuf bytes version of the graph
def gen_linear_plot(slope):
    linear_prediction = x_data * slope
    plt.plot(x_data, y_data, 'b.', label='data')
    plt.plot(x_data, linear_prediction, 'r-', linewidth=3, label='predicted line')
    plt.legend(loc='upper left')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


# Add image to tensorboard (plot the linear fit!)
slope = sess.run(m)
plot_buf = gen_linear_plot(slope[0])
# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# Add the batch dimension
image = tf.expand_dims(image, 0)
# Add image summary
image_summary_op = tf.summary.image("Linear Plot", image)
image_summary = sess.run(image_summary_op)
log_writer.add_summary(image_summary, i)
log_writer.close()

