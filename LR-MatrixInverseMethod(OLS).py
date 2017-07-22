# y = b1 + b2x2 + b3x3 + ... + bkxk + u
# u is the error
# target is minimize the u^2
# every y has a u, so target is minimize the sum(u^2)
# u is (y - bx), u^2 in matrix is u'*u
# so minimize the (y - bx)'(y - bx) = y'y - 2b'x'y + b'x'xb
# differential above is that 2 * sum(Yi - b1 - b2x2 - ... - bkxk)(-1) = 0
# => nb1 + b2*sum(x2i) + ... + bk * sum(xki) = sum(yi)
# => b1*sum(x2i) + ... + bk * sum(x2i * xki) = sum(x2i * yi)
# => ...
# => b1*sum(xki) + ... + bk * sum(xki * xki) = sum(xki * yi)
# and the matrix is x'x * b = x'y
# all we want to know is b, the coefficient, b = (x'x).inverse() * x'y
# x is [[1,1,1,1...],[x1,x2,...]]
# so beautiful @_@        -- Wenchi

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

X = np.column_stack((np.transpose(x_vals), np.transpose(np.matrix(np.repeat(1, 100)))))
y = np.transpose(np.matrix(y_vals))

x_ten = tf.constant(X)
y_ten = tf.constant(y)

x_tenT = tf.transpose(x_ten)
x_tenM = tf.matmul(x_tenT, x_ten)
x_tenI = tf.matrix_inverse(x_tenM)
product = tf.matmul(x_tenI, x_tenT)
solution = tf.matmul(product, y_ten)

solution_eval = sess.run(solution)
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope is :' + str(slope))
print('y_intercept is :' + str(y_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()

