'''
Learning tensorflow for first time
'''

import tensorflow as tf

x = tf.constant(35, name = 'x')
y = tf.Variable(x + 5, name = 'y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))


# E 1)

import tensorflow as tf

x = tf.constant([35, 40, 45], name = 'x')
y = tf.Variable(x + 5, name = 'y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))

# E 2)

import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size = 10000)

x = tf.constant(data, name = 'x')
y = tf.Variable( (5 * x**2) - (3 * x) + 15, name = 'y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))


# E 3)

import tensorflow as tf

x = tf.Variable(0, name = 'x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        x += 1
        print(session.run(x))


# E 4)

import tensorflow as tf
import numpy as np

m = 10000
n = tf.Variable(0, name = 'x')
mean = tf.Variable(0, name = 'mean')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        random_numbers = np.random.randint(1000, size = m )
        sum_random_numbers = np.sum(random_numbers)
        n += m
        mean = (sum_random_numbers / n) + (mean * (n-m)/n)
        print(session.run(mean))


# E 5)

import tensorflow as tf

x = tf.constant(35, name = 'x')
print(x)

y = tf.Variable(x + 5, name='y')


with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/basic", session.graph)
    model = tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))

