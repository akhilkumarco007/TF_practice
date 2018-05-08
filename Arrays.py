'''
Practicing image manipulation using tensorflow and numpy
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

path = '/home/akhil/Downloads/MarshOrchid.jpg'

# load the image
image = mpimg.imread(path)
height, width, depth = image.shape

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()


# E1)

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


path = '/home/akhil/Downloads/MarshOrchid.jpg'

image = mpimg.imread(path)

height, width, depth = image.shape

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    x = tf.reverse_sequence(x, np.ones((width,)) * height, 1, batch_dim=0)

    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()


# E2)

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


path = '/home/akhil/Downloads/MarshOrchid.jpg'

image = mpimg.imread(path)

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    shape = tf.shape(x)
    session.run(model)
    shape = session.run(shape)
    x = tf.transpose(x, perm=[1, 0, 2])
    x = tf.reverse_sequence(x, np.ones((shape[1],)) * shape[0], 1, batch_dim=0)

    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()



# E3)

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

path = '/home/akhil/Downloads/MarshOrchid.jpg'

image = mpimg.imread(path)

height, width, depth = image.shape

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.reverse_sequence(x, np.ones((width,)) * height, 0, batch_dim=1)

    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()


# E4)

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


path = '/home/akhil/Downloads/MarshOrchid.jpg'

image = mpimg.imread(path)

height, width, depth = image.shape

cut_image = image[:(height/2), :, :]

cut_height, cut_width, cut_depth = cut_image.shape

x = tf.Variable(cut_image, name='x')
y = tf.Variable(cut_image, name='y')
model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.reverse_sequence(x, np.ones((cut_width,)) * cut_height, 0, batch_dim=1)
    x = tf.reverse_sequence(x, np.ones((cut_height)) * cut_width, 1, batch_dim=0)
    x = tf.concat([y, x], 0)
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()
