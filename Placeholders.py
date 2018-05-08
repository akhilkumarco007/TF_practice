# learning and practicing about placeholders in tensorflow

####
import tensorflow as tf

x = tf.placeholder('float', None)
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)

#####
import tensorflow as tf


x = tf.placeholder('float', [None, 3])
y = x * 2

with tf.Session() as session:
    x_data = [[1, 2, 3],
              [4, 5, 6]]
    result = session.run(y, feed_dict={x: x_data})
    print (result)

#####

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

path = '/home/akhil/Downloads/MarshOrchid.jpg'

raw_image = mpimg.imread(path)

image = tf.placeholder('uint8', [None, None, 3])
slice = tf.slice(image, [1000, 0 ,0], [3000, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image})
    print(result.shape)

plt.imshow(result)
plt.show()


# E2)

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

path = '/home/akhil/Downloads/MarshOrchid.jpg'

raw_image = mpimg.imread(path)
height, width, depth = raw_image.shape

image = tf.placeholder('uint8', [None, None, 3])

with tf.Session() as session:
    slice1 = tf.slice(image, [0, 0, 0], [height / 2, width / 2, -1])
    slice2 = tf.slice(image, [0, width / 2, 0], [height / 2, -1, -1])
    slice3 = tf.slice(image, [height / 2, 0, 0], [-1, width / 2, -1])
    slice4 = tf.slice(image, [height / 2, width / 2, 0], [-1, -1, -1])
    join1 = tf.concat([slice1, slice2], 1)
    join2 = tf.concat([slice3, slice4], 1)
    reconstructed = tf.concat([join1, join2], 0)
    result = session.run(reconstructed, feed_dict={image: raw_image})
    print(result.shape)

plt.imshow(result)
plt.show()

# E3)

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

path = '/home/akhil/Downloads/MarshOrchid.jpg'

raw_image = mpimg.imread(path)
height, width, depth = raw_image.shape

image = tf.placeholder('float32', [None, None, 3])
gray_scale_image = tf.reduce_mean(image, 2)

with tf.Session() as session:
    result = session.run(gray_scale_image, feed_dict={image: raw_image})
    print(result.shape)

plt.imshow(result, cmap='gray')
plt.show()