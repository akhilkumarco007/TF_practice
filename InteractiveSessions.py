# coding: utf-8

# In[1]:


print 'hello'


# In[ ]:





# In[ ]:





# In[2]:


import tensorflow as tf


# In[3]:


session = tf.InteractiveSession()


# In[4]:


x = tf.constant(list(range(10)))


# In[5]:


print(x.eval())


# In[6]:


session.close()


# In[7]:


import resource


# In[8]:


print '{} Kb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[9]:


import numpy as np
session = tf.InteractiveSession()


# In[10]:


X = tf.constant(np.eye(10000))
Y = tf.constant(np.random.randn(10000, 300))


# In[11]:


print '{} Kb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[12]:


Z = tf.matmul(X, Y)


# In[13]:


print '{} Kb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[ ]:





# In[14]:


Z.eval()


# In[15]:


print '{} Kb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[16]:


session.close()


# In[17]:


print '{} Kb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[1]:


print '{} Kb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[ ]:


# Exercises

# E1

# coding: utf-8

# In[1]:


import resource


# In[3]:


print "{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[5]:


import tensorflow as tf
import numpy as np
session = tf.InteractiveSession()


# In[8]:


X = tf.constant(np.random.randint(10, high= 100, size=10000000))


# In[9]:


print "{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[10]:


X = tf.to_float(X)


# In[11]:


print "{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# In[12]:


X.eval()


# In[13]:


print "{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

# E2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

path = '/home/akhil/Downloads/MarshOrchid.jpg'

# load the image
image = mpimg.imread(path)

X = tf.Variable(image, name='X')

X = tf.image.random_brightness(X, 0.5)

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    result = sess.run(X)

plt.imshow(result)
plt.show()