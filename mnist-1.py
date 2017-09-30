from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#The MNIST data is split into three parts: 
#55,000 data points of training data (mnist.train), 
#10,000 points of test data (mnist.test), 
#and 5,000 points of validation data (mnist.validation).

#every MNIST data point has two parts: 
#an image of a handwritten digit and a corresponding label. 
#We'll call the images "x" and the labels "y". 
#Both the training set and test set contain images and their corresponding labels; 
#for example the training images are mnist.train.images
#and the training labels are mnist.train.labels

#Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers
#We can flatten this array into a vector of 28x28 = 784 numbers. 
#It doesn't matter how we flatten the array, as long as we're consistent between images.

#The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]. 
#The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image. 
#Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.

#Each image in MNIST has a corresponding label, 
#a number between 0 and 9 representing the digit drawn in the image.

#We're going to want our labels as "one-hot vectors". 
#A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. 
#In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. 
#For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. 
#Consequently, mnist.train.labels is a [55000, 10] array of floats.


#x isn't a specific value. It's a placeholder, 
#a value that we'll input when we ask TensorFlow to run a computation. 
#We want to be able to input any number of MNIST images, 
#each flattened into a 784-dimensional vector. 
#We represent this as a 2-D tensor of floating-point numbers, 
#with a shape [None, 784], (Here None means that a dimension can be of any length.)
x = tf.placeholder(tf.float32, [None, 784])

#Notice that W has a shape of [784, 10] because, we want to multiply,
#the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes. 
#b has a shape of [10] so we can add it to the output.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implement softmax model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#defining placeholder for input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

#First, tf.log computes the logarithm of each element of y. 
#Next, we multiply each element of y_ with the corresponding element of tf.log(y).
#Then tf.reduce_sum adds the elements in the second dimension of y, 
#due to the reduction_indices=[1] parameter. 

#Finally, tf.reduce_mean computes the mean over all the examples in the batch.
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#The raw formulation of cross-entropy, can be numerically unstable.
#So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
#outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Minimise cross_entropy with gradient descent.
#What TensorFlow actually does here, behind the scenes, 
#is to add new operations to your graph which implement backpropagation and gradient descent. 
#Then it gives you back a single operation which, when run, 
#does a step of gradient descent training slightly tweaking your variables to reduce the loss.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.03).minimize(cross_entropy)

#start interactive session
sess = tf.InteractiveSession()

#initialise variables w and b (weights and biases)
tf.global_variables_initializer().run()

#Training with 1000 iterations.#
#Each step of the loop, we get a 100 data point sized batch from the training set.
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) #get batch from mnist
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) #feed batch into optimizer

#tf.argmax is an extremely useful function which gives you the,
#index of the highest entry in a tensor along some axis. 
#For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, 
#while tf.argmax(y_,1) is the correct label. 
#We can use tf.equal to check if our prediction matches the truth.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#That gives us a list of booleans. 
#To determine what fraction are correct, 
#we cast to floating point numbers and then take the mean.
#For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#calculate accuracy on the test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()