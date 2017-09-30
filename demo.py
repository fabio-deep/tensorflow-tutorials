import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#3                                # a rank 0 tensor; this is a scalar with shape []
#[1., 2., 3.]                     # a rank 1 tensor; this is a vector with shape [3]
#[[1., 2., 3.], [4., 5., 6.]]     # a rank 2 tensor; a matrix with shape [2, 3]
#[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

#Like all TensorFlow constants, 
#it takes no inputs, and it outputs a value it stores internally. 
#We can create two floating point Tensors node1 and node2 as follows
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

#We can build more complicated computations by, 
#combining Tensor nodes with operations (Operations are also nodes).
#we can add our two constant nodes and produce a new graph as follows.
node3 = tf.add(node1, node2)

#A graph can be parameterized to accept external inputs, known as placeholders. 
#A placeholder is a promise to provide a value later.
u = tf.placeholder(tf.float32)
v = tf.placeholder(tf.float32)

# "+" provides a shortcut for tf.add(a, b)
adder_node = u + v 

#add another operation to the computation graph
add_and_triple = adder_node * 3.

#To make the model trainable, 
#we need to be able to modify the graph to get new outputs with the same input. 
#Variables allow us to add trainable parameters to a graph. 
#They are constructed with a type and initial value
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

#To evaluate the model on training data, 
#we need a y placeholder to provide the desired values, 
#and we need to write a loss function.
y = tf.placeholder(tf.float32)

#Using a standard loss model for linear regression, 
#which sums the squares of the deltas between the current model and the provided data. 
#linear_model - y creates a vector where,
#each element is the corresponding example's error delta. 
#We call tf.square to square that error. 
#we sum all the squared errors to create a single scalar,
#that abstracts the error of all examples using tf.reduce_sum:
loss = tf.reduce_sum(tf.square(linear_model - y))

#Gradient descent modifies each variable according to,
#the magnitude of the derivative of loss with respect to that variable.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#Constants are initialized when you call tf.constant, and their value can never change. 
#By contrast, variables are not initialized when you call tf.Variable. 
#To initialize all the variables in a TensorFlow program, 
#you must explicitly call a special operation as follows:
init = tf.global_variables_initializer()

#A variable is initialized to the value provided to tf.Variable,
#but can be changed using operations like tf.assign
#For example, W =-1 and b = 1 are the optimal parameters for our model. 
#We can change W and b accordingly:
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

#=====================SESSION==========================
#To actually evaluate the nodes, 
#we must run the computational graph within a session. 
#A session encapsulates the control and state of the TensorFlow runtime.
sess = tf.Session()

#init is a handle to the TensorFlow sub-graph that initializes all the global variables. 
#Until we call sess.run, the variables are uninitialized.
sess.run(init)

#runs the session, output is: [3.0, 4.0]
print(sess.run([node1, node2]))

#output is: node3: Tensor("Add:0", shape=(), dtype=float32); sess.run(node3): 7.0
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

#Like a function or a lambda in which we define,
#two input parameters (a and b) and then an operation on them. 
#We can evaluate this graph with multiple inputs by,
#using the feed_dict argument to the run method to feed values to the placeholders:
print(sess.run(adder_node, {u: 3, v: 4.5}))
print(sess.run(adder_node, {u: [1, 3], v: [2, 4]}))

#We can make the computational graph more complex by adding another operation:
#output 22.5
print(sess.run(add_and_triple, {u: 3, v: 4.5}))

#Since x is a placeholder, 
#we can evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

#passing the values of x to compute loss function with the desired output y
#output is 23.66 
print('Initial loss:', sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#run the assign params to change w and b to optimal values and rerun loss
#output is 0 loss
sess.run([fixW, fixb])
print('Perfect loss:', sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

sess.run(init)
#training loop of 1000 iterations
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
# Print the cost every 100 iterations
    if i % 100 == 0:
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

sess.close()
