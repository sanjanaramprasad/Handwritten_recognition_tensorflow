import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
#import utils
import logging
import dataset


sess = tf.InteractiveSession()
ds = dataset.read_data_sets('/home/ubuntu/',one_hot=True)
print ds.train.images.shape
print ds.train.labels.shape
print ds.test.images.shape
print ds.test.labels.shape
x = tf.placeholder(tf.float32, shape=[None, 4096])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

learning_rate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=ds.train.images.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


### First Layer #######
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,64,64,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)




#Second layer###


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#####Third Layer #######
W_conv3 = weight_variable([5,5,64,128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


#######Fourth Layer########
W_conv4 = weight_variable([5, 5, 128, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_pool3 , W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

#########Fifth Layer################
W_conv5 = weight_variable([5,5,256,512])
b_conv5 = bias_variable([512])
h_conv5 = tf.nn.relu(conv2d(h_pool4 , W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv4)


#Fully Connected Layer#########
W_fc1 = weight_variable([2*2 *512, 1024])
b_fc1 = bias_variable([1024])

h_pool5_flat = tf.reshape(h_pool5, [-1, 2*2*512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2




cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#cost= tf.nn.l2_loss(y_conv - y_, name="squared_error_cost")
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
for i in range(20000):
	#print i
	batch_xs , batch_ys = ds.train.next_batch(100)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
        	x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		#print(sess.run(y_conv , feed_dict={x:batch[0]})) 
    		print("step %d, training accuracy %g"%(i, train_accuracy))
  	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
	
	if i%200 == 0:
		validation_accuracy = accuracy.eval(feed_dict={
                x:ds.validation.images, y_: ds.validation.labels, keep_prob: 1.0})
		print("step %d, validation accuracy%g"%(i, validation_accuracy))
        #train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: ds.test.images, y_: ds.test.labels, keep_prob: 1.0}))
print("Done!!")
