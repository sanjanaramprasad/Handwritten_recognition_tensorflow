import tensorflow as tf
import dataset
from tensorflow.python.framework.ops import reset_default_graph
import utils
import logging

reset_default_graph()

g = tf.get_default_graph()
[op.name for op in g.get_operations()]

ds = dataset.read_data_sets('/home/ops/data/',one_hot=True)
X= tf.placeholder(tf.float32 , [None, 8192])
Y = tf.placeholder(tf.float32 , [None , 2])

X_tensor = tf.reshape(X,[-1,128,64,1])

filter_size = 5
n_filters_in = 1
n_filters_out = 68
n_output = 2
W_1 = tf.get_variable( name = 'W' , shape=[filter_size , filter_size ,n_filters_in , n_filters_out],
                       initializer= tf.random_normal_initializer())

b_1 = tf.get_variable(name = 'b' , shape=[n_filters_out] , initializer = tf.constant_initializer())

h_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input= X_tensor , filter = W_1 , strides=[1,2,2,1] , padding = 'SAME'), b_1))

n_filters_in = 68
n_filters_out = 136

W_2 = tf.get_variable(name = 'W2' , shape = [filter_size, filter_size, n_filters_in, n_filters_out], initializer= tf.random_normal_initializer())
b_2 = tf.get_variable( name = 'b2' , shape = [n_filters_out] , initializer= tf.constant_initializer())
h_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input = h_1 , filter = W_2 , strides= [1,2,2,1] , padding = 'SAME') , b_2))

h_2_flat = tf.reshape(h_2 , [-1 , 32*16 *n_filters_out])

h_3 , W = utils.linear(h_2_flat , 256 , activation = tf.nn.relu , name = 'fc_1')

Y_pred , W = utils.linear(h_3 , n_output , activation = tf.nn.softmax , name = 'fc_2' )

cross_entropy = -tf.reduce_sum(Y*tf.log(Y_pred + 1e-12))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y_pred, 1 ) , tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , 'float'))


sess = tf.Session()
sess.run(tf.initialize_all_variables())



n_epochs = 10
for epoch_i in range(n_epochs):
    for batch_xs, batch_ys in ds.train.next_batch():
        sess.run(optimizer, feed_dict={
            X: batch_xs,
            Y: batch_ys
        })
        print(sess.run(accuracy , feed_dict = {X: batch_xs , Y:batch_ys}))
    print("Epoch number %s"%epoch_i)
    valid = ds.validation
    print("Validation",sess.run(accuracy,
                   feed_dict={
                       X: valid.images,
                       Y: valid.labels
                   }))
print("Optimzation done")
print(sess.run(accuracy , feed_dict={X:ds.test.images, Y:ds.test.labels}))
