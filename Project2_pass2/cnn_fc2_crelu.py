import tensorflow as tf
#from tensorflow.python.framework.ops import reset_default_graph
import dataset
import config
import os
import numpy as np

def train():
    sess = tf.InteractiveSession()
    ds = dataset.read_data_sets(config.images_dir, one_hot=True)
    print(ds.train.images.shape)
    print(ds.train.labels.shape)
    print(ds.test.images.shape)
    print(ds.test.labels.shape)



    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial , name)

    def weight_variable(shape , name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial , name = name)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    def batch_norm(x, n_out, phase_train, scope='bn'):
        with tf.variable_scope(scope):
            # pred = tf.placeholder(tf.bool , shape = [])
            # pred = True
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def conv_layer(input, channels_in, channels_out , phase_train ,act='relu', pool=False ,name ='conv'):
        with tf.name_scope(name):
            w = weight_variable([3,3,channels_in,channels_out] , name)
            b = bias_variable([channels_out], name)
            conv = conv2d(input , w)
            if act == 'crelu':
                act = tf.nn.crelu(batch_norm(conv , channels_out , phase_train ) + b)
            else:
                act = tf.nn.relu(batch_norm(conv , channels_out,phase_train) + b)
            tf.summary.histogram('weights' , w)
            tf.summary.histogram('biases' , b)
            tf.summary.histogram('activations_crelu' , act)
            if pool:
                h_pool = max_pool_2x2(act)
                return h_pool
            return act

    def fc_layer(input, channels_in , channels_out, keep_prob, name='fc',logits = False):
        with tf.name_scope(name):
            w = weight_variable([channels_in , channels_out] , name)
            b =bias_variable([channels_out] , name)
            h_fc =tf.nn.relu(tf.matmul(input,w) + b)
            if logits:
                return tf.matmul(input,w) + b
            return tf.nn.dropout(h_fc , keep_prob)


    #### Setting up the graph#####
    '''placeholders for inputs'''
    x = tf.placeholder(tf.float32, shape=[None, 4096],name = 'x')
    y_ = tf.placeholder(tf.float32, shape=[None, 2] , name = 'labels')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(x, [-1, 64, 64, 1])
    tf.summary.image('input',x_image,3)

    ####Convolution Architecture####
    conv1 = conv_layer(x_image,1,8,phase_train,act = 'crelu' , pool = False ,name='conv1')
    conv2 = conv_layer(conv1 , 16,16,phase_train,act = 'crelu' , pool=True , name ='conv2')
    conv3 = conv_layer(conv2, 32, 32, phase_train, act = 'relu' , pool = False , name ='conv3')
    conv4 = conv_layer(conv3 , 32, 32, phase_train , act = 'relu' , pool = True , name = 'conv4' )
    conv5 = conv_layer(conv4 , 32 , 64 , phase_train, act = 'relu', pool = False , name = 'conv5')
    conv6 = conv_layer(conv5 , 64,64, phase_train, act ='relu' , pool=True, name = 'conv6')


    ####Fully Connected Layer############
    flattened = tf.reshape(conv6 , [-1, 8*8*64])
    fc1 = fc_layer(flattened , 8*8*64 ,1700,keep_prob , name = 'fc1' )
    #fc2 = fc_layer(fc1 , 1700 , 900 , keep_prob , name ='fc2')
    logits = fc_layer(fc1 , 1700, 2 ,keep_prob , name = 'fc3' , logits= True)

    ### loss function ####

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    tf.summary.scalar('cross_entropy' , cross_entropy)

    ####optimizer###

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    #### accuracy ######
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy' , accuracy)
    #training_accuracy = tf.summary.scalar('training_accuracy' , accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'train/'+config.current_model))
    validation_writer = tf.summary.FileWriter(os.path.join(config.log_dir,'validation/'+config.current_model))
    writer.add_graph(sess.graph)
    writer.flush()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(config.epoch):
        display_step = 0
        total_tr = 0
        good_tr =0
        for batch_xs, batch_ys in ds.train.next_batch():
            display_step += 1
            if display_step % 100 == 0:
                summary , training_accuracy = sess.run([merged_summary , accuracy] , feed_dict={
                    x: batch_xs, y_: batch_ys, phase_train.name: False, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (display_step, training_accuracy))
                writer.add_summary(summary , display_step)
                writer.flush()
            summary , _ = sess.run([merged_summary , train_step] , feed_dict = {x: batch_xs, y_: batch_ys, phase_train.name: True, keep_prob: 0.6})
            #[train_loss,_,loss] = sess.run([training_loss,train_step,cross_entropy],feed_dict={x: batch_xs, y_: batch_ys, phase_train.name: True, keep_prob: 0.6})
            #writer.add_summary(summary, display_step)
            #writer.flush()
        total =0
        good = 0
        step =0
        accuracy_sumv = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        for batch_x,batch_y in ds.validation.next_batch():
            step+=1
            total += batch_x.shape[0]
            if step%100 == 0:
                validation_summary , validation_acc= sess.run([merged_summary, accuracy] ,feed_dict={
                    x: batch_x, y_: batch_y, phase_train.name: False, keep_prob: 1.0})
                validation_writer.add_summary(validation_summary , step)
                validation_writer.flush()
            good += accuracy_sumv.eval(feed_dict={
                    x: batch_x, y_: batch_y, phase_train.name: False, keep_prob: 1.0})
        average = good/total
        #validation_accuracy = tf.summary.scalar('validation_accuracy', average)
        #validation_writer.add_summary(validation_accuracy , i)
        #validation_writer.flush()
        print("step %d, validation accuracy%g" % (display_step, average))
    #saver.save(sess , os.path.join(config.log_dir,'projector/'+config.current_model,'model.ckpt') , global_step = config.epoch)
    validation_writer.close()
    writer.close()

    total_test = 0
    accuracy_sumt = tf.reduce_sum(tf.cast(correct_prediction , tf.float32))
    good_test =0
    for batch_xt,batch_yt in ds.test.next_batch():
        total_test+= batch_xt.shape[0]
        good_test+= accuracy_sumt.eval(feed_dict = { x:batch_xt , y_:batch_yt , phase_train.name : False, keep_prob:1.0})
    print("step %d, test accuracy%g" % ( 30, good_test/total_test))
    print("Done!!")

    #writer.close()
    sess.close()


if __name__ == '__main__':
    train()















