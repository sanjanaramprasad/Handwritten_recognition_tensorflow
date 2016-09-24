import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

cur_dir = os.getcwd()
print("resizing images")
print("current directory:",cur_dir)

def modify_image(image):
    resized = tf.image.resize_images(image, 56, 56, 1)
    resized.set_shape([56,56,3])
    return resized

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    return image

def get_file_names():
    filenames = []
    for file in os.listdir(cur_dir):
        if file.endswith(".png"):
            filenames.append(file)
    print filenames
    return filenames
def inputs():
    filenames = get_file_names()
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    return filenames , reshaped_image

with tf.Graph().as_default():
    filenames, image = inputs()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for name in filenames:
        print name
        img = sess.run(image)
        img = np.squeeze(img, 2)
        print img.shape
        img = Image.fromarray(img)
        img.save(os.path.join('/home/sanjana/Handwritten_and_preprocessed', name ))