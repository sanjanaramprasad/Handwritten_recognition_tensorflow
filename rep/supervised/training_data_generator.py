import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
import Image
import random


def make_dir(training_directory , testing_directory):
    if not os.path.exists(training_directory):
        os.makedirs(training_directory)
    if not os.path.exists(testing_directory):
        os.makedirs(testing_directory)

def match_images(input_dir, dirs, files):
    split_counter = 0
    for i in range(0, len(files)):
        print (i)
        img1 = plt.imread(os.path.join(input_dir, files[i]))
        img1 = (img1 * 255).round().astype(np.uint8)
        img1 = imresize(img1, (64, 64))
        for j in range(i+1 , len(files)):
            if (files[j][:4] == files[i][:4]):
                name = "match"+str(files[i][:10]) + "_"+ str(files[j][:10]) + ".png"
            	img2 = plt.imread(os.path.join(input_dir, files[j]))
            	img2 = (img2 * 255).round().astype(np.uint8)
                img2 = imresize(img2, (64, 64))
                img = np.vstack((img1, img2))
            	img = Image.fromarray(img)
                if(split_counter < 8000):
                    split_counter+=1
                    img.save(os.path.join(dirs[1], name))
                else:
            	    img.save(os.path.join(dirs[0], name))


def non_match_images(input_dir , dirs ,files):
    split_counter = 0
    for i in range(1, 1569):
        if(i < 10):
            key = "000" + str(i)
        elif (i < 100):
            key = "00" + str(i)
        elif (i <1000):
            key = "0" + str(i)
        else :
            key = str(i)
        shortlisted_file_names = [ filename for filename in files if filename[:4] == key]
        if shortlisted_file_names:
            shortlisted = list(set(files) - set(shortlisted_file_names))
            for j in range(0,54):
                img1_key = random.choice(shortlisted_file_names)
                img1 = plt.imread(os.path.join(input_dir, img1_key))
                img1 = (img1 * 255).round().astype(np.uint8)
                img1 = imresize(img1, (64, 64))
                img2_key = random.choice(shortlisted)
                shortlisted = list(set(shortlisted) - set(img2_key))
                img2 = plt.imread(os.path.join(input_dir, img2_key))
                name = "mis_match" + str(img1_key[:10]) + "_" + str(img2_key[:10]) + ".png"
                img2 = (img2 * 255).round().astype(np.uint8)
                img2 = imresize(img2, (64, 64))
                img = np.vstack((img1, img2))
                img = Image.fromarray(img)
                if(split_counter < 8000):
                    split_counter+=1
                    img.save(os.path.join(dirs[1], name))
                else:
                    img.save(os.path.join(dirs[0], name))



if __name__ ==  '__main__':
    print "here"
    input_dir = '/home/sanjana/PycharmProjects/DeepLearning/images/'
    files = [ input_file for input_file in os.listdir(input_dir) if '.png' in input_file]
    output_dir = os.getcwd()
    training_dir = 'training_data_v2'
    testing_dir = 'testing_data_v2'
    make_dir(training_dir , testing_dir)
    training_dir = output_dir + '/'+training_dir+'/'
    testing_dir = output_dir +'/'+testing_dir+'/'
    dirs = [training_dir , testing_dir]
    match_images(input_dir , dirs , files)
    non_match_images(input_dir , dirs, files)
