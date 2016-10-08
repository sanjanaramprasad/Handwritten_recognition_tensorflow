import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.misc import imresize
import Image
import random



def match_images(files):
    for i in range(0, len(files)):
        print (i)
        img1 = plt.imread(os.path.join('/home/sanjana/Handwritten_and_preprocessed_v2', files[i]))
        img1 = (img1 * 255).round().astype(np.uint8)
        for j in range(i+1 , len(files)):
            if (files[j][:4] == files[i][:4]):
                #print files[i][:10]
                #print files[j][:10]
                name = "match"+str(files[i][:10]) + "_"+ str(files[j][:10]) + ".png"
            	#else:
                    #name = "non_match"+ str(files[i][:10]) + "_"+ str(files[j][:10]) +".png"
            	img2 = plt.imread(os.path.join('/home/sanjana/Handwritten_and_preprocessed_v2', files[j]))
            	img2 = (img2 * 255).round().astype(np.uint8)
            	img = img1 & img2
            	img = Image.fromarray(img)
            	img.save(os.path.join('/home/sanjana/Handwritten_and_preprocessed_v2/training_data_matched/', name))


def non_match_images(files):
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
        print shortlisted_file_names
        if shortlisted_file_names:
            shortlisted = list(set(files) - set(shortlisted_file_names))
            for j in range(0,54):
                img1_key = random.choice(shortlisted_file_names)
                #print img1_key
                img1 = plt.imread(os.path.join('/home/sanjana/Handwritten_and_preprocessed_v2/', img1_key))
                img1 = (img1 * 255).round().astype(np.uint8)
                img2_key = random.choice(shortlisted)
                shortlisted = list(set(shortlisted) - set(img2_key))
                img2 = plt.imread(os.path.join('/home/sanjana/Handwritten_and_preprocessed_v2/', img2_key))
                print img1_key , img2_key
                name = "mis_match" + str(img1_key[:10]) + "_" + str(img2_key[:10]) + ".png"
                img2 = (img2 * 255).round().astype(np.uint8)
                img = img1 & img2
                img = Image.fromarray(img)
                img.save(os.path.join('/home/sanjana/Handwritten_and_preprocessed_v2/training_data_mismatch/', name))




input_dir = '/home/sanjana/PycharmProjects/DeepLearning/images/'
#input_dir = '/home/sanjana/Handwritten_and_preprocessed_v2/'
files = [ input_file for input_file in os.listdir(input_dir) if '.png' in input_file]
'''for each in files:
    img = plt.imread(os.path.join(input_dir, each))
    #img2 = plt.imread(files[1])
    img = (img * 255).round().astype(np.uint8)
    img = imresize(img, (64,64))
    img = Image.fromarray(img)
    img.save(os.path.join('/home/sanjana/Handwritten_and_preprocessed_v2', each))
    #print img'''
match_images(files)
non_match_images(files)
