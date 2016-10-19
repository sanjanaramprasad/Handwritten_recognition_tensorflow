import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imresize
import pickle
import random
from PIL import Image


def matching_training_data(input_dir, resize_x, resize_y ):

    print("Input directory path:", input_dir)
    matching_image_dict = {}
    input_file_names = os.listdir(str(input_dir))
    index_in_file = 0
    print("Number of input files:",len(input_file_names))
    print("Processing MATCHING images and creating a pickle file starts")
    for i in range(0, len(input_file_names)):
        print("I:",i)
        first_img = plt.imread(os.path.join(input_dir,input_file_names[i]))
        if first_img is not None:
            first_img = first_img[:, :, 0]
            first_img = imresize(first_img, (resize_x, resize_y))
            first_img = first_img.ravel()
            first_img = np.multiply(first_img, 1.0 / 255.0)
            # counter_to_break = 0
            for j in range(i + 1, len(input_file_names)):
                # print("J",j)
                '''Check for first four
                    check to discard files split into two from being a part of training data'''
                if  input_file_names[j][:4] == input_file_names[i][:4] and input_file_names[i][4] != input_file_names[i][5]:
                    second_img = plt.imread(os.path.join(input_dir,input_file_names[j]))
                    if second_img is not None:
                        second_img = second_img[:, :, 0]
                        second_img = imresize(second_img, (resize_x, resize_y))
                        second_img = second_img.ravel()
                        second_img = np.multiply(second_img, 1.0 / 255.0)
                        '''Create a dictionary with
                           key - number autoincremented
                           value - list with three elements [0] - label, [1] - first image 1d, [2] - second image 1d '''
                        matching_image_dict[index_in_file] = ['0', first_img, second_img]

                        # Increment the index counter
                        index_in_file += 1
                    # counter_to_break += 1
                # if counter_to_break >= number_of_samples:
                #     break;

    # Done creating a dictionary for matching images
    # Flush to disk
    print("Number of elements in the MATCHING pickle dictionary:",len(matching_image_dict))
    # output_file = open(matching_output_file_name, 'wb')
    # pickle.dump(matching_image_dict, output_file, protocol=2)
    # output_file.close()
    return matching_image_dict, index_in_file


def not_matching_training_data(input_dir, output_file_name, index_in_file_matching, matching_image_dict, resize_x, resize_y):
    print("Input directory path:", input_dir)
    print("Output file name:", output_file_name)
    not_matching_image_dict = matching_image_dict
    input_file_names = os.listdir(str(input_dir))
    index_in_file = index_in_file_matching + 1
    print("Number of input files:",len(input_file_names))
    print("Processing NOT MATCHING images and creating a pickle file starts")
    for i in range(1, 1568):
        print("I:",i)
        if (i < 10):
            key = "000" + str(i)
        elif (i < 100):
            key = "00" + str(i)
        elif (i < 1000):
            key = "0" + str(i)
        else:
            key = str(i)
        shortlisted_file_names = [filename for filename in input_file_names if filename[:4] == key]
        if shortlisted_file_names:
            shortlisted = list(set(input_file_names) - set(shortlisted_file_names))
            for j in range(1, 54):
                # print("J:",j)
                first_img_key = random.choice(shortlisted_file_names)
                first_img = plt.imread(os.path.join(input_dir, first_img_key))
                if first_img is not None:
                    first_img = first_img[:, :, 0]
                    first_img = imresize(first_img, (resize_x, resize_y))
                    first_img = first_img.ravel()
                    first_img = np.multiply(first_img, 1.0 / 255.0)
                    second_img_key = random.choice(shortlisted)
                    shortlisted = list(set(shortlisted) - set(second_img_key))
                    second_img = plt.imread(os.path.join(input_dir, second_img_key))
                    if second_img is not None:
                        second_img = second_img[:, :, 0]
                        second_img = imresize(second_img, (resize_x, resize_y))
                        second_img = second_img.ravel()
                        second_img = np.multiply(second_img, 1.0 / 255.0)
                        '''Create a dictionary with
                                   key - number autoincremented
                                   value - list with three elements [0] - label, [1] - first image 1d, [2] - second image 1d '''
                        not_matching_image_dict[index_in_file] = ['0', first_img, second_img]
                        # Increment the index counter
                        index_in_file += 1
    # Done creating a dictionary for matching images
    # Flush to disk
    print("Number of elements written to pickle file:",len(not_matching_image_dict))
    output_file = open(output_file_name, 'wb')
    pickle.dump(not_matching_image_dict, output_file, protocol=2)
    output_file.close()

input_dir = "/Users/nikhilshekhar/deep_learning_working_directory/images"
#input_dir = "/Users/nikhilshekhar/deep_learning_working_directory/test_images"
output_file = "pre_processed.p"

# Resize input image to resize_x * resize_y
resize_x = 24
resize_y = 32

# Call method to create matching training data
matching_image_dict, index_in_file = matching_training_data(input_dir,resize_x,resize_y)
# matching_image_dict = {}
# index_in_file = 0


# Call method to create not - matching training data
not_matching_training_data(input_dir, output_file, index_in_file, matching_image_dict, resize_x, resize_y)
