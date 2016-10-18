import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imresize
import pickle


def matching_training_data(input_dir,matching_output_file_name, resize_x,resize_y):

    print("Input directory path:", input_dir)
    print("Output pickled MATCHING file name:", matching_output_file_name)
    matching_image_dict = {}
    input_file_names = os.listdir(str(input_dir))
    index_in_file = 0
    print("Processing MATCHING images and creating a pickle file starts")
    for i in range(0, len(input_file_names)):
        print(i)
        first_img = plt.imread(os.path.join(input_dir,input_file_names[i]))
        if first_img is not None:
            first_img = first_img[:, :, 0]
            first_img = imresize(first_img, (resize_x, resize_y))
            first_img = first_img.ravel()
            first_img = np.multiply(first_img, 1.0 / 255.0)
            for j in range(i + 1, len(input_file_names)):
                '''Check for first four
                    check to include files split into two '''
                if input_file_names[j][:4] == input_file_names[i][:4] and input_file_names[i][4] != input_file_names[i][5]:
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

    # Done creating a dictionary for matching images
    # Flush to disk
    print("Number of elements written to the  MATCHING pickle file:",len(matching_image_dict))
    output_file = open(matching_output_file_name, 'wb')
    pickle.dump(matching_image_dict, output_file)
    output_file.close()


def not_matching_training_data(input_dir,not_matching_output_file_name, resize_x, resize_y, number_of_mismatches):

    print("Input directory path:", input_dir)
    print("Output pickled NOT MATCHING file name:", not_matching_output_file_name)
    not_matching_image_dict = {}
    input_file_names = os.listdir(str(input_dir))
    index_in_file = 0
    print("Processing NOT MATCHING images and creating a pickle file starts")
    for i in range(0, len(input_file_names)):
        print(i)
        first_img = plt.imread(os.path.join(input_dir,input_file_names[i]))
        if first_img is not None:
            first_img = first_img[:, :, 0]
            first_img = imresize(first_img, (resize_x, resize_y))
            first_img = first_img.ravel()
            first_img = np.multiply(first_img, 1.0 / 255.0)
            counter_to_break = 0
            for j in range(i + 1, len(input_file_names)):
                '''Check for first four
                    check to include files split into two '''
                if input_file_names[j][:4] != input_file_names[i][:4] and j < counter_to_break:
                    second_img = plt.imread(os.path.join(input_dir,input_file_names[j]))
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
                        counter_to_break += 1

    # Done creating a dictionary for matching images
    # Flush to disk
    print("Number of elements written to the NOT MATCHING pickle file:",len(not_matching_image_dict))
    output_file = open(not_matching_output_file_name, 'wb')
    pickle.dump(not_matching_image_dict, output_file)
    output_file.close()

input_dir = "/Users/nikhilshekhar/deep_learning_working_directory/test_images"
matching_output_file = "/Users/nikhilshekhar/deep_learning_working_directory/matching_pre_processed.pickle"
not_matching_output_file = "/Users/nikhilshekhar/deep_learning_working_directory/not_matching_pre_processed.pickle"

# Resize input image to resize_x * resize_y
resize_x = 150
resize_y = 230

# Call method to create matching training data
matching_training_data(input_dir,matching_output_file,resize_x,resize_y)

# Number of mismatch data wanted
number_of_mismatches = 80000

# Donot change this - remains constant
number_of_person_sampled = 1500
number_of_mis_match_per_person_to_be_generated = int(number_of_mismatches/number_of_person_sampled)

# Call method to create not - matching training data
print("Number of NOT MATCHING samples created per person:",number_of_mis_match_per_person_to_be_generated)
not_matching_training_data(input_dir,matching_output_file,resize_x,resize_y,number_of_mis_match_per_person_to_be_generated)



