Last login: Mon Nov  7 13:23:23 on ttys001
You have mail.
Nikhils-MacBook-Pro-2:deep_learning_working_directory nikhilshekhar$ scp nikhil@mr-0xg7:/home/nikhil/data/pipeline.py . 
nikhil@mr-0xg7's password: 
pipeline.py                                                                                                                                                               100%   13KB  13.0KB/s   00:00    
Nikhils-MacBook-Pro-2:deep_learning_working_directory nikhilshekhar$ vim pipeline.py 
























































import luigi
import config
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
import Image
import random
import gzip
import os
import struct
import re, shutil
import math

import ast
class TupleParameter(luigi.Parameter):
    def parse(self, x):
        #print x
        return tuple(ast.literal_eval(str(x)))


class match_images(luigi.Task):
    dirs_1 = TupleParameter()
    files_1 = TupleParameter()
    def output(self):
        return luigi.LocalTarget('generated_match_images')
    def run(self):
        with self.output().open('w') as out_file:
                index = 0
                while( index < len(self.files_1) - 1):
                        #print index
                        matches_counter = 1
                        person_images = []
                        #print "------------------------"
                        #print self.files_1[index]
                        person_images.append(self.files_1[index])
                        flag = True
                        while(flag):
                                next_index = index + matches_counter
                                #print next_index
                                if (next_index < len(self.files_1)) and (self.files_1[index][:4] == self.files_1[next_index][:4]):
                                        #print self.files_1[next_index]
                                        person_images.append(self.files_1[next_index])
                                        matches_counter+=1
                                else:
                                        flag = False
                        permutations = 0
                        for p in range(1 , len(person_images)-1):
                                permutations+=p
                        #print "PERMUTATIONS" , permutations
                        if(len(person_images) > 1):
                                split_count_test = 0
                                split_count_valid = 0
                                for i in range(0 , len(person_images)):
                                        img1 = plt.imread(os.path.join(config.input_dir , person_images[i]))
                                        img1 = (img1*255).round().astype(np.uint8)
                                        img1 = imresize(img1, (32 , 64))
                                        for j in range(i+1,len(person_images)):
                                                #print j
                                                name = 'match' + str(person_images[i][:10]) + "_"+ str(person_images[j][:10]) + ".png"
                                                img2 = plt.imread(os.path.join(config.input_dir , person_images[j]))

