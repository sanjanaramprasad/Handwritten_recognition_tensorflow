import luigi
import config
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from PIL import Image
import random
import gzip
import os
import struct
import re


import ast
class TupleParameter(luigi.Parameter):
    def parse(self, x):
        print x
        return tuple(ast.literal_eval(str(x)))


class match_images(luigi.Task):
    dirs_1 = TupleParameter()
    files_1 = TupleParameter()
    def output(self):
        return luigi.LocalTarget('generated_match_images')
    def run(self):
        with self.output().open('w') as out_file:
            print "matching"
            split_counter = 0
            for i in range(0, len(self.files_1)):
                print (i)
                img1 = plt.imread(os.path.join(config.input_dir, self.files_1[i]))
                img1 = (img1 * 255).round().astype(np.uint8)
                img1 = imresize(img1, (64, 64))
                for j in range(i + 1, len(self.files_1)):
                    if (self.files_1[j][:4] == self.files_1[i][:4]):
                        name = "match" + str(self.files_1[i][:10]) + "_" + str(self.files_1[j][:10]) + ".png"
                        img2 = plt.imread(os.path.join(config.input_dir, self.files_1[j]))
                        img2 = (img2 * 255).round().astype(np.uint8)
                        img2 = imresize(img2, (64, 64))
                        img = np.vstack((img1, img2))
                        img = Image.fromarray(img)
                        if (split_counter < 8000):
                            split_counter += 1
                            img.save(os.path.join(self.dirs_1[1], name))
                        else:
                            img.save(os.path.join(self.dirs_1[0], name))
            out_file.write("Status : done")


class non_match_images(luigi.Task):
    dirs = TupleParameter()
    files = TupleParameter()
    def output(self):
        return luigi.LocalTarget('generated_non_match_images')
    def run(self):
        with self.output().open('w') as out_file:
            print "non matching"
            split_counter = 0
            for i in range(1, 1569):
                if (i < 10):
                    key = "000" + str(i)
                elif (i < 100):
                    key = "00" + str(i)
                elif (i < 1000):
                    key = "0" + str(i)
                else:
                    key = str(i)
                shortlisted_file_names = [filename for filename in self.files if filename[:4] == key]
                if shortlisted_file_names:
                    shortlisted = list(set(self.files) - set(shortlisted_file_names))
                    for j in range(0, 54):
                        img1_key = random.choice(shortlisted_file_names)
                        img1 = plt.imread(os.path.join(config.input_dir, img1_key))
                        img1 = (img1 * 255).round().astype(np.uint8)
                        img1 = imresize(img1, (64, 64))
                        img2_key = random.choice(shortlisted)
                        shortlisted = list(set(shortlisted) - set(img2_key))
                        img2 = plt.imread(os.path.join(config.input_dir, img2_key))
                        name = "mis_match" + str(img1_key[:10]) + "_" + str(img2_key[:10]) + ".png"
                        img2 = (img2 * 255).round().astype(np.uint8)
                        img2 = imresize(img2, (64, 64))
                        img = np.vstack((img1, img2))
                        img = Image.fromarray(img)
                        if (split_counter < 8000):
                            split_counter += 1
                            img.save(os.path.join(self.dirs[1], name))
                        else:
                            img.save(os.path.join(self.dirs[0], name))
            out_file.write("Status : done")


class training_data_generator(luigi.Task):
    def output(self):
        return luigi.LocalTarget('training_data_generator')
    def run(self):
        with self.output().open('w') as out_file:
            print "running training_data"
            files = [input_file for input_file in os.listdir(config.input_dir) if '.png' in input_file]
            output_dir = os.getcwd()
            if not os.path.exists(config.training_dir):
                os.makedirs(config.training_dir)
            if not os.path.exists(config.testing_dir):
                os.makedirs(config.testing_dir)
            training_dir = output_dir + '/' + config.training_dir + '/'
            testing_dir = output_dir + '/' + config.testing_dir + '/'
            dirs = [training_dir, testing_dir]
            yield match_images(dirs , files)
            yield non_match_images(dirs, files)
            out_file.write("Status :done")



class encode(luigi.Task):
    input_dir = luigi.Parameter()
    output_filename = luigi.Parameter()
    label = luigi.Parameter()
    def output(self):
        target = str(self.output_filename) + "_encoding_status"
        return luigi.LocalTarget(target)

    def run(self):
        print "running encoding"
        with self.output().open('w') as out_file:
            if str(self.input_dir)[-1] != '/':
                self.input_dir += '/'

	    l = os.listdir(str(self.input_dir))
	    random.shuffle(l)
            fs = [self.input_dir + x for x in l if '.png' in x]
            num_imgs = len(fs)
            output_file = open(str(self.output_filename), "wb")
            print (num_imgs)
            if self.label == "label":
                magic_num = 2049
            else:
                magic_num = 2051
            output_file.write(struct.pack('>i', magic_num))
            output_file.write(struct.pack('>i', num_imgs))

            if self.label == "image":
                im = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.uint32)
                r, c = im.shape
                output_file.write(struct.pack('>i', r))
                output_file.write(struct.pack('>i', c))

            for img in range(num_imgs):
                print img
                if self.label == "label":
                    if re.match(self.input_dir + 'match.*', fs[img]):
                        #print fs[img]
                        target = 1
                    elif re.match(self.input_dir + 'mis_match.*', fs[img]):
                        #print fs[img]
                        target = 0
		    print target
                    target = np.uint32(target)
                    #print target
                    output_file.write(struct.pack('>B', target))
                else:
                    im = np.asarray(Image.open(fs[img]).convert('L'), dtype=np.uint32)
                    for i in xrange(im.shape[0]):
                        for j in xrange(im.shape[1]):
                            output_file.write(struct.pack('>B', im[i, j]))
            output_file.close()
            f_in = open(str(self.output_filename))
            f_out = gzip.open(str(self.output_filename) + '.gz', 'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(str(self.output_filename))
            out_file.write('Status : done')

class encode_data(luigi.Task):
    def requires(self):
        return training_data_generator()

    def output(self):
        return luigi.LocalTarget('encode_data')

    def run(self):
        with self.output().open('w') as out_file:
            print "encoding_data"
            output_dir = os.getcwd()
            training_dir = output_dir + '/' + config.training_dir + '/'
            testing_dir = output_dir + '/' + config.testing_dir + '/'
            yield encode(training_dir, 'training-images-and-ubyte',label="image")
            yield encode(training_dir, 'training-labels-and-ubyte',label="label")
            yield encode(testing_dir, 'testing-images-and-ubyte',label="image")
            yield encode(testing_dir, 'testing-labels-and-ubyte', label="label")
            out_file.write("Status : done")



if __name__ == '__main__':
    luigi.run()

