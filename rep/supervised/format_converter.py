import  numpy as np
import Image
import gzip
import os
import struct
import re

def encode_images(input_dir, output_filename, ext='.png'):
  # Ensure the input directory has a trailing slash
  if input_dir[-1] != '/':
    input_dir += '/'

  fs = [input_dir + x for x in np.sort(os.listdir(input_dir)) if ext in x]
  num_imgs = len(fs)
  output_file = open(output_filename, "wb")
  print num_imgs
  # Write items in the header
  # MNIST uses 2051 as a magic number in the header, so following convention here
  output_file.write(struct.pack('>i', 2051))       # Magic Number for train/test images
  output_file.write(struct.pack('>i', num_imgs))   # Number of images

  # Load the first image to get dimensions
  im = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.uint32)
  r,c = im.shape

  # Write the rest of the header
  output_file.write(struct.pack('>i', r))          # Number of rows in 1 image
  output_file.write(struct.pack('>i', c))          # Number of columns in 1 image

  # For each image, record the pixel values in the binary file
  for img in range(num_imgs):
    im = np.asarray(Image.open(fs[img]).convert('L'), dtype=np.uint32)
    for i in xrange(im.shape[0]):
      for j in xrange(im.shape[1]):
        output_file.write(struct.pack('>B', im[i,j]))

  # Close the file
  output_file.close()

  f_in = open(output_filename)
  f_out = gzip.open(output_filename + '.gz', 'wb')
  f_out.writelines(f_in)
  f_out.close()
  f_in.close()
  os.remove(output_filename)



#############################encoding labels the same way as images#################################################################################################################

def encode_labels(input_dir, output_filename, ext='.png' ):

      if input_dir[-1] != '/':
          input_dir += '/'

      fs = [input_dir + x for x in np.sort(os.listdir(input_dir)) if ext in x]
      num_imgs = len(fs)
      output_file = open(output_filename, "wb")

      print (num_imgs)
      # Write items in the header
      # MNIST uses 2051 as a magic number in the header, so following convention here
      output_file.write(struct.pack('>i', 2049))  # Magic Number for train/test images
      output_file.write(struct.pack('>i', num_imgs))  # Number of images

      # Load the first image to get dimensions
      print fs[0]
      im = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.uint32)

      # For each image, record the pixel values in the binary file
      count = 0
      for img in range(num_imgs):
        count+=1
        print count
        #print fs[img]
        if re.match(input_dir+'match.*', fs[img]):
            label = 1
        elif re.match(input_dir + 'mis_match.*' , fs[img]):
            label=0
        label = np.uint32(label)
        print label
        output_file.write(struct.pack('>B', label))

      # Close the file
      output_file.close()

      f_in = open(output_filename)
      f_out = gzip.open(output_filename + '.gz', 'wb')
      f_out.writelines(f_in)
      f_out.close()
      f_in.close()
      os.remove(output_filename)





encode_images('/home/sanjana/Handwritten_and_preprocessed_v2/training_data_and/' , 'training-images-and-ubyte')
encode_labels('/home/sanjana/Handwritten_and_preprocessed_v2/training_data_and/' , 'training-labels-and-ubyte')
encode_images('/home/sanjana/Handwritten_and_preprocessed_v2/testing_data/' , 'testing-images-and-ubyte')
encode_labels('/home/sanjana/Handwritten_and_preprocessed_v2/testing_data/' , 'testing-labels-and-ubyte')
