import pickle
import numpy as np



class Dataset(object):
        def __init__(self , image_pairs , labels):
                self._image_pairs = image_pairs
                self._labels = labels
                self._epochs_completed = 0
                self._index_in_epoch = 0
                #print image_pairs
                self._first_images = np.array([pair[0] for pair in image_pairs])
                self._second_images = np.array([pair[1] for pair in image_pairs])
                self._num_examples = image_pairs.shape[0]

        def image_pairs(self):
                return np.array(self._image_pairs)

        def labels(self):
                return np.array(self._labels)

        def first_images(self):
                return self._first_images

        def second_images(self):
                return self._second_images

        def next_batch(self, batch_size = 2):
                start = self._index_in_epoch
                self._index_in_epoch += batch_size
                if self._index_in_epoch > self._num_examples:
                        self._epochs_completed += 1
                        perm = np.arange(self._num_examples)
                        np.random.shuffle(perm)
                        self._first_images = self._first_images[perm]
                        self._second_images = self._second_images[perm]
                        self._labels = self._labels[perm]
                        # Start next epoch
                        start = 0
                        self._index_in_epoch = batch_size
                        assert batch_size <= self._num_examples
                end = self._index_in_epoch
                return self._first_images[start:end], self._second_images[start:end] , self._labels[start:end]


def extract_labels(data , num_classes , one_hot = False):
        labels = [value[0] for key , value in data.iteritems()]
        num_labels = len(labels)
        if one_hot:
                result = np.zeros(shape=(len(labels), 2))
                result[np.arange(len(labels)), labels] = 1
                return np.array(result)
        return np.array(labels)

def extract_image_pairs(data):
        image_pairs = [[value[1],value[2]] for key,value in data.iteritems()]
        return np.array(image_pairs)

def read_data_sets(test_size , validation_size):
        #######################splitting data ##################################################
        contents = pickle.load(open("matching_pre_processed.pickle" , 'rb'))
        keys = contents.keys()
        train_indices = keys[:len(contents)-(validation_size + test_size)]
        test_indices = keys[len(contents) - test_size :]
        validation_indices = keys[len(contents) - test_size - validation_size : len(contents) - test_size]
        train_images={k:contents[k] for k in train_indices}
        test_images = {index:contents[index] for index in test_indices}
        validation_images = {index:contents[index] for index in validation_indices}

        ################# Wrap it as a class of Dataset and return ################################ 
        train_image_pairs = extract_image_pairs(train_images)
        train_labels = extract_labels(train_images , num_classes = 2 , one_hot = True)
        #print train_labels
        validation_image_pairs = extract_image_pairs(validation_images)
        validation_labels = extract_labels(validation_images , num_classes = 2 , one_hot = True)
        test_image_pairs = extract_image_pairs(test_images)
        test_labels = extract_labels(test_images , num_classes = 2 , one_hot = True)
        train = Dataset(train_image_pairs , train_labels)
        #print train.image_pairs
        validation = Dataset(validation_image_pairs, validation_labels)
        test = Dataset(test_image_pairs , test_labels)
        return  train , validation , validation


        #return Dataset(train_images , test_images, validation_images)




data = read_data_sets(1,0)
print data[0].first_images().shape

        
  
