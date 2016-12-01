from imageCaptioning.model import *
from skimage import io
import pylab

model_path='/Users/nikhilshekhar/deep_learning_working_directory/project2_aws_models/model-58'
maxlen=30

# encoding="ISO-8859-1"
with open(vgg_path, mode='rb') as f:
    fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

images = tf.placeholder("float32", [1, 224, 224, 3])
tf.import_graph_def(graph_def, input_map={"images": images})

ixtoword = np.load('/Users/nikhilshekhar/deep_learning_working_directory/project2_aws_models/ixtoword.npy').tolist()
n_words = len(ixtoword)

sess = tf.InteractiveSession()

# Populate the objects
caption_generator = CaptionGenerator(dim_image=dim_image, dim_hidden=dim_hidden, dim_embed=dim_embed, batch_size=batch_size, n_lstm_steps=maxlen, n_words=n_words)

# Re-create the same Tensorflow graph which was used while training
graph = tf.get_default_graph()
fc7_tf, generated_words_tf = caption_generator.build_generator(maxlen=maxlen)
saver = tf.train.Saver()

# Read the weights file from disk and load it with the graph
saver.restore(sess, model_path)


''' This method generates the captions making use of the pre-trained LSTM model'''


def generate_captions(test_image_path=None):

    # display the image read
    pylab.imshow(io.imread(test_image_path))
    pylab.show()

    # Read in the image
    image = skimage.io.imread(test_image_path)

    # Resize the image to 1 * 224 * 224 * 3 as this is the expected shape of the tensor to be fed as input
    img = np.resize(image, (224, 224, 3))
    image_val = img[np.newaxis, ...]
    print(np.shape(image_val))

    # Apply the model on the tensor of the image generated above
    fc7 = sess.run(graph.get_tensor_by_name("import/fc7_relu:0"), feed_dict={images: image_val})
    generated_word_index = sess.run(generated_words_tf, feed_dict={fc7_tf: fc7})

    # Stack the words one after the other
    generated_word_index = np.hstack(generated_word_index)
    generated_words = [ixtoword[x] for x in generated_word_index]

    # Add a "." at the end of the caption generated
    punctuation = np.argmax(np.array(generated_words) == '.')+1
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)

    # Display the generated caption
    print(generated_sentence)


''' Feed an image as input to check the caption generated'''
generate_captions('/Users/nikhilshekhar/deep_learning_working_directory/Project-2/flickr30k-images/134206.jpg')

