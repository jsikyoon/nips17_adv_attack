"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import FastGradientMethod
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

from gene2 import build_generator

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    images[i][images[i]<-1.0]=-1.0;
    images[i][images[i]>1.0]=1.0;
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 256.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    is_training = tf.placeholder(tf.bool);
    x_adv,g_params,bn_var_num=build_generator(x_input,is_training,FLAGS);
    x_adv=x_adv*eps;
    x_adv=x_input+x_adv;

    # Run computation
    saver=tf.train.Saver(g_params);
 
    epsilon_value=int(FLAGS.max_epsilon);
    with tf.train.MonitoredSession() as sess:
      #saver.restore(sess,"my-models_iv3_rv250_rv2152_rv2101_iv4_avg2/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_rv250_rv2152_rv2101_iv4_avg/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_rv250_rv2152_rv2101_iv4/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_rv250_rv2152_avg2/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_rv250_rv2152/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_rv250_avg2/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_rv250/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_r3/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"aaa/my-model_"+str(epsilon_value)+".ckpt");
      #saver.restore(sess,"my-models_iv3_conv6/my-model_"+str(epsilon_value)+".ckpt");
      saver.restore(sess,"mark3_iv3_rv250_rv2152_rv2101_iv4_avg2/my-model_"+str(epsilon_value)+".ckpt");
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images,is_training:False})
        save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
