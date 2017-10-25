"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
from scipy.misc import imread

import tensorflow as tf
#from tensorflow.contrib.slim.nets import inception
from nets import inception
from nets import vgg
from nets import resnet_v2
from nets import resnet_v1
from nets import resnet_utils
from nets import vgg_preprocessing

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '/tmp/nips17/trained_models/', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_file_name', 'inception_v3.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'resnet_v2_50.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'inception_v4.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'inception_resnet_v2_2016_08_30.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'resnet_v2_101.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'resnet_v2_152.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'inception_v1.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'inception_v2.ckpt', 'checkpoint file name for inception network.')

#     'checkpoint_file_name', 'vgg_16.ckpt', 'checkpoint file name for inception network.')
#     'checkpoint_file_name', 'vgg_19.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'resnet_v1_50.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'resnet_v1_101.ckpt', 'checkpoint file name for inception network.')
#    'checkpoint_file_name', 'resnet_v1_152.ckpt', 'checkpoint file name for inception network.')

tf.flags.DEFINE_integer(
    'max_epsilon', 4, 'max infi norm')

tf.flags.DEFINE_string(
    'origin_img_dir', '/tmp/nips17/imgs/', 'Input directory with images.')

tf.flags.DEFINE_string(
    'input_dir', '/tmp/nips17/adv_imgs/', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', 'output_file', 'Output file to save labels.')

tf.flags.DEFINE_string(
    'true_label', 'true_label', 'True label file')

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
    if(FLAGS.checkpoint_file_name=="vgg_16.ckpt")or(FLAGS.checkpoint_file_name=="vgg_19.ckpt")or(FLAGS.checkpoint_file_name=="resnet_v1_50.ckpt")or(FLAGS.checkpoint_file_name=="resnet_v1_101.ckpt")or(FLAGS.checkpoint_file_name=="resnet_v1_152.ckpt"):
      with tf.gfile.Open(filepath) as f:
        image = imread(f, mode='RGB').astype(np.float)
      images[idx, :, :, :] = image
    else:
      with tf.gfile.Open(filepath) as f:
        image = imread(f, mode='RGB').astype(np.float) / 255.0
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


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  # max_epsilon over checking
  # get original images
  origin_img_list=np.sort(glob.glob(FLAGS.origin_img_dir+"*.png"));
  origin_imgs=np.zeros((len(origin_img_list),FLAGS.image_height,FLAGS.image_width,3),dtype=float);
  for i in range(len(origin_img_list)):
    origin_imgs[i]=imread(origin_img_list[i],mode='RGB').astype(np.float);
  # get adv images
  adv_img_list=np.sort(glob.glob(FLAGS.input_dir+"*.png"));
  adv_imgs=np.zeros((len(adv_img_list),FLAGS.image_height,FLAGS.image_width,3),dtype=float);
  for i in range(len(adv_img_list)):
    adv_imgs[i]=imread(adv_img_list[i],mode='RGB').astype(np.float);
  epsilon_list=np.linalg.norm(np.reshape(abs(origin_imgs-adv_imgs),[-1,FLAGS.image_height*FLAGS.image_width*3]),ord=np.inf,axis=1);
  #print(epsilon_list);exit(1);
  over_epsilon_list=np.zeros((len(origin_img_list),2),dtype=object);
  cnt=0;
  for i in range(len(origin_img_list)):
    file_name=origin_img_list[i].split("/")[-1];
    file_name=file_name.split(".")[0];
    over_epsilon_list[i,0]=file_name;
    if(epsilon_list[i]>FLAGS.max_epsilon):
      over_epsilon_list[i,1]="1";
      cnt+=1;
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    if(FLAGS.checkpoint_file_name=="inception_v3.ckpt"):
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
            x_input, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['Predictions'], 1)
    elif(FLAGS.checkpoint_file_name=="inception_v4.ckpt"):
      with slim.arg_scope(inception.inception_v4_arg_scope()):
        _, end_points = inception.inception_v4(
            x_input, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['Predictions'], 1)
    elif(FLAGS.checkpoint_file_name=="inception_resnet_v2_2016_08_30.ckpt"):
      with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        _, end_points = inception.inception_resnet_v2(
            x_input, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['Predictions'], 1)
    elif(FLAGS.checkpoint_file_name=="resnet_v2_101.ckpt"):
      x_input2 = tf.image.resize_bilinear(x_input,[224,224],align_corners=False);
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet_v2.resnet_v2_101(
            x_input2, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['predictions'], 1)
    elif(FLAGS.checkpoint_file_name=="resnet_v2_50.ckpt"):
      x_input2 = tf.image.resize_bilinear(x_input,[224,224],align_corners=False);
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet_v2.resnet_v2_50(
            x_input2, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['predictions'], 1)
    elif(FLAGS.checkpoint_file_name=="resnet_v2_152.ckpt"):
      x_input2 = tf.image.resize_bilinear(x_input,[224,224],align_corners=False);
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet_v2.resnet_v2_152(
            x_input2, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['predictions'], 1)
    elif(FLAGS.checkpoint_file_name=="inception_v1.ckpt"):
      x_input2 = tf.image.resize_bilinear(x_input,[224,224],align_corners=False);
      with slim.arg_scope(inception.inception_v1_arg_scope()):
        _, end_points = inception.inception_v1(
            x_input2, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['Predictions'], 1)
    elif(FLAGS.checkpoint_file_name=="inception_v2.ckpt"):
      x_input2 = tf.image.resize_bilinear(x_input,[224,224],align_corners=False);
      with slim.arg_scope(inception.inception_v2_arg_scope()):
        _, end_points = inception.inception_v2(
            x_input2, num_classes=num_classes, is_training=False)
      predicted_labels = tf.argmax(end_points['Predictions'], 1)

    # Resnet v1 and vgg are not working now
    elif(FLAGS.checkpoint_file_name=="vgg_16.ckpt"):
      x_input_list=tf.unstack(x_input,FLAGS.batch_size,0);
      for i in range(FLAGS.batch_size):
        x_input_list[i]=vgg_preprocessing.preprocess_image(x_input_list[i],224,224);
      x_input2=tf.stack(x_input_list,0);
      with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg.vgg_16(
            x_input2, num_classes=num_classes-1, is_training=False)
      predicted_labels = tf.argmax(end_points['vgg_16/fc8'], 1)+1
    elif(FLAGS.checkpoint_file_name=="vgg_19.ckpt"):
      x_input_list=tf.unstack(x_input,FLAGS.batch_size,0);
      for i in range(FLAGS.batch_size):
        x_input_list[i]=vgg_preprocessing.preprocess_image(x_input_list[i],224,224);
      x_input2=tf.stack(x_input_list,0);
      with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg.vgg_19(
            x_input2, num_classes=num_classes-1, is_training=False)
      predicted_labels = tf.argmax(end_points['vgg_19/fc8'], 1)+1
    elif(FLAGS.checkpoint_file_name=="resnet_v1_50.ckpt"):
      x_input_list=tf.unstack(x_input,FLAGS.batch_size,0);
      for i in range(FLAGS.batch_size):
        x_input_list[i]=vgg_preprocessing.preprocess_image(x_input_list[i],224,224);
      x_input2=tf.stack(x_input_list,0);
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet_v1.resnet_v1_50(
            x_input, num_classes=num_classes-1, is_training=False)
      predicted_labels = tf.argmax(end_points['predictions'], 1)+1
    elif(FLAGS.checkpoint_file_name=="resnet_v1_101.ckpt"):
      x_input_list=tf.unstack(x_input,FLAGS.batch_size,0);
      for i in range(FLAGS.batch_size):
        x_input_list[i]=vgg_preprocessing.preprocess_image(x_input_list[i],224,224);
      x_input2=tf.stack(x_input_list,0);
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet_v1.resnet_v1_101(
            x_input2, num_classes=num_classes-1, is_training=False)
      predicted_labels = tf.argmax(end_points['predictions'], 1)+1
    elif(FLAGS.checkpoint_file_name=="resnet_v1_152.ckpt"):
      x_input_list=tf.unstack(x_input,FLAGS.batch_size,0);
      for i in range(FLAGS.batch_size):
        x_input_list[i]=vgg_preprocessing.preprocess_image(x_input_list[i],224,224);
      x_input2=tf.stack(x_input_list,0);
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet_v1.resnet_v1_152(
            x_input2, num_classes=num_classes-1, is_training=False)
      predicted_labels = tf.argmax(end_points['predictions'], 1)+1
    
    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path+FLAGS.checkpoint_file_name,
        master=FLAGS.master)

    f=open(FLAGS.true_label,"r");
    t_label_list=np.array([i[:-1].split(",") for i in f.readlines()]);
    
    score=0;
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
          labels = sess.run(predicted_labels, feed_dict={x_input: images})
          for filename, label in zip(filenames, labels):
            f_name=filename.split(".")[0];
            t_label=int(t_label_list[t_label_list[:,0]==f_name,1][0]);
            if(t_label!=label):
              if(over_epsilon_list[over_epsilon_list[:,0]==f_name,1]!="1"):
                score+=1;
            #out_file.write('{0},{1}\n'.format(filename, label))
  print("Over max epsilon#: "+str(cnt));
  print(str(FLAGS.max_epsilon)+" max epsilon Score: "+str(score));


if __name__ == '__main__':
  tf.app.run()
