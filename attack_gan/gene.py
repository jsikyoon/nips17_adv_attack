import tensorflow as tf
import numpy as np

def conv_variable(weight_shape):
  w = weight_shape[0]
  h = weight_shape[1]
  input_channels  = weight_shape[2]
  output_channels = weight_shape[3]
  d = 1.0 / np.sqrt(input_channels * w * h)
  bias_shape = [output_channels]
  weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
  bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
  return weight, bias

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def build_generator(x_data, FLAGS):
    # Reference model : Learning Deconvlutional Network for Semantic Segmentation
    w_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_1 = tf.nn.conv2d(x_data, w_1, strides=[1, 2, 2, 1], padding='SAME')
    conv_1 = tf.nn.relu(h_1 + b_1)

    w_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_2 = tf.nn.conv2d(conv_1, w_2, strides=[1, 2, 2, 1], padding='SAME')
    conv_2 = tf.nn.relu(h_2 + b_2)

    w_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.1, shape=[256]))
    h_3 = tf.nn.conv2d(conv_2, w_3, strides=[1, 2, 2, 1], padding='SAME')
    conv_3 = tf.nn.relu(h_3 + b_3)

    w_f_4 = tf.Variable(tf.truncated_normal([1, 1, 256, 256], stddev=0.1))
    b_f_4 = tf.Variable(tf.constant(0.1, shape=[256]))
    h_f_4 = tf.nn.conv2d(conv_3, w_f_4, strides=[1, 1, 1, 1], padding='SAME')
    fc_4 = tf.nn.relu(h_f_4 + b_f_4)
    
    d_w_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
    d_b_3 = tf.Variable(tf.constant(0.1, shape=[128]))
    x_shape = tf.shape(conv_2)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], 128])
    d_h_3 = tf.nn.conv2d_transpose(fc_4, d_w_3, output_shape=out_shape ,strides=[1, 2, 2, 1], padding='SAME')
    deconv_3 = d_h_3 + d_b_3

    d_w_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
    d_b_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    x_shape = tf.shape(conv_1)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], 64])
    d_h_2 = tf.nn.conv2d_transpose(deconv_3, d_w_2, output_shape=out_shape ,strides=[1, 2, 2, 1], padding='SAME')
    deconv_2 = d_h_2 + d_b_2

    d_w_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
    d_b_1 = tf.Variable(tf.constant(0.1, shape=[3]))
    x_shape = tf.shape(x_data)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], 3])
    d_h_1 = tf.nn.conv2d_transpose(deconv_2, d_w_1, output_shape=out_shape ,strides=[1, 2, 2, 1], padding='SAME')
    deconv_1 = d_h_1 + d_b_1

    x_generate = tf.nn.tanh(deconv_1)

    g_params=[w_1, b_1, w_2, b_2, w_3, b_3, w_f_4, b_f_4, d_w_3, d_b_3, d_w_2, d_b_2, d_w_1, d_b_1]

    return x_generate, g_params

