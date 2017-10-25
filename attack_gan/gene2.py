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

def batch_norm_layer(x,training_phase,scope_bn,activation=None):
  return tf.cond(training_phase,
  lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
  updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
  lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
  updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))



def build_generator(x_data, is_training, FLAGS):
    # Small epsilon value for the BN transform
    epsilon = 1e-3;
    # Reference model : Learning Deconvlutional Network for Semantic Segmentation
    #conv_layer maximum is 9
    conv_layer_num=3;
    fil_num_list=[64,128,256];
    c_W=np.zeros(conv_layer_num,dtype=object);
    c_b=np.zeros(conv_layer_num,dtype=object);
    conv=np.zeros(conv_layer_num,dtype=object);
    mean_variance_var_list=[];
    input_data = batch_norm_layer(x_data,training_phase=is_training,scope_bn='gene_bn_input',activation=tf.identity)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_input');
    # conv layers
    c_W[0] = tf.Variable(tf.truncated_normal([3, 3, 3, fil_num_list[0]], stddev=0.1))
    c_b[0] = tf.Variable(tf.constant(0.1, shape=[fil_num_list[0]]))
    conv_res = tf.nn.conv2d(input_data, c_W[0], strides=[1, 2, 2, 1], padding='SAME')+c_b[0];
    conv[0] = batch_norm_layer(conv_res,training_phase=is_training,scope_bn='gene_bn_conv_0',activation=tf.nn.relu)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_conv_0');
    for i in range(1,conv_layer_num):
      c_W[i] = tf.Variable(tf.truncated_normal([3, 3, fil_num_list[i-1], fil_num_list[i]], stddev=0.1))
      c_b[i] = tf.Variable(tf.constant(0.1, shape=[fil_num_list[i]]))
      conv_res = tf.nn.conv2d(conv[i-1], c_W[i], strides=[1, 2, 2, 1], padding='SAME')+c_b[i];
      conv[i] = batch_norm_layer(conv_res,training_phase=is_training,scope_bn='gene_bn_conv_'+str(i),activation=tf.nn.relu)
      mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_conv_'+str(i));
    # fc layer
    f_w = tf.Variable(tf.truncated_normal([1, 1, fil_num_list[conv_layer_num-1], fil_num_list[conv_layer_num-1]], stddev=0.1))
    f_b = tf.Variable(tf.constant(0.1, shape=[fil_num_list[conv_layer_num-1]]))
    net_res = tf.nn.conv2d(conv[conv_layer_num-1], f_w, strides=[1, 1, 1, 1], padding='SAME')+f_b;
    net = batch_norm_layer(net_res,training_phase=is_training,scope_bn='gene_bn_fc',activation=tf.nn.relu)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_fc');
    # deconv layers
    d_W=np.zeros(conv_layer_num,dtype=object);
    d_b=np.zeros(conv_layer_num,dtype=object);
    for i in range(1,conv_layer_num):
      idx=conv_layer_num-i;
      d_W[idx] = tf.Variable(tf.truncated_normal([3, 3, fil_num_list[idx-1], fil_num_list[idx]], stddev=0.1))
      d_b[idx] = tf.Variable(tf.constant(0.1, shape=[fil_num_list[idx-1]]))
      x_shape = tf.shape(conv[idx-1]);
      out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], fil_num_list[idx-1]])
      deconv_res = tf.nn.conv2d_transpose(net, d_W[idx], output_shape=out_shape ,strides=[1, 2, 2, 1], padding='SAME')+d_b[idx]
      net = batch_norm_layer(deconv_res,training_phase=is_training,scope_bn='gene_bn_deconv_'+str(idx),activation=tf.nn.relu)
      mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_deconv_'+str(idx));
    d_W[0] = tf.Variable(tf.truncated_normal([3, 3, 3, fil_num_list[0]], stddev=0.1))
    d_b[0] = tf.Variable(tf.constant(0.1, shape=[3]))
    x_shape = tf.shape(x_data)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], 3])
    net = tf.nn.conv2d_transpose(net, d_W[0], output_shape=out_shape ,strides=[1, 2, 2, 1], padding='SAME')+d_b[0]
    x_generate = batch_norm_layer(net,training_phase=is_training,scope_bn='gene_bn_deconv_0',activation=tf.nn.tanh)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_deconv_0');
    
    bn_var_num=len(mean_variance_var_list); 
    g_params=list(c_W)+list(c_b)+[f_w,f_b]+list(d_W)+list(d_b)+mean_variance_var_list;
    
    return x_generate, g_params, bn_var_num

