import tensorflow as tf
import numpy as np
import os,sys,glob
import shutil
import argparse
from scipy.misc import imread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gene import build_generator
#from disc import InceptionModel

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

FLAGS=None;

def show_result(data,img_file):
  data[0]=(data[0]+1.0)/2.0;
  fig=plt.figure();
  plt.axis('off');
  plt.imshow(data[0]);
  fig.savefig(img_file);

def train():
  eps=2.0*float(FLAGS.max_epsilon)/256.0;
  tf.logging.set_verbosity(tf.logging.INFO);
  with tf.Graph().as_default():
    # Design architecture
    # input
    x_data = tf.placeholder(tf.float32, [None, FLAGS.img_height, FLAGS.img_width,3], name="x_data")
    y_label = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name="y_label")
    lam = tf.placeholder(tf.float32, [], name="lambda");

    # generator
    x_generated, g_params = build_generator(x_data,FLAGS);
    #x_generated = x_generated * eps;
    x_generated=tf.nn.l2_normalize(x_generated,3)*eps;
    
    x_generated = x_data + x_generated;

    # discriminator 
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_generated, num_classes=FLAGS.num_classes, is_training=False)
    predicted_labels = tf.argmax(end_points['Predictions'], 1);
    predicted_logits = end_points['Logits'];

    # loss and optimizer
    gen_acc=tf.reduce_mean(tf.cast(tf.equal(predicted_labels,tf.argmax(y_label,1)),tf.float32));

    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_logits, labels=y_label));

    infi_norm=tf.reduce_mean(tf.norm(tf.reshape(abs(x_data-x_generated),[-1,FLAGS.img_size]),ord=np.inf,axis=1));
    #infi_norm=tf.reduce_mean(tf.norm(tf.reshape(abs(x_data-x_generated),[-1,FLAGS.img_size]),ord=2,axis=1));

    #g_loss = 1/cross_entropy+infi_norm;
    #g_loss = -1*cross_entropy+0.3*infi_norm;
    #g_loss = infi_norm;
    #g_loss = gen_acc+0.01*infi_norm;
    
    g_loss=-1*cross_entropy + gen_acc;
    #g_loss = -1/cross_entropy+tf.nn.relu(infi_norm-16.0*2.0/256.0);

    optimizer = tf.train.AdamOptimizer(0.0001)

    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    # get the data and label
    img_list=np.sort(glob.glob(FLAGS.input_folder+"*.png"));
    total_data=np.zeros((len(img_list),FLAGS.img_height,FLAGS.img_width,3),dtype=float);
    for i in range(len(img_list)):
      total_data[i]=imread(img_list[i],mode='RGB').astype(np.float) / 255.0;
      total_data[i]=total_data[i]*2.0-1.0;  # 0~1 -> -1~1
    val_data=np.copy(total_data[0]);
    f=open(FLAGS.label_folder+"true_label","r");
    total_label2=np.array([i[:-1].split(",")[1] for i in f.readlines()],dtype=int);
    total_label=np.zeros((len(total_data),FLAGS.num_classes),dtype=int);
    for i in range(len(total_data)):
      total_label[i,total_label2[i]]=1;
    val_label=np.copy(total_label[0]);

    # shuffle
    total_idx=range(len(total_data)); np.random.shuffle(total_idx);
    total_data=total_data[total_idx];total_label=total_label[total_idx];

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables());
    saver2 = tf.train.Saver(g_params);
    
    """
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)
    """
    # initialization
    init = tf.global_variables_initializer();
    with tf.Session() as sess:
    #with tf.train.MonitoredSession() as sess:
      sess.run(init)
      saver.restore(sess,FLAGS.checkpoint_path);
      # tf board
      tf.summary.scalar('reverse_cross_entropy',cross_entropy);
      tf.summary.scalar('training_accuracy',gen_acc); 
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter('tmp/nips17/attack_gan/learn_gan', sess.graph)
      # training
      for i in range(FLAGS.max_epoch):
        if(i>100):
          lam_value=1.0;
        else:
          lam_value=1.0;
        tr_ce=0;
        tr_infi=0;
        tr_gen_acc=0;
        for j in range(len(total_data) / FLAGS.batch_size):
          batch_data=total_data[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size];
          batch_label=total_label[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size];
          tr_gen_acc_part,tr_ce_part,tr_infi_part,_=sess.run([gen_acc,cross_entropy,infi_norm,g_trainer],feed_dict={x_data: batch_data, y_label: batch_label,lam:lam_value});
          tr_ce+=tr_ce_part;
          tr_infi+=tr_infi_part;
          tr_gen_acc+=tr_gen_acc_part;
        print(str(i+1)+" Epoch Training Cross Entropy: "+str(tr_ce/(j+1))+", Infinity Norm: "+str(tr_infi/(j+1))+",Gen Acc: "+str(tr_gen_acc/(j+1)));
        total_idx=range(len(total_data)); np.random.shuffle(total_idx);
        total_data=total_data[total_idx];total_label=total_label[total_idx];
        #x_gen_val = sess.run(x_generated,feed_dict={x_data:[val_data],y_label:[val_label],lam:lam_value});
        #show_result(x_gen_val, FLAGS.output_folder+"sample{0}.jpg".format(i+1));
      saver2.save(sess,"my-models_iv_r3/my-model_"+str(FLAGS.max_epsilon)+".ckpt");

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_height', type=int, default=299,
                      help='Image Height')
  parser.add_argument('--img_width', type=int, default=299,
                      help='Image Width')
  parser.add_argument('--num_classes', type=int, default=1001,
                      help='The number of classes')
  parser.add_argument('--input_folder', type=str, default='/tmp/nips17/imgs/',
                      help='input folder path')
  parser.add_argument('--label_folder', type=str, default='',
                      help='label folder path')
  parser.add_argument('--output_folder', type=str, default='/tmp/nips17/adv_imgs/',
                      help='output folder path')
  parser.add_argument('--max_epoch', type=int, default=1000,
                      help='Max Epoch')
  parser.add_argument('--max_epsilon', type=int, default=4,
                      help='Max Epsilon')
  parser.add_argument('--batch_size', type=int, default=50,
                      help='The size of batches')
  parser.add_argument('--master', type=str, default='',
                      help='The address of the TensorFlow master to use.')
  parser.add_argument('--checkpoint_path', type=str, default='/tmp/nips17/inception_v3.ckpt',
                      help='Path to checkpoint for inception network.')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.img_size=FLAGS.img_height*FLAGS.img_width*3;
  train()
