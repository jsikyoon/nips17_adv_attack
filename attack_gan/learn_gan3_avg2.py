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

from nets import vgg
from nets import inception
from nets import resnet_v2
from nets import resnet_utils

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

    # generator
    x_generated, g_params = build_generator(x_data,FLAGS);
    x_generated = x_generated * eps;
    
    x_generated = x_data + x_generated;

    # discriminator(inception v3) 
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_generated, num_classes=FLAGS.num_classes, is_training=False)
    predicted_labels = tf.argmax(end_points['Predictions'], 1);
    predicted_logits = end_points['Logits'];
    disc_var_list=slim.get_model_variables();
    # discriminator(resnet v2 50)
    x_generated2=tf.image.resize_bilinear(x_generated,[224,224],align_corners=False); 
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points2 = resnet_v2.resnet_v2_50(
          x_generated2, num_classes=FLAGS.num_classes, is_training=False)
    predicted_labels2 = tf.argmax(end_points2['predictions'], 1);
    predicted_logits2 = end_points2['predictions'];
    disc_var_list2=slim.get_model_variables()[len(disc_var_list):];
    # discriminator(resnet v2 152)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points3 = resnet_v2.resnet_v2_152(
          x_generated2, num_classes=FLAGS.num_classes, is_training=False)
    predicted_labels3 = tf.argmax(end_points3['predictions'], 1);
    predicted_logits3 = end_points3['predictions'];
    disc_var_list3=slim.get_model_variables()[(len(disc_var_list)+len(disc_var_list2)):];

    # average
    predicted_prob_avg = (end_points['Predictions']+end_points2['predictions']+end_points3['predictions'])/5.0;
    predicted_labels_avg = tf.argmax((end_points['Predictions']+end_points2['predictions']+end_points3['predictions'])/5.0, 1);

    # loss and optimizer
    gen_acc=tf.reduce_mean(tf.cast(tf.equal(predicted_labels,tf.argmax(y_label,1)),tf.float32));
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_logits, labels=y_label));
    gen_acc2=tf.reduce_mean(tf.cast(tf.equal(predicted_labels2,tf.argmax(y_label,1)),tf.float32));
    cross_entropy2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_logits2, labels=y_label));
    gen_acc3=tf.reduce_mean(tf.cast(tf.equal(predicted_labels3,tf.argmax(y_label,1)),tf.float32));
    cross_entropy3=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_logits3, labels=y_label));
    gen_acc_avg=tf.reduce_mean(tf.cast(tf.equal(predicted_labels_avg,tf.argmax(y_label,1)),tf.float32));
    cross_entropy_avg=tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(predicted_prob_avg),1));
    infi_norm=tf.reduce_mean(tf.norm(tf.reshape(abs(x_data-x_generated),[-1,FLAGS.img_size]),ord=np.inf,axis=1));
    
    g_loss=-1*cross_entropy_avg;

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
    saver = tf.train.Saver(disc_var_list);
    saver2 = tf.train.Saver(disc_var_list2);
    saver3 = tf.train.Saver(disc_var_list3);
    saver_gen = tf.train.Saver(g_params);
    
    # initialization
    init = tf.global_variables_initializer();
    with tf.Session() as sess:
      sess.run(init)
      saver.restore(sess,FLAGS.checkpoint_path+FLAGS.checkpoint_file_name);
      saver2.restore(sess,FLAGS.checkpoint_path+FLAGS.checkpoint_file_name2);
      saver3.restore(sess,FLAGS.checkpoint_path+FLAGS.checkpoint_file_name3);
      # training
      for i in range(FLAGS.max_epoch):
        tr_infi=0;
        tr_ce=0;
        tr_gen_acc=0;
        tr_ce2=0;
        tr_gen_acc2=0;
        tr_ce3=0;
        tr_gen_acc3=0;
        tr_ce_avg=0;
        tr_gen_acc_avg=0;
        for j in range(len(total_data) / FLAGS.batch_size):
          batch_data=total_data[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size];
          batch_label=total_label[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size];
          acc_p_a,ce_p_a,acc_p3,ce_p3,acc_p2,ce_p2,acc_p,ce_p,infi_p,_=sess.run([gen_acc_avg,cross_entropy_avg,gen_acc3,cross_entropy3,gen_acc2,cross_entropy2,gen_acc,cross_entropy,infi_norm,g_trainer],feed_dict={x_data: batch_data, y_label: batch_label});
          tr_infi+=infi_p;
          tr_ce+=ce_p;
          tr_gen_acc+=acc_p;
          tr_ce2+=ce_p2;
          tr_gen_acc2+=acc_p2;
          tr_ce3+=ce_p3;
          tr_gen_acc3+=acc_p3;
          tr_ce_avg+=ce_p_a;
          tr_gen_acc_avg+=acc_p_a;
        print(str(i+1)+" Epoch InfiNorm:"+str(tr_infi/(j+1))+",CE: "+str(tr_ce/(j+1))+",Acc: "+str(tr_gen_acc/(j+1))+",CE2: "+str(tr_ce2/(j+1))+",Acc2: "+str(tr_gen_acc2/(j+1))+",CE3: "+str(tr_ce3/(j+1))+",Acc3: "+str(tr_gen_acc3/(j+1))+",Acc_avg: "+str(tr_gen_acc_avg/(j+1))+",CE_avg: "+str(tr_ce_avg/(j+1)));
        total_idx=range(len(total_data)); np.random.shuffle(total_idx);
        total_data=total_data[total_idx];total_label=total_label[total_idx];
      saver_gen.save(sess,"my-models_iv3_rv250_rv2152_avg2/my-model_"+str(FLAGS.max_epsilon)+".ckpt");

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
  parser.add_argument('--max_epoch', type=int, default=300,
                      help='Max Epoch')
  parser.add_argument('--max_epsilon', type=int, default=12,
                      help='Max Epsilon')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='The size of batches')
  parser.add_argument('--master', type=str, default='',
                      help='The address of the TensorFlow master to use.')
  parser.add_argument('--checkpoint_path', type=str, default='/tmp/nips17/trained_models/',
                      help='Path to checkpoint for inception network.')
  parser.add_argument('--checkpoint_file_name', type=str, default='inception_v3.ckpt',
                      help='checkpoint file name')
  parser.add_argument('--checkpoint_file_name2', type=str, default='resnet_v2_50.ckpt',
                      help='checkpoint file name2')
  parser.add_argument('--checkpoint_file_name3', type=str, default='resnet_v2_152.ckpt',
                      help='checkpoint file name3')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.img_size=FLAGS.img_height*FLAGS.img_width*3;
  train()
