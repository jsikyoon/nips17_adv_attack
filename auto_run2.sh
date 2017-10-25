#!/bin/bash

eps=10

cd attack_gan/
./run_attack.sh $eps
cd ../defense
python defense.py --max_epsilon=$eps --checkpoint_file_name=inception_v3.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=resnet_v2_50.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=inception_v4.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=inception_resnet_v2_2016_08_30.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=resnet_v2_101.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=resnet_v2_152.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=inception_v1.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=inception_v2.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=vgg_16.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=vgg_19.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=resnet_v1_50.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=resnet_v1_101.ckpt
python defense.py --max_epsilon=$eps --checkpoint_file_name=resnet_v1_152.ckpt
cd ..

