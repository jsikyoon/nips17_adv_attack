#!/bin/bash

for i in {4..8}
do
  # FGSM
  #cd sample_attacks/fgsm
  #./run_attack.sh $i
  #cd ../../defense
  # RANDOM NOISE
  #cd sample_attacks/random_noise
  #./run_attack.sh $i
  #cd ../../defense
  # DECONV
  cd attack_gan/
  #cd iter_llcm/
  ./run_attack.sh $i
  cd ../defense
  python defense.py --max_epsilon=$i
  cd ..
done

