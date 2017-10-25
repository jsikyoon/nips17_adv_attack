#!/bin/bash

for i in {5..16}
do
  echo $i "Learning Start " `date`
  python learn_gan_ll.py --max_epsilon=$i
  echo "Done"
done
