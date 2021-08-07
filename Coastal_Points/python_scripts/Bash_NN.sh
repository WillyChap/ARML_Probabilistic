#!/usr/bin/env bash



for i in $(seq 1 1 22)
do
   nohup python Run_NN_FineTune.py $i > out.NN.RUN.$i.out
####    nohup python Run_CNN_GatherData_FineTune.py $i $j > out.CNN.Gather.$i.out
done
