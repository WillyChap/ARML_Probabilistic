#!/usr/bin/env bash



for i in $(seq 1 1 22)
do
   nohup python Run_CNN_FineTune.py $i > out.CNN.RUN.$i.out
####    nohup python Run_CNN_GatherData_FineTune.py $i $j > out.CNN.Gather.$i.out
done
