#!/usr/bin/env bash

for i in $(seq 1 1 22)
do
#    nohup python Run_FCN_FineTune.py $i > out.FCN.RUN.$i.out
   nohup python Run_FCN_GatherData_FineTune.py $i > out.FCN.Gather.$i.out
done
