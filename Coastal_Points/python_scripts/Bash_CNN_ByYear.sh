#!/usr/bin/env bash


for j in $(seq 2 1 33)
do
for i in $(seq 9 1 9)
do
   nohup python Run_NN_FineTune_ByYear.py $i $j > out.CNN.RUN.$i.$j.out
   nohup python Run_NN_GatherData_FineTune_ByYear.py $i $j > out.CNN.Gather.$i.$j.out
done
done
