#!/usr/bin/env bash


for j in $(seq 2 1 33)
do
for i in $(seq 20 1 20)
do
   echo python Run_NN_FineTune_ByYear.py $i $j #> out.RUN.$i.$j.out
   echo python Run_NN_GatherData_FineTune_ByYear.py $i $j #> out.Gather.$i.$j.out
done
done
