#!/usr/bin/env bash

for i in $(seq 1 1 22)
do
   nohup python Run_EMOS_finetune.py $i > out.EMOS.RUN.$i.out
   nohup python Run_EMOS_GatherData.py $i > out.EMOS.Gather.$i.out
done
