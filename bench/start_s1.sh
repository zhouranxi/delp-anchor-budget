#!/bin/bash
# start_s1.sh -- server-side launcher (setsid-detached so ssh exit cannot kill it)
cd /mnt/mydisk/xinguan-dendritic/bench
setsid nohup bash launch_s1.sh 0 mnist     > s1_gpu0.log 2>&1 &
setsid nohup bash launch_s1.sh 1 fashion   > s1_gpu1.log 2>&1 &
setsid nohup bash launch_s1.sh 2 refsl1000 > s1_gpu2.log 2>&1 &
setsid nohup bash launch_s1.sh 3 l1000t20  > s1_gpu3.log 2>&1 &
