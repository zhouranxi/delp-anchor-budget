#!/bin/bash
# start_r8.sh -- server-side launcher (setsid-detached so ssh exit cannot kill it)
cd /mnt/mydisk/xinguan-dendritic/bench
setsid nohup bash launch_r8.sh 0 gamma05 > r8_gpu0.log 2>&1 &
setsid nohup bash launch_r8.sh 1 gamma09 > r8_gpu1.log 2>&1 &
setsid nohup bash launch_r8.sh 2 mgrid  > r8_gpu2.log 2>&1 &
setsid nohup bash launch_r8.sh 3 merge  > r8_gpu3.log 2>&1 &
