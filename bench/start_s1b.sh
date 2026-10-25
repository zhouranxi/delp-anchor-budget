#!/bin/bash
# start_s1b.sh -- server-side setsid launch for S1 B (4 GPUs, nohup-proof)
cd /mnt/mydisk/xinguan-dendritic/bench
rm -f s1b_done.marker
setsid bash launch_s1b.sh 0 mnist 10 s1b_mnist_t10.json </dev/null >/dev/null 2>&1 &
setsid bash launch_s1b.sh 1 mnist 20 s1b_mnist_t20.json </dev/null >/dev/null 2>&1 &
setsid bash launch_s1b.sh 2 fashion 10 s1b_fashion_t10.json </dev/null >/dev/null 2>&1 &
setsid bash launch_s1b.sh 3 fashion 20 s1b_fashion_t20.json </dev/null >/dev/null 2>&1 &
echo "launched 4 shards"
