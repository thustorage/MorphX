#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_name> <test_type>"
    exit 1
fi
if [ "$2" == "smsched" ]; then
    LD_PRELOAD=/home/rtx/gpu/sm-sched/runtime/build/libstub.so \
        /home/rtx/miniconda3/envs/smsched/bin/python ./test-${1}.py \
        | grep -Ev "cutlass|smsched" > ./logs/${1}.${2}.log
else
    /home/rtx/miniconda3/envs/baseline/bin/python ./test-${1}.py \
        | grep -Ev "cutlass|smsched" > ./logs/${1}.${2}.log
fi