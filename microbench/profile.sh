#!/bin/bash
for sms in {8..108..8}; do
    echo "Running green_ctx_bench with $sms SMs"
    ./build/green_ctx_bench $sms | grep "Context 2 finished" >> profile_green_ctx_bench.txt
done