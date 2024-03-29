#!/bin/bash

# Request resources:
# -N: Number of nodes
# -n: Number of tasks
# -c: Number of CPUs per task
# --mem: Memory per node (in MB)
# -t: Time limit (hours:minutes:seconds)
# Replace the values according to your requirements

salloc -N 1 \
       -n 1 \
       -c 5 \
       --mem=10GB \
       -t 5:00:00 \
       --gres=gpu:a100:1 \
