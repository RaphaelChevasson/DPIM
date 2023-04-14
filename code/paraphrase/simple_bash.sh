#!/usr/bin/env bash


cd /home_expes/chr06111/chr06111/NAG/paraphrase/

# Activate environment
source ../paraphrase/venv-paraphrase/bin/activate
source ../ewiser/venv-ewiser/bin/activate

# Simple check on the gpu we will be using
echo "node: $(hostname)"
echo "GPUs:"
# nvidia-smi --list-gpus
nvidia-smi --query-gpu=index,gpu_name,memory.total,driver_version --format=csv
echo "you are using GPU ${CUDA_VISIBLE_DEVICES}"
echo "your environment is $(which python)"

# Run the script
$@
