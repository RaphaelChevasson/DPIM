#!/usr/bin/env bash


# Activate environment
cd ../ProtAugment
source venv-protaugment/bin/activate

# Source the .envrc file, if it exists
if [[ -f ".envrc" ]]; then
    source .envrc
fi

# Simple check on the gpu we will be using
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "hostname: $(hostname)"
command -v nvidia-smi >/dev/null && {
    echo "GPU Devices:"
    nvidia-smi
} || {
    :
}

# Run the script
cd ../ProtAugment
PYTHONPATH=. python models/proto/protaugment.py $@
