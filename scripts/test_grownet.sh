#!/bin/bash
LABEL=default
MODEL=tiny16
DROPOUT=0
DROPEDGE=0
DEPTHRATE=0.5
POLICY=RandomPolicy

# Options can be passes by arguments
while getopts l:m:o:e:r:p: option
do
    case "${option}"
        in
        l) LABEL=${OPTARG};;
        m) MODEL=${OPTARG};;
        o) DROPOUT=${OPTARG};;
        e) DROPEDGE=${OPTARG};;
        r) DEPTHRATE=${OPTARG};;
        p) POLICY=${OPTARG};;
    esac
done

CHECKPOINT=ckpt_$LABEL.pth

# Test GrowNet
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=10000 \
    --expand-policy=$POLICY \
    --lr=2e-1 \
    --num-epochs=1 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT \
    --test-only
