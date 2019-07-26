#!/bin/bash
LABEL=default
MODEL=tiny16
DROPOUT=0
DROPEDGE=0
DEPTHRATE=0.5
EXPAND_POLICY=RandomPolicy
MONITOR_PARAM=max
MONITOR_STAT=cma

# Options can be passes by arguments
while getopts l:m:o:e:r:p:q:s: option
do
    case "${option}"
        in
        l) LABEL=${OPTARG};;
        m) MODEL=${OPTARG};;
        o) DROPOUT=${OPTARG};;
        e) DROPEDGE=${OPTARG};;
        r) DEPTHRATE=${OPTARG};;
        p) EXPAND_POLICY=${OPTARG};;
        q) MONITOR_PARAM=${OPTARG};;
        s) MONITOR_STAT=${OPTARG};;
    esac
done

CHECKPOINT=ckpt_$LABEL.pth
LABEL=$LABEL-rst-do0

# Remove previous works with the same label
rm -rf log/$LABEL
rm -rf checkpoint/$LABEL
rm -rf graph/$LABEL

# First run with reset model weights
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --monitor-param=$MONITOR_PARAM \
    --monitor-stat=$MONITOR_STAT \
    --expand-period=10000 \
    --expand-policy=$EXPAND_POLICY \
    --lr=2e-1 \
    --num-epochs=140 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT \
    --reset-model

# Rename the checkpoint and resume running
CHECKPOINT=ckpt_$LABEL.pth
cp checkpoint/$LABEL/ckpt_done139.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --monitor-param=$MONITOR_PARAM \
    --monitor-stat=$MONITOR_STAT \
    --expand-period=10000 \
    --expand-policy=$EXPAND_POLICY \
    --lr=2e-2 \
    --num-epochs=60 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_done199.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --monitor-param=$MONITOR_PARAM \
    --monitor-stat=$MONITOR_STAT \
    --expand-period=10000 \
    --expand-policy=$EXPAND_POLICY \
    --lr=2e-3 \
    --num-epochs=40 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
