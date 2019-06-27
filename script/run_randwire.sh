#!/bin/bash
LABEL=tiny16-do0-conn-str1
MODEL=tiny16
DROPOUT=0
DROPEDGE=0

# Options can be passes by arguments
while getopts l:m:o:e: option
do
    case "${option}"
        in
        l) LABEL=${OPTARG};;
        m) MODEL=${OPTARG};;
        o) DROPOUT=${OPTARG};;
        e) DROPEDGE=${OPTARG};;
    esac
done

CHECKPOINT=ckpt_$LABEL.pth

# Remove previous works with the same label
rm -rf log/$LABEL
rm -rf checkpoint/$LABEL
rm -rf graph/$LABEL

# Train and evaluate the network
python train_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --lr=2e-1 \
    --num-epochs=140 \
    --no-cosine-annealing \
    --no-label-smoothing
cp checkpoint/$LABEL/ckpt_test139.pth checkpoint/$CHECKPOINT
python train_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --lr=2e-2 \
    --num-epochs=60 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_test199.pth checkpoint/$CHECKPOINT
python train_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --lr=2e-3 \
    --num-epochs=40 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
