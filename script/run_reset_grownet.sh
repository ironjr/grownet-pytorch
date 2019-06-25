LABEL=valmon-mes-dr5-do0-v16-lr2-str1
MODEL=tiny16
DROPOUT=0
DROPEDGE=0
DEPTHRATE=0.5
POLICY=RandomPolicy
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
    --expand-period=10000 \
    --expand-policy=$POLICY \
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
    --expand-period=10000 \
    --expand-policy=$POLICY \
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
    --expand-period=10000 \
    --expand-policy=$POLICY \
    --lr=2e-3 \
    --num-epochs=40 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
