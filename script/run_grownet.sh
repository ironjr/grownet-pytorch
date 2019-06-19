LABEL=valmon-dr5-do0-rand-v16-lr2
MODEL=tiny16
DROPOUT=0
DROPEDGE=0
DEPTHRATE=0.5
POLICY=RandomPolicy
CHECKPOINT=ckpt_$LABEL.pth

rm checkpoint/$CHECKPOINT
rm -rf log/$LABEL
rm -rf checkpoint/$LABEL
rm -rf graph/$LABEL
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=1 \
    --expand-policy=$POLICY \
    --lr=2e-1 \
    --num-epochs=10 \
    --no-cosine-annealing \
    --no-label-smoothing
cp checkpoint/$LABEL/ckpt_done9.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=2 \
    --expand-policy=$POLICY \
    --lr=2e-1 \
    --num-epochs=30 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_done39.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=4 \
    --expand-policy=$POLICY \
    --lr=2e-1 \
    --num-epochs=40 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_done79.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=400 \
    --expand-policy=$POLICY \
    --lr=2e-1 \
    --num-epochs=60 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_done139.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=1000 \
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
    --expand-period=1000 \
    --expand-policy=$POLICY \
    --lr=2e-3 \
    --num-epochs=40 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_done239.pth checkpoint/$CHECKPOINT
