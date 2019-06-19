LABEL=tiny16-run8-do5-conn
MODEL=tiny16
DROPOUT=0.5
DROPEDGE=0
CHECKPOINT=ckpt_$LABEL.pth

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
