LABEL=tiny16-run5-da5
MODEL=tiny16
DROPOUT=0.5
DROPEDGE=0.1
CHECKPOINT=ckpt_$LABEL.pth

python train_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --lr=1e-1 \
    --num-epochs=150 \
    --no-cosine-annealing \
    --no-label-smoothing
cp checkpoint/$LABEL/ckpt_test149.pth checkpoint/$CHECKPOINT
python train_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --lr=1e-2 \
    --num-epochs=100 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_test249.pth checkpoint/$CHECKPOINT
python train_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --lr=1e-3 \
    --num-epochs=100 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
