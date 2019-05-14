LABEL=grow-run9-dr2-wide
MODEL=tiny16
DROPOUT=0
DROPEDGE=0
DEPTHRATE=0.2
CHECKPOINT=ckpt_$LABEL.pth

rm checkpoint/$CHECKPOINT
rm -rf log/$LABEL
rm -rf checkpoint/$LABEL
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=1 \
    --lr=1e-1 \
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
    --lr=1e-1 \
    --num-epochs=30 \
    --no-cosine-annealing \
    --no-label-smoothing
cp checkpoint/$LABEL/ckpt_done39.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=4 \
    --lr=1e-1 \
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
    --lr=1e-1 \
    --num-epochs=40 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
cp checkpoint/$LABEL/ckpt_done119.pth checkpoint/$CHECKPOINT
python train_grownet_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --depthrate=$DEPTHRATE \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --expand-period=8000 \
    --lr=1e-2 \
    --num-epochs=80 \
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
    --lr=1e-3 \
    --num-epochs=40 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT
