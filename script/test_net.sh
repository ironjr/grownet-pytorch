LABEL=tiny16-do0-conn-str1
MODEL=tiny16
DROPOUT=0
DROPEDGE=0
DEPTHRATE=0.5
POLICY=MaxEdgeStrengthPolicy
CHECKPOINT=ckpt_$LABEL.pth

# Test RandWire
python train_cifar.py \
    --label=$LABEL \
    --model=$MODEL \
    --dropout=$DROPOUT \
    --drop-edge=$DROPEDGE \
    --lr=2e-1 \
    --num-epochs=1 \
    --no-cosine-annealing \
    --no-label-smoothing \
    --resume \
    --checkpoint=checkpoint/$CHECKPOINT \
    --test-only

# Test GrowNet
# python train_grownet_cifar.py \
#     --label=$LABEL \
#     --model=$MODEL \
#     --depthrate=$DEPTHRATE \
#     --dropout=$DROPOUT \
#     --drop-edge=$DROPEDGE \
#     --expand-period=10000 \
#     --expand-policy=$POLICY \
#     --lr=2e-1 \
#     --num-epochs=1 \
#     --no-cosine-annealing \
#     --no-label-smoothing \
#     --resume \
#     --checkpoint=checkpoint/$CHECKPOINT \
#     --test-only
