LABEL=default
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
