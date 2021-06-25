# The name of experiment
name=fine_tune_ssrp

# Create dirs and make backup
output=snap/stage2/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/ssrp_finetune.py \
    --train mscoco_train_aug,mscoco_nominival_aug --valid mscoco_minival_aug \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --load_lxmert=/m/liyz/lxmert/snap/pretrain/stage1_imgaug/BEST_EVAL_LOSS_LXRT.pth \
    --optim bert --lr 1e-4 --epochs 20 \
    --tqdm --batchSize 128 \
    --output $output ${@:2}


