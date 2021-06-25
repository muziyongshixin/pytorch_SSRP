# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa_ssrp/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa_ssrp.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --load_lxmert snap/pretrain/stage1_imgaug/BEST_EVAL_LOSS_LXRT.pth \
    --load_probe_head snap/stage2/fine_tune_ssrp/Epoch15_probe_head.pth \
    --batchSize 48 --optim bert --lr 5e-5 --epochs 20 \
    --tqdm --output $output ${@:3}
