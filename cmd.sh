#!/usr/bin/env bash

set -x

# EXP_DIR=exps/r50_deformable_detr
# PY_ARGS=${@:1}

# original
# python -u main.py \
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS}

# finetune on ford
# python -u main.py \
#     --output_dir="exps/ford_try_0715" \
#     --coco_path /home/ymjian/Desktop/Load_Data_and_COCO_format/0624_test \
#     --dataset_file ford \
    # --resume "model_pt_files/r50_deformable_detr-checkpoint_no-class-head.pth" \
    # --epochs 50 \
    # ${PY_ARGS}

# eval 
python main.py -m dn_dab_deformable_detr \
    --output_dir logs/dab_deformable_detr/R50_from_scratch \
    --batch_size 1 \
    --coco_path /mnt/workspace/users/ymjian/Shuttle_Deepen_Data_ImgBased_front_cam \
    --resume /mnt/workspace/users/ymjian/test/exps/fine_tune_test/image_based/dn_d_detr_from_scratch/checkpoint.pth \
    --dataset_file ford2 \
    --transformer_activation relu \
    --use_dn \
    --eval