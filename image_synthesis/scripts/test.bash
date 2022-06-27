#!/usr/bin/env bash
flag="--name spade
      --dataset_mode custom
      --label_dir ../views_sem_image_transferred
      --image_dir ../../views_img
      --label_nc 42
      --use_vae
      --no_instance
      --gpu_ids 0
      --preprocess_mode none
      --batchSize 1
      --style_method original
      --mask_num 0
      --output_dir views_img_spade_original"

python test.py $flag
