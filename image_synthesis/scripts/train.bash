#!/usr/bin/env bash
flag="--name spade
      --dataset_mode custom
      --label_dir ../views_sem_image_transferred
      --image_dir ../../views_img
      --label_nc 42
      --mask_num 0
      --no_instance
      --use_vae
      --gpu_ids 0
      --preprocess_mode scale_shortside_and_crop
      --batchSize 16"

python train.py $flag
