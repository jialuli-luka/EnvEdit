ob_type=pano
feedback=sample

features=vit-16-ori-768-e2e
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/R2R/exprs/finetune/agent/

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 2000
      --batch_size 6
      --optim adamW

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5"

# train
# vitbase.e2e bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt

# first stage pretrain     --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt
# ../datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks/ckpts/model_step_130000.pt
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --features_aug vit-16-st-samefilter-768-e2e

# inference
# vitbase.e2e resume_file: ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen
#CUDA_VISIBLE_DEVICES='1' python r2r/main.py $flag \
#      --resume_file ../datasets/R2R/exprs/pretrain/agent1/ckpts/model_step_200000.pt  \
#      --test --submit
