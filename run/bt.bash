name=agent
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train augenvaugpath --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker40/state_dict/best_val_unseen_bleu
      --load snap/agent/state_dict/best_val_unseen
      --feature_extract img_features/CLIP-ViT-B-16-views.tsv
      --aug_env img_features/CLIP-ViT-B-16-views-st-samefilter.tsv
      --style_embedding style_original.tsv
      --style_embedding_aug spade2_style_st_samefilter.tsv
      --feature_size 512
      --log_every 500
      --aug_method specify
      --train_env both
      --valid_env original
      --angleFeatSize 128
      --accumulateGrad
      --featdropout 0.4
      --subout max --optim rms --lr 1e-4 --iters 300000 
      --maxAction 35"
mkdir -p snap/$name

# Try this with file logging:
CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
