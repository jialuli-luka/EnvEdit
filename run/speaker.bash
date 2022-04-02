name=speaker
flag="--attn soft --angleFeatSize 128
      --feature_size 512
      --feature_extract img_features/CLIP-ViT-B-16-views.tsv
      --aug_env img_features/CLIP-ViT-B-16-views-st-samefilter.tsv
      --train speaker
      --style_embedding style_original.tsv
      --train_env both
      --valid_env original
      --subout max --dropout 0.6 --optim adam --lr 1e-4 --iters 80000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this for file logging
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
