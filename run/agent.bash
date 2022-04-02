name=agent
flag="--attn soft --train augenv
      --featdropout 0.3
      --feature_size 512
      --angleFeatSize 128
      --log_every 500
      --feature_extract img_features/CLIP-ViT-B-16-views.tsv
      --aug_env img_features/CLIP-ViT-B-16-views-st-samefilter.tsv
      --division equal
      --train_env both
      --valid_env original
      --aug_method alternative
      --feedback sample
      --mlWeight 0.2
      --batchSize 64
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 35"
mkdir -p snap/$name

# Try this with file logging:
CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
