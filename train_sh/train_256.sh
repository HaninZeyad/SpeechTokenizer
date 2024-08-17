#!/bin/bash
#SBATCH -vv
#SBATCH -t 01:00:00
## SBATCH --mem-per-gpu 20G
##SBATCH -c 5
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J 256_10_linear2
#SBATCH --output Output_error_log/%x_%j.out
#SBATCH --error Output_error_log/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hatwany
#SBATCH --signal INT@60

#source /opt/packages/anaconda3-2022.10/etc/profile.d/conda.sh
source /ocean/projects/cis220031p/hatwany/env/bin/activate
conda activate nglam

# cd /ocean/projects/cis220031p/hatwany/SpeechTokenizer
#CONFIG="config/spt_base_256_cfg.json"
CONFIG="/ocean/projects/cis220031p/hatwany/SpeechTokenizer/config/spt_base_256_cfg_linear2.json"

# NPROC_PER_NODE=4
# CUDA_VISIBLE_DEVICES=1,2,6,7 torchrun \
#     --nnode 1 \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master_port 50025  \
# train_example.py \
#     --config ${CONFIG} \

# CUDA_VISIBLE_DEVICES=1,2,6,7 accelerate launch scripts/train_example.py\
#     --config ${CONFIG}\
   # --continue_train

accelerate launch scripts/train_example.py\
    --config ${CONFIG}\
