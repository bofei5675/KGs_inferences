#!/bin/bash
#SBATCH --job-name=inferences_drugbank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:p40:1

source activate capstone
cd ../
python main.py --data ./data/drugbank_2191_new/ --pretrained_emb False\
 --output_folder ./checkpoints/drugbank2191__new/ \
 --epochs_gat 800\
 --epochs_conv 200\
 --batch_size_conv 256 --batch_size_gat 80000


