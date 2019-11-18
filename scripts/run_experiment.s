#!/bin/bash
#SBATCH --job-name=inferences_drugbank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:p40:1

source activate capstone
cd ../
python main.py --data ./data/drugbank_1000/ --pretrained_emb False\
 --output_folder ./checkpoints/drugbank1000_3\
 --epochs_gat 400\
 --epochs_conv 100\
 --batch_size_conv 256 --batch_size_gat 80000


