#!/bin/bash
#SBATCH --job-name=inferences_drugbank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:p40:1

source activate capstone
cd ../
python main.py --data ./data/drugbank_100/ --pretrained_emb no\
 --output_folder ./checkpoints/drugbank100_new/ \
 --epochs_gat 30\
 --epochs_conv 10\
 --batch_size_conv 512 --batch_size_gat 80000\
 --tanh yes


