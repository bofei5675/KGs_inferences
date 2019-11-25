#!/bin/bash
#SBATCH --job-name=inferences_drugbank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:p40:1

source activate capstone
cd ../
python main.py --data ./data/drugbank_1861/ --pretrained_emb False\
 --output_folder ./checkpoints/drugbank1861/ \
 --epochs_gat 1200\
 --epochs_conv 30\
 --batch_size_conv 512 --batch_size_gat 80000\
 --load_gat /scratch/bz1030/relationPrediction/checkpoints/drugbank1861/gat/trained_1199.pth\
 --load_conv /scratch/bz1030/relationPrediction/checkpoints/drugbank1861/conv/trained_29.pth\
 --tanh no