#!/bin/bash
#SBATCH --job-name=inferences_drugbank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:p40:1

source activate capstone
cd ../
python main.py --data ./data/deepddi/ --pretrained_emb no\
 --output_folder ./checkpoints/deepddi_tanh/ \
 --epochs_gat 300\
 --epochs_conv 300\
 --load_gat /scratch/bz1030/relationPrediction/checkpoints/deepddi_tanh/gat/trained_899.pth\
 --batch_size_conv 512 --batch_size_gat 80000\
 --tanh yes


