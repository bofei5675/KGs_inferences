#!/bin/bash
#SBATCH --job-name=inferences_drugbank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:30:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python main.py --data ./data/drugbank_1000/ --pretrained_emb False\
 --output_folder ./checkpoints/drugbank1000/ 

