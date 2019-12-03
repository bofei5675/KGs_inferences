#!/bin/bash
#SBATCH --job-name=inferences_drugbank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source activate capstone
cd ../
python evaluate.py --data ./data/deepddi/ \
--output_folder /scratch/bz1030/relationPrediction/checkpoints/deepddi/ \
--tanh no


