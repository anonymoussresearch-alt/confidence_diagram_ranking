#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 120G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00                         # Runtime in D-HH:MM format
#SBATCH --job-name=llama1

# Load required modules
module load miniconda3/4.10.3
conda init bash
module load gcc/9.2.0 cuda/11.2
module load python/3.8.12 

python run_llama1.py