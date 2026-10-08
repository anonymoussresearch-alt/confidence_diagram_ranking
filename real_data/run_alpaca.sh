#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH --job-name=alpaca
#SBATCH --array=0-4

# Load required modules
module load miniconda3/4.10.3
conda init bash
module load gcc/9.2.0 cuda/11.2
module load python/3.8.12 

python run_alpaca.py