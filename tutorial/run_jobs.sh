#!/bin/bash
#SBATCH --job-name=pytorch_test
#SBATCH --partition=boost_usr_prod          ### Use `sinfo` to find available partitions
#SBATCH --gres=gpu:1    # Allocate 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=output.log
#SBATCH --error=error.log

module load cuda/12.3
module load anaconda3/2023.09-0

source activate test-env

python train.py
