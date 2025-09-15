#!/bin/bash
#SBATCH --job-name=sft-train                        # Job name 
#SBATCH --partition=boost_usr_prod                  # Partition (queue) name     
#SBATCH --gres=gpu:4                                # Number of GPUs (per node)                 
#SBATCH --nodes=1                                   # Total number of nodes to use           
#SBATCH --ntasks-per-node=1                         # Total number of tasks (processes) per node   
#SBATCH --ntasks=1                                  # Total number of tasks (processes) in the job  (must be ntask>=ntasks-per-node * nodes)            
#SBATCH --cpus-per-task=32                          # Number of CPU cores per task (max 32) 
#SBATCH --mem=256G                                  # Total memory per node (max, 256G)
#SBATCH --time=24:00:00                             # Time limit hrs:min:sec (max 24:00:00) 
#SBATCH --account=EUHPC_A06_067                     # Project account to charge (EUHPC_A06_067)  
#SBATCH --output=logs/output_sft.log                # Standard output and error log
#SBATCH --error=logs/error_sft.log

# Load necessary modules
module load cuda/12.1
module load anaconda3/2023.09-0

# Activate enviromnemnt
source activate sft

export HF_HOME="/leonardo_work/EUHPC_A06_067/hf_cache"
export PYTHONUNBUFFERED=TRUE        # Ensure real-time logging

# Multi-node launch
srun accelerate launch --config_file ./multi_gpu.yaml \
    sft_train.py \
    --model_name_or_path $HF_HOME/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 \
    --dataset_name "Vezora/Tested-143k-Python-Alpaca" \
    --cache_dir $HF_HOME \
    --train_split train \
    --output_dir $WORK/results \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 5.0e-06 \
    --optim adamw_torch_fused \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --adam_beta2 0.99 \
    --warmup_ratio 0.03 \
    --bf16 \
    --logging_steps 1 \
    --save_steps 10 \
    --eval_steps 100 \
    --completion_only_loss \
    --instruction_format pythonalpaca \
    --max_length 8192 \
    --use_liger \