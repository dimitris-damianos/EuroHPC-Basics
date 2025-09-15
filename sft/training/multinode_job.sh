#!/bin/bash
#SBATCH --job-name=multinode
#SBATCH --partition=boost_usr_prod                              
#SBATCH --cpus-per-task=8                   
#SBATCH --nodes=4      # nodes <= ntasks
#SBATCH --ntasks=4                             
#SBATCH --ntasks-per-node=1  
#SBATCH --gpus-per-node=4                  
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=./logs/multinode_output.log
#SBATCH --error=./logs/multinode_error.log

module load anaconda3/2023.09-0
source activate sft

export PYTHONUNBUFFERED=TRUE  
export HF_HOME="/leonardo_work/EUHPC_D19_095/hf_cache"

export LOGLEVEL=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
export NCCL_DEBUG=INFO
echo "environment: $(env | grep NCCL)"

num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
srun --label accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --machine_rank $SLURM_NODEID \
    --num_processes $num_processes \
    --num_machines $SLURM_NNODES \
    --dynamo_backend no \
    --mixed_precision 'bf16' \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    train.py \
    --whisper-path "openai/whisper-large-v3" \
    --llm-path "ilsp/Llama-Krikri-8B-Base"\
    --injection-layer 30\
    --dataset-path "/leonardo_work/EUHPC_D19_097/mllm/datasets/gpc50_pretrain" \
    --batch-size 8\
    --steps 1000\
    --epochs 100\
    --attn-implementation "flash_attention_2" \
    --frozen-whisper \
    --add-liger \
    --add-lora \
    --disable-tqdm \
    --resume
