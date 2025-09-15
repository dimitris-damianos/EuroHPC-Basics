## SFT Training
Before following this guide, make sure that you have locally saved model instances and datasets. Additionally, make sure that the dataset is pre-processed and already tokenized, to avoid any NCCL issues.

### Python script
The training script we provide is [sft_train.py](./sft_train.py). As you can see, it contains the standard transformers-based code, using the `SFTTrainer`. On python level, HPC system do not require any specific changes. 

The only, optional, consideration from the python side wuld be to use the `stdout` commands (like `print()`) with `flush=True`. This changes, along with setting the enviroment parameter `PYTHONBUFFERED=True` allows for real-time logging on the `stdout` file.

### SLURM file 
The SLURM files are required for commiting jobs to the SLURM scheduler and they are the only way to use the compute nodes. for this tutorila we would be using [train_job.sh](./train_job.sh)

These files contain at the top a configuration for the job that is about to run:
```bash
#!/bin/bash
#SBATCH --job-name=sft-train                        # Job name 
#SBATCH --partition=boost_usr_prod                  # Partition (queue) name     
#SBATCH --gres=gpu:4                                # Number of GPUs (per node)                 
#SBATCH --nodes=1                                   # Total number of nodes to use           
#SBATCH --ntasks-per-node=1                         # Total number of tasks (processes) per node   
#SBATCH --ntasks=1                                  # Total number of tasks in the job          
#SBATCH --cpus-per-task=32                          # Number of CPU cores per task (max 32) 
#SBATCH --mem=256G                                  # Total memory per node
#SBATCH --time=24:00:00                             # Time limit hrs:min:sec
#SBATCH --account=EUHPC_A06_067                     # Project account to charge 
#SBATCH --output=logs/output_sft.log                # Standard output and error log
#SBATCH --error=logs/error_sft.log
```

- `--job-name` :  Job name
- `--partition` : partition to use (use `sinfo` for available partitions)
- `--gres=gpu:4` : number of GPUs per node to use
- `--nodes` : number of compute nodes to use
- `--ntasks-per-node` : total tasks to spawn per node
- `--ntasks` : total tasks to spawn (must be `ntasks`>=`ntasks-per-node`)
- `--cpu-per-task` : CPU cores per spawned task (max 32)
- `--mem` : memory to use (max 256GB)
- `--time` : max job time (max 24 hours)  
- `--account` : Project account to charge (the project name)
- `--output` : stdout log file (for real time logging use `PYTHONBUFFERED=True` and `flush=True` )
- `--error` : stderror log file 

### Running a script
First you have to load the neccesary module (`anaconda`, `cuda` etc) and activate the `conda` enviroments inside the compute node. This is achieved using the following code:
```bash
module load cuda/12.1
module load anaconda3/2023.09-0
source activate sft
```
Then, using `srun` you can run any python script:
```bash
srun python script.py
```

### Commiting a job
To commit a job, you must first create a SLURM file (which contains a configuration as the one mentioned above) and the script/commands you want to run. 
With these commands on our SLURM script, we can commit a SLURM job using `sbatch script.sh` in our command line. For more details please refer to [train_job.sh](./train_job.sh). 

### Multi-GPU training (single node)
Multi-GPU training using `accelerate` is quite simple:
- Make sure that `--ntasks=1` and `--ntasks-per-node=1`, since the additional tasks will be spawned by `accelerate`.
- Make sure the `config.yaml` used in `accelerate` has the same number of `num_processes` as the `--gres=gpu:` parameter. If there is a mismatch,  multimple jobs may spawn on the same GPU, causing OOM.

To run the multi-gpu job simply run:
```bash
srun accelerate launch \
    --config_file ./config.yaml \
    script.py \
    --arg1 ... \
    ...
```
We provide a simple config file, [multi_gpu.yaml](./multi_gpu.yaml).

### Multi-node training
This is a bit more complicated. Since the node allocation happens dynamically, due to SLURM scheduler, we extract the neccesary information inside the script: 
```bash
export LOGLEVEL=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
export NCCL_DEBUG=INFO

num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
```
The number of nodes to use is defined from `--ntasks` and `--ntasks-per-node`. For example, `--ntasks=8` will use 8 node.
The `accelerate` config is created dynamically and the code can be run as:
```bash
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
    script.py \
    --args1 ...\
    --args2 \
```

Check [multinode_job.sh](./multinode_job.sh) for more details.

## Monitoring the job
### Using the `log` files
- The `stderror` and `stdout` log files can provide real-time monitoring of the job's output. 

- Using `squeue -u $USER` on cmd, the user can see all the scheduled jobs, their PID, the occupied nodes, the current runtime.

Direct access to compute nodes is prohibited, which makes the developer experience frustrating when out-of-memory (OOM) errors occur. To compensate for that, we provide a simple MemoryCallback function in [utils.py](./utils.py) that allows for direct memory monitoring during training. 

## Killing a job
To kill a job, you require either the job's PID or its name: 
- `scancel PID` 
- `scancel -n JOB_NAME`
