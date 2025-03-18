# EuroHPC-Basics
EuroHPC's resources basic usage guide.


### Create a conda enviroment
Use `module avail | grep conda` to search for available conda packages. Use the following commands to load the package and create a conda enviroment.

```bash
module load anaconda3/2023.9-0
conda create -n test-env python=3.10
```
To activate the created conda enviroment use teh `source` command, and install all required packages via `pip`.

```bash
source activate test-env
pip install torch torchvision torchaudio
```

### Download your datasets
All HPC systems share the same logical disk structure and file systems definition. The available data areas are defined, on all HPC clusters,  through predefined environment variables. 

Datasets are usually saved at `$SCRATCH` area, which provides 20TB of space, and is local, temporary (files are deleted afte 40 days) and user specific. 

To download MNIST dataset to your user-specific `$SCRATCH` directory, run
```bash
python download_data.py
```

### Schedule jobs
CINECA HPC uses Slurm job manager, which provides three functions:
 - Allocating access to resources (compute nodes) to users for a specified duration, allowing them to perform their work.

- Providing a framework for starting, executing, and monitoring work (usually parallel jobs) on the set of allocated nodes.

- Managing resource contention by handling the queue of pending jobs.

To run your scipts on HPC's compute nodes, you have to specify the tasks you want to execute, and the system will manage running these tasks and returning the results to you. If the resources are not available, SLURM will hold your jobs and run them when resources become available.

A simple SLURM job script `run_jobs.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=pytorch_test     # Name of job
#SBATCH --partition=boost_usr_prod  # Use `sinfo` to find available partitions
#SBATCH --gres=gpu:1                # Allocate 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00             # Time limit
#SBATCH --output=output.log         # Output file
#SBATCH --error=error.log           # Error file
```

To run your scripts using CUDA and the created encviroment, you have to use the following commands:
```bash
module load cuda/12.3
module load anaconda3/2023.09-0
source activate test-env
python train.py
```
Without them, the compute nodes won't have access to your conda enviroment.

To schedule your job script, run
```bash 
sbatch run_jobs.sh
```
This will create a batch job with specific `JOB_ID`.
To see more details abut your job use
```bash
scontrol show job JOB_ID
```

