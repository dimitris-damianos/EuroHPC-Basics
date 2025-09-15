# EUROHPC Guide

## Basic Tutorial
In the [tutorial/](./tutorial/) folder you can find a basic usage guide. We recommend to start with this tutorial, if you have no prior experience with SLURM-based HPC systems.

## SFT Tutorial
In the [sft/](./sft/) folder you can find a SFT-specific example, including data/model downlodaing, data preprocessing, multi-gpu training.

## Basic HPC Infrasructure
The nodes (machines) in a HPC system are categorized in: 
- **login nodes**: nodes where the user connects and can performs tasks
- **compute nodes**: nodes where heavy tasks are perfomed, unaccessible by users.

The login nodes allow for limited CPU/memory usage, and are the only ones connected to the internet. On the other hand, compute nodes are designed for heavy CPU/GPU tasks but have no internet connections.

Due to these limitations, users have to:
- **download base model locally**: models used in traing/inference must be saved locally before accessing compute nodes
- **download data locally**: the datasets must be saved locally before accessing compute nodes
- **download in batches**: in the case of large datasets (hundreds of GBs) it is recomended to download the data in smaller batches, since the download process will be killed after some time.

## Useful SLURM commands:
- *sbatch job.sh*: submits a job script to the SLURM scheduler, which then dispatches it to available compute nodes.
- *saldo -b*: allows user to monitor monthly and total GPU usage of their projects
- *user -u $USER*: allows user to monitor active jobs
