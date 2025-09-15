#SFT Tutorial
This is a SFT guide, where we present all the neccesary steps for downloading data, model locally, data preprocessing and SFT training. 

# Enviroment creation
We provide a tutorial enviroment. To create your enviroment locally, use the following instructions. 
```bash
module load anaconda3/2023.09-0        # Load the conda package

conda create -n sft python=3.10
source activate sft                     # Use source to activate conda envs
pip install -r requirements.txt
```
This step is required for all next steps, make sure the enviroment is created and can be succesfully used.

## Preprocessing
Follow the instructions in [preprocessing/](./preprocessing/)

## SFT Training
Follow the instructions in [training/](./training/)
