# Preprocessing
In order for the models and datasets to be used on the HPC, we follow these steps

## Datasets

1)  We download locally the dataset from hugging face by running the script download\_sft\_datasets.py
2)  We compress the dataset with the command:
   tar -czvf bigbio\_\_\_pubmed\_qa hf\_cache/bigbio\_\_\_pubmed\_qa
3)  We transfer the file to the hpc:
   scp bigbio\_\_\_pubmed\_qa [username@login.leonardo.cineca.it](mailto:username@login.leonardo.cineca.it):/leonardo\_work/project\_name/
4)  We decompress the file as follows:
   tar -xzf bigbio\_\_\_pubmed\_qa.tar.gz

## Models
Similarly to the datasets:
1) tar -czvf models--Qwen--Qwen3-0.6B.tar.gz hf\_cache/models--Qwen--Qwen3-0.6B
2) scp models--Qwen--Qwen3-0.6B.tar.gz [username@login.leonardo.cineca.it](mailto:username@login.leonardo.cineca.it):/leonardo\_work/project\_name/
3) tar -xzf models--Qwen--Qwen3-0.6B.tar.gz

## SFT formatting
With the script sft\_formatting.py the data format is converted into a format suitable for SFT (Supervised Fine-Tuning) – either prompt-completion pairs for instruction tuning or conversation for chat template.

## Using HF CACHE
Downloading models and datasets using the usual `transformers` and `datasets` methods, saves copies of said data on the cache. The compute nodes have access to these models, so it is a simple and viable alternative. 

The only possible isue that may occur is that the user mey be required to define explicity the path of cahced dataset/model inside the `hf_cache` directory. 

## Save space limits
The `$HOME` directory has a limit of 50GB and is not designed for heavy disk use (models/datasets)

The `$SCRATCH` directory provide a lot of shared space, but its contents are deleted after 40 days.

The `$WORK` directory provides 4T per project.