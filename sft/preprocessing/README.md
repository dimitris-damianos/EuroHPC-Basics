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

