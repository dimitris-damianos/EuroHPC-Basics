####Downloading models and datasets 
Προκειμένου τα μοντέλα και datasets να χρησιμοποιηθούν στον HPC, ακολουθούμε τα εξής βήματα
##Datasets
1) Κατεβάζουμε τοπικά το dataset από το hugging face τρέχοντας το script download_sft_datasets.py
2) Συμπιέζουμε το dataset με την εντολή: tar -czvf bigbio___pubmed_qa  hf_cache/bigbio___pubmed_qa
3) Μεταφέρουμε το αρχείο στον hpc: scp bigbio___pubmed_qa username@login.leonardo.cineca.it:/leonardo_work/project_name/ 
4) Αποσυμπιέζουμε το αρχείο ως εξής: tar -xzf bigbio___pubmed_qa.tar.gz

##Models
Όμοια με τα datasets: 
1) tar -czvf models--Qwen--Qwen3-0.6B.tar.gz  hf_cache/models--Qwen--Qwen3-0.6B
2) scp models--Qwen--Qwen3-0.6B.tar.gz username@login.leonardo.cineca.it:/leonardo_work/project_name/
3) tar -xzf models--Qwen--Qwen3-0.6B.tar.gz

##sft formatting
Με το script sft_formatting.py η μορφή των δεδομένων μετατρέπεται σε format κατάλληλο για SFT (Supervised Fine-Tuning)- είτε prompt-completion pairs για instruction tuning είτε coversation για chat template.
