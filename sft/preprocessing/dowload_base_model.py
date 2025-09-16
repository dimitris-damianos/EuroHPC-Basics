# download_base_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    cache_dir="./hf_cache",
    local_files_only=False,
    force_download=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-1.5B",
    cache_dir="./hf_cache",
    local_files_only=False,
    force_download=True
)

