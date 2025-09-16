from datasets import load_dataset

datasets = [
    "tatsu-lab/alpaca"
]


for d in datasets:
    print(f"Checking: {d}")
    try:
        ds = load_dataset(
            d,
            cache_dir="./hf_cache"
        )
        '''
        ds = load_dataset(
            "tatsu-lab/alpaca",
            "default",
        cache_dir="."
        )
        '''
        print(f">>> {d} already in cache or just downloaded.")
    except FileNotFoundError:
        print(f">>> {d} not found in cache. Downloading...")
        load_dataset(
            d,
            cache_dir="./hf_cache"
        )

