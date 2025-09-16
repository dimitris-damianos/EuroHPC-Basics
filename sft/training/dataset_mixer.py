# dataset_mixer.py

import os
import math
from typing import List, Dict
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict

from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)
logger.setLevel("INFO")

import multiprocessing

from sft_formatting import (  # format functions to convert datasets to prompt-completion and conversation format
    map_to_conversation_with_system_message,
    map_mathinstruct_to_prompt_completion,
    map_mathinstruct_to_conversation,
    map_metamathqa_to_conversation,
    map_pythonalpaca_to_prompt_completion,
    map_pythonalpaca_to_conversation,
    map_aya_to_prompt_completion,
    map_aya_to_conversation,
    map_aya_with_language_to_prompt_completion,
    map_aya_with_language_to_conversation,
    map_alpaca_to_prompt_completion,
    map_alpaca_to_conversation,
    map_copa_to_prompt_completion,
    map_copa_to_conversation,
    map_socialiqa_to_prompt_completion,
    map_socialiqa_to_conversation,
    map_pubmedqa_to_prompt_completion,
    map_pubmedqa_to_conversation,
    map_piqa_to_prompt_completion,
    map_piqa_to_conversation,
    map_ecare_causal_reasoning_to_prompt_completion,
    map_ecare_causal_reasoning_to_conversation,
    map_ecare_explanation_generation_to_prompt_completion,
    map_ecare_explanation_generation_to_conversation,
    map_moleculeqa_to_prompt_completion,
    map_moleculeqa_to_conversation,
    map_casehold_to_prompt_completion,
    map_casehold_to_conversation,
    map_finqa_to_prompt_completion,
    map_finqa_to_conversation,
    map_story_generation_to_prompt_completion,
    map_story_generation_to_conversation,
    map_news_summarization_to_prompt_completion,
    map_news_summarization_to_conversation,
    map_moral_stories_moral_action_to_prompt_completion,
    map_moral_stories_moral_action_to_conversation,
    map_moral_stories_immoral_action_to_prompt_completion,
    map_moral_stories_immoral_action_to_conversation,
    map_wikiqa_to_prompt_completion,
    map_wikiqa_to_conversation,
)

MAP_FUNCTIONS = {
    "insert_system_message": map_to_conversation_with_system_message,
    "mathinstruct": map_mathinstruct_to_prompt_completion,
    "mathinstruct_chat": map_mathinstruct_to_conversation,
    "metamathqa_chat": map_metamathqa_to_conversation,
    "pythonalpaca": map_pythonalpaca_to_prompt_completion,
    "pythonalpaca_chat": map_pythonalpaca_to_conversation,
    "piqa": map_piqa_to_prompt_completion,
    "piqa_chat": map_piqa_to_conversation,
    "pubmedqa": map_pubmedqa_to_prompt_completion,
    "pubmedqa_chat": map_pubmedqa_to_conversation,
    "aya": map_aya_to_prompt_completion,
    "aya_chat": map_aya_to_conversation,
    "aya_with_language": map_aya_with_language_to_prompt_completion,
    "aya_with_language_chat": map_aya_with_language_to_conversation,
    "alpaca": map_alpaca_to_prompt_completion,
    "alpaca_chat": map_alpaca_to_conversation,
    "copa": map_copa_to_prompt_completion,
    "copa_chat": map_copa_to_conversation,
    "socialiqa": map_socialiqa_to_prompt_completion,
    "socialiqa_chat": map_socialiqa_to_conversation,
    "e-care_causal_reasoning": map_ecare_causal_reasoning_to_prompt_completion,
    "e-care_causal_reasoning_chat": map_ecare_causal_reasoning_to_conversation,
    "e-care_explanation_generation": map_ecare_explanation_generation_to_prompt_completion,
    "e-care_explanation_generation_chat": map_ecare_explanation_generation_to_conversation,
    "moleculeqa": map_moleculeqa_to_prompt_completion,
    "moleculeqa_chat": map_moleculeqa_to_conversation,
    "casehold": map_casehold_to_prompt_completion,
    "casehold_chat": map_casehold_to_conversation,
    "finqa": map_finqa_to_prompt_completion,
    "finqa_chat": map_finqa_to_conversation,
    "story": map_story_generation_to_prompt_completion,
    "story_chat": map_story_generation_to_conversation,
    "news": map_news_summarization_to_prompt_completion,
    "news_chat": map_news_summarization_to_conversation,
    "moral_actions": map_moral_stories_moral_action_to_prompt_completion,
    "moral_actions_chat": map_moral_stories_moral_action_to_conversation,
    "immoral_actions": map_moral_stories_immoral_action_to_prompt_completion,
    "immoral_actions_chat": map_moral_stories_immoral_action_to_conversation,
    "wikiqa": map_wikiqa_to_prompt_completion,
    "wikiqa_chat": map_wikiqa_to_conversation,
}

def mix_datasets_with_mapping(
    dataset_configs: List[Dict],
    splits: List[str],
    shuffle_init: bool = True,
    shuffle_post: bool = True,
    cache_dir: str = "./hf_cache",
    # accelerator: Accelerator = None,
) -> DatasetDict:
    mixed = {}
 
    for split in splits:
        split_subsets = []

        for entry in dataset_configs:
            dataset_id = entry["name"]
            num_samples = entry.get("num_samples", None)
            frac = entry.get("fraction", None)
            map_fn_key = entry["map_fn"]
            config_name = entry.get("config_name", None)
            data_files = entry.get("data_files", None)

            entry_splits = entry.get("splits", ["train"])
            if split not in entry_splits:
                continue

            assert map_fn_key in MAP_FUNCTIONS, f"Unknown map function: {map_fn_key}"
            map_fn = MAP_FUNCTIONS[map_fn_key]

            try:
                if os.path.isdir(dataset_id) and not data_files:
                    dataset = load_from_disk(os.path.join(dataset_id, split))
                elif data_files:
                    dataset = load_dataset(
                        path=dataset_id,
                        split=split,
                        data_files=data_files,
                        cache_dir=cache_dir
                    )
                else:
                    dataset = load_dataset(
                        path=dataset_id,
                        name=config_name,
                        split=split,
                        cache_dir=cache_dir
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset {dataset_id} [{split}]: {e}")

            if len(dataset) == 0:
                logger.warning(f"[{dataset_id}][{split}]: skipped (no samples)")
                continue

            if shuffle_init:
                dataset = dataset.shuffle(seed=42)

            if num_samples is not None and frac is not None:
                raise ValueError(f"Cannot set both 'num_samples' and 'fraction' for dataset {dataset_id}")
            if num_samples is not None:
                if num_samples <= 0:
                    raise ValueError(f"Invalid num_samples={num_samples} for {dataset_id}. Must be > 0.")
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                logger.info(f"[{dataset_id}][{split}]: loaded {len(dataset)} samples, using {num_samples} samples")
            elif frac is not None:
                if frac < 0:
                    raise ValueError(f"Invalid fraction {frac} for dataset {dataset_id}. Must be >= 0.")
                if frac < 1.0:
                    dataset = dataset.select(range(int(frac * len(dataset))))
                elif frac > 1.0:
                    concat_times = math.floor(frac)
                    extra = frac - concat_times
                    dataset = concatenate_datasets([dataset] * concat_times + [
                        dataset.select(range(int(extra * len(dataset))))
                    ])
                logger.info(f"[{dataset_id}][{split}]: loaded {len(dataset)} samples, using {frac*100:.1f}%")
            else:
                logger.info(f"[{dataset_id}][{split}]: loaded {len(dataset)} samples (full dataset)")

            # FIXME: or pass the accelerator from sft.py
            # if accelerator is not None:
            #     accelerator.print(f"[{dataset_id}][{split}]: loaded {len(dataset)} samples, using {frac*100:.1f}%")
            # else:
            #     logger.info(f"[{dataset_id}][{split}]: loaded {len(dataset)} samples, using {frac*100:.1f}%")
            dataset = dataset.map(map_fn, remove_columns=list(dataset.features), num_proc=multiprocessing.cpu_count())
            dataset = dataset.filter(lambda x: x["messages"] is not None)  # NOTE: for conversational format - added to quickly filter out some bad examples
            logger.info(f"{len(dataset)} samples after filtering")
            if len(split_subsets) > 0:
                first_keys = set(split_subsets[0].features)
                current_keys = set(dataset.features)
                assert current_keys == first_keys, (
                    f"Inconsistent fields across datasets: expected {first_keys}, got {current_keys}"
                )
            split_subsets.append(dataset)

        if not split_subsets:
            continue
        combined = concatenate_datasets(split_subsets)
        if shuffle_post:
            combined = combined.shuffle(seed=42)
        mixed[split] = combined

    return DatasetDict(mixed)

