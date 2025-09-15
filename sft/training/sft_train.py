# sft.py

import argparse
import os
import datetime
import yaml

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    SFTTrainer,
    SFTConfig,
    DataCollatorForCompletionOnlyLM,
    clone_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from peft import LoraConfig, AutoPeftModelForCausalLM
from accelerate import Accelerator
from accelerate.state import PartialState

import wandb

from dataset_mixer import mix_datasets_with_mapping

import multiprocessing
from utils import MemoryUsageCallback

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

def make_parser():
    parser = argparse.ArgumentParser(description="Qwen experts with SFT.")
    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./hf_cache")

    # Dataset
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--data_files", type=str, default=None)
    parser.add_argument("--dataset_mix_config", type=str, default=None)
    parser.add_argument(
        "--instruction_format",
        type=str,
        choices = [
            "insert_system_message",
            "mathinstruct",
            "mathinstruct_chat",
            "metamathqa_chat",
            "pythonalpaca",
            "pythonalpaca_chat",
            "piqa",
            "piqa_chat",
            "pubmedqa",
            "pubmedqa_chat",
            "aya",
            "aya_chat",
            "aya_with_language",
            "aya_with_language_chat",
            "alpaca",
            "alpaca_chat",
            "copa",
            "copa_chat",
            "socialiqa",
            "socialiqa_chat",
            "e-care_causal_reasoning",
            "e-care_causal_reasoning_chat",
            "e-care_explanation_generation",
            "e-care_explanation_generation_chat",
            "moleculeqa",
            "moleculeqa_chat",
            "casehold",
            "casehold_chat",
            "finqa",
            "finqa_chat",
            "story",
            "story_chat",
            "news",
            "news_chat",
            "moral_actions",
            "moral_actions_chat",
            "immoral_actions",
            "immoral_actions_chat",
            "wikiqa",
            "wikiqa_chat",
        ],
        required=False,
    )

    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_liger", action="store_true")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--completion_only_loss", action="store_true")
    parser.add_argument("--assistant_only_loss", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--neftune_noise_alpha", type=float, default=None)
    parser.add_argument("--activation_offloading", action="store_true")

    # Optimizer and scheduler
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # LoRA
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_rslora", action="store_true")

    # Other settings
    parser.add_argument("--push_to_hub", action="store_true")
    
    return parser

def make_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if args.packing else "right"
    tokenizer.truncation_side = "right"
    
    return tokenizer

def make_model(args, quantization_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  
    )
    model.config.attn_implementation = "flash_attention_2"
    model.config.use_cache = False if args.gradient_checkpointing else True
    
    return model

def make_dataset(args, accelerator):
    config = None
    if args.dataset_mix_config:  # datamix
        with open(args.dataset_mix_config, "r") as f:
            config = yaml.safe_load(f)
        splits = config.get("splits", [args.train_split, args.eval_split])
        dataset = mix_datasets_with_mapping(
            dataset_configs=config["datasets"],
            splits=splits,
            shuffle_init=config.get("shuffle_init", True),
            shuffle_post=config.get("shuffle_post", True),
            cache_dir=args.cache_dir,
            # accelerator=accelerator,
        )
        train_dataset = dataset[args.train_split]
        if args.eval_split and args.eval_split in dataset:
            eval_dataset = dataset[args.eval_split]
        else:
            eval_dataset = None

    else:  # single dataset
        if args.data_files:
            import json
            data_files = json.loads(args.data_files)
            dataset = load_dataset(
                path=args.dataset_name,
                data_files=data_files,
                cache_dir=args.cache_dir
            )
        else:
            dataset = load_dataset(
                path=args.dataset_name,
                name=args.dataset_config_name,
                cache_dir=args.cache_dir
            )
        map_functions = {
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
        # convert dataset to prompt-completion (instruction) or conversational format
        mapping_fn = map_functions[args.instruction_format]

        train_dataset = dataset[args.train_split].map(
            mapping_fn,
            remove_columns=list(dataset[args.train_split].features),
            num_proc=multiprocessing.cpu_count(),
        )
        if args.eval_split:
            eval_dataset = dataset[args.eval_split].map(
                mapping_fn,
                remove_columns=list(dataset[args.eval_split].features),
                num_proc=multiprocessing.cpu_count(),
            )
        else:
            eval_dataset = None

    accelerator.print(f"Loaded {len(train_dataset)} training samples.")
    if eval_dataset:
        accelerator.print(f"Loaded {len(eval_dataset)} evaluation samples.\n")

    return train_dataset, eval_dataset, config

def get_lora_config(args):
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft_cfg

def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.dataset_mix_config:
        if not os.path.exists(args.dataset_mix_config):
            raise ValueError(f"Dataset mix config file not found: {args.dataset_mix_config}")
    elif not args.dataset_name or not args.instruction_format:
        raise ValueError("You must provide either --dataset_mix_config or both --dataset_name and --instruction_format.")

    accelerator = Accelerator()
    accelerator.print(f"Training config:\n{args}\n")

    # Initialize wandb
    project_name = os.environ.get("WANDB_PROJECT", "default_project_name")
    run_name = f"{project_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if accelerator.is_main_process:
        wandb.init(
            project=project_name,
            name=run_name,
            mode="offline",
        )

    accelerator.print("Initializing tokenizer and model...")
    tokenizer = make_tokenizer(args)
    PartialState().wait_for_everyone()

    quantization_config = None
    model = make_model(args, quantization_config)
    
    gradient_checkpointing_kwargs = None
    if args.gradient_checkpointing:
        gradient_checkpointing_kwargs = {'use_reentrant':False}

    
    # NOTE: uncommment - necessary for Qwen!, FIXME: give as param - or do it with chat_template_path?
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n\n        {{- '<|im_start|>' + message.role }}\n        {% generation %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- content }}\n            {%- endif %}\n        {%- else %}\n            {{- content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>' }}\n        {% endgeneration %}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

    PartialState().wait_for_everyone()

    accelerator.print("Loading dataset...")

    train_dataset, eval_dataset, data_config = make_dataset(args, accelerator)
    
    PartialState().wait_for_everyone()

    peft_cfg = None
    if args.use_peft:  # qLoRA -> quantize model
        accelerator.print("Using LoRA")
        peft_cfg = get_lora_config(args)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        tf32=True,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        use_liger_kernel=args.use_liger,
        eval_strategy="steps" if eval_dataset else "no",
        push_to_hub=False,
        optim=args.optim,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        packing=args.packing,
        padding_free=True,  # TODO: add as param - requires flash_attention_2
        # model_init_kwargs={"attn_implementation": "flash_attention_2"},
        completion_only_loss=args.completion_only_loss,
        assistant_only_loss=args.assistant_only_loss,
        max_length=args.max_length,
        neftune_noise_alpha=args.neftune_noise_alpha,
        activation_offloading=args.activation_offloading,
        eos_token="<|im_end|>",  # uncomment for conversational format!
        ddp_timeout=18000,  # avoid nccl errors when tokenizing large datasets
    )
    accelerator.print(f"SFT config:\n{sft_config}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        processing_class=tokenizer,
        peft_config=peft_cfg,
        callbacks=[MemoryUsageCallback(step_interval=args.logging_steps)],
    )
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    accelerator.print(f"\nTrainable parameters: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")

    PartialState().wait_for_everyone()  

    trainer.train(resume_from_checkpoint=args.resume)
    
    accelerator.print("Training complete.")

    accelerator.print("Saving trainer state...")
    trainer.save_state()

    trainer.accelerator.wait_for_everyone()  # save model on main process
    accelerator.print("Saving model...")
    trainer.save_model()
    accelerator.print(f"Model saved to {args.output_dir}")

    if args.push_to_hub:
        accelerator.print("Pushing to hub...")
        trainer.push_to_hub()

    trainer.accelerator.wait_for_everyone()  # save tokenizer and info on main process
    if trainer.accelerator.is_main_process:
        accelerator.print("Saving tokenizer...")
        tokenizer.save_pretrained(args.output_dir)

        # save SFT info
        if args.dataset_mix_config:
            dataset_names = [dataset["name"] for dataset in data_config["datasets"]]
        else:
            dataset_names = [args.dataset_name]
        kwargs = {
            "finetuned_from": args.model_name_or_path,
            "dataset": dataset_names,
        }
        # trainer.create_model_card(**kwargs)  # deprecated
        # restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(args.output_dir)

    if args.use_peft:
        accelerator.print("Merging LoRA adapters into base model...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(
            args.output_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        accelerator.print(f"Merged model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
