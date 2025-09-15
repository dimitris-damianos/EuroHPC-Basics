from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch
import threading
import time
import subprocess
import datetime

class MemoryUsageCallback(TrainerCallback):
    def __init__(self, step_interval=1):
        self.step_interval = step_interval
    def get_nvidia_smi_memory(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return int(result.stdout.strip().split('\n')[0])  # MB
        except Exception as e:
            print(f"[MemoryCallback] Failed to read nvidia-smi: {e}")
            return -1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.step_interval == 0:
            nvidia_mem = self.get_nvidia_smi_memory()

            print(f"[{datetime.datetime.now()}][Step {state.global_step}] Nvidia-smi: {nvidia_mem} MB")

            torch.cuda.reset_peak_memory_stats()