# https://pypi.org/project/nvidia-ml-py/
# pip install nvidia-ml-py
import pynvml as nvml
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class MemoryUsageCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        fb_info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
        # you'll notice this is slightly higher than the summary you get in nvidia-smi.
        # that's because it's used + reserved.
        # you can compute used+reserved via nvidia-smi yourself:
        # nvidia-smi -i 0 -q -d MEMORY
        print(f'Used {fb_info.used >> 20}MiB / {fb_info.total >> 20}MiB')