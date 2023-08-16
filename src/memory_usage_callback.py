# https://pypi.org/project/nvidia-ml-py/
# pip install nvidia-ml-py
import pynvml as nvml
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class MemoryUsageCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        nvml.nvmlInit()
        device_count: int = nvml.nvmlDeviceGetCount()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(did) for did in range(device_count)]

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        overall_used = 0
        overall_total = 0
        for did, handle in enumerate(self.handles):
            fb_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            overall_used += fb_info.used
            overall_total += fb_info.total
            # you'll notice this is slightly higher than the summary you get in nvidia-smi.
            # that's because it's used + reserved.
            # you can compute used+reserved via nvidia-smi yourself:
            # nvidia-smi -i 0 -q -d MEMORY
            print(f'Device {did}: Used {fb_info.used >> 20}MiB / {fb_info.total >> 20}MiB')
        if len(self.handles) > 1:
            print(f'Overall: Used {overall_used >> 20}MiB / {overall_total >> 20}MiB')