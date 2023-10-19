from __future__ import annotations
# https://pypi.org/project/nvidia-ml-py/
# pip install nvidia-ml-py
import pynvml as nvml
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import NamedTuple, List, Callable
import torch

def justify(bytes: int) -> str:
    return str(bytes).rjust(5)

def to_MiB(bytes: int) -> int:
    return bytes >> 20

class NVMLMemoryStats(NamedTuple):
    used_bytes: int
    total_bytes: int

def nvml_memory_usage(handle) -> NVMLMemoryStats:
    fb_info = nvml.nvmlDeviceGetMemoryInfo(handle)
    return NVMLMemoryStats(used_bytes=fb_info.used, total_bytes=fb_info.total)

class TorchMemoryStats(NamedTuple):
    used_bytes: int
    used_plus_reserved_bytes: int

def torch_memory_usage(device=0) -> TorchMemoryStats:
    used = torch.cuda.memory_allocated(device)
    used_plus_reserved = torch.cuda.memory_reserved(device)
    return TorchMemoryStats(used_bytes=used, used_plus_reserved_bytes=used_plus_reserved)

class MemoryUsageCallback(TrainerCallback):
    visible_nvml_device_ixs: List[int]
    substep: int
    make_memory_str: Callable[[MemoryUsageCallback, str], str]
    # when grad acc is disabled: there is no microstep detail to print. set True to justify to the same width anyway, to make results easier to compare
    justify_empty_microstep_detail: bool
    def __init__(self, brief=True) -> None:
        super().__init__()
        nvml.nvmlInit()
        device_count: int = nvml.nvmlDeviceGetCount()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(did) for did in range(device_count)]
        # if you use `CUDA_VISIBLE_DEVICES` to hide devices: nvml will see them but torch won't.
        # for example, if you have 2xA40 (NVML device 0 and device 1),
        # CUDA_VISIBLE_DEVICES=1 will hide nvml device 0.
        # so cuda:0 will correspond to nvml device 1.
        self.visible_nvml_device_ixs = [torch.cuda._get_nvml_device_index(ix) for ix in range(torch.cuda.device_count())]
        self.substep = 0
        self.make_memory_str = self.make_memory_str_brief if brief else self.make_memory_str_
        self.justify_empty_microstep_detail = True

    def make_memory_str_(self, qualifier: str) -> str:
        overall_nvml_used = 0
        overall_nvml_total = 0
        out = f'{qualifier}\n  NVML memory stats (used+reserved, all processes):'
        # for did, handle in enumerate(self.handles):
        for nvml_did in self.visible_nvml_device_ixs:
            nvml_handle = self.handles[nvml_did]
            used_bytes, total_bytes = nvml_memory_usage(nvml_handle)
            overall_nvml_used += used_bytes
            overall_nvml_total += total_bytes
            # you'll notice this is slightly higher than the summary you get in nvidia-smi.
            # that's because it's used + reserved.
            # you can compute used+reserved via nvidia-smi yourself:
            # nvidia-smi -i 0 -q -d MEMORY
            out += f'\n    Device {nvml_did}: Used {to_MiB(used_bytes)} MiB / {to_MiB(total_bytes)} MiB'
        if len(self.visible_nvml_device_ixs) > 1:
            out += f'\n    Overall: Used {to_MiB(overall_nvml_used)} MiB / {to_MiB(overall_nvml_total)} MiB'

        overall_torch_used = 0
        overall_torch_used_plus_reserved_bytes = 0
        out += '\n  Torch memory stats (allocated, reserved):'
        for torch_did in range(torch.cuda.device_count()):
            nvml_did: int = self.visible_nvml_device_ixs[torch_did]
            used_bytes, used_plus_reserved_bytes = torch_memory_usage(torch_did)
            overall_torch_used += used_bytes
            overall_torch_used_plus_reserved_bytes += used_plus_reserved_bytes
            # Allocated/resident includes stuff like optimizer state
            # Reserved includes temporary state like gradients
            out += f'\n    Device {nvml_did}: Used {to_MiB(used_plus_reserved_bytes)} MiB (Allocated: {to_MiB(used_bytes)} MiB, Reserved {to_MiB(used_plus_reserved_bytes-used_bytes)} MiB)'
        if len(self.visible_nvml_device_ixs) > 1:
            out += f'\n    Overall: Used {to_MiB(overall_torch_used_plus_reserved_bytes)} MiB (Allocated: {to_MiB(overall_torch_used)} MiB, Reserved {to_MiB(overall_torch_used_plus_reserved_bytes-overall_torch_used)} MiB)'
        return out
    
    def make_memory_str_brief(self, qualifier: str) -> str:
        spacer: str = ''.rjust(len(qualifier))
        overall_torch_used_bytes = 0
        overall_torch_used_plus_reserved_bytes = 0
        overall_nvml_used_bytes = 0
        overall_nvml_total_bytes = 0
        lines: List[str] = []
        for torch_did in range(torch.cuda.device_count()):
            nvml_did: int = self.visible_nvml_device_ixs[torch_did]
            nvml_handle = self.handles[nvml_did]
            nvml_used_bytes, nvml_total_bytes = nvml_memory_usage(nvml_handle)
            overall_nvml_used_bytes += nvml_used_bytes
            overall_nvml_total_bytes += nvml_total_bytes
            torch_used_bytes, torch_used_plus_reserved_bytes = torch_memory_usage(torch_did)
            overall_torch_used_bytes += torch_used_bytes
            overall_torch_used_plus_reserved_bytes += torch_used_plus_reserved_bytes
            lines.append(f'Device {nvml_did}: Torch Used {justify(to_MiB(torch_used_plus_reserved_bytes))} MiB (Allocated: {justify(to_MiB(torch_used_bytes))} MiB, Reserved {justify(to_MiB(torch_used_plus_reserved_bytes-torch_used_bytes))} MiB), NVML {justify(to_MiB(nvml_used_bytes))} / {justify(to_MiB(nvml_total_bytes))} MiB')
        if len(self.visible_nvml_device_ixs) > 1:
            lines.append(f'Overall:  Torch Used {justify(to_MiB(overall_torch_used_plus_reserved_bytes))} MiB (Allocated: {justify(to_MiB(overall_torch_used_bytes))} MiB, Reserved {justify(to_MiB(overall_torch_used_plus_reserved_bytes-overall_torch_used_bytes))} MiB), NVML {justify(to_MiB(overall_nvml_used_bytes))} / {justify(to_MiB(overall_nvml_total_bytes))} MiB')
        first, *rest = lines
        out: str = '\n'.join((
            f'{qualifier} {first}',
            *(f'{spacer} {line}' for line in rest)
        ))
        return out
    
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.synchronize()
        print(self.make_memory_str(f'step {state.global_step}, microstep {self.substep}'))
        self.substep += 1
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.substep = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.synchronize()
        microstep_detail = '(end)'.rjust(13) if self.substep or self.justify_empty_microstep_detail else ''
        qualifier = f'step {state.global_step - 1}{microstep_detail}'
        print(self.make_memory_str(qualifier))
        

