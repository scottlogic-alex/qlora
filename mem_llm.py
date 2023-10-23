from __future__ import annotations
import torch
from torch import BoolTensor, LongTensor
from torch.cuda.amp import autocast
from torch.optim import SGD
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig, PretrainedConfig#, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import argparse
from typing import List, NamedTuple#, Union
from logging import getLogger, Formatter, StreamHandler
import logging

logger = getLogger(__file__)
log_fmt_nominal = '{message}'

rank0_logger = logger.getChild('rank0')
rank0_logger.setLevel(logging.INFO)
rank0_handler = StreamHandler()
rank0_handler.setFormatter(Formatter(fmt=log_fmt_nominal, style='{'))

all_logger = logger.getChild('all')
all_logger.setLevel(logging.INFO)
all_handler = StreamHandler()
all_handler.setFormatter(Formatter(fmt=log_fmt_nominal, style='{'))
all_logger.addHandler(all_handler)

class TorchCudaMemoryBytes(NamedTuple):
  alloc: int
  alloc_plus_reserved: int

  def __add__(self, other: TorchCudaMemoryBytes) -> TorchCudaMemoryBytes:
    return TorchCudaMemoryBytes(
      alloc=self.alloc + other.alloc,
      alloc_plus_reserved=self.alloc_plus_reserved + other.alloc_plus_reserved,
    )

def mib_str(bytes: int) -> str:
  return f'{f"{bytes/1024**2:.2f}".rjust(8)} MiB'

def device_mem(device_ix=0) -> TorchCudaMemoryBytes:
  alloc: int = torch.cuda.memory_allocated(device_ix)
  alloc_plus_reserved: int = torch.cuda.memory_reserved(device_ix)
  return TorchCudaMemoryBytes(
    alloc=alloc,
    alloc_plus_reserved=alloc_plus_reserved,
  )

def mem_summary(mem_metric: TorchCudaMemoryBytes, params: int) -> str:
  alloc, alloc_plus_reserved = mem_metric
  reserved: int = alloc_plus_reserved-alloc
  bytes_per_param = alloc_plus_reserved/params
  return f'{mib_str(alloc_plus_reserved)}; {f"{bytes_per_param:.2f}".rjust(5)} bytes/param (of which {mib_str(alloc)} alloc, {mib_str(reserved)} reserved)'

def mem(preamble: str, params: int, multi_device: bool, device_id: int):
  if multi_device and torch.cuda.device_count() > 1:
    out_lines: List[str] = []
    total_metric = TorchCudaMemoryBytes(0, 0)
    once_preamble: str = preamble
    for device_ix in range(torch.cuda.device_count()):
      mem_metric: TorchCudaMemoryBytes = device_mem(device_ix)
      total_metric += mem_metric
      summary: str = mem_summary(mem_metric, params)
      out_lines.append(f'{once_preamble} d{device_ix}: {summary}')
      once_preamble = ''.rjust(len(preamble))
    total_summary: str = mem_summary(total_metric, params)
    out_lines.append(f'{once_preamble}  =: {total_summary}')
    return '\n'.join(out_lines)
  else:
    mem_metric: TorchCudaMemoryBytes = device_mem(device_id)
    summary: str = mem_summary(mem_metric, params)
    return f'{preamble}: {summary}'

def main():
  parser = argparse.ArgumentParser(prog='LLM finetuning memory measurer')
  parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m-deduped')
  parser.add_argument('--cache_dir', type=str, default=None)
  parser.add_argument('--batch_size', type=int, default=1, help='per-process batch size')
  parser.add_argument('--seq_len', type=int, default=8)
  parser.add_argument('--grad_ckpt', action='store_true')
  parser.add_argument('--steps', type=int, default=1)
  parser.add_argument('--microsteps', type=int, default=1)
  parser.add_argument('--device_map_auto', action='store_true')
  parser.add_argument('--skip_ckpt_load', action='store_true')
  parser.add_argument('--mixed_bf16', action='store_true')
  # if we run distributed: torchrun passes this option in. we use dist.get_rank() instead, to stick closer to the torch distributed tutorial.
  parser.add_argument('--local-rank', type=int, default=None)
  args = parser.parse_args()

  is_distributed: bool = args.local_rank is not None
  if is_distributed:
    assert not args.device_map_auto, "Have not implemented support for device_map='auto' in distributed mode"
    dist.init_process_group("nccl")
    rank: int = dist.get_rank()
    all_handler.setFormatter(Formatter(fmt=f'r{rank} {log_fmt_nominal}', style='{'))
    rank0_handler.setFormatter(Formatter(fmt=f'   {log_fmt_nominal}', style='{'))
  else:
    rank = 0
  if rank == 0:
    rank0_logger.addHandler(rank0_handler)
    rank0_logger.setLevel(logging.INFO)
  else:
    rank0_logger.setLevel(logging.ERROR)
  
  device_id: int = rank % torch.cuda.device_count()
  device = torch.device(device_id)

  rank0_logger.info(f"grad acc:  {'en' if args.microsteps > 1 else 'dis'}abled")
  rank0_logger.info(f"precision: {'mixed' if args.mixed_bf16 else 'uniform'}")

  realloc_each_microstep = True
  optim_set_to_none = True

  device=torch.device('cuda', device_id)

  if args.skip_ckpt_load:
    assert not args.device_map_auto, "for some reason device_map='auto' support isn't implemented for AutoModelForCausalLM#from_config()"
    config: PretrainedConfig = AutoConfig.from_pretrained(
      args.model_name,
      cache_dir=args.cache_dir,
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_config(config)
  else:
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
      args.model_name,
      cache_dir=args.cache_dir,
      device_map='auto' if args.device_map_auto else None,
    )
  param_count = sum([p.numel() for p in model.parameters()])
  rank0_logger.info(f'param count: {param_count}')

  # tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(
  #   args.model_name,
  #   cache_dir=args.cache_dir,
  #   padding_side="right",
  # )
  # if args.batch_size > 1:
  #   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  #   model.resize_token_embeddings(len(tokenizer))
  if not args.device_map_auto:
    model.to(device)

  if args.grad_ckpt:
    model.gradient_checkpointing_enable()

  # accelerate uses these for the general case, but I think they're no-ops in our case; the model begins in train mode, and zero_grad does nothing if your grads are None
  # model.train()
  # model.zero_grad()

  vocab_size: int = model.config.vocab_size

  if is_distributed:
    model = DDP(model, device_ids=[device_id])

  optim = SGD(model.parameters(), lr=2e-5, momentum=0.)

  precision_ctx = autocast(dtype=torch.bfloat16, cache_enabled=True) if args.mixed_bf16 else nullcontext()

  step_indicator_padding = ''.rjust(9)
  microstep_indicator_padding = ''.rjust(14)

  for step in range(args.steps):
    step_indicator = f'[step {step}] ' if args.steps > 1 else step_indicator_padding
    for microstep in range(args.microsteps):
      microstep_indicator = f'[microstep {microstep}] ' if args.microsteps > 1 else microstep_indicator_padding
      step_and_micro_indicator = f'{step_indicator}{microstep_indicator}'
      if realloc_each_microstep or step == 0 and microstep == 0:
        input_ids: LongTensor = torch.randint(vocab_size, (args.batch_size, args.seq_len), device=device, requires_grad=False)
        labels: LongTensor = input_ids.clone()
        labels[:,:4] = -100
        attention_mask: BoolTensor = torch.ones_like(input_ids, dtype=torch.bool)

      with precision_ctx:
        model_out: CausalLMOutputWithPast = model.forward(
          input_ids=input_ids,
          labels=labels,
          attention_mask=attention_mask,
        )
        del model_out.logits, model_out.past_key_values

      if args.microsteps > 1:
        model_out.loss /= args.microsteps
      model_out.loss.backward()
      del model_out

      all_logger.info(mem(f'{step_and_micro_indicator}after loss backward', params=param_count, multi_device=args.device_map_auto, device_id=device_id))

    optim.step()
    optim.zero_grad(set_to_none=optim_set_to_none)
    all_logger.info(mem(f'{step_indicator}{microstep_indicator_padding}after  zero_grad())', params=param_count, multi_device=args.device_map_auto, device_id=device_id))

    if is_distributed:
      # the tutorials says we should destroy our process group, but when I tried this it complained that there was no process group.
      # dist.destroy_process_group()
      pass
  
if __name__ == "__main__":
  main()