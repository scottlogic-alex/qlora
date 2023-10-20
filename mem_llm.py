import torch
from torch import BoolTensor, LongTensor, FloatTensor
from torch.cuda.amp import autocast
from torch.optim import SGD
from contextlib import nullcontext
from transformers import GPTNeoXForCausalLM#, GPTNeoXTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast
import argparse

def mib_str(bytes: int) -> str:
  return f'{f"{bytes/1024**2:.2f}".rjust(8)} MiB'

def mem(params: int, device_ix=0):
  alloc: int = torch.cuda.memory_allocated(device_ix)
  alloc_plus_reserved: int = torch.cuda.memory_reserved(device_ix)
  reserved: int = alloc_plus_reserved-alloc
  bytes_per_param = alloc_plus_reserved/params
  return f'{mib_str(alloc_plus_reserved)}; {f"{bytes_per_param:.2f}".rjust(5)} bytes/param (of which {mib_str(alloc)} alloc, {mib_str(reserved)} reserved)'

def main():
  parser = argparse.ArgumentParser(prog='LLM finetuning memory measurer')
  parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m-deduped')
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--seq_len', type=int, default=8)
  parser.add_argument('--grad_ckpt', action='store_true')
  parser.add_argument('--steps', type=int, default=1)
  parser.add_argument('--microsteps', type=int, default=1)
  parser.add_argument('--mixed_bf16', action='store_true')
  args = parser.parse_args()

  print(f'''grad acc {'en' if args.microsteps > 1 else 'dis'}abled
precision: {'mixed' if args.mixed_bf16 else 'uniform'}''')

  realloc_each_microstep = True
  optim_set_to_none = True

  device=torch.device('cuda')

  model: GPTNeoXForCausalLM = GPTNeoXForCausalLM.from_pretrained(
    args.model_name,
    use_safetensors=True,
  )
  param_count = sum([p.numel() for p in model.parameters()])

  # tokenizer: GPTNeoXTokenizerFast = GPTNeoXTokenizerFast.from_pretrained(
  #   args.model_name,
  #   padding_side="right",
  # )
  # if args.batch_size > 1:
  #   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  #   model.resize_token_embeddings(len(tokenizer))
  model.to(device)

  if args.grad_ckpt:
    model.gradient_checkpointing_enable()
  model.train()
  model.zero_grad()

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
        input_ids: LongTensor = torch.randint(model.config.vocab_size, (args.batch_size, args.seq_len), device=device, requires_grad=False)
        labels: LongTensor = input_ids.clone()
        labels[:,:4] = -100
        attention_mask: BoolTensor = torch.ones_like(input_ids, dtype=torch.bool)

      with precision_ctx:
        model_out: CausalLMOutputWithPast = model.forward(
          input_ids=input_ids,
          labels=labels,
          attention_mask=attention_mask,
        )
        loss: FloatTensor = model_out['loss']

      if args.microsteps > 1:
        loss /= args.microsteps
      loss.backward()

      print(f'{step_and_micro_indicator}after loss backward: {mem(param_count)}')

    optim.step()
    optim.zero_grad(set_to_none=optim_set_to_none)
    print(f'{step_indicator}{microstep_indicator_padding}after  zero_grad()): {mem(param_count)}')
  
if __name__ == "__main__":
  main()