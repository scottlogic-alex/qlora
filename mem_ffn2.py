import torch
from torch import FloatTensor
from torch.nn import Linear, MSELoss, Module, Sequential, GELU
from torch.cuda.amp import autocast
from torch.optim import SGD
from typing import List
from contextlib import nullcontext
import argparse

def mib_str(bytes: int) -> str:
  return f'{f"{bytes/1024**2:.2f}".rjust(8)} MiB'

def mem(params: int, device_ix=0):
  alloc: int = torch.cuda.memory_allocated(device_ix)
  alloc_plus_reserved: int = torch.cuda.memory_reserved(device_ix)
  reserved: int = alloc_plus_reserved-alloc
  bytes_per_param = alloc_plus_reserved/params
  return f'{mib_str(alloc_plus_reserved)}; {f"{bytes_per_param:.2f}".rjust(5)} bytes/param (of which {mib_str(alloc)} alloc, {mib_str(reserved)} reserved)'

device=torch.device('cuda')

class FFN(Module):
  layers: Sequential
  def __init__(self, n_layers: int, in_dim: int, hidden_dim: int, out_dim: int, bias: bool, device=None, dtype=None) -> None:
    super().__init__()
    assert n_layers > 0, "FFN requires at least 1 layer"
    layers: List[Linear] = []

    for layer_ix in range(n_layers):
      in_features: int = in_dim if layer_ix == 0 else hidden_dim
      out_features: int = out_dim if layer_ix == n_layers-1 else hidden_dim
      layer = Linear(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
      layers.append(layer)
      if layer_ix != n_layers - 1 and n_layers > 1:
        gate = GELU()
        layers.append(gate)
    self.layers = Sequential(*layers)
  
  def forward(self, x: FloatTensor) -> FloatTensor:
    x: FloatTensor = self.layers(x)
    return x

def main():
  parser = argparse.ArgumentParser(prog='FFN memory measurer')
  parser.add_argument('--in_dim', type=int, default=4096)
  parser.add_argument('--hidden_dim', type=int, default=16384)
  parser.add_argument('--out_dim', type=int, default=4096)
  parser.add_argument('--n_layers', type=int, default=3)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--steps', type=int, default=1)
  parser.add_argument('--microsteps', type=int, default=1)
  parser.add_argument('--mixed_bf16', action='store_true')
  args = parser.parse_args()

  print(f'''grad acc {'en' if args.microsteps > 1 else 'dis'}abled
precision: {'mixed' if args.mixed_bf16 else 'uniform'}''')

  realloc_each_microstep = True
  optim_set_to_none=True

  model = FFN(
    n_layers=args.n_layers,
    in_dim=args.in_dim,
    hidden_dim=args.hidden_dim,
    out_dim=args.out_dim,
    device=device,
    bias=False,
  )
  param_count = sum([p.numel() for p in model.parameters()])

  optim = SGD(model.parameters(), lr=2e-5, momentum=0.)
  loss_fn = MSELoss()

  precision_ctx = autocast(dtype=torch.bfloat16, cache_enabled=True) if args.mixed_bf16 else nullcontext()

  step_indicator_padding = ''.rjust(9)
  microstep_indicator_padding = ''.rjust(14)

  for step in range(args.steps):
    step_indicator = f'[step {step}] ' if args.steps > 1 else step_indicator_padding
    for microstep in range(args.microsteps):
      microstep_indicator = f'[microstep {microstep}] ' if args.microsteps > 1 else microstep_indicator_padding
      step_and_micro_indicator = f'{step_indicator}{microstep_indicator}'
      if realloc_each_microstep or step == 0 and microstep == 0:
        x = torch.randn(args.batch_size, args.in_dim, device=device, requires_grad=False)
        y_true = torch.randn(args.batch_size, args.out_dim, device=device, requires_grad=False)

      with precision_ctx:
        y_pred = model.forward(x)
        loss = loss_fn.forward(y_pred, y_true)

      if args.microsteps > 1:
        loss /= args.microsteps
      loss.backward()

      print(f'{step_and_micro_indicator}after loss backward: {mem(param_count)}')

    optim.step()
    optim.zero_grad(set_to_none=optim_set_to_none)
    print(f'{step_indicator}{microstep_indicator_padding}after  zero_grad()): {mem(param_count)}')
  
if __name__ == "__main__":
  main()