import torch
from torch import FloatTensor
from torch.nn import Linear, MSELoss, Module, Sequential, GELU
from torch.cuda.amp import autocast
from torch.optim import AdamW, SGD
from typing import List
from contextlib import nullcontext

def gib_str(bytes: int) -> str:
  return f'{bytes/1024**3:.2f}GiB'

def mib_str(bytes: int) -> str:
  return f'{bytes/1024**2:.2f}MiB'

def mem(device_ix=0):
  # return torch.cuda.memory_summary()
  alloc: int = torch.cuda.memory_allocated(device_ix)
  total: int = torch.cuda.memory_reserved(device_ix)
  reserved: int = total-alloc
  return f'{mib_str(alloc)} alloc, {mib_str(reserved)} reserved, {mib_str(total)} total'

def pretty_mem(
  preamble: str,
  context: str,
  device_ix=0
):
  alloc: int = torch.cuda.memory_allocated(device_ix)
  total: int = torch.cuda.memory_reserved(device_ix)
  reserved: int = total-alloc
  return f'{preamble}{context.rjust(20)} {mib_str(alloc).rjust(10)} alloc {mib_str(reserved).rjust(10)} reserved {mib_str(total).rjust(10)} total'

device=torch.device('cuda')

layer_count = 3
in_dim = 4096
hidden_dim = 16384
out_dim = 4096
batch_size = 1024
print(f'batch={batch_size}')

use_mixed = True
print(f'precision: {"mixed" if use_mixed else "uniform"}')

cache_enabled = True
if use_mixed:
  print(f'cache_enabled: {cache_enabled}')

realloc_each_microstep = True
print(f'realloc_each_microstep: {realloc_each_microstep}')

optim_set_to_none=True
print(f'optim_set_to_none: {optim_set_to_none}')

class LoggingSequential(Sequential):
  def forward(self, input: FloatTensor, step_and_micro_indicator = '') -> FloatTensor:
    for ix, module in enumerate(self):
      input: FloatTensor = module(input)
      layer_label = 'G' if isinstance(module, GELU) else 'L'
      dense_ix = int(ix / 2)
      print(pretty_mem(step_and_micro_indicator, f'after {layer_label}{dense_ix}.forward:'))
    return input

class Model(Module):
  layers: LoggingSequential
  def __init__(self, layer_count: int, in_dim: int, hidden_dim: int, out_dim: int, bias: bool, device=None, dtype=None) -> None:
    super().__init__()
    assert layer_count > 0
    layers: List[Linear] = []
    for layer_ix in range(layer_count):
      in_features: int = in_dim if layer_ix == 0 else hidden_dim
      out_features: int = out_dim if layer_ix == layer_count-1 else hidden_dim
      layer = Linear(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
      layers.append(layer)
      if layer_ix != layer_count - 1:
        gate = GELU()
        layers.append(gate)
    self.layers = LoggingSequential(*layers)
  
  def forward(self, x: FloatTensor, step_and_micro_indicator = '') -> FloatTensor:
    x: FloatTensor = self.layers(x, step_and_micro_indicator=step_and_micro_indicator)
    return x

model = Model(
  layer_count=layer_count,
  in_dim=in_dim,
  hidden_dim=hidden_dim,
  out_dim=out_dim,
  device=device,
  bias=False,
)
print(model)
print(f'after declare model: {mem()}')

# optim = AdamW(model.parameters(), lr=2e-5)
momentum=0.
optim = SGD(model.parameters(), lr=2e-5, momentum=momentum)
optim_extra_desc = f', mom={momentum}' if isinstance(optim, SGD) else ''
print(f'after declare optim ({type(optim).__name__}{optim_extra_desc}): {mem()}')

loss_fn = MSELoss()

precision_ctx = autocast(dtype=torch.bfloat16, cache_enabled=cache_enabled) if use_mixed else nullcontext()

steps = 1
microsteps = 1
for step in range(steps):
  step_indicator = f'[step {step}] ' if steps > 1 else ''
  for microstep in range(microsteps):
    microstep_indicator = f'[microstep {microstep}] ' if microsteps > 1 else ''
    step_and_micro_indicator = f'{step_indicator}{microstep_indicator}'

    if realloc_each_microstep or step == 0 and microstep == 0:
      x = torch.randn(batch_size, in_dim, device=device, requires_grad=False)
      y_true = torch.randn(batch_size, out_dim, device=device, requires_grad=False)
      print(pretty_mem(step_and_micro_indicator, f'after declare x/y:'))

    with precision_ctx:
      y_pred = model.forward(x)
      # y_pred.retain_grad()
      # print(pretty_mem(step_and_micro_indicator, f'after model.forward:'))

      # y_pred2 = y_pred.float()
      # print(f'after y_pred cast: {mem()}')
      loss = loss_fn.forward(y_pred, y_true)
      del y_pred
      # loss.retain_grad()
      print(pretty_mem(step_and_micro_indicator, f'after loss:'))

    if microsteps > 1:
      loss /= microsteps
    loss.backward()
    print(pretty_mem(step_and_micro_indicator, f'after backward:'))
    del loss
    print(pretty_mem(step_indicator, 'after del loss'))

  optim.step()
  print(pretty_mem(step_indicator, 'after optim.step'))
  
  optim.zero_grad(set_to_none=optim_set_to_none)
  print(pretty_mem(step_indicator, f'after optim.zero_grad ({optim_set_to_none})'))

print(f'model     (f32): {mib_str(sum([p.numel() for p in model.parameters()])*4)}')
print(f'model.in  (f32): {mib_str(in_dim*hidden_dim*4)}')
if layer_count > 2:
  print(f'model.mid (f32): {mib_str(hidden_dim**2*4)}')
  print(f'activ.mid (f32): {mib_str(batch_size*hidden_dim*4)}')
print(f'model.out (f32): {mib_str(hidden_dim*out_dim*4)}')
if use_mixed:
  print(f'model     (f16): {mib_str(sum([p.numel() for p in model.parameters()])*2)}')
  print(f'model.in  (f16): {mib_str(in_dim*hidden_dim*2)}')
  if layer_count > 2:
    print(f'model.mid (f16): {mib_str(hidden_dim**2*2)}')
  print(f'activ.mid (f16): {mib_str(batch_size*hidden_dim*2)}')
  print(f'model.out (f16): {mib_str(hidden_dim*out_dim*2)}')
print(f'x         (f32): {mib_str(batch_size*in_dim*4)}')
print(f'y_true    (f32): {mib_str(batch_size*out_dim*4)}')
if use_mixed:
  print(f'y_pred    (f16): {mib_str(batch_size*out_dim*2)}')
else:
  print(f'y_pred    (f32): {mib_str(batch_size*out_dim*4)}')
# torch.cuda.memory_snapshot()
# [(f"{m['address']:02x}"[3:-5], mib_str(m['allocated_size'])) for m in torch.cuda.memory_snapshot()]
# print('\n'.join([f"""{f"{m['address']:02x}"[3:-5]}: {mib_str(m['allocated_size']).rjust(9)} alloc""" for m in torch.cuda.memory_snapshot()]))
# print('\n'.join([f"""{f"{m['address']:02x}"[3:-5]}: {mib_str(m['allocated_size']).rjust(9)} alloc, {mib_str(m['total_size']).rjust(9)} total""" for m in torch.cuda.memory_snapshot()]))
pass