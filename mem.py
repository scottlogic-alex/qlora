import torch
from torch.nn import Linear, MSELoss
from torch.cuda.amp import autocast
from torch.optim import AdamW, SGD
from contextlib import nullcontext

def gib_str(bytes: int) -> str:
  return f'{bytes/1024**3:.2f}GiB'

def mib_str(bytes: int) -> str:
  return f'{bytes/1024**2:.2f}MiB'

def mem():
  # return torch.cuda.memory_summary()
  return f'{mib_str(torch.cuda.memory_allocated(0))} alloc, {mib_str(torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))} reserved'

device=torch.device('cuda')

in_dim = 16384
out_dim = 4096
batch_size = 4096
print(f'batch={batch_size}')

x = torch.randn(batch_size, in_dim, device=device, requires_grad=True)
y_true = torch.randn(batch_size, out_dim, device=device, requires_grad=False)
print(f'after declare x/y: {mem()}')

model = Linear(in_features=in_dim, out_features=out_dim, device=device, bias=False)
print(f'after declare model: {mem()}')

# optim = AdamW(model.parameters(), lr=2e-5)
momentum=0.
optim = SGD(model.parameters(), lr=2e-5, momentum=momentum)
optim_extra_desc = f', momentum={momentum}' if isinstance(optim, SGD) else ''
print(f'after declare optim ({type(optim).__name__}{optim_extra_desc}): {mem()}')

loss_fn = MSELoss()

use_mixed = True
precision_ctx = autocast(dtype=torch.bfloat16, cache_enabled=True) if use_mixed else nullcontext()
print(f'precision: {"mixed" if use_mixed else "uniform"}')

steps = 1
microsteps = 2
for step in range(steps):
  step_indicator = f'[step {step}] ' if steps > 1 else ''
  for microstep in range(microsteps):
    microstep_indicator = f'[microstep {microstep}] ' if microsteps > 1 else ''
    with precision_ctx:
      y_pred = model.forward(x)
      # y_pred.retain_grad()
      print(f'{step_indicator}{microstep_indicator}after model.forward: {mem()}')

      # y_pred2 = y_pred.float()
      # print(f'after y_pred cast: {mem()}')
      loss = loss_fn.forward(y_pred, y_true)
      # loss.retain_grad()
      print(f'{step_indicator}{microstep_indicator}after loss: {mem()}')

    if microsteps > 1:
      loss /= microsteps
    loss.backward()
    print(f'{step_indicator}{microstep_indicator}after backward: {mem()}')

  optim.step()
  print(f'{step_indicator}after optim.step: {mem()}')
  set_to_none=False
  optim.zero_grad(set_to_none=set_to_none)
  print(f'{step_indicator}after optim.zero_grad ({set_to_none}): {mem()}')

print(f'model  (f32): {mib_str(model.weight.numel()*4)}\nmodel  (f16): {mib_str(model.weight.numel()*2)}\nx      (f32): {mib_str(x.numel()*4)}\ny_true (f32): {mib_str(y_true.numel()*4)}\ny_pred (f16): {mib_str(y_pred.numel()*2)}')
# torch.cuda.memory_snapshot()
# [(f"{m['address']:02x}"[3:-5], mib_str(m['allocated_size'])) for m in torch.cuda.memory_snapshot()]
# print('\n'.join([f"""{f"{m['address']:02x}"[3:-5]}: {mib_str(m['allocated_size']).rjust(9)} alloc""" for m in torch.cuda.memory_snapshot()]))
# print('\n'.join([f"""{f"{m['address']:02x}"[3:-5]}: {mib_str(m['allocated_size']).rjust(9)} alloc, {mib_str(m['total_size']).rjust(9)} total""" for m in torch.cuda.memory_snapshot()]))
pass