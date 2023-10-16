import torch
from torch.nn import Linear, MSELoss
from torch.cuda.amp import autocast
from torch.optim import AdamW, SGD
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

in_dim = 16384
out_dim = 4096
batch_size = 4096
print(f'batch={batch_size}')

use_mixed = False
print(f'precision: {"mixed" if use_mixed else "uniform"}')

model = Linear(in_features=in_dim, out_features=out_dim, device=device, bias=False)
print(f'after declare model: {mem()}')

# optim = AdamW(model.parameters(), lr=2e-5)
momentum=0.
optim = SGD(model.parameters(), lr=2e-5, momentum=momentum)
optim_extra_desc = f', mom={momentum}' if isinstance(optim, SGD) else ''
print(f'after declare optim ({type(optim).__name__}{optim_extra_desc}): {mem()}')

loss_fn = MSELoss()

precision_ctx = autocast(dtype=torch.bfloat16, cache_enabled=True) if use_mixed else nullcontext()

steps = 1
microsteps = 2
for step in range(steps):
  step_indicator = f'[step {step}] ' if steps > 1 else ''
  for microstep in range(microsteps):
    microstep_indicator = f'[microstep {microstep}] ' if microsteps > 1 else ''
    step_and_micro_indicator = f'{step_indicator}{microstep_indicator}'

    x = torch.randn(batch_size, in_dim, device=device, requires_grad=True)
    y_true = torch.randn(batch_size, out_dim, device=device, requires_grad=False)
    print(pretty_mem(step_and_micro_indicator, f'after declare x/y:'))

    with precision_ctx:
      y_pred = model.forward(x)
      # y_pred.retain_grad()
      print(pretty_mem(step_and_micro_indicator, f'after model.forward:'))

      # y_pred2 = y_pred.float()
      # print(f'after y_pred cast: {mem()}')
      loss = loss_fn.forward(y_pred, y_true)
      # loss.retain_grad()
      print(pretty_mem(step_and_micro_indicator, f'after loss:'))

    if microsteps > 1:
      loss /= microsteps
    loss.backward()
    print(pretty_mem(step_and_micro_indicator, f'after backward:'))
    del x.grad

  optim.step()
  print(f'{step_indicator}after optim.step: {mem()}')
  set_to_none=False
  optim.zero_grad(set_to_none=set_to_none)
  print(f'{step_indicator}after optim.zero_grad ({set_to_none}): {mem()}')

print(f'model  (f32): {mib_str(model.weight.numel()*4)}')
if use_mixed:
  print(f'model  (f16): {mib_str(model.weight.numel()*2)}')
print(f'x      (f32): {mib_str(x.numel()*4)}')
print(f'y_true (f32): {mib_str(y_true.numel()*4)}')
if use_mixed:
  print(f'y_pred (f16): {mib_str(y_pred.numel()*2)}')
else:
  print(f'y_pred (f32): {mib_str(y_pred.numel()*4)}')
# torch.cuda.memory_snapshot()
# [(f"{m['address']:02x}"[3:-5], mib_str(m['allocated_size'])) for m in torch.cuda.memory_snapshot()]
# print('\n'.join([f"""{f"{m['address']:02x}"[3:-5]}: {mib_str(m['allocated_size']).rjust(9)} alloc""" for m in torch.cuda.memory_snapshot()]))
# print('\n'.join([f"""{f"{m['address']:02x}"[3:-5]}: {mib_str(m['allocated_size']).rjust(9)} alloc, {mib_str(m['total_size']).rjust(9)} total""" for m in torch.cuda.memory_snapshot()]))
pass