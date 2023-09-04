import torch
from torch import load, Tensor
from typing import Dict, OrderedDict
from safetensors import safe_open
from safetensors.torch import save_file

device=torch.device('cpu')
in_dir: str = 'output_13b_alpaca_special/checkpoint-1664'
out_dir: str = in_dir

embed_tokens: OrderedDict[str, Tensor] = load(f'{in_dir}/embed_tokens.pt', weights_only=False, map_location=device)
lm_head: OrderedDict[str, Tensor] = load(f'{in_dir}/lm_head.pt', weights_only=False, map_location=device)

out_path: str = f'{out_dir}/overlays.safetensors'
tensors: Dict[str, Tensor] = {
  'base_model.model.base_model.model.model.embed_tokens.weight': embed_tokens['weight'],
  'base_model.model.base_model.model.lm_head.weight': lm_head['weight'],
}
save_file(tensors, out_path)

with safe_open(out_path, framework='pt', device='cpu') as f:
  loaded = safe_open(out_path, framework='pt')
  loaded_tensors: Dict[str, Tensor] = {}
  for key in f.keys():
    loaded_tensors[key] = f.get_tensor(key)
print(loaded_tensors)