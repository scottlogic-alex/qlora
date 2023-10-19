from torch.nn import Module
from bitsandbytes.nn import Params4bit

def count_model_params(model: Module) -> int:
  return sum([p.numel()*2 if isinstance(p, Params4bit) else p.numel() for p in model.parameters()])