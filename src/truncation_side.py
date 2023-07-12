from transformers import PreTrainedTokenizer
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Optional

@dataclass
class truncation_side(ContextDecorator):
  tokenizer: PreTrainedTokenizer
  truncation_side: str
  orig_truncation_side: Optional[str] = None

  def __enter__(self):
    self.orig_truncation_side = self.tokenizer.truncation_side
    return self

  def __exit__(self, *exc):
    self.tokenizer.truncation_side = self.orig_truncation_side
    return False