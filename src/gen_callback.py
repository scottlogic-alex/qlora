from transformers import (   
	StoppingCriteria,
	StoppingCriteriaList,
	TrainerCallback,
	TrainingArguments,
	TrainerControl,
	TrainerState,
	LlamaTokenizer,
)
from dataclasses import dataclass, field
from peft import PeftModelForCausalLM
from datasets import Dataset
from typing import List, Iterator, Iterable, TypedDict
from torch import LongTensor, FloatTensor

from .callback_text_iterator_streamer import CallbackTextIteratorStreamer
from .collation import Collator
from .iteration import nth, repeatedly

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

@dataclass
class StopOnTokens(StoppingCriteria):
	stop_token_ids: List[int]
	def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
		for stop_id in self.stop_token_ids:
			if input_ids[0][-1] == stop_id:
				return True
		return False

@dataclass
class GenerationCallback(TrainerCallback):
	model: PeftModelForCausalLM
	# tokenizer: LlamaTokenizer
	dataset: Dataset
	collator: Collator
	favourite_prompt: str = field(init=False)
	data_it: Iterable[str] = field(init=False)
	def __post_init__(self):
		i: Iterator[str] = iter(self.dataset['input'])
		# What is $\sqrt{53}$ in simplest radical form?
		self.favourite_prompt = nth(i, 3)
		self.data_it = repeatedly(self.dataset['input'])


	def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		"""
		Event called at the beginning of a training step. If using gradient accumulation, one training step might take
		several inputs.
		"""
		prompts: List[str] = [self.favourite_prompt, next(self.data_it)]
		# tokenized_prompts: TokenizerOutput = self.tokenizer(prompts, return_tensors='pt', truncation=True)
		for input in prompts:
			streamer=CallbackTextIteratorStreamer()
			# TODO
