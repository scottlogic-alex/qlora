from transformers import (   
	GenerationConfig,
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
from typing import List, Iterable, TypedDict, Tuple, Iterator
from torch import LongTensor, FloatTensor
from itertools import tee, chain, repeat

from .callback_text_iterator_streamer import CallbackTextIteratorStreamer
from .collation import Collator, CollatedData, DataInstance
from .iteration import nth, repeatedly, roundrobin

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
	tokenizer: LlamaTokenizer
	dataset: Dataset
	collator: Collator
	generation_config: GenerationConfig
	test_instances: Iterator[DataInstance] = field(init=False)
	def __post_init__(self):
		io: Iterable[Tuple[str, str]] = zip(self.dataset['input'], self.dataset['output'])
		it: Iterable[DataInstance] = (DataInstance(input=input, output=output) for input, output in io)
		# What is $\sqrt{53}$ in simplest radical form?
		it0, it1 = tee(it, 2)
		favourite_sample: DataInstance = nth(it0, 2)
		data_it: Iterable[DataInstance] = repeatedly(it1)

		# alternates between our favourite, and a random
		self.test_instances = iter(roundrobin(repeat(favourite_sample), data_it))

		stop_token_ids: List[int] = [self.tokenizer.eos_token_id]
		stop = StopOnTokens(stop_token_ids)
		self.stopping_criteria = StoppingCriteriaList([stop])

	def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		"""
		Event called at the beginning of a training step. If using gradient accumulation, one training step might take
		several inputs.
		"""
		sample: DataInstance = next(self.test_instances)
		collated: CollatedData = self.collator([sample])
		# for sample in samples:
		streamer=CallbackTextIteratorStreamer()
		prediction: LongTensor = self.model.generate(
			input_ids=collated['input_ids'].to(self.model.device),
			attention_mask=collated['attention_mask'].to(self.model.device),
			generation_config=self.generation_config,
			do_sample=self.generation_config.temperature > 0.,
			stopping_criteria=self.stopping_criteria,
			# TODO: streamer probably doesn't make sense for batches
			streamer=streamer,
		)
		decodeds: List[str] = self.tokenizer.decode(prediction[0, collated['input_ids'].size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
		for decoded in decodeds:
			# TODO: wandb
			print(decoded)
		pass # put breakpoint here
