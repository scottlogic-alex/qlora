from transformers import (   
	BatchEncoding,
	GenerationConfig,
	StoppingCriteriaList,
	TensorType,
	TrainerCallback,
	TrainingArguments,
	TrainerControl,
	TrainerState,
	LlamaTokenizer,
)
from dataclasses import dataclass, field
from peft import PeftModelForCausalLM
from datasets import Dataset
from typing import List, Iterable, TypedDict, Iterator, Dict, Literal
from torch import LongTensor, no_grad
from itertools import tee, cycle
from enum import Enum, auto

from .callback_text_iterator_streamer import CallbackTextIteratorStreamer
from .iteration import nth, repeatedly
from .truncation_side import truncation_side
from .stop_on_tokens import StopOnTokens

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor
  
class SampleSource(Enum):
	Favourite = auto(),
	Sequential = auto(),

@dataclass
class GenerationCallback(TrainerCallback):
	model: PeftModelForCausalLM
	tokenizer: LlamaTokenizer
	dataset: Dataset
	generation_config: GenerationConfig

	source_max_len: int
	target_max_len: int
	truncate_toward_center: bool
	use_bos_token_in_prompt: bool

	report_to_wandb: bool
	generate_steps: int

	favourite_sample: str = field(init=False)
	data_it: Iterable[str] = field(init=False)
	sample_source: Iterator[SampleSource] = field(init=False)
	def __post_init__(self):
		# What is $\sqrt{53}$ in simplest radical form?
		it0, it1 = tee(self.dataset['input'], 2)
		self.favourite_sample = nth(it0, 2)
		self.data_it = repeatedly(it1)
		self.sample_source = iter(cycle((SampleSource.Favourite, SampleSource.Sequential)))

		stop_token_ids: List[int] = [self.tokenizer.eos_token_id]
		stop = StopOnTokens(stop_token_ids)
		self.stopping_criteria = StoppingCriteriaList([stop])

	def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		"""
		Event called at the beginning of a training step. If using gradient accumulation, one training step might take
		several inputs.
		"""
		if state.global_step % self.generate_steps > 0:
			return
		sample_source: SampleSource = next(self.sample_source)

		sample: str = self.favourite_sample if sample_source is SampleSource.Favourite else next(self.data_it)

		instruction: str = sample[sample.find('Instruction:\n')+len('Instruction:\n'):]
		instruction: str = instruction[:instruction.find('\n\n### Response:\n')]

		if self.use_bos_token_in_prompt:
			sample = f"{self.tokenizer.bos_token}{sample}"

		with truncation_side(self.tokenizer, 'left'), no_grad():
			encoded: BatchEncoding = self.tokenizer(
				sample,
				max_length=self.source_max_len,
				truncation=True,
				add_special_tokens=False,
				return_tensors=TensorType.PYTORCH,
			)

		response = ''
		def on_text(message: str, stream_end = False):
			nonlocal response
			response += message
			print(message, end='', flush=True)

		streamer = CallbackTextIteratorStreamer(self.tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=False)

		print(instruction)
		with no_grad():
			prediction: LongTensor = self.model.generate(
				input_ids=encoded['input_ids'].to(self.model.device),
				attention_mask=encoded['attention_mask'].to(self.model.device),
				generation_config=self.generation_config,
				do_sample=self.generation_config.temperature > 0.,
				stopping_criteria=self.stopping_criteria,
				streamer=streamer,
			)
		print('')

		# decoded: str = self.tokenizer.decode(prediction[0, encoded['input_ids'].size(-1):], skip_special_tokens=False, clean_up_tokenization_spaces=True)

		if self.report_to_wandb:
			import wandb
			metric_key: Literal['prompt_fav', 'prompt_rand'] = 'prompt_fav' if sample_source is SampleSource.Favourite else 'prompt_rand'
			table = wandb.Table(data=[[instruction, response]], columns=['Instruction', 'Response'])
			metrics: Dict[Literal['prompt_fav', 'prompt_rand'], wandb.Table] = {
 				metric_key: table,
			}
			wandb.log(metrics, step=state.global_step)
		pass # put breakpoint here
